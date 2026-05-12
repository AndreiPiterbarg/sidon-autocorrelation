"""Prototype 3 stacked pruning improvements + combined chain.

Tested on the 100 saved F+Q-survivors at (n_half=4, m=10, d=8, c=1.281).
Survivors are loaded from _L_order2_survivors.npy.

Prototype 1 (M2): exact per-window N_i*c_i sum vs F's (ell-1)*W_int overestimate.
Prototype 2 (Positivity): augment Q-LP and L-SDP with a_i >= 0 (i.e. delta_i >= -c_i/m).
Prototype 3 (Inner BnB): bisect delta-cell along chosen coord, recurse L-Shor on each child.

Combined: F + Q + L_pos + M2 + InnerBnB chain.

Outputs:
  _pruning_proto.json
  console table

Constraint: <30 min total wall on local Windows. Use multiprocessing with N_WORKERS=12.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from multiprocessing import Pool

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

# ---- Config ---------------------------------------------------------
N_HALF = 4
M = 10
D = 2 * N_HALF
C_TARGET = 1.281
N_WORKERS = 12
SURV_PATH = os.path.join(_dir, '_L_order2_survivors.npy')
OUT_JSON = os.path.join(_dir, '_pruning_proto.json')

# ---- Per-worker globals ---------------------------------------------
_G = {}


def _worker_init():
    from _Q_bench import _build_windows, _enum_balanced_signs
    from _L_bench import _build_A_matrices
    windows, ell_int_sums = _build_windows(D)
    A_mats = _build_A_matrices(D, windows)
    sigmas = _enum_balanced_signs(D)
    _G['windows'] = windows
    _G['ell_int_sums'] = ell_int_sums
    _G['A_mats'] = A_mats
    _G['sigmas'] = sigmas


# ---------------------------------------------------------------------
# PROTOTYPE 1 — M2 (numba batch kernel)
# ---------------------------------------------------------------------
def proto1_m2_kills(survivors):
    from _M2_bench import prune_E
    batch = np.ascontiguousarray(survivors, dtype=np.int32)
    warm = batch[:1].copy()
    prune_E(warm, N_HALF, M, C_TARGET)  # JIT warmup
    t0 = time.time()
    survived = prune_E(batch, N_HALF, M, C_TARGET)
    wall = time.time() - t0
    kills = int(np.sum(~survived))
    return kills, wall, np.where(survived)[0].tolist()


# ---------------------------------------------------------------------
# PROTOTYPE 2a — Q-LP with positivity (vertex filter)
# ---------------------------------------------------------------------
def _q_lp_pos(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    """Q-LP retaining only sigma vertices that are positivity-feasible.

    Original sigma in {-1,+1}^d w/ sum=0 corresponds to extreme points of
    {|delta|_inf <= 1/m, sum delta = 0}. With a_i >= 0, delta_i >= -c_i/m;
    if c_i = 0 then delta_i >= 0 — so any sigma with sigma_i = -1 at i where
    c_i = 0 is INFEASIBLE under positivity. Dropping such sigmas is sound
    (relaxes the inner sup, can only weaken — no it actually TIGHTENS the
    Q lower bound on delta-sup, making Q-prune EASIER). Wait — Q computes
    sup over delta of (TV(b) - TV(a)); restricting delta to a smaller set
    REDUCES the sup, hence INCREASES the lower bound on inf TV(a) =>
    MORE pruning. So vertex filter is sound + tightens. (We also account
    for vertices on the new positivity face, but that requires non-vertex
    enumeration; this prototype just tries the cheap vertex-restriction.)
    """
    from scipy.optimize import linprog
    from _Q_bench import _composition_window_data

    d = len(c_int)
    # Filter sigmas: drop any sigma with sigma_i = -1 where c_int[i] == 0.
    zero_mask = (np.asarray(c_int) == 0)
    if zero_mask.any():
        # Keep sigma if (sigma == -1) AND zero_mask never both true at any i.
        bad = (sigmas == -1) & zero_mask[None, :]
        keep = ~bad.any(axis=1)
        sig_pos = sigmas[keep]
    else:
        sig_pos = sigmas
    if len(sig_pos) == 0:
        return True  # cell vertex set empty under positivity -> tight => prune

    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m

    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    V = ws.astype(np.float64) * inv_4nl - ell_int_sums.astype(np.float64) * inv_4nl - cs_m2

    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M_mat = sig_pos.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M_mat / (2.0 * n_d)

    n_win = len(windows)
    n_sigma = len(sig_pos)
    nvar = n_win + 1
    c_obj = np.zeros(nvar); c_obj[-1] = -1.0
    A_ub = np.zeros((n_sigma, nvar))
    A_ub[:, :n_win] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_sigma)
    A_eq = np.zeros((1, nvar)); A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_win + [(None, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if not res.success:
            return False
        return (-res.fun) > 1e-9 * m * m
    except Exception:
        return False


def _proto2a_one(args):
    c_int = np.asarray(args, dtype=np.int32)
    return _q_lp_pos(c_int, _G['windows'], _G['ell_int_sums'], _G['sigmas'],
                      N_HALF, M, C_TARGET)


# ---------------------------------------------------------------------
# Shor SDP feasibility on user-supplied (lo, hi) box -- supports BnB
# ---------------------------------------------------------------------
def _shor_box(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
              eps_margin=1e-9):
    """Order-1 Shor (Lasserre) SDP feasibility on the box [lo, hi].

    Returns (pruned_bool, status_string).
    """
    import cvxpy as cp
    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    # Quick infeasibility check: sum(lo) > nm or sum(hi) < nm => infeasible
    if lo.sum() > nm + 1e-9 or hi.sum() < nm - 1e-9:
        return True, 'box_sum_infeasible'

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0, x >= lo, x <= hi, cp.sum(x) == nm]
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver='CLARABEL', verbose=False, max_iter=300)
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'
    return prob.status == 'infeasible', prob.status


def _make_baseline_box(c_int):
    c = np.asarray(c_int, dtype=np.float64)
    return np.maximum(0.0, c - 1.0), c + 1.0


# ---------------------------------------------------------------------
# Inner BnB on delta-cell
# ---------------------------------------------------------------------
def _bnb_prune(c_int, A_mats, windows, n_half, m, c_target, max_depth=5):
    info = {'solves': 0, 'pruned': False, 'depth_at_kill': -1}
    lo0, hi0 = _make_baseline_box(c_int)

    # Iterative DFS to allow time-bounding.
    def recurse(lo, hi, depth):
        info['solves'] += 1
        ok, status = _shor_box(c_int, lo, hi, A_mats, windows, n_half, m, c_target)
        if ok:
            return True, depth
        if depth >= max_depth:
            return False, depth
        widths = hi - lo
        i_split = int(np.argmax(widths))
        if widths[i_split] < 1e-9:
            return False, depth
        mid = 0.5 * (lo[i_split] + hi[i_split])
        hi_L = hi.copy(); hi_L[i_split] = mid
        ok_L, d_L = recurse(lo, hi_L, depth + 1)
        if not ok_L:
            return False, d_L
        lo_R = lo.copy(); lo_R[i_split] = mid
        ok_R, d_R = recurse(lo_R, hi, depth + 1)
        if not ok_R:
            return False, d_R
        return True, max(d_L, d_R)

    pruned, depth_max = recurse(lo0, hi0, 0)
    info['pruned'] = pruned
    info['depth_at_kill'] = depth_max if pruned else -1
    return info


def _proto3_one(args):
    c_int_list, max_depth = args
    c_int = np.asarray(c_int_list, dtype=np.int32)
    return _bnb_prune(c_int, _G['A_mats'], _G['windows'],
                       N_HALF, M, C_TARGET, max_depth=max_depth)


def _proto_L_baseline_one(c_int_list):
    c_int = np.asarray(c_int_list, dtype=np.int32)
    lo, hi = _make_baseline_box(c_int)
    pruned, status = _shor_box(c_int, lo, hi, _G['A_mats'], _G['windows'],
                                N_HALF, M, C_TARGET)
    return pruned, str(status)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    survivors = np.load(SURV_PATH)
    n_surv = len(survivors)
    print(f"Loaded {n_surv} F+Q survivors from {SURV_PATH}")
    print(f"Config: n_half={N_HALF}, m={M}, d={D}, c_target={C_TARGET}, N_WORKERS={N_WORKERS}\n")

    results = {
        'config': {'n_half': N_HALF, 'm': M, 'd': D, 'c_target': C_TARGET,
                   'n_survivors': int(n_surv), 'n_workers': N_WORKERS},
        'tests': {},
    }

    surv_list = [survivors[i].tolist() for i in range(n_surv)]

    # ---- BASELINE: L order-1 Shor on each survivor
    print("=== Baseline: L order-1 Shor (single-box, no BnB) ===")
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_worker_init) as pool:
        out_L = pool.map(_proto_L_baseline_one, surv_list)
    wall_L = time.time() - t0
    pruned_L_mask = [bool(p) for p, s in out_L]
    statuses_L = {}
    for _, s in out_L:
        statuses_L[s] = statuses_L.get(s, 0) + 1
    kills_L = int(sum(pruned_L_mask))
    print(f"  kills: {kills_L}/{n_surv}  wall: {wall_L:.2f}s  per-comp: {wall_L/n_surv*1000:.1f}ms")
    print(f"  statuses: {statuses_L}")
    results['tests']['baseline_L'] = {
        'kills': kills_L, 'n': n_surv, 'wall_s': wall_L,
        'wall_per_comp_s': wall_L / n_surv,
        'statuses': statuses_L,
    }

    # ---- PROTO 1: M2
    print("\n=== Proto 1: M2 (exact per-window N_i*c_i) ===")
    kills_M2, wall_M2, M2_surv_idx = proto1_m2_kills(survivors)
    print(f"  kills: {kills_M2}/{n_surv}  wall: {wall_M2:.3f}s  per-comp: {wall_M2/n_surv*1e6:.1f}us")
    results['tests']['M2'] = {
        'kills': kills_M2, 'n': n_surv, 'wall_s': wall_M2,
        'wall_per_comp_s': wall_M2 / n_surv,
        'surviving_idx': M2_surv_idx,
    }

    # ---- PROTO 2a: Q-LP with positivity vertex filter
    print("\n=== Proto 2a: Q-LP with positivity (vertex filter) ===")
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_worker_init) as pool:
        out2a = pool.map(_proto2a_one, surv_list)
    wall_2a = time.time() - t0
    kills_2a = int(sum(out2a))
    print(f"  kills: {kills_2a}/{n_surv}  wall: {wall_2a:.2f}s  per-comp: {wall_2a/n_surv*1000:.1f}ms")
    results['tests']['proto2a_Qpos'] = {
        'kills': kills_2a, 'n': n_surv, 'wall_s': wall_2a,
        'wall_per_comp_s': wall_2a / n_surv,
    }

    # ---- PROTO 2b: L-Shor positivity == baseline (lo = max(0, c-1))
    print("\n=== Proto 2b: L-Shor with positivity ===")
    print(f"  L-Shor's box uses lo=max(0,c-1) >= 0; positivity built-in => same as baseline_L: {kills_L}/{n_surv}.")
    results['tests']['proto2b_Lpos'] = {
        'kills': kills_L, 'n': n_surv,
        'note': 'L-Shor already encodes a_i >= 0 via lo = max(0, c-1)',
    }

    # ---- PROTO 3: Inner BnB on delta-cell
    print("\n=== Proto 3: Inner BnB on delta-cell ===")
    L_survs_idx = [i for i, p in enumerate(pruned_L_mask) if not p]
    print(f"  Running BnB on {len(L_survs_idx)} L-survivors.")
    bnb_kills = set()
    out3 = []
    if len(L_survs_idx) == 0:
        print("  No L-survivors -- skipping BnB.")
        results['tests']['proto3_innerBnB'] = {'skipped': True, 'reason': 'no L-survivors'}
        max_depth_used = 0
    else:
        per_solve = wall_L / n_surv
        max_depth = 5
        worst_solves = 2 ** (max_depth + 1) - 1
        est_parallel = per_solve * worst_solves * len(L_survs_idx) / N_WORKERS
        print(f"  per-solve baseline {per_solve*1000:.1f}ms; worst-case tree size {worst_solves}; "
              f"est parallel wall {est_parallel:.1f}s")
        if est_parallel > 1500:
            max_depth = 4
            print(f"  Reducing max_depth -> {max_depth}")
        max_depth_used = max_depth

        L_surv_args = [(survivors[i].tolist(), max_depth) for i in L_survs_idx]
        t0 = time.time()
        with Pool(N_WORKERS, initializer=_worker_init) as pool:
            out3 = pool.map(_proto3_one, L_surv_args)
        wall_3 = time.time() - t0
        print(f"  BnB wall: {wall_3:.2f}s  per-Lsurv: {wall_3/len(L_survs_idx):.2f}s")

        kills_at_depth = {k: 0 for k in range(max_depth + 1)}
        total_solves = 0
        for orig_idx, info in zip(L_survs_idx, out3):
            total_solves += info['solves']
            if info['pruned']:
                bnb_kills.add(orig_idx)
                d_kill = info['depth_at_kill']
                for k in range(d_kill, max_depth + 1):
                    kills_at_depth[k] += 1
        print(f"  BnB extra kills (over baseline L), by max-depth allowed:")
        for k in range(max_depth + 1):
            extra = kills_at_depth[k]
            tot = kills_L + extra
            print(f"    depth<={k}: extra={extra:3d}  total={tot}/{n_surv} ({100*tot/n_surv:.1f}%)")
        print(f"  Total SDP solves: {total_solves}")
        results['tests']['proto3_innerBnB'] = {
            'wall_s': wall_3,
            'wall_per_Lsurv_s': wall_3 / len(L_survs_idx),
            'wall_per_comp_s': wall_3 / n_surv,
            'max_depth': max_depth,
            'n_L_survs_input': len(L_survs_idx),
            'extra_kills_by_depth': kills_at_depth,
            'total_kills_by_depth_with_L': {k: kills_L + v for k, v in kills_at_depth.items()},
            'total_sdp_solves': total_solves,
        }

    # ---- COMBINED CHAIN
    print("\n=== Combined: F+Q (given) + M2 + L-pos + InnerBnB ===")
    killed_M2_set = set(range(n_surv)) - set(M2_surv_idx)
    killed_L_set = {i for i, p in enumerate(pruned_L_mask) if p}
    all_killed = killed_M2_set | killed_L_set | bnb_kills
    kills_combined = len(all_killed)
    print(f"  Combined kills: {kills_combined}/{n_surv} ({100*kills_combined/n_surv:.1f}%)")
    print(f"    by M2:           {len(killed_M2_set)}")
    print(f"    by L (baseline): {len(killed_L_set)}")
    print(f"    by BnB (extra over L): {len(bnb_kills - killed_L_set)}")
    results['tests']['combined'] = {
        'kills': kills_combined, 'n': n_surv,
        'kills_by_M2': len(killed_M2_set),
        'kills_by_L': len(killed_L_set),
        'kills_by_BnB_extra_over_L': len(bnb_kills - killed_L_set),
    }

    # ---- TABLE
    print("\n=== TABLE ===")
    hdr = f"{'method':<24} {'kills/100':>10} {'wall(s)':>10} {'/comp(s)':>12}"
    print(hdr)
    print('-' * len(hdr))

    def _row(name, key, kills_override=None):
        t = results['tests'].get(key, {})
        if kills_override is not None:
            kk = kills_override
        elif 'kills' in t:
            kk = t['kills']
        else:
            kk = '-'
        wall = t.get('wall_s', '-')
        per = t.get('wall_per_comp_s', '-')
        wall_s = f"{wall:.2f}" if isinstance(wall, (int, float)) else str(wall)
        per_s = f"{per:.4f}" if isinstance(per, (int, float)) else str(per)
        print(f"{name:<24} {str(kk):>10} {wall_s:>10} {per_s:>12}")

    _row('baseline F+Q+L', 'baseline_L')
    _row('Proto1 M2', 'M2')
    _row('Proto2a Q-pos', 'proto2a_Qpos')
    _row('Proto2b L-pos (=L)', 'proto2b_Lpos')
    if 'proto3_innerBnB' in results['tests'] and not results['tests']['proto3_innerBnB'].get('skipped'):
        md = results['tests']['proto3_innerBnB']['max_depth']
        bnb_total = results['tests']['proto3_innerBnB']['total_kills_by_depth_with_L'][md]
        _row(f'Proto3 BnB(d<={md})', 'proto3_innerBnB', kills_override=bnb_total)
    _row('Combined', 'combined')

    with open(OUT_JSON, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")


if __name__ == '__main__':
    main()

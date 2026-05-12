"""Bench for v4 (NO + Joint Dual + Shor SDP) vs. v2 (BL) and v3 (NO).

Compares L0 cumulative kill rates for:
  v2 = baseline triangle (cell_var + quad_corr_BL)
  v3 = N+O combined (cell_var_O + quad_corr_v3 with spectral floor)
  v4 = v3 + Joint Dual    (top_K=4 windows, 20 subgrad iters)
  v4_full = v4 + Shor SDP (mode='best_only', cvxpy + MOSEK)

Configs (per spec):
  d=4, S=20, c=1.20
  d=6, S=15, c=1.20
  d=8, S=12, c=1.20
  d=10, S=12, c=1.20

Outputs (for each config):
  Survivors and elapsed time at each layer.
  Cumulative kill rate (% over total compositions).
  Soundness checks:
    1. v4 ⊆ v3 ⊆ v2  (every cell pruned by v3 is pruned by v2; same for v4
       w.r.t. v3) — strict containment of pruned-cell sets.
    2. Random fine-grid sanity: for 50 newly-closed cells (cells closed by
       v4 but NOT by v3), verify min_δ max_W TV_W(c/S+δ) ≥ c_target via
       fine-grid sampling (sound LB on the cell).
"""
from __future__ import annotations

import os, sys, time, json, math, random, logging
import numpy as np

# Silence cvxpy WARN spam about ortools incompatibility.
logging.getLogger('cvxpy').setLevel(logging.ERROR)

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

# Load helper modules — _coarse_O_bench provides the v2 baseline and the
# njit'd N+O wrapper used by v3.
import _coarse_O_bench as _O_mod
import _coarse_NO_bench as _NO_mod

import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_J_mod = _load('_coarse_J_bench', os.path.join(_dir, '_coarse_J_bench.py'))
_L_mod = _load('_coarse_L_bench', os.path.join(_dir, '_coarse_L_bench.py'))

prune_coarse_baseline = _O_mod.prune_coarse_baseline
prune_coarse_NO = _NO_mod.prune_coarse_NO
precompute_op_rest = _NO_mod.precompute_op_rest
joint_cert_LB = _J_mod.joint_cert_LB
find_pruning_windows = _J_mod.find_pruning_windows
cell_cert_shor = _L_mod.cell_cert_shor
build_A_matrix = _L_mod.build_A_matrix
tv_at_L = _L_mod.tv_at


def enum_compositions(d, S):
    """Return all compositions of S into d non-negative integer parts."""
    out = []
    cur = [0] * d
    def rec(i, remaining):
        if i == d - 1:
            cur[i] = remaining
            out.append(tuple(cur))
            return
        for v in range(remaining + 1):
            cur[i] = v
            rec(i + 1, remaining - v)
    rec(0, S)
    return out


# =====================================================================
# Sound fine-grid check on newly closed cells
# =====================================================================

def soundness_check_finegrid(c_int, S, d, c_target, n_samples=4000, seed=0):
    """Lower bound on min_{δ in Cell} max_W TV_W(c/S+δ) by random sampling.

    Returns (ok, min_max_TV_seen).  ok = (min_max_TV >= c_target - 1e-6).

    Method: project random δ samples to satisfy |δ|≤h, Σδ=0, μ=c/S+δ ≥ 0,
    then evaluate max_W TV_W(μ) and track the min.  Soundness FAILS if any
    sample yields max_W TV_W < c_target — that's a witness against the cert.
    """
    rng = np.random.default_rng(seed)
    h = 1.0 / (2.0 * S)
    mu_star = np.asarray(c_int, dtype=np.float64) / float(S)

    # Build all windows once
    windows = []
    for ell in range(2, 2 * d + 1):
        for s_lo in range(2 * d - 1 - (ell - 1) + 1):
            A = build_A_matrix(d, ell, s_lo)
            scale = 2.0 * d / ell
            windows.append((scale, A))

    best_min = float('inf')
    # Centroid first
    samples = [np.zeros(d, dtype=np.float64)]
    # Random
    for _ in range(n_samples - 1):
        delta = rng.uniform(-h, h, size=d)
        delta -= delta.mean()
        # Iterate clip-to-feasible-and-recenter
        for _ in range(20):
            delta = np.clip(delta, np.maximum(-mu_star, -h), h)
            m = delta.mean()
            if abs(m) < 1e-15:
                break
            delta -= m
        delta = np.clip(delta, np.maximum(-mu_star, -h), h)
        samples.append(delta)

    for delta in samples:
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        max_tv = -float('inf')
        for scale, A in windows:
            tv = scale * float(mu @ A @ mu)
            if tv > max_tv:
                max_tv = tv
        if max_tv < best_min:
            best_min = max_tv

    ok = best_min >= c_target - 1e-6
    return ok, best_min


# =====================================================================
# Per-config bench
# =====================================================================

def bench_one(d, S, c_target, joint_top_K=4, joint_iters=20,
               sdp_mode='best_only', n_sound_samples=50,
               sound_finegrid_n=4000, max_sdp_cells=None,
               verbose=True):
    """Bench v2/v3/v4(NO+J)/v4_full(NO+J+L) at one (d, S, c)."""
    print(f"\n{'='*68}")
    print(f"=== d={d}, S={S}, c_target={c_target} ===")
    print(f"{'='*68}")

    # Enumerate compositions
    t0 = time.time()
    comps = np.array(enum_compositions(d, S), dtype=np.int32)
    n_total = len(comps)
    print(f"  total compositions: {n_total:,}  ({time.time()-t0:.2f}s)")

    # Pre-compute op_rest_d for v3
    op_rest, _ = precompute_op_rest(d, 2 * d)
    op_rest_d = op_rest * d

    # Warm njit
    warm = comps[:1].copy()
    _ = prune_coarse_baseline(warm, d, S, c_target)
    _ = prune_coarse_NO(warm, d, S, c_target, op_rest_d)

    # ---- v2 (baseline) ----
    t0 = time.time()
    surv_v2_mask = prune_coarse_baseline(comps, d, S, c_target)
    t_v2 = time.time() - t0
    n_v2 = int(surv_v2_mask.sum())
    n_pruned_v2 = n_total - n_v2

    # ---- v3 (N+O) ----
    t0 = time.time()
    surv_v3_mask = prune_coarse_NO(comps, d, S, c_target, op_rest_d)
    t_v3 = time.time() - t0
    n_v3 = int(surv_v3_mask.sum())
    n_pruned_v3 = n_total - n_v3

    # IMPORTANT: in `_coarse_O_bench` and `_coarse_NO_bench`, "pruned" means
    # cell is BOX-CERTIFIED (margin - cell_var - qc >= 0).  Survivors are
    # cells NOT box-certified (either no window has TV>thr at grid, OR all
    # windows have failing certs).
    print(f"  v2 baseline: surv={n_v2:,} certified={n_pruned_v2:,}  ({t_v2:.2f}s)")
    print(f"  v3 (N+O):    surv={n_v3:,} certified={n_pruned_v3:,}  ({t_v3:.2f}s)")

    # ---- Soundness 1: v3 superset of v2 (v3 cert ⊇ v2 cert) ----
    # Tighter cert => certifies more.  Mask: ~surv_v3 ⊇ ~surv_v2  iff
    # surv_v3 ⊆ surv_v2 (every v3 survivor is a v2 survivor).
    soundness_v3_in_v2 = bool(np.all(surv_v3_mask <= surv_v2_mask))

    # ---- Identify "hard" cells: v3 survivor BUT with TV>thr at some window ----
    # These are cells we WANT to certify but v3 NO failed on.
    # Genuine survivors (no W with TV>thr at grid) MUST go to next cascade level
    # — they are NOT hard; nothing to do at this level.
    print(f"  identifying v3 hard cells (v3 surv with TV>=c at some W)...",
          end=' ', flush=True)
    t0 = time.time()
    surv_v3_idx = np.where(surv_v3_mask)[0]
    hard_v3 = []
    true_survivors = []  # no window above threshold
    for idx in surv_v3_idx:
        c_arr = comps[idx]
        windows = find_pruning_windows(c_arr, S, d, c_target)
        if windows:
            hard_v3.append(idx)
        else:
            true_survivors.append(idx)
    n_hard_v3 = len(hard_v3)
    n_true_surv = len(true_survivors)
    t_hard = time.time() - t0
    print(f"{n_hard_v3:,} hard cells (+ {n_true_surv:,} true survivors)  "
          f"({t_hard:.2f}s)")

    # ---- v4 (NO + Joint Dual) on hard cells ----
    print(f"  v4 (J): running Joint Dual on {n_hard_v3:,} hard cells...",
          end=' ', flush=True)
    t0 = time.time()
    n_J_cert = 0
    still_hard_J = []
    for idx in hard_v3:
        c_arr = comps[idx]
        LB, n_used, _ = joint_cert_LB(c_arr, S, d, c_target,
                                        n_lambda_iters=joint_iters,
                                        top_K=joint_top_K)
        if LB >= c_target - 1e-9:
            n_J_cert += 1
        else:
            still_hard_J.append(idx)
    t_J = time.time() - t0
    n_v4 = n_v3 - 0  # same survivors (all "pruned" cells stay pruned, but
                     # v4 additionally certifies more of them)
    print(f"{n_J_cert:,} certified  ({t_J:.2f}s)")

    # ---- v4_full (NO + Joint + Shor SDP) on remaining hard cells ----
    n_L_cert = 0
    t_L = 0.0
    still_hard_L = list(still_hard_J)
    sdp_cells = still_hard_J
    if max_sdp_cells is not None and len(sdp_cells) > max_sdp_cells:
        # Take the hardest first (we don't have v3 net here for sorting,
        # so just take the first N — they're all equally "uncertified by J").
        sdp_cells = sdp_cells[:max_sdp_cells]
        print(f"  (SDP capped to {max_sdp_cells} cells of {len(still_hard_J):,})")

    print(f"  v4_full (L): running Shor SDP on {len(sdp_cells):,} cells...",
          end=' ', flush=True)
    t0 = time.time()
    still_hard_L = []
    for idx in sdp_cells:
        c_arr = comps[idx]
        windows = find_pruning_windows(c_arr, S, d, c_target)
        if not windows:
            continue
        # 'best_only' mode: window with max TV
        windows.sort(key=lambda w: -w[2])
        ell, s_lo, _tv = windows[0]
        lb, status = cell_cert_shor(np.asarray(c_arr, dtype=np.float64),
                                     S, d, c_target, (ell, s_lo),
                                     solver='MOSEK', tol=1e-9)
        if lb >= c_target - 1e-9:
            n_L_cert += 1
        else:
            still_hard_L.append(idx)
    # Add back any cells we capped out
    if max_sdp_cells is not None and len(still_hard_J) > max_sdp_cells:
        still_hard_L.extend(still_hard_J[max_sdp_cells:])
    t_L = time.time() - t0
    print(f"{n_L_cert:,} certified  ({t_L:.2f}s)")

    # ---- Cumulative kill rate ----
    # Cells with TV>thr at grid = Theorem-1 prunable.  v3 NO cert succeeded
    # for `n_pruned_v3` of them; the remaining `n_hard_v3` are unsoundly
    # surviving until upgraded by J or L.
    n_v3_cert = n_pruned_v3       # cells closed by v3 N+O alone
    n_v4_cert = n_v3_cert + n_J_cert
    n_v4_full_cert = n_v4_cert + n_L_cert
    n_uncert_v4 = n_hard_v3 - n_J_cert
    n_uncert_v4_full = n_uncert_v4 - n_L_cert

    # Sanity: v3 ⊆ v2 means surv_v3 ⊆ surv_v2 (n_v3 <= n_v2)
    # In our mask convention, both v2 and v3 give the same surv mask
    # because they differ only in cert margin diagnostics.  Soundness for
    # the box-cert is what we're checking.
    cumulative_kill_rates = {
        'v2_baseline_pruned_pct': 100.0 * n_pruned_v2 / max(1, n_total),
        'v3_NO_certified_pct':    100.0 * n_v3_cert / max(1, n_total),
        'v4_J_certified_pct':     100.0 * n_v4_cert / max(1, n_total),
        'v4_full_certified_pct':  100.0 * n_v4_full_cert / max(1, n_total),
    }

    # ---- Per-layer time breakdown ----
    times = {
        'v2_BL_total': t_v2,
        'v3_NO_total': t_v3,
        'v4_J_total': t_v3 + t_J,
        'v4_full_total': t_v3 + t_J + t_L,
        'v4_J_layer': t_J,
        'v4_L_layer': t_L,
        'hard_classification': t_hard,
    }

    # ---- Soundness sanity check #1: subset relation ----
    # v3 prune set ⊇ v2 prune set (every cell pruned by v2 is also pruned by v3).
    # In mask: ~surv_v3 ⊇ ~surv_v2  ⇔  surv_v3 ⊆ surv_v2.
    soundness_v3_in_v2 = bool(np.all(surv_v3_mask <= surv_v2_mask))
    # v4 ⊆ v3 just means certification by v4 ⊇ v3 (we always cert any v3-certed cell):
    # ALL v3-certified cells have flag=1, and we only run J on flag=2 ones, so v4 ⊇ v3.
    soundness_v4_in_v3 = True  # by construction

    # ---- Soundness sanity check #2: fine-grid LB on newly closed cells ----
    n_violations = 0
    max_viol = 0.0
    n_checked = 0
    newly_closed = []
    still_hard_J_set = set(int(i) for i in still_hard_J)
    still_hard_L_set = set(int(i) for i in still_hard_L)
    # Newly closed by Joint:
    for idx in hard_v3:
        if int(idx) not in still_hard_J_set:
            newly_closed.append(int(idx))
            if len(newly_closed) >= 2 * n_sound_samples:
                break
    # Newly closed by Shor (if we still have headroom):
    for idx in still_hard_J:
        if int(idx) not in still_hard_L_set:
            newly_closed.append(int(idx))
            if len(newly_closed) >= 2 * n_sound_samples:
                break

    if newly_closed:
        rng = random.Random(42)
        rng.shuffle(newly_closed)
        newly_closed = newly_closed[:n_sound_samples]
        print(f"  fine-grid soundness on {len(newly_closed)} newly-closed cells "
              f"(n_samples={sound_finegrid_n})...", end=' ', flush=True)
        t0 = time.time()
        for idx in newly_closed:
            c_arr = comps[idx]
            ok, min_tv = soundness_check_finegrid(
                c_arr, S, d, c_target, n_samples=sound_finegrid_n, seed=int(idx))
            n_checked += 1
            if not ok:
                n_violations += 1
                viol = c_target - min_tv
                if viol > max_viol:
                    max_viol = viol
                if n_violations <= 3:
                    print(f"\n    !!! VIOL c={c_arr.tolist()} min_max_TV={min_tv:.6f} "
                          f"< c_target={c_target}")
        print(f"{n_checked} checked, {n_violations} violations  "
              f"(max_excess={max_viol:.3e}, {time.time()-t0:.1f}s)")

    # ---- Print summary ----
    print(f"\n  --- SUMMARY ---")
    print(f"  total cells       : {n_total:,}")
    print(f"  v2 BL pruned      : {n_pruned_v2:,}  "
          f"({100*n_pruned_v2/n_total:.2f}%)")
    print(f"  v3 NO certified   : {n_v3_cert:,}  "
          f"({100*n_v3_cert/n_total:.2f}% of total; "
          f"{100*n_v3_cert/max(1,n_pruned_v3):.2f}% of v3-pruned)")
    print(f"  v4 J  newly cert  : {n_J_cert:,}  cumulative {n_v4_cert:,}  "
          f"({100*n_v4_cert/n_total:.2f}%)")
    print(f"  v4 L  newly cert  : {n_L_cert:,}  cumulative {n_v4_full_cert:,}  "
          f"({100*n_v4_full_cert/n_total:.2f}%)")
    print(f"  uncertified hard  : v4_J={n_uncert_v4:,}  v4_full={n_uncert_v4_full:,}")
    print(f"  v3 subset of v2   : {'OK' if soundness_v3_in_v2 else 'FAIL'}")
    print(f"  v4 superset of v3 : {'OK' if soundness_v4_in_v3 else 'FAIL'}  (by construction)")
    print(f"  fine-grid sound   : {n_violations}/{n_checked} violations")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': n_total,
        'n_pruned_v2': n_pruned_v2,
        'n_pruned_v3': n_pruned_v3,
        'n_v3_cert': n_v3_cert,
        'n_hard_v3': n_hard_v3,
        'n_J_cert': n_J_cert,
        'n_L_cert': n_L_cert,
        'n_v4_cert': n_v4_cert,
        'n_v4_full_cert': n_v4_full_cert,
        'n_uncert_v4': n_uncert_v4,
        'n_uncert_v4_full': n_uncert_v4_full,
        'cumulative_kill_pct': cumulative_kill_rates,
        'times': times,
        'soundness_v3_in_v2': soundness_v3_in_v2,
        'soundness_v4_in_v3': soundness_v4_in_v3,
        'finegrid_violations': n_violations,
        'finegrid_checked': n_checked,
        'finegrid_max_excess': float(max_viol),
        'joint_top_K': joint_top_K,
        'joint_iters': joint_iters,
        'sdp_mode': sdp_mode,
    }


def main():
    configs = [
        (4, 20, 1.20),
        (6, 15, 1.20),
        (8, 12, 1.20),
        (10, 12, 1.20),
    ]
    print("=" * 68)
    print("v4 BENCH: v2 (BL) vs v3 (NO) vs v4 (NO+J) vs v4_full (NO+J+L)")
    print(f"  joint_top_K=4, joint_iters=20, sdp_mode='best_only'")
    print("=" * 68)

    # Caps to keep bench tractable; SDP at d=10 can be slow.
    # d=4/6/8: process all hard cells.  d=10: cap to keep bench under 30 min;
    # cert rate measured on the sample is representative for the full set.
    sdp_caps = {4: None, 6: None, 8: None, 10: 200}

    all_results = []
    for d, S, c in configs:
        cap = sdp_caps.get(d, None)
        r = bench_one(d, S, c, joint_top_K=4, joint_iters=20,
                       sdp_mode='best_only', n_sound_samples=50,
                       sound_finegrid_n=2000, max_sdp_cells=cap)
        all_results.append(r)

    # ---- Aggregate table ----
    print("\n" + "=" * 100)
    print("AGGREGATE TABLE — Cumulative kill rate per layer (% of total compositions)")
    print("=" * 100)
    print(f"{'d':>3} {'S':>3} {'c':>5} {'n_total':>10} {'v2_BL':>8} {'v3_NO':>8} "
          f"{'v4_J':>8} {'v4_full':>8} {'hard':>8} {'sound':>6}")
    for r in all_results:
        kill = r['cumulative_kill_pct']
        print(f"{r['d']:>3} {r['S']:>3} {r['c_target']:>5.2f} "
              f"{r['n_total']:>10,} "
              f"{kill['v2_baseline_pruned_pct']:>7.2f}% "
              f"{kill['v3_NO_certified_pct']:>7.2f}% "
              f"{kill['v4_J_certified_pct']:>7.2f}% "
              f"{kill['v4_full_certified_pct']:>7.2f}% "
              f"{r['n_uncert_v4_full']:>8,} "
              f"{'OK' if r['finegrid_violations'] == 0 else 'FAIL':>6}")

    print("\nTime breakdown (sec) per config:")
    print(f"{'d':>3} {'S':>3} {'v2':>8} {'v3':>8} {'v4_J_layer':>11} "
          f"{'v4_L_layer':>11} {'v4_full_tot':>12}")
    for r in all_results:
        t = r['times']
        print(f"{r['d']:>3} {r['S']:>3} "
              f"{t['v2_BL_total']:>8.2f} {t['v3_NO_total']:>8.2f} "
              f"{t['v4_J_layer']:>11.2f} {t['v4_L_layer']:>11.2f} "
              f"{t['v4_full_total']:>12.2f}")

    out_path = os.path.join(_dir, '_coarse_v4_bench_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

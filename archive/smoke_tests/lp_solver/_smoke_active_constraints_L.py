"""Smoke test: active-window constraint filtering for L-bench's Shor SDP.

Idea
====
`_L_bench._shor_feasibility` adds Tr(A_W X) <= thr_W for EVERY window W (n_win
= O(d^2) of them).  Many windows have a HUGE slack at the composition's
nominal point (TV_W(c) << thr), so their constraints are slack at the
optimum and contribute only to solver wall time.

Define an "active" window for composition c at threshold tau = c_target * m^2:
    active_W := { W :  TV_W(c) >= tau - delta }    (delta = safety margin)
where TV_W(c) is computed exactly from c (integer composition).

Strategy (sound)
----------------
  1. Solve SDP with active_W constraints only.
  2. If status is "infeasible" (Farkas certificate) -> prune confirmed.
     This is sound: a smaller constraint set is a LOOSER relaxation, so
     active-infeasible => full-infeasible => original-infeasible.
  3. If status is "optimal" (or anything else) -> we cannot conclude.
     Re-solve with the FULL constraint set (current always-full path).

Soundness sketch
----------------
  Let P_full be the SDP feasible set with all window constraints.
  Let P_act  be the SDP feasible set with only active_W constraints.
  Since active_W subseteq all_W, we have P_full subseteq P_act.
  So if P_act is empty (infeasible), P_full is also empty (infeasible),
  hence the original integer problem is infeasible -> safe prune.
  If P_act is nonempty, we cannot conclude P_full is nonempty
  (it could still be empty); so we re-solve with full constraints.

The "constraint generation / cutting plane" interpretation: we generate
the constraints likely to be tight (the active ones); if they're enough
to cut the cell empty, great; otherwise add the rest.

Comparison
----------
We benchmark on the FIRST 60 Q-survivors at (n_half=4, m=10, c=1.28, d=8):
- Full path: full SDP every time (current baseline).
- Active path: active SDP first, fall back to full only if not infeasible.

Soundness check: prune count must MATCH between the two paths (same set of
compositions identified as L-prunable). If active path prunes fewer or
mistakenly prunes a comp full-path doesn't, that's a failure.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, _enum_balanced_signs, prune_Q_one
from _L_bench import (_build_A_matrices, _detect_solver,
                       _shor_feasibility as _shor_full)

import cvxpy as cp


# ----------------------------------------------------------------------
# Compute TV_W(c) for ALL windows in one shot (vectorized)
# ----------------------------------------------------------------------
def compute_TV_W_at_c(c_int, A_mats, windows, n_half, m):
    """Return per-window TV_W(c) (in m^2 units, comparable to c_target*m^2)
    using x = m * a = c (integer composition values).

    m^2 * TV_W(a) = (1/(4n*ell)) * x^T A_W x.
    """
    x = c_int.astype(np.float64)
    n_d = float(n_half)
    out = np.empty(len(windows), dtype=np.float64)
    for wi, (A_mat, (ell, _)) in enumerate(zip(A_mats, windows)):
        v = float(x @ A_mat @ x)
        out[wi] = v / (4.0 * n_d * float(ell))   # m^2 * TV_W
    return out


# ----------------------------------------------------------------------
# Active-only Shor SDP (subset of windows)
# ----------------------------------------------------------------------
def _shor_feasibility_active(c_int, lo, hi, A_mats, windows, active_idx,
                              n_half, m, c_target, solver='MOSEK',
                              tol=1e-9, eps_margin=1e-9, verbose=False):
    """Test SDP feasibility using only active_idx window constraints.

    SOUND for pruning: active_W is a SUBSET of all_W, so active-infeasible
    implies full-infeasible (the cell with fewer constraints is a SUPERSET).
    """
    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

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

    # ONLY the active-window constraints
    for wi in active_idx:
        A_mat = A_mats[wi]
        (ell, _) = windows[wi]
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)

    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=verbose,
                        eps_abs=tol, eps_rel=tol, max_iter=300)
        elif solver == 'SCS':
            prob.solve(solver='SCS', verbose=verbose, eps=tol, max_iters=20000)
        else:
            return False, 'NO_SOLVER'
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    status = prob.status
    if status == 'infeasible':
        return True, status
    return False, status


def _make_cell(c_int, m):
    c = np.asarray(c_int, dtype=np.float64)
    lo = np.maximum(0.0, c - 1.0)
    hi = c + 1.0
    return lo, hi


# ----------------------------------------------------------------------
# Driver: generate Q-survivors at (n_half=4, m=10), time both paths
# ----------------------------------------------------------------------
def main():
    n_half = 4
    m = 10
    c_target = 1.28
    d = 2 * n_half
    n_target = 60               # benchmark on first 60 Q-survivors
    safety_pcts = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)   # delta = safety_pct * tau, sweep to find best

    actual_solver = _detect_solver('auto')
    print(f"[smoke] solver = {actual_solver}")
    print(f"[smoke] config: n_half={n_half}, m={m}, c_target={c_target}, d={d}")

    # Build LP scaffolding
    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"[smoke] n_win = {n_win}")

    # Warm up
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # Generate Q-survivors
    print(f"[smoke] generating Q-survivors ...")
    q_survs = []
    n_proc = 0
    S_half = 2 * n_half * m
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_surv_idx = np.where(sF)[0]
        for idx in f_surv_idx:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, margin=1e-9):
                q_survs.append(c_int.copy())
            if len(q_survs) >= n_target:
                break
        if len(q_survs) >= n_target:
            break
    print(f"[smoke] got {len(q_survs)} Q-survivors after scanning {n_proc} comps")

    # ========================================================================
    # PATH 1: full SDP (always include all n_win window constraints) -- baseline
    # ========================================================================
    print(f"\n[smoke] PATH 1 (FULL): running full SDP on {len(q_survs)} comps ...")
    full_t0 = time.time()
    full_results = []   # list of (pruned, status, time_sec)
    for k, c_int in enumerate(q_survs):
        t1 = time.time()
        pruned, status = _shor_full(c_int, *_make_cell(c_int, m), A_mats, windows,
                                       n_half, m, c_target, solver=actual_solver,
                                       tol=1e-9, eps_margin=1e-9)
        full_results.append((pruned, status, time.time() - t1))
    t_full_total = time.time() - full_t0
    n_full_pruned = sum(1 for r in full_results if r[0])
    print(f"[smoke] FULL: pruned {n_full_pruned}/{len(q_survs)} in {t_full_total:.2f}s "
          f"(median {1000*np.median([r[2] for r in full_results]):.0f}ms / SDP)")

    # ========================================================================
    # PATH 2: active-only first, fallback to full if not infeasible
    # ========================================================================
    # Sweep over safety_pcts to find the best one (delta = pct * tau)
    print(f"\n[smoke] PATH 2 (ACTIVE+FALLBACK): sweeping safety pct ...")
    cs_m2 = c_target * m * m

    # Precompute TV_W(c) for each Q-survivor
    tv_at_c = [compute_TV_W_at_c(c_int, A_mats, windows, n_half, m)
               for c_int in q_survs]

    best_path2 = None
    best_path2_total_time = float('inf')

    sweep_results = []
    for safety_pct in safety_pcts:
        delta = safety_pct * cs_m2
        thr_active = cs_m2 - delta   # window is active if TV_W(c) >= thr_active

        n_active_per_comp = []
        for tv_c in tv_at_c:
            n_act = int(np.sum(tv_c >= thr_active))
            n_active_per_comp.append(n_act)

        n_active_min = min(n_active_per_comp)
        n_active_max = max(n_active_per_comp)
        n_active_med = int(np.median(n_active_per_comp))
        print(f"  pct={safety_pct:.2f}  active_count: med={n_active_med} "
              f"min={n_active_min} max={n_active_max} (out of {n_win})")

        # Skip "no active windows" case (would fall back 100%)
        if n_active_med == 0:
            sweep_results.append({
                'safety_pct': safety_pct,
                'note': 'all comps had 0 active windows (all fall-backs)',
                'skipped': True,
            })
            continue

        # Run path 2
        path2_t0 = time.time()
        path2_results = []   # (pruned, status_active, fallback_taken, time_total)
        n_fallback = 0
        n_pruned_active_only = 0
        for k, c_int in enumerate(q_survs):
            tv_c = tv_at_c[k]
            active_idx = np.where(tv_c >= thr_active)[0].tolist()
            t1 = time.time()
            pruned, status = _shor_feasibility_active(
                c_int, *_make_cell(c_int, m), A_mats, windows, active_idx,
                n_half, m, c_target, solver=actual_solver,
                tol=1e-9, eps_margin=1e-9)
            t_active = time.time() - t1

            fallback = False
            t_full_part = 0.0
            status_full = None
            if not pruned:
                # fallback: full SDP
                fallback = True
                n_fallback += 1
                t2 = time.time()
                pruned_full, status_full = _shor_full(
                    c_int, *_make_cell(c_int, m), A_mats, windows,
                    n_half, m, c_target, solver=actual_solver,
                    tol=1e-9, eps_margin=1e-9)
                t_full_part = time.time() - t2
                pruned = pruned_full   # use full's verdict
            else:
                n_pruned_active_only += 1

            path2_results.append({
                'pruned': pruned,
                'active_status': status,
                'fallback': fallback,
                'fallback_status': status_full,
                't_active': t_active,
                't_full': t_full_part,
                't_total': t_active + t_full_part,
                'n_active': len(active_idx),
            })
        t_path2_total = time.time() - path2_t0

        n_path2_pruned = sum(1 for r in path2_results if r['pruned'])

        # Soundness check vs full path
        soundness_ok = True
        soundness_msg = ''
        for k, (rf, rp) in enumerate(zip(full_results, path2_results)):
            if rf[0] != rp['pruned']:
                soundness_ok = False
                soundness_msg += (f"  *** MISMATCH @ k={k}: full prune={rf[0]} "
                                    f"vs active+fb={rp['pruned']}\n")

        speedup = t_full_total / max(1e-9, t_path2_total)
        print(f"  pct={safety_pct:.2f}: total={t_path2_total:.2f}s "
              f"(active_only_pruned={n_pruned_active_only}, fallback={n_fallback}, "
              f"all_pruned={n_path2_pruned}/{len(q_survs)}); "
              f"speedup vs FULL = {speedup:.2f}x "
              f"{'OK' if soundness_ok else 'SOUNDNESS_FAIL'}")
        if not soundness_ok:
            print(soundness_msg)

        sweep_results.append({
            'safety_pct': safety_pct,
            'n_active_med': n_active_med,
            'n_active_min': n_active_min,
            'n_active_max': n_active_max,
            'n_active_avg': float(np.mean(n_active_per_comp)),
            'total_time': t_path2_total,
            'n_active_only_pruned': n_pruned_active_only,
            'n_fallback': n_fallback,
            'n_pruned_total': n_path2_pruned,
            'soundness_ok': soundness_ok,
            'speedup_vs_full': speedup,
            'med_t_active_ms': 1000.0 * float(np.median([r['t_active'] for r in path2_results])),
            'med_t_full_ms_on_fallback': (1000.0 * float(np.median([r['t_full'] for r in path2_results if r['fallback']])) if n_fallback > 0 else 0.0),
        })

        if soundness_ok and t_path2_total < best_path2_total_time:
            best_path2_total_time = t_path2_total
            best_path2 = sweep_results[-1]

    # Summary
    print(f"\n[smoke] Summary:")
    print(f"  Full baseline:    {t_full_total:.2f}s, {n_full_pruned}/{len(q_survs)} pruned")
    if best_path2 is not None:
        print(f"  Best active path: {best_path2['total_time']:.2f}s "
              f"({best_path2['safety_pct']*100:.0f}% safety), "
              f"{best_path2['n_pruned_total']}/{len(q_survs)} pruned, "
              f"speedup = {best_path2['speedup_vs_full']:.2f}x")
        if best_path2['speedup_vs_full'] >= 1.05:
            verdict = f"ACTIVE_CONSTRAINTS_SPEEDUP: {best_path2['speedup_vs_full']:.2f}x"
        else:
            verdict = (f"NO_GAIN (best ratio = {best_path2['speedup_vs_full']:.2f}x "
                       f"at pct={best_path2['safety_pct']:.2f})")
    else:
        print(f"  No usable active path (all unsound or all fallback)")
        verdict = "NO_GAIN"

    out = {
        'config': {
            'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
            'n_target': n_target, 'n_win': n_win, 'safety_pcts': list(safety_pcts),
            'solver': actual_solver,
            'n_q_survivors_actually_tested': len(q_survs),
        },
        'full_baseline': {
            'total_time': t_full_total,
            'n_pruned': n_full_pruned,
            'n_total': len(q_survs),
            'med_solve_ms': 1000.0 * float(np.median([r[2] for r in full_results])),
        },
        'sweep': sweep_results,
        'best_path2': best_path2,
        'verdict': verdict,
    }
    with open('_smoke_active_constraints_L.json', 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\n[smoke] wrote _smoke_active_constraints_L.json")
    print(f"\n{verdict}")


if __name__ == '__main__':
    main()

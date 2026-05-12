#!/usr/bin/env python
r"""
Optimized constraint-generation (CG) Lasserre sweep.

Approach 2 from the scalable Lasserre analysis: lazily add PSD window
localizing constraints.  Converges to the EXACT full Lasserre bound
with typically ~10-30 active windows (vs ~d² in full).

Key optimizations over lasserre_scalable.py solve_cg:
  1. Phase-1 direct optimization (min t, scalar windows only) → 1 solve
     instead of ~15 binary-search feasibility checks.
  2. Vectorized batch violation checking: stack all localizing matrices,
     compute all eigenvalues at once via np.linalg.eigvalsh on 3D array.
  3. Tight binary-search range: start from phase-1 lb, not 0.5.
  4. Aggressive + diverse window addition per round.
  5. Adaptive termination when bound improvement < tolerance.

Usage:
  python tests/lasserre_cg_sweep.py
  python tests/lasserre_cg_sweep.py --d 16 --order 3
  python tests/lasserre_cg_sweep.py --d 8 --order 3 --full  # compare to full
"""

import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
                          ObjectiveSense, SolutionStatus)
import time
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
from lasserre_scalable import (
    _precompute, _build_base_constraints, _add_psd_window,
)

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
}


# =====================================================================
# Vectorized batch violation checking
# =====================================================================

def _batch_check_violations(y_vals, t_val, P, ab_eiej_idx,
                            active_windows, tol=1e-6):
    """Check ALL non-active windows for PSD violations in one batch.

    Instead of looping over windows with per-window eigvalsh,
    builds a (n_check, n_loc, n_loc) array and calls eigvalsh once.

    Returns: list of (window_index, min_eigenvalue), sorted most-violated first.
    """
    n_loc = P['n_loc']
    n_win = P['n_win']
    M_mats = P['M_mats']

    if n_loc == 0 or ab_eiej_idx is None:
        return []

    t_pick_arr = np.array(P['t_pick'])

    # T-part: t_val * y[t_pick] reshaped to (n_loc, n_loc)
    L_t = t_val * y_vals[t_pick_arr].reshape(n_loc, n_loc)

    # Precompute y values at all ab+ei+ej positions: (n_loc, n_loc, d, d)
    safe_idx = np.clip(ab_eiej_idx, 0, len(y_vals) - 1)
    y_abij = y_vals[safe_idx]
    y_abij[ab_eiej_idx < 0] = 0.0

    # Identify non-active, non-zero windows
    check_indices = []
    M_stack_list = []
    for w in range(n_win):
        if w in active_windows:
            continue
        Mw = M_mats[w]
        if np.all(Mw == 0):
            continue
        check_indices.append(w)
        M_stack_list.append(Mw)

    if not check_indices:
        return []

    # Stack window matrices: (n_check, d, d)
    M_stack = np.stack(M_stack_list)  # (n_check, d, d)

    # Batch compute quadratic parts: (n_check, n_loc, n_loc)
    # L_q[w, a, b] = sum_{i,j} y_abij[a,b,i,j] * M_stack[w,i,j]
    L_q_all = np.einsum('abij,wij->wab', y_abij, M_stack)

    # Full localizing matrices: L_w = L_t - L_q
    L_all = L_t[np.newaxis, :, :] - L_q_all  # (n_check, n_loc, n_loc)

    # Symmetrize (numerical safety)
    L_all = 0.5 * (L_all + np.swapaxes(L_all, -2, -1))

    # Batch eigenvalue computation
    all_eigs = np.linalg.eigvalsh(L_all)  # (n_check, n_loc)
    min_eigs = all_eigs[:, 0]  # smallest eigenvalue per window

    # Collect violations
    violated_mask = min_eigs < -tol
    violations = [
        (check_indices[i], float(min_eigs[i]))
        for i in np.where(violated_mask)[0]
    ]

    # Sort by most violated (most negative eigenvalue first)
    violations.sort(key=lambda x: x[1])
    return violations


# =====================================================================
# Optimized CG solver
# =====================================================================

def solve_cg_optimized(d, c_target, order=3, n_bisect=15,
                       add_upper_loc=True, max_cg_rounds=20,
                       max_add_per_round=20, tol=1e-7,
                       verbose=True):
    """Optimized constraint-generation Lasserre solver.

    Phase 1: Direct optimization with scalar window constraints only.
             Solves min t s.t. t >= f_W(y) for all W, base PSD.
             Gets initial lb and optimal y* in ONE solver call.

    Phase 2: CG loop — check PSD violations at y*, add violated
             windows as PSD constraints, binary-search in tight range.
             Repeat until no violations or bound converges.

    Returns: dict with lb, proven, timing, n_active_windows, etc.
    """
    t_total = time.time()

    # ── Precompute ──
    P = _precompute(d, order, verbose)
    n_win = P['n_win']
    n_y = P['n_y']
    n_loc = P['n_loc']

    # Precompute ab_eiej_idx for violation checking
    ab_eiej_idx = None
    if P['AB_loc_hash'] is not None:
        EE_hash = P['bases'][:, None] + P['bases'][None, :]
        ABIJ_hash = (P['AB_loc_hash'][:, :, None, None]
                     + EE_hash[None, None, :, :])
        ab_eiej_idx = _hash_lookup(ABIJ_hash, P['sorted_h'], P['sort_o'])

    # ================================================================
    # Phase 1: Direct optimization (scalar window constraints only)
    # ================================================================
    if verbose:
        print(f"\n  Phase 1: Direct optimization (scalar windows)...",
              flush=True)

    t_p1 = time.time()
    mdl_lin = Model("cg_phase1")
    y_lin = mdl_lin.variable("y", n_y, Domain.greaterThan(0.0))
    t_var = mdl_lin.variable("t")

    _build_base_constraints(mdl_lin, y_lin, P, add_upper_loc, verbose)

    # Scalar window constraints: t >= f_W(y) for all W
    F_mosek = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all = Expr.mul(F_mosek, y_lin)
    ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep = Expr.flatten(Expr.mul(ones_col, t_var))
    mdl_lin.constraint("win_scalar", Expr.sub(t_rep, f_all),
                       Domain.greaterThan(0.0))

    mdl_lin.objective(ObjectiveSense.Minimize, t_var)
    mdl_lin.solve()

    pstatus = mdl_lin.getPrimalSolutionStatus()
    if pstatus not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
        mdl_lin.dispose()
        raise RuntimeError(f"Phase 1 failed: primal status = {pstatus}")

    lb_linear = t_var.level()[0]
    y_star = np.array(y_lin.level())
    p1_time = time.time() - t_p1

    mdl_lin.dispose()

    if verbose:
        print(f"    lb_linear = {lb_linear:.10f}  ({p1_time:.2f}s)")

    # Check PSD violations at y*
    violations = _batch_check_violations(
        y_star, lb_linear, P, ab_eiej_idx, set())

    if verbose:
        print(f"    PSD violations: {len(violations)}", flush=True)
        if violations:
            print(f"    Worst: min_eig = {violations[0][1]:.6e}")

    if not violations:
        # No violations → linear bound IS the full Lasserre bound
        elapsed = time.time() - t_total
        proven = lb_linear >= c_target - 1e-6
        if verbose:
            print(f"\n  No PSD violations — linear = exact!")
            print(f"  Lower bound: {lb_linear:.10f}")
            print(f"  Total time: {elapsed:.1f}s")
            if proven:
                print(f"  *** PROVEN: val({d}) >= {c_target} ***")
        return {
            'lb': lb_linear, 'proven': proven,
            'elapsed': elapsed, 'p1_time': p1_time,
            'd': d, 'order': order, 'mode': 'cg',
            'n_active_windows': 0, 'n_cg_rounds': 0,
            'lb_linear': lb_linear,
        }

    # ================================================================
    # Phase 2: CG loop with binary search
    # ================================================================
    if verbose:
        print(f"\n  Phase 2: Constraint generation...", flush=True)

    t_p2 = time.time()
    mdl = Model("cg_phase2")
    y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
    t_param = mdl.parameter("t")

    _build_base_constraints(mdl, y, P, add_upper_loc, False)

    # Scalar window constraints (same as phase 1)
    F_mosek2 = Matrix.sparse(n_win, n_y, P['f_r'], P['f_c'], P['f_v'])
    f_all2 = Expr.mul(F_mosek2, y)
    ones_col2 = Matrix.dense(n_win, 1, [1.0] * n_win)
    t_rep2 = Expr.flatten(Expr.mul(ones_col2,
                          Expr.reshape(t_param, 1, 1)))
    mdl.constraint("win_scalar", Expr.sub(t_rep2, f_all2),
                   Domain.greaterThan(0.0))

    mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    active_windows = set()
    best_lb = lb_linear
    build_time = time.time() - t_p2

    def check_feasible(t_val):
        t_param.setValue(t_val)
        try:
            mdl.solve()
            ps = mdl.getPrimalSolutionStatus()
            return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
        except Exception:
            return False

    for cg_round in range(max_cg_rounds):
        # Add most-violated windows from last violation check
        n_add = min(max_add_per_round, len(violations))
        added_this_round = []
        for w, min_eig in violations[:n_add]:
            if w not in active_windows:
                _add_psd_window(mdl, y, t_param, w, P, ab_eiej_idx)
                active_windows.add(w)
                added_this_round.append((w, min_eig))

        if verbose:
            print(f"\n  CG round {cg_round + 1}: "
                  f"+{len(added_this_round)} PSD windows "
                  f"(total: {len(active_windows)})", flush=True)
            for w, eig in added_this_round[:5]:
                ell, s = P['windows'][w]
                print(f"    window ({ell},{s}): min_eig={eig:.6e}")

        # Binary search in TIGHT range [best_lb, best_lb * 1.05 + 0.1]
        lo = best_lb
        hi = best_lb * 1.05 + 0.1

        # Ensure hi is feasible
        while not check_feasible(hi):
            hi *= 1.5
            if hi > 100:
                break
        if hi > 100 and not check_feasible(hi):
            if verbose:
                print(f"    WARNING: infeasible up to t={hi:.2f}")
            break

        # Binary search
        n_steps = n_bisect
        for step in range(n_steps):
            mid = (lo + hi) / 2
            if check_feasible(mid):
                hi = mid
            else:
                lo = mid

        lb_round = lo
        improvement = lb_round - best_lb
        best_lb = max(best_lb, lb_round)

        if verbose:
            print(f"    lb = {lb_round:.10f} "
                  f"(+{improvement:.2e} over previous)", flush=True)

        # Extract y* at feasible boundary
        t_param.setValue(hi)
        mdl.solve()
        ps = mdl.getPrimalSolutionStatus()
        if ps not in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
            if verbose:
                print(f"    Could not extract y* (status={ps})")
            break

        y_star = np.array(y.level())

        # Check for remaining violations
        violations = _batch_check_violations(
            y_star, hi, P, ab_eiej_idx, active_windows)

        if verbose:
            print(f"    Remaining violations: {len(violations)}", flush=True)
            if violations:
                print(f"    Worst: min_eig = {violations[0][1]:.6e}")

        if not violations:
            if verbose:
                print(f"    *** Converged — no more violations ***")
            break

        # Adaptive termination: if improvement is tiny, stop
        if improvement < tol and cg_round >= 2:
            if verbose:
                print(f"    Improvement {improvement:.2e} < tol {tol:.2e}"
                      f" — stopping.")
            break

    mdl.dispose()

    elapsed = time.time() - t_total
    p2_time = time.time() - t_p2
    proven = best_lb >= c_target - 1e-6

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  Phase 1 (linear): {lb_linear:.10f}  ({p1_time:.1f}s)")
        print(f"  Phase 2 (CG):     {best_lb:.10f}  ({p2_time:.1f}s)")
        print(f"  CG lift:          +{best_lb - lb_linear:.2e}")
        print(f"  Active windows:   {len(active_windows)}/{n_win}")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Margin over {c_target}: {best_lb - c_target:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")
        print(f"  {'='*50}")

    return {
        'lb': best_lb, 'proven': proven,
        'elapsed': elapsed, 'p1_time': p1_time, 'p2_time': p2_time,
        'd': d, 'order': order, 'mode': 'cg',
        'n_active_windows': len(active_windows),
        'n_cg_rounds': cg_round + 1 if violations is not None else 0,
        'lb_linear': lb_linear,
        'n_win_total': n_win,
    }


# =====================================================================
# Full Lasserre (for comparison)
# =====================================================================

def solve_full_for_comparison(d, c_target, order, n_bisect=12):
    """Solve full Lasserre for comparison. Returns lb or None on OOM."""
    try:
        from lasserre_fusion import solve_lasserre_fusion
        r = solve_lasserre_fusion(d, c_target, order=order,
                                   n_bisect=n_bisect, verbose=False)
        return r['lb']
    except Exception as e:
        return None


# =====================================================================
# Sweep
# =====================================================================

def gap_closure(lb, d):
    v = val_d_known.get(d, 0)
    if v <= 1:
        return 0
    return max(0, (lb - 1.0) / (v - 1.0) * 100)


def run_one(desc, d, order, c_target=1.28, compare_full=False, **kwargs):
    """Run one CG configuration and optionally compare to full Lasserre."""
    print(f"\n{'#'*65}")
    print(f"# {desc}")
    print(f"# d={d}, order={order} (L{order}, degree {2*order})")
    print(f"{'#'*65}\n", flush=True)

    t0 = time.time()
    try:
        r = solve_cg_optimized(d, c_target, order=order, **kwargs)
        elapsed = time.time() - t0
        lb = r['lb']
        gc = gap_closure(lb, d)

        result = {
            'desc': desc, 'lb': lb, 'gc': gc, 'time': elapsed,
            'd': d, 'order': order, 'status': 'ok',
            'n_active': r['n_active_windows'],
            'n_total': r['n_win_total'],
            'lb_linear': r['lb_linear'],
            'n_rounds': r['n_cg_rounds'],
        }

        # Optionally compare to full Lasserre
        if compare_full:
            print(f"\n  Comparing to full Lasserre...", flush=True)
            lb_full = solve_full_for_comparison(d, c_target, order)
            if lb_full is not None:
                result['lb_full'] = lb_full
                diff = abs(lb - lb_full)
                print(f"  Full Lasserre lb: {lb_full:.10f}")
                print(f"  CG lb:            {lb:.10f}")
                print(f"  Difference:       {diff:.2e}")
                if diff < 1e-4:
                    print(f"  *** MATCH (within 1e-4) ***")
                else:
                    print(f"  *** MISMATCH ***")
            else:
                result['lb_full'] = None
                print(f"  Full Lasserre: OOM or failed")

        v = val_d_known.get(d, 0)
        v_str = f"{v:.3f}" if v else "?"
        print(f"\n  => lb={lb:.6f}, val({d})={v_str}, "
              f"gap_closure={gc:.1f}%, active_windows={r['n_active_windows']}, "
              f"time={elapsed:.1f}s\n")

        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return {
            'desc': desc, 'lb': 0, 'gc': 0, 'time': elapsed,
            'd': d, 'order': order, 'status': str(e),
            'n_active': 0, 'n_total': 0, 'lb_linear': 0, 'n_rounds': 0,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimized CG Lasserre sweep (Approach 2)")
    parser.add_argument('--d', type=int, default=0,
                        help="Run single d (0 = full sweep)")
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--full', action='store_true',
                        help="Compare to full Lasserre where feasible")
    parser.add_argument('--bisect', type=int, default=15)
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--max-add', type=int, default=20)
    parser.add_argument('--no-upper-loc', dest='upper_loc',
                        action='store_false', default=True)
    args = parser.parse_args()

    print("=" * 65)
    print("OPTIMIZED CONSTRAINT-GENERATION LASSERRE SWEEP")
    print("Approach 2: exact full Lasserre bound via lazy PSD addition")
    print("=" * 65)
    print()

    cg_kw = dict(
        n_bisect=args.bisect,
        max_cg_rounds=args.max_rounds,
        max_add_per_round=args.max_add,
        add_upper_loc=args.upper_loc,
    )

    if args.d > 0:
        # Single configuration
        run_one(f"L{args.order} d={args.d}",
                args.d, args.order, args.c_target,
                compare_full=args.full, **cg_kw)
        return

    # Full sweep
    results = []

    # --- Tier 1: Sanity checks + comparison to full ---
    results.append(run_one(
        "L2 d=4 (sanity)", 4, 2,
        compare_full=True, **cg_kw))
    results.append(run_one(
        "L3 d=4 (sanity)", 4, 3,
        compare_full=True, **cg_kw))
    results.append(run_one(
        "L2 d=8 (sanity)", 8, 2,
        compare_full=True, **cg_kw))
    results.append(run_one(
        "L3 d=8 (compare full)", 8, 3,
        compare_full=args.full, **cg_kw))

    # --- Tier 2: Medium scale ---
    results.append(run_one(
        "L3 d=10", 10, 3, **cg_kw))
    results.append(run_one(
        "L3 d=12", 12, 3, **cg_kw))
    results.append(run_one(
        "L4 d=8", 8, 4, **cg_kw))

    # --- Tier 3: Previously infeasible ---
    results.append(run_one(
        "L3 d=16 (was OOM@60GB full)", 16, 3, **cg_kw))
    results.append(run_one(
        "L4 d=10 (was OOM@>1TB full)", 10, 4, **cg_kw))

    # --- Tier 4: Frontier ---
    results.append(run_one(
        "L4 d=16 (new territory)", 16, 4, **cg_kw))

    # --- Summary table ---
    print("\n" + "=" * 100)
    print(f"{'Config':<35} {'lb_cg':>10} {'lb_lin':>10} "
          f"{'CG lift':>10} {'val(d)':>8} {'Gap%':>7} "
          f"{'Win':>6} {'Rnds':>5} {'Time':>7}")
    print("-" * 100)
    for r in results:
        v = val_d_known.get(r['d'], 0)
        v_str = f"{v:.3f}" if v else "?"
        lb_str = f"{r['lb']:.6f}" if r['lb'] > 0 else "---"
        ll_str = f"{r['lb_linear']:.6f}" if r.get('lb_linear', 0) > 0 else "---"
        lift = r['lb'] - r.get('lb_linear', 0) if r['lb'] > 0 else 0
        lift_str = f"{lift:.2e}" if lift > 0 else "0"
        gc_str = f"{r['gc']:.1f}%" if r['lb'] > 0 else "---"
        t_str = f"{r['time']:.1f}s"
        win_str = f"{r['n_active']}/{r['n_total']}" if r['n_total'] > 0 else "---"
        rnd_str = str(r['n_rounds']) if r.get('n_rounds', 0) > 0 else "0"
        status = "" if r['status'] == 'ok' else f" [{r['status'][:20]}]"
        print(f"{r['desc']:<35} {lb_str:>10} {ll_str:>10} "
              f"{lift_str:>10} {v_str:>8} {gc_str:>7} "
              f"{win_str:>6} {rnd_str:>5} {t_str:>7}{status}")

    # Full Lasserre comparison column (if available)
    has_full = any(r.get('lb_full') is not None for r in results)
    if has_full:
        print()
        print(f"{'Config':<35} {'lb_cg':>10} {'lb_full':>10} {'diff':>10}")
        print("-" * 65)
        for r in results:
            if r.get('lb_full') is not None:
                diff = abs(r['lb'] - r['lb_full'])
                match = "OK" if diff < 1e-4 else "MISMATCH"
                print(f"{r['desc']:<35} {r['lb']:.6f}   "
                      f"{r['lb_full']:.6f}   {diff:.2e}  {match}")

    print("=" * 100)


if __name__ == '__main__':
    main()

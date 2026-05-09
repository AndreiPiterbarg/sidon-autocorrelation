"""Tight split-cell SDP for L-survivors at d=10 (n=5, m=5, c=1.28).

EXTENDS `_smoke_split_cell_SDP.py` with:
  1) DIRECT MOSEK API (`l_direct.prune_L_direct`-style; ~17 ms / SDP at d=10)
     reused per-worker via a shared `mosek.Env` (no CVXPY overhead).
  2) RECURSIVE BINARY SPLIT to depth K (default K=2):  whenever a sub-cell
     SDP is `optimal` (looks feasible), split it again into 2^d sub-sub-cells
     of half the width on each axis, and SDP each.  Iterate to depth K.
     If at depth K we still find `optimal`, accept the parent as NOT
     split-prunable (sound — only Farkas-certified infeasibility prunes).
  3) NO early termination: enumerate ALL feasible sub-cells at depth 1 in
     order to know exactly which ones need recursion.
  4) Built on `_smoke_split_cell_SDP`'s worker pool.

SOUNDNESS
=========
Each prune is backed by Farkas/dual-infeasibility certificates from MOSEK
on every leaf sub-cell.  At depth K, if any leaf returns `optimal` we
declare the parent NOT split-prunable.  Sub-cells partition the parent
cell (their union covers it modulo measure-zero faces), so:

  ALL leaves INFEASIBLE  =>  parent INFEASIBLE  =>  composition pruned.

USAGE
=====
  python _smoke_split_cell_SDP_tight.py
  N_WORKERS=12 K_DEPTH=2 python _smoke_split_cell_SDP_tight.py
  TARGET_ONLY_STUCK=1 python _smoke_split_cell_SDP_tight.py

Env vars:
  N_WORKERS         number of worker processes (default = cores - 2, max 12)
  K_DEPTH           recursion depth (default 2)
  TARGET_ONLY_STUCK if set, only test the 2 stuck cells; faster smoke run
  WALL_BUDGET_MIN   wall-clock budget in minutes (default 55)
"""
from __future__ import annotations
import os
import sys
import time
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))


# ----------------------------------------------------------------------
# Direct-MOSEK SDP feasibility on an ARBITRARY box (lo, hi).
# Mirrors `cloninger-steinerberger/cpu/l_direct.prune_L_direct` but takes
# `lo, hi` as args so we can use it on sub-cells / sub-sub-cells.
# ----------------------------------------------------------------------
def _shor_feasibility_box(c_int, lo, hi, A_mats, windows, n_half, m,
                            c_target, env, tol=1e-9, eps_margin=1e-9):
    """Direct-MOSEK Shor SDP feasibility on `lo <= x <= hi`.

    Returns:
        (pruned: bool, status_str) where pruned iff status is `prim_infeas_cer`
        (Farkas certificate).
    """
    import mosek
    d = len(c_int)
    bar_dim = d + 1
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    # Pre-screen: box-sum infeasibility (sum x must equal nm)
    s_lo = float(np.sum(lo))
    s_hi = float(np.sum(hi))
    if nm < s_lo - 1e-9 or nm > s_hi + 1e-9:
        return True, 'box_sum_pre_infeasible'

    def _coeffs(alpha_const, x_coef, X_coef_lower):
        subi, subj, val = [], [], []
        if alpha_const != 0.0:
            subi.append(0); subj.append(0); val.append(float(alpha_const))
        for i in range(d):
            cf = x_coef[i]
            if cf != 0.0:
                subi.append(i + 1); subj.append(0); val.append(0.5 * float(cf))
        for (i, j), cf in X_coef_lower.items():
            if cf == 0.0:
                continue
            if i == j:
                subi.append(i + 1); subj.append(j + 1); val.append(float(cf))
            else:
                ii, jj = (i, j) if i > j else (j, i)
                subi.append(ii + 1); subj.append(jj + 1); val.append(0.5 * float(cf))
        return subi, subj, val

    try:
        with env.Task(0, 0) as task:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, tol)
            task.putintparam(mosek.iparam.intpnt_max_iterations, 200)
            task.putintparam(mosek.iparam.log, 0)
            task.putintparam(mosek.iparam.num_threads, 1)

            task.appendbarvars([bar_dim])
            task.putobjsense(mosek.objsense.minimize)

            def _add(subi, subj, vals, bk, blk, buk):
                cidx = task.getnumcon()
                task.appendcons(1)
                if len(subi) > 0:
                    aid = task.appendsparsesymmat(bar_dim, subi, subj, vals)
                    task.putbaraij(cidx, 0, [aid], [1.0])
                task.putconbound(cidx, bk, blk, buk)
                return cidx

            # Y[0,0] = 1
            sI, sJ, sV = _coeffs(1.0, np.zeros(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

            # Box: lo[i] <= x_i <= hi[i]
            for i in range(d):
                xc = np.zeros(d); xc[i] = 1.0
                sI, sJ, sV = _coeffs(0.0, xc, {})
                _add(sI, sJ, sV, mosek.boundkey.ra, lo[i], hi[i])

            # Sum: sum x = nm
            sI, sJ, sV = _coeffs(0.0, np.ones(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, nm, nm)

            # Diag McCormick
            for i in range(d):
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.ra,
                     lo[i] * lo[i], hi[i] * hi[i])
                xc = np.zeros(d); xc[i] = -2.0 * lo[i]
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -lo[i] * lo[i], 0.0)
                xc = np.zeros(d); xc[i] = -2.0 * hi[i]
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -hi[i] * hi[i], 0.0)
                xc = np.zeros(d); xc[i] = -(lo[i] + hi[i])
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.up, 0.0, -lo[i] * hi[i])

            # Off-diag RLT
            for i in range(d):
                for j in range(i + 1, d):
                    li, lj = lo[i], lo[j]
                    ui, uj = hi[i], hi[j]
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -li
                    sI, sJ, sV = _coeffs(li * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(ui * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -li
                    sI, sJ, sV = _coeffs(li * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(ui * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

            # Window constraints: Tr(A_W X) <= 4n*ell*(c_target*m^2 + eps_margin*m^2)
            for A_mat, (ell, _) in zip(A_mats, windows):
                thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
                Xc = {}
                for ii in range(d):
                    Xc[(ii, ii)] = float(A_mat[ii, ii])
                    for jj in range(ii):
                        Xc[(ii, jj)] = 2.0 * float(A_mat[ii, jj])
                sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.up, -1e30, thr)

            try:
                task.optimize()
            except mosek.Error as e:
                return False, f"optimize-error: {e}"

            try:
                solsta = task.getsolsta(mosek.soltype.itr)
            except mosek.Error:
                return False, "getsolsta-error"

            if solsta == mosek.solsta.prim_infeas_cer:
                return True, "infeasible"
            if solsta == mosek.solsta.optimal:
                return False, "optimal"
            return False, f"solsta={solsta}"
    except Exception as e:
        return False, f"EXC:{type(e).__name__}"


# ----------------------------------------------------------------------
# Worker initializer + shared cache (per process).
# ----------------------------------------------------------------------
_WORKER_CACHE = {}

def _worker_init():
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _here)
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger'))
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger', 'cpu'))


def _get_cache(d, n_half, m, c_target):
    """Lazy: build windows, A_mats, env once per worker process."""
    global _WORKER_CACHE
    key = ('cache', d, n_half, m, c_target)
    if key not in _WORKER_CACHE:
        from _Q_bench import _build_windows
        from _L_bench import _build_A_matrices
        import mosek
        windows, _ = _build_windows(d)
        A_mats = _build_A_matrices(d, windows)
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass
        _WORKER_CACHE[key] = (windows, A_mats, env)
    return _WORKER_CACHE[key]


# ----------------------------------------------------------------------
# Build a sub-cell box from depth-K binary refinement.
# At depth 1 with sigma in {-1,+1}^d we have the standard half-cell.
# At depth K with k in [0, 2^K)^d we further sub-divide the sigma-cell
# uniformly, producing a box of width (1/2^K) of the parent cell width
# (which itself is the half of the original cell width 1/m -> 1/(m*2^K)).
# ----------------------------------------------------------------------
def _subcell_box(c_int, depth_path):
    """Build (lo, hi) for a sub-cell given a "depth path".

    depth_path: list of length K+1; depth_path[0] is sigma in {+1,-1}^d
                (the depth-1 sub-cell), and for k >= 1, depth_path[k] is
                an array in {0,1}^d that further halves the previous box.

    For depth K (no recursion): depth_path = [sigma] (length 1), gives
    the standard half-cell as in the baseline.
    """
    d = len(c_int)
    sigma = depth_path[0]
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            # δ_i ≥ 0  =>  x_i ≤ c_i, lo = max(0, c-1), hi = c
            lo[i] = max(0.0, ci - 1.0)
            hi[i] = ci
        else:
            lo[i] = ci
            hi[i] = ci + 1.0
    # Further halving for each refinement step
    for k_path in depth_path[1:]:
        for i in range(d):
            mid = 0.5 * (lo[i] + hi[i])
            if k_path[i] == 0:
                hi[i] = mid
            else:
                lo[i] = mid
    return lo, hi


# ----------------------------------------------------------------------
# Worker function: test one sub-cell with optional recursion to depth K.
#
# Returns: (path_id, depth_label, pruned, status, t_seconds)
# where `path_id` is a frozen tuple identifying the sub-cell.
# ----------------------------------------------------------------------
def _worker_one_path(args):
    """Test SDP feasibility on the leaf sub-cell `path`."""
    path_pickle, c_int_list, n_half, m, c_target = args
    c_int = np.asarray(c_int_list, dtype=np.int32)
    d = len(c_int)
    windows, A_mats, env = _get_cache(d, n_half, m, c_target)

    # Reconstruct depth_path
    depth_path = [np.asarray(p, dtype=np.int8) for p in path_pickle]

    lo, hi = _subcell_box(c_int, depth_path)
    t0 = time.time()
    pruned, status = _shor_feasibility_box(c_int, lo, hi, A_mats, windows,
                                              n_half, m, c_target, env,
                                              tol=1e-9, eps_margin=1e-9)
    t_one = time.time() - t0
    return path_pickle, bool(pruned), str(status), float(t_one)


# ----------------------------------------------------------------------
# Recursive split-prune driver for one composition.
# ----------------------------------------------------------------------
def split_prune_recursive(c_int, n_half, m, c_target, k_depth=2,
                            n_workers=12, max_recursive_subcells=20000,
                            verbose=True):
    """Test split-prune at depth K via parallel worker pool.

    Algorithm:
      Frontier := {(σ,) : σ ∈ {-1,+1}^d}    # depth-1 sub-cells (1024)
      For depth in [1, K]:
          run SDP on every cell in `frontier` in parallel
          frontier_next := []
          for each cell:
              if SDP infeasible (Farkas):  this branch is pruned
              else (status=optimal or numerical issue): keep in frontier_next
                  if depth < K: split into 2^d sub-sub-cells
          frontier := frontier_next
          if frontier empty: SPLIT-PRUNED  (return True)
      # At depth K, if frontier still nonempty: NOT split-prunable.
      # (We accept these as feasible — sound termination.)

    Returns dict with split_pruned, n_inf, n_opt_at_leaf, t_total, etc.
    """
    d = len(c_int)
    c_int_list = c_int.tolist()
    t_global = time.time()

    # Initial frontier: 2^d depth-1 sub-cells (sigma vectors)
    sigmas = list(product([1, -1], repeat=d))
    frontier = [(np.asarray(s, dtype=np.int8),) for s in sigmas]

    stats_per_depth = []  # (depth, n_total, n_inf, n_opt, n_other, wall_t)

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_worker_init) as ex:
        for depth in range(1, k_depth + 1):
            n_total = len(frontier)
            t_depth = time.time()
            if verbose:
                print(f"      depth={depth}: {n_total} sub-cells to test "
                      f"(n_workers={n_workers})")

            # Defensive cap: at depth 2 with many sub-cells optimal at depth 1,
            # we could blow up. If the would-be frontier is too big, give up
            # cleanly (parent NOT split-prunable; sound termination).
            if n_total > max_recursive_subcells:
                if verbose:
                    print(f"      ABORT: frontier too large ({n_total} > "
                          f"{max_recursive_subcells}); accept parent NOT split-prunable")
                stats_per_depth.append({
                    'depth': depth, 'n_total': n_total, 'n_inf': 0,
                    'n_opt': 0, 'n_other': 0, 'wall_t': 0.0, 'aborted': True
                })
                return {
                    'split_pruned': False, 'aborted': True,
                    'k_reached': depth - 1,
                    'stats_per_depth': stats_per_depth,
                    't_total': time.time() - t_global,
                }

            # Build args list
            args_list = []
            for path in frontier:
                path_pickle = tuple(arr.tolist() for arr in path)
                args_list.append((path_pickle, c_int_list, n_half, m, c_target))

            futs = {ex.submit(_worker_one_path, a): i
                     for i, a in enumerate(args_list)}

            n_inf = 0
            n_opt = 0
            n_other = 0
            statuses = {}
            next_optimal_paths = []
            completed = 0
            t_solve = 0.0
            for fut in as_completed(futs):
                path_pickle, pruned, status, t_one = fut.result()
                completed += 1
                t_solve += t_one
                statuses[status] = statuses.get(status, 0) + 1
                if pruned:
                    n_inf += 1
                elif status == 'optimal':
                    n_opt += 1
                    next_optimal_paths.append([np.asarray(p, dtype=np.int8)
                                                 for p in path_pickle])
                else:
                    # Numerical/MOSEK error — to remain SOUND, we must not
                    # claim prune. Treat as feasible for the purpose of
                    # this depth — keep in next frontier (so it gets a
                    # finer split next round, or accept feasibility at K).
                    n_other += 1
                    next_optimal_paths.append([np.asarray(p, dtype=np.int8)
                                                 for p in path_pickle])
                if verbose and completed % 256 == 0:
                    print(f"        ... {completed}/{n_total}  "
                          f"(inf={n_inf}, opt={n_opt}, oth={n_other}, "
                          f"t={time.time()-t_depth:.1f}s)")

            wall_t = time.time() - t_depth
            stats_per_depth.append({
                'depth': depth, 'n_total': n_total, 'n_inf': n_inf,
                'n_opt': n_opt, 'n_other': n_other, 'wall_t': wall_t,
                't_solve_sum': t_solve,
                'statuses': statuses,
            })
            if verbose:
                print(f"      depth={depth} done: inf={n_inf}, opt={n_opt}, "
                      f"oth={n_other}, wall={wall_t:.1f}s, "
                      f"sum_solve_t={t_solve:.1f}s, "
                      f"statuses={statuses}")

            # If no surviving (optimal or numerical) sub-cell, parent is pruned.
            if len(next_optimal_paths) == 0:
                return {
                    'split_pruned': True, 'aborted': False,
                    'k_reached': depth,
                    'stats_per_depth': stats_per_depth,
                    't_total': time.time() - t_global,
                }

            # Build the next frontier: refine each surviving cell into 2^d.
            if depth < k_depth:
                next_frontier = []
                halvings = list(product([0, 1], repeat=d))
                for path in next_optimal_paths:
                    for h in halvings:
                        new_path = list(path) + [np.asarray(h, dtype=np.int8)]
                        next_frontier.append(tuple(new_path))
                frontier = next_frontier
            else:
                # At max depth and still have un-pruned cells -> NOT split-prunable.
                pass

    # If we get here, depth == k_depth and frontier still has un-pruned cells
    return {
        'split_pruned': False, 'aborted': False,
        'k_reached': k_depth,
        'stats_per_depth': stats_per_depth,
        't_total': time.time() - t_global,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    n_half = 5
    m = 5
    c_target = 1.28
    d = 2 * n_half
    n_workers = int(os.environ.get('N_WORKERS',
                                   max(1, min(12, (os.cpu_count() or 4) - 2))))
    k_depth = int(os.environ.get('K_DEPTH', 2))
    target_only_stuck = bool(int(os.environ.get('TARGET_ONLY_STUCK', '0')))
    wall_budget_min = float(os.environ.get('WALL_BUDGET_MIN', 55))
    max_recursive_subcells = int(os.environ.get('MAX_RECURSIVE_SUBCELLS', 20000))

    cache = os.path.join(_HERE, '_smoke_split_cell_l_survivors.json')
    if not os.path.exists(cache):
        raise RuntimeError(f"Need cached L-survivors at {cache}; "
                           "run _smoke_split_cell_SDP.py first.")
    with open(cache) as fp:
        data = json.load(fp)
    assert (data['n_half'] == n_half and data['m'] == m
            and data['c_target'] == c_target)
    l_survivors = [np.asarray(c, dtype=np.int32) for c in data['l_survivors']]

    # Optional: target only the 2 known stuck cells.
    stuck_cells = [
        np.array([20, 6, 7, 8, 9, 9, 8, 7, 6, 20], dtype=np.int32),
        np.array([21, 5, 7, 8, 9, 9, 8, 7, 5, 21], dtype=np.int32),
    ]
    if target_only_stuck:
        target = stuck_cells
    else:
        target = l_survivors

    print(f"Solver: MOSEK (direct API), n_workers: {n_workers}")
    print(f"K_DEPTH: {k_depth}, MAX_RECURSIVE_SUBCELLS: {max_recursive_subcells}")
    print(f"WALL_BUDGET_MIN: {wall_budget_min}, "
          f"TARGET_ONLY_STUCK: {target_only_stuck}")
    print(f"\nProcessing {len(target)} compositions at d={d}")

    results = []
    n_split_pruned = 0
    t_start = time.time()
    for i, c_int in enumerate(target):
        elapsed_min = (time.time() - t_start) / 60.0
        if elapsed_min > wall_budget_min:
            print(f"\nWall budget {wall_budget_min} min exceeded at survivor {i+1}; "
                  f"stopping.")
            break
        print(f"\n  [{i+1}/{len(target)}] c={c_int.tolist()}  "
              f"(elapsed={elapsed_min:.1f} min)")
        out = split_prune_recursive(c_int, n_half, m, c_target,
                                       k_depth=k_depth, n_workers=n_workers,
                                       max_recursive_subcells=max_recursive_subcells,
                                       verbose=True)
        if out['split_pruned']:
            n_split_pruned += 1
            print(f"    >>> SPLIT-PRUNED at depth {out['k_reached']} "
                  f"({out['t_total']:.1f}s)")
        else:
            print(f"    >>> NOT split-pruned (k_reached={out['k_reached']}, "
                  f"aborted={out['aborted']})  ({out['t_total']:.1f}s)")
        results.append({
            'c_int': c_int.tolist(),
            'split_pruned': bool(out['split_pruned']),
            'k_reached': int(out['k_reached']),
            'aborted': bool(out['aborted']),
            'stats_per_depth': out['stats_per_depth'],
            't_total': float(out['t_total']),
        })

    total_t = time.time() - t_start
    print(f"\n\n=========================================================")
    print(f"FINAL: split-pruned {n_split_pruned} of {len(results)} "
          f"(target={len(target)}) at d={d}")
    print(f"Total wall: {total_t:.1f}s ({total_t/60:.1f} min)")
    print(f"=========================================================\n")

    out_path = os.path.join(_HERE, '_smoke_split_cell_SDP_tight.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
            'solver': 'MOSEK_direct',
            'n_workers': n_workers,
            'k_depth': k_depth,
            'max_recursive_subcells': max_recursive_subcells,
            'target_only_stuck': target_only_stuck,
            'wall_budget_min': wall_budget_min,
            'n_target': len(target),
            'n_tested': len(results),
            'n_split_pruned': n_split_pruned,
            'total_time_s': float(total_t),
            'results': results,
        }, fp, indent=2)
    print(f"Wrote {out_path}")
    return n_split_pruned, len(results)


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()
    main()

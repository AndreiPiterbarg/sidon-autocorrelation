"""Smoke test: cell-splitting SDP for L-survivors at d=10 (n=5, m=5, c=1.28).

PREMISE
=======
At each L-survivor cell (composition c, integer length d, sum 4nm), the
parent SDP uses box  max(0,c_i-1) <= x_i <= c_i+1.  Now SPLIT the cell into
sub-cells indexed by σ ∈ {-1,+1}^d.  For sub-cell σ:
    σ_i = +1  =>  δ_i ≥ 0  =>  x_i ≤ c_i, so  lo=max(0, c_i-1), hi=c_i.
    σ_i = -1  =>  δ_i ≤ 0  =>  x_i ≥ c_i, so  lo=c_i, hi=c_i+1.

The half-cell SDP is a TIGHTER feasibility test (smaller box).  If ALL sub-cells
SDP-INFEASIBLE, the parent cell is too — sound, by union of relaxations.

We enumerate all 2^d = 1024 sub-cells per L-survivor at d=10.

PARALLELIZATION
===============
Each sub-cell SDP is independent.  Use ProcessPoolExecutor with N=12 workers
to parallelize the 1024 sub-cells per survivor.  Sequential test took 980s
per survivor; parallel target is ~80s.

USAGE: python _smoke_split_cell_SDP.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

# We import these only in the main process — workers re-import on demand.
from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, _shor_feasibility, prune_L_one, _detect_solver


# ---------------------------------------------------------------------------
# Step 1: enumerate (and cache) the L-survivors at (n_half=5, m=5, c=1.28)
# ---------------------------------------------------------------------------
def find_l_survivors(n_half=5, m=5, c_target=1.28, solver='MOSEK',
                      cache_path=None, verbose=True):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fp:
            data = json.load(fp)
        if (data['n_half'] == n_half and data['m'] == m
                and data['c_target'] == c_target):
            if verbose:
                print(f"  Loaded {len(data['l_survivors'])} cached L-survivors "
                      f"from {cache_path}")
            return [np.asarray(c, dtype=np.int32) for c in data['l_survivors']]

    d = 2 * n_half
    S_half = 2 * n_half * m
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)

    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    if verbose:
        print(f"  Scanning compositions: n_half={n_half}, m={m}, c={c_target} "
              f"(d={d}, S_half={S_half}) ...")

    t0 = time.time()
    l_survivors = []
    n_proc = n_F = n_Q = n_L = 0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                     batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        n_F += len(f_idx)
        for idx in f_idx:
            c_int = batch[idx]
            if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target, margin=1e-9):
                continue
            n_Q += 1
            pruned, status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                          c_target, solver=solver, order=1,
                                          tol=1e-9, eps_margin=1e-9)
            if not pruned:
                l_survivors.append(c_int.copy())
                n_L += 1
        if verbose:
            print(f"    progress: proc={n_proc:,}  F-surv={n_F}  Q-surv={n_Q}  "
                  f"L-surv={n_L}  ({time.time()-t0:.1f}s)")

    if verbose:
        print(f"  Found {len(l_survivors)} L-survivors in {time.time()-t0:.1f}s")
    if cache_path:
        with open(cache_path, 'w') as fp:
            json.dump({'n_half': n_half, 'm': m, 'c_target': c_target,
                        'l_survivors': [c.tolist() for c in l_survivors]}, fp,
                       indent=2)
        if verbose:
            print(f"  Cached to {cache_path}")
    return l_survivors


# ---------------------------------------------------------------------------
# Worker initializer + function — runs one sub-cell SDP feasibility test.
# We cache windows + A_mats per process via a module-level dict.
# ---------------------------------------------------------------------------
_WORKER_CACHE = {}

def _worker_init():
    """Initializer: import path setup and pre-build windows once per worker."""
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _here)
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger'))
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger', 'cpu'))


def _worker_one_subcell(args):
    """Worker: test SDP feasibility of one sub-cell.

    Args (tuple): (sigma_idx, c_int_list, sigma_list, n_half, m, c_target, solver)
    Returns: (sigma_idx, sub_pruned: bool, status: str, t: float, pre_inf: bool)
    """
    global _WORKER_CACHE
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target, solver = args

    import numpy as _np
    from _Q_bench import _build_windows
    from _L_bench import _build_A_matrices, _shor_feasibility

    c_int = _np.asarray(c_int_list, dtype=_np.int32)
    sigma = _np.asarray(sigma_list, dtype=_np.int8)
    d = len(c_int)
    nm = float(4 * n_half * m)

    # Build sub-cell box
    lo = _np.empty(d, dtype=_np.float64)
    hi = _np.empty(d, dtype=_np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            lo[i] = max(0.0, ci - 1.0)
            hi[i] = ci
        else:
            lo[i] = ci
            hi[i] = ci + 1.0

    # Quick pre-infeasibility check by box sum
    s_lo = float(_np.sum(lo))
    s_hi = float(_np.sum(hi))
    if nm < s_lo - 1e-9 or nm > s_hi + 1e-9:
        return sigma_idx, True, 'box_sum_pre_infeasible', 0.0, True

    # Cached windows + A_mats per worker
    cache_key = ('wA', d)
    if cache_key not in _WORKER_CACHE:
        windows, _ = _build_windows(d)
        A_mats = _build_A_matrices(d, windows)
        _WORKER_CACHE[cache_key] = (windows, A_mats)
    windows, A_mats = _WORKER_CACHE[cache_key]

    t0 = time.time()
    pruned, status = _shor_feasibility(c_int, lo, hi, A_mats, windows,
                                          n_half, m, c_target, solver=solver,
                                          tol=1e-9, eps_margin=1e-9, verbose=False)
    return sigma_idx, bool(pruned), str(status), time.time() - t0, False


def split_prune_one_parallel(c_int, n_half, m, c_target, solver='MOSEK',
                              n_workers=12, early_terminate=True, verbose=True):
    """Test split-pruning of one composition using a worker pool.

    Returns:
        (split_pruned: bool, n_inf: int, n_total: int, t_total: float,
         feasible_sigma: optional list)
    """
    d = len(c_int)
    sigmas = list(product([1, -1], repeat=d))
    n_total = len(sigmas)
    c_int_list = c_int.tolist()

    # Submit all sub-cells; count infeasible.  If any feasible, we can early-stop.
    t0 = time.time()
    n_inf = 0
    n_pre_inf = 0
    feasible_sigma = None
    statuses = {}
    completed = 0

    args_list = [(idx, c_int_list, list(sigma), n_half, m, c_target, solver)
                  for idx, sigma in enumerate(sigmas)]

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_worker_init) as ex:
        futs = {ex.submit(_worker_one_subcell, args): args[0]
                 for args in args_list}
        try:
            for fut in as_completed(futs):
                sigma_idx, sub_pruned, status, t, pre_inf = fut.result()
                completed += 1
                if pre_inf:
                    n_pre_inf += 1
                statuses[status] = statuses.get(status, 0) + 1
                if sub_pruned:
                    n_inf += 1
                else:
                    if feasible_sigma is None:
                        feasible_sigma = list(sigmas[sigma_idx])
                    if early_terminate:
                        # Cancel pending; can't fully cancel running, but stop processing
                        for f in futs:
                            if not f.done():
                                f.cancel()
                        break
                if verbose and completed % 128 == 0:
                    print(f"      ... {completed}/{n_total} sub-cells tested  "
                          f"(inf={n_inf}, pre_inf={n_pre_inf}, "
                          f"{time.time()-t0:.1f}s)")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    t_total = time.time() - t0
    split_pruned = (feasible_sigma is None)
    return split_pruned, n_inf, completed, t_total, feasible_sigma, statuses


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    n_half = 5
    m = 5
    c_target = 1.28
    d = 2 * n_half
    # Default 12 workers (leave a couple of cores for OS/IO).  MOSEK
    # supports unlimited concurrent solves under one license.
    n_workers = int(os.environ.get('N_WORKERS', max(1, min(12, (os.cpu_count() or 4) - 2))))
    solver = _detect_solver('MOSEK')
    print(f"Solver: {solver}, n_workers: {n_workers}")
    if solver != 'MOSEK':
        print("WARNING: non-MOSEK solver detected")

    # Step 1: get L-survivors (cached after first run)
    cache = os.path.join(_HERE, '_smoke_split_cell_l_survivors.json')
    print(f"\n[1] Identifying L-survivors at (n_half={n_half}, m={m}, c={c_target})")
    l_survivors = find_l_survivors(n_half, m, c_target, solver=solver,
                                     cache_path=cache, verbose=True)
    if len(l_survivors) == 0:
        print("No L-survivors; nothing to split-prune.")
        return
    print(f"  -> {len(l_survivors)} L-survivors to test")

    # Step 2: split-prune each (in parallel over sub-cells)
    print(f"\n[2] Splitting cells: 2^{d} = {2**d} sub-cells per survivor "
          f"({n_workers} parallel workers)")
    results = []
    n_split_pruned = 0
    t_global = time.time()

    for i, c_int in enumerate(l_survivors):
        print(f"\n  Survivor {i+1}/{len(l_survivors)}: c={c_int.tolist()}")
        sp_pruned, n_inf, n_done, t_one, feasible_sigma, statuses = (
            split_prune_one_parallel(c_int, n_half, m, c_target, solver=solver,
                                      n_workers=n_workers, early_terminate=True,
                                      verbose=True))
        if sp_pruned:
            n_split_pruned += 1
            print(f"    SPLIT-PRUNED: {n_inf}/{n_done} sub-cells INFEASIBLE  "
                  f"(no feasible sub-cell found)  ({t_one:.1f}s)")
        else:
            print(f"    NOT split-pruned: feasible sub-cell found at "
                  f"sigma={feasible_sigma}  ({n_inf}/{n_done} "
                  f"sub-cells INFEASIBLE before stopping)  ({t_one:.1f}s)")
        print(f"    statuses: {statuses}")
        results.append({
            'c_int': c_int.tolist(),
            'split_pruned': bool(sp_pruned),
            'n_sub_infeasible': int(n_inf),
            'n_sub_tested': int(n_done),
            'time_s': float(t_one),
            'feasible_sigma': feasible_sigma,
            'statuses': statuses,
        })
        elapsed = time.time() - t_global
        print(f"    [running total: {n_split_pruned}/{i+1} split-pruned, "
              f"{elapsed:.1f}s elapsed]")

        if elapsed > 25 * 60:
            print(f"\n  WALL-TIME LIMIT (25 min) -- stopping after survivor {i+1}")
            break

    print(f"\n\n=========================================================")
    print(f"FINAL: split-prunes {n_split_pruned} of {len(results)} L-survivors "
          f"(out of {len(l_survivors)} total)")
    print(f"Total time: {time.time() - t_global:.1f}s")
    print(f"=========================================================\n")

    out_path = os.path.join(_HERE, '_smoke_split_cell_SDP_results.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
            'solver': solver,
            'n_workers': n_workers,
            'n_l_survivors_total': len(l_survivors),
            'n_l_survivors_tested': len(results),
            'n_split_pruned': n_split_pruned,
            'total_time_s': float(time.time() - t_global),
            'results': results,
        }, fp, indent=2)
    print(f"Wrote {out_path}")
    return n_split_pruned, len(results)


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()  # required on Windows
    main()

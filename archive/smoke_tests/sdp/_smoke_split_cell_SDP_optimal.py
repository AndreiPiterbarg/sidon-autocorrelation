"""OPTIMAL split-cell SDP for L-survivors at d=10 (n=5, m=5, c=1.28).

Combines all validated speedups from the search + optional tightness mode.

SPEED OPTIMIZATIONS (all sound):
  (1) Direct MOSEK Task API via `cloninger-steinerberger/cpu/l_direct.py`
      with `lo_override`, `hi_override` — bypasses CVXPY (84% of L SDP wall in
      baseline).  Validated 20.7x speedup at d=6 in `_smoke_mosek_direct.py`.
  (2) Reused `mosek.Env` per worker — created once in worker init, used for
      ALL 1024 sub-cells of a parent.  Saves ~20 ms/SDP license init.
  (3) Smart sigma ordering — process sub-cells whose box CENTER mean-sum is
      closest to 4nm FIRST.  Those are most likely feasible; if any is
      feasible, parent is NOT split-prunable and we stop (early termination).
      For TRULY split-prunable parents (the 1024-infeasible ones) order
      doesn't matter — we have to do all of them anyway.
  (4) Box-sum pre-screen (microseconds): if `Σ lo > 4nm` or `Σ hi < 4nm`,
      sub-cell is infeasible; skip SDP.
  (5) Per-survivor pool of all 1024 sub-cells distributed across workers
      (NOT one survivor per worker).  Each worker burns through sub-cells
      sequentially with its persistent Env.

TIGHTNESS MODE (optional, --depth=2 flag):
  When a sub-cell's SDP returns `optimal` (feasible), recursively SPLIT
  that sub-cell into 2^d sub-sub-cells (each half-width).  If all are
  infeasible, the original sub-cell is too.  At depth=2, total sub-cells
  per parent ≤ 2^d × 2^d = 1M; in practice few sub-cells require recursion.
  Sound: each sub-cell SDP uses MOSEK Farkas certificate.

USAGE:
  python _smoke_split_cell_SDP_optimal.py            # speed mode (depth=1)
  python _smoke_split_cell_SDP_optimal.py --depth=2  # tightness mode
  python _smoke_split_cell_SDP_optimal.py --max-cells=2  # quick test on 2 cells
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


# ---------------------------------------------------------------------------
# Worker globals — initialized once per worker process.
# ---------------------------------------------------------------------------
_W_ENV = None
_W_A_MATS = None
_W_WINDOWS = None
_W_PRUNE_DIRECT = None


def _worker_init(d_child):
    global _W_ENV, _W_A_MATS, _W_WINDOWS, _W_PRUNE_DIRECT
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _here)
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger'))
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger', 'cpu'))
    _mlf = _os.environ.get('MOSEKLM_LICENSE_FILE')
    if not _mlf:
        for _cand in ('/home/ubuntu/mosek/mosek.lic',
                       _os.path.expanduser('~/mosek/mosek.lic'),
                       'C:/mosek/mosek.lic'):
            if _os.path.exists(_cand):
                _os.environ['MOSEKLM_LICENSE_FILE'] = _cand
                break
    _os.environ.setdefault('MSK_IPAR_NUM_THREADS', '1')
    import mosek
    from _Q_bench import _build_windows
    from _L_bench import _build_A_matrices
    from l_direct import prune_L_direct
    _W_ENV = mosek.Env()
    try:
        _W_ENV.checkoutlicense(mosek.feature.pton)
    except Exception:
        pass
    windows, _ = _build_windows(d_child)
    _W_A_MATS = _build_A_matrices(d_child, windows)
    _W_WINDOWS = windows
    _W_PRUNE_DIRECT = prune_L_direct


# ---------------------------------------------------------------------------
# Sub-cell helpers
# ---------------------------------------------------------------------------
def _subcell_box(c_int, sigma):
    """Return (lo, hi) box for sub-cell sigma ∈ {±1}^d.
       sigma_i = +1  =>  x_i ∈ [max(0, c_i-1), c_i]   (δ_i ≥ 0)
       sigma_i = -1  =>  x_i ∈ [c_i, c_i+1]            (δ_i ≤ 0)
    """
    d = len(c_int)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            lo[i] = max(0.0, ci - 1.0); hi[i] = ci
        else:
            lo[i] = ci; hi[i] = ci + 1.0
    return lo, hi


def _box_sum_pre_infeasible(lo, hi, n_half, m, eps=1e-9):
    """If Σ lo > 4nm or Σ hi < 4nm, sub-cell is INFEASIBLE (sound prune)."""
    nm = float(4 * n_half * m)
    return (np.sum(lo) > nm + eps) or (np.sum(hi) < nm - eps)


def _smart_sigma_order(c_int, n_half, m):
    """Return sigmas ordered: most-likely-feasible first.

    A sub-cell is most likely feasible when its box CENTER's mean sum is
    closest to 4nm (the equality plane passes through the center).  Sort
    sigmas by `|mean_sum - 4nm|` ascending.  When the parent is NOT
    split-prunable, this order finds the feasible sub-cell fast and
    enables early termination.  When the parent IS split-prunable, all
    1024 sub-cells must be tested anyway (no penalty).
    """
    d = len(c_int)
    s_target = 4 * n_half * m
    cs = c_int.tolist() if hasattr(c_int, 'tolist') else list(c_int)
    sigmas = list(product([1, -1], repeat=d))

    def slack(sigma):
        # Mean sum: for sigma_i=+1, mean = c_i - 0.5 ; for -1, c_i + 0.5
        ms = 0.0
        for i in range(d):
            ms += (cs[i] - 0.5) if sigma[i] > 0 else (cs[i] + 0.5)
        return abs(ms - s_target)

    sigmas.sort(key=slack)
    return sigmas


# ---------------------------------------------------------------------------
# Per-sub-cell SDP runner (worker-side).
# ---------------------------------------------------------------------------
def _run_subcell(args):
    """Test SDP feasibility of one sub-cell.  All worker globals must be set.

    Args (tuple): (sigma_idx, c_int_list, sigma_list, n_half, m, c_target)
    Returns: (sigma_idx, sub_pruned: bool, status: str, t: float)
    """
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target = args
    c_int = np.asarray(c_int_list, dtype=np.int32)
    sigma = np.asarray(sigma_list, dtype=np.int8)
    lo, hi = _subcell_box(c_int, sigma)

    if _box_sum_pre_infeasible(lo, hi, n_half, m):
        return sigma_idx, True, 'box_sum_pre_infeasible', 0.0

    t0 = time.time()
    try:
        pruned, status = _W_PRUNE_DIRECT(
            c_int, _W_A_MATS, _W_WINDOWS, n_half, m, c_target,
            env=_W_ENV, lo_override=lo, hi_override=hi)
    except Exception as e:
        return sigma_idx, False, f'EXC:{type(e).__name__}:{e}', time.time() - t0
    return sigma_idx, bool(pruned), str(status), time.time() - t0


# ---------------------------------------------------------------------------
# Recursive sub-split for tightness mode (depth >= 2).
# ---------------------------------------------------------------------------
def _recursive_subsplit(c_int, lo, hi, n_half, m, c_target, depth, max_depth):
    """Recursively split [lo, hi] into 2^d half-boxes; return True iff ALL
    sub-sub-cells are SDP-infeasible.  Single-process (called from a worker).

    Sound: each leaf SDP uses Farkas certificate.  At max_depth, if a leaf
    is `optimal`, return False (sub-cell is feasible at this resolution =
    parent NOT recursively-split-prunable at this depth).
    """
    d = len(c_int)
    # Test current box's SDP
    pruned, status = _W_PRUNE_DIRECT(
        c_int, _W_A_MATS, _W_WINDOWS, n_half, m, c_target,
        env=_W_ENV, lo_override=lo, hi_override=hi)
    if pruned:
        return True
    if status != 'optimal' or depth >= max_depth:
        return False

    # Recurse: split each coord at midpoint into 2^d sub-cells
    mid = 0.5 * (lo + hi)
    for sigma in product([0, 1], repeat=d):
        sub_lo = np.empty(d, dtype=np.float64)
        sub_hi = np.empty(d, dtype=np.float64)
        for i in range(d):
            if sigma[i] == 0:
                sub_lo[i] = lo[i]; sub_hi[i] = mid[i]
            else:
                sub_lo[i] = mid[i]; sub_hi[i] = hi[i]
        # Pre-screen
        if _box_sum_pre_infeasible(sub_lo, sub_hi, n_half, m):
            continue
        if not _recursive_subsplit(c_int, sub_lo, sub_hi, n_half, m,
                                     c_target, depth + 1, max_depth):
            return False
    return True


def _run_subcell_recursive(args):
    """Worker: test sub-cell, recursing if it returns `optimal` and depth allows."""
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target, max_depth = args
    c_int = np.asarray(c_int_list, dtype=np.int32)
    sigma = np.asarray(sigma_list, dtype=np.int8)
    lo, hi = _subcell_box(c_int, sigma)

    if _box_sum_pre_infeasible(lo, hi, n_half, m):
        return sigma_idx, True, 'box_sum_pre_infeasible', 0.0

    t0 = time.time()
    try:
        pruned = _recursive_subsplit(c_int, lo, hi, n_half, m, c_target,
                                       depth=1, max_depth=max_depth)
    except Exception as e:
        return sigma_idx, False, f'EXC:{type(e).__name__}:{e}', time.time() - t0
    status = 'infeasible' if pruned else 'optimal_at_depth'
    return sigma_idx, bool(pruned), status, time.time() - t0


# ---------------------------------------------------------------------------
# Per-parent driver.
# ---------------------------------------------------------------------------
def split_prune_one(c_int, n_half, m, c_target, n_workers=12,
                     early_terminate=True, max_depth=1, verbose=False):
    """Split-prune one composition.

    Returns (split_pruned, n_inf, n_done, t_total, feasible_sigma, statuses).
    """
    d = len(c_int)
    sigmas = _smart_sigma_order(c_int, n_half, m)
    n_total = len(sigmas)
    c_int_list = c_int.tolist() if hasattr(c_int, 'tolist') else list(c_int)

    if max_depth > 1:
        args_list = [(idx, c_int_list, list(sigma), n_half, m, c_target, max_depth)
                      for idx, sigma in enumerate(sigmas)]
        worker_fn = _run_subcell_recursive
    else:
        args_list = [(idx, c_int_list, list(sigma), n_half, m, c_target)
                      for idx, sigma in enumerate(sigmas)]
        worker_fn = _run_subcell

    n_inf = 0
    feasible_sigma = None
    statuses = {}
    completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_worker_init,
                              initargs=(d,)) as ex:
        futs = [ex.submit(worker_fn, args) for args in args_list]
        try:
            for fut in futs:
                sigma_idx, sub_pruned, status, t = fut.result()
                completed += 1
                statuses[status] = statuses.get(status, 0) + 1
                if sub_pruned:
                    n_inf += 1
                else:
                    feasible_sigma = list(sigmas[sigma_idx])
                    if early_terminate:
                        for f in futs[completed:]:
                            f.cancel()
                        break
                if verbose and completed % 128 == 0:
                    el = time.time() - t0
                    print(f"      ... {completed}/{n_total}  inf={n_inf}  "
                          f"({el:.1f}s, {1000*el/max(1,completed):.1f} ms/sub-cell)")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    t_total = time.time() - t0
    split_pruned = (feasible_sigma is None)
    return split_pruned, n_inf, completed, t_total, feasible_sigma, statuses


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=5)
    ap.add_argument('--m', type=int, default=5)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--depth', type=int, default=1,
                     help='recursion depth: 1 = standard binary, 2 = recurse on stuck sub-cells')
    ap.add_argument('--max_cells', type=int, default=14,
                     help='max number of L-survivors to test')
    ap.add_argument('--n_workers', type=int,
                     default=int(os.environ.get('N_WORKERS',
                                                  max(1, min(12, (os.cpu_count() or 4) - 2)))))
    ap.add_argument('--out', type=str, default='_smoke_split_cell_SDP_optimal.json')
    args = ap.parse_args()

    cache = os.path.join(_HERE, '_smoke_split_cell_l_survivors.json')
    if not os.path.exists(cache):
        print(f"  ERR: no cached L-survivors at {cache}; run baseline first.")
        return
    with open(cache) as fp:
        data = json.load(fp)
    l_survivors = [np.asarray(c, dtype=np.int32) for c in data['l_survivors']]
    if not l_survivors:
        print("No L-survivors; nothing to do.")
        return
    print(f"Loaded {len(l_survivors)} L-survivors from {cache}")

    n_workers = args.n_workers
    print(f"n_workers={n_workers}, depth={args.depth}, max_cells={args.max_cells}")

    results = []
    n_split = 0
    t_global = time.time()
    for i, c_int in enumerate(l_survivors):
        if i >= args.max_cells:
            break
        print(f"\n[{i+1}/{min(args.max_cells, len(l_survivors))}] "
              f"c={c_int.tolist()}")
        sp, n_inf, n_done, t_one, feasible, statuses = split_prune_one(
            c_int, args.n_half, args.m, args.c_target,
            n_workers=n_workers, early_terminate=True,
            max_depth=args.depth, verbose=False)
        if sp:
            n_split += 1
            print(f"  SPLIT-PRUNED in {t_one:.1f}s ({n_inf}/{n_done} sub-cells inf, "
                  f"avg {1000*t_one/max(1,n_done):.1f} ms/sub-cell)")
        else:
            print(f"  NOT split-pruned (feasible at sigma={feasible}) "
                  f"({n_inf}/{n_done} inf before stop, {t_one:.1f}s)")
        print(f"  statuses: {statuses}")
        results.append({
            'c_int': c_int.tolist(),
            'split_pruned': bool(sp),
            'n_sub_infeasible': int(n_inf),
            'n_sub_tested': int(n_done),
            'time_s': float(t_one),
            'feasible_sigma': feasible,
            'statuses': statuses,
        })

    total = time.time() - t_global
    print(f"\n========================================")
    print(f"FINAL: {n_split}/{len(results)} split-pruned in {total:.1f}s")
    print(f"========================================")

    out_path = os.path.join(_HERE, args.out)
    with open(out_path, 'w') as fp:
        json.dump({
            'n_half': args.n_half, 'm': args.m, 'c_target': args.c_target,
            'd': 2 * args.n_half, 'depth': args.depth,
            'n_workers': n_workers,
            'n_l_survivors_tested': len(results),
            'n_split_pruned': n_split,
            'total_time_s': total,
            'results': results,
        }, fp, indent=2)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()
    main()

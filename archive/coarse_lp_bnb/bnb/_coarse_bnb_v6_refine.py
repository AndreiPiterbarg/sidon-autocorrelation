"""v6 BnB + d-doubling refinement driver.

Runs the full proof workflow for C_{1a} >= c_target:
  Level 0 (d_start): full canonical enumeration + numba F + v6 cascade
                     → produces "open" cells.
  Level k (d = 2^k · d_start): each open cell c at d generates Π(c_i+1) children
                     at 2d (S invariant), canonically deduped across all parents.
                     Numba F-screens the children, residue goes through v6.
                     "Open" of level k feeds level k+1.
  Stop:              all cells certified, OR d reaches max_d, OR frontier=0.

Children construction:
  parent c = (c_0,...,c_{d-1}) with Σ c = S.
  child c' = (a_0, c_0−a_0, a_1, c_1−a_1, ..., a_{d-1}, c_{d-1}−a_{d-1})
        with 0 ≤ a_i ≤ c_i.  Σ c' = Σ c = S.  len(c') = 2d.

Soundness (matches `_refine_1275_to_d16.py`):
  Every measure on the parent d-cell decomposes uniquely into 2d-coordinate
  masses lying in SOME child 2d-cell (modulo measure-zero boundaries).  Hence
  if every child certifies max_W TV_W > c_target, the parent does too.
  v6 cascade is sound; this refinement loop is sound by union-over-children.

Performance:
  - Stage A (numba F-screen) at each level: vectorized over the full frontier,
    typically prunes >99% in seconds.
  - Stage B (v6 cascade) runs only on F-residue, parallel via mp.Pool(N).
  - Canonical reversal dedup cuts the d=2d child set by ~50%.
"""
from __future__ import annotations
import os, sys, time, json, logging, argparse
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import multiprocessing as mp
import numpy as np
from itertools import product

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from compositions import generate_canonical_compositions_batched
from _d16_F_bench import _prune_coarse_count_cell


# ============================================================
# Worker state per (d, S, c_target, max_depth)
# ============================================================
_W = {'d': None, 'S': None, 'c_target': None, 'windows': None,
       'max_depth': 3, 'joint_K': 4, 'joint_fw_iters': 5}


def _worker_init(d, S, c_target, max_depth=3, joint_K=4, joint_fw_iters=5):
    import warnings
    warnings.filterwarnings('ignore')
    import _coarse_bnb_v6 as v6
    _W['d'], _W['S'], _W['c_target'] = d, S, c_target
    _W['max_depth'] = max_depth
    _W['joint_K'] = joint_K
    _W['joint_fw_iters'] = joint_fw_iters
    _W['windows'] = v6.build_all_windows(d)
    # Warm SDP template
    c = np.zeros(d); c[0] = S
    cell = v6.Cell.from_integer_composition(c.astype(np.float64), S)
    try:
        cache = v6.CellCache.build(cell)
        v6.tier_L_single_v6(cache, _W['windows'][len(_W['windows'])//2], c_target)
    except Exception:
        pass


def _worker_cert(c_tuple):
    import _coarse_bnb_v6 as v6
    c = np.asarray(c_tuple, dtype=np.float64)
    try:
        r = v6.certify_composition(c, _W['S'], _W['d'], _W['c_target'],
                                      windows=_W['windows'],
                                      max_depth=_W['max_depth'],
                                      joint_K=_W['joint_K'],
                                      joint_fw_iters=_W['joint_fw_iters'])
        return (c_tuple, r.tier_used, int(r.depth_used), float(r.bound),
                bool(r.certified))
    except Exception as e:
        return (c_tuple, f'ERROR:{type(e).__name__}', -1, 0.0, False)


# ============================================================
# Children generation + canonical dedup
# ============================================================

def gen_children_array(parents: list, d_parent: int) -> np.ndarray:
    """Generate all children (parent c → 2d compositions via per-coord split).

    Returns int32 array (N_children, 2*d_parent).  No dedup yet.
    """
    out_rows = []
    for p in parents:
        opts = [[(a, p[i] - a) for a in range(p[i] + 1)]
                  for i in range(d_parent)]
        for combo in product(*opts):
            row = np.empty(2 * d_parent, dtype=np.int32)
            for i, (a, b) in enumerate(combo):
                row[2 * i] = a
                row[2 * i + 1] = b
            out_rows.append(row)
    if not out_rows:
        return np.zeros((0, 2 * d_parent), dtype=np.int32)
    return np.stack(out_rows, axis=0)


def canonical_dedup_keys(arr: np.ndarray) -> np.ndarray:
    """Canonical reversal dedup.  Returns deduped rows as int32 array.

    Key = lex-min(tuple(c), tuple(c[::-1])).
    """
    if arr.shape[0] == 0:
        return arr
    rev = arr[:, ::-1]
    take_left = np.zeros(arr.shape[0], dtype=bool)
    # row-wise lex compare
    for i in range(arr.shape[1]):
        col_l = arr[:, i]
        col_r = rev[:, i]
        if i == 0:
            decided = col_l != col_r
            take_left[col_l < col_r] = True
            cur_undecided = ~decided
        else:
            new_mask = cur_undecided & (col_l != col_r)
            take_left[new_mask & (col_l < col_r)] = True
            cur_undecided = cur_undecided & (col_l == col_r)
    canonical = np.where(take_left[:, None], arr, rev)
    # Deduplicate via byte view
    void_view = np.ascontiguousarray(canonical).view(
        np.dtype((np.void, canonical.dtype.itemsize * canonical.shape[1])))
    _, unique_idx = np.unique(void_view, return_index=True)
    return canonical[np.sort(unique_idx)]


# ============================================================
# Numba F-screen (single batch)
# ============================================================

def numba_F_pass(cells_arr: np.ndarray, d: int, S: int, c_target: float
                    ) -> tuple:
    """Returns (residue_idx, grid_surv_idx, n_grid_pruned).

    cells_arr: int32 (N, d).
    """
    if cells_arr.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0
    # Warm JIT once per d
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, 1.0)
    survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
        cells_arr, d, S, c_target)
    residue_mask = (~survived) & neg_mask  # cell-uncertain
    grid_surv_idx = np.where(survived)[0]
    residue_idx = np.where(residue_mask)[0]
    n_grid_pruned = int(((~survived) & (~neg_mask)).sum())
    return residue_idx, grid_surv_idx, n_grid_pruned


# ============================================================
# Stage A: initial canonical enumeration + F + v6
# ============================================================

def stage_initial(d, S, c_target, batch_size=400_000, max_depth=3,
                    workers=96, chunksize=4, time_budget=14400.0,
                    log_every_s=5.0):
    log = print
    log(f"\n{'='*72}\nSTAGE 0: initial enum + F-screen + v6 cascade\n"
         f"  d={d}  S={S}  c_target={c_target}  workers={workers}\n{'='*72}", flush=True)

    # JIT warm
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, 1.0)

    residue = []  # cell-uncertain comps (need v6 cascade)
    grid_surv = []  # need d-refinement
    n_total = n_grid_pruned = n_grid_surv = n_cell_uncertain = 0
    min_net_seen = np.inf
    t0 = time.time()
    batch_i = 0
    for batch in generate_canonical_compositions_batched(d, S, batch_size=batch_size):
        batch_i += 1
        batch_i32 = batch.astype(np.int32)
        survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
            batch_i32, d, S, c_target)
        n_total += len(batch)
        n_grid_pruned += int((~survived).sum())
        n_grid_surv += int(survived.sum())
        n_cell_uncertain += int(n_neg)
        if min_net < min_net_seen:
            min_net_seen = float(min_net)
        residue_mask = (~survived) & neg_mask
        if residue_mask.any():
            for idx in np.where(residue_mask)[0]:
                residue.append(tuple(int(x) for x in batch[idx]))
        if survived.any():
            for idx in np.where(survived)[0]:
                grid_surv.append(tuple(int(x) for x in batch[idx]))
        if batch_i <= 3 or batch_i % 5 == 0:
            log(f"  batch {batch_i:>3}: tot={n_total:>9,} grid_pruned={n_grid_pruned:>9,}  "
                 f"uncertain={n_cell_uncertain:>7,} surv={n_grid_surv:>6,} "
                 f"min_net={min_net_seen:.4f} ({time.time()-t0:.1f}s)", flush=True)

    log(f"\n  F-screen done: total={n_total:,} residue={len(residue):,} "
         f"survivors={len(grid_surv):,} ({time.time()-t0:.1f}s)", flush=True)

    # v6 cascade in parallel on residue
    open_cells = []
    counts: dict = {}
    if residue:
        log(f"\n  v6 cascade on {len(residue):,} residue cells (Pool x {workers})",
             flush=True)
        ctx = mp.get_context('spawn')
        t1 = time.time()
        last = t1
        n_done = 0
        with ctx.Pool(processes=workers,
                        initializer=_worker_init,
                        initargs=(d, S, c_target, max_depth, 4, 5)) as pool:
            for result in pool.imap_unordered(_worker_cert, residue,
                                                chunksize=chunksize):
                n_done += 1
                c_tup, tier, depth, bound, certified = result
                counts[tier] = counts.get(tier, 0) + 1
                if not certified:
                    open_cells.append(list(c_tup))
                now = time.time()
                if now - last >= log_every_s:
                    el = now - t1
                    rate = n_done / max(el, 1e-9)
                    eta = (len(residue) - n_done) / max(rate, 1e-9)
                    closed_n = n_done - len(open_cells)
                    log(f"    [{n_done:>6}/{len(residue):,} {100*n_done/len(residue):5.1f}%]  "
                         f"closed={closed_n} open={len(open_cells)}  "
                         f"rate={rate:.0f}/s  el={el:.0f}s eta={eta:.0f}s", flush=True)
                    last = now
        log(f"  v6 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
             f"({time.time()-t1:.1f}s)\n  by tier: {counts}", flush=True)
    return {'d': d, 'S': S, 'c_target': c_target,
            'n_total': n_total, 'grid_pruned': n_grid_pruned,
            'grid_survivors': grid_surv, 'open_cells': open_cells,
            'counts': counts, 'min_net': min_net_seen,
            'elapsed_s': time.time() - t0}


# ============================================================
# Stage B: doubling refinement (level k -> k+1)
# ============================================================

def stage_refine_level(open_parents: list, d_parent: int, S: int,
                          c_target: float, workers=96, max_depth=3,
                          chunksize=4, log_every_s=5.0):
    """One level of refinement: parents at d_parent → children at 2*d_parent."""
    d_child = 2 * d_parent
    log = print
    log(f"\n{'='*72}\nREFINE LEVEL: d={d_parent} -> d={d_child}  "
         f"({len(open_parents)} parents)\n{'='*72}", flush=True)
    if not open_parents:
        return {'d_child': d_child, 'children': 0, 'open_cells': [],
                  'grid_survivors': [], 'counts': {}}
    # Generate + canonical-dedup all children
    t0 = time.time()
    arr_raw = gen_children_array(open_parents, d_parent)
    log(f"  raw children: {arr_raw.shape[0]:,}  ({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    arr_dedup = canonical_dedup_keys(arr_raw)
    log(f"  unique canonical: {arr_dedup.shape[0]:,}  ({time.time()-t1:.1f}s)", flush=True)
    del arr_raw

    # Numba F-screen at d_child
    t2 = time.time()
    residue_idx, surv_idx, n_pruned = numba_F_pass(arr_dedup, d_child, S, c_target)
    log(f"  F-screen at d={d_child}: pruned={n_pruned:,}  "
         f"residue={len(residue_idx):,}  survivors={len(surv_idx):,}  "
         f"({time.time()-t2:.1f}s)", flush=True)
    grid_surv = [tuple(int(x) for x in arr_dedup[i]) for i in surv_idx]
    if len(residue_idx) == 0:
        log(f"  level done: NO residue, no SDP needed.", flush=True)
        return {'d_child': d_child, 'children': int(arr_dedup.shape[0]),
                  'open_cells': [], 'grid_survivors': grid_surv,
                  'counts': {}, 'elapsed_s': time.time() - t0}
    # v6 cascade on residue (parallel)
    residue_list = [tuple(int(x) for x in arr_dedup[i]) for i in residue_idx]
    del arr_dedup
    open_cells = []
    counts: dict = {}
    log(f"\n  v6 cascade on {len(residue_list):,} residue at d={d_child} "
         f"(Pool x {workers})", flush=True)
    ctx = mp.get_context('spawn')
    t3 = time.time()
    last = t3
    n_done = 0
    with ctx.Pool(processes=workers,
                    initializer=_worker_init,
                    initargs=(d_child, S, c_target, max_depth, 4, 5)) as pool:
        for result in pool.imap_unordered(_worker_cert, residue_list,
                                            chunksize=chunksize):
            n_done += 1
            c_tup, tier, depth, bound, certified = result
            counts[tier] = counts.get(tier, 0) + 1
            if not certified:
                open_cells.append(list(c_tup))
            now = time.time()
            if now - last >= log_every_s:
                el = now - t3
                rate = n_done / max(el, 1e-9)
                eta = (len(residue_list) - n_done) / max(rate, 1e-9)
                closed_n = n_done - len(open_cells)
                log(f"    [{n_done:>6}/{len(residue_list):,} {100*n_done/len(residue_list):5.1f}%]  "
                     f"closed={closed_n} open={len(open_cells)}  "
                     f"rate={rate:.0f}/s  el={el:.0f}s eta={eta:.0f}s", flush=True)
                last = now
    log(f"  level v6 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
         f"({time.time()-t3:.1f}s)\n  by tier: {counts}", flush=True)
    return {'d_child': d_child, 'children': int(len(residue_list) + n_pruned + len(surv_idx)),
              'open_cells': open_cells, 'grid_survivors': grid_surv,
              'counts': counts, 'elapsed_s': time.time() - t0}


# ============================================================
# Main: full pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d_start', type=int, default=8)
    ap.add_argument('--S', type=int, default=16)
    ap.add_argument('--c_target', type=float, default=1.25)
    ap.add_argument('--max_d', type=int, default=64)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--chunksize', type=int, default=4)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--out', type=str, default='_prove_125_refine_pod.json')
    args = ap.parse_args()
    print(f"\n{'#'*72}\n# REFINEMENT PROOF C_1a >= {args.c_target}\n"
          f"# d_start={args.d_start}  S={args.S}  max_d={args.max_d}  "
          f"workers={args.workers}\n# Start: {time.strftime('%Y-%m-%d %H:%M:%S')}"
          f"\n{'#'*72}", flush=True)
    t_global = time.time()
    levels = []
    # Level 0: full canonical enumeration
    L0 = stage_initial(args.d_start, args.S, args.c_target,
                          workers=args.workers, chunksize=args.chunksize,
                          max_depth=args.max_depth)
    levels.append({'d': args.d_start, **{k: v for k, v in L0.items()
                                           if k not in ('open_cells', 'grid_survivors')},
                     'n_open': len(L0['open_cells']),
                     'n_grid_survivors': len(L0['grid_survivors'])})
    open_cells = L0['open_cells'] + L0['grid_survivors']
    # Convert to tuples for refinement
    open_parents = [tuple(int(x) for x in c) for c in open_cells]
    d_cur = args.d_start
    while d_cur < args.max_d and open_parents:
        Lk = stage_refine_level(open_parents, d_cur, args.S, args.c_target,
                                  workers=args.workers, max_depth=args.max_depth,
                                  chunksize=args.chunksize)
        levels.append({'d': 2*d_cur, **{k: v for k, v in Lk.items()
                                            if k not in ('open_cells', 'grid_survivors')},
                         'n_open': len(Lk['open_cells']),
                         'n_grid_survivors': len(Lk['grid_survivors'])})
        # Next frontier: open + grid_survivors
        open_parents = [tuple(int(x) for x in c) for c in
                         Lk['open_cells'] + Lk['grid_survivors']]
        d_cur = 2 * d_cur

    total_wall = time.time() - t_global
    if not open_parents:
        verdict = f"PROOF COMPLETE: C_1a >= {args.c_target}"
    else:
        verdict = (f"INCOMPLETE: {len(open_parents):,} cells unclosed at d={d_cur}"
                   f"  (max_d reached)")
    print(f"\n{'#'*72}\n# VERDICT: {verdict}\n# Total wall: {total_wall:.1f}s"
          f"\n{'#'*72}", flush=True)
    out = {'d_start': args.d_start, 'S': args.S, 'c_target': args.c_target,
           'max_d': args.max_d, 'workers': args.workers,
           'verdict': verdict, 'levels': levels,
           'final_open_count': len(open_parents),
           'final_d': d_cur, 'total_wall_s': total_wall,
           'final_open_sample': [list(c) for c in open_parents[:50]]}
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2,
                    default=lambda x: float(x) if isinstance(x, np.floating)
                    else int(x) if isinstance(x, np.integer) else str(x))
    print(f"  saved: {args.out}", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()

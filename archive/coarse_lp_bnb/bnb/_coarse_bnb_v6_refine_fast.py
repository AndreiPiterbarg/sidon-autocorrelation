"""Fast v6 BnB + d-doubling refinement.

Speedups vs `_coarse_bnb_v6_refine.py`:
  (1) MASTER-SIDE VECTORIZED PRE-SCREEN
      Run B1 (lo·lo corner) and B1u (Σμ=1 complement corner) on the ENTIRE
      residue batch via numpy einsum over (N, d, d) × (n_W, d, d) — chunked
      to bound memory.  Closes 30-50% of cells with no Python loop / no Pool
      dispatch / no SDP.  Sound: same bound as v6's tier_B1, tier_B1u.
  (2) LIGHTER WORKER CASCADE
      max_depth=2 (was 4), joint_K=3, joint_fw_iters=2 — most "easy" cells
      close at L_single after 1 SDP; cap deep BnB / FW costs.
  (3) LARGER CHUNKSIZE (16) for Pool — reduces IPC overhead.
  (4) CHUNK RESIDUE INTO MEMORY-BOUNDED SLICES — process 200k cells per
      master pre-screen pass to keep RAM under ~10 GB regardless of total.

All optimizations preserve soundness (same SDP, same bounds).  Master
pre-screens use vectorized identical-math versions of v6's per-cell ops.
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

from compositions import generate_canonical_compositions_batched
from _d16_F_bench import _prune_coarse_count_cell


# ============================================================
# Worker state (lighter config)
# ============================================================
_W = {'d': None, 'S': None, 'c_target': None, 'windows': None,
       'max_depth': 2, 'joint_K': 3, 'joint_fw_iters': 2}


def _worker_init(d, S, c_target, max_depth=2, joint_K=3, joint_fw_iters=2):
    import warnings
    warnings.filterwarnings('ignore')
    import _coarse_bnb_v6 as v6
    _W['d'], _W['S'], _W['c_target'] = d, S, c_target
    _W['max_depth'] = max_depth
    _W['joint_K'] = joint_K
    _W['joint_fw_iters'] = joint_fw_iters
    _W['windows'] = v6.build_all_windows(d)
    # warm SDP template
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
# Vectorized master-side B1 + B1u pre-screen (CHUNKED)
# ============================================================

def master_prescreen_chunk(batch_arr: np.ndarray, S: int,
                              A_stack: np.ndarray, Q_coef_vec: np.ndarray,
                              JmA_stack: np.ndarray, c_target: float,
                              chunk: int = 100_000) -> np.ndarray:
    """Run vectorized B1 + B1u over `batch_arr` (N, d) int32.

    Returns boolean mask of length N: True iff cell is CLOSED by either
    B1 or B1u (i.e. some window gives strictly positive LB).
    Memory bounded to ~chunk · d² · 8 + chunk · n_W · 8 bytes.
    """
    N = batch_arr.shape[0]
    d = batch_arr.shape[1]
    closed_mask = np.zeros(N, dtype=bool)
    h = 1.0 / (2.0 * S)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        c_int = batch_arr[start:stop].astype(np.float64)
        # lo = max(0, c/S - h); hi = c/S + h
        lo = np.maximum(0.0, c_int / S - h)
        hi = c_int / S + h
        # outer products per cell: (chunk, d, d)
        lo_outer = lo[:, :, None] * lo[:, None, :]
        hi_outer = hi[:, :, None] * hi[:, None, :]
        # B1 LB per (cell, W): einsum
        quad_b1 = np.einsum('nij,wij->nw', lo_outer, A_stack)
        b1_NW = Q_coef_vec[None, :] * quad_b1 - c_target  # (chunk, W)
        # B1u LB per (cell, W): 1 - Σ (J−A) hi⊗hi
        quad_b1u = np.einsum('nij,wij->nw', hi_outer, JmA_stack)
        b1u_NW = Q_coef_vec[None, :] * (1.0 - quad_b1u) - c_target
        # Take max over both screens for each cell
        b_max_per_cell = np.maximum(b1_NW.max(axis=1), b1u_NW.max(axis=1))
        closed_mask[start:stop] = b_max_per_cell > 0.0
    return closed_mask


def precompute_window_tensors(d: int):
    """Return (A_stack, Q_coef_vec, JmA_stack) for d."""
    import _coarse_bnb_v6 as v6
    windows = v6.build_all_windows(d)
    A_stack = np.array([W.A for W in windows], dtype=np.float64)
    Q_coef_vec = np.array([W.Q_coef for W in windows], dtype=np.float64)
    JmA_stack = 1.0 - A_stack
    return A_stack, Q_coef_vec, JmA_stack


# ============================================================
# Children generation + canonical dedup (same as slow version)
# ============================================================

def gen_children_array(parents: list, d_parent: int) -> np.ndarray:
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
    if arr.shape[0] == 0:
        return arr
    rev = arr[:, ::-1]
    take_left = np.zeros(arr.shape[0], dtype=bool)
    cur_undecided = None
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
    void_view = np.ascontiguousarray(canonical).view(
        np.dtype((np.void, canonical.dtype.itemsize * canonical.shape[1])))
    _, unique_idx = np.unique(void_view, return_index=True)
    return canonical[np.sort(unique_idx)]


def numba_F_pass(cells_arr: np.ndarray, d: int, S: int, c_target: float):
    if cells_arr.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, 1.0)
    survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
        cells_arr, d, S, c_target)
    residue_mask = (~survived) & neg_mask
    grid_surv_idx = np.where(survived)[0]
    residue_idx = np.where(residue_mask)[0]
    n_grid_pruned = int(((~survived) & (~neg_mask)).sum())
    return residue_idx, grid_surv_idx, n_grid_pruned


# ============================================================
# Stage 0: initial canonical enumeration + F + pre-screen + v6
# ============================================================

def stage0_initial(d, S, c_target, batch_size=400_000, workers=96,
                    chunksize=16, log_every_s=5.0,
                    max_depth=6, joint_K=4, joint_fw_iters=5):
    log = print
    log(f"\n{'='*72}\nSTAGE 0: enum + F + master-prescreen + v6\n"
         f"  d={d} S={S} c_target={c_target} workers={workers}\n{'='*72}", flush=True)
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, 1.0)

    # Precompute window tensors for master pre-screen
    A_stack, Q_coef_vec, JmA_stack = precompute_window_tensors(d)
    log(f"  windows: {A_stack.shape[0]} at d={d}", flush=True)

    residue_arrays = []  # list of int32 (b, d)
    grid_surv = []
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
            residue_arrays.append(batch_i32[residue_mask].copy())
        if survived.any():
            for idx in np.where(survived)[0]:
                grid_surv.append(tuple(int(x) for x in batch[idx]))
        if batch_i <= 3 or batch_i % 5 == 0:
            log(f"  batch {batch_i:>3}: tot={n_total:>9,} pruned={n_grid_pruned:>9,} "
                 f"residue={n_cell_uncertain:>7,} surv={n_grid_surv:>5,} "
                 f"min_net={min_net_seen:.4f} ({time.time()-t0:.1f}s)", flush=True)
    if not residue_arrays:
        return {'d': d, 'S': S, 'c_target': c_target, 'open_cells': [],
                  'grid_survivors': grid_surv, 'n_total': n_total,
                  'elapsed_s': time.time() - t0}
    residue_arr = np.concatenate(residue_arrays, axis=0)
    log(f"\n  Stage-0 F done: residue={residue_arr.shape[0]:,} "
         f"(t={time.time()-t0:.1f}s)", flush=True)

    # Master pre-screen
    t_ps = time.time()
    closed_by_ps = master_prescreen_chunk(residue_arr, S, A_stack, Q_coef_vec,
                                              JmA_stack, c_target,
                                              chunk=100_000)
    n_closed_ps = int(closed_by_ps.sum())
    log(f"  master pre-screen (B1+B1u): closed={n_closed_ps:,} / "
         f"{residue_arr.shape[0]:,}  ({time.time()-t_ps:.1f}s)", flush=True)
    survivors = residue_arr[~closed_by_ps]
    log(f"  → Pool residue: {survivors.shape[0]:,}", flush=True)

    # v6 Pool
    open_cells = []
    counts: dict = {'B1+B1u (pre-screen)': n_closed_ps}
    if survivors.shape[0] > 0:
        residue_list = [tuple(int(x) for x in row) for row in survivors]
        del survivors
        log(f"\n  v6 cascade on Pool x {workers} (max_depth={max_depth}, "
             f"joint_K={joint_K}, FW={joint_fw_iters})",
             flush=True)
        ctx = mp.get_context('spawn')
        t3 = time.time(); last = t3; n_done = 0
        with ctx.Pool(processes=workers,
                        initializer=_worker_init,
                        initargs=(d, S, c_target, max_depth, joint_K,
                                    joint_fw_iters)) as pool:
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
                    log(f"    [{n_done:>7}/{len(residue_list):,} "
                         f"{100*n_done/len(residue_list):5.1f}%]  "
                         f"closed={closed_n} open={len(open_cells)}  "
                         f"rate={rate:.0f}/s el={el:.0f}s eta={eta:.0f}s",
                         flush=True)
                    last = now
        log(f"  v6 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
             f"({time.time()-t3:.1f}s)\n  tiers: {counts}", flush=True)
    return {'d': d, 'S': S, 'c_target': c_target, 'n_total': n_total,
              'grid_pruned': n_grid_pruned, 'grid_survivors': grid_surv,
              'open_cells': open_cells, 'counts': counts,
              'min_net': min_net_seen, 'elapsed_s': time.time() - t0}


# ============================================================
# Level k -> k+1 doubling
# ============================================================

def stage_refine_level(open_parents: list, d_parent: int, S: int,
                          c_target: float, workers=96, chunksize=16,
                          max_depth=2, joint_K=3, joint_fw_iters=2,
                          log_every_s=5.0):
    d_child = 2 * d_parent
    log = print
    log(f"\n{'='*72}\nREFINE: d={d_parent} -> d={d_child}  "
         f"({len(open_parents)} parents)\n{'='*72}", flush=True)
    if not open_parents:
        return {'d_child': d_child, 'children': 0, 'open_cells': [],
                  'grid_survivors': [], 'counts': {}}
    t0 = time.time()
    # Generate raw children + canonical dedup
    arr_raw = gen_children_array(open_parents, d_parent)
    log(f"  raw children: {arr_raw.shape[0]:,}  ({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    arr_dedup = canonical_dedup_keys(arr_raw)
    log(f"  unique canonical: {arr_dedup.shape[0]:,}  ({time.time()-t1:.1f}s)", flush=True)
    del arr_raw

    # Numba F-screen at d_child
    t2 = time.time()
    residue_idx, surv_idx, n_pruned = numba_F_pass(arr_dedup, d_child, S, c_target)
    log(f"  F at d={d_child}: pruned={n_pruned:,} residue={len(residue_idx):,} "
         f"survivors={len(surv_idx):,} ({time.time()-t2:.1f}s)", flush=True)
    grid_surv = [tuple(int(x) for x in arr_dedup[i]) for i in surv_idx]

    if len(residue_idx) == 0:
        return {'d_child': d_child, 'children': int(arr_dedup.shape[0]),
                  'open_cells': [], 'grid_survivors': grid_surv,
                  'counts': {}, 'elapsed_s': time.time() - t0}

    # Master vectorized pre-screen at d_child
    A_stack, Q_coef_vec, JmA_stack = precompute_window_tensors(d_child)
    residue_arr = arr_dedup[residue_idx]
    del arr_dedup
    t_ps = time.time()
    closed_by_ps = master_prescreen_chunk(residue_arr, S, A_stack, Q_coef_vec,
                                              JmA_stack, c_target,
                                              chunk=50_000)
    n_closed_ps = int(closed_by_ps.sum())
    log(f"  master pre-screen: closed={n_closed_ps:,} / {residue_arr.shape[0]:,} "
         f"({time.time()-t_ps:.1f}s)", flush=True)
    survivors = residue_arr[~closed_by_ps]
    log(f"  → Pool residue: {survivors.shape[0]:,}", flush=True)

    counts: dict = {'B1+B1u (pre-screen)': n_closed_ps}
    open_cells = []
    if survivors.shape[0] > 0:
        residue_list = [tuple(int(x) for x in row) for row in survivors]
        del survivors
        log(f"\n  v6 cascade on Pool x {workers}", flush=True)
        ctx = mp.get_context('spawn')
        t3 = time.time(); last = t3; n_done = 0
        with ctx.Pool(processes=workers,
                        initializer=_worker_init,
                        initargs=(d_child, S, c_target, max_depth, joint_K,
                                    joint_fw_iters)) as pool:
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
                    log(f"    [{n_done:>7}/{len(residue_list):,} "
                         f"{100*n_done/len(residue_list):5.1f}%]  "
                         f"closed={closed_n} open={len(open_cells)} "
                         f"rate={rate:.0f}/s el={el:.0f}s eta={eta:.0f}s",
                         flush=True)
                    last = now
        log(f"  v6 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
             f"({time.time()-t3:.1f}s)\n  tiers: {counts}", flush=True)
    return {'d_child': d_child, 'children': int(residue_arr.shape[0] + n_pruned + len(surv_idx)),
              'open_cells': open_cells, 'grid_survivors': grid_surv,
              'counts': counts, 'elapsed_s': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d_start', type=int, default=10)
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.25)
    ap.add_argument('--max_d', type=int, default=80)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--chunksize', type=int, default=16)
    # L0 (stage 0): high-closure config; L1+ (refinement levels): fast config
    ap.add_argument('--l0_max_depth', type=int, default=6)
    ap.add_argument('--l0_joint_K', type=int, default=4)
    ap.add_argument('--l0_joint_fw_iters', type=int, default=5)
    ap.add_argument('--lk_max_depth', type=int, default=3)
    ap.add_argument('--lk_joint_K', type=int, default=3)
    ap.add_argument('--lk_joint_fw_iters', type=int, default=3)
    ap.add_argument('--out', type=str, default='_prove_125_refine_fast_pod.json')
    args = ap.parse_args()
    print(f"\n{'#'*72}\n# FAST REFINEMENT  c={args.c_target}\n"
          f"# d_start={args.d_start} S={args.S} max_d={args.max_d} "
          f"workers={args.workers}\n# {time.strftime('%Y-%m-%d %H:%M:%S')}"
          f"\n{'#'*72}", flush=True)
    t_global = time.time()
    levels = []
    L0 = stage0_initial(args.d_start, args.S, args.c_target,
                          workers=args.workers, chunksize=args.chunksize,
                          max_depth=args.l0_max_depth,
                          joint_K=args.l0_joint_K,
                          joint_fw_iters=args.l0_joint_fw_iters)
    levels.append({'d': args.d_start, **{k: v for k, v in L0.items()
                                            if k not in ('open_cells', 'grid_survivors')},
                     'n_open': len(L0['open_cells']),
                     'n_grid_survivors': len(L0['grid_survivors'])})
    open_parents = [tuple(int(x) for x in c) for c in
                     L0['open_cells'] + L0['grid_survivors']]
    d_cur = args.d_start
    while d_cur < args.max_d and open_parents:
        Lk = stage_refine_level(open_parents, d_cur, args.S, args.c_target,
                                  workers=args.workers, chunksize=args.chunksize,
                                  max_depth=args.lk_max_depth,
                                  joint_K=args.lk_joint_K,
                                  joint_fw_iters=args.lk_joint_fw_iters)
        levels.append({'d': 2*d_cur, **{k: v for k, v in Lk.items()
                                            if k not in ('open_cells', 'grid_survivors')},
                         'n_open': len(Lk['open_cells']),
                         'n_grid_survivors': len(Lk['grid_survivors'])})
        open_parents = [tuple(int(x) for x in c) for c in
                         Lk['open_cells'] + Lk['grid_survivors']]
        d_cur = 2 * d_cur
    total_wall = time.time() - t_global
    if not open_parents:
        verdict = f"PROOF COMPLETE: C_1a >= {args.c_target}"
    else:
        verdict = f"INCOMPLETE: {len(open_parents):,} cells unclosed at d={d_cur}"
    print(f"\n{'#'*72}\n# VERDICT: {verdict}\n# Total: {total_wall:.1f}s"
          f"\n{'#'*72}", flush=True)
    out = {'d_start': args.d_start, 'S': args.S, 'c_target': args.c_target,
           'max_d': args.max_d, 'workers': args.workers,
           'verdict': verdict, 'levels': levels,
           'final_open_count': len(open_parents), 'final_d': d_cur,
           'total_wall_s': total_wall,
           'final_open_sample': [list(c) for c in open_parents[:50]]}
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2,
                    default=lambda x: float(x) if isinstance(x, np.floating)
                    else int(x) if isinstance(x, np.integer) else str(x))
    print(f"  saved: {args.out}", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()

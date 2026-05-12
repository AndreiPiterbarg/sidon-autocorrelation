"""v7 refinement driver: v6_refine_fast architecture + v7 worker (BD, skip-gate, signing cuts).
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
# Import master pre-screen and helpers from v6 refine fast
from _coarse_bnb_v6_refine_fast import (
    master_prescreen_chunk, precompute_window_tensors,
    gen_children_array, canonical_dedup_keys, numba_F_pass,
)


_W = {'d': None, 'S': None, 'c_target': None, 'windows': None,
       'max_depth': 6, 'joint_K': 4, 'joint_fw_iters': 5}


def _worker_init(d, S, c_target, max_depth=6, joint_K=4, joint_fw_iters=5):
    import warnings
    warnings.filterwarnings('ignore')
    import _coarse_bnb_v7 as v7
    _W['d'], _W['S'], _W['c_target'] = d, S, c_target
    _W['max_depth'] = max_depth
    _W['joint_K'] = joint_K
    _W['joint_fw_iters'] = joint_fw_iters
    _W['windows'] = v7.build_all_windows(d)
    c = np.zeros(d); c[0] = S
    cell = v7.Cell.from_integer_composition(c.astype(np.float64), S)
    try:
        cache = v7.CellCache.build(cell)
        v7.tier_L_single_v7(cache, _W['windows'][len(_W['windows'])//2], c_target)
    except Exception:
        pass


def _worker_cert(c_tuple):
    import _coarse_bnb_v7 as v7
    c = np.asarray(c_tuple, dtype=np.float64)
    try:
        r = v7.certify_composition(c, _W['S'], _W['d'], _W['c_target'],
                                      windows=_W['windows'],
                                      max_depth=_W['max_depth'],
                                      joint_K=_W['joint_K'],
                                      joint_fw_iters=_W['joint_fw_iters'])
        return (c_tuple, r.tier_used, int(r.depth_used), float(r.bound),
                bool(r.certified))
    except Exception as e:
        return (c_tuple, f'ERROR:{type(e).__name__}', -1, 0.0, False)


def stage0_initial(d, S, c_target, batch_size=400_000, workers=96,
                    chunksize=16, log_every_s=5.0,
                    max_depth=6, joint_K=4, joint_fw_iters=5):
    log = print
    log(f"\n{'='*72}\nSTAGE 0 v7: enum + F + master-prescreen + v7 cascade\n"
         f"  d={d} S={S} c_target={c_target} workers={workers}\n{'='*72}", flush=True)
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, 1.0)
    A_stack, Q_coef_vec, JmA_stack = precompute_window_tensors(d)
    log(f"  windows: {A_stack.shape[0]} at d={d}", flush=True)
    residue_arrays = []
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
    log(f"\n  F done: residue={residue_arr.shape[0]:,} ({time.time()-t0:.1f}s)",
         flush=True)
    t_ps = time.time()
    closed_by_ps = master_prescreen_chunk(residue_arr, S, A_stack, Q_coef_vec,
                                              JmA_stack, c_target,
                                              chunk=100_000)
    n_closed_ps = int(closed_by_ps.sum())
    log(f"  master pre-screen: closed={n_closed_ps:,} ({time.time()-t_ps:.1f}s)",
         flush=True)
    survivors = residue_arr[~closed_by_ps]
    log(f"  → Pool residue: {survivors.shape[0]:,}", flush=True)
    open_cells = []
    counts: dict = {'B1+B1u (pre-screen)': n_closed_ps}
    if survivors.shape[0] > 0:
        residue_list = [tuple(int(x) for x in row) for row in survivors]
        del survivors
        log(f"\n  v7 cascade Pool x {workers} "
             f"(max_depth={max_depth}, joint_K={joint_K}, FW={joint_fw_iters})",
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
        log(f"  v7 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
             f"({time.time()-t3:.1f}s)\n  tiers: {counts}", flush=True)
    return {'d': d, 'S': S, 'c_target': c_target, 'n_total': n_total,
              'grid_pruned': n_grid_pruned, 'grid_survivors': grid_surv,
              'open_cells': open_cells, 'counts': counts,
              'min_net': min_net_seen, 'elapsed_s': time.time() - t0}


def stage_refine_level(open_parents, d_parent, S, c_target,
                          workers=96, chunksize=16,
                          max_depth=3, joint_K=3, joint_fw_iters=3,
                          log_every_s=5.0):
    d_child = 2 * d_parent
    log = print
    log(f"\n{'='*72}\nREFINE v7: d={d_parent} -> d={d_child}  "
         f"({len(open_parents)} parents)\n{'='*72}", flush=True)
    if not open_parents:
        return {'d_child': d_child, 'children': 0, 'open_cells': [],
                  'grid_survivors': [], 'counts': {}}
    t0 = time.time()
    arr_raw = gen_children_array(open_parents, d_parent)
    log(f"  raw children: {arr_raw.shape[0]:,}  ({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    arr_dedup = canonical_dedup_keys(arr_raw)
    log(f"  unique canonical: {arr_dedup.shape[0]:,}  ({time.time()-t1:.1f}s)", flush=True)
    del arr_raw
    t2 = time.time()
    residue_idx, surv_idx, n_pruned = numba_F_pass(arr_dedup, d_child, S, c_target)
    log(f"  F at d={d_child}: pruned={n_pruned:,} residue={len(residue_idx):,} "
         f"surv={len(surv_idx):,} ({time.time()-t2:.1f}s)", flush=True)
    grid_surv = [tuple(int(x) for x in arr_dedup[i]) for i in surv_idx]
    if len(residue_idx) == 0:
        return {'d_child': d_child, 'children': int(arr_dedup.shape[0]),
                  'open_cells': [], 'grid_survivors': grid_surv,
                  'counts': {}, 'elapsed_s': time.time() - t0}
    A_stack, Q_coef_vec, JmA_stack = precompute_window_tensors(d_child)
    residue_arr = arr_dedup[residue_idx]
    del arr_dedup
    t_ps = time.time()
    closed_by_ps = master_prescreen_chunk(residue_arr, S, A_stack, Q_coef_vec,
                                              JmA_stack, c_target,
                                              chunk=50_000)
    n_closed_ps = int(closed_by_ps.sum())
    log(f"  master pre-screen: closed={n_closed_ps:,} ({time.time()-t_ps:.1f}s)",
         flush=True)
    survivors = residue_arr[~closed_by_ps]
    log(f"  → Pool residue: {survivors.shape[0]:,}", flush=True)
    counts: dict = {'B1+B1u (pre-screen)': n_closed_ps}
    open_cells = []
    if survivors.shape[0] > 0:
        residue_list = [tuple(int(x) for x in row) for row in survivors]
        del survivors
        log(f"\n  v7 cascade Pool x {workers}", flush=True)
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
        log(f"  v7 done: closed={n_done-len(open_cells)} open={len(open_cells)} "
             f"({time.time()-t3:.1f}s)\n  tiers: {counts}", flush=True)
    return {'d_child': d_child,
              'children': int(residue_arr.shape[0] + n_pruned + len(surv_idx)),
              'open_cells': open_cells, 'grid_survivors': grid_surv,
              'counts': counts, 'elapsed_s': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d_start', type=int, default=12)
    ap.add_argument('--S', type=int, default=12)
    ap.add_argument('--c_target', type=float, default=1.25)
    ap.add_argument('--max_d', type=int, default=96)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--chunksize', type=int, default=16)
    ap.add_argument('--l0_max_depth', type=int, default=6)
    ap.add_argument('--l0_joint_K', type=int, default=4)
    ap.add_argument('--l0_joint_fw_iters', type=int, default=5)
    ap.add_argument('--lk_max_depth', type=int, default=3)
    ap.add_argument('--lk_joint_K', type=int, default=3)
    ap.add_argument('--lk_joint_fw_iters', type=int, default=3)
    ap.add_argument('--out', type=str, default='_prove_125_v7_pod.json')
    args = ap.parse_args()
    print(f"\n{'#'*72}\n# v7 REFINEMENT  c={args.c_target}\n"
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
    verdict = (f"PROOF COMPLETE: C_1a >= {args.c_target}" if not open_parents
               else f"INCOMPLETE: {len(open_parents):,} unclosed at d={d_cur}")
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

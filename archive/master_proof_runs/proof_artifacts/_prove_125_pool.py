"""Parallel pod prove of C_{1a} >= c_target using v6 (sound) cascade.

Maximizes pod usage:
  Stage 1 : numba F-screen with all available threads.
  Stage 2 : multiprocessing.Pool(N_WORKERS); each worker imports v6, builds
            its SDP template on first call (per-d, DPP-clean, cached), then
            certifies stage-1 residue cells streamed via imap_unordered.

Soundness:
  Uses _coarse_bnb_v6 only (sound L_joint, Lagrangian + Frank–Wolfe).
  v4's unsound max-min epigraph is NOT in this pipeline.

Usage:
  python3 _prove_125_pool.py --d 8 --S 16 --c_target 1.25 --workers 96 \
                              --out _prove_125_pod.json
"""
from __future__ import annotations
import os, sys, time, json, logging, argparse
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import multiprocessing as mp
import numpy as np

# Make sure local dir is on path
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

# Limit child thread count (MOSEK pinned in v6 template; numpy/openblas may grab cores)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from compositions import generate_canonical_compositions_batched
from _d16_F_bench import _prune_coarse_count_cell


# ============================================================
# Worker globals (initialized once per worker process)
# ============================================================

_W_windows = None
_W_d = None
_W_S = None
_W_c_target = None
_W_bundle = None
_W_max_depth = 3
_W_joint_K = 4
_W_joint_fw_iters = 5


def _worker_init(d, S, c_target, max_depth=3, joint_K=4, joint_fw_iters=5):
    """Initializer: import v6, build windows + SDP template once per worker."""
    global _W_windows, _W_d, _W_S, _W_c_target, _W_bundle
    global _W_max_depth, _W_joint_K, _W_joint_fw_iters
    # Suppress CVXPY warnings inside worker
    import warnings
    warnings.filterwarnings('ignore')
    import _coarse_bnb_v6 as v6
    _W_windows = v6.build_all_windows(d)
    _W_d, _W_S, _W_c_target = d, S, c_target
    _W_max_depth = max_depth
    _W_joint_K = joint_K
    _W_joint_fw_iters = joint_fw_iters
    _W_bundle = v6.get_bundle(_W_windows)
    # Warm SDP template (single dummy solve to compile the parametric SDP)
    _W = _W_windows[len(_W_windows) // 2]
    c = np.zeros(d); c[0] = S
    cell = v6.Cell.from_integer_composition(c.astype(np.float64), S)
    cache = v6.CellCache.build(cell)
    try:
        v6.tier_L_single_v6(cache, _W, c_target)
    except Exception:
        pass


def _worker_cert(c_int_tuple):
    """Certify one composition.  Returns (c_tuple, tier_used, depth, bound).

    The caller side feeds tuples of int (picklable) — we reconstruct float[d].
    """
    import _coarse_bnb_v6 as v6
    c = np.asarray(c_int_tuple, dtype=np.float64)
    try:
        r = v6.certify_composition(c, _W_S, _W_d, _W_c_target,
                                      windows=_W_windows,
                                      max_depth=_W_max_depth,
                                      joint_K=_W_joint_K,
                                      joint_fw_iters=_W_joint_fw_iters)
        return (tuple(int(x) for x in c_int_tuple),
                  r.tier_used, int(r.depth_used), float(r.bound),
                  bool(r.certified))
    except Exception as e:
        return (tuple(int(x) for x in c_int_tuple),
                  f'ERROR:{type(e).__name__}', -1, 0.0, False)


# ============================================================
# Stage 1: numba F-screen on full canonical enumeration
# ============================================================

def stage1(d, S, c_target, batch_size=400_000, time_budget=1800.0,
              log_every=10):
    """Returns dict with residue (cell-uncertain) compositions + grid_survivors."""
    log = print
    log(f"\n{'='*72}")
    log(f"STAGE 1: numba F-screen on canonical enumeration")
    log(f"  d={d}  S={S}  c_target={c_target}  batch_size={batch_size}")
    log(f"{'='*72}")
    # JIT warm
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, c_target)

    residue = []
    grid_survivors = []
    n_total = n_grid_pruned = n_grid_surv = n_cell_uncertain = 0
    min_net_seen = np.inf
    t0 = time.time()
    batch_i = 0
    for batch in generate_canonical_compositions_batched(d, S,
                                                            batch_size=batch_size):
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
                grid_survivors.append(tuple(int(x) for x in batch[idx]))
        if batch_i <= 3 or batch_i % log_every == 0:
            el = time.time() - t0
            rate = n_total / max(el, 1e-9)
            log(f"  batch {batch_i:>3}: total={n_total:>10,}  "
                 f"grid_pruned={n_grid_pruned:>10,}  uncertain={n_cell_uncertain:>8,}  "
                 f"surv={n_grid_surv:>6,}  rate={rate/1e3:.1f}K/s  el={el:.1f}s  "
                 f"min_net={min_net_seen:.6f}", flush=True)
        if time.time() - t0 > time_budget:
            log("  STAGE 1 BUDGET HIT"); break
    el = time.time() - t0
    log(f"\n  STAGE 1 SUMMARY: total={n_total:,}  grid_pruned={n_grid_pruned:,}  "
         f"uncertain={n_cell_uncertain:,}  surv={n_grid_surv:,}  "
         f"min_net={min_net_seen:.6f}  ({el:.1f}s)")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': n_total, 'n_grid_pruned': n_grid_pruned,
        'n_grid_surv': n_grid_surv, 'n_cell_uncertain': n_cell_uncertain,
        'min_net_seen': min_net_seen, 'elapsed_s': el,
        'residue': residue, 'grid_survivors': grid_survivors,
    }


# ============================================================
# Stage 2: mp.Pool over residue
# ============================================================

def stage2(residue, d, S, c_target, workers=96, chunksize=8,
              time_budget=10800.0, log_every_s=5.0,
              max_depth=3, joint_K=4, joint_fw_iters=5):
    log = print
    log(f"\n{'='*72}")
    log(f"STAGE 2: mp.Pool({workers}) on {len(residue):,} residue cells (v6)")
    log(f"  chunksize={chunksize}  time_budget={time_budget}s")
    log(f"{'='*72}")
    if not residue:
        return {'closed_counts': {}, 'open_count': 0, 'open_samples': [],
                'elapsed_s': 0.0}
    ctx = mp.get_context('spawn')  # safer than fork with cvxpy/mosek state
    t0 = time.time()
    counts: dict = {}
    open_cells: list = []
    n_done = 0
    last_log = t0
    with ctx.Pool(processes=workers,
                  initializer=_worker_init,
                  initargs=(d, S, c_target, max_depth, joint_K, joint_fw_iters)) as pool:
        log(f"  pool ready, dispatching residue ...", flush=True)
        for result in pool.imap_unordered(_worker_cert, residue,
                                            chunksize=chunksize):
            n_done += 1
            c_tup, tier, depth, bound, certified = result
            counts[tier] = counts.get(tier, 0) + 1
            if not certified:
                open_cells.append({'c': list(c_tup), 'tier': tier,
                                     'depth': depth, 'bound': bound})
            now = time.time()
            if now - last_log >= log_every_s:
                elapsed = now - t0
                rate = n_done / max(elapsed, 1e-9)
                remaining = (len(residue) - n_done) / max(rate, 1e-9)
                # Compact summary: total certified + per-tier counts + open + ETA
                closed_n = n_done - len(open_cells)
                tier_brief = ' '.join(f'{k}={v}' for k, v in
                                       sorted(counts.items(), key=lambda x: -x[1]))
                # Read load average for pod utilization
                try:
                    with open('/proc/loadavg') as fp:
                        loadavg = fp.read().split()[0]
                except Exception:
                    loadavg = '?'
                log(f"  [{n_done:>6}/{len(residue):,} {100*n_done/len(residue):5.1f}%]  "
                     f"closed={closed_n} open={len(open_cells)}  rate={rate:.1f}/s  "
                     f"el={elapsed:.0f}s eta={remaining:.0f}s  load1={loadavg}",
                     flush=True)
                log(f"      tiers: {tier_brief}", flush=True)
                last_log = now
            if now - t0 > time_budget:
                log("  STAGE 2 BUDGET HIT", flush=True); break
    el = time.time() - t0
    log(f"\n  STAGE 2 SUMMARY:")
    log(f"    closed counts:  {counts}")
    log(f"    open:           {len(open_cells)}")
    log(f"    elapsed:        {el:.1f}s")
    return {'closed_counts': counts, 'open_count': len(open_cells),
            'open_samples': open_cells[:50], 'elapsed_s': el,
            'n_processed': n_done}


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=8)
    ap.add_argument('--S', type=int, default=16)
    ap.add_argument('--c_target', type=float, default=1.25)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--chunksize', type=int, default=8)
    ap.add_argument('--time_stage1', type=float, default=1800.0)
    ap.add_argument('--time_stage2', type=float, default=10800.0)
    ap.add_argument('--max_depth', type=int, default=3)
    ap.add_argument('--joint_K', type=int, default=4)
    ap.add_argument('--joint_fw_iters', type=int, default=5)
    ap.add_argument('--out', type=str, default='_prove_125_pod.json')
    args = ap.parse_args()
    print(f"\n{'#'*72}\n# PROVE C_1a >= {args.c_target}  "
          f"(d={args.d} S={args.S} workers={args.workers})\n# Pod: $(nproc)  "
          f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*72}", flush=True)
    t_global = time.time()
    s1 = stage1(args.d, args.S, args.c_target,
                  time_budget=args.time_stage1)
    s2 = stage2(s1['residue'], args.d, args.S, args.c_target,
                  workers=args.workers, chunksize=args.chunksize,
                  time_budget=args.time_stage2,
                  max_depth=args.max_depth, joint_K=args.joint_K,
                  joint_fw_iters=args.joint_fw_iters)
    total = time.time() - t_global
    print(f"\n{'#'*72}\n# VERDICT @ d={args.d} S={args.S} c={args.c_target}\n{'#'*72}")
    print(f"  Stage 1: processed={s1['n_total']:,}  "
          f"grid_survivors={s1['n_grid_surv']:,}  "
          f"residue={len(s1['residue']):,}")
    print(f"  Stage 2: closed={s2.get('n_processed',0) - s2['open_count']}  "
          f"open={s2['open_count']}")
    cells_unclosed = s1['n_grid_surv'] + s2['open_count']
    if cells_unclosed == 0:
        print(f"\n  *** PROOF COMPLETE: C_1a >= {args.c_target} at d={args.d}, S={args.S} ***")
    else:
        print(f"\n  Proof INCOMPLETE: {cells_unclosed:,} cells unclosed at this level.")
    print(f"  Total wall: {total:.1f}s")
    out = {
        'd': args.d, 'S': args.S, 'c_target': args.c_target,
        'workers': args.workers,
        'stage1': {k: v for k, v in s1.items() if k not in ('residue', 'grid_survivors')},
        'stage1_residue_count': len(s1['residue']),
        'stage1_survivor_count': len(s1['grid_survivors']),
        'stage2': s2,
        'total_wall_s': total,
    }
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2,
                    default=lambda x: float(x) if isinstance(x, np.floating)
                    else int(x) if isinstance(x, np.integer) else str(x))
    print(f"  saved: {args.out}")


if __name__ == '__main__':
    mp.freeze_support()
    main()

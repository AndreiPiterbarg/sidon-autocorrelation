"""Attempt to prove C_{1a} >= 1.25 via the hybrid coarse cascade.

OPTIMAL CONFIG (laptop, ~1 hour budget):

  d=12, S=12   c_target=1.25
  d=16, S=16   c_target=1.25    (fallback if d=12 has too much residue)

Pipeline:
  Stage 1: Numba _prune_coarse (F-style) on FULL canonical enumeration.
           Reports (a) grid-pruned mask, (b) per-cell cell-net.
           Cells with cell-net > 0 are CERTIFIED here.

  Stage 2: For residue (grid-pruned but cell-net ≤ 0), apply v4 cell-cert
           (B1 μ-space corner → tier F → tier L_single → tier L_joint → split).

  Stage 3: For grid-survivors (cell-net irrelevant), refine to L1 (d → 2d)
           and recurse. Or report.

NEVER PALINDROMIC. Canonical enumeration only.
Heavy logging at every stage.
"""
from __future__ import annotations
import os, sys, time, logging, json
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_canonical_compositions_batched
import _coarse_bnb_v4 as v4
from _d16_F_bench import _prune_coarse_count_cell  # numba grid+cell-net


def _log(msg=""):
    print(msg, flush=True)


def stage1_numba(d, S, c_target, batch_size=200_000, time_budget=900.0,
                  log_every=10):
    """Stage 1: full enum + numba F. Returns residue compositions (cell-net ≤ 0)
    and grid-survivors (need refinement)."""
    _log(f"\n{'='*72}")
    _log(f"STAGE 1: Numba F + cell-net on FULL canonical enumeration")
    _log(f"  d={d}  S={S}  c_target={c_target}")
    _log(f"  batch_size={batch_size}  time_budget={time_budget}s")
    _log(f"{'='*72}")
    # JIT warmup
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    t_w = time.time()
    _prune_coarse_count_cell(warm, d, S, c_target)
    _log(f"  JIT warm done in {time.time()-t_w:.1f}s")

    n_total_processed = 0
    n_grid_pruned = 0
    n_grid_surv = 0
    n_cell_certified_F = 0
    n_cell_uncertain = 0
    residue = []  # compositions needing v4 tier_L
    grid_survivors = []  # need d-refinement
    min_net_seen = np.inf
    t0 = time.time()
    batch_i = 0

    for batch in generate_canonical_compositions_batched(d, S,
                                                            batch_size=batch_size):
        batch_i += 1
        batch_i32 = batch.astype(np.int32)
        survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
            batch_i32, d, S, c_target)
        n_total_processed += len(batch)
        n_p = int((~survived).sum())
        n_s = int(survived.sum())
        n_neg_batch = int(n_neg)
        n_grid_pruned += n_p
        n_grid_surv += n_s
        n_cell_uncertain += n_neg_batch
        n_cell_certified_F += (n_p - n_neg_batch)
        if min_net < min_net_seen:
            min_net_seen = float(min_net)
        # Collect residue (grid-pruned + cell-uncertain)
        residue_mask = (~survived) & neg_mask
        if residue_mask.any():
            for idx in np.where(residue_mask)[0]:
                residue.append(batch[idx].astype(np.int64).copy())
        # Collect grid-survivors (need d-refinement)
        if survived.any():
            for idx in np.where(survived)[0]:
                grid_survivors.append(batch[idx].astype(np.int64).copy())
        # Logging
        if batch_i <= 5 or batch_i % log_every == 0:
            elapsed = time.time() - t0
            rate = n_total_processed / max(elapsed, 1e-9)
            _log(f"  batch {batch_i:>4}: processed={n_total_processed:>12,}  "
                 f"grid_pruned={n_grid_pruned:>12,}  "
                 f"cell_uncertain={n_cell_uncertain:>9,}  "
                 f"grid_surv={n_grid_surv:>9,}  "
                 f"rate={rate/1e3:.1f}K/s  elapsed={elapsed:.1f}s  "
                 f"min_net={min_net_seen:.6f}")
        if time.time() - t0 > time_budget:
            _log(f"  STAGE 1 TIME BUDGET REACHED")
            break

    elapsed = time.time() - t0
    _log(f"\n  STAGE 1 SUMMARY:")
    _log(f"    processed:              {n_total_processed:,}")
    _log(f"    grid-pruned:            {n_grid_pruned:,} "
         f"({100*n_grid_pruned/max(n_total_processed,1):.4f}%)")
    _log(f"    grid-survivors:         {n_grid_surv:,} "
         f"(need d-refinement)")
    _log(f"    cell-certified by F:    {n_cell_certified_F:,} "
         f"({100*n_cell_certified_F/max(n_total_processed,1):.4f}%)")
    _log(f"    cell-uncertain residue: {n_cell_uncertain:,} "
         f"({100*n_cell_uncertain/max(n_total_processed,1):.4f}%)  "
         f"→ Stage 2 (v4 L tier)")
    _log(f"    min_net seen:           {min_net_seen:.6f}")
    _log(f"    elapsed:                {elapsed:.1f}s")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'processed': n_total_processed,
        'grid_pruned': n_grid_pruned,
        'grid_surv': n_grid_surv,
        'cell_certified_F': n_cell_certified_F,
        'cell_uncertain': n_cell_uncertain,
        'min_net_seen': min_net_seen,
        'elapsed_s': elapsed,
        'residue': residue,
        'grid_survivors': grid_survivors,
    }


def stage2_v4(residue, d, S, c_target, time_budget=600.0):
    """Stage 2: residue → v4 tier_L cascade."""
    _log(f"\n{'='*72}")
    _log(f"STAGE 2: v4 cell-cert (B1→F→L→L_joint→split) on {len(residue):,} residue cells")
    _log(f"  time_budget={time_budget}s")
    _log(f"{'='*72}")
    windows = v4.build_all_windows(d)
    # Warm SDP templates
    v4.get_sdp_template(d)
    v4.get_joint_template(d, 4)
    t0 = time.time()
    counts = {'B1': 0, 'empty': 0, 'F': 0, 'L': 0, 'L_joint': 0, 'split': 0}
    open_cells = []
    for k, c in enumerate(residue):
        if time.time() - t0 > time_budget:
            _log(f"  STAGE 2 TIME BUDGET REACHED at {k}/{len(residue)}")
            for cc in residue[k:]:
                open_cells.append(cc)
            break
        if k > 0 and k % max(1, len(residue) // 30) == 0:
            elapsed = time.time() - t0
            n_done = sum(counts.values())
            _log(f"  [{k:>6}/{len(residue):,}] closed={n_done:>5}  "
                 f"(B1={counts['B1']} empty={counts['empty']} F={counts['F']} "
                 f"L={counts['L']} Lj={counts['L_joint']} split={counts['split']})  "
                 f"open={len(open_cells)}  elapsed={elapsed:.1f}s")
        try:
            r = v4.certify_composition(c.astype(np.float64), S, d, c_target,
                                          windows=windows, max_depth=3)
        except Exception as e:
            open_cells.append(c)
            continue
        if r.certified:
            counts[r.tier_used] = counts.get(r.tier_used, 0) + 1
        else:
            open_cells.append(c)

    elapsed = time.time() - t0
    _log(f"\n  STAGE 2 SUMMARY:")
    _log(f"    closed by tier: B1={counts['B1']} empty={counts['empty']} "
         f"F={counts['F']} L={counts['L']} L_joint={counts['L_joint']} "
         f"split={counts['split']}")
    _log(f"    still open:     {len(open_cells)}")
    _log(f"    elapsed:        {elapsed:.1f}s")
    if open_cells:
        _log(f"    sample open compositions (first 5):")
        for c in open_cells[:5]:
            _log(f"      {c.tolist()}")
    return {
        'closed_counts': counts,
        'open_count': len(open_cells),
        'elapsed_s': elapsed,
        'open_samples': [c.tolist() for c in open_cells[:20]],
    }


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=12)
    ap.add_argument('--S', type=int, default=12)
    ap.add_argument('--c_target', type=float, default=1.25)
    ap.add_argument('--time_stage1', type=float, default=1500.0)
    ap.add_argument('--time_stage2', type=float, default=1500.0)
    ap.add_argument('--batch_size', type=int, default=200_000)
    ap.add_argument('--out', type=str, default='_prove_125_result.json')
    args = ap.parse_args()

    t_global = time.time()
    _log(f"\n{'#'*72}")
    _log(f"# PROVE C_{{1a}} >= {args.c_target}")
    _log(f"# Config: d={args.d}, S={args.S}")
    _log(f"# Time budget: {args.time_stage1+args.time_stage2:.0f}s")
    _log(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"{'#'*72}")

    s1 = stage1_numba(args.d, args.S, args.c_target,
                        batch_size=args.batch_size,
                        time_budget=args.time_stage1, log_every=5)
    s2 = stage2_v4(s1['residue'], args.d, args.S, args.c_target,
                     time_budget=args.time_stage2)

    _log(f"\n{'#'*72}")
    _log(f"# FINAL VERDICT @ d={args.d}, S={args.S}, c={args.c_target}")
    _log(f"{'#'*72}")
    _log(f"  Stage 1 processed: {s1['processed']:,}")
    _log(f"  Stage 1 grid-survivors (NEED d-refinement, NOT closed): {s1['grid_surv']:,}")
    _log(f"  Stage 1 cell-certified by F: {s1['cell_certified_F']:,}")
    _log(f"  Stage 1 cell-uncertain residue: {s1['cell_uncertain']:,}")
    _log(f"  Stage 2 closed by v4: {sum(s2['closed_counts'].values())}")
    _log(f"  Stage 2 still open after v4: {s2['open_count']}")
    total_unclosed = s1['grid_surv'] + s2['open_count']
    if total_unclosed == 0 and s1['processed'] >= s1.get('total', s1['processed']):
        _log(f"\n  *** PROOF COMPLETE: every cell certified at d={args.d}, S={args.S} ***")
        _log(f"  *** C_{{1a}} >= {args.c_target} ***")
    else:
        _log(f"\n  Proof INCOMPLETE: {total_unclosed:,} cells unclosed at this level.")
        _log(f"  (grid-survivors need d-refinement; v4-open need split/SDP)")
    _log(f"\n  Total wall time: {time.time()-t_global:.1f}s")

    # Save JSON (without numpy arrays)
    out = {
        'd': args.d, 'S': args.S, 'c_target': args.c_target,
        'stage1': {k: v for k, v in s1.items() if k not in ('residue', 'grid_survivors')},
        'stage2': s2,
        'total_wall_s': time.time() - t_global,
    }
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
    _log(f"\n  Saved: {args.out}")

"""Split-and-SDP killer for the 8 K=128 survivors.

For each surviving box (which K=128 couldn't certify), recursively SPLIT
along the widest axis to produce children that K=0/K=16 SDP can certify.

The math: McCormick LP gap is O(hw²). Each split halves a box's widest
dimension, so the LP gap of children shrinks by 4×. After 3 splits (8
children per parent), the gap is 64× smaller — easily below 1e-3 of
the target margin. K=0 SDP should certify nearly all children.

ALGORITHM
---------
  pending = load(final_survivors.npz)         # 8 boxes
  iter = 0
  while pending:
      iter += 1
      # 1. Split each pending box along widest axis to a SPLIT_DEPTH
      children = []
      for box in pending:
          children.extend(split_to_depth(box, SPLIT_DEPTH))
      # 2. LP-cert filter
      children = [c for c in children if not lp_cert(c)]
      # 3. K-ladder over children
      results = run_k_ladder(children, k_ladder=[0, 16, 32, 64, 128])
      # 4. Continue with K=128 survivors as next iter's pending
      pending = results.final_survivors
      if iter > MAX_ITERS: break

Each iter expects: ~8 → 64 children → ~1 K=128 survivor → 8 grandchildren → ...
Should converge in 2-3 iters.

OUTPUT
------
  runs/<tag>/kill_survivors_iter_N/  # per iter dir
  runs/<tag>/kill_survivors_journal.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from cert_pipeline.k_ladder import (
    SurvivorBox, run_k_ladder, survivors_to_npz, survivors_from_npz,
    DEFAULT_K_LADDER,
)
from cert_pipeline.box_journal import canonical_box_hash


class MathInsufficient(Exception):
    """Raised when a box cannot be split further (every integer axis has
    width <= 1, i.e. the box is a single dyadic-2^60 grid point) but
    has not been certified.

    A point box has val_B = max_W f(mu) at that point. If the cert
    failed at that point it means f(mu) < target — so val(d) < target
    and the proof attempt has failed by mathematical insufficiency, NOT
    a configuration / time-budget issue. Caller must raise the dimension
    d (or lower the target) — no amount of extra splitting will help.
    """


def split_box_to_depth(lo_int: List[int], hi_int: List[int],
                        target_depth: int) -> List[tuple]:
    """Split a box along its top-`target_depth` widest INTEGER axes,
    bisecting each ONCE. Produces up to 2^target_depth children. Each
    child has the chosen axes halved (integer midpoint) and other axes
    unchanged.

    WHY MULTI-AXIS (not same axis target_depth times):
    The McCormick LP gap on a box has a term
        scale_W * sum_{(i,j) in S_W} (hi_i - lo_i)(hi_j - lo_j)
    per window W. Splitting only the single widest axis shrinks the gap
    by 4x ONLY for windows whose binding (i,j) pairs include that axis;
    other windows see no improvement. By splitting `target_depth`
    DIFFERENT widest axes once each, we touch many more (i,j) pairs and
    bring the LP gap down across nearly all windows simultaneously.
    Same child count, much faster gap reduction.

    SATURATION HANDLING:
    Only axes with integer width >= 2 are splittable on the dyadic-2^60
    grid (a width-1 axis bisected at floor((lo+hi)/2) yields lo=mid).
    If FEWER than `target_depth` axes are splittable we still return
    2^k children where k = number of splittable axes (<= target_depth).
    If ZERO axes are splittable the box is a single grid point — we
    raise `MathInsufficient` so the caller can abort with a clean
    "lower target / raise d" message instead of silently looping.
    """
    if target_depth <= 0:
        return [(list(lo_int), list(hi_int))]
    # Rank axes by integer width (descending). Splittable axes have
    # integer width >= 2.
    widths = sorted(
        ((hi - lo, i) for i, (lo, hi) in enumerate(zip(lo_int, hi_int))),
        reverse=True,
    )
    splittable_axes: List[int] = [i for (w, i) in widths if w >= 2]
    if not splittable_axes:
        # Single grid point — caller should escalate (raise d / lower t).
        raise MathInsufficient(
            f'box is a single dyadic-2^60 grid point '
            f'(no axis with int width >= 2). lo_int[0..3]={lo_int[:3]}'
        )
    pick: List[int] = splittable_axes[:target_depth]
    # BFS-style: start with the single input box, bisect each picked
    # axis in turn so children = 2^len(pick).
    children: List[tuple] = [(list(lo_int), list(hi_int))]
    for axis in pick:
        new_children: List[tuple] = []
        for (clo, chi) in children:
            mid = (clo[axis] + chi[axis]) // 2
            # Re-check splittability for THIS specific child (axis may
            # have been narrowed by an upstream split — possible only
            # if the same axis appears twice in `pick`, which can't
            # happen since splittable_axes has unique entries).
            if mid <= clo[axis] or mid >= chi[axis]:
                # Conservatively keep the child unsplit on this axis
                # (rest of `pick` may still bisect successfully).
                new_children.append((clo, chi))
                continue
            left_lo, left_hi = list(clo), list(chi)
            left_hi[axis] = mid
            right_lo, right_hi = list(clo), list(chi)
            right_lo[axis] = mid
            new_children.append((left_lo, left_hi))
            new_children.append((right_lo, right_hi))
        children = new_children
    return children


def split_survivors(survivors: List[SurvivorBox],
                     split_depth: int) -> List[SurvivorBox]:
    """Split each survivor box `split_depth` times → list of children.

    Propagates `iters_survived` (children inherit parent count + 1) and
    surfaces `MathInsufficient` UP to the caller (annotated with the
    offending box hash) so the orchestrator can abort with a clean
    diagnosis. Children whose box doesn't intersect the simplex are
    dropped (sound: those boxes contain no feasible mu).
    """
    from interval_bnb.box import SCALE as _SCALE
    children: List[SurvivorBox] = []
    for s in survivors:
        try:
            kids = split_box_to_depth(s.lo_int, s.hi_int, split_depth)
        except MathInsufficient as e:
            raise MathInsufficient(
                f'box {s.hash} (depth={s.depth}, '
                f'iters_survived={s.iters_survived}) cannot be split: {e}'
            ) from e
        next_iters = s.iters_survived + 1
        next_depth = s.depth + split_depth
        for (clo, chi) in kids:
            # Drop boxes that don't intersect the simplex (sound:
            # empty intersection = no feasible mu = nothing to certify).
            if sum(clo) > _SCALE or sum(chi) < _SCALE:
                continue
            chash = canonical_box_hash(clo, chi)
            vol = float(np.prod([(h - l) / _SCALE for l, h in zip(clo, chi)]))
            children.append(SurvivorBox(
                hash=chash, lo_int=clo, hi_int=chi,
                depth=next_depth, volume=vol,
                lp_val=None,  # recomputed by lp_cert_filter
                src=f'split({s.hash})',
                iters_survived=next_iters,
            ))
    return children


def lp_cert_filter(survivors: List[SurvivorBox], d: int,
                    target: float) -> tuple[List[SurvivorBox], int]:
    """Compute LP for each child and split into (LP-failing, lp_cert_count)."""
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    from interval_bnb.box import SCALE as _SCALE
    windows = build_windows(d)
    out: List[SurvivorBox] = []
    n_lp_cert = 0
    for s in survivors:
        lo = np.array([float(x) / _SCALE for x in s.lo_int], dtype=np.float64)
        hi = np.array([float(x) / _SCALE for x in s.hi_int], dtype=np.float64)
        try:
            lp = float(bound_epigraph_lp_float(lo, hi, windows, d))
        except Exception:
            lp = float('nan')
        if np.isfinite(lp) and lp >= target:
            n_lp_cert += 1
            continue
        s.lp_val = lp if np.isfinite(lp) else None
        out.append(s)
    return out, n_lp_cert


def main():
    ap = argparse.ArgumentParser(
        description='Split-and-SDP killer for K-ladder survivors.')
    ap.add_argument('--input', type=str, required=True,
                    help='Path to survivors npz (e.g., final_survivors.npz)')
    ap.add_argument('--d', type=int, default=22)
    ap.add_argument('--target', type=str, default='1.2805')
    ap.add_argument('--output-dir', type=str, required=True,
                    help='Where to write kill_survivors_iter_N/ subdirs')
    ap.add_argument('--split-depth', type=int, default=4,
                    help='How many splits per box per iter (2^N children)')
    ap.add_argument('--max-iters', type=int, default=4,
                    help='Maximum kill iterations before giving up')
    ap.add_argument('--k-ladder', type=str, default='0,16,32,64,128')
    ap.add_argument('--sdp-time-limit-s', type=float, default=600.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_f = float(args.target)
    k_ladder = [int(k) for k in args.k_ladder.split(',')]

    pending = survivors_from_npz(args.input)
    print(f"\n#### KILL-SURVIVORS LOOP ####")
    print(f"  input: {args.input} ({len(pending)} boxes)")
    print(f"  target: {args.target}")
    print(f"  split_depth: {args.split_depth} (=> {2**args.split_depth} children/box)")
    print(f"  max_iters: {args.max_iters}")
    print(f"  output_dir: {out_dir}")

    iter_log: List[dict] = []
    iter_n = 0
    t_total = time.time()
    while pending and iter_n < args.max_iters:
        iter_n += 1
        iter_dir = out_dir / f'iter_{iter_n:03d}'
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== KILL ITER {iter_n} === ({len(pending)} pending)")

        t0 = time.time()
        children = split_survivors(pending, args.split_depth)
        print(f"  split: {len(pending)} → {len(children)} children "
              f"(after simplex filter)")
        survivors_to_npz(children,
                          str(iter_dir / 'children_after_split.npz'))

        t1 = time.time()
        survivors, n_lp_cert = lp_cert_filter(children, args.d, target_f)
        print(f"  lp_filter: {n_lp_cert} LP-cert  "
              f"{len(survivors)} LP-failing → SDP")

        if not survivors:
            print(f"  iter {iter_n}: ALL children certified by LP. KILLED.")
            iter_log.append({
                'iter': iter_n, 'pending_in': len(pending),
                'children': len(children), 'lp_cert': n_lp_cert,
                'sdp_input': 0, 'sdp_cert': 0, 'survivors': 0,
                'wall_s': time.time() - t0,
            })
            pending = []
            break

        t2 = time.time()
        ladder = run_k_ladder(args.d, target_f, survivors,
                                iter_dir / 'k_ladder', _REPO,
                                k_ladder=k_ladder,
                                time_limit_s=args.sdp_time_limit_s)
        n_sdp_cert = ladder.total_cert
        n_survivors_next = len(ladder.final_survivors)
        print(f"  k-ladder: {ladder.total_input} input  "
              f"{n_sdp_cert} cert  {n_survivors_next} survivors_next")
        iter_log.append({
            'iter': iter_n, 'pending_in': len(pending),
            'children': len(children), 'lp_cert': n_lp_cert,
            'sdp_input': len(survivors), 'sdp_cert': n_sdp_cert,
            'survivors': n_survivors_next,
            'wall_s': time.time() - t0,
            'split_s': t1 - t0, 'lp_s': t2 - t1,
            'sdp_s': time.time() - t2,
        })
        pending = ladder.final_survivors
        print(f"  iter {iter_n}: pending={len(pending)} → next iter")

    final_n = len(pending)
    summary = {
        'input': args.input,
        'd': args.d,
        'target': args.target,
        'split_depth': args.split_depth,
        'k_ladder': k_ladder,
        'max_iters': args.max_iters,
        'iters_run': iter_n,
        'final_survivors_count': final_n,
        'total_wall_s': time.time() - t_total,
        'iter_log': iter_log,
        'verdict': 'KILLED_ALL' if final_n == 0
                   else f'INCOMPLETE_{final_n}_SURVIVORS',
    }
    (out_dir / 'kill_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))
    survivors_to_npz(pending, str(out_dir / 'final_survivors_after_kill.npz'))
    print(f"\n{'#'*70}")
    print(f"# KILL DONE  iters={iter_n}  remaining={final_n}  "
          f"wall={summary['total_wall_s']:.0f}s ({summary['total_wall_s']/60:.1f} min)")
    print(f"# verdict: {summary['verdict']}")
    print(f"# output: {out_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()

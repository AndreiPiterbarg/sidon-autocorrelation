"""Random-walk convergence probe.

Idea: at each "level" take a random subset of survivors (no seed, fresh each
time), split each by --split-depth, run LP filter, count survivors. Repeat
on the LP-fail children. Track the per-parent multiplier:

    multiplier = (#LP-fails after split) / (#parents in subset)

If multiplier < 1, the cascade contracts at this level. If > 1 it expands.
The split factor is 2^split-depth (so per-axis we have 2^split children
before LP). The "free" multiplier (no LP closes anything) is 2^split-depth.
LP closes some fraction; we want the residual multiplier.

We track this until either:
  - multiplier drops below 1 (cascade contracts)
  - some max-levels limit is hit
  - ZERO survivors (already converged)

USAGE:
    python -u _convergence_probe.py \
        --input runs_local/d22_t1p2805_split_K9/iter_005/survivors_after_kladder.npz \
        --d 22 --target 1.2805 --split-depth 4 --subset-size 8 --max-levels 20

Note: at split-depth=4 → 16 children per parent. Subset 8 boxes → 128 children
per level. LP per child is fast (sub-second). Whole probe takes <1 hour.

Lower split-depth = faster per level but slower convergence.
Higher split-depth = each level closer to "real" run iteration.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cert_pipeline.k_ladder import SurvivorBox, survivors_from_npz
from cert_pipeline.kill_survivors import (
    split_survivors, lp_cert_filter, MathInsufficient,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='npz with current survivors (e.g. survivors_after_kladder.npz)')
    ap.add_argument('--d', type=int, default=22)
    ap.add_argument('--target', type=float, default=1.2805)
    ap.add_argument('--split-depth', type=int, default=4,
                    help='axes to bisect per level (2^k children per parent)')
    ap.add_argument('--subset-size', type=int, default=8,
                    help='random parents picked at each level')
    ap.add_argument('--max-levels', type=int, default=20)
    ap.add_argument('--out', default='_convergence_probe.json')
    args = ap.parse_args()

    seed = int(time.time_ns() & 0xFFFFFFFF)
    random.seed(seed)
    np.random.seed(seed)
    print(f'seed (fresh each run): {seed}', flush=True)

    survivors = survivors_from_npz(args.input)
    if not survivors:
        print('input has 0 survivors — already converged.', flush=True)
        return
    print(f'loaded {len(survivors)} survivors from {args.input}', flush=True)
    print(f'depth range: {min(s.depth for s in survivors)}'
          f'..{max(s.depth for s in survivors)}', flush=True)
    print(f'split-depth={args.split_depth} (= {2**args.split_depth} children/parent)',
          flush=True)
    print(f'subset-size={args.subset_size}, max-levels={args.max_levels}',
          flush=True)
    print(f'target={args.target}, d={args.d}', flush=True)
    print('=' * 80, flush=True)

    pool = list(survivors)
    log = []
    free_mult = 2 ** args.split_depth

    for level in range(1, args.max_levels + 1):
        if not pool:
            print(f'\n[level {level}] pool empty — CONVERGED at level {level-1}.',
                  flush=True)
            break
        n_pick = min(args.subset_size, len(pool))
        # uniform random WITHOUT replacement (no seed; fresh draw each level)
        picked = random.sample(pool, n_pick)

        t0 = time.time()
        try:
            children = split_survivors(picked, args.split_depth)
        except MathInsufficient as e:
            print(f'\n[level {level}] MathInsufficient: {e}', flush=True)
            log.append({
                'level': level, 'n_parents': n_pick,
                'error': 'MathInsufficient',
            })
            break
        n_children = len(children)
        t_split = time.time() - t0

        t0 = time.time()
        lp_fails, n_lp_cert = lp_cert_filter(children, args.d, args.target)
        t_lp = time.time() - t0

        n_lp_fail = len(lp_fails)
        # per-parent multiplier (LP-fail children per parent box)
        mult_per_parent = n_lp_fail / n_pick if n_pick > 0 else 0.0
        # raw close fraction (across the children)
        close_frac = n_lp_cert / n_children if n_children > 0 else 0.0

        depth_now = picked[0].depth
        contracts = 'CONTRACTS' if mult_per_parent < 1.0 else 'expands'
        print(f'\n[level {level}] depth_in={depth_now}+{args.split_depth} -> '
              f'depth_out={depth_now + args.split_depth}', flush=True)
        print(f'  parents picked: {n_pick}', flush=True)
        print(f'  children produced: {n_children} (free mult = {free_mult}; '
              f'simplex-clip lost {n_pick * free_mult - n_children})', flush=True)
        print(f'  LP cert: {n_lp_cert}/{n_children} = {100*close_frac:.1f}%', flush=True)
        print(f'  LP fail: {n_lp_fail}', flush=True)
        print(f'  >>> per-parent mult = {n_lp_fail}/{n_pick} = '
              f'{mult_per_parent:.3f} ({contracts})', flush=True)
        print(f'  t_split={t_split:.1f}s  t_lp={t_lp:.1f}s', flush=True)

        log.append({
            'level': level,
            'n_parents': n_pick,
            'depth_in': depth_now,
            'depth_out': depth_now + args.split_depth,
            'n_children': n_children,
            'n_lp_cert': n_lp_cert,
            'n_lp_fail': n_lp_fail,
            'close_frac': close_frac,
            'mult_per_parent': mult_per_parent,
            't_split_s': t_split,
            't_lp_s': t_lp,
        })

        if n_lp_fail == 0:
            print(f'\n[level {level}] all children LP-certed — CASCADE CLOSES HERE.',
                  flush=True)
            break

        # Pool for next level: the LP-failing children themselves
        pool = lp_fails

    # Summary
    print('\n' + '=' * 80, flush=True)
    print('SUMMARY (per-parent multiplier vs depth)', flush=True)
    print('=' * 80, flush=True)
    print(f'{"level":>5} {"depth_out":>10} {"close%":>8} '
          f'{"mult/parent":>12} {"verdict":>10}', flush=True)
    contracted_at = None
    for r in log:
        if 'error' in r:
            print(f'{r["level"]:>5} {"-":>10} {"-":>8} {"-":>12} ERROR', flush=True)
            continue
        verdict = 'contracts' if r['mult_per_parent'] < 1.0 else 'expands'
        if contracted_at is None and r['mult_per_parent'] < 1.0:
            contracted_at = r['level']
        print(f'{r["level"]:>5} {r["depth_out"]:>10} '
              f'{100*r["close_frac"]:>7.1f}% '
              f'{r["mult_per_parent"]:>12.3f} {verdict:>10}', flush=True)

    print()
    if contracted_at is not None:
        print(f'>>> Cascade FIRST CONTRACTS at level {contracted_at} '
              f'(depth_out = {log[contracted_at-1]["depth_out"]}).', flush=True)
    else:
        print('>>> Cascade did NOT contract within probe range — still expanding.',
              flush=True)

    import json
    with open(args.out, 'w') as f:
        json.dump({'seed': seed, 'args': vars(args), 'log': log}, f, indent=2)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()

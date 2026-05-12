"""Cascade feasibility sweep for cloud CPU.

Tests all (m, n_half) combinations for a given c_target.
At each level:
  1. Computes EXACT total children count (fast numpy, no enumeration)
  2. If total > budget, kills the config immediately
  3. Otherwise, processes a sample of parents to measure expansion
  4. Projects next level's parent count and repeats

The budget is based on measured throughput × cores × hours.

Usage:
    python tests/feasibility_sweep.py --c_target 1.35 --cores 64 --hours 100
    python tests/feasibility_sweep.py --c_target 1.40 --cores 64 --hours 50
"""
import argparse
import sys
import os
import time
import math
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused

C_UPPER = 1.5029


def x_cap_for(m, d_child, c_target):
    n_half_child = d_child // 2
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    xc = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    xc_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    return max(min(xc, xc_cs), 0)


def children_counts(parents, m, d_child, c_target):
    """Vectorized: number of children per parent (exact, no enumeration)."""
    xc = x_cap_for(m, d_child, c_target)
    B = parents.astype(np.int64)
    lo = np.maximum(0, 2 * B - xc)
    hi = np.minimum(2 * B, xc)
    eff = np.maximum(hi - lo + 1, 0)
    # Use float64 to avoid int overflow on large products
    return np.prod(eff.astype(np.float64), axis=1), xc


def run_config(n_half, m, c_target, budget, sample_n, max_levels):
    """Test one (n_half, m) config. Returns dict with results."""
    corr = correction(m)
    eff = c_target + corr
    if eff >= C_UPPER:
        print(f"  VACUOUS (eff={eff:.4f} >= {C_UPPER})")
        return {'status': 'vacuous'}

    # --- L0 ---
    t0 = time.time()
    l0 = run_level0(n_half, m, c_target, verbose=False)
    l0_time = time.time() - t0
    survivors = l0['survivors']
    n0 = len(survivors)
    print(f"  L0: {n0:,} surv ({l0_time:.1f}s)", flush=True)

    if n0 == 0:
        print(f"  PROVEN at L0!")
        return {'status': 'proven', 'proven_at': 'L0'}

    d_parent = 2 * n_half
    nhp = n_half
    cumulative_children = 0
    projected_parents = float(n0)  # tracks projected full parent count

    for lvl in range(1, max_levels + 1):
        d_child = 2 * d_parent
        nhc = 2 * nhp
        n_parents = len(survivors)

        if n_parents == 0:
            print(f"  L{lvl}: 0 parents -> PROVEN!")
            return {'status': 'proven', 'proven_at': f'L{lvl}'}

        # --- EXACT total children at this level ---
        # Compute from actual survivors, scale by projected/actual ratio
        counts, xc = children_counts(survivors, m, d_child, c_target)
        avg_children = float(np.mean(counts))
        total_children_this_level = avg_children * projected_parents
        proj_parents = projected_parents

        remaining_budget = budget - cumulative_children
        pct = total_children_this_level / budget * 100

        print(f"  L{lvl}: {n_parents:,} parents (proj {proj_parents:.2e}), "
              f"d={d_parent}->{d_child}, x_cap={xc}", flush=True)
        print(f"    Total children: {total_children_this_level:.2e} "
              f"({pct:.1f}% of budget)", flush=True)

        if total_children_this_level > remaining_budget:
            print(f"    EXCEEDS BUDGET: {total_children_this_level:.2e} > "
                  f"{remaining_budget:.2e} remaining "
                  f"({total_children_this_level/remaining_budget:.0f}x over)")
            return {'status': 'budget_exceeded', 'failed_at': f'L{lvl}',
                    'total_children': total_children_this_level,
                    'budget_pct': pct}

        cumulative_children += total_children_this_level
        print(f"    Cumulative: {cumulative_children:.2e} "
              f"({cumulative_children/budget*100:.1f}% of budget)", flush=True)

        # --- Process RANDOM sample to measure expansion ---
        # Use random parents (not lightest) for unbiased expansion estimate.
        # Skip parents > 500M children to avoid blocking forever.
        MAX_PER_PARENT = 500_000_000
        processable_mask = counts <= MAX_PER_PARENT
        n_processable = int(processable_mask.sum())

        if n_processable == 0:
            print(f"    No processable parents (lightest={counts.min():.2e})")
            print(f"    Children p1={np.percentile(counts,1):.2e} "
                  f"p50={np.percentile(counts,50):.2e} "
                  f"p99={np.percentile(counts,99):.2e}")
            print(f"    Cannot measure expansion (all parents too heavy)")
            return {'status': 'unmeasurable', 'failed_at': f'L{lvl}',
                    'budget_pct': cumulative_children/budget*100,
                    'lightest_parent': float(counts.min())}

        # Random sample from processable parents
        processable_idx = np.where(processable_mask)[0]
        rng = np.random.RandomState(lvl * 1000 + m)
        n_to_process = min(sample_n, n_processable)
        chosen = rng.choice(processable_idx, n_to_process, replace=False)
        sample_parents = survivors[chosen]

        pct_processable = n_processable / n_parents * 100
        print(f"    Processing {n_to_process} random parents "
              f"({n_processable}/{n_parents} = {pct_processable:.0f}% "
              f"under {MAX_PER_PARENT:,})...", flush=True)

        total_surv = 0
        total_tested = 0
        next_surv = []

        for i, parent in enumerate(sample_parents):
            t1 = time.time()
            s, nc = process_parent_fused(parent, m, c_target, nhc)
            dt = time.time() - t1
            total_tested += nc
            total_surv += len(s)
            if len(s) > 0:
                next_surv.append(s)

            # Print every parent
            print(f"      p{i}: {len(s):,}/{nc:,} surv/children ({dt:.1f}s)",
                  flush=True)

        avg_surv = total_surv / n_to_process
        expansion = avg_surv
        projected_next = proj_parents * expansion

        print(f"    Expansion: {expansion:,.0f}x "
              f"({total_surv:,}/{total_tested:,} from {n_to_process} parents)",
              flush=True)
        print(f"    Projected next level: {projected_next:.2e} parents",
              flush=True)

        if total_surv == 0:
            print(f"    EXPANSION = 0 -> CASCADE CONVERGES AT L{lvl}!")
            return {'status': 'converges', 'proven_at': f'L{lvl}',
                    'budget_pct': cumulative_children/budget*100,
                    'children_tested': total_tested,
                    'parents_tested': n_to_process}

        # Prepare next level
        if next_surv:
            survivors = np.vstack(next_surv)
        else:
            survivors = np.empty((0, d_child), dtype=np.int32)
        projected_parents = projected_next

        d_parent = d_child
        nhp = nhc

    return {'status': 'max_levels', 'budget_pct': cumulative_children/budget*100}


def main():
    ap = argparse.ArgumentParser(
        description='Cascade feasibility sweep for cloud CPU')
    ap.add_argument('--c_target', type=float, default=1.35)
    ap.add_argument('--cores', type=int, default=64)
    ap.add_argument('--hours', type=int, default=100)
    ap.add_argument('--m_values', type=str, default='20,25,30',
                    help='Comma-separated m values')
    ap.add_argument('--n_half_values', type=str, default='2',
                    help='Comma-separated n_half values')
    ap.add_argument('--sample', type=int, default=20,
                    help='Parents to sample per level')
    ap.add_argument('--max_levels', type=int, default=6)
    ap.add_argument('--rate', type=float, default=0.0,
                    help='Measured children/sec/core (0 = measure from L1)')
    args = ap.parse_args()

    m_values = [int(x) for x in args.m_values.split(',')]
    n_half_values = [int(x) for x in args.n_half_values.split(',')]

    # Compute budget
    # If rate not provided, use conservative estimate
    rate = args.rate if args.rate > 0 else 6.0e6  # conservative default
    budget = rate * args.cores * args.hours * 3600
    print(f"{'='*60}")
    print(f"Cascade Feasibility Sweep")
    print(f"{'='*60}")
    print(f"c_target: {args.c_target}")
    print(f"Compute: {args.cores} cores × {args.hours}h")
    print(f"Rate: {rate/1e6:.1f}M children/sec/core")
    print(f"Budget: {budget:.2e} children")
    print(f"Configs: m={m_values}, n_half={n_half_values}")
    print(f"Sample: {args.sample} parents/level")
    print(f"{'='*60}\n")

    results = []

    for n_half in n_half_values:
        for m in m_values:
            print(f"\n--- n_half={n_half}, m={m} ---", flush=True)
            t0 = time.time()
            result = run_config(n_half, m, args.c_target, budget,
                                args.sample, args.max_levels)
            result['n_half'] = n_half
            result['m'] = m
            result['time'] = time.time() - t0
            results.append(result)
            print(f"  [{result['status']}] ({result['time']:.1f}s)\n", flush=True)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: c_target={args.c_target}, budget={budget:.2e}")
    print(f"{'='*60}")
    print(f"{'n_half':>6} {'m':>4} {'status':>20} {'detail':>30}")
    print(f"{'-'*65}")
    for r in results:
        detail = ''
        if r['status'] == 'converges':
            detail = f"PROVEN at {r['proven_at']}, {r['budget_pct']:.1f}% budget"
        elif r['status'] == 'budget_exceeded':
            detail = f"failed at {r['failed_at']}, {r['budget_pct']:.0f}% budget"
        elif r['status'] == 'unmeasurable':
            detail = (f"fits at {r['failed_at']} ({r['budget_pct']:.1f}%) "
                      f"but lightest={r['lightest_parent']:.1e}")
        elif r['status'] == 'proven':
            detail = f"at {r['proven_at']}"
        elif r['status'] == 'vacuous':
            detail = ''
        print(f"{r['n_half']:>6} {r['m']:>4} {r['status']:>20} {detail:>30}")

    feasible = [r for r in results if r['status'] in ('converges', 'proven')]
    if feasible:
        print(f"\nFEASIBLE CONFIGS:")
        for r in feasible:
            print(f"  n_half={r['n_half']}, m={r['m']}: "
                  f"{r['status']} at {r.get('proven_at','?')}")
    else:
        print(f"\nNO FEASIBLE CONFIGS FOUND for c_target={args.c_target}")


if __name__ == '__main__':
    main()

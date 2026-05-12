"""Deep cascade sweep with random child sampling for levels where full
enumeration is infeasible.

For each level, samples random parents, generates random children from
each parent's cursor ranges, tests them for pruning, and estimates the
survival rate and expansion factor.

Usage:
    python tests/deep_cascade_sweep.py --m 16 --n_half 1 --c_target 1.30
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import (run_level0, process_parent_fused,
                         _compute_bin_ranges, _prune_dynamic_int32,
                         _tighten_ranges)


def sample_random_children(parent_int, lo_arr, hi_arr, n_samples, rng):
    """Generate n_samples random children by sampling each cursor uniformly."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    children = np.empty((n_samples, d_child), dtype=np.int32)
    for pos in range(d_parent):
        lo, hi = int(lo_arr[pos]), int(hi_arr[pos])
        # cursor value for position pos
        cursor_vals = rng.integers(lo, hi + 1, size=n_samples)
        # child bins: (cursor, 2*parent - cursor)
        children[:, 2 * pos] = cursor_vals
        children[:, 2 * pos + 1] = 2 * int(parent_int[pos]) - cursor_vals
    return children


def estimate_level_sampled(survivors, m, c_target, n_half_child,
                           n_parent_samples, n_child_samples_per_parent,
                           rng, use_flat_threshold=False, level=0):
    """Estimate survival rate at a level by sampling random children."""
    d_parent = survivors.shape[1]
    d_child = 2 * d_parent
    n_parents = len(survivors)

    # Sample parents randomly
    if n_parents > n_parent_samples:
        parent_idx = rng.choice(n_parents, n_parent_samples, replace=False)
        sampled_parents = survivors[parent_idx]
    else:
        sampled_parents = survivors
        n_parent_samples = n_parents

    total_children_tested = 0
    total_survivors_found = 0
    total_cart_product = 0
    parent_details = []
    survivor_children = []

    for i, parent in enumerate(sampled_parents):
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            parent_details.append({
                'idx': i, 'cart_product': 0, 'tested': 0,
                'survived': 0, 'rate': 0.0
            })
            continue
        lo_arr, hi_arr, cart_product = result

        # Tighten ranges via arc consistency
        cart_product_raw = _tighten_ranges(parent, lo_arr, hi_arr, m, c_target,
                                           n_half_child, use_flat_threshold)
        if cart_product_raw == 0:
            parent_details.append({
                'idx': i, 'cart_product': 0, 'tested': 0,
                'survived': 0, 'rate': 0.0
            })
            continue

        # Recompute cart product in Python (arbitrary precision) to avoid
        # int64 overflow in Numba's return value
        cart_product = 1
        for pos in range(len(parent)):
            rng_size = int(hi_arr[pos]) - int(lo_arr[pos]) + 1
            if rng_size <= 0:
                cart_product = 0
                break
            cart_product *= rng_size

        if cart_product == 0:
            parent_details.append({
                'idx': i, 'cart_product': 0, 'tested': 0,
                'survived': 0, 'rate': 0.0
            })
            continue

        total_cart_product += cart_product

        # If small enough, enumerate fully
        if cart_product <= n_child_samples_per_parent:
            surv, n_ch = process_parent_fused(parent, m, c_target, n_half_child,
                                               use_flat_threshold=use_flat_threshold)
            n_tested = n_ch
            n_survived = len(surv)
            if n_survived > 0:
                survivor_children.append(surv)
        else:
            # Sample random children
            n_to_sample = n_child_samples_per_parent
            children = sample_random_children(parent, lo_arr, hi_arr,
                                              n_to_sample, rng)
            # Test them
            survived_mask = _prune_dynamic_int32(children, n_half_child, m,
                                                  c_target, use_flat_threshold)
            n_tested = n_to_sample
            n_survived = int(survived_mask.sum())
            if n_survived > 0:
                survivor_children.append(children[survived_mask])

        total_children_tested += n_tested
        total_survivors_found += n_survived

        rate = n_survived / n_tested if n_tested > 0 else 0.0
        parent_details.append({
            'idx': i, 'cart_product': cart_product, 'tested': n_tested,
            'survived': n_survived, 'rate': rate
        })

        print(f"    [P{i+1}/{n_parent_samples}] cart={cart_product:,.0f}, "
              f"tested={n_tested:,}, surv={n_survived:,} "
              f"(rate={rate:.6f})", flush=True)

    # Compute statistics
    rates = [d['rate'] for d in parent_details if d['tested'] > 0]
    cart_products = [d['cart_product'] for d in parent_details if d['cart_product'] > 0]

    if not rates:
        return None, parent_details

    # Weighted average survival rate (weight by cart product size)
    weighted_num = sum(d['survived'] for d in parent_details)
    weighted_den = sum(d['tested'] for d in parent_details)
    overall_rate = weighted_num / weighted_den if weighted_den > 0 else 0.0

    # Per-parent expansion = rate * cart_product
    expansions = []
    for d in parent_details:
        if d['tested'] > 0 and d['cart_product'] > 0:
            expansions.append(d['rate'] * d['cart_product'])

    avg_expansion = np.mean(expansions) if expansions else 0.0
    median_expansion = np.median(expansions) if expansions else 0.0
    avg_cart = np.mean(cart_products) if cart_products else 0.0
    median_cart = np.median(cart_products) if cart_products else 0.0

    # Estimated total survivors = avg_survivors_per_parent * n_parents
    est_total_survivors = avg_expansion * n_parents
    est_total_children = avg_cart * n_parents

    # Collect survivor children for next level
    if survivor_children:
        all_surv = np.vstack(survivor_children)
    else:
        all_surv = np.empty((0, d_child), dtype=np.int32)

    stats = {
        'overall_rate': overall_rate,
        'avg_expansion': avg_expansion,
        'median_expansion': median_expansion,
        'avg_cart': avg_cart,
        'median_cart': median_cart,
        'est_total_survivors': est_total_survivors,
        'est_total_children': est_total_children,
        'n_parents': n_parents,
        'n_parents_sampled': n_parent_samples,
        'total_tested': total_children_tested,
        'total_survived': total_survivors_found,
        'rates': rates,
        'expansions': expansions,
        'survivor_children': all_surv,
    }
    return stats, parent_details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=16)
    ap.add_argument('--n_half', type=int, default=1)
    ap.add_argument('--c_target', type=float, default=1.30)
    ap.add_argument('--n_parent_samples', type=int, default=30,
                    help='parents to sample per level')
    ap.add_argument('--n_child_samples', type=int, default=50000,
                    help='random children to test per parent')
    ap.add_argument('--max_levels', type=int, default=8,
                    help='max cascade levels to explore')
    ap.add_argument('--use_flat_threshold', action='store_true')
    ap.add_argument('--seed', type=int, default=None,
                    help='RNG seed (default: random)')
    args = ap.parse_args()

    m, n_half, c_target = args.m, args.n_half, args.c_target
    flat = args.use_flat_threshold

    # Truly random seed if not specified
    if args.seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    else:
        seed = args.seed
    rng = np.random.default_rng(seed)

    corr = correction(m, n_half)
    d0 = 2 * n_half
    S0 = 4 * n_half * m
    n_compositions = count_compositions(d0, S0)

    print("=" * 70)
    print(f"DEEP CASCADE SWEEP")
    print(f"=" * 70)
    print(f"Config: m={m}, n_half={n_half} (d0={d0}), c_target={c_target}, flat={flat}")
    print(f"RNG seed: {seed}")
    print(f"Parent samples/level: {args.n_parent_samples}")
    print(f"Child samples/parent: {args.n_child_samples:,}")
    print(f"L0: d={d0}, S={S0}, compositions={n_compositions:,}")
    print(f"    correction={corr:.6f}, threshold={c_target+corr:.6f}")
    print()

    # --- L0: run fully ---
    t0 = time.time()
    result = run_level0(n_half, m, c_target, verbose=True,
                        use_flat_threshold=flat)
    survivors = result['survivors']
    n_surv = result['n_survivors']
    print(f"\nL0 done: {n_surv:,} survivors in {time.time()-t0:.1f}s")

    if n_surv == 0:
        print("PROVEN at L0!")
        return

    # --- Cascade levels ---
    cumulative_expansion = float(n_surv)
    for level in range(1, args.max_levels + 1):
        d_parent = survivors.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2
        n_parents = len(survivors)

        print(f"\n{'=' * 70}")
        print(f"L{level}: d_parent={d_parent} -> d_child={d_child}, "
              f"{n_parents:,} parents (est total={cumulative_expansion:,.0f})")
        print(f"{'=' * 70}")

        # Correction at child dimension
        corr_child = correction(m, n_half_child)
        print(f"    correction(m={m}, n_half={n_half_child})={corr_child:.6f}")

        t0 = time.time()

        # Check typical cart product size from a few random parents
        sample_carts = []
        n_probe = min(10, n_parents)
        for _ in range(n_probe):
            p = survivors[rng.integers(0, n_parents)]
            r = _compute_bin_ranges(p, m, c_target, d_child, n_half_child)
            if r is not None:
                # Recompute in Python to avoid int64 overflow
                lo_a, hi_a, _ = r
                cp = 1
                for pos in range(len(p)):
                    cp *= int(hi_a[pos]) - int(lo_a[pos]) + 1
                sample_carts.append(cp)
        if sample_carts:
            med_cart = np.median(sample_carts)
            print(f"    probe cart products: median={med_cart:,.0f}, "
                  f"range=[{min(sample_carts):,.0f}, {max(sample_carts):,.0f}]")

        # Decide: full enumeration or sampling
        use_full = False
        if n_parents <= args.n_parent_samples and sample_carts:
            if max(sample_carts) <= 10_000_000:
                use_full = True

        if use_full:
            print(f"    Mode: FULL enumeration")
            all_surv = []
            total_ch = 0
            total_sv = 0
            for i, parent in enumerate(survivors):
                surv_i, n_ch_i = process_parent_fused(
                    parent, m, c_target, n_half_child,
                    use_flat_threshold=flat)
                total_ch += n_ch_i
                n_sv_i = len(surv_i)
                total_sv += n_sv_i
                if n_sv_i > 0:
                    all_surv.append(surv_i)
                print(f"    [P{i+1}/{n_parents}] "
                      f"children={n_ch_i:,}, surv={n_sv_i:,}", flush=True)

            elapsed = time.time() - t0
            avg_surv = total_sv / n_parents if n_parents > 0 else 0
            expansion_factor = avg_surv
            cumulative_expansion *= expansion_factor

            print(f"\n  L{level} FULL summary ({elapsed:.1f}s):")
            print(f"    total children:  {total_ch:,}")
            print(f"    total survivors: {total_sv:,}")
            print(f"    avg surv/parent: {avg_surv:.1f}")
            print(f"    expansion factor: {expansion_factor:.2f}x")
            print(f"    cumulative est survivors: {cumulative_expansion:,.0f}")

            if total_sv == 0:
                print(f"\n*** ALL PRUNED at L{level}! Cascade converges. ***")
                return

            survivors = np.vstack(all_surv)
        else:
            print(f"    Mode: SAMPLING ({args.n_child_samples:,} children/parent)")

            stats, details = estimate_level_sampled(
                survivors, m, c_target, n_half_child,
                args.n_parent_samples, args.n_child_samples,
                rng, use_flat_threshold=flat, level=level)

            elapsed = time.time() - t0

            if stats is None:
                print(f"\n  L{level}: no parents with valid ranges, stopping")
                return

            expansion_factor = stats['avg_expansion']
            cumulative_expansion *= expansion_factor

            print(f"\n  L{level} SAMPLING summary ({elapsed:.1f}s):")
            print(f"    parents sampled:     {stats['n_parents_sampled']} "
                  f"of {stats['n_parents']:,}")
            print(f"    total tested:        {stats['total_tested']:,}")
            print(f"    total survived:      {stats['total_survived']:,}")
            print(f"    overall surv rate:   {stats['overall_rate']:.8f}")
            print(f"    avg cart product:    {stats['avg_cart']:,.0f}")
            print(f"    median cart product: {stats['median_cart']:,.0f}")
            print(f"    avg surv/parent (rate*cart): {stats['avg_expansion']:,.1f}")
            print(f"    median surv/parent:  {stats['median_expansion']:,.1f}")
            print(f"    est total survivors: {stats['est_total_survivors']:,.0f}")
            print(f"    est total children:  {stats['est_total_children']:,.0f}")
            print(f"    cumulative est surv: {cumulative_expansion:,.0f}")

            # Detailed rate distribution
            rates = stats['rates']
            if rates:
                nonzero_rates = [r for r in rates if r > 0]
                print(f"\n    Rate distribution ({len(rates)} parents):")
                print(f"      mean:    {np.mean(rates):.8f}")
                print(f"      median:  {np.median(rates):.8f}")
                print(f"      min:     {np.min(rates):.8f}")
                print(f"      max:     {np.max(rates):.8f}")
                print(f"      std:     {np.std(rates):.8f}")
                print(f"      nonzero: {len(nonzero_rates)}/{len(rates)}")
                if nonzero_rates:
                    print(f"      mean(nonzero): {np.mean(nonzero_rates):.8f}")

            exps = stats['expansions']
            if exps:
                print(f"\n    Expansion distribution (surv/parent):")
                print(f"      mean:   {np.mean(exps):,.1f}")
                print(f"      median: {np.median(exps):,.1f}")
                print(f"      min:    {np.min(exps):,.1f}")
                print(f"      max:    {np.max(exps):,.1f}")
                print(f"      std:    {np.std(exps):,.1f}")

            if stats['total_survived'] == 0:
                print(f"\n*** ALL SAMPLED CHILDREN PRUNED at L{level}! ***")
                print(f"    ({stats['total_tested']:,} tested, 0 survived)")
                print(f"    Upper bound on rate: < {1/stats['total_tested']:.2e}")
                print(f"\n*** No survivors to seed next level. Stopping. ***")
                return

            # Use found survivors for next level
            survivors = stats['survivor_children']
            if len(survivors) == 0:
                print(f"\n*** No survivor children collected. Stopping. ***")
                return

            print(f"\n    Collected {len(survivors):,} actual survivors "
                  f"for next level seeding")

    print(f"\n{'=' * 70}")
    print(f"Reached max level {args.max_levels}. Final survivors: {len(survivors):,}")
    print(f"Cumulative estimated survivors: {cumulative_expansion:,.0f}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

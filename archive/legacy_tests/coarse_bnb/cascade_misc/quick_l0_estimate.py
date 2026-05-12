"""Quick L0 survival-rate estimator via random sampling.

Instead of enumerating all compositions (billions), draw uniform random
compositions on the simplex, run them through the same pruning pipeline,
and estimate survival rate.  Then project total survivors and run L1+
on the sampled survivors.

Usage:
    python tests/quick_l0_estimate.py --m_values 15,20,25,30 --n_half 3 --c_target 1.30 --n_samples 500000
"""
import argparse
import os
import sys
import time

import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions, asymmetry_prune_mask, _canonical_mask
from run_cascade import _prune_dynamic_int32, process_parent_fused


def sample_compositions_uniform(d, S, n_samples, rng=None):
    """Sample n_samples uniform random compositions of S into d parts.

    Uses the "stars and bars" method: sample d-1 breakpoints uniformly
    from {0, ..., S}, sort them, and take differences.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample breakpoints
    breaks = rng.integers(0, S + 1, size=(n_samples, d - 1))
    breaks = np.sort(breaks, axis=1)

    # Prepend 0 and append S, take differences
    zeros = np.zeros((n_samples, 1), dtype=breaks.dtype)
    fulls = np.full((n_samples, 1), S, dtype=breaks.dtype)
    extended = np.concatenate([zeros, breaks, fulls], axis=1)
    compositions = np.diff(extended, axis=1).astype(np.int32)

    return compositions


def estimate_l0(n_half, m, c_target, n_samples, rng=None):
    """Estimate L0 survival rate by sampling."""
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    corr = correction(m, n_half)

    print(f"\n  m={m}: d={d}, S={S}, total_comps={n_total:,}, corr={corr:.6f}, "
          f"thresh={c_target+corr:.6f}")

    t0 = time.time()

    # Sample random compositions
    batch = sample_compositions_uniform(d, S, n_samples, rng=rng)

    # Canonical filter
    canon = _canonical_mask(batch)
    canon_rate = canon.sum() / len(batch)
    batch_canon = batch[canon]

    if len(batch_canon) == 0:
        print(f"  m={m}: No canonical samples")
        return None

    # Asymmetry filter
    needs_check = asymmetry_prune_mask(batch_canon, n_half, m, c_target)
    asym_survive_rate = needs_check.sum() / len(batch_canon)
    candidates = batch_canon[needs_check]

    if len(candidates) == 0:
        elapsed = time.time() - t0
        print(f"  m={m}: All pruned by asymmetry ({elapsed:.1f}s)")
        return {
            'm': m, 'n_samples': n_samples, 'n_total': n_total,
            'survival_rate': 0.0, 'projected_survivors': 0,
            'canon_rate': canon_rate, 'asym_survive_rate': asym_survive_rate,
            'test_survive_rate': 0.0, 'elapsed': elapsed,
            'survivors': np.empty((0, d), dtype=np.int32)
        }

    # Test-value pruning (same as real L0)
    survived_mask = _prune_dynamic_int32(candidates, n_half, m, c_target)
    n_survived = survived_mask.sum()
    test_survive_rate = n_survived / len(candidates) if len(candidates) > 0 else 0.0

    survivors = candidates[survived_mask]

    # Overall survival rate (among all sampled compositions)
    overall_rate = n_survived / n_samples

    # Project to full L0
    # Account for canonical: ~half of compositions are canonical
    projected_survivors = int(overall_rate * n_total)

    elapsed = time.time() - t0

    print(f"  m={m}: {n_survived:,}/{n_samples:,} survived "
          f"(rate={overall_rate:.6f})")
    print(f"  m={m}: canon_rate={canon_rate:.3f}, "
          f"asym_survive={asym_survive_rate:.3f}, "
          f"test_survive={test_survive_rate:.4f}")
    print(f"  m={m}: Projected L0 survivors: ~{projected_survivors:,} "
          f"(of {n_total:,})")
    print(f"  m={m}: {elapsed:.1f}s")

    return {
        'm': m, 'n_samples': n_samples, 'n_total': n_total,
        'survival_rate': overall_rate,
        'projected_survivors': projected_survivors,
        'canon_rate': canon_rate,
        'asym_survive_rate': asym_survive_rate,
        'test_survive_rate': test_survive_rate,
        'elapsed': elapsed,
        'survivors': survivors
    }


def run_l1_on_survivors(survivors, m, c_target, n_half, max_parents=10000):
    """Run L1 refinement on L0 survivors (or a sample of them)."""
    if len(survivors) == 0:
        return {'l1_survivors': 0, 'l1_children': 0, 'l1_expansion': 0.0}

    n_half_child = 2 * n_half  # L1 has double the n_half

    if len(survivors) > max_parents:
        idx = np.random.default_rng(42).choice(len(survivors), max_parents, replace=False)
        sampled = survivors[idx]
        scale = len(survivors) / max_parents
    else:
        sampled = survivors
        scale = 1.0

    print(f"\n  [L1] Processing {len(sampled):,} parents "
          f"(of {len(survivors):,} L0 survivors)...")

    t0 = time.time()
    total_surv = 0
    total_children = 0

    for i, parent in enumerate(sampled):
        surv, n_ch = process_parent_fused(parent, m, c_target, n_half_child)
        total_surv += len(surv)
        total_children += n_ch

        if (i + 1) % max(1, len(sampled) // 5) == 0:
            elapsed = time.time() - t0
            print(f"       {i+1:,}/{len(sampled):,} | "
                  f"{total_surv:,} survivors | {elapsed:.1f}s")

    elapsed = time.time() - t0
    projected_surv = int(total_surv * scale)
    expansion = total_surv / len(sampled) if len(sampled) > 0 else 0

    print(f"  [L1] Done: {total_surv:,} survivors from {len(sampled):,} parents "
          f"({elapsed:.1f}s)")
    print(f"  [L1] Expansion: {expansion:.1f}x, "
          f"Projected: ~{projected_surv:,}")

    return {
        'l1_survivors': total_surv,
        'l1_children': total_children,
        'l1_expansion': expansion,
        'l1_projected': projected_surv,
        'l1_elapsed': elapsed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_values', type=str, default='15,20,25,30')
    parser.add_argument('--n_half', type=int, default=3)
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--n_samples', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max_l1_parents', type=int, default=5000)
    args = parser.parse_args()

    m_values = [int(x) for x in args.m_values.split(',')]
    rng = np.random.default_rng(args.seed)

    print(f"Quick L0 Estimator: n_half={args.n_half}, c_target={args.c_target}, "
          f"n_samples={args.n_samples:,}")
    print(f"m values: {m_values}")
    print("=" * 70)

    results = []
    for m in m_values:
        # Check vacuity
        corr = correction(m)
        if args.c_target + corr >= 1.5029:
            print(f"\n  m={m}: VACUOUS (thresh={args.c_target+corr:.4f} >= 1.5029)")
            continue

        r = estimate_l0(args.n_half, m, args.c_target, args.n_samples, rng=rng)
        if r is not None and len(r['survivors']) > 0:
            l1 = run_l1_on_survivors(r['survivors'], m, args.c_target,
                                      args.n_half, args.max_l1_parents)
            r.update(l1)
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: n_half={args.n_half}, c_target={args.c_target}, "
          f"samples={args.n_samples:,}")
    print("=" * 70)
    print(f"{'m':>5} | {'L0 rate':>10} | {'Proj L0 surv':>14} | "
          f"{'L0 total comps':>16} | {'L1 exp':>8} | {'L1 proj':>10}")
    print("-" * 70)
    for r in results:
        if r is None:
            continue
        l1_exp = f"{r.get('l1_expansion', 0):.1f}x" if 'l1_expansion' in r else "N/A"
        l1_proj = f"{r.get('l1_projected', 0):,}" if 'l1_projected' in r else "N/A"
        print(f"{r['m']:>5} | {r['survival_rate']:>10.6f} | "
              f"{r['projected_survivors']:>14,} | "
              f"{r['n_total']:>16,} | {l1_exp:>8} | {l1_proj:>10}")


if __name__ == '__main__':
    main()

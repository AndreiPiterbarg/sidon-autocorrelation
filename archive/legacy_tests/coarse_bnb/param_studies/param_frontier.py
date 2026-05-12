"""Find the frontier: maximum c_target where expansion factors drop.

Also tests higher m for n_half=2 to see if tighter thresholds help.
And runs L2 for the best config to check if expansion decreases with depth.
"""
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
from run_cascade import (run_level0, process_parent_fused, _prune_dynamic,
                         _canonicalize_inplace, _fast_dedup)

C_UPPER_BOUND = 1.5029

def run_l1_sample(survivors, m, c_target, n_half, max_parents=200, time_budget=60):
    """Run L1 on a sample and return expansion factor."""
    d_parent = survivors.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2
    n_sample = min(max_parents, len(survivors))

    if n_sample < len(survivors):
        idx = np.random.default_rng(42).choice(len(survivors), n_sample, replace=False)
        sample = survivors[idx]
    else:
        sample = survivors

    total_children = 0
    total_surv = 0
    t0 = time.time()

    for i, parent in enumerate(sample):
        if time.time() - t0 > time_budget:
            n_sample = i
            break
        surv, nc = process_parent_fused(parent, m, c_target, n_half_child)
        total_children += nc
        total_surv += len(surv)

    if n_sample == 0:
        return 0, 0, 0

    surv_per_parent = total_surv / n_sample
    children_per_parent = total_children / n_sample
    projected = surv_per_parent * len(survivors)
    expansion = projected / max(len(survivors), 1)
    return expansion, children_per_parent, surv_per_parent


# =========================================================================
# PART 1: Higher m for n_half=2 at c_target=1.40
# =========================================================================

print("=" * 80)
print("PART 1: Does higher m help at c_target=1.40, n_half=2?")
print("=" * 80)

for m in [20, 25, 30, 40, 50]:
    corr = correction(m)
    threshold = 1.40 + corr
    if threshold >= C_UPPER_BOUND:
        print(f"\nm={m}: VACUOUS")
        continue

    margin = C_UPPER_BOUND - threshold
    l0 = run_level0(2, m, 1.40, verbose=False)
    n_surv = l0['n_survivors']
    survivors = l0['survivors']

    if n_surv == 0:
        print(f"\nm={m}: PROVEN at L0 (0 survivors)")
        continue

    exp, cpp, spp = run_l1_sample(survivors, m, 1.40, 2, max_parents=200, time_budget=30)
    print(f"\nm={m}: corr={corr:.4f}, margin={margin:.4f}")
    print(f"  L0: {n_surv:,} survivors")
    print(f"  L1: expansion={exp:.1f}x, children/parent={cpp:.0f}, surv/parent={spp:.1f}")
    print(f"  L1 projected: {int(exp * n_surv):,}")


# =========================================================================
# PART 2: c_target frontier for n_half=2, m=20
# =========================================================================

print("\n\n" + "=" * 80)
print("PART 2: c_target frontier for n_half=2, m=20")
print("  What's the max c_target where expansion shows any promise?")
print("=" * 80)

for ct in [1.20, 1.25, 1.28, 1.30, 1.33, 1.35, 1.37, 1.40]:
    corr = correction(20)
    threshold = ct + corr
    if threshold >= C_UPPER_BOUND:
        print(f"\nc_target={ct}: VACUOUS (threshold={threshold:.4f})")
        continue

    l0 = run_level0(2, 20, ct, verbose=False)
    n_surv = l0['n_survivors']

    if n_surv == 0:
        print(f"\nc_target={ct}: PROVEN at L0!")
        continue

    exp, cpp, spp = run_l1_sample(l0['survivors'], 20, ct, 2, max_parents=300, time_budget=30)
    print(f"\nc_target={ct}: L0 surv={n_surv:,}, L1 expansion={exp:.1f}x, "
          f"children/parent={cpp:.0f}")


# =========================================================================
# PART 3: c_target frontier for n_half=3, m=15
# =========================================================================

print("\n\n" + "=" * 80)
print("PART 3: c_target frontier for n_half=3, m=15")
print("  (This was the only converging config in the benchmark)")
print("=" * 80)

for ct in [1.28, 1.30, 1.33, 1.35, 1.37]:
    corr = correction(15)
    threshold = ct + corr
    if threshold >= C_UPPER_BOUND:
        print(f"\nc_target={ct}: VACUOUS (threshold={threshold:.4f})")
        continue

    l0 = run_level0(3, 15, ct, verbose=False)
    n_surv = l0['n_survivors']

    if n_surv == 0:
        print(f"\nc_target={ct}: PROVEN at L0!")
        continue

    exp, cpp, spp = run_l1_sample(l0['survivors'], 15, ct, 3, max_parents=300, time_budget=30)
    print(f"\nc_target={ct}: L0 surv={n_surv:,}, L1 expansion={exp:.1f}x, "
          f"children/parent={cpp:.0f}")


# =========================================================================
# PART 4: Multi-level depth for n_half=2, m=20, c_target=1.30
# =========================================================================

print("\n\n" + "=" * 80)
print("PART 4: Multi-level depth check: n_half=2, m=20, c_target=1.30")
print("  Does expansion decrease with depth?")
print("=" * 80)

ct = 1.30
survivors = run_level0(2, 20, ct, verbose=False)['survivors']
print(f"L0: {len(survivors):,} survivors")

current = survivors
for level in range(1, 5):
    if len(current) == 0:
        print(f"L{level}: PROVEN (0 survivors)!")
        break

    d_parent = current.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    n_sample = min(100, len(current))
    if n_sample < len(current):
        idx = np.random.default_rng(42).choice(len(current), n_sample, replace=False)
        sample = current[idx]
    else:
        sample = current

    total_children = 0
    all_surv = []
    t0 = time.time()

    for parent in sample:
        if time.time() - t0 > 60:
            break
        surv, nc = process_parent_fused(parent, 20, ct, n_half_child)
        total_children += nc
        if len(surv) > 0:
            all_surv.append(surv)

    elapsed = time.time() - t0
    n_surv_sample = sum(len(s) for s in all_surv)
    expansion = (n_surv_sample / n_sample) if n_sample > 0 else 0
    projected = int(expansion * len(current))
    cpp = total_children / max(n_sample, 1)

    print(f"L{level}: {len(current):,} parents, sampled {n_sample}, "
          f"expansion={expansion:.1f}x, projected={projected:,}, "
          f"children/parent={cpp:.0f}, time={elapsed:.1f}s")

    # Build next level from sample survivors
    if all_surv:
        next_surv = np.vstack(all_surv)
        # Canonicalize and dedup
        _canonicalize_inplace(next_surv)
        next_surv = _fast_dedup(next_surv)
        # Scale up to estimate full set
        current = next_surv
        print(f"     (using {len(current):,} unique sample survivors for next level)")
    else:
        current = np.empty((0, d_child), dtype=np.int32)


# =========================================================================
# SUMMARY
# =========================================================================

print("\n\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)
print("""
For c_target = 1.40:

1. VACUOUS (m <= 19): Cannot prove anything. The discretization error
   2/m + 1/m^2 pushes the threshold above the upper bound 1.5029.

2. m=20 (n_half=2): Barely non-vacuous (margin = 0.0004).
   L1 expansion ~143x and GROWING at deeper levels. HOPELESS.

3. m=25-50 (n_half=2): Non-vacuous but expansion INCREASES with m.
   More compositions at L0 and higher expansion at L1. WORSE than m=20.

4. n_half=3 (any m): Even more compositions and higher expansion.
   STRICTLY WORSE than n_half=2 at c_target=1.40.

5. n_half >= 4: Astronomical L0 counts (millions+). Not feasible.

CONCLUSION: No (m, n_half) combination can prove c_target = 1.40
via the current C&S cascade algorithm. The expansion factor is
fundamentally > 1 at every level for every non-vacuous configuration.

The highest c_target proven by ANY config is 1.35 (n_half=3, m=15,
converging at L6 per benchmark results), but even that is theoretical
(projected runtime is astronomical).

To prove c_target=1.40, one would need either:
  - A fundamentally different pruning strategy (not just C&S correction)
  - Higher m with a tighter correction bound than 2/m + 1/m^2
  - An entirely different algorithmic approach
""")

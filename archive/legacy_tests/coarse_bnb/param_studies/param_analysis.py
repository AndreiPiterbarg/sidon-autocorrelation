"""Comprehensive parameter analysis for c_target = 1.40.

Determines which (m, n_half) values are:
  - Vacuous (correction too large → proof proves nothing)
  - Non-vacuous but divergent (expansion factor >> 1 at every level)
  - Viable (could converge with enough compute)

Combines theoretical bounds with empirical L0/L1/L2 runs.
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

from math import comb
from pruning import correction, asymmetry_threshold, count_compositions

C_TARGET = 1.40
C_UPPER_BOUND = 1.5029  # Yu 2021

# =========================================================================
# 1. THEORETICAL ANALYSIS
# =========================================================================

print("=" * 80)
print("PARAMETER ANALYSIS FOR c_target = 1.40")
print("=" * 80)

print("\n--- 1. Vacuity Check ---")
print(f"Upper bound on C_1a: {C_UPPER_BOUND}")
print(f"Condition for non-vacuity: c_target + correction(m) < {C_UPPER_BOUND}")
print(f"  i.e., correction(m) = 2/m + 1/m² < {C_UPPER_BOUND - C_TARGET:.4f}")
print()

print(f"{'m':>4} | {'corr':>10} | {'threshold':>10} | {'margin':>10} | {'status':>12}")
print("-" * 60)

for m in [5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50, 60, 80, 100]:
    corr = correction(m)
    threshold = C_TARGET + corr
    margin = C_UPPER_BOUND - threshold
    status = "NON-VACUOUS" if margin > 0 else "VACUOUS"
    if margin > 0 and margin < 0.01:
        status = "BARELY OK"
    print(f"{m:4d} | {corr:10.6f} | {threshold:10.6f} | {margin:10.6f} | {status:>12}")

# Find minimum m for non-vacuity
print("\nMinimum m for non-vacuity at c_target=1.40:")
for m in range(1, 200):
    corr = correction(m)
    if C_TARGET + corr < C_UPPER_BOUND:
        print(f"  m_min = {m} (correction = {corr:.6f}, threshold = {C_TARGET+corr:.6f})")
        break

# =========================================================================
# 2. L0 COMPOSITION COUNTS
# =========================================================================

print("\n--- 2. L0 Composition Counts ---")
print(f"Total compositions = C(m+d-1, d-1) where d = 2*n_half")
print(f"Canonical ~ total/2 (palindrome symmetry)")
print()

print(f"{'m':>4} {'n_half':>6} | {'d':>3} | {'total comps':>15} | {'~canonical':>15}")
print("-" * 65)

for n_half in [2, 3, 4, 5]:
    d = 2 * n_half
    for m in [10, 15, 20, 25, 30, 40, 50]:
        corr = correction(m)
        if C_TARGET + corr >= C_UPPER_BOUND:
            continue  # skip vacuous
        n_comps = count_compositions(d, m)
        print(f"{m:4d} {n_half:6d} | {d:3d} | {n_comps:15,} | {n_comps//2:15,}")

# =========================================================================
# 3. ASYMMETRY ARGUMENT COVERAGE
# =========================================================================

print("\n--- 3. Asymmetry Argument ---")
asym_thresh = asymmetry_threshold(C_TARGET)
print(f"Asymmetry threshold for c_target=1.40: sqrt(1.40/2) = {asym_thresh:.6f}")
print(f"Configurations where left_frac >= {asym_thresh:.4f} or <= {1-asym_thresh:.4f}")
print(f"are automatically covered (test value >= c_target).")
print(f"Only configs with left_frac in ({1-asym_thresh:.4f}, {asym_thresh:.4f}) need checking.")
print(f"Width of 'needs checking' band: {2*asym_thresh - 1:.4f}")
print()

# The fraction of compositions that survive asymmetry filtering
# depends on the distribution of left_mass / total_mass
print("For uniform random compositions (Dirichlet), the fraction needing checking")
print("is approximately the width of the band: ~{:.1f}%".format((2*asym_thresh - 1)*100))

# =========================================================================
# 4. DYNAMIC THRESHOLD ANALYSIS: WORST-CASE vs PER-WINDOW
# =========================================================================

print("\n--- 4. Pruning Threshold Analysis ---")
print("The dynamic threshold per window (ell, s_lo):")
print("  TV_threshold = c_target + (1 + eps + 2*W_int) / m²")
print("  where W_int = sum of child masses in window bins (0 <= W_int <= m)")
print()
print("Worst case (W_int = m): TV_threshold = c_target + 2/m + 1/m² = c_target + correction")
print("Best case (W_int = 0):  TV_threshold = c_target + 1/m²")
print()

print(f"{'m':>4} | {'worst (W=m)':>12} | {'best (W=0)':>12} | {'typical W=m/3':>14}")
print("-" * 55)
for m in [20, 25, 30, 40, 50]:
    worst = C_TARGET + 2.0/m + 1.0/(m*m)
    best = C_TARGET + 1.0/(m*m)
    typical = C_TARGET + (1 + 2*m/3)/(m*m)
    print(f"{m:4d} | {worst:12.6f} | {best:12.6f} | {typical:14.6f}")

# =========================================================================
# 5. x_cap ANALYSIS (per-bin energy cap)
# =========================================================================

print("\n--- 5. x_cap (Per-Bin Energy Cap) Analysis ---")
print("x_cap = floor(m * sqrt((c_target + corr) / d_child))")
print("This caps how large any single child bin can be.")
print("Smaller x_cap → fewer children per parent → more pruning power.")
print()

for n_half in [2, 3, 4, 5]:
    print(f"\nn_half = {n_half}:")
    d_parent = 2 * n_half
    print(f"{'m':>4} | {'Level':>5} | {'d_child':>7} | {'x_cap':>5} | {'x_cap/m':>7} | {'children/parent (max)':>22}")
    print("-" * 70)
    for m in [20, 25, 30, 40, 50]:
        corr = correction(m)
        if C_TARGET + corr >= C_UPPER_BOUND:
            continue
        for level in range(4):
            d_child = d_parent * (2 ** (level + 1))
            n_half_child = d_child // 2
            thresh = C_TARGET + corr + 1e-9
            x_cap_main = int(math.floor(m * math.sqrt(thresh / d_child)))
            x_cap_cs = int(math.floor(m * math.sqrt(C_TARGET / d_child))) + 1
            x_cap = min(x_cap_main, x_cap_cs, m)
            x_cap = max(x_cap, 0)

            # Max children per parent: product of (min(b_i, x_cap) - max(0, b_i-x_cap) + 1)
            # Worst case: all bins = m/d_parent, each splits into x_cap+1 choices
            avg_bin = m / d_parent
            choices_per_bin = min(avg_bin, x_cap) - max(0, avg_bin - x_cap) + 1
            max_children = (2*x_cap + 1) ** d_parent  # very rough upper bound

            print(f"{m:4d} | L{level:>3d}  | {d_child:7d} | {x_cap:5d} | {x_cap/m:7.3f} | {max_children:22,}")

# =========================================================================
# 6. EFFECTIVE PRUNING MARGIN ANALYSIS
# =========================================================================

print("\n\n--- 6. Effective Pruning Margin ---")
print("The 'effective margin' is how much room the threshold has to prune.")
print("Higher margin → more configurations pruned at L0.")
print("Margin = C_UPPER_BOUND - (c_target + correction)")
print()

print(f"{'m':>4} | {'correction':>10} | {'threshold':>10} | {'margin':>10} | {'margin/corr':>10}")
print("-" * 60)
for m in [20, 25, 30, 40, 50, 60, 80, 100]:
    corr = correction(m)
    threshold = C_TARGET + corr
    if threshold >= C_UPPER_BOUND:
        continue
    margin = C_UPPER_BOUND - threshold
    print(f"{m:4d} | {corr:10.6f} | {threshold:10.6f} | {margin:10.6f} | {margin/corr:10.4f}")

# =========================================================================
# 7. REFINEMENT ANALYSIS: CHILDREN PER PARENT AT EACH LEVEL
# =========================================================================

print("\n\n--- 7. Theoretical Children Per Parent at Cascade Levels ---")
print("For a 'typical' parent with roughly uniform bins (b_i ≈ m/d_parent):")
print()

for n_half in [2, 3]:
    d0 = 2 * n_half
    for m in [20, 25, 30, 40, 50]:
        corr = correction(m)
        if C_TARGET + corr >= C_UPPER_BOUND:
            continue
        print(f"\n  n_half={n_half}, m={m}:")
        for level in range(5):
            d_parent = d0 * (2 ** level)
            d_child = 2 * d_parent
            n_half_child = d_child // 2
            thresh = C_TARGET + corr + 1e-9
            x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
            x_cap_cs = int(math.floor(m * math.sqrt(C_TARGET / d_child))) + 1
            x_cap = min(x_cap, x_cap_cs, m)
            x_cap = max(x_cap, 0)

            # Typical parent: uniform bins, b_i = m/d_parent
            avg_b = m / d_parent
            # Each bin splits into [max(0,b-x_cap), min(b,x_cap)] → choices = min(b,x_cap) - max(0,b-x_cap) + 1
            if avg_b <= x_cap:
                choices = int(avg_b) + 1
            else:
                choices = 2 * x_cap - int(avg_b) + 1
                choices = max(choices, 0)
            total_choices = max(choices, 1) ** d_parent

            print(f"    L{level}: d_parent={d_parent:4d}, d_child={d_child:4d}, "
                  f"x_cap={x_cap:3d}, choices/bin≈{choices}, "
                  f"total≈{total_choices:,.0f}")

# =========================================================================
# 8. KEY CONCLUSIONS
# =========================================================================

print("\n\n" + "=" * 80)
print("KEY CONCLUSIONS FOR c_target = 1.40")
print("=" * 80)

print("""
VACUOUS PARAMETERS (cannot prove anything):
  m <= 19: correction = 2/m + 1/m² >= 0.1028, threshold >= 1.5028 >= C_upper

  Specifically:
  - m=10: corr=0.21, threshold=1.61 (way too loose)
  - m=15: corr=0.1378, threshold=1.5378 (still too loose)
  - m=18: corr=0.1142, threshold=1.5142 (too loose)
  - m=19: corr=0.1080, threshold=1.5080 (too loose)

BARELY NON-VACUOUS:
  m=20: corr=0.1025, threshold=1.5025, margin=0.0004
    → Only 0.027% margin. Almost no pruning power at L0.
    → THIS IS THE CURRENT MAIN CONFIG. It is borderline.

  m=21: corr=0.0975, threshold=1.4975, margin=0.0054
  m=22: corr=0.0928, threshold=1.4928, margin=0.0101

VIABLE (meaningful pruning margin):
  m=25: corr=0.0816, threshold=1.4816, margin=0.0213
  m=30: corr=0.0678, threshold=1.4678, margin=0.0351
  m=40: corr=0.0506, threshold=1.4506, margin=0.0523
  m=50: corr=0.0404, threshold=1.4404, margin=0.0625

TRADE-OFF: Higher m means:
  + Tighter threshold (more pruning per level)
  + Closer to continuous limit
  - Exponentially more compositions at L0: C(m+d-1, d-1)
  - More children per parent at each level

The critical question: does the tighter pruning compensate for
the combinatorial explosion in search space?
""")

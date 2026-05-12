"""
Complete analysis: How does the known upper bound constrain the best
provable lower bound?

THE KEY RELATIONSHIP:
  Pruning threshold = c_target + (3 + 2*W_int) / m^2
  Non-vacuous requires: threshold < C_upper  (for at least some windows)

  => c_target < C_upper - (3 + 2*W_int) / m^2

  Best case (W_int=0):   c_target_max = C_upper - 3/m^2
  Worst case (W_int=m):   c_target_max = C_upper - 2/m - 3/m^2

Three levels of constraint:
  1. FULLY NON-VACUOUS: ALL windows can prune
     => c_target < C_upper - (3 + 2*m)/m^2 = C_upper - 2/m - 3/m^2
  2. PARTIALLY NON-VACUOUS: at least the best window (W_int=0) can prune
     => c_target < C_upper - 3/m^2
  3. CASCADE CONVERGENCE: expansion factor < 1 (empirical, stricter)
     => c_target << C_upper (depends on m, n_half, compute budget)
"""
import math

print("=" * 90)
print("UPPER BOUND vs BEST PROVABLE LOWER BOUND: COMPLETE ANALYSIS")
print("=" * 90)

# =========================================================================
# 1. THE FORMULA
# =========================================================================
print("""
THE FUNDAMENTAL CONSTRAINT
===========================

The pruning rule is:
  prune if TV_discrete > c_target + (3 + 2*W_int) / m^2

where W_int = integer mass in the window's bins (0 <= W_int <= m).

For pruning to WORK (be non-vacuous), the threshold must be BELOW C_upper:
  c_target + (3 + 2*W_int) / m^2  <  C_upper

Solving for c_target:
  c_target  <  C_upper - (3 + 2*W_int) / m^2

This gives us two bounds:

  THEORETICAL MAX (best window, W_int=0):
    c_target_max  =  C_upper - 3/m^2

  PRACTICAL MAX (all windows work, W_int=m):
    c_target_practical  =  C_upper - 2/m - 3/m^2

NOTE: These are NECESSARY conditions, not sufficient. The cascade must
also CONVERGE (expansion factor < 1), which is a stricter empirical
constraint. But vacuity is the hard mathematical wall.
""")

# =========================================================================
# 2. TABLE: For current upper bound (1.5029)
# =========================================================================
C_UPPER_CURRENT = 1.5029

print(f"CURRENT UPPER BOUND: C_upper = {C_UPPER_CURRENT}")
print("-" * 90)
print(f"{'m':>4} | {'correction':>10} | {'c_max (all windows)':>20} | {'c_max (best window)':>20} | {'min m for c=1.28':>16}")
print("-" * 90)

for m in [10, 15, 20, 25, 30, 40, 50, 75, 100, 200]:
    corr_worst = 2.0/m + 3.0/(m*m)       # all windows non-vacuous
    corr_best = 3.0/(m*m)                  # at least best window works
    c_max_all = C_UPPER_CURRENT - corr_worst
    c_max_best = C_UPPER_CURRENT - corr_best
    print(f"{m:4d} | {corr_worst:10.6f} | {c_max_all:20.4f} | {c_max_best:20.4f} |")

# =========================================================================
# 3. THE INTERESTING TABLE: Given various C_upper values
# =========================================================================
print()
print()
print("=" * 90)
print("IF THE UPPER BOUND CHANGES: What can we prove?")
print("=" * 90)
print()
print("For each hypothetical C_upper, showing c_target_max (all windows non-vacuous)")
print("Formula: c_target_max = C_upper - 2/m - 3/m^2")
print()

upper_bounds = [1.50, 1.45, 1.42, 1.40, 1.38, 1.36, 1.34, 1.32, 1.30, 1.29, 1.2802]
m_values = [15, 20, 25, 30, 40, 50, 75, 100, 200, 500]

# Header
header = f"{'C_upper':>8} |"
for m in m_values:
    header += f" {'m='+str(m):>7} |"
print(header)
print("-" * len(header))

for cu in upper_bounds:
    row = f"{cu:8.4f} |"
    for m in m_values:
        c_max = cu - 2.0/m - 3.0/(m*m)
        if c_max < 1.0:
            row += f" {'<1.00':>7} |"
        elif c_max < 1.2802:
            row += f" {c_max:7.4f}*|"  # below current best known lower bound
        else:
            row += f" {c_max:7.4f} |"

    print(row)

print()
print("* = below current best known lower bound (1.2802), so no improvement possible")

# =========================================================================
# 4. MINIMUM m REQUIRED for each (C_upper, c_target) pair
# =========================================================================
print()
print()
print("=" * 90)
print("MINIMUM m REQUIRED (all windows non-vacuous)")
print("=" * 90)
print()
print("Given C_upper and desired c_target, minimum m such that:")
print("  c_target < C_upper - 2/m - 3/m^2")
print("  => 2/m + 3/m^2 < C_upper - c_target  (= gap)")
print("  => m > 2/gap  (approximate, ignoring 3/m^2 term)")
print()

c_targets = [1.28, 1.30, 1.33, 1.35, 1.37, 1.40, 1.42, 1.45]

header = f"{'C_upper':>8} |"
for ct in c_targets:
    header += f" {'c='+str(ct):>7} |"
print(header)
print("-" * len(header))

for cu in upper_bounds:
    row = f"{cu:8.4f} |"
    for ct in c_targets:
        gap = cu - ct
        if gap <= 0:
            row += f" {'IMPOSS':>7} |"
        else:
            # Find minimum m: 2/m + 3/m^2 < gap
            found = False
            for m in range(2, 10001):
                if 2.0/m + 3.0/(m*m) < gap:
                    row += f" {m:>7} |"
                    found = True
                    break
            if not found:
                row += f" {'>10000':>7} |"
    print(row)

print()
print("IMPOSS = c_target >= C_upper, so proof is mathematically impossible")
print("(there exists a function achieving ||f*f||_inf = C_upper < c_target)")

# =========================================================================
# 5. THE COMPLETE PICTURE
# =========================================================================
print()
print()
print("=" * 90)
print("THE COMPLETE PICTURE")
print("=" * 90)
print("""
THREE NESTED CONSTRAINTS on what c_target we can prove:

1. MATHEMATICAL IMPOSSIBILITY (hardest wall):
   c_target > C_1a (the true constant)
   => there EXISTS a function with ||f*f||_inf < c_target
   => that function's discretization will NEVER be pruned
   => proof CANNOT succeed at ANY m

2. VACUITY WALL (second wall, depends on C_upper):
   c_target > C_upper - 2/m - 3/m^2
   => the pruning threshold exceeds C_upper for some windows
   => near-optimal configs (TV ~ C_upper) escape pruning
   => increasing m pushes this wall back (correction shrinks)
   => BUT: more m = more compositions = slower cascade

3. CONVERGENCE WALL (practical wall, empirical):
   Even when non-vacuous, the cascade may DIVERGE
   (expansion factor > 1 at every level)
   => this is the ACTUAL limiting factor in practice
   => from benchmarks: convergence only seen at c_target <= 1.35

RELATIONSHIP:
   c_target_convergence  <  c_target_nonvacuous  <  C_1a  <=  C_upper

   |<--provable-->|<--non-vacuous but diverges-->|<--vacuous-->|<--impossible-->|
   1.28          ~1.35                        ~1.40-1.50      C_1a         C_upper

KEY INSIGHT: A new tighter upper bound squeezes the picture FROM THE RIGHT:
   - Walls 2 and 3 shift LEFT (less room for the correction term)
   - Wall 1 shifts LEFT (true constant might be lower)
   - Previously proven bounds remain valid (they're to the LEFT)
   - But the FRONTIER for new proofs gets harder

EXAMPLE SCENARIOS:
""")

scenarios = [
    (1.5029, "Current knowledge"),
    (1.45,   "Moderate improvement"),
    (1.40,   "Major improvement"),
    (1.35,   "Near current best lower bound"),
    (1.30,   "Very close to 1.2802"),
]

for cu, label in scenarios:
    print(f"  C_upper = {cu:.4f} ({label}):")

    # Best provable with m=20 (all windows)
    c_m20 = cu - 2.0/20 - 3.0/400
    # Best provable with m=50
    c_m50 = cu - 2.0/50 - 3.0/2500
    # Best provable with m=100
    c_m100 = cu - 2.0/100 - 3.0/10000

    # Minimum m to prove at least 1.2802
    gap_1280 = cu - 1.2802
    if gap_1280 > 0:
        m_min = None
        for m in range(2, 10001):
            if 2.0/m + 3.0/(m*m) < gap_1280:
                m_min = m
                break
        m_str = str(m_min) if m_min else ">10000"
    else:
        m_str = "IMPOSSIBLE"

    print(f"    m=20:  max c_target = {c_m20:.4f}  (non-vacuous)")
    print(f"    m=50:  max c_target = {c_m50:.4f}  (non-vacuous)")
    print(f"    m=100: max c_target = {c_m100:.4f}  (non-vacuous)")
    print(f"    Min m to beat 1.2802: {m_str}")
    print()

# =========================================================================
# 6. DOES THE UPPER BOUND AFFECT PROVEN RESULTS?
# =========================================================================
print("=" * 90)
print("DOES A NEW UPPER BOUND INVALIDATE EXISTING PROOFS?")
print("=" * 90)
print("""
NO — with one caveat.

A completed proof (0 survivors) says:
  "Every discretized mass distribution at resolution m has
   max_window TV > c_target + correction"

This is a FACT about the enumeration. It does not reference C_upper.
The correction term comes from Lemma 3 (a mathematical theorem about
discretization error), not from the upper bound.

THE CAVEAT: If someone proves C_upper < c_target (i.e., the true
constant is BELOW what we claimed to prove), then we have a
CONTRADICTION. Either:
  (a) Our proof has a bug (wrong correction, missed survivors, etc.)
  (b) Their upper bound construction has a bug
  (c) Both

This cannot happen if both proofs are correct, because:
  c_target <= C_1a <= C_upper  (by definition)

So a valid lower bound proof and a valid upper bound proof can
never contradict each other. They just narrow the interval
[lower, upper] containing C_1a.
""")

"""
Critical vacuity table: For each (C_upper, c_target) pair, what is the
minimum m required for ALL windows to be non-vacuous?

Non-vacuous requires: c_target < C_upper - 2/m - 3/m^2
Solving: 2/m + 3/m^2 < C_upper - c_target  (= gap)
"""
import math

# Dense grid of upper bounds from 1.5029 down to 1.29
upper_bounds = [
    1.5029, 1.50, 1.49, 1.48, 1.47, 1.46,
    1.45, 1.44, 1.43, 1.42, 1.41, 1.40,
    1.39, 1.38, 1.37, 1.36, 1.35,
]

# Dense grid of c_targets from 1.28 up to 1.50
c_targets = [
    1.28, 1.29, 1.30, 1.31, 1.32, 1.33, 1.34, 1.35,
    1.36, 1.37, 1.38, 1.39, 1.40, 1.41, 1.42, 1.43,
    1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.50,
]


def min_m_for_nonvacuous(gap):
    """Find minimum m such that 2/m + 3/m^2 < gap."""
    if gap <= 0:
        return None  # impossible
    for m in range(2, 100001):
        if 2.0 / m + 3.0 / (m * m) < gap:
            return m
    return None  # need m > 100000


print("=" * 200)
print("CRITICAL VACUITY TABLE: Minimum m for non-vacuous pruning (all windows)")
print("Formula: 2/m + 3/m^2 < C_upper - c_target")
print("=" * 200)
print()
print("  '.' = impossible (c_target >= C_upper)")
print("  Numbers = minimum m required")
print()

# Header row
header = f"{'C_upper':>8} |"
for ct in c_targets:
    header += f" {ct:5.2f} |"
print(header)
print("-" * len(header))

for cu in upper_bounds:
    row = f"{cu:8.4f} |"
    for ct in c_targets:
        gap = cu - ct
        if gap <= 0:
            row += f"    . |"
        else:
            m_min = min_m_for_nonvacuous(gap)
            if m_min is None:
                row += f"  >1K |"
            elif m_min > 999:
                row += f"  >1K |"
            else:
                row += f" {m_min:4d} |"
    print(row)

print()
print()
print("=" * 200)
print("INTERPRETATION GUIDE")
print("=" * 200)
print("""
HOW TO READ THIS TABLE:

  Pick a row (hypothetical C_upper) and a column (desired c_target).
  The cell tells you the MINIMUM m needed for the pruning to be non-vacuous.

  REMEMBER: larger m means MORE compositions and MORE survivors per level.
  Empirically:
    m=15: ~3K-11K L0 compositions (n_half=3), cascade converges for c<=1.35
    m=20: ~467-11K L0 compositions, cascade DIVERGES for all tested c_targets
    m=30: ~5K L0 compositions, untested but expected worse
    m=50+: tens of thousands of L0 compositions, likely hopeless for convergence

  So even though m=50 is "non-vacuous", the cascade almost certainly diverges.
  Non-vacuity is NECESSARY but far from SUFFICIENT.

KEY OBSERVATIONS:

  1. Along each row, m increases left-to-right (higher c_target = tighter margin)
  2. Along each column, m increases bottom-to-top (lower C_upper = tighter margin)
  3. The diagonal where c_target ~ C_upper requires m -> infinity (impossible)
  4. Cells with m <= 20 are the "easy" zone — feasible with current code
  5. Cells with m = 20-50 are the "hard" zone — maybe feasible with GPU
  6. Cells with m > 50 are likely impractical for cascade convergence
""")

# Also print a "best provable c_target" table for each (C_upper, m) pair
print()
print("=" * 200)
print("BEST NON-VACUOUS c_target for each (C_upper, m) pair")
print("Formula: c_target_max = C_upper - 2/m - 3/m^2")
print("=" * 200)
print()

m_values = [10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50, 60, 75, 100, 150, 200]

header = f"{'C_upper':>8} |"
for m in m_values:
    header += f"  m={m:<4}|"
print(header)
print("-" * len(header))

for cu in upper_bounds:
    row = f"{cu:8.4f} |"
    for m in m_values:
        c_max = cu - 2.0 / m - 3.0 / (m * m)
        if c_max < 1.20:
            row += f"  <1.2 |"
        elif c_max < 1.2802:
            row += f" {c_max:.3f}*|"
        else:
            row += f" {c_max:.3f} |"
    print(row)

print()
print("* = below current best known lower bound (1.2802)")
print()
print("EXAMPLE READINGS:")
print("  Row C_upper=1.45, column m=20: best non-vacuous c_target = 1.342")
print("  Row C_upper=1.40, column m=30: best non-vacuous c_target = 1.330")
print("  Row C_upper=1.35, column m=50: best non-vacuous c_target = 1.309")

"""Mathematically precise vacuity analysis for c_target = 1.40.

NOTE: This analysis was written under the old coarse-grid (S=m) parameterization
and references "Formula A" (with 4n/ell factor) and "Formula B" (MATLAB/flat).
The project has since switched to the C&S fine grid (S = 4nm), where the
threshold formula is:
  threshold = floor((c_target*m^2 + 3 + W_int/(2n) + eps) * 4n*ell)
The vacuity conclusions and correction-function comparisons below may need
re-evaluation under the fine-grid parameterization.

Goes back to first principles. Verifies the ACTUAL formulas in the code
against what the proofs claim. Trusts nothing.

Key findings to verify:
1. What formula does the code ACTUALLY implement?
2. Is that formula proven sound?
3. What is the exact vacuity condition?
4. The correction() function in pruning.py -- does it match the pruning code?
"""
import math
import os
import sys
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction

C_TARGET = 1.40
C_UPPER = 1.5029  # Matolcsi & Vinuesa (2010)

print("=" * 80)
print("PRECISE VACUITY ANALYSIS FOR c_target = 1.40")
print("=" * 80)

# =========================================================================
# FACT 1: What formula does the code implement?
# =========================================================================
print("""
FACT 1: THE CODE'S ACTUAL PRUNING FORMULA
=========================================

From run_cascade.py (all 5 pruning kernels):

  cs_corr_base = c_target * m^2 + 3.0 + eps_margin    [eps_margin = 1e-9 * m^2]
  ct_base_ell  = cs_corr_base * ell / (4*n_half)
  w_scale      = 2.0 * ell / (4*n_half)
  dyn_x        = ct_base_ell + w_scale * W_int

  Expanded:
  dyn_x = (c_target * m^2 + 3 + eps + 2*W_int) * ell/(4n)

  PRUNE if:  ws_int > floor(dyn_x * (1 - 4*DBL_EPS))

  where ws_int = sum of integer autoconvolution values in the window.

Converting to test-value space (TV = ws_int * 4n / (m^2 * ell)):

  TV_threshold = c_target + (3 + eps + 2*W_int) / m^2

  NOTE: the threshold is INDEPENDENT of ell. The ell/(4n) factor in the
  integer threshold cancels when converting to TV space.

  This is the MATLAB formula (Formula B) with +3 instead of +1.
""")

# =========================================================================
# FACT 2: What does correction() return vs what the code uses?
# =========================================================================
print("FACT 2: INCONSISTENCY IN correction() FUNCTION")
print("=" * 50)
print()

for m in [15, 20, 25, 30, 50]:
    corr_func = correction(m)
    # Code's actual worst-case (W_int = m):
    actual_worst = (3 + 2*m) / (m*m)
    # Code's actual without W: (W_int = 0):
    actual_best = 3.0 / (m*m)

    print(f"m={m}:")
    print(f"  correction() returns: {corr_func:.6f}  (= 2/m + 1/m^2)")
    print(f"  Code's worst case:    {actual_worst:.6f}  (= (3 + 2m)/m^2 = 2/m + 3/m^2)")
    print(f"  Code's best case:     {actual_best:.6f}  (= 3/m^2, when W_int=0)")
    print(f"  MISMATCH: correction() uses +1, code uses +3. Diff = {actual_worst - corr_func:.6f}")
    print()

# =========================================================================
# FACT 3: Vacuity under the code's ACTUAL threshold
# =========================================================================
print()
print("FACT 3: VACUITY UNDER THE CODE'S ACTUAL THRESHOLD")
print("=" * 50)
print()
print("Vacuity at the WIDEST window (W_int = m, maximum threshold):")
print("  threshold = c_target + (3 + 2m)/m^2")
print(f"  Must be < C_upper = {C_UPPER} for widest window to prune")
print()

print(f"{'m':>4} | {'(3+2m)/m^2':>10} | {'threshold':>10} | {'margin':>10} | status")
print("-" * 65)

for m in range(15, 61):
    worst_corr = (3 + 2*m) / (m*m)
    threshold = C_TARGET + worst_corr
    margin = C_UPPER - threshold
    if abs(margin) < 0.02 or m in [15, 20, 21, 22, 25, 30, 40, 50]:
        status = "VACUOUS" if margin <= 0 else "OK"
        if margin > 0 and margin < 0.005:
            status = "BARELY"
        print(f"{m:4d} | {worst_corr:10.6f} | {threshold:10.6f} | {margin:10.6f} | {status}")

print()
print("Minimum m where widest window is non-vacuous:")
for m in range(1, 200):
    worst_corr = (3 + 2*m) / (m*m)
    if C_TARGET + worst_corr < C_UPPER:
        print(f"  m_min = {m} (threshold = {C_TARGET + worst_corr:.6f})")
        break

# =========================================================================
# FACT 4: But narrower windows have lower W_int
# =========================================================================
print()
print()
print("FACT 4: PER-WINDOW VACUITY DEPENDS ON W_int, NOT ell")
print("=" * 50)
print("""
The TV-space threshold is:
  TV > c_target + (3 + 2*W_int) / m^2

This does NOT depend on ell!  Only on W_int (the mass in the window's bins).

A window is "vacuous" (cannot prune near-optimal configs with TV ~ 1.50) when:
  c_target + (3 + 2*W_int) / m^2 >= C_upper
  (3 + 2*W_int) / m^2 >= C_upper - c_target = 0.1029
  W_int >= (0.1029 * m^2 - 3) / 2
""")

print(f"{'m':>4} | {'W_int_max':>9} | {'W_vacuous':>10} | note")
print("-" * 50)
for m in [20, 21, 22, 25, 30, 40, 50]:
    w_vacuous_thresh = (0.1029 * m * m - 3) / 2
    w_max = m
    print(f"{m:4d} | {w_max:9d} | {w_vacuous_thresh:10.1f} | "
          f"windows with W_int > {int(w_vacuous_thresh)} are vacuous")

# =========================================================================
# FACT 5: Soundness status of the formula
# =========================================================================
print()
print()
print("FACT 5: SOUNDNESS STATUS")
print("=" * 50)
print("""
The code uses Formula B (MATLAB threshold): the ENTIRE correction scales with ell/(4n).

Soundness status per proof/discretization_error_proof.md:

  FORMULA A (Theorem 3.7, proven):
    TV > c_target + (4n/ell) * (1/m^2 + 2W/m)
    Integer: ws_int > c_target*m^2*ell/(4n) + 1 + 2*W_int
    Status: PROVEN SOUND. The 4n/ell factor is a valid (loose) upper bound.

  FORMULA B (MATLAB, code uses):
    TV > c_target + (3 + 2*W_int) / m^2    [the +3 is code's version; MATLAB uses +1]
    Integer: ws_int > (c_target*m^2 + 3 + 2*W_int) * ell/(4n)
    Status: UNPROVEN. Per-window counterexample exists (d=4, m=10, error exceeds
            Formula B bound by factor 3.1). Composition-level soundness is OPEN.

  KEY DISTINCTION: Formula A is PROVEN but makes narrow windows useless.
                   Formula B is UNPROVEN but makes narrow windows effective.
""")

# =========================================================================
# FACT 6: Under Formula A (proven), what's vacuous?
# =========================================================================
print()
print("FACT 6: VACUITY UNDER FORMULA A (PROVEN THRESHOLD)")
print("=" * 50)
print()
print("Formula A per-window threshold in TV space:")
print("  TV > c_target + (4n/ell) * (1/m^2 + 2W/m)")
print()
print("For this to prune configs with TV ~ C_upper = 1.5029:")
print("  (4n/ell) * (1/m^2 + 2*W_int/(m^2)) < C_upper - c_target = 0.1029")
print()

for m in [20, 25, 30, 50]:
    print(f"\nm={m}:")
    for n_half in [2, 3]:
        for level in [0, 1, 2, 3]:
            d = 2 * n_half * (2 ** level)
            n = d // 2  # n_half_at_this_level
            max_ell = 2 * d
            # Widest window: 4n/ell = 4n/(4n) = 1 → same as Formula B
            corr_widest = (1 + 2*m) / (m*m)
            # ell = d (half-width): 4n/ell = 4n/(2n) = 2
            corr_half = 2 * (1 + 2*m) / (m*m)
            # ell = 2 (narrowest): 4n/ell = 2n
            corr_narrow = 2*n * (1 + 2*m) / (m*m)

            ok_widest = C_TARGET + corr_widest < C_UPPER
            ok_half = C_TARGET + corr_half < C_UPPER
            ok_narrow = C_TARGET + corr_narrow < C_UPPER

            if level <= 1:
                print(f"  n_half_orig={n_half}, L{level}: d={d}, n={n}")
                print(f"    ell=2 (narrow): corr={corr_narrow:.4f}, "
                      f"threshold={C_TARGET+corr_narrow:.4f} {'OK' if ok_narrow else 'VACUOUS'}")
                print(f"    ell={d} (half):  corr={corr_half:.4f}, "
                      f"threshold={C_TARGET+corr_half:.4f} {'OK' if ok_half else 'VACUOUS'}")
                print(f"    ell={max_ell} (wide): corr={corr_widest:.4f}, "
                      f"threshold={C_TARGET+corr_widest:.4f} {'OK' if ok_widest else 'VACUOUS'}")


# =========================================================================
# SUMMARY
# =========================================================================
print()
print()
print("=" * 80)
print("MATHEMATICAL VERDICT")
print("=" * 80)
print(f"""
FOR c_target = 1.40, C_upper = {C_UPPER}:

1. CORRECTION FUNCTION IS WRONG
   correction() returns 2/m + 1/m^2, but the pruning code uses (3 + 2*W_int)/m^2.
   The worst case (W_int=m) is 2/m + 3/m^2, not 2/m + 1/m^2.

2. UNDER THE CODE'S FORMULA (Formula B with +3):
   - m <= 20: The WIDEST window is vacuous (threshold > {C_UPPER}).
     But this does NOT mean pruning fails -- narrower windows with W_int < m
     still prune. The TV threshold depends on W_int, not on ell.
   - m = 21: First value where widest window is non-vacuous (margin 0.00086).
   - For any m, windows with W_int > floor((0.1029*m^2 - 3)/2) are vacuous.

3. FORMULA B IS NOT PROVEN SOUND
   The code uses Formula B (MATLAB formula). This formula has a known
   counterexample at the per-window level (d=4, m=10, error 3.1x larger
   than Formula B predicts). Its composition-level soundness is OPEN.

   The original C&S paper also uses Formula B and proved C_1a >= 1.28.
   Their proof's rigor depends on Formula B being sound.

4. UNDER FORMULA A (PROVEN SOUND):
   - Only the widest window (ell = 4n) gives the same threshold as Formula B.
   - All narrower windows have correction multiplied by 4n/ell >> 1.
   - At L0 (d=4, n=2): ell=2 correction is 4x larger. Threshold = 1.81.
   - At L2 (d=16, n=8): ell=2 correction is 16x larger. Threshold = 3.04.
   - Formula A makes the cascade dramatically weaker.

5. WHICH (m, n_half) ARE VACUOUS?

   Under Formula B (code, unproven):
     m <= 19, any n_half:  FULLY vacuous (no window can prune anything)
     m = 20, any n_half:   Widest window vacuous, but cascade still works
     m >= 21, any n_half:  Non-vacuous at all windows
     n_half >= 3: REDUNDANT (strictly worse than n_half=2 at c_target=1.40)
     n_half >= 4: INFEASIBLE (too many L0 compositions)

   Under Formula A (proven):
     At every level, only the widest window contributes meaningfully.
     Convergence at c_target=1.40 is unknown -- the cascade is far weaker.
     The proven threshold may be too conservative for c_target=1.40 to work.

6. THE FUNDAMENTAL PROBLEM
   Proving c_target=1.40 requires Formula B to be sound. Formula B is
   unproven. The proven Formula A is too weak. This is not a parameter
   choice problem -- it is a PROOF GAP.
""")

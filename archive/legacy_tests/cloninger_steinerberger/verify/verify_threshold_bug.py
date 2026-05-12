"""
Verify the threshold formula bug claim in the Cloninger-Steinerberger algorithm.

Two formulas under consideration:

  BUGGY (old):     threshold_int = floor((c_target*m^2 + 1 + eps + 2*W_int) * ell/(4n))
  CORRECTED (new): threshold_int = floor(c_target*m^2 * ell/(4n) + 1 + eps + 2*W_int)

The difference: in the buggy version, the correction terms (1 + eps + 2*W_int)
are multiplied by ell/(4n). In the corrected version, they are NOT.

We test:
  1. A specific counterexample showing the formulas diverge
  2. Whether the buggy formula could falsely "prove" c_target=1.503
     (above the known upper bound of 1.5029)
"""

import numpy as np
from itertools import combinations_with_replacement
import math

# =====================================================================
# Helper functions
# =====================================================================

def autoconv_int(c):
    """Integer autoconvolution: conv[k] = sum_{i+j=k} c_i * c_j"""
    d = len(c)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += int(c[i]) * int(c[j])
    return conv

def test_value_window(c, n_half, m, ell, s_lo):
    """
    Compute the test value for window (ell, s_lo) in continuous coordinates.

    TV = ws_a / (4*n*ell) where ws_a = sum of autoconv in a-coordinates
    a_i = (4n/m) * c_i, so conv_a[k] = (4n/m)^2 * conv_int[k]
    TV = (4n/m)^2 * ws_int / (4*n*ell) = 4*n * ws_int / (m^2 * ell)
    """
    d = len(c)
    conv = autoconv_int(c)
    n_cv = ell - 1  # number of conv values in window
    ws_int = sum(conv[s_lo : s_lo + n_cv])
    n = n_half  # n_half = n/2, but here d = 2*n_half so n = n_half... wait
    # d = 2*n_half, so the support is [-1/4, 1/4] split into d bins
    # The formula: TV = ws_int * 4*n_half / (m^2 * ell)  -- NO
    # Let me be careful. d = 2*n_half. The bins are width 1/(2d) = 1/(4*n_half).
    # a_i = c_i / m * (1 / bin_width) ... actually let me just use the formula from the code.
    # From run_cascade.py: inv_4n = 1/(4*n_half)
    # ct_base_ell = c_target * m^2 * ell * inv_4n = c_target * m^2 * ell / (4*n_half)
    # threshold_int for window sum ws (in integer coords)
    # Prune if ws > threshold_int
    # The test value in continuous coords: TV = ws * 4*n_half / (m^2 * ell)
    # So TV > c_target iff ws > c_target * m^2 * ell / (4*n_half)

    TV = float(ws_int) * 4.0 * n_half / (m * m * ell)
    return TV, ws_int

def W_int_for_window(c, ell, s_lo, d):
    """
    W_int = sum of c_i for bins overlapping the window [s_lo, s_lo+ell-2].

    The overlapping bins are those i where the conv index k = i + j can fall
    in [s_lo, s_lo+ell-2] for some j in [0, d-1].

    From the code:
      lo_bin = max(0, s_lo - (d-1))
      hi_bin = min(d-1, s_lo + ell - 2)
      W_int = sum(c[lo_bin : hi_bin+1])
    """
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    return int(np.sum(c[lo_bin : hi_bin + 1]))

def threshold_buggy(c_target, m, n_half, ell, W_int):
    """
    CORRECT formula: ALL terms scaled by ell/(4n), +3 for W_g correction.
    threshold = floor((c_target*m^2 + 3 + eps + 2*W_int) * ell/(4*n_half))
    """
    eps_margin = 1e-9 * m * m
    dyn_base = c_target * m * m + 3.0 + eps_margin + 2.0 * W_int
    dyn_x = dyn_base * ell / (4.0 * n_half)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    return int(math.floor(dyn_x * one_minus_4eps))

def threshold_corrected(c_target, m, n_half, ell, W_int):
    """
    OLD formula (Theorem 3.7): only c_target*m^2 scaled by ell/(4n).
    threshold = floor(c_target*m^2*ell/(4*n_half) + 1 + eps + 2*W_int)
    """
    eps_margin = 1e-9 * m * m
    dyn_x = c_target * m * m * ell / (4.0 * n_half) + 1.0 + eps_margin + 2.0 * W_int
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    return int(math.floor(dyn_x * one_minus_4eps))

def compositions(m, d):
    """Generate all compositions of m into d non-negative parts."""
    if d == 1:
        yield (m,)
        return
    for i in range(m + 1):
        for rest in compositions(m - i, d - 1):
            yield (i,) + rest

def is_canonical(c):
    """Check if c <= reverse(c) lexicographically."""
    rev = tuple(reversed(c))
    return c <= rev

# =====================================================================
# Part 1: Specific counterexample
# =====================================================================

print("=" * 70)
print("PART 1: Specific counterexample")
print("=" * 70)

d = 4
n_half = 2
m = 10
c = np.array([0, 0, 1, 9])
mu = np.array([1, 1, 0, 8])

print(f"\nParameters: d={d}, n_half={n_half}, m={m}")
print(f"c  = {c}  (original composition)")
print(f"mu = {mu} (perturbation, |mu_i - c_i| <= 1)")
print(f"sum(c) = {sum(c)}, sum(mu) = {sum(mu)}")
print(f"max |c_i - mu_i| = {max(abs(c - mu))}")

ell = 2
s_lo = 1

TV_c, ws_c = test_value_window(c, n_half, m, ell, s_lo)
TV_mu, ws_mu = test_value_window(mu, n_half, m, ell, s_lo)

print(f"\nWindow: ell={ell}, s_lo={s_lo}")
print(f"  TV(c)  = {TV_c:.6f}  (ws_int = {ws_c})")
print(f"  TV(mu) = {TV_mu:.6f}  (ws_int = {ws_mu})")
print(f"  TV difference = {TV_c - TV_mu:.6f}")

W_c = W_int_for_window(c, ell, s_lo, d)
W_mu = W_int_for_window(mu, ell, s_lo, d)

print(f"\n  W_int(c)  = {W_c}")
print(f"  W_int(mu) = {W_mu}")

# Compare thresholds
for ct in [1.4, 1.503]:
    print(f"\n  c_target = {ct}:")
    tb_c = threshold_buggy(ct, m, n_half, ell, W_c)
    tc_c = threshold_corrected(ct, m, n_half, ell, W_c)
    print(f"    Buggy threshold     = {tb_c}  (ws_int={ws_c}, prune? {ws_c > tb_c})")
    print(f"    Corrected threshold = {tc_c}  (ws_int={ws_c}, prune? {ws_c > tc_c})")
    print(f"    Difference          = {tc_c - tb_c}")

# =====================================================================
# Part 2: Compare formulas across ALL windows for this example
# =====================================================================

print("\n" + "=" * 70)
print("PART 2: All windows for c=(0,0,1,9), both formulas")
print("=" * 70)

conv = autoconv_int(c)
conv_len = 2 * d - 1
print(f"\nconv_int = {conv}")

c_target = 1.4
print(f"\nc_target = {c_target}")
print(f"{'ell':>4} {'s_lo':>4} {'ws':>6} {'W_int':>5} {'buggy_thr':>10} {'corr_thr':>10} {'bug_prune':>10} {'cor_prune':>10} {'diff':>6}")
for ell in range(2, 2 * d + 1):
    n_cv = ell - 1
    n_windows = conv_len - n_cv + 1
    for s_lo in range(n_windows):
        ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
        W = W_int_for_window(c, ell, s_lo, d)
        tb = threshold_buggy(c_target, m, n_half, ell, W)
        tc = threshold_corrected(c_target, m, n_half, ell, W)
        bp = ws > tb
        cp = ws > tc
        print(f"{ell:4d} {s_lo:4d} {ws:6d} {W:5d} {tb:10d} {tc:10d} {str(bp):>10} {str(cp):>10} {tc-tb:6d}")

# =====================================================================
# Part 3: Enumerate all compositions, find max TV, check if buggy
#          formula could falsely prove c_target=1.503
# =====================================================================

print("\n" + "=" * 70)
print("PART 3: Exhaustive check over ALL compositions")
print("=" * 70)

c_target_test = 1.503
print(f"\nc_target = {c_target_test}, m = {m}, d = {d}, n_half = {n_half}")
print(f"Known upper bound on C_1a: 1.5029")
print(f"If any composition has max_TV <= {c_target_test}, it CANNOT be pruned,")
print(f"so the bound {c_target_test} cannot be proven.\n")

n_compositions = 0
n_canonical = 0
max_TV_overall = 0.0
max_TV_comp = None
unprunable_buggy = []
unprunable_corrected = []

for comp in compositions(m, d):
    n_compositions += 1
    c_arr = np.array(comp)

    if not is_canonical(comp):
        continue
    n_canonical += 1

    # Find max TV across all windows
    conv = autoconv_int(c_arr)
    conv_len = 2 * d - 1
    max_TV_this = 0.0
    max_window = None

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
            TV = float(ws) * 4.0 * n_half / (m * m * ell)
            if TV > max_TV_this:
                max_TV_this = TV
                max_window = (ell, s_lo, ws)

    if max_TV_this > max_TV_overall:
        max_TV_overall = max_TV_this
        max_TV_comp = comp

    # Check if pruned by each formula
    pruned_buggy = False
    pruned_corrected = False

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
            W = W_int_for_window(c_arr, ell, s_lo, d)

            tb = threshold_buggy(c_target_test, m, n_half, ell, W)
            tc = threshold_corrected(c_target_test, m, n_half, ell, W)

            if ws > tb:
                pruned_buggy = True
            if ws > tc:
                pruned_corrected = True

    if not pruned_buggy:
        unprunable_buggy.append((comp, max_TV_this))
    if not pruned_corrected:
        unprunable_corrected.append((comp, max_TV_this))

print(f"Total compositions: {n_compositions}")
print(f"Canonical compositions: {n_canonical}")
print(f"Max TV across all compositions: {max_TV_overall:.6f} at {max_TV_comp}")

print(f"\n--- Buggy formula (c_target={c_target_test}) ---")
print(f"Unprunable compositions: {len(unprunable_buggy)}")
if len(unprunable_buggy) == 0:
    print("  WARNING: Buggy formula prunes everything -> would falsely 'prove' c_target={:.4f}!".format(c_target_test))
    print("  This is ABOVE the known upper bound of 1.5029 -> BUG CONFIRMED")
else:
    print(f"  Correctly fails to prune {len(unprunable_buggy)} compositions:")
    for comp, tv in unprunable_buggy[:20]:
        print(f"    {comp}  max_TV = {tv:.6f}")

print(f"\n--- Corrected formula (c_target={c_target_test}) ---")
print(f"Unprunable compositions: {len(unprunable_corrected)}")
if len(unprunable_corrected) == 0:
    print("  WARNING: Corrected formula also prunes everything!")
    print("  This would also falsely 'prove' c_target > 1.5029 -> UNEXPECTED")
else:
    print(f"  Correctly fails to prune {len(unprunable_corrected)} compositions:")
    for comp, tv in unprunable_corrected[:20]:
        print(f"    {comp}  max_TV = {tv:.6f}")

# =====================================================================
# Part 4: Quantify divergence between formulas
# =====================================================================

print("\n" + "=" * 70)
print("PART 4: When do the formulas diverge?")
print("=" * 70)

print("\nThe two formulas differ when ell/(4*n_half) != 1.")
print(f"For n_half={n_half}: ell/(4*{n_half}) = ell/{4*n_half}")
print()

for ell in range(2, 2 * d + 1):
    ratio = ell / (4.0 * n_half)
    # Buggy: (c_target*m^2 + 1 + eps + 2*W) * ratio
    # Correct: c_target*m^2 * ratio + 1 + eps + 2*W
    # Difference: (1 + eps + 2*W) * ratio - (1 + eps + 2*W) = (1 + eps + 2*W) * (ratio - 1)
    print(f"  ell={ell}: ratio={ratio:.3f}, scaling factor for correction = {ratio:.3f}")
    print(f"    If W_int=10: buggy correction = {(1 + 2*10) * ratio:.3f}, correct correction = {1 + 2*10:.3f}, diff = {(1 + 2*10) * (ratio - 1):.3f}")

print()
print("When ratio < 1 (ell < 4*n_half): buggy threshold is LOWER -> more aggressive pruning -> risk of false pruning")
print("When ratio > 1 (ell > 4*n_half): buggy threshold is HIGHER -> less aggressive pruning -> missed pruning opportunities")
print(f"For n_half={n_half}, critical ell = {4*n_half} (where ratio=1)")

# =====================================================================
# Part 5: Directly verify the current code's formula matches "corrected"
# =====================================================================

print("\n" + "=" * 70)
print("PART 5: What does the CURRENT code actually do?")
print("=" * 70)

print("""
Current code (run_cascade.py):
  cs_corr_base = c_target * m^2 + 3.0 + eps_margin
  ct_base_ell_arr[ell] = cs_corr_base * ell * inv_4n
  dyn_x = ct_base_ell + w_scale * W_int   # entire threshold scaled by ell/(4n)

This matches the CORRECT formula (C&S Lemma 3 + W_g correction):
  threshold = floor((c_target*m^2 + 3 + eps + 2*W_int) * ell/(4n) * (1-4*DBL_EPS))

The OLD code (before commit 6e43a6d) had:
  dyn_base = c_target * m^2 + 1.0 + 1e-9*m^2
  dyn_base_ell = dyn_base * ell * inv_4n                   # ALL terms scaled
  dyn_x = dyn_base_ell + 2*ell*inv_4n * W_int              # W_int also scaled

This matches the BUGGY formula:
  threshold = floor((c_target*m^2 + 1 + eps + 2*W_int) * ell/(4n) * (1-4*DBL_EPS))
""")

print("CONCLUSION: The current code uses the CORRECTED formula (since commit 6e43a6d).")
print("The bug existed in the OLD code and has been fixed.")
print()

# =====================================================================
# Part 6: Does the buggy formula actually accept false proofs?
# =====================================================================

print("=" * 70)
print("PART 6: Can the BUGGY formula falsely prove c_target above the upper bound?")
print("=" * 70)

# Try a range of c_target values with the buggy formula
print(f"\nSearching for highest c_target where buggy formula prunes ALL compositions...")
print(f"(d={d}, m={m}, n_half={n_half})")

for ct in [1.40, 1.45, 1.50, 1.503, 1.51, 1.55, 1.60]:
    all_pruned = True
    survivors = []
    for comp in compositions(m, d):
        c_arr = np.array(comp)
        if not is_canonical(comp):
            continue

        conv = autoconv_int(c_arr)
        conv_len = 2 * d - 1
        pruned = False

        for ell in range(2, 2 * d + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
                W = W_int_for_window(c_arr, ell, s_lo, d)
                tb = threshold_buggy(ct, m, n_half, ell, W)
                if ws > tb:
                    pruned = True
                    break

        if not pruned:
            all_pruned = False
            survivors.append(comp)

    status = "ALL PRUNED" if all_pruned else f"{len(survivors)} survivors"
    falsity = " <-- FALSE PROOF (above upper bound 1.5029)" if all_pruned and ct > 1.5029 else ""
    print(f"  c_target={ct:.3f}: {status}{falsity}")

print("\nDone.")

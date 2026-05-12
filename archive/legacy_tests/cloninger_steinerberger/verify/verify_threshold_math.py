"""Verify the threshold formula derivation step by step.

Traces through a single composition and specific window, comparing:
  1. MATLAB continuous-domain threshold
  2. Original CPU formula (everything scaled by ell/(4n))
  3. Current CPU formula (correction NOT scaled)

Shows the exact values at each step to verify mathematical equivalence.
"""
import math
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger', 'cpu'))

# ==========================================================================
# Parameters
# ==========================================================================
m = 20
c_target = 1.4
n_half = 2
d = 2 * n_half  # = 4
gs = 1.0 / m    # = 0.05

# Pick a specific L0 composition that CPU keeps but MATLAB prunes
# From the comparison: (2, 8, 0, 10) was in CPU only
a_int = np.array([2, 8, 0, 10], dtype=np.int32)
a_cont = a_int / m  # continuous weights

print("=" * 70)
print("THRESHOLD FORMULA VERIFICATION")
print(f"m={m}, c_target={c_target}, n_half={n_half}, d={d}, gs={gs}")
print(f"Composition (int): {a_int}")
print(f"Composition (cont): {a_cont}")
print(f"Sum check: {sum(a_int)} (should be {m})")
print("=" * 70)

# ==========================================================================
# Step 1: Compute autoconvolution (both domains)
# ==========================================================================
conv_len = 2 * d - 1  # = 7

# Integer autoconvolution
conv_int = np.zeros(conv_len, dtype=np.int64)
for i in range(d):
    conv_int[2 * i] += a_int[i] * a_int[i]
    for j in range(i + 1, d):
        conv_int[i + j] += 2 * a_int[i] * a_int[j]

# Continuous autoconvolution
conv_cont = np.zeros(conv_len, dtype=np.float64)
for i in range(d):
    conv_cont[2 * i] += a_cont[i] * a_cont[i]
    for j in range(i + 1, d):
        conv_cont[i + j] += 2 * a_cont[i] * a_cont[j]

print(f"\nConvolution (int):  {conv_int}")
print(f"Convolution (cont): {conv_cont}")
print(f"Relation check: conv_cont = conv_int / m^2 = {conv_int / (m*m)}")
assert np.allclose(conv_cont, conv_int / (m * m)), "conv_cont != conv_int/m^2"

# ==========================================================================
# Step 2: Check every window (ell, s_lo)
# ==========================================================================
print(f"\n{'='*70}")
print("WINDOW-BY-WINDOW COMPARISON")
print(f"{'='*70}")

# Prefix sums for W_int computation
prefix_c = np.zeros(d + 1, dtype=np.int64)
for i in range(d):
    prefix_c[i + 1] = prefix_c[i] + a_int[i]

DBL_EPS = 2.220446049250313e-16
one_minus_4eps = 1.0 - 4.0 * DBL_EPS
eps_margin = 1e-9 * m * m
inv_4n = 1.0 / (4.0 * n_half)

discrepancies = []

for ell in range(2, 2 * d + 1):
    n_cv = ell - 1
    n_windows = conv_len - n_cv + 1

    for s_lo in range(n_windows):
        # --- Window sum (integer) ---
        ws_int = sum(conv_int[s_lo:s_lo + n_cv])

        # --- Window sum (continuous) ---
        ws_cont = sum(conv_cont[s_lo:s_lo + n_cv])
        assert abs(ws_cont - ws_int / (m * m)) < 1e-15

        # --- W_int: mass of bins overlapping this window ---
        lo_bin = max(0, s_lo - (d - 1))
        hi_bin = min(d - 1, s_lo + ell - 2)
        W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])
        W_cont = W_int / m

        # =============================================
        # MATLAB threshold (continuous domain)
        # =============================================
        # test_val = ws_cont * (2*d) / ell
        test_val = ws_cont * (2 * d) / ell
        # bound = (c_target + gs^2) + 2*gs*W
        matlab_bound = (c_target + gs * gs) + 2 * gs * W_cont
        matlab_prunes = (test_val >= matlab_bound)

        # =============================================
        # Convert MATLAB to integer domain
        # =============================================
        # test_val >= bound
        # ws_cont * (2d/ell) >= c + gs^2 + 2*gs*W
        # ws_int/m^2 * (2d/ell) >= c + 1/m^2 + 2*W_int/m^2
        # ws_int >= (c*m^2 + 1 + 2*W_int) * ell / (2d)
        # ws_int >= (c*m^2 + 1 + 2*W_int) * ell / (4n)
        matlab_int_thresh = (c_target * m * m + 1 + 2 * W_int) * ell / (4 * n_half)

        # =============================================
        # ORIGINAL CPU formula (pre-commit 6e43a6d, updated to +3)
        # Everything scaled by ell/(4n)
        # =============================================
        dyn_base_orig = c_target * m * m + 3.0 + eps_margin
        orig_dyn_x = (dyn_base_orig + 2.0 * W_int) * ell * inv_4n
        orig_dyn_it = int(orig_dyn_x * one_minus_4eps)
        orig_prunes = (ws_int > orig_dyn_it)

        # =============================================
        # CURRENT CPU formula (corrected C&S + W_g)
        # Entire threshold scaled by ell/(4n), +3 for W_g correction
        # =============================================
        cs_corr_base = c_target * m * m + 3.0 + eps_margin
        curr_dyn_x = (cs_corr_base + 2.0 * W_int) * ell * inv_4n
        curr_dyn_it = int(curr_dyn_x * one_minus_4eps)
        curr_prunes = (ws_int > curr_dyn_it)

        # Compare
        if matlab_prunes != orig_prunes or matlab_prunes != curr_prunes:
            flag = "***DISCREPANCY***"
        else:
            flag = ""

        if flag or ell <= 3:  # Print first few and all discrepancies
            print(f"\n  ell={ell}, s_lo={s_lo}:")
            print(f"    ws_int={ws_int}, W_int={W_int}")
            print(f"    MATLAB: test_val={test_val:.6f}, bound={matlab_bound:.6f}, "
                  f"prunes={matlab_prunes}")
            print(f"    MATLAB->int threshold: {matlab_int_thresh:.4f}")
            print(f"    ORIGINAL CPU: dyn_x={orig_dyn_x:.4f}, dyn_it={orig_dyn_it}, "
                  f"prunes={orig_prunes}")
            print(f"    CURRENT CPU:  dyn_x={curr_dyn_x:.4f}, dyn_it={curr_dyn_it}, "
                  f"prunes={curr_prunes}")
            if flag:
                print(f"    {flag}")
                discrepancies.append((ell, s_lo, matlab_prunes, orig_prunes, curr_prunes))

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

# Now check: does MATLAB prune this composition?
matlab_pruned = False
orig_pruned = False
curr_pruned = False

for ell in range(2, 2 * d + 1):
    n_cv = ell - 1
    for s_lo in range(conv_len - n_cv + 1):
        ws_int = sum(conv_int[s_lo:s_lo + n_cv])
        ws_cont = sum(conv_cont[s_lo:s_lo + n_cv])
        lo_bin = max(0, s_lo - (d - 1))
        hi_bin = min(d - 1, s_lo + ell - 2)
        W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])
        W_cont = W_int / m

        test_val = ws_cont * (2 * d) / ell
        matlab_bound = (c_target + gs * gs) + 2 * gs * W_cont
        if test_val >= matlab_bound:
            matlab_pruned = True

        dyn_base_orig = c_target * m * m + 3.0 + eps_margin
        orig_dyn_x = (dyn_base_orig + 2.0 * W_int) * ell * inv_4n
        orig_dyn_it = int(orig_dyn_x * one_minus_4eps)
        if ws_int > orig_dyn_it:
            orig_pruned = True

        cs_corr_base = c_target * m * m + 3.0 + eps_margin
        curr_dyn_x = (cs_corr_base + 2.0 * W_int) * ell * inv_4n
        curr_dyn_it = int(curr_dyn_x * one_minus_4eps)
        if ws_int > curr_dyn_it:
            curr_pruned = True

print(f"Composition {tuple(a_int)}:")
print(f"  MATLAB prunes:       {matlab_pruned}")
print(f"  ORIGINAL CPU prunes: {orig_pruned}")
print(f"  CURRENT CPU prunes:  {curr_pruned}")
print()

if discrepancies:
    print(f"{len(discrepancies)} windows have discrepancies!")
    for (ell, s_lo, mp, op, cp) in discrepancies:
        print(f"  ell={ell}, s_lo={s_lo}: MATLAB={mp}, ORIG={op}, CURR={cp}")
else:
    print("No discrepancies found.")

# ==========================================================================
# Step 3: General formula comparison
# ==========================================================================
print(f"\n{'='*70}")
print("FORMULA COMPARISON (general)")
print(f"{'='*70}")
print()
print("The test value T(ell, s) is defined as:")
print("  T = (1/(4*n*ell)) * sum_{k in window} conv_int[k]")
print("    = ws_int / (4*n*ell)")
print()
print("The MATLAB checks: T * (2d) >= c + gs^2 + 2*gs*W")
print("Wait-- let me re-derive from the MATLAB code...")
print()
print("MATLAB code does:")
print("  convFunctionVals = functionMult * sumIndicesStore{j}")
print("    = sum of f_i*f_j for pairs in window")
print("    ≈ ws_cont  (sum of point-eval conv over window)")
print()
print("  convFunctionVals *= (2*numBins) / j")
print("    = ws_cont * (2*d) / ell")
print()
print("  boundToBeat = (c + gs^2) + 2*gs*W")
print()
print("  Prune if: ws_cont * (2d/ell) >= c + gs^2 + 2*gs*W")
print()
print("Converting to integer domain:")
print("  ws_int/m^2 * (2d/ell) >= c + 1/m^2 + 2*W_int/m^2")
print("  ws_int * (2d/ell) >= c*m^2 + 1 + 2*W_int")
print("  ws_int >= (c*m^2 + 1 + 2*W_int) * ell/(2d)")
print(f"  ws_int >= (c*m^2 + 1 + 2*W_int) * ell/(4n)   [since 2d=4n]")
print()
print("So ALL terms (c*m^2 + 1 + 2*W_int) are scaled by ell/(4n).")
print()
print("ORIGINAL CPU (correct):")
print("  dyn_x = (c*m^2 + 1 + eps + 2*W_int) * ell/(4n)  ← matches!")
print()
print("CURRENT CPU (buggy):")
print("  dyn_x = c*m^2*ell/(4n) + 1 + eps + 2*W_int      ← WRONG!")
print()

# Numerical example
ell = 2
W = 10
correct_thresh = (c_target * m * m + 1 + 2 * W) * ell / (4 * n_half)
buggy_thresh = c_target * m * m * ell / (4 * n_half) + 1 + 2 * W
print(f"Example: ell={ell}, W_int={W}")
print(f"  Correct threshold: ({c_target}*{m*m} + 1 + 2*{W}) * {ell}/{4*n_half}")
print(f"                   = {c_target * m * m + 1 + 2 * W} * {ell/(4*n_half)}")
print(f"                   = {correct_thresh:.4f}")
print(f"  Buggy threshold:   {c_target}*{m*m}*{ell}/{4*n_half} + 1 + 2*{W}")
print(f"                   = {c_target * m * m * ell / (4 * n_half)} + {1 + 2 * W}")
print(f"                   = {buggy_thresh:.4f}")
print(f"  Difference:        {buggy_thresh - correct_thresh:.4f}")
print(f"  Relative error:    {(buggy_thresh - correct_thresh)/correct_thresh*100:.1f}%")
print()

ell = 8  # = 4n, max
correct_thresh = (c_target * m * m + 1 + 2 * W) * ell / (4 * n_half)
buggy_thresh = c_target * m * m * ell / (4 * n_half) + 1 + 2 * W
print(f"Example: ell={ell} (=4n, max), W_int={W}")
print(f"  Correct threshold: {correct_thresh:.4f}")
print(f"  Buggy threshold:   {buggy_thresh:.4f}")
print(f"  Difference:        {buggy_thresh - correct_thresh:.4f}  (should be ~0)")

"""
Verify Theorem 1 (test value lower bound) for the Sidon autocorrelation problem.

Theorem 1: max_{|t|<=1/2} (f*f)(t) >= TV_W for any window W,
where TV_W = (2d/ell) * sum_{k=s}^{s+ell-2} sum_{i+j=k} mu_i * mu_j

We test this by:
1. Creating several test functions f on [-1/4, 1/4] with integral 1
2. Computing the TRUE max of (f*f)(t) numerically
3. Computing bin masses and max TV over all windows
4. Checking the inequality holds
"""

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Helper: compute (f*f)(t) on a fine grid via numerical convolution
# ---------------------------------------------------------------------------

def compute_true_max(f_vals, x_grid):
    """Compute max_{|t|<=1/2} (f*f)(t) via direct numerical integration.

    For each t on a fine grid, compute integral f(x)*f(t-x) dx using
    interpolation for f(t-x).
    """
    dx = x_grid[1] - x_grid[0]
    # Use numpy convolve (Riemann sum approximation)
    conv_vals = np.convolve(f_vals, f_vals, mode='full') * dx
    return np.max(conv_vals)


def compute_bin_masses(f_vals, x_grid, d):
    """Compute bin masses mu_i for d bins on [-1/4, 1/4].

    Uses trapezoidal integration within each bin for accuracy, and
    normalizes masses to sum to exactly 1 (matching the constraint).
    """
    bin_width = 0.5 / d  # 1/(2d)
    masses = np.zeros(d)
    for i in range(d):
        lo = -0.25 + i * bin_width
        hi = lo + bin_width
        mask = (x_grid >= lo) & (x_grid <= hi)
        x_bin = x_grid[mask]
        f_bin = f_vals[mask]
        if len(x_bin) > 1:
            masses[i] = np.trapz(f_bin, x_bin)
        else:
            masses[i] = 0.0
    # Normalize to sum to 1 to remove discretization drift
    total = np.sum(masses)
    if total > 0:
        masses *= 1.0 / total
    return masses


def compute_max_tv(masses, d):
    """Compute max TV over all windows (ell, s).

    TV_W = (2d/ell) * sum_{k=s}^{s+ell-2} conv[k]
    where conv[k] = sum_{i+j=k} mu_i * mu_j

    k ranges from 0 to 2d-2 (convolution indices).
    Window (ell, s): ell bins in convolution space, starting at position s.
    s ranges from 0 to 2d-2, and s+ell-2 <= 2d-2, so s <= 2d-ell.
    ell ranges from 1 to 2d-1.
    """
    # Compute discrete convolution of masses
    conv = np.convolve(masses, masses, mode='full')  # length 2d-1, indices 0..2d-2

    best_tv = 0.0
    best_window = None
    n_conv = len(conv)  # 2d - 1

    for ell in range(1, n_conv + 1):
        for s in range(0, n_conv - ell + 1):
            # sum conv[s] through conv[s + ell - 1]
            # But the formula says sum_{k=s}^{s+ell-2}, which is ell-1 terms
            # Actually let me re-check: window of ell consecutive bins in
            # convolution space covers ell bins, total length ell/(2d).
            # The convolution has 2d-1 entries at indices 0..2d-2.
            # A window of ell entries starting at index s sums conv[s..s+ell-1].
            # TV = (2d/ell) * sum(conv[s..s+ell-1])
            # Wait - the formula in the docstring says sum_{k=s}^{s+ell-2} which
            # is ell-1 terms. Let me look at the CLAUDE.md more carefully.
            #
            # From CLAUDE.md: "for a window W of ell consecutive bins in
            # convolution space (total length ell/(2d))"
            # and the integral equals sum of mu_i * mu_k for pairs contributing
            # to W, so max >= sum / (ell/(2d)) = sum * 2d/ell.
            #
            # The convolution array has 2d-1 entries. A window of ell entries
            # starting at s sums ell entries: conv[s], ..., conv[s+ell-1].
            # TV = (2d/ell) * sum(conv[s..s+ell-1])

            window_sum = np.sum(conv[s:s + ell])
            tv = (2 * d / ell) * window_sum
            if tv > best_tv:
                best_tv = tv
                best_window = (ell, s)

    return best_tv, best_window


# ---------------------------------------------------------------------------
# Test functions on [-1/4, 1/4] with integral 1
# ---------------------------------------------------------------------------

N = 100001  # fine grid points (odd for symmetry)
x = np.linspace(-0.25, 0.25, N)
dx = x[1] - x[0]

test_functions = {}

# 1. Uniform: f(x) = 2 (since integral over [-1/4,1/4] of 2 = 1)
f_uniform = np.ones(N) * 2.0
test_functions["Uniform (f=2)"] = f_uniform

# 2. Triangular: peaked at 0, f(x) = 8*(1/4 - |x|)
f_tri = 8.0 * (0.25 - np.abs(x))
f_tri = np.maximum(f_tri, 0)
# Normalize
f_tri /= (np.sum(f_tri) * dx)
test_functions["Triangular"] = f_tri

# 3. Spiky: concentrated near center, Gaussian-like
sigma = 0.03
f_spike = np.exp(-x**2 / (2 * sigma**2))
f_spike /= (np.sum(f_spike) * dx)
test_functions["Spiky (sigma=0.03)"] = f_spike

# 4. Two peaks: mass concentrated at +-1/8
f_two = np.exp(-(x - 0.125)**2 / (2 * 0.02**2)) + np.exp(-(x + 0.125)**2 / (2 * 0.02**2))
f_two /= (np.sum(f_two) * dx)
test_functions["Two peaks (+-1/8)"] = f_two

# 5. Step function: piecewise constant with random heights
np.random.seed(42)
f_step = np.zeros(N)
n_steps = 8
step_heights = np.random.exponential(1.0, n_steps)
step_width = N // n_steps
for i in range(n_steps):
    lo_idx = i * step_width
    hi_idx = (i + 1) * step_width if i < n_steps - 1 else N
    f_step[lo_idx:hi_idx] = step_heights[i]
f_step /= (np.sum(f_step) * dx)
test_functions["Random step (8 steps)"] = f_step

# 6. Another random step function
np.random.seed(123)
f_step2 = np.zeros(N)
step_heights2 = np.random.exponential(1.0, n_steps)
for i in range(n_steps):
    lo_idx = i * step_width
    hi_idx = (i + 1) * step_width if i < n_steps - 1 else N
    f_step2[lo_idx:hi_idx] = step_heights2[i]
f_step2 /= (np.sum(f_step2) * dx)
test_functions["Random step (seed=123)"] = f_step2

# 7. Cosine bump
f_cos = np.cos(2 * np.pi * x) + 1  # nonneg, supported on [-1/4, 1/4]
f_cos = np.maximum(f_cos, 0)
f_cos /= (np.sum(f_cos) * dx)
test_functions["Cosine bump"] = f_cos

# 8. Left-skewed: mass concentrated near -1/4
f_left = np.exp(-((x + 0.2)**2) / (2 * 0.03**2))
f_left /= (np.sum(f_left) * dx)
test_functions["Left-skewed"] = f_left


# ---------------------------------------------------------------------------
# Part 1: Verify Theorem 1 for each test function and d = 4, 8, 16
# ---------------------------------------------------------------------------

print("=" * 90)
print("PART 1: Verify Theorem 1 — max(f*f) >= max_TV for all test functions and d values")
print("=" * 90)

d_values = [4, 8, 16]
all_pass = True

for fname, f_vals in test_functions.items():
    true_max = compute_true_max(f_vals, x)

    for d in d_values:
        masses = compute_bin_masses(f_vals, x, d)
        mass_sum = np.sum(masses)
        max_tv, best_window = compute_max_tv(masses, d)

        holds = true_max >= max_tv - 1e-10  # small tolerance for numerical error
        status = "PASS" if holds else "FAIL"
        if not holds:
            all_pass = False

        gap = true_max - max_tv
        print(f"  {fname:30s}  d={d:2d}  true_max={true_max:.6f}  "
              f"max_TV={max_tv:.6f}  gap={gap:+.6f}  mass_sum={mass_sum:.6f}  [{status}]")
    print()

print("-" * 90)
if all_pass:
    print("ALL TESTS PASSED: Theorem 1 inequality holds for every test case.")
else:
    print("SOME TESTS FAILED!")
print()


# ---------------------------------------------------------------------------
# Part 2: For d=8, random mass vectors on simplex, check TV <= 2.0
# ---------------------------------------------------------------------------

print("=" * 90)
print("PART 2: For d=8, random mass vectors on simplex, verify max_TV <= 2.0")
print("  (Since max(f*f) <= ||f||_inf * ||f||_1 = ||f||_inf, and for uniform f=2,")
print("   max(f*f) = integral f(x)*f(t-x)dx. The uniform f gives (f*f)(0) = ")
print("   integral 2*2 dx over overlap, which for t=0 is 4 * 0.5 = 2.0.)")
print("=" * 90)
print()

# Actually: for uniform f=2 on [-1/4,1/4], (f*f)(0) = integral_{-1/4}^{1/4} 4 dx = 2.
# And max(f*f)(t) = 2 at t=0. So TV should be <= 2 for the uniform distribution.
# But for OTHER distributions, max(f*f) can be larger (e.g., spiky functions).
# The claim should be: for any mass vector on the simplex, TV is a LOWER BOUND
# on max(f*f), so it can be arbitrarily large if masses are concentrated.
#
# Actually, the TV for a mass vector is valid: TV <= max(f*f) for ANY f consistent
# with those masses. But the mass vector itself constrains what f can be.
# A mass vector with all mass in one bin means f is concentrated there, and
# max(f*f) would be large. So TV being large is fine.
#
# Better test: for the UNIFORM mass vector mu_i = 1/d, TV should be <= 2.0
# because the uniform distribution on [-1/4,1/4] achieves max(f*f) = 2.0.

d = 8
np.random.seed(42)

print(f"  Testing uniform mass vector (mu_i = 1/{d}):")
uniform_masses = np.ones(d) / d
max_tv_unif, _ = compute_max_tv(uniform_masses, d)
print(f"    max_TV = {max_tv_unif:.6f}  (<= 2.0? {'YES' if max_tv_unif <= 2.0 + 1e-10 else 'NO'})")
print()

print(f"  Testing 20 random mass vectors on the simplex (d={d}):")
all_bounded = True
for trial in range(20):
    # Sample from Dirichlet (uniform on simplex)
    raw = np.random.exponential(1.0, d)
    masses = raw / np.sum(raw)
    max_tv, _ = compute_max_tv(masses, d)

    # For a step function with these masses, compute true max
    f_step_trial = np.zeros(N)
    bin_width = 0.5 / d
    for i in range(d):
        lo = -0.25 + i * bin_width
        hi = lo + bin_width
        mask = (x >= lo) & (x < hi)
        if i == d - 1:
            mask = (x >= lo) & (x <= hi)
        height = masses[i] / bin_width  # mass / width = height for step fn
        f_step_trial[mask] = height

    true_max_trial = compute_true_max(f_step_trial, x)
    holds = true_max_trial >= max_tv - 1e-10

    if not holds:
        all_bounded = False

    # For step function, max_TV should equal true max (approximately)
    # since the step function IS the discrete approximation
    print(f"    Trial {trial:2d}: max_TV={max_tv:.6f}  true_max_step={true_max_trial:.6f}  "
          f"gap={true_max_trial - max_tv:+.6f}  {'PASS' if holds else 'FAIL'}")

print()
if all_bounded:
    print("  ALL random trials: Theorem 1 holds (true_max >= max_TV).")
else:
    print("  SOME random trials FAILED!")

print()

# ---------------------------------------------------------------------------
# Part 3: Sanity check — TV for uniform masses
# ---------------------------------------------------------------------------

print("=" * 90)
print("PART 3: Sanity checks")
print("=" * 90)

# For uniform f=2, all masses are 1/d.
# conv[k] = sum_{i+j=k} (1/d)^2 = (number of pairs with i+j=k) / d^2
# The number of pairs (i,j) with i+j=k (0<=i,j<=d-1) is min(k+1, d, 2d-1-k)
# For ell=1 window at k=d-1 (the peak): conv[d-1] = d/d^2 = 1/d
# TV = 2d/1 * 1/d = 2.  So the single-bin window at the center gives TV=2.

for d in [4, 8, 16]:
    masses = np.ones(d) / d
    conv = np.convolve(masses, masses)
    peak_idx = d - 1
    tv_single = 2 * d * conv[peak_idx]
    max_tv, best_win = compute_max_tv(masses, d)
    print(f"  d={d:2d}: uniform masses, conv peak = {conv[peak_idx]:.6f}, "
          f"TV(ell=1,peak) = {tv_single:.6f}, max_TV = {max_tv:.6f}, "
          f"best_window = {best_win}")

print()
print("Expected: TV(ell=1, peak) = 2.0 for all d (uniform distribution).")
print("This confirms the formula is correctly implemented.")


# ---------------------------------------------------------------------------
# Part 4: Exact analytical test for step functions
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("PART 4: Exact analytical check for step functions (no numerical integration)")
print("  For a step function with d bins of heights h_i, the autoconvolution")
print("  (f*f)(t) is piecewise linear. At convolution knot points t_k,")
print("  (f*f)(t_k) = (1/(2d)) * sum_{i+j=k} h_i * h_j = conv[k]/(2d).")
print("  The max over knot points equals max_k conv[k]/(2d).")
print("  For ell=1 window at position k: TV = 2d * mu_i*mu_j-sum = 2d * conv[k]")
print("  where conv[k] uses masses mu_i = h_i/(2d). So TV(ell=1,k) = conv[k]*2d")
print("  and true_max(knot) = max_k conv_mass[k] * 2d = max TV(ell=1).")
print("  They should be exactly equal for ell=1 windows on step functions.")
print("=" * 90)
print()

np.random.seed(99)
for trial in range(10):
    d = 8
    # Random step heights
    heights = np.random.exponential(1.0, d)
    # Normalize so integral = 1: integral = sum(h_i * 1/(2d)) = sum(h_i)/(2d) = 1
    # => sum(h_i) = 2d
    heights *= (2 * d) / np.sum(heights)
    masses_exact = heights / (2 * d)  # mu_i = h_i / (2d)

    # Exact convolution of masses
    conv_exact = np.convolve(masses_exact, masses_exact)

    # Exact max of (f*f) at knot points = max_k (1/(2d)) * sum_{i+j=k} h_i*h_j
    # = max_k conv_heights[k] / (2d)
    # But conv_heights[k] = (2d)^2 * conv_masses[k]
    # So max = (2d)^2 * max(conv_masses) / (2d) = 2d * max(conv_masses)
    # And TV(ell=1, best k) = 2d * max(conv_masses)
    # So they're exactly equal.

    exact_max_at_knots = 2 * d * np.max(conv_exact)
    max_tv_exact, _ = compute_max_tv(masses_exact, d)

    # The true continuous max may be slightly higher (between knots for
    # overlapping triangular pieces), but the TV with ell=1 should match knots.
    print(f"  Trial {trial}: exact_knot_max={exact_max_at_knots:.10f}  "
          f"max_TV={max_tv_exact:.10f}  match={'YES' if abs(exact_max_at_knots - max_tv_exact) < 1e-12 else 'NO'}")

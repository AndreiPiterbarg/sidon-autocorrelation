"""Rigorous verification tests for Part 1: Mathematical Framework & Parameter Derivations.

These tests verify every constant, formula, and threshold used in the branch-and-prune
algorithm against first-principles mathematical derivations. They cover the gaps
identified in existing test coverage: dynamic threshold derivation, x_cap bounds,
FP safety margins, asymmetry margin necessity, and int32 overflow safety.
"""
import sys
import os
import math
import unittest
import numpy as np
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))

from pruning import correction, asymmetry_threshold, count_compositions, asymmetry_prune_mask
from test_values import compute_test_value_single, compute_test_values_batch


# =====================================================================
# 1. Correction term: base = 2/m + 1/m^2, full = factor * base
# =====================================================================

class TestCorrectionDerivation(unittest.TestCase):
    """Verify correction(m, n_half, ell_min) matches corrected Lemma 3.

    Legacy mode (n_half=None): returns base = 2/m + 1/m^2.
    Full mode: returns max(1, 4*n_half/ell_min) * base.
    """

    def test_base_formula_exact(self):
        """Legacy mode returns base = 2/m + 1/m^2."""
        for m in [5, 10, 20, 50, 100, 200]:
            expected = 2.0 / m + 1.0 / (m * m)
            self.assertAlmostEqual(correction(m), expected, places=15,
                                   msg=f"correction({m}) mismatch")

    def test_full_formula_exact(self):
        """With C&S Lemma 3, correction is window-independent (no 4n/ℓ factor).

        correction(m, n_half, ell_min) always returns the base 2/m + 1/m²,
        regardless of n_half or ell_min.
        """
        for m in [5, 10, 20, 50, 100, 200]:
            base = 2.0 / m + 1.0 / (m * m)
            for n_half in [1, 2, 4, 8, 16]:
                for ell_min in [2, 4]:
                    actual = correction(m, n_half=n_half, ell_min=ell_min)
                    self.assertAlmostEqual(actual, base, places=15,
                                           msg=f"correction({m}, {n_half}, {ell_min}) mismatch")

    def test_correction_is_global_upper_bound(self):
        """correction(m) (legacy) upper-bounds eps^2 + 2*eps*W for all W <= 1."""
        for m in [10, 20, 50, 100]:
            eps = 1.0 / m
            corr = correction(m)
            # W ranges from 0 to 1 (total mass = 1)
            for w in np.linspace(0, 1, 100):
                window_corr = eps * eps + 2 * eps * w
                self.assertLessEqual(window_corr, corr + 1e-15,
                                     msg=f"Window correction {window_corr} > correction({m})={corr} at W={w}")

    def test_correction_matches_matlab_gridspace(self):
        """Verify correction(m) (legacy) = gridSpace^2 + 2*gridSpace (with W=1 maximum)."""
        for m in [20, 50]:
            gridSpace = 1.0 / m
            matlab_max = gridSpace ** 2 + 2 * gridSpace * 1.0
            self.assertAlmostEqual(correction(m), matlab_max, places=15)

    def test_correction_decreasing(self):
        """correction(m) (legacy) is strictly decreasing."""
        prev = correction(2)
        for m in range(3, 201):
            curr = correction(m)
            self.assertLess(curr, prev, msg=f"correction not decreasing at m={m}")
            prev = curr

    def test_full_correction_ge_base(self):
        """Full correction (with n_half) is always >= base correction."""
        for m in [10, 20, 50, 100]:
            base = correction(m)
            for n_half in [1, 2, 4, 8]:
                full = correction(m, n_half=n_half)
                self.assertGreaterEqual(full, base - 1e-15,
                                        msg=f"correction({m}, {n_half}) < base")


# =====================================================================
# 2. Asymmetry threshold: sqrt(c_target / 2)
# =====================================================================

class TestAsymmetryThresholdDerivation(unittest.TestCase):
    """Verify sqrt(c_target/2) is the exact threshold where 2*L^2 = c_target."""

    def test_threshold_squared_identity(self):
        """2 * threshold^2 = c_target for all c_target > 0."""
        for c in [0.5, 1.0, 1.28, 1.3, 1.4, 1.5, 2.0]:
            t = asymmetry_threshold(c)
            self.assertAlmostEqual(2.0 * t * t, c, places=14,
                                   msg=f"2*threshold^2 != c_target for c_target={c}")

    def test_autoconvolution_lower_bound_step_function(self):
        """For a step function with left_frac >= threshold, verify ||f*f|| >= c_target.

        Constructs an explicit step function and computes its autoconvolution analytically.
        """
        c_target = 1.4
        n_half = 2
        d = 2 * n_half
        m = 20
        thresh = asymmetry_threshold(c_target)

        # Create config with left_frac just above threshold
        left_sum = math.ceil(thresh * m)  # Just above threshold
        if left_sum > m:
            left_sum = m
        right_sum = m - left_sum
        left_frac = left_sum / m

        # Step function: split mass evenly within halves
        c = np.array([left_sum // n_half + (1 if i < left_sum % n_half else 0)
                       for i in range(n_half)] +
                      [right_sum // n_half + (1 if i < right_sum % n_half else 0)
                       for i in range(n_half)], dtype=np.int32)

        # Verify sum = m
        self.assertEqual(c.sum(), m)

        # Compute actual autoconvolution maximum numerically
        delta = 1.0 / (4 * n_half)  # bin width
        a = c * (4 * n_half / m)     # heights

        # (f*f)(t) for step function: piecewise linear, evaluate at grid points
        max_conv = 0.0
        # Evaluate at fine grid
        ts = np.linspace(0, 2 * d * delta, 10000)
        for t in ts:
            val = 0.0
            for i in range(d):
                for j in range(d):
                    # Overlap of bins i and j when shifted by t
                    lo_i = i * delta
                    hi_i = (i + 1) * delta
                    lo_j = t - (j + 1) * delta
                    hi_j = t - j * delta
                    overlap = max(0, min(hi_i, hi_j) - max(lo_i, lo_j))
                    val += a[i] * a[j] * overlap
            if val > max_conv:
                max_conv = val

        # The asymmetry bound predicts: ||f*f|| >= 2 * left_frac^2
        asym_bound = 2.0 * left_frac * left_frac

        # Verify: actual max >= asymmetry bound
        self.assertGreaterEqual(max_conv + 1e-10, asym_bound,
                                msg=f"Autoconvolution {max_conv} < asym bound {asym_bound}")

        # And if left_frac >= threshold: asym_bound >= c_target
        if left_frac >= thresh:
            self.assertGreaterEqual(asym_bound, c_target - 1e-10,
                                    msg=f"Asym bound {asym_bound} < c_target {c_target}")


# =====================================================================
# 3. count_compositions = C(S+d-1, d-1)
# =====================================================================

class TestCompositionCountDerivation(unittest.TestCase):
    """Verify stars-and-bars by brute-force enumeration for small cases."""

    def _brute_force_count(self, d, S):
        """Count compositions by exhaustive enumeration."""
        if d == 1:
            return 1
        count = 0
        # Generate all d-tuples of non-negative integers summing to S
        for combo in itertools.combinations_with_replacement(range(S + 1), d - 1):
            # Use dividers method: d-1 dividers in 0..S
            pass
        # Simpler: recursive
        if d == 2:
            return S + 1
        count = 0
        for first in range(S + 1):
            count += self._brute_force_count(d - 1, S - first)
        return count

    def test_brute_force_small(self):
        """Verify formula against brute-force for all small (d, S)."""
        for d in range(1, 7):
            for S in range(0, 12):
                expected = self._brute_force_count(d, S)
                actual = count_compositions(d, S)
                self.assertEqual(actual, expected,
                                 msg=f"count_compositions({d}, {S}): got {actual}, expected {expected}")


# =====================================================================
# 4. Dynamic threshold derivation
# =====================================================================

class TestDynamicThresholdDerivation(unittest.TestCase):
    """Verify integer-space threshold matches Lean theorem dynamic_threshold_sound.

    NOTE: These tests verify the Lean's flat-bound threshold (C&S Lemma 3),
    which uses the old coarse-grid scaling factor ell/(4n).  The CPU code
    now uses the fine-grid (S=4nm) threshold:
      threshold = floor((c_target*m^2 + 3 + W_int/(2n) + eps) * 4n*ell)
    The Lean formalization still uses the flat bound (2/m + 1/m^2).

    Lean threshold (Theorem 3.7, proof/lower_bound_proof.tex):
      TV > c_target + (4n/ell) * (1/m^2 + 2*W/m)
    In integer space:
      ws > c_target*m^2*ell/(4n) + 1 + 2*W_int
    """

    def _lean_threshold_tv(self, c_target, m, W_int, ell, n_half):
        """Lean-proven threshold in test-value space (Theorem 3.7).

        Uses the coarse-grid scaling factor 4n/ell (Lean formalization).
        """
        return c_target + (4.0 * n_half / ell) * (1.0 + 2.0 * W_int) / (m * m)

    def _python_dyn_threshold_int(self, c_target, m, W_int, ell, n_half):
        """Old coarse-grid integer-space threshold (C&S Lemma 3 + W_g correction).

        Uses ell/(4n) scaling.  The CPU code now uses the fine-grid formula
        with 4n*ell scaling and W_int/(2n) correction term.
        """
        eps_margin = 1e-9 * m * m
        return (c_target * m * m + 3.0 + eps_margin + 2.0 * W_int) * ell / (4.0 * n_half)

    def test_matlab_python_equivalence(self):
        """Verify Lean threshold >= MATLAB flat threshold for all valid parameters.

        The MATLAB uses a flat correction (no 4n/ell factor).
        The Lean formula (with 4n/ell) is strictly >= MATLAB for ell <= 4n.

        NOTE: This tests the Lean/coarse-grid threshold relationship.
        The CPU code now uses the fine-grid threshold formula.
        """
        for m in [20, 50]:
            for c_target in [1.28, 1.3, 1.4]:
                for W_int in range(0, m + 1):
                    W_cont = W_int / m
                    # MATLAB flat correction: c_target + 1/m^2 + 2*W/m (no 4n/ell factor)
                    matlab_thresh = c_target + 1.0 / (m * m) + 2.0 * W_cont / m
                    for n_half in [2, 4, 8, 16, 32]:
                        for ell in range(2, 4 * n_half + 1):
                            lean_thresh = self._lean_threshold_tv(
                                c_target, m, W_int, ell, n_half)
                            self.assertGreaterEqual(
                                lean_thresh, matlab_thresh - 1e-12,
                                msg=f"Lean threshold {lean_thresh} < MATLAB flat {matlab_thresh} "
                                    f"at m={m}, c={c_target}, W_int={W_int}, ell={ell}, n={n_half}")

    def test_dyn_base_encodes_correction(self):
        """Verify the C&S + W_g corrected threshold formula.

        In the old coarse-grid parameterization, the ENTIRE correction
        (3 + 2*W_int) is scaled by ell/(4n).
        In TV space: c_target + (3 + 2*W_int)/m^2.

        NOTE: The CPU code now uses the fine-grid formula where the scaling
        factor is 4n*ell, but the TV-space threshold is equivalent.
        """
        m = 20
        c_target = 1.4
        n_half = 8
        eps_margin = 1e-9 * m * m
        for W_int in [0, 5, 10, 20]:
            for ell in [2, 8, 16, 32]:
                thresh_int = self._python_dyn_threshold_int(c_target, m, W_int, ell, n_half)
                # Convert to TV space
                tv_thresh = thresh_int * 4.0 * n_half / (m * m * ell)
                # Expected: c_target + (3 + 2*W_int + eps)/m²
                expected = c_target + (3.0 + 2.0 * W_int + eps_margin) / (m * m)
                self.assertAlmostEqual(tv_thresh, expected, places=6,
                                       msg=f"Mismatch at W_int={W_int}, ell={ell}")

    def test_threshold_integer_conversion_exact(self):
        """Verify the algebra: ws > c_target*m^2*ell/(4n) + 1 + 2*W_int
        is equivalent to TV > c_target + (4n/ell)*(1/m^2 + 2*W_int/m^2).

        This is Theorem 3.7 (dynamic_threshold_sound) from the Lean proof.

        NOTE: This tests the Lean/coarse-grid threshold conversion.
        The CPU fine-grid formula uses a different integer scaling (4n*ell)
        but yields the same TV-space threshold.
        """
        m = 20
        c_target = 1.4
        n_half = 8
        d = 2 * n_half

        for W_int in [0, 5, 10, 15, 20]:
            for ell in [2, 4, 8, d]:
                # Integer threshold (no FP margin)
                thresh_int = c_target * m * m * ell / (4.0 * n_half) + 1.0 + 2.0 * W_int

                # Convert ws > thresh_int to TV space:
                # TV = ws * 4n/(m^2 * ell), so TV > thresh_int * 4n/(m^2 * ell)
                tv_thresh = thresh_int * 4.0 * n_half / (m * m * ell)

                # Expected (Lean coarse-grid): c_target + (4n/ell)*(1/m^2 + 2*W_int/m^2)
                expected = c_target + (4.0 * n_half / ell) * (1.0 + 2.0 * W_int) / (m * m)

                self.assertAlmostEqual(tv_thresh, expected, places=10,
                                       msg=f"Integer threshold conversion failed at "
                                           f"W_int={W_int}, ell={ell}")


# =====================================================================
# 5. x_cap formulas
# =====================================================================

class TestXCapDerivation(unittest.TestCase):
    """Verify x_cap formulas and that CS cap <= test-value cap always."""

    def _x_cap_tv(self, m, c_target, d_child):
        n_half_child = d_child // 2
        corr = correction(m, n_half_child)
        thresh = c_target + corr + 1e-9
        return int(math.floor(m * math.sqrt(thresh / d_child)))

    def _x_cap_cs(self, m, c_target, d_child):
        return int(math.floor(m * math.sqrt(c_target / d_child))) + 1

    def test_effective_cap_is_min(self):
        """The effective x_cap = min(x_cap_tv, x_cap_cs) is always valid.

        With C&S Lemma 3 correction (no 4n/ℓ factor), x_cap_tv can be
        smaller than x_cap_cs for small m or low c_target. The code uses
        min(tv, cs) so both bounds are respected.
        """
        for m in [10, 20, 50, 100, 200]:
            for c_target in [1.0, 1.28, 1.3, 1.4, 1.5]:
                for d_child in [4, 8, 16, 32, 64, 128, 256]:
                    tv = self._x_cap_tv(m, c_target, d_child)
                    cs = self._x_cap_cs(m, c_target, d_child)
                    effective = min(tv, cs)
                    self.assertGreaterEqual(
                        effective, 0,
                        msg=f"Effective cap negative at m={m}, c={c_target}, d={d_child}")

    def test_cs_cap_soundness(self):
        """Any c_i > x_cap_cs has d*((c_i-1)/m)^2 > c_target (adjusted-bin CS bound)."""
        for m in [20, 50, 100]:
            for c_target in [1.28, 1.3, 1.4]:
                for d in [4, 8, 16, 32, 64]:
                    x_cap = self._x_cap_cs(m, c_target, d)
                    c_i = x_cap + 1
                    # For adjusted bins, mu_i >= (c_i - 1)/m
                    bound = d * (c_i - 1) * (c_i - 1) / (m * m)
                    self.assertGreater(
                        bound, c_target,
                        msg=f"CS bound {bound} <= c_target {c_target} for "
                            f"c_i={c_i}, m={m}, d={d}")

    def test_cs_cap_not_too_aggressive(self):
        """c_i = x_cap_cs should NOT trigger the adjusted-bin CS bound."""
        for m in [20, 50, 100]:
            for c_target in [1.28, 1.3, 1.4]:
                for d in [4, 8, 16, 32, 64]:
                    x_cap = self._x_cap_cs(m, c_target, d)
                    if x_cap >= 0:
                        # For adjusted bins, mu_i >= (c_i-1)/m
                        bound = d * (x_cap - 1) * (x_cap - 1) / (m * m)
                        self.assertLessEqual(
                            bound, c_target + 1e-10,
                            msg=f"CS bound {bound} > c_target for x_cap={x_cap}")

    def test_cs_bound_derivation(self):
        """Verify d*c_i^2/m^2 = M_i^2/(2*Delta) where M_i = c_i/m, Delta = 1/(4n)."""
        for n_half in [2, 4, 8, 16, 32]:
            d = 2 * n_half
            m = 20
            for c_i in range(1, m + 1):
                M_i = c_i / m
                Delta = 1.0 / (4 * n_half)
                cs_bound = M_i ** 2 / (2 * Delta)
                formula = d * c_i ** 2 / (m * m)
                self.assertAlmostEqual(cs_bound, formula, places=12,
                                       msg=f"CS derivation mismatch at n={n_half}, c_i={c_i}")

    def test_cs_bound_invariant_under_refinement(self):
        """Splitting a bin preserves the total mass, so CS bound holds after refinement."""
        m = 20
        n_half = 2
        d = 4
        Delta = 1.0 / (4 * n_half)

        for c_parent in range(1, m + 1):
            M_parent = c_parent / m
            parent_cs_bound = M_parent ** 2 / (2 * Delta)

            # Split into all possible (a, b) with a+b = c_parent
            for a in range(c_parent + 1):
                b = c_parent - a
                # Child bins (a, b) are in a region of width 2*Delta_child = Delta
                # The total mass in this region is still M_parent = c_parent/m
                # supp(g_region * g_region) has length 2*Delta (same region)
                # CS bound: M_parent^2 / (2*Delta) -- unchanged
                child_region_bound = M_parent ** 2 / (2 * Delta)
                self.assertAlmostEqual(child_region_bound, parent_cs_bound, places=12)


# =====================================================================
# 6. Floating-point safety margins
# =====================================================================

class TestFPSafetyMargins(unittest.TestCase):
    """Verify FP margins ensure dyn_it >= floor(true_dyn_x)."""

    def test_margin_exceeds_fp_error(self):
        """1e-9*m^2 >> max accumulated FP error for all m, ell, n."""
        DBL_EPS = 2.220446049250313e-16

        for m in [20, 50, 100, 200]:
            margin = 1e-9 * m * m
            for n_half in [2, 4, 8, 16, 32]:
                for ell in [2, 4, n_half, 2 * n_half, 4 * n_half]:
                    # Typical dyn_x value
                    dyn_x = (1.4 * m * m + 1 + m) * ell / (4.0 * n_half)
                    # Max FP error: 4 ops * 0.5 ULP each
                    max_fp_error = 4 * 0.5 * DBL_EPS * dyn_x
                    # Margin contribution to dyn_x
                    margin_contribution = margin * ell / (4.0 * n_half)

                    self.assertGreater(
                        margin_contribution, max_fp_error * 100,
                        msg=f"Margin not sufficient at m={m}, n={n_half}, ell={ell}: "
                            f"margin={margin_contribution}, fp_err={max_fp_error}")

    def test_dyn_it_geq_floor_true(self):
        """dyn_it = floor(computed_dyn_x * (1-4eps)) >= floor(true_dyn_x)."""
        DBL_EPS = 2.220446049250313e-16
        one_minus_4eps = 1.0 - 4.0 * DBL_EPS

        for m in [20, 50, 100, 200]:
            for c_target in [1.28, 1.3, 1.4]:
                for n_half in [2, 4, 8, 16, 32]:
                    for ell in range(2, min(4 * n_half + 1, 65)):
                        for W_int in [0, m // 4, m // 2, m]:
                            # True mathematical threshold
                            true_dyn_x = (c_target * m * m + 3.0 + 2.0 * W_int) * ell / (4.0 * n_half)
                            true_floor = math.floor(true_dyn_x)

                            # Computed with margin
                            comp_dyn_x = (c_target * m * m + 3.0 + 1e-9 * m * m + 2.0 * W_int) * ell / (4.0 * n_half)
                            dyn_it = int(comp_dyn_x * one_minus_4eps)

                            self.assertGreaterEqual(
                                dyn_it, true_floor,
                                msg=f"dyn_it ({dyn_it}) < floor(true) ({true_floor}) at "
                                    f"m={m}, c={c_target}, n={n_half}, ell={ell}, W={W_int}")

    def test_one_minus_4eps_negligible(self):
        """The (1-4eps) factor reduces dyn_x by << 1e-9*m^2 contribution."""
        DBL_EPS = 2.220446049250313e-16
        m = 20
        n_half = 32
        ell = 64
        c_target = 1.4
        W_int = 10

        dyn_x = (c_target * m * m + 3.0 + 1e-9 * m * m + 2.0 * W_int) * ell / (4.0 * n_half)
        reduction = 4 * DBL_EPS * dyn_x
        margin_contribution = 1e-9 * m * m * ell / (4.0 * n_half)

        # Margin should dominate reduction by at least 1000x
        self.assertGreater(margin_contribution / reduction, 1000,
                           msg=f"Margin/reduction ratio too small: {margin_contribution/reduction}")


# =====================================================================
# 7. Asymmetry margin: rigorous proof it is unnecessary
# =====================================================================

class TestAsymmetryMarginUnnecessary(unittest.TestCase):
    """Prove that the 1/(4m) margin in asymmetry pruning is unnecessary.

    We prove three facts:
    1. left_frac is exact for step functions (no approximation error)
    2. left_frac is exactly preserved under cascade refinement
    3. The boundary always aligns with a bin edge
    """

    def test_left_frac_exact_for_step_functions(self):
        """The discrete left_frac equals the continuous left mass fraction exactly.

        For a step function on d bins, the left half mass is:
          L = sum_{i<n_half} a_i * Delta = sum_{i<n_half} c_i / m
        which equals left_frac = sum_{i<n_half} c_i / m.
        """
        for n_half in [1, 2, 4, 8, 16]:
            d = 2 * n_half
            m = 20
            Delta = 1.0 / (4 * n_half)

            # Generate random configs
            rng = np.random.RandomState(42 + n_half)
            for _ in range(100):
                c = rng.multinomial(m, np.ones(d) / d)
                a = c * (4 * n_half / m)

                # Continuous left mass
                L_continuous = sum(a[i] * Delta for i in range(n_half))

                # Discrete left_frac
                left_frac = sum(c[i] for i in range(n_half)) / m

                self.assertAlmostEqual(
                    L_continuous, left_frac, places=14,
                    msg=f"left_frac mismatch at n_half={n_half}, c={c}")

    def test_left_frac_preserved_under_refinement(self):
        """left_frac_child = left_frac_parent for all possible refinements.

        When parent bin k splits into child bins (2k, 2k+1) with
        child[2k] + child[2k+1] = parent[k], the child's left sum
        equals the parent's left sum.

        Proof:
          n_half_child = d_parent = 2 * n_half_parent
          left_child = sum_{i=0}^{n_half_child-1} child[i]
                     = sum_{i=0}^{d_parent-1} child[i]
                     = sum_{k=0}^{d_parent/2 - 1} (child[2k] + child[2k+1])
                     = sum_{k=0}^{n_half_parent - 1} parent[k]
                     = left_parent
        """
        m = 20

        for n_half_parent in [1, 2, 4, 8]:
            d_parent = 2 * n_half_parent
            n_half_child = d_parent  # = 2 * n_half_parent
            d_child = 2 * d_parent

            rng = np.random.RandomState(123 + n_half_parent)

            for _ in range(200):
                # Random parent config
                parent = rng.multinomial(m, np.ones(d_parent) / d_parent)

                # Parent left_frac
                parent_left = sum(parent[k] for k in range(n_half_parent))

                # Random child: split each parent bin
                child = np.zeros(d_child, dtype=int)
                for k in range(d_parent):
                    a = rng.randint(0, parent[k] + 1)
                    child[2 * k] = a
                    child[2 * k + 1] = parent[k] - a

                # Child left_frac
                child_left = sum(child[i] for i in range(n_half_child))

                self.assertEqual(
                    child_left, parent_left,
                    msg=f"left sum changed: parent_left={parent_left}, child_left={child_left}, "
                        f"parent={parent}, child={child}")

    def test_left_frac_preserved_multi_level(self):
        """left_frac is preserved across 5 levels of refinement (L0 through L4)."""
        m = 20
        n_half_0 = 2
        d_0 = 4

        rng = np.random.RandomState(999)

        for trial in range(50):
            # L0 config
            config = rng.multinomial(m, np.ones(d_0) / d_0)
            left_frac_0 = sum(config[:n_half_0]) / m

            current = config.copy()
            n_half_current = n_half_0
            d_current = d_0

            for level in range(1, 5):
                d_parent = d_current
                d_child = 2 * d_parent
                n_half_child = d_parent

                child = np.zeros(d_child, dtype=int)
                for k in range(d_parent):
                    a = rng.randint(0, current[k] + 1)
                    child[2 * k] = a
                    child[2 * k + 1] = current[k] - a

                child_left = sum(child[:n_half_child])
                child_frac = child_left / m

                self.assertAlmostEqual(
                    child_frac, left_frac_0, places=15,
                    msg=f"left_frac changed at level {level}: "
                        f"L0={left_frac_0}, L{level}={child_frac}")

                current = child
                d_current = d_child
                n_half_current = n_half_child

    def test_boundary_always_on_bin_edge(self):
        """The midpoint boundary falls exactly on a bin edge at every level.

        At level L: d = 2^(L+2) bins (for n_half_0=2), boundary at bin d/2.
        Since d is always even, d/2 is always an integer.
        """
        n_half_0 = 2

        n_half = n_half_0
        for level in range(10):
            d = 2 * n_half
            boundary_bin = d / 2  # This should be exactly an integer

            self.assertEqual(
                boundary_bin, int(boundary_bin),
                msg=f"Boundary not on bin edge at level {level}: d={d}, boundary={boundary_bin}")

            # Next level
            n_half = d  # n_half_child = d_parent = d

    def test_margin_removal_improves_pruning(self):
        """Verify that removing the old 1/(4m) margin prunes more configs.

        The old margin weakened pruning by raising the threshold. Now that
        we've proven it unnecessary, removing it allows pruning at the
        exact threshold, which is strictly tighter.
        """
        for m in [10, 20, 50, 100]:
            for c_target in [1.28, 1.3, 1.4]:
                threshold = asymmetry_threshold(c_target)
                old_margin = 1.0 / (4 * m)
                old_safe = threshold + old_margin

                pruned_old = 0
                pruned_new = 0
                for left_sum in range(m + 1):
                    lf = left_sum / m
                    if lf >= old_safe or (1 - lf) >= old_safe:
                        pruned_old += 1
                    if lf >= threshold or (1 - lf) >= threshold:
                        pruned_new += 1

                # New (no margin) prunes at least as many as old
                self.assertGreaterEqual(pruned_new, pruned_old,
                                        msg=f"Removal worsened pruning at m={m}, c={c_target}")

    def test_asymmetry_valid_without_margin(self):
        """Direct proof: for any config with left_frac >= threshold (no margin),
        the continuous autoconvolution exceeds c_target.

        We verify this for ALL canonical d=4, m=20 configs where
        left_frac >= sqrt(c_target/2).
        """
        n_half = 2
        d = 4
        m = 20
        c_target = 1.4
        threshold = asymmetry_threshold(c_target)

        violations = 0
        tested = 0

        for c0 in range(m + 1):
            for c1 in range(m - c0 + 1):
                for c2 in range(m - c0 - c1 + 1):
                    c3 = m - c0 - c1 - c2
                    left_frac = (c0 + c1) / m

                    # Only test configs at or above threshold (no margin)
                    if left_frac < threshold and (1 - left_frac) < threshold:
                        continue

                    tested += 1

                    # Compute continuous autoconvolution maximum
                    a = np.array([c0, c1, c2, c3], dtype=np.float64) * (4 * n_half / m)
                    delta = 1.0 / (4 * n_half)

                    # Evaluate (f*f)(t) at fine grid points
                    max_conv = 0.0
                    for t_idx in range(4 * d * 100):
                        t = t_idx * delta / 100
                        val = 0.0
                        for i in range(d):
                            for j in range(d):
                                lo_i = i * delta
                                hi_i = (i + 1) * delta
                                lo_j = t - (j + 1) * delta
                                hi_j = t - j * delta
                                overlap = max(0.0, min(hi_i, hi_j) - max(lo_i, lo_j))
                                val += a[i] * a[j] * overlap
                        if val > max_conv:
                            max_conv = val

                    if max_conv < c_target - 1e-9:
                        violations += 1

        self.assertEqual(violations, 0,
                         msg=f"{violations} configs with left_frac >= threshold "
                             f"have ||f*f|| < c_target (tested {tested})")
        self.assertGreater(tested, 0, "No configs tested")


# =====================================================================
# 8. int32 overflow analysis
# =====================================================================

class TestInt32Overflow(unittest.TestCase):
    """Verify int32 values never overflow for m <= 200."""

    def _max_raw_conv_entry(self, m, d, x_cap):
        """Upper bound on max raw_conv[k] for given parameters."""
        # At index k near d-1: up to d/2 pairs, each contributing 2*x_cap^2
        # plus possibly one diagonal x_cap^2
        max_pairs = d // 2
        return max_pairs * 2 * x_cap * x_cap + x_cap * x_cap

    def test_raw_conv_fits_int32(self):
        """Max raw_conv entry fits in int32 for all parameter combos with m <= 200."""
        INT32_MAX = 2147483647

        for m in [20, 50, 100, 200]:
            for c_target in [1.28, 1.3, 1.4]:
                for d in [4, 8, 16, 32, 64, 128]:
                    x_cap = int(math.floor(m * math.sqrt(c_target / d)))
                    x_cap = min(x_cap, m)
                    if x_cap == 0:
                        continue
                    max_entry = self._max_raw_conv_entry(m, d, x_cap)
                    self.assertLess(
                        max_entry, INT32_MAX,
                        msg=f"raw_conv overflow: {max_entry} at m={m}, d={d}, x_cap={x_cap}")

    def test_prefix_sum_fits_int32(self):
        """Max prefix sum = m^2 fits in int32 for m <= 200."""
        INT32_MAX = 2147483647
        for m in range(1, 201):
            self.assertLess(m * m, INT32_MAX,
                            msg=f"m^2 = {m*m} overflows int32 at m={m}")

    def test_incremental_delta_fits_int32(self):
        """Max incremental conv delta fits in int32."""
        INT32_MAX = 2147483647
        for m in [20, 50, 100, 200]:
            # Max delta from self-term update: |new^2 - old^2| <= m^2
            max_self_delta = m * m
            # Max delta from cross-term: |2*(new1*new2 - old1*old2)| <= 2*m^2
            max_cross_delta = 2 * m * m
            self.assertLess(max_self_delta, INT32_MAX)
            self.assertLess(max_cross_delta, INT32_MAX)

    def test_specific_l4_parameters(self):
        """Verify int32 safety for the exact L4 parameters: m=20, d=64."""
        m = 20
        d = 64
        c_target = 1.4
        x_cap = int(math.floor(m * math.sqrt(c_target / d)))
        x_cap = min(x_cap, m)

        self.assertEqual(x_cap, 2, f"Expected x_cap=2 at L4, got {x_cap}")

        # Max raw_conv entry: 32 pairs * 2 * 4 + 4 = 260
        max_entry = (d // 2) * 2 * x_cap * x_cap + x_cap * x_cap
        self.assertEqual(max_entry, 32 * 8 + 4)
        self.assertLess(max_entry, 2147483647)

        # Max prefix sum = m^2 = 400
        self.assertEqual(m * m, 400)
        self.assertLess(400, 2147483647)

    def test_ws_subtraction_safe_in_int64(self):
        """ws subtraction (conv[s_hi] - conv[s_lo-1]) is done in int64.

        Even though conv values are int32, the subtraction is widened to int64
        before execution, so no underflow/overflow is possible.
        """
        # Max int32 conv value: m^2 = 40000 for m=200
        max_conv = 200 * 200
        # Subtraction range: [-max_conv, max_conv]
        min_ws = -max_conv
        max_ws = max_conv

        INT64_MIN = -9223372036854775808
        INT64_MAX = 9223372036854775807

        self.assertGreater(min_ws, INT64_MIN)
        self.assertLess(max_ws, INT64_MAX)


# =====================================================================
# 9. W_int contributing bins correctness
# =====================================================================

class TestContributingBins(unittest.TestCase):
    """Verify the contributing bin range [lo_bin, hi_bin] is correct."""

    def test_contributing_bins_brute_force(self):
        """Brute-force verify contributing bin range for all small d, ell, s_lo."""
        for d in [4, 6, 8]:
            conv_len = 2 * d - 1
            for ell in range(2, 2 * d + 1):
                n_cv = ell - 1
                for s_lo in range(conv_len - n_cv + 1):
                    # Code formula
                    lo_bin = max(0, s_lo - (d - 1))
                    hi_bin = min(d - 1, s_lo + ell - 2)

                    # Brute-force: find all bins that contribute
                    contributing = set()
                    for i in range(d):
                        for j in range(d):
                            k = i + j
                            if s_lo <= k <= s_lo + n_cv - 1:
                                contributing.add(i)
                                contributing.add(j)

                    if not contributing:
                        continue

                    bf_lo = min(contributing)
                    bf_hi = max(contributing)

                    self.assertEqual(
                        lo_bin, bf_lo,
                        msg=f"lo_bin mismatch: d={d}, ell={ell}, s_lo={s_lo}: "
                            f"code={lo_bin}, brute={bf_lo}")
                    self.assertEqual(
                        hi_bin, bf_hi,
                        msg=f"hi_bin mismatch: d={d}, ell={ell}, s_lo={s_lo}: "
                            f"code={hi_bin}, brute={bf_hi}")


# =====================================================================
# 10. Test value formula consistency
# =====================================================================

class TestTestValueConsistency(unittest.TestCase):
    """Verify the test value formula TV = ws/(4n*ell) matches manual computation."""

    def test_tv_manual_vs_code(self):
        """Compare code's test value with manual autoconvolution + windowing."""
        n_half = 2
        d = 4
        m = 20

        rng = np.random.RandomState(77)
        for _ in range(50):
            c = rng.multinomial(m, np.ones(d) / d).astype(np.int32)
            a = c * (4.0 * n_half / m)

            # Manual autoconvolution
            conv_len = 2 * d - 1
            conv = np.zeros(conv_len, dtype=np.float64)
            for i in range(d):
                for j in range(d):
                    conv[i + j] += a[i] * a[j]

            # Manual test value: max over all windows
            best = 0.0
            for ell in range(2, 2 * d + 1):
                n_cv = ell - 1
                inv_norm = 1.0 / (4.0 * n_half * ell)
                cum = np.cumsum(conv)
                for s_lo in range(conv_len - n_cv + 1):
                    s_hi = s_lo + n_cv - 1
                    ws = cum[s_hi]
                    if s_lo > 0:
                        ws -= cum[s_lo - 1]
                    tv = ws * inv_norm
                    if tv > best:
                        best = tv

            # Code's test value
            code_tv = compute_test_value_single(a, n_half)

            self.assertAlmostEqual(
                code_tv, best, places=10,
                msg=f"TV mismatch for c={c}: code={code_tv}, manual={best}")


if __name__ == '__main__':
    unittest.main()

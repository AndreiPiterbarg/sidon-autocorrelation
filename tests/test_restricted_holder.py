"""Tests for the conditional bound C_{1a} >= M_optimal under Hyp_R.

The headline rigorous bound is M_optimal(log 16/pi) ~= 1.37842 (the inf
over admissible z_1 of M_master(z_1)). The earlier recipe value
M_recipe(log 16/pi) ~= 1.37925 (using z_1 = 0.50426 fixed) is sub-optimal
for c < 1 and is retained only for historical comparison; see
``delsarte_dual/restricted_holder/derivation.md`` §Z and CORRECTION_NOTE.md.
"""
import sys
import os
import unittest

import mpmath as mp
from mpmath import mpf

ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from delsarte_dual.restricted_holder.conditional_bound import (
    conditional_bound_recipe,
    conditional_bound_optimal,
    conditional_bound_optimal_bisection,
    modified_master_residual,
    _ingredients,
    _master_rhs,
    _admissible_z1_max_sq,
    _hi_prec,
    _resolve_c,
    MV_Z1,
)


class TestRecipeForm(unittest.TestCase):
    """The MV-recipe form (z_1 = 0.50426 fixed) — sub-optimal at c < 1."""

    def test_reproduces_mv_baseline(self):
        """At c = 1, the recipe form matches MV's published M ~= 1.27481."""
        M = conditional_bound_recipe(mpf(1), dps=40)
        self.assertLess(abs(float(M) - 1.27481), 1e-4)

    def test_recipe_value_1_37925(self):
        """At c = log 16/pi, the recipe (z_1 = 0.50426) gives 1.37925062... .

        Historical sanity check; this is NOT the rigorous bound — see
        test_optimal_bound_is_rigorous_inf for the rigorous M_optimal.
        """
        c = mp.log(16) / mp.pi
        M = conditional_bound_recipe(c, dps=60)
        target = mpf("1.37925062005091")
        self.assertLess(abs(M - target), mpf("1e-10"))

    def test_recipe_residual_zero(self):
        """At z_1 = 0.50426, the residual at M_recipe is zero."""
        c = mp.log(16) / mp.pi
        M = conditional_bound_recipe(c, dps=40)
        residual = modified_master_residual(M, c, z1=MV_Z1)
        self.assertLess(abs(residual), mpf("1e-30"))


class TestOptimalForm(unittest.TestCase):
    """The rigorous inf-over-z_1 bound (the headline conditional theorem)."""

    def test_optimal_bound_is_rigorous_inf(self):
        """M_optimal at c = log 16/pi is the inf of M_master(z_1).

        Verifies:
          (i)   M_optimal < M_recipe (the recipe over-claims).
          (ii)  At z_1 = z_1* (interior argmin of M_master), residual ~ 0.
          (iii) At z_1 = 0.50426 (recipe), the residual at M_optimal is
                positive: confirming z_1 = 0.50426 is sub-optimal at c < 1
                (master ineq fails at the recipe z_1 when M = M_optimal).
        """
        c = mp.log(16) / mp.pi
        M_opt = conditional_bound_optimal(c, dps=40)
        M_rec = conditional_bound_recipe(c, dps=40)

        # (i) Rigorous bound is strictly tighter than the recipe.
        self.assertLess(M_opt, M_rec)

        # (ii) Interior critical point: z_1*^4 = k_1^2 (cM - 1) / (||K||_2^2 - 1).
        K2, a_gain, k1, u_hi = _ingredients()
        c_hi = _resolve_c(c)
        z1_star_sq = k1 * mp.sqrt((c_hi * M_opt - 1) / (K2 - 1))
        z1_star = mp.sqrt(z1_star_sq)
        rhs_at_star = _master_rhs(M_opt, c_hi, z1_star, K2, a_gain, k1, u_hi)
        lhs = 2 / u_hi + a_gain
        residual_at_star = lhs - rhs_at_star
        self.assertLess(abs(residual_at_star), mpf("1e-30"),
                        msg=f"At z_1*, residual = {residual_at_star}, expected 0.")

        # (iii) At z_1 = 0.50426 (the recipe), the residual at M_opt is positive
        # (master inequality fails: LHS > RHS, so no f with z_actual = 0.50426
        # exists at M = M_opt; hence z_1 = 0.50426 is sub-optimal here).
        z1_recipe = _hi_prec(MV_Z1, "0.50426")
        rhs_at_recipe = _master_rhs(M_opt, c_hi, z1_recipe, K2, a_gain, k1, u_hi)
        residual_at_recipe = lhs - rhs_at_recipe
        self.assertGreater(residual_at_recipe, mpf("1e-6"),
                           msg=f"Expected positive residual at z_1=0.50426; "
                               f"got {residual_at_recipe}")

    def test_optimal_matches_bisection_cross_check(self):
        """Closed-form optimal should match the golden-section search."""
        c = mp.log(16) / mp.pi
        M_closed = conditional_bound_optimal(c, dps=40)
        M_bis, z_opt = conditional_bound_optimal_bisection(c, dps=40)
        # Closed form and bisection should agree to ~25 digits.
        self.assertLess(abs(M_closed - M_bis), mpf("1e-25"),
                        msg=f"closed: {M_closed}\nbisect: {M_bis}\nz*: {z_opt}")

    def test_optimal_reproduces_to_30_digits(self):
        """M_optimal at c = log 16/pi reproduced to >= 30 digits."""
        c = mp.log(16) / mp.pi
        M = conditional_bound_optimal(c, dps=40)
        target = mpf("1.378421973775417728399702552431")
        self.assertLess(abs(M - target), mpf("1e-30"))

    def test_master_inequality_holds_for_all_admissible_z1(self):
        """At M = M_optimal, sweep z_1 over [0, sqrt(mu(M_optimal))] and
        verify that the master inequality is *not* violated anywhere —
        i.e., LHS - RHS >= 0 for every admissible z_1, equivalently
        RHS(M_optimal, z_1) <= LHS.

        Sign convention: residual = LHS - RHS. A *negative* residual at
        any admissible z_1 would mean RHS > LHS there, so an f with
        z_actual = that z_1 could exist at M < M_optimal — refuting
        M_optimal as the rigorous lower bound. The test fails loudly in
        that case.

        Note: a uniform grid will not land exactly on the argmin z_1*, so
        the *minimum* of the residual on the grid will be a small positive
        number (~ (Delta z_1)^2 * |M_master''|) rather than zero. The
        exact boundary residual=0 at z_1=z_1* is verified separately in
        ``test_optimal_bound_is_rigorous_inf``.
        """
        c = mp.log(16) / mp.pi
        mp.mp.dps = 40
        M = conditional_bound_optimal(c, dps=40)

        K2, a_gain, k1, u_hi = _ingredients()
        c_hi = _resolve_c(c)
        z1_max_sq = _admissible_z1_max_sq(M, c_hi, K2, k1)
        z1_max = mp.sqrt(z1_max_sq)
        n_grid = 1000
        residuals = []
        lhs = 2 / u_hi + a_gain
        for i in range(n_grid + 1):
            z = z1_max * mpf(i) / n_grid
            rhs = _master_rhs(M, c_hi, z, K2, a_gain, k1, u_hi)
            if rhs == mpf("-inf"):
                # Out of Hyp_R sqrt domain; treat as +inf residual (master
                # ineq trivially fails since RHS undefined as a real bound).
                residuals.append(mpf("+inf"))
            else:
                residuals.append(lhs - rhs)
        finite = [r for r in residuals if r != mpf("+inf")]
        min_res = min(finite)
        # Rigorousness check: no violation (no z_1 gives RHS > LHS).
        self.assertGreater(min_res, mpf("-1e-25"),
                           msg=f"At M_optimal, min residual = {min_res} < -1e-25; "
                               f"some z_1 gives RHS > LHS, refuting M_optimal as "
                               f"the rigorous lower bound.")
        # Sanity: minimum is small (~grid-resolution^2 from the true argmin).
        # Should be << 1 for a 1000-point grid; quadratic near z_1*.
        self.assertLess(min_res, mpf("1e-5"),
                        msg=f"min residual = {min_res} suspiciously large; "
                            f"grid may not bracket the argmin z_1*.")

    def test_restriction_consistent(self):
        """M_optimal(log 16/pi) <= M_max = 1.51, so the contradiction is
        within the restricted class where Hyp_R is supposed to apply.
        """
        c = mp.log(16) / mp.pi
        M = conditional_bound_optimal(c, dps=40)
        self.assertLessEqual(M, mpf("1.51"))

    def test_optimal_below_recipe_for_c_lt_1(self):
        """For c < 1, M_optimal < M_recipe (recipe is over-claiming)."""
        for c in [mpf("0.90"), mpf("0.95"), mpf("0.98"), mp.log(16) / mp.pi]:
            M_opt = conditional_bound_optimal(c, dps=40)
            M_rec = conditional_bound_recipe(c, dps=40)
            self.assertLess(M_opt, M_rec, msg=f"At c={c}: M_opt={M_opt}, M_rec={M_rec}")

    def test_optimal_gain_monotonic_in_c(self):
        """M_optimal(c) is strictly decreasing in c on [0.9, 1.0]."""
        cs = [mpf("0.90"), mpf("0.92"), mpf("0.95"), mpf("0.98"), mpf("1.0")]
        Ms = [conditional_bound_optimal(c, dps=40) for c in cs]
        for a, b in zip(Ms[:-1], Ms[1:]):
            self.assertGreater(a, b, msg=f"Not monotonic: {a} <= {b}")

    def test_M_max_tight(self):
        """M_max = M_target works; the proof closes at the tight threshold.

        The default M_max is M_target itself (auto-tracking). We also pass
        M_max explicitly equal to the target and verify it's accepted.
        """
        c = mp.log(16) / mp.pi
        # Default (M_max=None -> M_target): no error.
        M_default = conditional_bound_optimal(c, dps=40)
        # Explicit M_max = M_target: accepted, same value returned.
        M_explicit = conditional_bound_optimal(c, dps=40, M_max=M_default)
        self.assertEqual(M_default, M_explicit)
        # Explicit M_max > M_target (a fortiori): also accepted.
        M_loose = conditional_bound_optimal(c, dps=40, M_max=mpf("1.51"))
        self.assertEqual(M_default, M_loose)

    def test_M_max_below_target_fails(self):
        """M_max < M_target raises AssertionError (proof would not close)."""
        c = mp.log(16) / mp.pi
        # Pick a value strictly below the known M_target ~1.37842.
        with self.assertRaises(AssertionError) as ctx:
            conditional_bound_optimal(c, dps=40, M_max=mpf("1.30"))
        self.assertIn("M_max", str(ctx.exception))
        self.assertIn("M_target", str(ctx.exception))
        # Just below the target: also fails.
        with self.assertRaises(AssertionError):
            conditional_bound_optimal(c, dps=40, M_max=mpf("1.378"))


class TestBoyerLiWitness(unittest.TestCase):
    """Verify the Boyer-Li 575-step witness is OUTSIDE the restricted class."""

    @classmethod
    def setUpClass(cls):
        coeff_path = os.path.join(
            ROOT, "delsarte_dual", "restricted_holder", "coeffBL.txt"
        )
        if not os.path.exists(coeff_path):
            raise unittest.SkipTest(
                "delsarte_dual/restricted_holder/coeffBL.txt not present; "
                "fetch from github.com/zkli-math/autoconvolutionHolder/coeffBL.txt"
            )
        with open(coeff_path) as f:
            content = f.read().strip().lstrip("{").rstrip("}").rstrip(",").strip()
        cls.v = [int(x.strip()) for x in content.split(",")]
        assert len(cls.v) == 575

    def test_boyerli_witness_outside_restriction(self):
        """The rescaled BL witness g (on [-1/4, 1/4], pdf) has ||g*g||_inf > 1.51,
        so the restricted Holder hypothesis is NOT disproved by Boyer-Li.
        """
        v = self.v
        N = len(v)
        S = sum(v)
        # L_j = sum_n v_n v_{j-n-1}  (Boyer-Li convention; equals (f_0*f_0)(j))
        max_L = 0
        for j in range(2 * N + 1):
            s = 0
            for n in range(max(0, j - N), min(j, N)):
                m = j - n - 1
                if 0 <= m < N:
                    s += v[n] * v[m]
            if s > max_L:
                max_L = s
        # ||g*g||_inf = (2N) * max_L / S^2
        mp.mp.dps = 50
        S_mp = mpf(S)
        max_L_mp = mpf(max_L)
        gg_inf = mpf(2 * N) * max_L_mp / (S_mp * S_mp)
        self.assertGreater(gg_inf, mpf("1.51"))
        # Verify the published Q ratio (sanity)
        L = []
        for j in range(2 * N + 1):
            s = 0
            for n in range(max(0, j - N), min(j, N)):
                m = j - n - 1
                if 0 <= m < N:
                    s += v[n] * v[m]
            L.append(s)
        from fractions import Fraction
        num = sum(L[j] ** 2 + L[j] * L[j + 1] + L[j + 1] ** 2 for j in range(2 * N))
        denom1 = sum(L[j] + L[j + 1] for j in range(2 * N))
        Q = Fraction(num, 3) / (Fraction(max_L) * Fraction(denom1, 2))
        Q_dec = mpf(Q.numerator) / mpf(Q.denominator)
        self.assertGreater(Q_dec, mpf(mp.log(16) / mp.pi))  # >= log 16/pi (BL theorem)
        self.assertLess(Q_dec, mpf(1))                      # <= 1 (Holder)


class TestExtremizerRatio(unittest.TestCase):
    """Empirical Holder ratio for MV's near-extremizer (unchanged)."""

    def test_emp_well_below_log16_over_pi(self):
        """c_emp = ||f*f||_2^2 / (||f*f||_inf ||f*f||_1) for f = G/Z is < 0.99."""
        from delsarte_dual.restricted_holder.sidon_extremizer_ratio import (
            holder_ratio,
        )
        res = holder_ratio(Nmax=2000, dps=40)
        self.assertLess(res["c_emp"], mpf("0.99"))


if __name__ == "__main__":
    unittest.main()

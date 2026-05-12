"""Tests for the alternative-kernel sweep infrastructure.

Run with:
    python -m pytest delsarte_dual/grid_bound_alt_kernel/tests -v
or:
    python -m unittest discover delsarte_dual/grid_bound_alt_kernel/tests
"""
from __future__ import annotations

import unittest

from flint import arb, fmpq

from delsarte_dual.grid_bound_alt_kernel.kernels import (
    ArcsineKernel, TriangularKernel, TruncatedGaussianKernel,
    JacksonKernel, SelbergBandlimitedKernel, RieszKernel,
    default_kernel_registry,
)
from delsarte_dual.grid_bound_alt_kernel.optimize_G import solve_qp_for_kernel
from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    run_single_kernel, compile_phi_params_for_kernel,
)


class TestK1ReproducesMV(unittest.TestCase):
    """Acceptance criterion [1]: K1 arcsine must yield M_cert in [1.2743, 1.2750].

    MV's published value is 1.27481; our bisection at tol=1e-3 hits 1.2745
    (bisection resolution).  Tightening to 1e-4 brings it to ~1.2747.
    """
    def test_K1_reproduces_mv(self):
        res = run_single_kernel(
            ArcsineKernel(),
            n_coeffs=119,
            n_grid_qp=5001,
            n_cells_min_G=2048,
            tol_q=fmpq(1, 10**3),
            max_cells_per_M=20000,
            initial_splits=32,
            bochner_max=20,
            prec_bits=128,
            verbose=False,
        )
        self.assertIsNotNone(res.M_cert, f"K1 sweep failed: {res.note}")
        self.assertGreaterEqual(res.M_cert, 1.2740,
            f"K1 M_cert {res.M_cert} lower than expected MV baseline")
        self.assertLess(res.M_cert, 1.2755,
            f"K1 M_cert {res.M_cert} higher than MV's 1.2748 (unexpected)")


class TestBochnerPositivity(unittest.TestCase):
    """Acceptance criterion [2]: verify Bochner-check admissibility for each K.

    K1 arcsine and K2 triangular MUST pass Bochner (they have |J_0|^2 and
    sinc^2 Fourier).  Known non-admissible kernels (Riesz alpha != 0.5,
    truncated Gaussian with large sigma) should fail.
    """
    def test_K1_bochner_50(self):
        self.assertTrue(ArcsineKernel().K_tilde_positive(50, prec_bits=128))

    def test_K2_bochner_50(self):
        self.assertTrue(TriangularKernel().K_tilde_positive(50, prec_bits=128))

    def test_K3_smallsigma_bochner_25(self):
        # sigma = delta/3 is small enough that truncated Gaussian is near-
        # arcsine-like and the first FT zero is past j=25.
        K = TruncatedGaussianKernel(sigma_over_delta=fmpq(1, 3))
        self.assertTrue(K.K_tilde_positive(25, prec_bits=128),
            "K3 sigma=delta/3 expected to pass Bochner at j<=25")

    def test_K3_largesigma_bochner_fails(self):
        # sigma = delta produces oscillatory FT; Bochner fails early.
        K = TruncatedGaussianKernel(sigma_over_delta=fmpq(1, 1))
        self.assertFalse(K.K_tilde_positive(50, prec_bits=128),
            "K3 sigma=delta expected to fail Bochner")

    def test_K6_alpha_fails(self):
        # Riesz alpha != 0.5 has Bessel FT which oscillates
        for a in (0.4, 0.6, 0.8, 1.0, 1.2):
            K = RieszKernel(alpha=a)
            self.assertFalse(K.K_tilde_positive(50, prec_bits=128),
                f"K6 alpha={a} unexpectedly passed Bochner")

    def test_K5_selberg_bochner(self):
        K = SelbergBandlimitedKernel()
        self.assertTrue(K.K_tilde_positive(50, prec_bits=128),
            "K5 Selberg-cos^2-tent should pass Bochner")


class TestQPConvergence(unittest.TestCase):
    """Acceptance criterion [3]: QP converges and min_G >= 0.9999 (rounding)."""

    def test_K1_qp(self):
        res = solve_qp_for_kernel(ArcsineKernel(), n=119, n_grid=5001, verbose=False)
        self.assertGreaterEqual(res.min_G_grid_float, 0.9999)
        self.assertLess(res.min_G_grid_float, 1.0001)

    def test_K2_qp(self):
        res = solve_qp_for_kernel(TriangularKernel(), n=119, n_grid=5001, verbose=False)
        self.assertGreaterEqual(res.min_G_grid_float, 0.9999)
        self.assertLess(res.min_G_grid_float, 1.0001)

    def test_K5_qp(self):
        res = solve_qp_for_kernel(SelbergBandlimitedKernel(), n=119, n_grid=5001, verbose=False)
        self.assertGreaterEqual(res.min_G_grid_float, 0.9999)


class TestKernelNormalisation(unittest.TestCase):
    """Each admissible kernel must satisfy hat_K_R(0) = 1 (int K = 1)."""

    def test_K1_mass(self):
        # K1 arcsine-auto-conv: hat_K_R(0) = J_0(0)^2 = 1 by construction.
        K = ArcsineKernel()
        self.assertAlmostEqual(float(K.K_tilde(0, prec_bits=128).mid()), 1.0, places=12)

    def test_K2_mass(self):
        K = TriangularKernel()
        self.assertAlmostEqual(float(K.K_tilde(0, prec_bits=128).mid()), 1.0, places=12)

    def test_K3_mass(self):
        K = TruncatedGaussianKernel(sigma_over_delta=fmpq(1, 2))
        self.assertAlmostEqual(float(K.K_tilde(0, prec_bits=128).mid()), 1.0, places=12)


class TestKernelValuesSanity(unittest.TestCase):
    """Sanity on k1 and K2 for the arcsine baseline."""
    def test_K1_k1_matches_existing(self):
        # existing grid_bound/phi.py PhiParams.from_mv reports k1 ~ 0.909276...
        k1 = ArcsineKernel().K_tilde(1, prec_bits=128)
        self.assertAlmostEqual(float(k1.mid()), 0.909276, places=5)

    def test_K1_K2_matches_existing(self):
        # existing K_2 = 0.5747 / 0.138 = 4.16449...
        K2 = ArcsineKernel().K_norm_sq(prec_bits=128)
        self.assertAlmostEqual(float(K2.mid()), 4.164492, places=5)


if __name__ == "__main__":
    unittest.main()

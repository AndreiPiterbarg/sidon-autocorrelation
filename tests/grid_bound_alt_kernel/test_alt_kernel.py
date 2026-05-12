"""Tests for the multi-scale arcsine production pipeline.

Run from the repository root with

    ``pytest tests/grid_bound_alt_kernel/``.
"""
from __future__ import annotations

import unittest

from flint import arb, fmpq

from delsarte_dual.grid_bound_alt_kernel.kernels import (
    ArcsineKernel,
    MultiScaleArcsineKernel,
)
from delsarte_dual.grid_bound_alt_kernel.optimize_G import solve_qp_for_kernel
from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    production_kernel,
    run_single_kernel,
)


class TestArcsineBaseline(unittest.TestCase):
    """The single-scale arcsine kernel must reproduce the Matolcsi-Vinuesa
    lower bound ``1.27481``.
    """

    def test_reproduces_mv_baseline(self):
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
        self.assertIsNotNone(
            res.M_cert_float, f"single-scale sweep failed: {res.note}"
        )
        self.assertGreaterEqual(
            res.M_cert_float, 1.2740,
            f"M_cert {res.M_cert_float} below the MV baseline",
        )
        self.assertLess(
            res.M_cert_float, 1.2755,
            f"M_cert {res.M_cert_float} above the MV ceiling (unexpected)",
        )


class TestBochnerPositivity(unittest.TestCase):
    """Both production kernels must pass the Bochner check (admissibility
    property K4).
    """

    def test_arcsine_bochner(self):
        self.assertTrue(
            ArcsineKernel().K_tilde_positive(50, prec_bits=128)
        )

    def test_multiscale_bochner(self):
        K = production_kernel()
        self.assertTrue(K.K_tilde_positive(50, prec_bits=128))


class TestQPConvergence(unittest.TestCase):
    """The QP must converge with min G ≈ 1 on the grid."""

    def test_arcsine_qp(self):
        res = solve_qp_for_kernel(
            ArcsineKernel(), n=119, n_grid=5001, verbose=False
        )
        self.assertGreaterEqual(res.min_G_grid_float, 0.9999)
        self.assertLess(res.min_G_grid_float, 1.0001)

    def test_multiscale_qp(self):
        res = solve_qp_for_kernel(
            production_kernel(), n=200, n_grid=2001, verbose=False
        )
        self.assertGreaterEqual(res.min_G_grid_float, 0.9985)


class TestKernelNormalisation(unittest.TestCase):
    """Each admissible kernel must satisfy ``hat K(0) = 1`` (mass one)."""

    def test_arcsine_mass(self):
        K = ArcsineKernel()
        self.assertAlmostEqual(
            float(K.K_tilde(0, prec_bits=128).mid()), 1.0, places=12
        )

    def test_multiscale_mass(self):
        K = production_kernel()
        self.assertAlmostEqual(
            float(K.K_tilde(0, prec_bits=128).mid()), 1.0, places=12
        )


class TestArcsineAnchors(unittest.TestCase):
    """Single-scale arcsine ``k_1`` and ``K_2`` at the MV parameters."""

    def test_k1(self):
        k1 = ArcsineKernel().K_tilde(1, prec_bits=128)
        self.assertAlmostEqual(float(k1.mid()), 0.909276, places=5)

    def test_K2(self):
        K2 = ArcsineKernel().K_norm_sq(prec_bits=128)
        self.assertAlmostEqual(float(K2.mid()), 4.164492, places=5)


class TestMultiScaleAnchors(unittest.TestCase):
    """Multi-scale 3-arcsine production anchors at writeup parameters.

    Compares to the reference values quoted in the writeup and in
    ``lean/Sidon/MultiScale.lean``.
    """

    def test_multiscale_k1(self):
        K = production_kernel()
        k1 = K.K_tilde(1, prec_bits=192)
        self.assertAlmostEqual(float(k1.mid()), 0.92124659, places=7)

    def test_multiscale_lambdas_sum_to_one(self):
        K = production_kernel()
        self.assertEqual(sum(K.lambdas, fmpq(0)), fmpq(1))


if __name__ == "__main__":
    unittest.main()

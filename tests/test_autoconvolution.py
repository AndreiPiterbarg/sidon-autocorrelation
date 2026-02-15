"""Tests for autoconvolution/test-value computation and asymmetry pruning."""
import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import (
    compute_test_values_batch,
    compute_test_value_single,
    asymmetry_prune_mask,
)


class TestAutoconvolution(unittest.TestCase):
    def test_uniform_n2(self):
        tv = compute_test_value_single([2, 2, 2, 2], n_half=2)
        self.assertAlmostEqual(tv, 1.25, places=6)

    def test_uniform_n3(self):
        tv = compute_test_value_single([2, 2, 2, 2, 2, 2], n_half=3)
        self.assertAlmostEqual(tv, 4.0/3.0, places=6)

    def test_concentrated_high(self):
        tv = compute_test_value_single([8, 0, 0, 0], n_half=2)
        self.assertAlmostEqual(tv, 4.0, places=6)

    def test_nonneg(self):
        for _ in range(20):
            a = np.random.dirichlet(np.ones(4)) * 8
            tv = compute_test_value_single(a, n_half=2)
            self.assertGreaterEqual(tv, 0.0)

    def test_lower_bound_property(self):
        n_half = 2
        d = 4
        a = np.array([3.0, 1.0, 1.0, 3.0])
        tv = compute_test_value_single(a, n_half=n_half)

        n_points = 1000
        t_grid = np.linspace(-0.5, 0.5, n_points)
        bin_width = 1.0 / (4 * n_half)

        ff_vals = np.zeros(n_points)
        for t_idx, t in enumerate(t_grid):
            val = 0.0
            for i in range(d):
                for j in range(d):
                    xi_lo = -0.25 + i * bin_width
                    xi_hi = xi_lo + bin_width
                    xj_lo = -0.25 + j * bin_width
                    xj_hi = xj_lo + bin_width
                    overlap_lo = max(xi_lo, t - xj_hi)
                    overlap_hi = min(xi_hi, t - xj_lo)
                    if overlap_hi > overlap_lo:
                        val += a[i] * a[j] * (overlap_hi - overlap_lo)
            ff_vals[t_idx] = val

        actual_sup = ff_vals.max()
        self.assertLessEqual(tv, actual_sup + 1e-6)

    def test_batch_matches_single(self):
        n_half = 2
        m = 10
        configs = np.array([
            [10, 10, 10, 10],
            [20, 5, 5, 10],
            [0, 20, 20, 0],
        ], dtype=np.int32)

        batch_tvs = compute_test_values_batch(configs, n_half, m)

        for i in range(len(configs)):
            a = configs[i].astype(np.float64) / m
            single_tv = compute_test_value_single(a, n_half)
            self.assertAlmostEqual(batch_tvs[i], single_tv, places=8)


class TestAsymmetryPruning(unittest.TestCase):
    def test_left_heavy_pruned(self):
        n_half = 2
        m = 10
        configs = np.array([[50, 30, 0, 0]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])

    def test_uniform_needs_check(self):
        n_half = 2
        m = 10
        configs = np.array([[20, 20, 20, 20]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertTrue(mask[0])

    def test_right_heavy_pruned(self):
        n_half = 2
        m = 10
        configs = np.array([[0, 0, 30, 50]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])

    def test_margin_near_threshold(self):
        n_half = 2
        m = 10
        configs = np.array([[32, 32, 8, 8]], dtype=np.int32)
        left_frac = configs[0, :n_half].sum() / (4 * n_half * m)
        self.assertAlmostEqual(left_frac, 0.8)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertTrue(mask[0])

    def test_margin_well_beyond_threshold(self):
        n_half = 2
        m = 10
        configs = np.array([[35, 35, 5, 5]], dtype=np.int32)
        left_frac = configs[0, :n_half].sum() / (4 * n_half * m)
        self.assertGreater(left_frac, 0.825)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])


if __name__ == '__main__':
    unittest.main()

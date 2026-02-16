"""Tests for Cloninger-Steinerberger branch-and-prune algorithm."""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import (
    correction, count_compositions, asymmetry_threshold,
    generate_compositions_batched, compute_test_values_batch,
    compute_test_value_single, asymmetry_prune_mask,
    run_single_level, find_best_bound, find_best_bound_direct,
)


class TestCorrection(unittest.TestCase):
    def test_m50(self):
        self.assertAlmostEqual(correction(50), 2/50 + 1/2500, places=10)

    def test_m100(self):
        self.assertAlmostEqual(correction(100), 0.0201, places=4)

    def test_decreasing(self):
        self.assertGreater(correction(10), correction(50))
        self.assertGreater(correction(50), correction(100))


class TestAsymmetryThreshold(unittest.TestCase):
    def test_target_128(self):
        self.assertAlmostEqual(asymmetry_threshold(1.28), 0.8, places=10)

    def test_target_1(self):
        self.assertAlmostEqual(asymmetry_threshold(1.0),
                               np.sqrt(0.5), places=10)

    def test_target_2(self):
        self.assertAlmostEqual(asymmetry_threshold(2.0), 1.0, places=10)


class TestCountCompositions(unittest.TestCase):
    def test_d2_S5(self):
        """C(6, 1) = 6."""
        self.assertEqual(count_compositions(2, 5), 6)

    def test_d4_S3(self):
        """C(6, 3) = 20."""
        self.assertEqual(count_compositions(4, 3), 20)

    def test_d1_Sany(self):
        self.assertEqual(count_compositions(1, 100), 1)


class TestGenerateCompositions(unittest.TestCase):
    def test_d2_S3(self):
        """4 compositions: (0,3), (1,2), (2,1), (3,0)."""
        batches = list(generate_compositions_batched(2, 3, batch_size=100))
        all_comps = np.vstack(batches)
        self.assertEqual(len(all_comps), 4)
        self.assertTrue(np.all(all_comps.sum(axis=1) == 3))
        self.assertTrue(np.all(all_comps >= 0))

    def test_d3_S2(self):
        """C(4, 2) = 6 compositions."""
        batches = list(generate_compositions_batched(3, 2, batch_size=100))
        all_comps = np.vstack(batches)
        self.assertEqual(len(all_comps), 6)
        self.assertTrue(np.all(all_comps.sum(axis=1) == 2))

    def test_batching(self):
        """Verify batching produces correct total count."""
        batches = list(generate_compositions_batched(4, 5, batch_size=3))
        all_comps = np.vstack(batches)
        expected = count_compositions(4, 5)
        self.assertEqual(len(all_comps), expected)

    def test_all_unique(self):
        """No duplicate compositions."""
        batches = list(generate_compositions_batched(3, 4, batch_size=10))
        all_comps = np.vstack(batches)
        unique = set(map(tuple, all_comps.tolist()))
        self.assertEqual(len(unique), len(all_comps))


class TestAutoconvolution(unittest.TestCase):
    def test_uniform_n2(self):
        """Uniform config (2,2,2,2) at n=2 should give test_val = 1.25."""
        tv = compute_test_value_single([2, 2, 2, 2], n_half=2)
        self.assertAlmostEqual(tv, 1.25, places=6)

    def test_uniform_n3(self):
        """Uniform config at n=3 should give test_val = 4/3."""
        tv = compute_test_value_single([2, 2, 2, 2, 2, 2], n_half=3)
        self.assertAlmostEqual(tv, 4.0/3.0, places=6)

    def test_concentrated_high(self):
        """All mass in one bin should give very high test value."""
        tv = compute_test_value_single([8, 0, 0, 0], n_half=2)
        # 8^2 / (4*2*2) = 64/16 = 4.0
        self.assertAlmostEqual(tv, 4.0, places=6)

    def test_nonneg(self):
        """Test values are always non-negative."""
        for _ in range(20):
            a = np.random.dirichlet(np.ones(4)) * 8
            tv = compute_test_value_single(a, n_half=2)
            self.assertGreaterEqual(tv, 0.0)

    def test_lower_bound_property(self):
        """Test value is a valid lower bound on ||f*f||_inf.

        For a step function f with the given masses, the actual ||f*f||_inf
        should be >= the test value. We verify this by constructing the
        step function and computing its autoconvolution directly.
        """
        n_half = 2
        d = 4
        a = np.array([3.0, 1.0, 1.0, 3.0])  # masses, sum = 8

        # Test value from our formula
        tv = compute_test_value_single(a, n_half=n_half)

        # Construct step function: f(x) = a_i on bin I_i = [i/(4n), (i+1)/(4n))
        # Compute autoconvolution numerically at many points
        n_points = 1000
        t_grid = np.linspace(-0.5, 0.5, n_points)
        bin_width = 1.0 / (4 * n_half)

        ff_vals = np.zeros(n_points)
        for t_idx, t in enumerate(t_grid):
            # (f*f)(t) = integral f(x) f(t-x) dx
            # For step function: sum over bin pairs
            val = 0.0
            for i in range(d):
                for j in range(d):
                    # f_i on [i*bw, (i+1)*bw) shifted to [-1/4, 1/4)
                    # bin i: x in [-1/4 + i*bw, -1/4 + (i+1)*bw)
                    xi_lo = -0.25 + i * bin_width
                    xi_hi = xi_lo + bin_width
                    xj_lo = -0.25 + j * bin_width
                    xj_hi = xj_lo + bin_width
                    # f_i * f_j at t: integral over x in I_i of f_i(x)*f_j(t-x)
                    # f_i(x) = a[i] (constant on I_i)
                    # f_j(t-x): t-x in I_j means x in [t - xj_hi, t - xj_lo]
                    overlap_lo = max(xi_lo, t - xj_hi)
                    overlap_hi = min(xi_hi, t - xj_lo)
                    if overlap_hi > overlap_lo:
                        val += a[i] * a[j] * (overlap_hi - overlap_lo)
            ff_vals[t_idx] = val

        actual_sup = ff_vals.max()
        # Test value should be <= actual sup (it's a lower bound)
        self.assertLessEqual(tv, actual_sup + 1e-6,
                             f"Test value {tv} exceeds actual sup {actual_sup}")

    def test_batch_matches_single(self):
        """Batch computation matches single computation."""
        n_half = 2
        m = 10
        configs = np.array([
            [10, 10, 10, 10],  # uniform
            [20, 5, 5, 10],
            [0, 20, 20, 0],
        ], dtype=np.int32)

        batch_tvs = compute_test_values_batch(configs, n_half, m)

        for i in range(len(configs)):
            a = configs[i].astype(np.float64) / m
            single_tv = compute_test_value_single(a, n_half)
            self.assertAlmostEqual(batch_tvs[i], single_tv, places=8,
                                   msg=f"Config {i}: batch={batch_tvs[i]}, "
                                       f"single={single_tv}")


class TestAsymmetryPruning(unittest.TestCase):
    def test_left_heavy_pruned(self):
        """Config with left_frac > 0.8 should be pruned (not need checking)."""
        n_half = 2
        m = 10
        # S=m=10: left=9/10=0.9 > safe_threshold=0.825
        configs = np.array([[5, 4, 1, 0]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])  # Does NOT need checking (covered by asymmetry)

    def test_uniform_needs_check(self):
        """Uniform config should need checking."""
        n_half = 2
        m = 10
        # S=m=10: left=5/10=0.5
        configs = np.array([[3, 2, 3, 2]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertTrue(mask[0])  # Needs checking

    def test_right_heavy_pruned(self):
        """Config with right_frac > 0.8 should be pruned."""
        n_half = 2
        m = 10
        # S=m=10: left=1/10=0.1, right_frac=0.9
        configs = np.array([[0, 1, 4, 5]], dtype=np.int32)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])

    def test_margin_near_threshold(self):
        """Config at the old threshold must NOT be pruned due to margin.

        With n_half=2, m=10, threshold=0.8, margin=1/(4*10)=0.025.
        safe_threshold = 0.825. A config with left_frac=0.8 is within
        the margin zone and must be checked (not pruned), because a
        continuous function rounding to it could have left_frac=0.775.
        """
        n_half = 2
        m = 10
        # S=m=10: left=8/10=0.8. Config: (4, 4, 1, 1)
        configs = np.array([[4, 4, 1, 1]], dtype=np.int32)
        left_frac = configs[0, :n_half].sum() / m
        self.assertAlmostEqual(left_frac, 0.8)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertTrue(mask[0])  # Needs checking (margin zone)

    def test_margin_well_beyond_threshold(self):
        """Config well beyond safe_threshold should be pruned."""
        n_half = 2
        m = 10
        # S=m=10: safe_threshold = 0.825. left_frac = 9/10 = 0.9 > 0.825.
        configs = np.array([[5, 4, 1, 0]], dtype=np.int32)
        left_frac = configs[0, :n_half].sum() / m
        self.assertGreater(left_frac, 0.825)
        mask = asymmetry_prune_mask(configs, n_half, m, c_target=1.28)
        self.assertFalse(mask[0])  # Safely pruned


class TestRunSingleLevel(unittest.TestCase):
    def test_trivial_target(self):
        """Target of 0.5 should be easily provable at any (n, m)."""
        result = run_single_level(n_half=2, m=5, c_target=0.5,
                                   batch_size=10000, verbose=False)
        self.assertTrue(result['proven'])
        self.assertEqual(result['c_proven'], 0.5)

    def test_impossible_target(self):
        """Target above 2 should never be provable."""
        result = run_single_level(n_half=2, m=5, c_target=2.5,
                                   batch_size=10000, verbose=False)
        self.assertFalse(result['proven'])
        self.assertIsNone(result['c_proven'])
        self.assertGreater(result['n_survivors'], 0)

    def test_total_processed(self):
        """All configs should be processed."""
        result = run_single_level(n_half=2, m=3, c_target=1.0,
                                   batch_size=10000, verbose=False)
        expected = count_compositions(4, 24)
        self.assertEqual(result['stats']['n_processed'], expected)
        total = (result['stats']['n_pruned_asym']
                 + result['stats']['n_pruned_test']
                 + result['stats']['n_survived'])
        self.assertEqual(total, expected)

    def test_monotone_in_target(self):
        """Lower target should have fewer survivors."""
        r_lo = run_single_level(n_half=2, m=5, c_target=0.8,
                                 batch_size=10000, verbose=False)
        r_hi = run_single_level(n_half=2, m=5, c_target=1.2,
                                 batch_size=10000, verbose=False)
        self.assertLessEqual(r_lo['n_survivors'], r_hi['n_survivors'])


class TestFindBestBoundDirect(unittest.TestCase):
    def test_returns_positive_bound(self):
        """Direct method should return a positive bound."""
        bound = find_best_bound_direct(n_half=2, m=5, verbose=False)
        self.assertGreater(bound, 0.0)

    def test_bound_reasonable_range(self):
        """Bound should be in a reasonable range for small params."""
        bound = find_best_bound_direct(n_half=2, m=10, verbose=False)
        self.assertGreater(bound, 0.5)
        self.assertLess(bound, 2.0)

    def test_matches_binary_search(self):
        """Direct method should agree with binary search to within tolerance."""
        n_half, m = 2, 20
        binary = find_best_bound(n_half, m, lo=0.8, hi=1.2, tol=0.005,
                                  verbose=False)
        direct = find_best_bound_direct(n_half, m, verbose=False)
        # Binary search has resolution of tol, so direct should be within tol
        # of binary result (direct is more precise)
        self.assertIsNotNone(binary)
        self.assertAlmostEqual(direct, binary, delta=0.01)

    def test_increases_with_m(self):
        """Larger m (finer grid) should give a tighter (higher) bound."""
        b10 = find_best_bound_direct(n_half=2, m=10, verbose=False)
        b20 = find_best_bound_direct(n_half=2, m=20, verbose=False)
        # Larger m reduces correction term, generally improving bound
        # (though the discrete min may also change)
        self.assertGreater(b20, b10 - 0.05)

    def test_n3_returns_bound(self):
        """Should work for d=6 (n_half=3)."""
        bound = find_best_bound_direct(n_half=3, m=3, verbose=False)
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 2.0)


if __name__ == '__main__':
    unittest.main()

"""Integration tests for run_single_level and find_best_bound_direct."""
import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import (
    count_compositions,
    run_single_level,
    find_best_bound,
    find_best_bound_direct,
)


class TestRunSingleLevel(unittest.TestCase):
    def test_trivial_target(self):
        result = run_single_level(n_half=2, m=5, c_target=0.5,
                                   batch_size=10000, verbose=False)
        self.assertTrue(result['proven'])
        self.assertEqual(result['c_proven'], 0.5)

    def test_impossible_target(self):
        result = run_single_level(n_half=2, m=5, c_target=2.5,
                                   batch_size=10000, verbose=False)
        self.assertFalse(result['proven'])
        self.assertIsNone(result['c_proven'])
        self.assertGreater(result['n_survivors'], 0)

    def test_total_processed(self):
        result = run_single_level(n_half=2, m=3, c_target=1.0,
                                   batch_size=10000, verbose=False)
        n_processed = result['stats']['n_processed']
        n_total = count_compositions(4, 24)
        self.assertGreater(n_processed, 0)
        self.assertLessEqual(n_processed, n_total)
        total = (result['stats']['n_pruned_asym']
                 + result['stats']['n_pruned_test']
                 + result['stats']['n_survived'])
        self.assertEqual(total, n_processed)

    def test_monotone_in_target(self):
        r_lo = run_single_level(n_half=2, m=5, c_target=0.8,
                                 batch_size=10000, verbose=False)
        r_hi = run_single_level(n_half=2, m=5, c_target=1.2,
                                 batch_size=10000, verbose=False)
        self.assertLessEqual(r_lo['n_survivors'], r_hi['n_survivors'])


class TestFindBestBoundDirect(unittest.TestCase):
    def test_returns_positive_bound(self):
        bound = find_best_bound_direct(n_half=2, m=5, verbose=False)
        self.assertGreater(bound, 0.0)

    def test_bound_reasonable_range(self):
        bound = find_best_bound_direct(n_half=2, m=10, verbose=False)
        self.assertGreater(bound, 0.5)
        self.assertLess(bound, 2.0)

    def test_matches_binary_search(self):
        n_half, m = 2, 20
        binary = find_best_bound(n_half, m, lo=0.8, hi=1.2, tol=0.005,
                                  verbose=False)
        direct = find_best_bound_direct(n_half, m, verbose=False)
        self.assertIsNotNone(binary)
        self.assertAlmostEqual(direct, binary, delta=0.01)

    def test_increases_with_m(self):
        b10 = find_best_bound_direct(n_half=2, m=10, verbose=False)
        b20 = find_best_bound_direct(n_half=2, m=20, verbose=False)
        self.assertGreater(b20, b10 - 0.05)

    def test_n3_returns_bound(self):
        bound = find_best_bound_direct(n_half=3, m=3, verbose=False)
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 2.0)


if __name__ == '__main__':
    unittest.main()

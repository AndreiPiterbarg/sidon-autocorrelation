"""Heavy benchmarks at production scales where optimization matters.

Two distinct bottleneck profiles at realistic parameter ranges:
  d=4 large-m:  n=2, m=400  --  5.5B configs.  Bottleneck: enumeration breadth.
  d=6 higher-d: n=3, m=16   --  2.3B configs.  Bottleneck: d^2 conv inner loop.
"""
import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import (
    count_compositions,
    compute_test_values_batch,
    compute_test_value_single,
    run_single_level,
    find_best_bound_direct,
)


class TestHeavy(unittest.TestCase):

    # -- d=4 production scale (n=2, m=400, 5.5B configs) --

    def test_run_single_level_n2_m400(self):
        """n=2, m=400: 5.5B configs, target=0.9 should be proven."""
        n_half, m, target = 2, 400, 0.9
        n_total = count_compositions(2 * n_half, 4 * n_half * m)
        self.assertEqual(n_total, 5_471_579_201)
        result = run_single_level(n_half, m, target,
                                  batch_size=100000, verbose=False)
        self.assertTrue(result['proven'])
        self.assertEqual(result['n_survivors'], 0)

    def test_direct_bound_n2_m400(self):
        """n=2, m=400: single-pass direct bound ~1.087."""
        bound = find_best_bound_direct(n_half=2, m=400, verbose=False)
        self.assertGreater(bound, 1.0)
        self.assertLess(bound, 1.15)

    # -- d=6 production scale (n=3, m=16, 2.3B configs) --

    def test_run_single_level_n3_m16(self):
        """n=3, m=16: 2.3B configs at d=6, target=0.9 should be proven."""
        n_half, m, target = 3, 16, 0.9
        n_total = count_compositions(2 * n_half, 4 * n_half * m)
        self.assertEqual(n_total, 2_349_279_569)
        result = run_single_level(n_half, m, target,
                                  batch_size=100000, verbose=False)
        self.assertTrue(result['proven'])
        self.assertEqual(result['n_survivors'], 0)

    def test_direct_bound_n3_m16(self):
        """n=3, m=16: d=6 single-pass bound ~1.04."""
        bound = find_best_bound_direct(n_half=3, m=16, verbose=False)
        self.assertGreater(bound, 0.9)
        self.assertLess(bound, 1.2)

    # -- Correctness at production scale --

    def test_batch_vs_single_at_scale(self):
        """Batch and single test values agree at n=2, m=400."""
        n_half, m = 2, 400
        d = 2 * n_half
        S = 4 * n_half * m
        rng = np.random.RandomState(42)
        configs = [
            np.array([S // d] * d, dtype=np.int32),
            np.array([S, 0, 0, 0], dtype=np.int32),
            np.array([0, S // 2, S // 2, 0], dtype=np.int32),
        ]
        for _ in range(10):
            parts = rng.multinomial(S, np.ones(d) / d)
            configs.append(parts.astype(np.int32))
        batch = np.array(configs, dtype=np.int32)

        batch_tvs = compute_test_values_batch(batch, n_half, m)
        for i in range(len(batch)):
            a = batch[i].astype(np.float64) / m
            single_tv = compute_test_value_single(a, n_half)
            self.assertAlmostEqual(
                batch_tvs[i], single_tv, places=8,
                msg=f"Config {i}: batch={batch_tvs[i]}, single={single_tv}",
            )


if __name__ == '__main__':
    unittest.main()

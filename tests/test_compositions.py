"""Tests for composition generators and reversal symmetry."""
import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import (
    count_compositions,
    generate_compositions_batched,
    generate_canonical_compositions_batched,
    _canonical_mask,
    compute_test_value_single,
)


class TestGenerateCompositions(unittest.TestCase):
    def test_d2_S3(self):
        batches = list(generate_compositions_batched(2, 3, batch_size=100))
        all_comps = np.vstack(batches)
        self.assertEqual(len(all_comps), 4)
        self.assertTrue(np.all(all_comps.sum(axis=1) == 3))
        self.assertTrue(np.all(all_comps >= 0))

    def test_d3_S2(self):
        batches = list(generate_compositions_batched(3, 2, batch_size=100))
        all_comps = np.vstack(batches)
        self.assertEqual(len(all_comps), 6)
        self.assertTrue(np.all(all_comps.sum(axis=1) == 2))

    def test_batching(self):
        batches = list(generate_compositions_batched(4, 5, batch_size=3))
        all_comps = np.vstack(batches)
        expected = count_compositions(4, 5)
        self.assertEqual(len(all_comps), expected)

    def test_all_unique(self):
        batches = list(generate_compositions_batched(3, 4, batch_size=10))
        all_comps = np.vstack(batches)
        unique = set(map(tuple, all_comps.tolist()))
        self.assertEqual(len(unique), len(all_comps))


class TestReversalSymmetry(unittest.TestCase):
    def test_canonical_count_d4(self):
        d, S = 4, 12
        all_batches = list(generate_compositions_batched(d, S))
        all_comps = np.vstack(all_batches)
        n_total = len(all_comps)

        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_comps = np.vstack(canon_batches)
        n_canon = len(canon_comps)

        n_palindromes = sum(1 for c in all_comps if list(c) == list(c[::-1]))
        self.assertEqual(n_canon, (n_total + n_palindromes) // 2)

    def test_canonical_count_d6(self):
        d, S = 6, 12
        all_batches = list(generate_compositions_batched(d, S))
        all_comps = np.vstack(all_batches)
        n_total = len(all_comps)

        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_comps = np.vstack(canon_batches)
        n_canon = len(canon_comps)

        n_palindromes = sum(1 for c in all_comps if list(c) == list(c[::-1]))
        self.assertEqual(n_canon, (n_total + n_palindromes) // 2)

    def test_palindromes_included(self):
        d, S = 4, 8
        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_comps = np.vstack(canon_batches)

        all_batches = list(generate_compositions_batched(d, S))
        all_comps = np.vstack(all_batches)
        palindromes = [tuple(c) for c in all_comps if list(c) == list(c[::-1])]

        canon_set = set(map(tuple, canon_comps.tolist()))
        for p in palindromes:
            self.assertIn(p, canon_set, f"Palindrome {p} missing from canonical")

    def test_no_duplicates(self):
        d, S = 4, 12
        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_comps = np.vstack(canon_batches)
        unique = set(map(tuple, canon_comps.tolist()))
        self.assertEqual(len(unique), len(canon_comps))

    def test_all_canonical(self):
        d, S = 4, 12
        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_comps = np.vstack(canon_batches)
        mask = _canonical_mask(canon_comps)
        self.assertTrue(mask.all(), "Non-canonical composition in output")

    def test_canonical_covers_all_d4(self):
        d, S = 4, 8
        all_batches = list(generate_compositions_batched(d, S))
        all_comps = np.vstack(all_batches)

        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_set = set(map(tuple, np.vstack(canon_batches).tolist()))

        for comp in all_comps:
            rev = tuple(comp[::-1])
            fwd = tuple(comp)
            self.assertTrue(fwd in canon_set or rev in canon_set,
                            f"{fwd} and its reverse {rev} both missing")

    def test_test_value_symmetric(self):
        rng = np.random.RandomState(123)
        n_half = 2
        for _ in range(50):
            a = rng.dirichlet(np.ones(4)) * 8
            a_rev = a[::-1]
            tv = compute_test_value_single(a, n_half)
            tv_rev = compute_test_value_single(a_rev, n_half)
            self.assertAlmostEqual(tv, tv_rev, places=10,
                                   msg=f"a={a}, rev={a_rev}")

    def test_canonical_gen_d6_covers_all(self):
        d, S = 6, 6
        all_batches = list(generate_compositions_batched(d, S))
        all_comps = np.vstack(all_batches)

        canon_batches = list(generate_canonical_compositions_batched(d, S))
        canon_set = set(map(tuple, np.vstack(canon_batches).tolist()))

        for comp in all_comps:
            rev = tuple(comp[::-1])
            fwd = tuple(comp)
            self.assertTrue(fwd in canon_set or rev in canon_set,
                            f"{fwd} and its reverse {rev} both missing")


if __name__ == '__main__':
    unittest.main()

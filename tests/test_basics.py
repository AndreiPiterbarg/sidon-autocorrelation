"""Tests for basic utilities: correction, asymmetry threshold, composition counting."""
import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'cloninger-steinerberger'))

from core import correction, asymmetry_threshold, count_compositions


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
        self.assertEqual(count_compositions(2, 5), 6)

    def test_d4_S3(self):
        self.assertEqual(count_compositions(4, 3), 20)

    def test_d1_Sany(self):
        self.assertEqual(count_compositions(1, 100), 1)


if __name__ == '__main__':
    unittest.main()

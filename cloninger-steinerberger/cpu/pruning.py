"""Pruning utilities for branch-and-prune algorithm.

Correction terms, asymmetry thresholds, composition counting,
and symmetry masks.
"""
import numpy as np
import numba
from math import comb


def correction(m):
    """Discretization error bound (Lemma 3): C_{1a} >= b_{n,m} - 2/m - 1/m^2."""
    return 2.0 / m + 1.0 / (m * m)


def asymmetry_threshold(c_target):
    """Minimum left-mass fraction for the asymmetry argument to give >= c_target.

    If left_mass_frac >= threshold (or right_mass_frac >= threshold),
    then ||f*f||_inf >= 2 * threshold^2 >= c_target.
    """
    return np.sqrt(c_target / 2.0)


def count_compositions(d, S):
    """Number of non-negative integer vectors of length d summing to S.
    Equals C(S + d - 1, d - 1).
    """
    return comb(S + d - 1, d - 1)


def asymmetry_prune_mask(batch_int, n_half, m, c_target):
    """Return boolean mask: True for configs NOT covered by asymmetry argument.

    Returns True for configs that NEED test-value checking.
    """
    d = 2 * n_half
    total = float(4 * n_half * m)
    threshold = asymmetry_threshold(c_target)

    # Discretization margin: continuous left_frac can differ from discrete
    # by up to n_half * (1/m) / (4*n_half) = 1/(4m).
    margin = 1.0 / (4.0 * m)

    left = batch_int[:, :n_half].sum(axis=1).astype(np.float64)
    left_frac = left / total

    # Asymmetry safely covers: left_frac >= threshold + margin
    #                       or left_frac <= (1 - threshold) - margin
    # Need checking: everything else
    safe_threshold = threshold + margin
    needs_check = (left_frac > 1 - safe_threshold) & (left_frac < safe_threshold)
    return needs_check


@numba.njit(parallel=True, cache=True)
def _canonical_mask(batch_int):
    """Return bool mask: True for b where b <= rev(b) lexicographically."""
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    result = np.ones(B, dtype=numba.boolean)
    for b in numba.prange(B):
        for i in range(d // 2):
            j = d - 1 - i
            if batch_int[b, i] < batch_int[b, j]:
                break  # canonical
            elif batch_int[b, i] > batch_int[b, j]:
                result[b] = False
                break
    return result

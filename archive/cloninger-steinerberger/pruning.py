"""Pruning utilities for branch-and-prune algorithm.

Correction terms, asymmetry thresholds, composition counting,
and symmetry masks.
"""
import numpy as np
import numba
from math import comb


def correction(m, n_half=None, ell_min=2):
    """Discretization error bound (C&S Lemma 3).

    C&S Lemma 3 bounds ||g*g - f*f||_∞ ≤ 2/m + 1/m² (pointwise on the
    autoconvolution, NOT per-window on the test value).  Valid because
    the code uses the paper's fine grid B_{n,m} where heights are
    multiples of 1/m (integer compositions sum to 4nm), giving
    ||ε||_∞ ≤ 1/m.

    The n_half and ell_min parameters are accepted for API compatibility
    but are no longer used in the correction formula.
    """
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
    n_half can be a float (e.g. 1.5 for d=3 starting dimension).
    """
    d = batch_int.shape[1]
    total = float(2 * d * m)  # Fine grid: integer coords sum to 2*d*m
    threshold = asymmetry_threshold(c_target)

    # No discretization margin needed: left_frac = sum(c_i for left bins) / m
    # is exact for piecewise-constant functions on the discrete grid, and is
    # preserved exactly under refinement (child bins sum to parent bins).
    # See docs/verification_part1_framework.md, Verification 8 for full proof.

    left_bins = d // 2  # floor(d/2) bins as "left half"
    left = batch_int[:, :left_bins].sum(axis=1).astype(np.float64)
    left_frac = left / total

    # Asymmetry covers: left_frac >= threshold or left_frac <= 1 - threshold
    # Need checking: everything in between
    needs_check = (left_frac > 1 - threshold) & (left_frac < threshold)
    return needs_check


def block_mass_prune_mask(batch_int, n_half_child, m, c_target,
                          use_flat_threshold=False):
    """Return boolean mask: True for parents NOT prunable by the block mass
    invariant.  Parents where the mask is False are guaranteed to have ALL
    children pruned (for every cursor assignment, some window exceeds the
    threshold) and can be skipped entirely.

    Mathematical basis:
    For k consecutive parent bins [i, i+k-1] with total integer mass M
    (in the parent's fine grid), the 2k child bins always have total mass 2M
    (regardless of cursor values, since child[2j]+child[2j+1] = 2*parent[j]).
    The autoconvolution sum over the 4k-1 conv positions that are fully
    internal to this child block equals (2M)^2 = 4M^2.  Cross-terms from
    bins outside the block are non-negative (all masses >= 0), so the window
    sum ws >= 4M^2.

    We check: 4M^2 > threshold(ell=4k, W_int_max).
    W_int_max = S_child (conservative: largest W_int gives highest threshold).
    If this holds for ANY contiguous block, the parent is prunable.

    Derivation of ell = 4k:
    A block of k parent bins produces 2k child bins. Their autoconvolution
    has 2*(2k)-1 = 4k-1 conv positions. In the pruning kernel, a window
    of ell "output bins" sums n_cv = ell-1 consecutive conv values. To sum
    all 4k-1 values we need n_cv = 4k-1, hence ell = 4k.
    """
    return _block_mass_prune_mask_jit(batch_int, n_half_child, m, c_target,
                                       use_flat_threshold)


@numba.njit(parallel=True, cache=True)
def _block_mass_prune_mask_jit(batch_int, n_half_child, m, c_target,
                                use_flat_threshold):
    """JIT implementation of block mass invariant pruning.

    Uses per-parent W_int bounds for tighter thresholds: for a block at
    parent position i with k bins, the window overlaps specific parent bins
    whose total mass bounds W_int from above.
    """
    B = batch_int.shape[0]
    d_parent = batch_int.shape[1]
    needs_expansion = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    n_half_d = np.float64(n_half_child)
    d_child = 2 * d_parent
    four_n = 4.0 * n_half_d
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    d_child_m1 = d_child - 1

    # Flat C&S Lemma 3 correction: (2/m + 1/m^2)*m^2 = 2m + 1
    flat_corr = 2.0 * m_d + 1.0

    # Precompute scale factor per ell (used with per-parent W_int)
    max_ell = 2 * d_child
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    # Flat threshold per ell (window-independent, for use_flat_threshold mode)
    flat_thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    if use_flat_threshold:
        for ell in range(2, max_ell + 1):
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_arr[ell]
            flat_thr_arr[ell] = np.int64(dyn_x)

    for b in numba.prange(B):
        # Prefix sum of parent masses for fast block-sum queries
        prefix = np.empty(d_parent + 1, dtype=np.int64)
        prefix[0] = 0
        for i in range(d_parent):
            prefix[i + 1] = prefix[i] + np.int64(batch_int[b, i])

        pruned = False
        for k in range(1, d_parent + 1):
            if pruned:
                break
            ell = 4 * k
            if ell > max_ell:
                break

            # Check all contiguous blocks of k parent bins
            for i in range(d_parent - k + 1):
                M = prefix[i + k] - prefix[i]
                four_M_sq = np.int64(4) * M * M

                if use_flat_threshold:
                    thr = flat_thr_arr[ell]
                else:
                    # Compute W_int_max for this specific block and parent.
                    # Window: s_lo = 4*i, ell = 4*k, covering conv
                    # positions [4i, 4i+4k-2].
                    # Bins overlapping this window:
                    s_lo = 4 * i
                    lo_bin = s_lo - d_child_m1
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child_m1:
                        hi_bin = d_child_m1
                    # Map child bins to parent bins:
                    # child[2j], child[2j+1] belong to parent j.
                    # Parent bins overlapping: ceil(lo_bin/2) to hi_bin//2
                    p_lo = (lo_bin + 1) // 2  # ceil(lo_bin/2)
                    p_hi = hi_bin // 2
                    if p_lo < 0:
                        p_lo = 0
                    if p_hi >= d_parent:
                        p_hi = d_parent - 1
                    if p_lo <= p_hi:
                        # W_int <= 2 * sum(parent[p_lo..p_hi])
                        W_int_max = np.int64(2) * (prefix[p_hi + 1]
                                                     - prefix[p_lo])
                    else:
                        W_int_max = np.int64(0)

                    corr_w = 1.0 + np.float64(W_int_max) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_arr[ell]
                    thr = np.int64(dyn_x)

                if four_M_sq > thr:
                    pruned = True
                    break

        if pruned:
            needs_expansion[b] = False

    return needs_expansion


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

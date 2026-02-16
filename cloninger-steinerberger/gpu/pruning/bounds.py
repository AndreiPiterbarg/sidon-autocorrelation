"""Mathematical pruning bounds for the Cloninger-Steinerberger algorithm.

Each bound is a lower bound on the test value:
    test_val(c) = max_{window(k,ell)} (1/(4*n_half*ell)) * sum_{k<=i+j<=k+ell-2} a_i*a_j

where a_i = c_i / m, sum(c_i) = 4*n_half*m.

All bounds have signature: bound_fn(c, n_half, m) -> float
and satisfy: bound_fn(c, n_half, m) <= true_test_value(c, n_half, m)
"""
import numpy as np
import sys
import os

# Add parent paths so we can import the CPU modules
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from test_values import compute_test_value_single


def true_test_value(c, n_half, m):
    """Ground-truth test value via full autoconvolution + window max.

    Parameters
    ----------
    c : array-like of int
        Integer mass vector (c_i = m * a_i), sum = 4*n_half*m.
    n_half : int
        Paper's n (half the number of bins).
    m : int
        Grid resolution.

    Returns
    -------
    float : The exact test value.
    """
    c = np.asarray(c, dtype=np.float64)
    a = c / m
    return compute_test_value_single(a, n_half)


# ============================================================
# Bound 1: Contiguous Block Sum
# ============================================================

def block_sum_bound(c, n_half, m, q, p):
    """Lower bound from a contiguous block of p bins starting at index q.

    Theorem: For any contiguous block {c_q, ..., c_{q+p-1}}:
        test_val >= block_sum^2 / (4*n_half * 2*p * m^2)

    Proof: The self-convolution of the block occupies conv indices
    [2q, 2(q+p-1)], spanning 2p-1 values (window length ell=2p).
    Cross-terms from elements outside the block are non-negative
    (all c_i >= 0), so window_sum >= block_sum^2 / m^2.

    Parameters
    ----------
    c : array-like of int
        Integer mass vector.
    q : int
        Starting index of the block.
    p : int
        Number of bins in the block (p >= 1).

    Returns
    -------
    float : Lower bound on test value.
    """
    c = np.asarray(c, dtype=np.int64)
    block_sum = int(np.sum(c[q:q+p]))
    return block_sum * block_sum / (4.0 * n_half * 2 * p * m * m)


def all_block_sum_bounds(c, n_half, m):
    """Maximum block-sum bound over all valid (q, p) combinations.

    Returns
    -------
    float : max over all blocks of block_sum_bound(c, n_half, m, q, p).
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    best = 0.0
    for p in range(1, d + 1):
        ell = 2 * p
        if ell > d:
            break  # ell cannot exceed d in the window-max formula
        inv_norm = 1.0 / (4.0 * n_half * ell * m * m)
        for q in range(d - p + 1):
            block_sum = int(np.sum(c[q:q+p]))
            val = block_sum * block_sum * inv_norm
            if val > best:
                best = val
    return best


def new_block_sum_bounds(c, n_half, m):
    """Block-sum bounds at intermediate ell NOT currently checked in CUDA.

    For D=6: checks p=2 (ell=4) and p=3 (ell=6) blocks that are
    not the edge half-sums (which are already checked).

    For D=4: checks p=2 (ell=4) interior block {c1,c2}.

    Returns
    -------
    float : max of the new (currently unchecked) block-sum bounds.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    best = 0.0

    if d == 6:
        # ell=4 (p=2): adjacent pairs - edges {c0,c1} and {c4,c5} are already
        # covered by ell=6 half-sum. New checks: {c1,c2}, {c2,c3}, {c3,c4}
        inv_4 = 1.0 / (4.0 * n_half * 4 * m * m)
        for q in [1, 2, 3]:
            s = int(c[q] + c[q+1])
            val = s * s * inv_4
            if val > best:
                best = val

        # ell=6 (p=3): interior triples {c1,c2,c3} and {c2,c3,c4}
        # Edge triples {c0,c1,c2} and {c3,c4,c5} are already checked as half-sums.
        inv_6 = 1.0 / (4.0 * n_half * 6 * m * m)
        for q in [1, 2]:
            s = int(c[q] + c[q+1] + c[q+2])
            val = s * s * inv_6
            if val > best:
                best = val

    elif d == 4:
        # ell=4 (p=2): interior pair {c1,c2}
        # Edge pairs {c0,c1} and {c2,c3} are already checked as half-sums.
        inv_4 = 1.0 / (4.0 * n_half * 4 * m * m)
        s = int(c[1] + c[2])
        val = s * s * inv_4
        if val > best:
            best = val

    return best


# ============================================================
# Bound 2: Two-Max Enhanced ell=2
# ============================================================

def two_max_bound(c, n_half, m):
    """Lower bound using the two largest elements at ell=2.

    Theorem: If max1, max2 are the two largest c_i:
        test_val >= max(max1^2, 2*max1*max2) / (4*n_half * 2 * m^2)

    Proof: conv[i+j] >= 2*c_i*c_j/m^2 for i != j (other terms
    non-negative). Taking i,j as positions of the two largest gives
    the bound. When max2 >= max1/2, the cross-term dominates.

    Returns
    -------
    float : Lower bound on test value.
    """
    c = np.asarray(c, dtype=np.int64)
    sorted_c = np.sort(c)[::-1]
    max1 = int(sorted_c[0])
    max2 = int(sorted_c[1]) if len(c) > 1 else 0
    inv_norm = 1.0 / (4.0 * n_half * 2 * m * m)
    return max(max1 * max1, 2 * max1 * max2) * inv_norm


def two_max_improvement(c, n_half, m):
    """Return only the improvement over the existing max-element bound.

    The existing ell=2 bound uses max1^2. This returns the two-max
    bound minus the existing bound (positive when 2*max1*max2 > max1^2,
    i.e., when max2 > max1/2).

    Returns
    -------
    float : Additional bound value from two-max (0 if no improvement).
    """
    c = np.asarray(c, dtype=np.int64)
    sorted_c = np.sort(c)[::-1]
    max1 = int(sorted_c[0])
    max2 = int(sorted_c[1]) if len(c) > 1 else 0
    inv_norm = 1.0 / (4.0 * n_half * 2 * m * m)
    existing = max1 * max1 * inv_norm
    two_max = max(max1 * max1, 2 * max1 * max2) * inv_norm
    return two_max - existing


# ============================================================
# Bound 3: Specific Cross-Term
# ============================================================

def cross_term_bound(c, n_half, m, i, j):
    """Lower bound from a specific cross-term c_i * c_j at ell=2.

    Theorem: For any pair (i,j) with i != j:
        test_val >= 2*c_i*c_j / (4*n_half * 2 * m^2)

    Proof: conv[i+j] includes term 2*a_i*a_j (other terms non-negative).
    The ell=2 window at k=i+j has test value >= 2*c_i*c_j/(4*n_half*2*m^2).

    Returns
    -------
    float : Lower bound on test value.
    """
    c = np.asarray(c, dtype=np.int64)
    return 2.0 * int(c[i]) * int(c[j]) / (4.0 * n_half * 2 * m * m)


def cross_term_c0c5_bound(c, n_half, m):
    """Cross-term bound 2*c0*c5 for D=6 configs.

    Targets symmetric edge-heavy configs where c0 and c5 are both
    large but left/right sums are balanced (surviving asymmetry test).

    Returns
    -------
    float : Lower bound on test value, or 0.0 if d != 6.
    """
    c = np.asarray(c, dtype=np.int64)
    if len(c) != 6:
        return 0.0
    return 2.0 * int(c[0]) * int(c[5]) / (4.0 * n_half * 2 * m * m)


# ============================================================
# Bound 5: Full central conv value at ell=2 ("conv5 bound")
# ============================================================

def central_conv_ell2_bound(c, n_half, m):
    """Lower bound from the full central convolution value conv[d-1] at ell=2.

    Theorem: conv[d-1] = 2*sum_{i < d/2} c_i * c_{d-1-i} (exact for even d).
    Since all terms are non-negative, this IS the exact value (not just a
    lower bound). The ell=2 window at position k=d-1 gives:
        test_val >= conv[d-1] / (4*n_half * 2 * m^2)

    For D=6: conv[5] = 2*(c0*c5 + c1*c4 + c2*c3). Subsumes cross_term_c0c5.
    For D=4: conv[3] = 2*(c0*c3 + c1*c2).

    Returns
    -------
    float : Lower bound on test value.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    conv_center = 0
    for i in range(d // 2):
        conv_center += int(c[i]) * int(c[d - 1 - i])
    conv_center *= 2
    return conv_center / (4.0 * n_half * 2 * m * m)


# ============================================================
# Bound 6: Adjacent conv values at ell=2 ("conv4/conv6 bounds")
# ============================================================

def adjacent_conv_ell2_bounds(c, n_half, m):
    """Lower bound from conv[d-2] and conv[d] at ell=2.

    Theorem: conv[k] = sum_{i+j=k} c_i*c_j / m^2. For specific k:
    - conv[d-2]: includes c_{d/2-1}^2 (diagonal) + cross-terms at i+j=d-2
    - conv[d]:   includes c_{d/2}^2 (diagonal) + cross-terms at i+j=d

    These are exact conv values, used as ell=2 bounds:
        test_val >= conv[k] / (4*n_half * 2 * m^2)

    For D=6:
    - conv[4] = c2^2 + 2*(c0*c4 + c1*c3)
    - conv[6] = c3^2 + 2*(c1*c5 + c2*c4)

    For D=4:
    - conv[2] = c1^2 + 2*c0*c2
    - conv[4] = c2^2 + 2*c1*c3

    Returns
    -------
    float : max(conv[d-2], conv[d]) / (4*n_half * 2 * m^2).
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    inv_norm = 1.0 / (4.0 * n_half * 2 * m * m)

    # conv[d-2]: sum of c_i*c_j where i+j = d-2, with i,j in [0, d-1]
    conv_lo = 0
    k_lo = d - 2
    for i in range(d):
        j = k_lo - i
        if 0 <= j < d:
            conv_lo += int(c[i]) * int(c[j])

    # conv[d]: sum of c_i*c_j where i+j = d, with i,j in [0, d-1]
    conv_hi = 0
    k_hi = d
    for i in range(d):
        j = k_hi - i
        if 0 <= j < d:
            conv_hi += int(c[i]) * int(c[j])

    return max(conv_lo, conv_hi) * inv_norm


# ============================================================
# Bound 7: ell=3 sliding window bounds
# ============================================================

def ell3_central_bounds(c, n_half, m):
    """Lower bound from ell=3 windows using adjacent conv values.

    Theorem: The ell=3 window at position k covers conv values
    conv[k] and conv[k+1]:
        window_sum(k, 3) = conv[k] + conv[k+1]
        test_val >= (conv[k] + conv[k+1]) / (4*n_half * 3 * m^2)

    We compute the three central conv values (conv[d-2], conv[d-1], conv[d])
    and check two ell=3 windows:
    - (conv[d-2] + conv[d-1]) / (4*n_half * 3 * m^2)
    - (conv[d-1] + conv[d])   / (4*n_half * 3 * m^2)

    These are exact ell=3 test values — not lower bounds of windows,
    but the true window sums at these specific positions.

    Returns
    -------
    float : max of the two ell=3 window bounds.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    inv_norm_3 = 1.0 / (4.0 * n_half * 3 * m * m)

    # Compute conv[d-2], conv[d-1], conv[d]
    conv_vals = {}
    for k in [d - 2, d - 1, d]:
        s = 0
        for i in range(d):
            j = k - i
            if 0 <= j < d:
                s += int(c[i]) * int(c[j])
        conv_vals[k] = s

    window_a = (conv_vals[d - 2] + conv_vals[d - 1]) * inv_norm_3
    window_b = (conv_vals[d - 1] + conv_vals[d]) * inv_norm_3
    return max(window_a, window_b)


# ============================================================
# Bound 8: Partial ell=4 center window bound
# ============================================================

def partial_ell4_center_bound(c, n_half, m):
    """Lower bound from the ell=4 center window using three conv values.

    Theorem: The ell=4 window at position k=d-2 covers conv values
    conv[d-2], conv[d-1], conv[d]:
        window_sum(d-2, 4) = conv[d-2] + conv[d-1] + conv[d]
        test_val >= (conv[d-2] + conv[d-1] + conv[d]) / (4*n_half * 4 * m^2)

    This is the exact ell=4 test value at the center position.

    Returns
    -------
    float : The ell=4 center window bound.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    inv_norm_4 = 1.0 / (4.0 * n_half * 4 * m * m)

    # Compute conv[d-2] + conv[d-1] + conv[d]
    total = 0
    for k in [d - 2, d - 1, d]:
        for i in range(d):
            j = k - i
            if 0 <= j < d:
                total += int(c[i]) * int(c[j])

    return total * inv_norm_4


# ============================================================
# Bound 4: Sum-of-Squares Pigeonhole
# ============================================================

def sum_of_squares_bound(c, n_half, m):
    """Lower bound from sum of squared elements (pigeonhole on diagonal).

    Theorem:
        test_val >= sum(c_i^2) / (d * 4*n_half * 2 * m^2)

    Proof: sum conv[2k] >= sum c_k^2/m^2 (diagonal entries of
    autoconvolution). There are d diagonal entries. By pigeonhole,
    max conv[2k] >= sum(c_k^2)/(d*m^2). The ell=2 window containing
    this max diagonal entry has test value >= sum(c_k^2)/(d*4*n_half*2*m^2).

    Note: Weak due to 1/d factor. Mainly useful as a Python-only check.

    Returns
    -------
    float : Lower bound on test value.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    sum_sq = int(np.sum(c * c))
    return sum_sq / (d * 4.0 * n_half * 2 * m * m)


# ============================================================
# Existing bounds (for reference / benchmarking comparison)
# ============================================================

def existing_max_element_bound(c, n_half, m):
    """Existing ell=2 bound: max(c_i)^2 / (4*n_half*2*m^2)."""
    c = np.asarray(c, dtype=np.int64)
    max_c = int(np.max(c))
    return max_c * max_c / (4.0 * n_half * 2 * m * m)


def existing_half_sum_bound(c, n_half, m):
    """Existing ell=D bound: max(left_half_sum, right_half_sum)^2 / (4*n_half*D*m^2)."""
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    left = int(np.sum(c[:d//2]))
    right = int(np.sum(c[d//2:]))
    max_half = max(left, right)
    return max_half * max_half / (4.0 * n_half * d * m * m)


def existing_asymmetry_bound(c, n_half, m):
    """Existing asymmetry bound (from Lemma on half-mass dominance).

    If left_frac = sum(c[:n_half]) / S, then the asymmetry argument gives
    test_val >= 2*(max(left_frac, 1-left_frac) - margin)^2
    where margin = 1/(4m).

    Returns the effective bound value.
    """
    c = np.asarray(c, dtype=np.float64)
    d = len(c)
    S = np.sum(c)
    left_frac = np.sum(c[:d//2]) / S
    margin = 1.0 / (4.0 * m)
    dom = max(left_frac, 1.0 - left_frac)
    asym_base = max(dom - margin, 0.0)
    return 2.0 * asym_base * asym_base


# ============================================================
# Combined bounds
# ============================================================

def best_new_bound(c, n_half, m):
    """Maximum of all NEW bounds (not currently in CUDA pipeline).

    Returns
    -------
    float : max of all new bounds including conv-value bounds.
    """
    vals = [
        new_block_sum_bounds(c, n_half, m),
        two_max_bound(c, n_half, m),
        sum_of_squares_bound(c, n_half, m),
        central_conv_ell2_bound(c, n_half, m),
        adjacent_conv_ell2_bounds(c, n_half, m),
        ell3_central_bounds(c, n_half, m),
        partial_ell4_center_bound(c, n_half, m),
    ]
    if len(c) == 6:
        vals.append(cross_term_c0c5_bound(c, n_half, m))
    return max(vals)


def best_existing_bound(c, n_half, m):
    """Maximum of all existing bounds (currently in CUDA pipeline).

    Returns
    -------
    float : max of existing_max_element_bound, existing_half_sum_bound,
            existing_asymmetry_bound.
    """
    return max(
        existing_max_element_bound(c, n_half, m),
        existing_half_sum_bound(c, n_half, m),
        existing_asymmetry_bound(c, n_half, m),
    )

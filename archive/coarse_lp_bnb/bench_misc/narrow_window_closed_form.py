"""Closed-form narrow-window certificates for ell <= 5.

Sound LB on min_{mu in cell} TV_W(mu), exploiting:
  TV_W(mu) = (2d/ell) * sum_{i+j in [s, s+ell-2]} mu_i mu_j
and that lo_i >= 0 (cell clipped to [0,1]^d), so the min over the box
[lo, hi]^d of any monomial mu_i mu_j is lo_i * lo_j. The simplex
constraint sum mu = 1 only tightens this LB.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _narrow_min_tv_lb(lo, d, ell, s):
    """O(ell*d) closed-form LB on min over cell of mu^T A_W mu, ell <= 5.

    Returns scale * (mu^T A_W mu)_min where scale = 2d/ell, computed by
    replacing each mu_i mu_j with lo_i * lo_j (sound since lo >= 0 in cell).
    """
    s_hi = s + ell - 2
    quad = 0.0
    # Iterate over antidiagonals k = i+j in [s, s_hi]
    for k in range(s, s_hi + 1):
        # i + j = k, 0 <= i, j <= d-1, i <= j
        i_lo = 0 if k - (d - 1) < 0 else k - (d - 1)
        i_hi = k // 2
        for i in range(i_lo, i_hi + 1):
            j = k - i
            if j > d - 1:
                continue
            if i == j:
                quad += lo[i] * lo[i]
            else:
                quad += 2.0 * lo[i] * lo[j]
    return (2.0 * d / ell) * quad


@njit(cache=True)
def _box_certify_cell_hybrid(mu_center, d, delta_q, c_target, ell_closed_max):
    """Hybrid: closed-form narrow (ell <= ell_closed_max) + caller does vertex
    enum for wider. Returns (best_min_tv, certified_by_narrow).
    """
    h = 0.5 * delta_q
    lo = np.empty(d, dtype=np.float64)
    for i in range(d):
        v = mu_center[i] - h
        lo[i] = v if v > 0.0 else 0.0

    best = 0.0
    for ell in range(2, ell_closed_max + 1):
        for s in range(2 * d - ell + 1):
            tv_lb = _narrow_min_tv_lb(lo, d, ell, s)
            if tv_lb > best:
                best = tv_lb
            if best >= c_target:
                return best, True
    return best, False

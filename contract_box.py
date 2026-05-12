"""HC4-revise / FBBT contraction of a box against the simplex equality
constraint sum(mu_i) = 1.

Given an axis-aligned box [lo, hi] and the simplex equality
    sum_i mu_i = 1,
each coordinate satisfies
    mu_j = 1 - sum_{i != j} mu_i
        in [1 - (sum_i hi_i - hi_j),  1 - (sum_i lo_i - lo_j)].
So we can tighten in place:
    hi_j_new = min(hi_j, 1 - sum_{i != j} lo_i) = min(hi_j, 1 - lo_sum + lo_j)
    lo_j_new = max(lo_j, 1 - sum_{i != j} hi_i) = max(lo_j, 1 - hi_sum + hi_j).

The two rules are interleaved: tightening lo_j shrinks lo_sum and may
unlock further contraction on hi_k for k != j (and vice versa), so we
iterate to a fixed point (or up to `max_iters` sweeps).

SOUNDNESS. The contraction removes only points that VIOLATE the simplex
equality: any feasible mu (i.e. mu in the box AND on the simplex) lies
in the contracted box by the elementary derivation above. Thus every
rigorous LB on (mu^T M_W mu) over the contracted box is also a valid LB
over the (original box ∩ simplex), which is what the BnB cover
ultimately needs.

If after contraction lo_j > hi_j for any j, the box is empty: it does
not intersect the simplex, and may be pruned for free.

This module deliberately performs the contraction in float64 — it is
used as a *cheap pre-pass* for the rigor cascade. The rigor-bearing
integer endpoints (`Box.lo_int`, `Box.hi_int`) are recomputed by the
caller via outward rounding (floor for lo, ceil for hi) so the
integer box still CONTAINS the float box. For an alternative that
performs contraction directly on the integer endpoints (rigorous and
reproducible), see `Box.tighten_to_simplex` in `interval_bnb/box.py`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


EMPTY = "empty"


def contract_box(
    lo: np.ndarray, hi: np.ndarray, *,
    max_iters: int = 20, tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """HC4-revise the simplex equality on the box [lo, hi].

    Parameters
    ----------
    lo, hi : np.ndarray, shape (d,)
        Box endpoints. The arrays are NOT mutated.
    max_iters : int
        Cap on outer sweeps. Each sweep is O(d) work.
    tol : float
        Stop when no axis is tightened by more than `tol` in a sweep.

    Returns
    -------
    lo_new, hi_new : np.ndarray
        Contracted endpoints. If `is_empty` is True, these are returned
        as-is from the last partial sweep and SHOULD NOT be consumed.
    is_empty : bool
        True iff lo_new[j] > hi_new[j] for some j after contraction
        (i.e. the box does not intersect {sum mu = 1, mu >= 0}).
    """
    lo = np.array(lo, dtype=np.float64, copy=True)
    hi = np.array(hi, dtype=np.float64, copy=True)
    d = lo.shape[0]
    for _ in range(max_iters):
        lo_sum = float(lo.sum())
        hi_sum = float(hi.sum())
        # Rule 1 (upper bounds): hi_j <= 1 - sum_{i!=j} lo_i = 1 - lo_sum + lo_j.
        new_hi = np.minimum(hi, 1.0 - lo_sum + lo)
        # Rule 2 (lower bounds): lo_j >= 1 - sum_{i!=j} hi_i = 1 - hi_sum + hi_j.
        # Use the freshly contracted hi for tighter immediate effect.
        new_hi_sum = float(new_hi.sum())
        new_lo = np.maximum(lo, 1.0 - new_hi_sum + new_hi)
        # Empty check.
        if np.any(new_lo > new_hi + 1e-300):
            return new_lo, new_hi, True
        # Convergence check.
        delta = max(
            float(np.max(hi - new_hi, initial=0.0)),
            float(np.max(new_lo - lo, initial=0.0)),
        )
        lo = new_lo
        hi = new_hi
        if delta <= tol:
            break
    # Final empty check (defensive).
    is_empty = bool(np.any(lo > hi + 1e-300))
    return lo, hi, is_empty

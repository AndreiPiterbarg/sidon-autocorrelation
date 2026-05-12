"""Lower-bound routines for min_{mu in B cap Delta_d} mu^T M_W mu.

Three progressively tighter bounds are provided.

1. Natural interval form (`bound_natural`):
       mu_i mu_j in [lo_i lo_j, hi_i hi_j]      (since mu >= 0)
   Summing the lower endpoints gives a valid lower bound. This equals
   the McCormick bound evaluated at mu = lo (the box's lower corner)
   and requires NO LP solve. Sharp when boxes shrink around lo.

2. McCormick linear envelope (`bound_mccormick_lp`):
   For each pair (i, j) in the support of M_W,
       mu_i mu_j >= lo_j * mu_i + lo_i * mu_j - lo_i * lo_j
   (the concave-hull "SW face" of the bilinear surface on a box with
   non-negative endpoints). Summing these inequalities over all pairs
   yields a LINEAR lower bound
       mu^T M_W mu / scale  >=  g^T mu  +  c0        (*)
   where g[k] = sum_{(k, j) in pairs_all} lo[j] and
         c0   = -sum_{(i, j) in pairs_all, i<=j} {lo_i lo_j (x2 for i<j)}.
   More succinctly, c0 = -sum_{(i,j) in pairs_all} lo_i lo_j because
   pairs_all already double-counts off-diagonal entries.
   Minimising (*) over the domain {mu : sum mu = 1, lo <= mu <= hi,
   mu_i <= mu_j for sym cuts} is a small LP; we use scipy's HiGHS dual
   simplex. Tighter than natural whenever the LP optimum lies above lo.

3. Rigorous Fraction replay (`bound_natural_exact`, `bound_mccormick_exact`):
   Same formulas computed in fractions.Fraction. Returns a Fraction;
   used to re-verify each certified leaf.

Correctness of McCormick bound:
   For mu_i, mu_j in [a_i, b_i] x [a_j, b_j] with a >= 0,
       (mu_i - a_i)(mu_j - a_j) >= 0
   => mu_i mu_j >= a_j mu_i + a_i mu_j - a_i a_j
   valid pointwise, hence valid after summation.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .windows import WindowMeta


# Sentinel: a Fraction so large that any certification target is exceeded.
# Chosen larger than any plausible M_W spectral radius (val(d) is O(1)).
# Used on the empty-domain branch of exact McCormick routines so an
# infeasible box vacuously satisfies any finite target (rather than
# silently returning Fraction(0) which could mis-certify).
_FRACTION_INF = Fraction(10**18)


# ---------------------------------------------------------------------
# 1. Natural interval bound (no simplex use)
# ---------------------------------------------------------------------

def bound_natural(lo: np.ndarray, hi: np.ndarray, w: WindowMeta) -> float:
    """Lower bound via natural interval: mu_i mu_j >= lo_i lo_j.
    Equivalent to scale * lo^T A_W lo.
    Weak when lo is near 0 because it ignores the simplex constraint."""
    A = _adjacency_matrix(w, len(lo))
    return w.scale * float(lo @ A @ lo)


def bound_natural_exact(lo_q: Sequence[Fraction], hi_q: Sequence[Fraction],
                        w: WindowMeta) -> Fraction:
    """Exact Fraction version of bound_natural."""
    total = Fraction(0)
    for (i, j) in w.pairs_all:
        total += lo_q[i] * lo_q[j]
    return w.scale_q * total


# ---------------------------------------------------------------------
# 1b. Autoconvolution complement bound (uses simplex Sigma mu = 1)
# ---------------------------------------------------------------------

def bound_autoconv(lo: np.ndarray, hi: np.ndarray, w: WindowMeta,
                   d: int) -> float:
    """LB via the identity sum_{s=0}^{2d-2} c_s(mu) = (sum mu)^2 = 1.

    For window W with sum-support S_W,  c_s >= 0 for every s, so
        sum_{s in S_W} c_s = 1 - sum_{s not in S_W} c_s
                          >= 1 - sum_{(i,j): i+j not in S_W} hi_i hi_j
    = 1 - hi^T (ones - A_W) hi = 1 - (sum hi)^2 + hi^T A_W hi.
    Floor at 0 since mu^T M_W mu >= 0 always.
    """
    A = _adjacency_matrix(w, d)
    sum_hi = float(hi.sum())
    hAh = float(hi @ A @ hi)
    lb = 1.0 - sum_hi * sum_hi + hAh
    if lb < 0.0:
        lb = 0.0
    return w.scale * lb


def bound_autoconv_exact(lo_q: Sequence[Fraction], hi_q: Sequence[Fraction],
                         w: WindowMeta, d: int) -> Fraction:
    in_support = _pair_in_support_mask(w, d)
    comp = Fraction(0)
    for i in range(d):
        for j in range(d):
            if not in_support[i, j]:
                comp += hi_q[i] * hi_q[j]
    lb = Fraction(1) - comp
    if lb < 0:
        lb = Fraction(0)
    return w.scale_q * lb


_MASK_CACHE: dict = {}
_COMP_CACHE: dict = {}


def _pair_in_support_mask(w: WindowMeta, d: int) -> np.ndarray:
    key = (w.ell, w.s_lo, d)
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    m = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(d):
            s = i + j
            if w.s_lo <= s <= w.s_lo + w.ell - 2:
                m[i, j] = True
    _MASK_CACHE[key] = m
    return m


_PAIR_IDX_CACHE: dict = {}
_COMP_IDX_CACHE: dict = {}


def _pair_idx_arrays(w: WindowMeta, d: int):
    """(ii, jj) int arrays of all ordered (i,j) with (i,j) in support."""
    key = (w.ell, w.s_lo, d)
    got = _PAIR_IDX_CACHE.get(key)
    if got is not None:
        return got
    mask = _pair_in_support_mask(w, d)
    ii, jj = np.where(mask)
    _PAIR_IDX_CACHE[key] = (ii, jj)
    return ii, jj


def _comp_idx_arrays(w: WindowMeta, d: int):
    """(ii, jj) int arrays of all ordered (i,j) NOT in support."""
    key = (w.ell, w.s_lo, d)
    got = _COMP_IDX_CACHE.get(key)
    if got is not None:
        return got
    mask = _pair_in_support_mask(w, d)
    ii, jj = np.where(~mask)
    _COMP_IDX_CACHE[key] = (ii, jj)
    return ii, jj


# ---------------------------------------------------------------------
# 2. McCormick LP lower bound
# ---------------------------------------------------------------------

def _mccormick_linear(lo: np.ndarray, w: WindowMeta) -> Tuple[np.ndarray, float]:
    """Return (g, c0) such that for every mu in the box,
            mu^T M_W mu / scale  >=  g.mu + c0.

    For each ORDERED pair (i,j) in the support of M_W, the SW McCormick
    face contributes  lo_j * mu_i + lo_i * mu_j - lo_i * lo_j. Summing
    gives g = 2 * A_W @ lo  and  c0 = -lo^T A_W lo, where A_W[i,j] = 1
    iff (i,j) in pairs_all (the M_W/scale_W indicator matrix).
    """
    # Diagonal mu_i^2: tangent under-estimator (valid LB; secant would
    # be an UPPER bound since x^2 is convex).
    A = _adjacency_matrix(w, len(lo))
    Alo = A @ lo
    g = 2.0 * Alo
    c0 = -float(lo @ Alo)
    return g, c0


_ADJ_CACHE: dict = {}
_TENSOR_CACHE: dict = {}


def _adjacency_matrix(w: WindowMeta, d: int) -> np.ndarray:
    """Cached (d x d) 0/1 adjacency matrix A_W with A[i,j]=1 iff
    (i,j) in pairs_all."""
    key = (w.ell, w.s_lo, d)
    A = _ADJ_CACHE.get(key)
    if A is not None:
        return A
    A = np.zeros((d, d), dtype=np.float64)
    ii, jj = _pair_idx_arrays(w, d)
    A[ii, jj] = 1.0
    _ADJ_CACHE[key] = A
    return A


def gap_weighted_split_axis(
    lo: np.ndarray, hi: np.ndarray,
    w: WindowMeta, d: int,
) -> int:
    """Pick the axis whose split will most tighten the winning window's
    McCormick bound, with a width-based regulariser so no axis can be
    starved of splits.

    Raw gap score: `widths[k] * (A_W @ widths)[k]`.

    Problem with raw: a window whose support is a SINGLE pair (k, k)
    (e.g. ell=2 s_lo=2k) has `coverage[j] = 0` for j != k, so the raw
    score is zero everywhere except axis k. Gap-weighted then picks
    axis k repeatedly until widths[k] shrinks past the integer
    denominator (2**-60), raising a split-depth-exhausted error.

    Regulariser: normalise width and gap-contribution each to [0,1]
    and blend them. The blended score is always > 0 for the widest
    axis, so every axis eventually gets split.
    """
    widths = hi - lo
    A = _adjacency_matrix(w, d)
    # (A @ widths)[k] = sum of widths[j] over (k, j) in supp(W).
    coverage = A @ widths
    gap_score = widths * coverage
    w_max = float(widths.max())
    g_max = float(gap_score.max())
    if w_max <= 0.0:
        # All axes at zero width -- caller will detect min_box_width
        # and abort. Pick axis 0 arbitrarily.
        return 0
    # Normalised blend. Weight gap 70% and width 30%; the width term
    # guarantees no axis gets starved indefinitely.
    w_norm = widths / w_max
    g_norm = gap_score / g_max if g_max > 0 else np.zeros_like(widths)
    score = 0.7 * g_norm + 0.3 * w_norm
    return int(np.argmax(score))


def window_tensor(windows: Sequence[WindowMeta], d: int):
    """Return (A_tensor, scales) where A_tensor is (W, d, d) and
    scales is (W,). Cached by the identity of the windows list."""
    key = (id(windows), d)
    got = _TENSOR_CACHE.get(key)
    if got is not None:
        return got
    W = len(windows)
    A = np.zeros((W, d, d), dtype=np.float64)
    scales = np.empty(W, dtype=np.float64)
    for wi, w in enumerate(windows):
        A[wi] = _adjacency_matrix(w, d)
        scales[wi] = w.scale
    _TENSOR_CACHE[key] = (A, scales)
    return A, scales


# ---------------------------------------------------------------------
# Batched multi-window bound evaluation (the hot path).
# ---------------------------------------------------------------------

def batch_bounds(
    lo: np.ndarray, hi: np.ndarray,
    A_tensor: np.ndarray, scales: np.ndarray,
    target_c: float,
    *,
    cache: Optional[dict] = None,
):
    """Evaluate bounds for EVERY window in a single vectorised pass.
    Backwards-compatible wrapper; prefer `batch_bounds_full` for callers
    that also want the per-box cache for rank-1 updates of children.
    """
    lb, w_idx, which, _mu, _cache = batch_bounds_full(
        lo, hi, A_tensor, scales, target_c,
    )
    if cache is not None:
        # Populate legacy dict for backwards compat.
        Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, _lo, _hi = _cache
        cache["Alo"] = Alo; cache["Ahi"] = Ahi
        cache["loAlo"] = loAlo; cache["hiAhi"] = hiAhi
        cache["lo_sum"] = lo_sum; cache["hi_sum"] = hi_sum
    return lb, w_idx, which, _mu


# ---------------------------------------------------------------------
# Rank-1 incremental evaluation (DFS fast path)
# ---------------------------------------------------------------------
#
# When a parent box splits at axis k, one of its children changes exactly
# ONE coordinate of lo (right child) or hi (left child) -- both arrays
# are shared otherwise. The cached projections can then be updated in
# O(W) per child instead of recomputed in O(W * d^2) from scratch:
#
#     new_lo = lo + delta * e_k       (delta > 0 for right child)
#     new_Alo[w, :] = Alo[w, :] + A[w, :, k] * delta
#     new_loAlo[w]  = loAlo[w] + 2 * delta * Alo[w, k] + delta^2 * A[w, k, k]
#     new_lo_sum    = lo_sum + delta
#
# Symmetric formulas for hi / Ahi / hiAhi / hi_sum. Since M_W (and hence
# A_W) is 0/1, A[w, k, k] in (0, 1).
# ---------------------------------------------------------------------

def batch_bounds_full(
    lo: np.ndarray, hi: np.ndarray,
    A_tensor: np.ndarray, scales: np.ndarray,
    target_c: float,
):
    """Full compute (no parent cache). Returns
        (best_lb, best_idx, best_which, best_mu, cache_tuple).
    The `cache_tuple` can be passed to batch_bounds_rank1_{lo,hi} for
    a cheap O(W) evaluation of the child box.
    """
    Alo = A_tensor @ lo
    Ahi = A_tensor @ hi
    loAlo = (Alo * lo).sum(axis=1)
    hiAhi = (Ahi * hi).sum(axis=1)
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    lb, w_idx, which, mu = _eval_bounds_from_cached(
        lo, hi, A_tensor, scales, target_c,
        Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
    )
    cache_tuple = (Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, lo, hi)
    return lb, w_idx, which, mu, cache_tuple


def batch_bounds_rank1_lo(
    A_tensor: np.ndarray, scales: np.ndarray,
    parent_cache: tuple,
    new_lo: np.ndarray,
    changed_k: int,
    target_c: float,
):
    """Rank-1 update for the RIGHT child of a split (lo[k] increased).

    Parent's hi is reused (same reference); only new_lo is new.
    """
    p_Alo, p_Ahi, p_loAlo, p_hiAhi, p_lo_sum, p_hi_sum, p_lo, p_hi = parent_cache
    delta = float(new_lo[changed_k] - p_lo[changed_k])
    # new_Alo = p_Alo + A[:, :, k] * delta
    col = A_tensor[:, :, changed_k]  # (W, d)
    Alo = p_Alo + col * delta
    # new_loAlo = p_loAlo + 2*delta*p_Alo[:, k] + delta^2 * A[:, k, k]
    A_kk = A_tensor[:, changed_k, changed_k]  # (W,)
    loAlo = p_loAlo + 2.0 * delta * p_Alo[:, changed_k] + (delta * delta) * A_kk
    Ahi = p_Ahi
    hiAhi = p_hiAhi
    lo_sum = p_lo_sum + delta
    hi_sum = p_hi_sum
    lb, w_idx, which, mu = _eval_bounds_from_cached(
        new_lo, p_hi, A_tensor, scales, target_c,
        Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
    )
    new_cache = (Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, new_lo, p_hi)
    return lb, w_idx, which, mu, new_cache


def batch_bounds_rank1_hi(
    A_tensor: np.ndarray, scales: np.ndarray,
    parent_cache: tuple,
    new_hi: np.ndarray,
    changed_k: int,
    target_c: float,
):
    """Rank-1 update for the LEFT child of a split (hi[k] decreased)."""
    p_Alo, p_Ahi, p_loAlo, p_hiAhi, p_lo_sum, p_hi_sum, p_lo, p_hi = parent_cache
    delta = float(new_hi[changed_k] - p_hi[changed_k])  # negative
    col = A_tensor[:, :, changed_k]
    Ahi = p_Ahi + col * delta
    A_kk = A_tensor[:, changed_k, changed_k]
    hiAhi = p_hiAhi + 2.0 * delta * p_Ahi[:, changed_k] + (delta * delta) * A_kk
    Alo = p_Alo
    loAlo = p_loAlo
    lo_sum = p_lo_sum
    hi_sum = p_hi_sum + delta
    lb, w_idx, which, mu = _eval_bounds_from_cached(
        p_lo, new_hi, A_tensor, scales, target_c,
        Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
    )
    new_cache = (Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, p_lo, new_hi)
    return lb, w_idx, which, mu, new_cache


def batch_bounds_rank1(
    A_tensor: np.ndarray, scales: np.ndarray,
    parent_cache: dict,
    lo: np.ndarray, hi: np.ndarray,
    changed_axis: int, which_endpoint: str,
    target_c: float,
):
    """Compute bounds for a child box whose only change from the parent
    is either `lo[changed_axis]` (right child) or `hi[changed_axis]`
    (left child), via a rank-1 update of Alo / Ahi.

    Returns (best_lb, best_idx, best_which, best_mu, child_cache).
    """
    W, d, _ = A_tensor.shape
    parent_Alo = parent_cache["Alo"]
    parent_Ahi = parent_cache["Ahi"]
    parent_loAlo = parent_cache["loAlo"]
    parent_hiAhi = parent_cache["hiAhi"]
    parent_lo_sum = parent_cache["lo_sum"]
    parent_hi_sum = parent_cache["hi_sum"]
    parent_lo = parent_cache["lo"]
    parent_hi = parent_cache["hi"]

    col = A_tensor[:, :, changed_axis]  # (W, d)
    k = changed_axis
    if which_endpoint == "lo":
        delta = float(lo[k] - parent_lo[k])
        Alo = parent_Alo + col * delta
        Ahi = parent_Ahi  # unchanged
        # loAlo = lo^T A lo. Update:
        # Let l' = l + delta * e_k. Then
        #   l'^T A l' = l^T A l
        #               + delta * (A[:, k, :] l) + delta * (A[:, :, k]^T l)
        #               + delta^2 * A[:, k, k]
        # Since A is symmetric, A[:, k, :] = A[:, :, k]^T. So
        #   l'^T A l' = l^T A l + 2*delta*(A l)[k] + delta^2*A[k,k]
        # but (A l) is per window -> shape (W,). (A l)[:, k] means column k.
        # Actually we need Alo_old (which is A @ l_old) and then
        #   (A @ l_new)[:, j] = (A @ l_old)[:, j] + delta * A[:, j, k]
        # So column k is included. Good.
        # Val l'^T A l' = sum_j l'[j] * (A l')[:, j]
        # Can recompute cheaply: (Alo * lo).sum(axis=1).
        loAlo = (Alo * lo).sum(axis=1)
        hiAhi = parent_hiAhi  # unchanged
        lo_sum = parent_lo_sum + delta
        hi_sum = parent_hi_sum
    elif which_endpoint == "hi":
        delta = float(hi[k] - parent_hi[k])
        Alo = parent_Alo
        Ahi = parent_Ahi + col * delta
        loAlo = parent_loAlo
        hiAhi = (Ahi * hi).sum(axis=1)
        lo_sum = parent_lo_sum
        hi_sum = parent_hi_sum + delta
    else:
        raise ValueError(which_endpoint)

    result = _eval_bounds_from_cached(
        lo, hi, A_tensor, scales, target_c,
        Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
    )
    new_cache = {
        "Alo": Alo, "Ahi": Ahi, "loAlo": loAlo, "hiAhi": hiAhi,
        "lo_sum": lo_sum, "hi_sum": hi_sum,
        "lo": lo, "hi": hi,
    }
    return result + (new_cache,)


def _min_max_linear_box_simplex(g, lo, hi):
    """(min, max) of g^T mu over {sum mu = 1, lo <= mu <= hi} (greedy)."""
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-12 or hi_sum < 1.0 - 1e-12:
        return float("nan"), float("nan")
    remaining = 1.0 - lo_sum
    base = float((g * lo).sum())
    order_min = np.argsort(g, kind="stable")
    val_min = base
    rem = remaining
    for i in order_min:
        if rem <= 0:
            break
        cap = float(hi[i] - lo[i])
        add = cap if cap < rem else rem
        val_min += float(g[i]) * add
        rem -= add
    val_max = base
    rem = remaining
    for i in order_min[::-1]:
        if rem <= 0:
            break
        cap = float(hi[i] - lo[i])
        add = cap if cap < rem else rem
        val_max += float(g[i]) * add
        rem -= add
    return val_min, val_max


def shor_bound_float(lo: np.ndarray, hi: np.ndarray,
                      w: "WindowMeta", d: int) -> float:
    """P4: eigenvalue-Shor LB on min_{mu in box ∩ simplex} mu^T M_W mu.

    Decompose A_W = sum_k lam_k v_k v_k^T (eigh, float). Then
        mu^T M_W mu = scale_W * sum_k lam_k (v_k^T mu)^2.
    Per eigenmode, compute (a_k, b_k) = (min, max) of v_k^T mu over
    box ∩ simplex via greedy LP (O(d log d)); then for each k:
        L_k = 0 if 0 in [a_k, b_k] else min(a_k^2, b_k^2)  (lower bound)
        U_k = max(a_k^2, b_k^2)                            (upper bound)
        contribution = lam_k * L_k if lam_k >= 0 else lam_k * U_k

    Returns float LB; NOT rigorous (uses float eigenvectors). Used as a
    cheap pre-filter to gate the expensive int joint-face LP cert.
    Cost: O(d^3) for eigh + O(d * d log d) for the LPs ≈ 100 us at d=20.
    """
    A = _adjacency_matrix(w, d)
    eigvals, eigvecs = np.linalg.eigh(A)
    lb = 0.0
    for k in range(d):
        lam = float(eigvals[k])
        v = eigvecs[:, k]
        a, b = _min_max_linear_box_simplex(v, lo, hi)
        if not np.isfinite(a) or not np.isfinite(b):
            return float("-inf")
        a2 = a * a
        b2 = b * b
        U_k = a2 if a2 > b2 else b2
        if a <= 0.0 <= b:
            L_k = 0.0
        else:
            L_k = a2 if a2 < b2 else b2
        lb += lam * (L_k if lam >= 0 else U_k)
    if lb < 0:
        lb = 0.0
    return w.scale * lb


def eval_all_window_lbs_from_cached(
    lo: np.ndarray, hi: np.ndarray,
    A_tensor: np.ndarray, scales: np.ndarray,
    Alo: np.ndarray, Ahi: np.ndarray,
    loAlo: np.ndarray, hiAhi: np.ndarray,
    lo_sum: float, hi_sum: float,
) -> np.ndarray:
    """Compute lb_all (per-window max of autoconv, SW, NE) for ALL windows.

    Always evaluates autoconv + SW + NE (no short-circuit). Used by the
    P1-LITE top-K joint-face escalation path: when the primary winning
    window's int rigor fails on a deep box, try int joint-face dual cert
    on alternate top-K windows.

    Cost: same as the full _eval_bounds_from_cached path with both SW and
    NE computed (no early-exit). At d=20 with W~780 windows: ~30 us.
    """
    auto_raw = 1.0 - hi_sum * hi_sum + hiAhi
    lb_auto = scales * np.maximum(auto_raw, 0.0)
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        # Float-infeasible: only autoconv is sound (per existing
        # _eval_bounds_from_cached logic). Caller is responsible for
        # recognising this and not proceeding further.
        return lb_auto
    remaining = 1.0 - lo_sum
    lb_sw = _batch_min_linear_lb_only(
        2.0 * Alo, lo, hi, remaining, scales, loAlo,
    )
    lb_ne = _batch_min_linear_lb_only(
        2.0 * Ahi, lo, hi, remaining, scales, hiAhi,
    )
    return np.maximum(np.maximum(lb_auto, lb_sw), lb_ne)


def _eval_bounds_from_cached(
    lo: np.ndarray, hi: np.ndarray,
    A_tensor: np.ndarray, scales: np.ndarray,
    target_c: float,
    Alo: np.ndarray, Ahi: np.ndarray,
    loAlo: np.ndarray, hiAhi: np.ndarray,
    lo_sum: float, hi_sum: float,
):
    """Three bounds per window, max-combined:
       * autoconv   -- 1 - (sum hi)^2 + hiAhi          (uses simplex)
       * SW McCormick -- min over box-simplex of 2 A lo . mu - loAlo
       * NE McCormick -- min over box-simplex of 2 A hi . mu - hiAhi

    The natural interval bound (scale * loAlo) is dominated pointwise by
    the SW McCormick LP (LP value >= g . lo = 2 loAlo, so LB >= loAlo),
    so we do not compute it separately.
    """
    W, d, _ = A_tensor.shape
    # Soundness contract: the BnB driver must gate with
    # B.intersects_simplex() (exact integer arithmetic) before entering
    # this routine. The float feasibility check below is a cheap
    # secondary guard; a permissive tolerance catches gross misuse while
    # still admitting boxes that are feasible modulo float error.
    assert lo_sum <= 1.0 + 1e-10 and hi_sum >= 1.0 - 1e-10, (
        "bound_eval called on grossly infeasible box -- driver must gate "
        "with B.intersects_simplex() first"
    )
    auto_raw = 1.0 - hi_sum * hi_sum + hiAhi
    lb_auto = scales * np.maximum(auto_raw, 0.0)

    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        # Box is float-infeasible but passed the exact-integer gate by
        # at most one ulp. The autoconv bound is still a valid LB on the
        # (possibly tiny) box-intersect-simplex, so we return it; future
        # driver refactors that skip the exact gate would harmlessly
        # fall through here rather than silently certify garbage.
        best_idx = int(np.argmax(lb_auto))
        return float(lb_auto[best_idx]), best_idx, "autoconv", None

    # Cheap autoconv may already certify.
    if float(lb_auto.max()) >= target_c:
        best_idx = int(np.argmax(lb_auto))
        return float(lb_auto[best_idx]), best_idx, "autoconv", None

    remaining = 1.0 - lo_sum
    lb_sw = _batch_min_linear_lb_only(
        2.0 * Alo, lo, hi, remaining, scales, loAlo,
    )
    lb_best = np.maximum(lb_auto, lb_sw)
    best_idx = int(np.argmax(lb_best))
    best_lb = float(lb_best[best_idx])

    # SW + autoconv certifies: skip NE face.
    if best_lb >= target_c:
        which = "mccormick" if lb_sw[best_idx] >= lb_auto[best_idx] else "autoconv"
        return best_lb, best_idx, which, None

    lb_ne = _batch_min_linear_lb_only(
        2.0 * Ahi, lo, hi, remaining, scales, hiAhi,
    )
    lb_mcc = np.maximum(lb_sw, lb_ne)
    lb_all = np.maximum(lb_auto, lb_mcc)
    best_idx = int(np.argmax(lb_all))
    best_lb = float(lb_all[best_idx])
    which = "mccormick" if lb_mcc[best_idx] >= lb_auto[best_idx] else "autoconv"
    return best_lb, best_idx, which, None


def _batch_min_linear_over_simplex(
    G: np.ndarray, lo: np.ndarray, hi: np.ndarray, remaining: float,
    scales: np.ndarray, c0_raw: np.ndarray,
):
    """For each row g_w of G (shape (W, d)), solve
        min g_w . mu  over { sum mu = 1, lo <= mu <= hi }
    by greedy sort. Returns:
        lb (W,): scales * (min_val - c0_raw)
        (sort_idx, added, sorted_cap): raw arrays so the caller can
          reconstruct mu_star for one winning window later, avoiding
          the expensive np.add.at scatter for every window.
    c0_raw is WITHOUT the minus sign. For SW face, c0_raw = loAlo;
    for NE face, c0_raw = hiAhi.
    """
    W, d = G.shape
    base_val = (G * lo).sum(axis=1)
    if remaining <= 1e-15:
        lb = scales * (base_val - c0_raw)
        return lb, None
    sort_idx = np.argsort(G, axis=1, kind="stable")
    sorted_g = np.take_along_axis(G, sort_idx, axis=1)
    cap_bcast = np.broadcast_to(hi - lo, (W, d))
    sorted_cap = np.take_along_axis(cap_bcast, sort_idx, axis=1)
    cum_cap = np.cumsum(sorted_cap, axis=1)
    cum_prev = np.empty_like(cum_cap)
    cum_prev[:, 0] = 0.0
    cum_prev[:, 1:] = cum_cap[:, :-1]
    added = np.clip(remaining - cum_prev, 0.0, sorted_cap)
    val_delta = (sorted_g * added).sum(axis=1)
    val = base_val + val_delta
    lb = scales * (val - c0_raw)
    return lb, (sort_idx, added)


def bound_mccormick_joint_face_dual_cert_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: "WindowMeta", d: int,
    target_num: int, target_den: int,
) -> bool:
    """Joint-face McCormick LB with dual certificate (integer-rigor safe).

    Solves in scipy.linprog HiGHS the tighter LP

        min_{mu,y}  sum_p y_p
        s.t.  y_p >= SW_p(mu),  y_p >= NE_p(mu),  (P ordered pairs)
              sum mu = 1, lo <= mu <= hi
              y_p free

    then extracts duals (ineqlin, eqlin marginals from scipy) and
    applies the Neumaier-Shcherbina correction: round scipy duals to
    denom 2^60, enforce stationarity exactly for FREE y_p variables,
    and absorb the residual at mu_k into lower/upper bound duals
    (which can take any value since mu has finite box bounds).

    Returns True iff the resulting DUAL bound

        LB_cert = b_ub . ineqlin + b_eq . eqlin
                + l_mu . lower_mu + u_mu . upper_mu

    satisfies  w.scale_q * LB_cert >= target_q  in exact integer
    arithmetic. This is ALWAYS a valid LB on primal (weak duality), so
    the certification is rigor-parity safe regardless of LP numerical
    accuracy.

    Short-circuits at several points:
      * P == 0: trivially cannot certify a positive target.
      * LP fails or float LP * scale < target * 0.999: skip cert.
      * Rounded dual sign-infeasible (rare, after clamp fallback).
    """
    from scipy.optimize import linprog
    pairs = w.pairs_all
    P = len(pairs)
    if P == 0:
        return False
    # Build the LP in float.
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for k in range(d):
        lo[k] = lo_int[k] / _SCALE
        hi[k] = hi_int[k] / _SCALE
    n_vars = P + d
    c = np.zeros(n_vars)
    c[:P] = 1.0
    A_ub = np.zeros((2 * P, n_vars))
    b_ub = np.zeros(2 * P)
    for p, (i, j) in enumerate(pairs):
        A_ub[p, p] = -1.0
        A_ub[p, P + i] += float(lo[j])
        A_ub[p, P + j] += float(lo[i])
        b_ub[p] = float(lo[i]) * float(lo[j])
        A_ub[P + p, p] = -1.0
        A_ub[P + p, P + i] += float(hi[j])
        A_ub[P + p, P + j] += float(hi[i])
        b_ub[P + p] = float(hi[i]) * float(hi[j])
    A_eq = np.zeros((1, n_vars))
    A_eq[0, P:] = 1.0
    b_eq = np.array([1.0])
    bounds = [(None, None)] * P + [(float(lo[k]), float(hi[k])) for k in range(d)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return False
    if not res.success:
        return False
    # Early bail: float LP value * scale can't reach target, even exact dual
    # won't be above LP value. Tolerance guards against tiny float error.
    target_f = target_num / target_den
    if w.scale * float(res.fun) < target_f * (1.0 - 1e-9):
        return False

    # --- Dual certificate: round, enforce stationarity, compute LB ---
    ineqlin = res.ineqlin.marginals  # (2P,), scipy sign: <= 0
    eqlin = res.eqlin.marginals      # (1,), free
    D = _SCALE  # denominator 2^60

    def _round(x: float) -> int:
        return int(round(float(x) * D))

    # scipy.linprog returns scipy_ineqlin <= 0 for <= constraints.
    # Stationarity for FREE variable y_p (col has -1 at rows p and P+p):
    #   c_{y_p} = A_ub^T . scipy_ineqlin  at y_p
    #   1 = -scipy_ineqlin[p] - scipy_ineqlin[P+p]
    # In integer at denom D: 1 * D = -sw_num[p] - ne_num[p]
    # i.e. sw_num[p] + ne_num[p] = -D.
    sw_num = [_round(ineqlin[p]) for p in range(P)]
    ne_num = [-D - sw_num[p] for p in range(P)]
    # Sign-feasibility: both sw_num[p] <= 0 AND ne_num[p] <= 0.
    for p in range(P):
        if sw_num[p] > 0 or ne_num[p] > 0:
            # Rounding pushed out of the feasible sign region. Fallback:
            # split the -D equally. Both become -D/2, satisfying signs.
            sw_num[p] = -(D >> 1)
            ne_num[p] = -D - sw_num[p]
    nu_num = _round(eqlin[0])

    # Residual r_mu[k] = c_{mu_k} - (A_ub^T ineqlin)_{mu_k} - (A_eq^T eqlin)_{mu_k}
    # c_{mu_k} = 0.
    # A_ub^T ineqlin at mu_k (denom _SCALE2 after multiplying lo_int * sw_num):
    #   SW row p=(i,j): column k has lo[j] if i==k + lo[i] if j==k.
    #     At denom _SCALE2: (lo_int[j] if i==k else 0 + lo_int[i] if j==k else 0) * sw_num[p]
    #   NE row similarly with hi_int.
    # A_eq^T eqlin at mu_k (denom D): 1 * nu_num. To match _SCALE2, multiply by _SCALE.
    r_mu = [0] * d  # at denom _SCALE2
    for p, (i, j) in enumerate(pairs):
        sw = sw_num[p]
        ne = ne_num[p]
        # (lo_int[j] * sw + hi_int[j] * ne) contributes to coord i via A_ub[p, P+i]
        #   coef: lo[j] in SW row p, hi[j] in NE row p. Both added to mu_i.
        r_mu[i] -= lo_int[j] * sw + hi_int[j] * ne
        # (lo_int[i] * sw + hi_int[i] * ne) contributes to coord j.
        r_mu[j] -= lo_int[i] * sw + hi_int[i] * ne
    for k in range(d):
        r_mu[k] -= nu_num * _SCALE  # A_eq^T eqlin at denom _SCALE2

    # Stationarity: c_{mu_k} = A_ub^T ineqlin + A_eq^T eqlin + lower[k] + upper[k]
    # So lower[k] + upper[k] = -(A_ub^T ineqlin + A_eq^T eqlin)_{mu_k} = r_mu[k].
    # To be dual-feasible with lower >= 0, upper <= 0: split r_mu[k].
    # lower_num[k] = max(r_mu[k], 0); upper_num[k] = min(r_mu[k], 0). Denom _SCALE2.

    # Certified LB (denom _SCALE * _SCALE2 = _SCALE3):
    #   LB = b_ub . ineqlin + b_eq . eqlin + l_mu . lower + u_mu . upper
    #   b_ub[SW p] = lo_int[i] * lo_int[j] / _SCALE2. ineqlin = sw_num[p] / _SCALE.
    #   Product at denom _SCALE3: lo_int[i] lo_int[j] sw_num[p].
    #   b_eq = 1, eqlin = nu_num / _SCALE. At denom _SCALE3: nu_num * _SCALE2.
    #   l_mu[k] = lo_int[k] / _SCALE, lower_num[k] / _SCALE2. Product _SCALE3:
    #     lo_int[k] * lower_num[k].
    lb_num = 0  # at denom _SCALE3 = 2**180
    for p, (i, j) in enumerate(pairs):
        lb_num += lo_int[i] * lo_int[j] * sw_num[p]
        lb_num += hi_int[i] * hi_int[j] * ne_num[p]
    lb_num += nu_num * _SCALE2
    for k in range(d):
        r = r_mu[k]
        if r >= 0:
            lb_num += lo_int[k] * r  # lower_num = r, upper_num = 0
        else:
            lb_num += hi_int[k] * r  # lower_num = 0, upper_num = r
    # Bound = scale_q * lb_num / _SCALE3. Cert iff >= target_q.
    # scale_q.num * lb_num * target_den >= target_num * scale_q.den * _SCALE3
    _SCALE3 = _SCALE * _SCALE2
    lhs = w.scale_q.numerator * lb_num * target_den
    rhs = target_num * w.scale_q.denominator * _SCALE3
    return lhs >= rhs


def bound_mccormick_joint_face_lp(
    lo: np.ndarray, hi: np.ndarray,
    w: "WindowMeta", scale: float,
) -> float:
    """Dual-face McCormick LP: min_{mu, y} sum y_p
        s.t.  y_p >= SW_p(mu)   (SW-face per pair)
              y_p >= NE_p(mu)   (NE-face per pair)
              sum mu = 1,  lo <= mu <= hi

    Uses scipy.linprog with HiGHS. Auxiliary variables y_p linearize
    the pointwise `max(SW, NE)` for each pair, giving a strictly
    tighter LB than max(sep_SW_min, sep_NE_min).

    Valid because at the LP minimum y_p = max(SW_p(mu), NE_p(mu)),
    so sum y_p = sum max(SW_p, NE_p) <= mu^T M_W mu / scale pointwise.
    """
    from scipy.optimize import linprog
    pairs = w.pairs_all  # ordered pairs, double-counts off-diagonal
    P = len(pairs)
    d = lo.shape[0]
    if P == 0:
        return 0.0
    # x = [y_0, ..., y_{P-1}, mu_0, ..., mu_{d-1}]
    n_vars = P + d
    # Objective: minimise sum of y_p.
    c = np.zeros(n_vars)
    c[:P] = 1.0
    # Inequality rows: -y_p + lo_j * mu_i + lo_i * mu_j <= lo_i lo_j (SW)
    #                  -y_p + hi_j * mu_i + hi_i * mu_j <= hi_i hi_j (NE)
    A_ub = np.zeros((2 * P, n_vars))
    b_ub = np.zeros(2 * P)
    for p, (i, j) in enumerate(pairs):
        # SW
        A_ub[p, p] = -1.0
        A_ub[p, P + i] += float(lo[j])
        A_ub[p, P + j] += float(lo[i])
        b_ub[p] = float(lo[i]) * float(lo[j])
        # NE
        A_ub[P + p, p] = -1.0
        A_ub[P + p, P + i] += float(hi[j])
        A_ub[P + p, P + j] += float(hi[i])
        b_ub[P + p] = float(hi[i]) * float(hi[j])
    # Equality: sum mu = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, P:] = 1.0
    b_eq = np.array([1.0])
    # Bounds: y unbounded below (may be negative); mu in [lo, hi].
    bounds = [(None, None)] * P + [(float(lo[k]), float(hi[k])) for k in range(d)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return float("-inf")
    if not res.success:
        return float("-inf")
    return scale * float(res.fun)


def bound_autoconv_plus_lp(
    lo: np.ndarray, hi: np.ndarray,
    w: "WindowMeta", d: int, scale: float,
) -> float:
    """Tighter autoconv LB via RLT-LP (Sherali-Adams level 1).

    Minimises Σ_{(i,j) ∈ pairs(W)} Y_{ij} over

        l ≤ μ ≤ h,  Σ μ = 1,
        Y_{ij} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j  (SW McC)
        Y_{ij} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j  (NE McC)
        Y_{ij} ≤ hi_j μ_i + lo_i μ_j − lo_i hi_j  (SE McC)
        Y_{ij} ≤ lo_j μ_i + hi_i μ_j − lo_j hi_i  (NW McC)
        Y_{ij} ≥ 0                                (μ ≥ 0 ⇒ μμ ≥ 0)
        Σ_j Y_{ij} = μ_i                          (RLT: μ_i · Σ μ = μ_i)

    Returns scale × LP_min, floored at 0. Strictly dominates the
    "hi·hi" upper bound used by the standard autoconv (any feasible
    point of the LP with Y_{ij} = μ_i μ_j is feasible, so LP_min is a
    valid LB; the added RLT equations tighten it).

    Equivalent reformulation: max Σ_{(i,j) ∉ pairs(W)} Y_{ij} and
    autoconv⁺ = 1 − max. Uses Σ_{all i,j} Y_{ij} = Σ_i μ_i = 1 via RLT.
    """
    from scipy.optimize import linprog
    pairs_support = set(w.pairs_all)
    # Variable layout: [μ_0..μ_{d-1}, Y_{0,0}, Y_{0,1}, ..., Y_{d-1,d-1}]
    n_mu = d
    n_y = d * d
    n_vars = n_mu + n_y
    y_idx = lambda i, j: n_mu + i * d + j

    # Objective: min Σ Y_{ij} for (i, j) in support.
    c = np.zeros(n_vars)
    for (i, j) in pairs_support:
        c[y_idx(i, j)] = 1.0

    # Inequality rows: 4 McCormick faces per pair.
    # Build as list first, then convert (scipy accepts None for empties).
    rows: list = []
    rhs: list = []
    lo_f = lo.astype(np.float64, copy=False)
    hi_f = hi.astype(np.float64, copy=False)
    for i in range(d):
        for j in range(d):
            li, lj = float(lo_f[i]), float(lo_f[j])
            hi_i, hj = float(hi_f[i]), float(hi_f[j])
            # SW:  Y ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j
            #  →  −Y + lo_j μ_i + lo_i μ_j ≤ lo_i lo_j
            row = np.zeros(n_vars); row[y_idx(i, j)] = -1.0
            row[i] += lj; row[j] += li
            rows.append(row); rhs.append(li * lj)
            # NE:  Y ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j
            #  →  −Y + hi_j μ_i + hi_i μ_j ≤ hi_i hi_j
            row = np.zeros(n_vars); row[y_idx(i, j)] = -1.0
            row[i] += hj; row[j] += hi_i
            rows.append(row); rhs.append(hi_i * hj)
            # SE:  Y ≤ hi_j μ_i + lo_i μ_j − lo_i hi_j
            #  →   Y − hi_j μ_i − lo_i μ_j ≤ −lo_i hi_j
            row = np.zeros(n_vars); row[y_idx(i, j)] = 1.0
            row[i] += -hj; row[j] += -li
            rows.append(row); rhs.append(-li * hj)
            # NW:  Y ≤ lo_j μ_i + hi_i μ_j − lo_j hi_i
            #  →   Y − lo_j μ_i − hi_i μ_j ≤ −lo_j hi_i
            row = np.zeros(n_vars); row[y_idx(i, j)] = 1.0
            row[i] += -lj; row[j] += -hi_i
            rows.append(row); rhs.append(-lj * hi_i)
    A_ub = np.asarray(rows, dtype=np.float64)
    b_ub = np.asarray(rhs, dtype=np.float64)

    # Equality: Σ μ = 1 AND Σ_j Y_{i,j} = μ_i for each i.
    A_eq = np.zeros((1 + d, n_vars), dtype=np.float64)
    b_eq = np.zeros(1 + d, dtype=np.float64)
    A_eq[0, :n_mu] = 1.0
    b_eq[0] = 1.0
    for i in range(d):
        A_eq[1 + i, i] = -1.0
        for j in range(d):
            A_eq[1 + i, y_idx(i, j)] = 1.0

    bounds = [(float(lo_f[k]), float(hi_f[k])) for k in range(d)]
    bounds += [(0.0, None)] * n_y

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return float("-inf")
    if not res.success:
        return float("-inf")
    val = float(res.fun)
    if val < 0.0:
        val = 0.0
    return scale * val


def _batch_min_linear_lb_only_numpy(
    G: np.ndarray, lo: np.ndarray, hi: np.ndarray, remaining: float,
    scales: np.ndarray, c0_raw: np.ndarray,
) -> np.ndarray:
    """Pure-numpy LB of min g.mu over {sum mu = 1, lo <= mu <= hi}.
    Each row g of G corresponds to a window; the greedy LP fills mass
    from lowest-g coord upward until remaining is exhausted.
    """
    W, d = G.shape
    base_val = (G * lo).sum(axis=1)
    if remaining <= 1e-15:
        return scales * (base_val - c0_raw)
    sort_idx = np.argsort(G, axis=1, kind="stable")
    sorted_g = np.take_along_axis(G, sort_idx, axis=1)
    cap_bcast = np.broadcast_to(hi - lo, (W, d))
    sorted_cap = np.take_along_axis(cap_bcast, sort_idx, axis=1)
    cum_cap = np.cumsum(sorted_cap, axis=1)
    cum_prev = np.empty_like(cum_cap)
    cum_prev[:, 0] = 0.0
    cum_prev[:, 1:] = cum_cap[:, :-1]
    added = np.clip(remaining - cum_prev, 0.0, sorted_cap)
    val_delta = (sorted_g * added).sum(axis=1)
    return scales * (base_val + val_delta - c0_raw)


try:
    import numba as _numba  # noqa: F401

    @_numba.njit(cache=True, fastmath=True, boundscheck=False)
    def _batch_min_linear_lb_only_numba(
        G, lo, hi, remaining, scales, c0_raw,
    ):
        """Numba-JIT equivalent of the numpy version. About 3-5x faster
        per call at d=16 because the per-window sort+fill runs as a
        tight native loop instead of Python-level vector ops with their
        per-call overhead.
        """
        W = G.shape[0]
        d = G.shape[1]
        lb_out = np.empty(W, dtype=np.float64)
        cap = np.empty(d, dtype=np.float64)
        for i in range(d):
            cap[i] = hi[i] - lo[i]
        if remaining <= 1e-15:
            for w in range(W):
                s = 0.0
                for i in range(d):
                    s += G[w, i] * lo[i]
                lb_out[w] = scales[w] * (s - c0_raw[w])
            return lb_out
        # Workspace
        order = np.empty(d, dtype=np.int64)
        # Per window: base g.lo, then greedy fill.
        for w in range(W):
            # Initialise order = [0..d-1]
            for i in range(d):
                order[i] = i
            # Insertion sort by G[w, order[i]] ascending — fast for small d.
            for i in range(1, d):
                key_idx = order[i]
                key_val = G[w, key_idx]
                j = i - 1
                while j >= 0 and G[w, order[j]] > key_val:
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = key_idx
            base = 0.0
            for i in range(d):
                base += G[w, i] * lo[i]
            rem = remaining
            val_delta = 0.0
            for t in range(d):
                if rem <= 0.0:
                    break
                idx = order[t]
                c = cap[idx]
                if c <= rem:
                    val_delta += G[w, idx] * c
                    rem -= c
                else:
                    val_delta += G[w, idx] * rem
                    rem = 0.0
            lb_out[w] = scales[w] * (base + val_delta - c0_raw[w])
        return lb_out

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _batch_min_linear_lb_only(
    G: np.ndarray, lo: np.ndarray, hi: np.ndarray, remaining: float,
    scales: np.ndarray, c0_raw: np.ndarray,
) -> np.ndarray:
    """Route to Numba if available, else pure numpy."""
    if _HAVE_NUMBA:
        return _batch_min_linear_lb_only_numba(
            G, lo, hi, remaining, scales, c0_raw,
        )
    return _batch_min_linear_lb_only_numpy(
        G, lo, hi, remaining, scales, c0_raw,
    )


def _reconstruct_mu_star(lo: np.ndarray, payload, window_idx: int) -> np.ndarray:
    """Reconstruct mu_star for a single window from the batched sort payload."""
    if payload is None:
        return lo.copy()
    sort_idx, added = payload
    s_idx = sort_idx[window_idx]
    add_row = added[window_idx]
    mu = lo.copy()
    # Non-scattered: s_idx is a permutation of [0..d-1], each unique.
    # So simple indexed assignment works.
    mu[s_idx] += add_row
    return mu


def _mccormick_linear_exact(
    lo_q: Sequence[Fraction], w: WindowMeta
) -> Tuple[List[Fraction], Fraction]:
    d = len(lo_q)
    g = [Fraction(0) for _ in range(d)]
    c0 = Fraction(0)
    # Diagonal mu_i^2: tangent under-estimator (valid LB; secant would
    # be an UPPER bound since x^2 is convex).
    for (i, j) in w.pairs_all:
        g[i] += lo_q[j]
        g[j] += lo_q[i]
        c0 -= lo_q[i] * lo_q[j]
    return g, c0


def bound_mccormick_exact_nosym(
    lo_q: Sequence[Fraction], hi_q: Sequence[Fraction], w: WindowMeta,
) -> Fraction:
    """Exact Fraction McCormick LB over {sum mu = 1, lo_q <= mu <= hi_q}
    (ignoring symmetry cuts -- a weaker LP than the fast path but still
    a valid LB). Solved via greedy sort on the linear coefficient g.
    Uses the SW face only."""
    g, c0 = _mccormick_linear_exact(lo_q, w)
    val = _rational_lp_simplex_box(g, lo_q, hi_q)
    if val is None:
        # Empty box-simplex intersection: vacuously certifies any finite
        # target. Return a large sentinel rather than Fraction(0), which
        # would silently mis-report if an upstream gate is ever skipped.
        return _FRACTION_INF
    return w.scale_q * (val + c0)


def bound_mccormick_ne_exact_nosym(
    lo_q: Sequence[Fraction], hi_q: Sequence[Fraction], w: WindowMeta,
) -> Fraction:
    """Exact Fraction NE-face McCormick LB.

    For each ordered pair (i,j) in support(W),
        mu_i mu_j >= hi_j mu_i + hi_i mu_j - hi_i hi_j.
    Summing gives a linear LB scale * (g_NE . mu + c0_NE) with
    g_NE = 2 * A_W . hi and c0_NE = -hi^T A_W hi. Minimise over
    {sum mu = 1, lo <= mu <= hi} by greedy sort.
    """
    d = len(lo_q)
    # Build g_NE and c0_NE in exact arithmetic.
    g = [Fraction(0) for _ in range(d)]
    c0 = Fraction(0)
    # Diagonal mu_i^2: tangent under-estimator at hi_i (valid LB; secant
    # would be an UPPER bound since x^2 is convex).
    for (i, j) in w.pairs_all:
        g[i] += hi_q[j]
        g[j] += hi_q[i]
        c0 -= hi_q[i] * hi_q[j]
    val = _rational_lp_simplex_box(g, lo_q, hi_q)
    if val is None:
        # Empty box-simplex intersection: vacuously certifies any finite
        # target. See bound_mccormick_exact_nosym for rationale.
        return _FRACTION_INF
    return w.scale_q * (val + c0)


def _rational_lp_simplex_box(
    g: Sequence[Fraction],
    lo_q: Sequence[Fraction],
    hi_q: Sequence[Fraction],
) -> Optional[Fraction]:
    """Compute min g.mu over {sum mu = 1, lo_q <= mu <= hi_q} exactly.

    Returns the minimum value, or None if the domain is empty.
    """
    lo_sum = sum(lo_q, Fraction(0))
    hi_sum = sum(hi_q, Fraction(0))
    one = Fraction(1)
    if lo_sum > one or hi_sum < one:
        return None
    d = len(g)
    # Sort indices ascending by g (most mass to smallest coefficient).
    order = sorted(range(d), key=lambda i: g[i])
    remaining = one - lo_sum
    val = sum(g[i] * lo_q[i] for i in range(d))
    for i in order:
        if remaining == 0:
            break
        cap = hi_q[i] - lo_q[i]
        add = cap if cap < remaining else remaining
        val += g[i] * add
        remaining -= add
    return val


# ---------------------------------------------------------------------
# Integer-arithmetic rigor gates (replaces Fraction bounds with raw
# Python ints at shared denominator 2**D_SHIFT). Mathematically
# identical to the Fraction path, 5--10x faster because no per-op GCD.
# Each `_ge` helper returns True iff the corresponding bound is
# >= target_q; we compare numerators directly in int arithmetic.
#
# Bound <expr> at denom 2^120, scale = scale_num / scale_den:
#     bound = scale_num * <expr> / (scale_den * 2^120)
#     bound >= target_num / target_den
#   iff scale_num * <expr> * target_den >= target_num * scale_den * 2^120
# ---------------------------------------------------------------------

from .box import SCALE as _SCALE, D_SHIFT as _D_SHIFT  # noqa: E402
_SCALE2 = _SCALE * _SCALE  # 2**120


def bound_natural_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, target_num: int, target_den: int,
) -> bool:
    """Natural interval LB >= target_q, all in integer arithmetic."""
    total = 0
    for (i, j) in w.pairs_all:
        total += lo_int[i] * lo_int[j]
    # bound = scale_q * total / 2^120
    lhs = w.scale_q.numerator * total * target_den
    rhs = target_num * w.scale_q.denominator * _SCALE2
    return lhs >= rhs


def bound_autoconv_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, d: int, target_num: int, target_den: int,
) -> bool:
    """Autoconv LB >= target_q, integer arithmetic.

    bound = scale * (1 - Σ_{(i,j) not in W} hi_i hi_j)
          = scale * (2^120 - Σ_comp hi_int[i] hi_int[j]) / 2^120.
    """
    in_support = _pair_in_support_mask(w, d)
    comp = 0
    for i in range(d):
        hi_i = hi_int[i]
        for j in range(d):
            if not in_support[i, j]:
                comp += hi_i * hi_int[j]
    if comp >= _SCALE2:
        return False  # clamped bound would be <= 0
    val_num = _SCALE2 - comp  # at denom 2^120
    lhs = w.scale_q.numerator * val_num * target_den
    rhs = target_num * w.scale_q.denominator * _SCALE2
    return lhs >= rhs


def _mccormick_sw_int_lp(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, d: int,
) -> Optional[int]:
    """Return the SW-McCormick LP minimum as an integer at denominator
    2**120, or None if the box-simplex domain is empty.

    g_int[k] = Σ lo_int[j] over ordered pairs (k, _) and (_, k) in supp W.
    c0_int   = - Σ_{(i,j) in pairs_all} lo_int[i] lo_int[j].
    LP min = greedy sort on g_int, fill mass from lo_int up to hi_int
             until Σ mu_int = 2**D_SHIFT.
    Full numerator at denom 2^120: g_int . mu_int + c0_int.
    """
    g_int = [0] * d
    c0_int = 0
    for (i, j) in w.pairs_all:
        g_int[i] += lo_int[j]
        g_int[j] += lo_int[i]
        c0_int -= lo_int[i] * lo_int[j]
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > _SCALE or hi_sum < _SCALE:
        return None
    remaining = _SCALE - lo_sum
    # Base LP val (at mu = lo_int): Σ g_int[k] * lo_int[k]
    val = 0
    for k in range(d):
        val += g_int[k] * lo_int[k]
    if remaining > 0:
        order = sorted(range(d), key=lambda k: g_int[k])
        for k in order:
            if remaining <= 0:
                break
            cap = hi_int[k] - lo_int[k]
            add = cap if cap < remaining else remaining
            val += g_int[k] * add
            remaining -= add
    return val + c0_int


def _mccormick_ne_int_lp(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, d: int,
) -> Optional[int]:
    """NE face: g based on hi_int instead of lo_int."""
    g_int = [0] * d
    c0_int = 0
    for (i, j) in w.pairs_all:
        g_int[i] += hi_int[j]
        g_int[j] += hi_int[i]
        c0_int -= hi_int[i] * hi_int[j]
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > _SCALE or hi_sum < _SCALE:
        return None
    remaining = _SCALE - lo_sum
    val = 0
    for k in range(d):
        val += g_int[k] * lo_int[k]
    if remaining > 0:
        order = sorted(range(d), key=lambda k: g_int[k])
        for k in order:
            if remaining <= 0:
                break
            cap = hi_int[k] - lo_int[k]
            add = cap if cap < remaining else remaining
            val += g_int[k] * add
            remaining -= add
    return val + c0_int


def bound_mccormick_sw_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, d: int, target_num: int, target_den: int,
) -> bool:
    val = _mccormick_sw_int_lp(lo_int, hi_int, w, d)
    if val is None:
        # Empty domain: vacuously certifies any finite target.
        return True
    lhs = w.scale_q.numerator * val * target_den
    rhs = target_num * w.scale_q.denominator * _SCALE2
    return lhs >= rhs


def bound_mccormick_ne_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    w: WindowMeta, d: int, target_num: int, target_den: int,
) -> bool:
    val = _mccormick_ne_int_lp(lo_int, hi_int, w, d)
    if val is None:
        return True
    lhs = w.scale_q.numerator * val * target_den
    rhs = target_num * w.scale_q.denominator * _SCALE2
    return lhs >= rhs


def bound_mccormick_lp(
    lo: np.ndarray, hi: np.ndarray, w: WindowMeta,
    sym_cuts: Sequence[Tuple[int, int]] = (),
) -> Tuple[float, Optional[np.ndarray]]:
    """Minimise the McCormick LB over {sum mu = 1, lo <= mu <= hi}.

    Fast sort-based LP (ignoring sym_cuts -- these are enforced by
    the initial box via Option C, so LP sees a weaker but still valid
    relaxation). Returns (lb, mu_star); lb = -inf if infeasible.

    The sym_cuts argument is kept for API compatibility but no longer
    used: including it would tighten the bound but prevent the O(d log d)
    closed-form LP solution and require a linear program (~100x slower).
    """
    del sym_cuts  # unused; see docstring.
    g, c0 = _mccormick_linear(lo, w)
    # Closed-form: fill mass from smallest g[i] up.
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    # Permissive guard: normal BnB usage has already gated via
    # B.intersects_simplex() (exact integer). A gross violation here
    # indicates misuse and must not silently certify.
    assert lo_sum <= 1.0 + 1e-10 and hi_sum >= 1.0 - 1e-10, (
        "bound_mccormick_lp called on grossly infeasible box -- driver "
        "must gate with B.intersects_simplex() first"
    )
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        # Float-infeasible: return a value that CANNOT certify.
        return float("-inf"), None
    remaining = 1.0 - lo_sum
    mu = lo.copy()
    val = float(g @ mu)
    if remaining > 0.0:
        order = np.argsort(g, kind="stable")
        for i in order:
            if remaining <= 0.0:
                break
            cap = hi[i] - lo[i]
            if cap <= remaining:
                mu[i] += cap
                val += g[i] * cap
                remaining -= cap
            else:
                mu[i] += remaining
                val += g[i] * remaining
                remaining = 0.0
    lb = w.scale * (val + c0)
    return lb, mu


# ---------------------------------------------------------------------
# Multi-window evaluation
# ---------------------------------------------------------------------

def bound_box_multiwindow(
    lo: np.ndarray, hi: np.ndarray,
    windows: Sequence[WindowMeta],
    sym_cuts: Sequence[Tuple[int, int]] = (),
    target_c: Optional[float] = None,
    order: Optional[Sequence[int]] = None,
    method: str = "combined",
    d: Optional[int] = None,
) -> Tuple[float, int, Optional[np.ndarray], str]:
    """Return (best_lb, best_window_index, mu_star_for_best, which_bound).

    method choices:
      "natural"   -- sum of lo_i lo_j * scale      (weakest; no simplex)
      "autoconv"  -- 1 - sum_{i+j not in W} hi_i hi_j (uses simplex)
      "mccormick" -- linear McCormick + greedy LP (mid; no sym cuts)
      "combined"  -- max of all three              (sharpest)

    In "combined", each window evaluates the cheap bounds (natural,
    autoconv) first, and only computes the expensive McCormick LP if
    neither already reaches target_c. The outer scan also short-circuits
    the moment any window certifies lb >= target_c.
    """
    if d is None:
        d = len(lo)
    # Cached scalars for the autoconv bound (sum_hi and its square):
    sum_hi = float(hi.sum()) if method in ("autoconv", "combined") else 0.0
    sum_hi_sq = sum_hi * sum_hi
    best_lb = float("-inf")
    best_idx = -1
    best_mu: Optional[np.ndarray] = None
    best_which = "none"
    idx_iter: Sequence[int] = order if order is not None else range(len(windows))
    target = float("inf") if target_c is None else float(target_c)
    for k in idx_iter:
        w = windows[k]
        mu_mcc: Optional[np.ndarray] = None
        # Cheap bounds first.
        if method in ("natural", "combined"):
            A = _adjacency_matrix(w, d)
            loAlo = float(lo @ A @ lo)
            lb_nat = w.scale * loAlo
        else:
            lb_nat = float("-inf")
        if method in ("autoconv", "combined"):
            if method == "autoconv":
                A = _adjacency_matrix(w, d)
            hAh = float(hi @ A @ hi)
            auto_raw = 1.0 - sum_hi_sq + hAh
            lb_auto = w.scale * (auto_raw if auto_raw > 0.0 else 0.0)
        else:
            lb_auto = float("-inf")
        lb_cheap = lb_nat if lb_nat >= lb_auto else lb_auto
        which_cheap = "natural" if lb_nat >= lb_auto else "autoconv"
        # McCormick only if cheap bounds insufficient to certify.
        if method in ("mccormick", "combined"):
            if method == "combined" and lb_cheap >= target and lb_cheap > best_lb:
                lb_mcc = float("-inf")  # skip expensive LP
            else:
                if method == "mccormick":
                    A = _adjacency_matrix(w, d)
                # g = 2 * A @ lo ; c0 = -lo @ A @ lo
                Alo = A @ lo
                g = 2.0 * Alo
                c0 = -(lb_nat / w.scale if method == "combined" else float(lo @ Alo))
                lb_mcc, mu_mcc = _mccormick_sort_lp(lo, hi, g, c0, w.scale)
        else:
            lb_mcc = float("-inf")
        # Pick best bound for this window.
        if lb_mcc >= lb_cheap:
            lb, which = lb_mcc, "mccormick"
        else:
            lb, which = lb_cheap, which_cheap
            mu_mcc = None
        if lb > best_lb:
            best_lb = lb
            best_idx = k
            best_mu = mu_mcc
            best_which = which
            if best_lb >= target:
                return best_lb, best_idx, best_mu, best_which
    return best_lb, best_idx, best_mu, best_which


def _mccormick_sort_lp(
    lo: np.ndarray, hi: np.ndarray, g: np.ndarray, c0: float, scale: float,
) -> Tuple[float, Optional[np.ndarray]]:
    """Greedy-sort LP: min g.mu + c0 over {sum mu = 1, lo <= mu <= hi}.
    Returns (scale*(min_val + c0), mu_star)."""
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    # Permissive guard: driver must gate with B.intersects_simplex()
    # (exact integer) before reaching here. A gross violation is a bug.
    assert lo_sum <= 1.0 + 1e-10 and hi_sum >= 1.0 - 1e-10, (
        "_mccormick_sort_lp called on grossly infeasible box -- driver "
        "must gate with B.intersects_simplex() first"
    )
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        # Float-infeasible: return a value that CANNOT certify.
        return float("-inf"), None
    remaining = 1.0 - lo_sum
    mu = lo.copy()
    val = float(g @ mu)
    if remaining > 0.0:
        order = np.argsort(g, kind="stable")
        for i in order:
            if remaining <= 0.0:
                break
            cap = hi[i] - lo[i]
            if cap <= remaining:
                mu[i] += cap
                val += g[i] * cap
                remaining -= cap
            else:
                mu[i] += remaining
                val += g[i] * remaining
                remaining = 0.0
    return scale * (val + c0), mu


def _best_of(lb_nat: float, lb_auto: float, lb_mcc: float) -> Tuple[float, str]:
    vals = [("natural", lb_nat), ("autoconv", lb_auto), ("mccormick", lb_mcc)]
    vals.sort(key=lambda t: t[1], reverse=True)
    return vals[0][1], vals[0][0]

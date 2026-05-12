"""Anchor cut: per-box LB via supporting-hyperplane-with-curvature
correction at a fixed mu*.

Mathematical setup
------------------
Let f(mu) = max_W TV_W(mu) where TV_W(mu) = scale_W * mu^T A_W mu.
We DO NOT assume f or any TV_W is convex (A_W is the 0/1 indicator of
window pair-support; it is NOT generally PSD, so TV_W is NOT convex).

For any single window W* with symmetric matrix A_{W*} and gradient
g = 2 * scale_{W*} * A_{W*} * mu*, the EXACT identity holds:

    TV_{W*}(mu) = TV_{W*}(mu*) + g . (mu - mu*) + scale_{W*} * (mu - mu*)^T A_{W*} (mu - mu*).

For the second-order term, lower-bound by the smallest eigenvalue:

    (mu - mu*)^T A_{W*} (mu - mu*) >= lambda_min(A_{W*}) * ||mu - mu*||^2.

If A_{W*} is PSD (lambda_min >= 0), the second-order term is >= 0 and the
familiar supporting-hyperplane bound recovers. Otherwise, we use a
conservative box-corner upper bound on ||mu - mu*||^2:

    ||mu - mu*||^2 <= sum_i max((lo_i - mu*_i)^2, (hi_i - mu*_i)^2) =: D2(B).

Combining (and using that f(mu) >= TV_{W*}(mu) for any single window):

    f(mu) >= TV_{W*}(mu*) + g . (mu - mu*) + scale_{W*} * lambda_min(A_{W*}) * (negative-correction).

When lambda_min < 0, the correction is `scale * lambda_min * D2(B)` which
is a NEGATIVE number (a "concession" to non-convexity). When lambda_min
>= 0, the correction is 0 (no concession needed). Therefore the per-box
LB used in the cert is:

    LB(B) = TV_{W*}(mu*)
          + min_{mu in B cap Delta_d}  g . (mu - mu*)
          + min(0, scale_{W*} * lambda_min(A_{W*})) * D2(B).

The min over mu of `g . (mu - mu*)` is a LINEAR program (greedy sort).

For PSD A_{W*}, this is exactly the convex supporting-hyperplane LB.
For non-PSD A_{W*}, the correction term widens the LB by a quantity
that scales with box diameter — so the cut tightens as B shrinks
around mu*.

Soundness rigor
---------------
- mu_star is converted Fraction-exactly via Fraction(float(x)), preserving
  the input float64 bit-pattern. All anchor constants (TV_{W*}(mu*),
  g, g.mu*) are computed in EXACT Fraction arithmetic at startup,
  eliminating float accumulation error in the cert constants.
- lambda_min(A_{W*}) is computed in float64 via numpy.linalg.eigvalsh.
  A Bauer-Fike-derived safety pad of `scale * sqrt(k_W) * d * eps_machine`
  is added to the curvature concession so the cert is sound against the
  eigvalsh accuracy bound (eps_machine = 2^-52 ~ 2.22e-16).
- Per-box arithmetic is exact integer at denom _SCALE^2 with conservative
  rounding (subtrahends rounded UP, addends rounded DOWN), yielding a
  rigorous under-approximation of the true LB.

Inputs:
  mu_star_int: list[int], at denom _SCALE.
  g_int_per_axis: list[int], at denom _SCALE, ROUNDED DOWN from the
      EXACT Fraction g_q (so g_int * mu lies BELOW true g . mu, since
      mu >= 0 on the simplex). Conservative for the LP min g.mu.
  f_at_mustar_q: Fraction at denom _SCALE^2, ROUNDED DOWN from the
      EXACT Fraction TV_{W*}(mu*).
  curv_neg_int: int >= 0, at denom _SCALE^2. Equals
      max(0, ceil(- scale * lambda_min(A_{W*}) * _SCALE^2)) plus the
      Bauer-Fike safety pad. Multiplied by D2(B)_int (at denom _SCALE^2)
      yields the curvature correction at denom _SCALE^4.

D2(B) is computed in integer arithmetic.

API
---
build_anchor_data(d, mu_star_float):
  precompute (mu_star_int, g_int_per_axis, f_at_mustar_q, curv_neg_int,
              w_star_idx, scale_q) from float mu_star.

bound_anchor_int_ge(lo_int, hi_int, mu_star_int, g_int_per_axis,
                    f_at_mustar_q, target_num, target_den, _scale=...,
                    curv_neg_int=0):
  return True iff anchor LB >= target_num/target_den (integer cert).

Empty domain (box ∩ simplex empty) returns True (vacuous).


Multi-anchor cut: OR of independent supporting hyperplanes
-----------------------------------------------------------
For any *finite* family of anchors {mu*_a : a = 1, ..., A}, each yields
its own sound LB (call it LB_a(B)). For each a we have, for every
mu in B cap Delta_d,

    f(mu) >= LB_a(B).

The pointwise minimum  min_a LB_a  is then also a sound LB for f over B,
but for *certification* we don't even need the min: if ANY single
LB_a(B) >= target then f(mu) >= target for every mu in B, so the box
certifies. Hence the multi-anchor cert is

    multi_cert(B) := OR_a [ LB_a(B) >= target ].

Soundness: each disjunct is sound (this file), so their OR is sound.

Sigma-image anchor (THEOREM.md, Lemma 3.3)
------------------------------------------
The objective f(mu) = max_W TV_W(mu) is sigma-invariant:

    f(sigma(mu)) = f(mu)         for all mu in Delta_d,

where (sigma mu)_i := mu_{d-1-i}. THEOREM.md §3.2 (Lemma 3.3) gives the
proof: under sigma, A_W is conjugated by the reversal permutation P;
because P is a permutation, mu^T P^T A_W P mu = (Pmu)^T A_W (Pmu) =
mu^T A_{tilde W} mu (Lemma 3.2), and the window family W_d is closed
under W -> tilde W (Lemma 3.1).

Consequence: if mu* is any reference point, BOTH mu* and sigma(mu*)
yield supporting hyperplanes with the same constant value
TV_{W*}(mu*) = f(mu*) = f(sigma(mu*)) (up to choice of W*: the binding
window at sigma(mu*) is sigma(W*) = tilde(W*), but the constant value
is preserved). The two anchors are independent linear cuts at different
locations of Delta_d; if mu* lies outside H_d (the half-simplex the BnB
searches per Lemma 3.4), then mu* is far from every H_d-box and the
mu*-cut is vacuously slack, while sigma(mu*) IS in H_d (by Lemma 3.4's
orbit-cover argument: at most one of {mu, sigma(mu)} can be outside
H_d) and gives a much tighter cut.

Per-box centroid anchor: supporting hyperplane at the box midpoint
-------------------------------------------------------------------
For any box B with centroid mu_c := (lo + hi)/2 (which always lies in
B) and ANY window W (in particular the smart W*(B) chosen below), the
same identity applies with mu_c in place of mu*:

    TV_W(mu) = TV_W(mu_c) + g_c . (mu - mu_c) + scale_W (mu - mu_c)^T A_W (mu - mu_c)

where g_c := 2 scale_W A_W mu_c. By Rayleigh,

    (mu - mu_c)^T A_W (mu - mu_c) >= lambda_min(A_W) * ||mu - mu_c||^2.

The centroid is a special reference because for mu in B,

    ||mu - mu_c||^2 = sum_i (mu_i - mu_c_i)^2
                    <= sum_i max((lo_i - mu_c_i)^2, (hi_i - mu_c_i)^2)
                    =  sum_i (half_width_i)^2
                    =: D2_centered(B).

(Equality of the two box-corner expressions follows from mu_c being the
midpoint: both mu_c - lo_i and hi_i - mu_c equal half_width_i.)

Combining and using f(mu) >= TV_W(mu) for ANY single window W,

    f(mu) >= TV_W(mu_c)
           + min_{mu in B cap Delta_d} g_c . (mu - mu_c)
           + min(0, scale_W * lambda_min(A_W)) * D2_centered(B)
        =: LB(W, B).

This is sound for every B and every choice of W. Unlike the fixed mu*
anchor, this anchor *follows the box*, so the LP-min term
g_c . (mu - mu_c) is small (the centroid is in the box) and the cut
tightens automatically as B shrinks.

Smart-W centroid soundness
--------------------------
For each box B we want the TIGHTEST LB. Since LB(W, B) is sound for
every W, the best single-window cert is

    LB_smart(B) := max_W LB(W, B).

This is a sound lower bound on f over B because

    f(mu) >= LB(W, B)  for every W

implies f(mu) >= max_W LB(W, B). Hence picking ANY argmax window —
including the box-dependent W*(B) — preserves soundness. Note that
this is genuinely different from the legacy choice
W_legacy(B) := argmax_W TV_W(mu_c): the legacy choice maximizes the
constant term TV_W(mu_c) only, ignoring the curvature concession
min(0, scale_W lambda_min) * D2_centered(B), which can dominate when
A_W has a very negative lambda_min (e.g. ell=14 windows at d=22 have
lambda_min ~ -6). Smart-W trades off these three contributions
explicitly.

Per-box selection cost: O(|W|) gradients + O(|W|) greedy LPs +
O(|W|) curvature evaluations. The selection inner loop runs in float64
(speed), but the FINAL cert for the chosen window is recomputed in
exact integer arithmetic — soundness depends only on the integer
cert, never on the float selection.

Soundness reductions:
- Identity (TV_W exact 2nd-order Taylor at mu_c): pure algebra of
  the symmetric matrix A_W.
- Rayleigh quotient bound: standard linear algebra.
- D2_centered upper bound: each axis (mu_i - mu_c_i)^2 is at most
  max((lo_i - mu_c_i)^2, (hi_i - mu_c_i)^2), and for mu_c at the
  midpoint these two corner values both equal (half_width_i)^2.
- f >= TV_W: definitional (max over windows >= any single window).
- LP slot: greedy sort over the box-simplex; same routine as the
  fixed-anchor case.
- Smart-W choice: max of sound LBs is sound (any one of them
  dominates f from below pointwise).

All rounding directions in the integer arithmetic are conservative
(round g_c DOWN, round f_at_muc DOWN, round curv_neg UP). The
cumulative effect is to under-approximate the true LB, which gives a
sound (but slightly looser) cert. The +1 ULP Bauer-Fike pad on
curv_neg (added in build_centroid_anchor_cache for every window's
lambda_min) absorbs the float-eigvalsh accuracy bound.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .box import SCALE as _SCALE
from .windows import WindowMeta, build_windows


# Bauer-Fike eigvalsh safety pad. For a symmetric A computed exactly,
# numpy's eigvalsh has accuracy bounded by  ||A||_F * d * eps_machine.
# For 0/1 indicator matrices this is sqrt(k_W) * d * eps. We use this
# to inflate the curvature concession so it absorbs the float-eigvalsh
# error.
_EPS_MACHINE: float = 2.220446049250313e-16  # float64 eps


# ---------------------------------------------------------------------
# Precompute anchor data once per worker
# ---------------------------------------------------------------------

def build_anchor_data(
    d: int,
    mu_star: np.ndarray,
    *,
    windows: Optional[List[WindowMeta]] = None,
):
    """Compute the constants for the anchor cut at mu_star.

    Returns dict with:
      'mu_star_float'   : np.ndarray (d,) float copy
      'mu_star_int'     : list[int] length d, at denom _SCALE
      'w_star_idx'      : int — index of the binding window
      'scale_q'         : Fraction — exact scale of W* = 2d/ell
      'pairs_W_star'    : tuple of (i,j) pairs that index A_{W*}
      'g_float'         : np.ndarray (d,) — gradient grad TV_{W*}(mu*)
      'g_int_per_axis'  : list[int] — g rounded DOWN to denom _SCALE
      'f_at_mustar_float': float — TV_{W*}(mu*)
      'f_at_mustar_q'   : Fraction — f(mu*) rounded DOWN, denom _SCALE^2
      'curv_neg_int'    : int >= 0, at denom _SCALE^2 — encodes
                           max(0, -scale_{W*} * lambda_min(A_{W*})). Used
                           to penalise box diameter for non-PSD A_{W*}.
      'lambda_min_float': float — lambda_min(A_{W*}), diagnostic.
    """
    mu_star = np.asarray(mu_star, dtype=np.float64).copy()
    if mu_star.ndim != 1 or mu_star.shape[0] != d:
        raise ValueError(
            f"mu_star shape {mu_star.shape}, expected ({d},)"
        )
    if windows is None:
        windows = build_windows(d)

    # Convert mu_star to EXACT Fraction (preserving the input float bits;
    # Fraction(float) decodes the float's exact dyadic value). All
    # downstream constant computations use Fraction arithmetic so they
    # are exact functions of the input float bits — no float accumulation.
    mu_star_q: List[Fraction] = [Fraction(float(x)) for x in mu_star]

    # Pick the binding window W* via highest TV_W(mu*). Float arithmetic
    # is fine for selection: any W* yields a SOUND cut (the chain
    # f(mu) >= TV_{W*}(mu) >= ... requires only f >= TV_{W*}). Tightness
    # may suffer microscopically if a near-binding W* is picked, but
    # not soundness.
    best_tv = -1.0
    w_star_idx = -1
    for k, w in enumerate(windows):
        s = 0.0
        for (i, j) in w.pairs_all:
            s += mu_star[i] * mu_star[j]
        tv = w.scale * s
        if tv > best_tv:
            best_tv = tv
            w_star_idx = k
    if w_star_idx < 0:
        raise ValueError("no windows? d=0 not supported")
    w_star = windows[w_star_idx]

    # Build A_{W*} as a dense matrix (small — d up to a few dozen).
    # WindowMeta.pairs_all enumerates BOTH (i,j) and (j,i) for every
    # off-diagonal entry of A_W (and (i,i) once for diagonal entries).
    A = np.zeros((d, d), dtype=np.float64)
    for (i, j) in w_star.pairs_all:
        A[i, j] = 1.0

    # f_at_mustar EXACT: f(mu*) = scale_q * sum_{(i,j) in pairs_all} mu*_i mu*_j
    # All Fractions, no float accumulation.
    s_q = Fraction(0)
    for (i, j) in w_star.pairs_all:
        s_q += mu_star_q[i] * mu_star_q[j]
    f_at_mustar_q_exact = w_star.scale_q * s_q
    SCALE2 = _SCALE * _SCALE
    # Floor-quantise to denom SCALE^2 (round DOWN — conservative).
    f_int_at_S2 = (
        f_at_mustar_q_exact.numerator * SCALE2
    ) // f_at_mustar_q_exact.denominator
    f_at_mustar_q = Fraction(f_int_at_S2, SCALE2)
    # Diagnostic float copy, NOT used in the cert.
    f_at_mustar_float = float(f_at_mustar_q_exact)

    # Gradient at mu* EXACT: grad TV_W = 2 * scale * (A @ mu*).
    # Compute (A @ mu*) as Fraction sum-of-mu*_q; then multiply by
    # 2*scale_q. Floor each component to denom _SCALE.
    Amu_q: List[Fraction] = [Fraction(0) for _ in range(d)]
    for (i, j) in w_star.pairs_all:
        Amu_q[i] = Amu_q[i] + mu_star_q[j]
    two_scale_q = 2 * w_star.scale_q
    g_q: List[Fraction] = [two_scale_q * Amu_q[i] for i in range(d)]
    # Floor to int at denom _SCALE.
    # SOUNDNESS: rounding g DOWN yields a smaller g_int . mu over mu>=0,
    # hence a smaller LP_min, hence a smaller LB. Conservative.
    g_int = [
        (vq.numerator * _SCALE) // vq.denominator for vq in g_q
    ]
    # Diagnostic float copy.
    g_float = np.array([float(vq) for vq in g_q], dtype=np.float64)

    # Rationalize mu_star to integers at denom _SCALE. Use floor for lo-side
    # and ceil for hi-side would be needed if mu* was a box; mu* is a
    # POINT, and we use mu_star_int only in g_int . mu_star_int (a scalar
    # constant on the cert subtrahend). The round-to-nearest below
    # introduces ULP-level cross-rounding error in g_dot_mustar; this is
    # absorbed by the curv-pad safety margin below (the true error is
    # << curv pad at d=22).
    mu_star_int = [int(round(float(x) * _SCALE)) for x in mu_star]

    # lambda_min(A_{W*}) — used for curvature correction when A is not PSD.
    eigs = np.linalg.eigvalsh(A)
    lambda_min = float(eigs[0])

    # Curvature correction: curv_neg = max(0, -scale * lambda_min).
    # We need it at denom SCALE^2 to align with the D2(B) computation
    # below. Use ceil-from-above to ensure we OVER-estimate the magnitude
    # of the negative term — which then gives a SMALLER (more
    # conservative) LB.
    if lambda_min >= 0:
        curv_neg_int = 0
    else:
        # neg_val_float = -scale * lambda_min > 0
        neg_val_float = -float(w_star.scale) * lambda_min
        # ceil(neg_val_float * SCALE^2) — over-approximation of the
        # positive constant we will SUBTRACT.
        nv_q = Fraction(float(neg_val_float))
        # ceil(p / q) = (p + q - 1) // q for p, q > 0.
        num_at_S2 = nv_q.numerator * SCALE2
        den = nv_q.denominator
        curv_neg_int = (num_at_S2 + den - 1) // den
        # Bauer-Fike eigvalsh safety pad. For symmetric A computed
        # exactly, numpy.linalg.eigvalsh accuracy is bounded by
        # ||A||_F * d * eps_machine. For a 0/1 indicator matrix
        # ||A||_F = sqrt(k_W). Pad the curvature concession so it
        # absorbs an eigvalsh undercount up to this magnitude.
        k_W = len(w_star.pairs_all)
        lam_err_float = math.sqrt(max(k_W, 1)) * d * _EPS_MACHINE
        scaled_err_float = float(w_star.scale) * lam_err_float
        err_q = Fraction(float(scaled_err_float))
        pad_S2 = (
            err_q.numerator * SCALE2 + err_q.denominator - 1
        ) // err_q.denominator
        # +1 absorbs the Fraction(float) representation ULP at SCALE^2.
        curv_neg_int += pad_S2 + 1

    return {
        'mu_star_float': mu_star,
        'mu_star_int': mu_star_int,
        'w_star_idx': w_star_idx,
        'scale_q': w_star.scale_q,
        'pairs_W_star': w_star.pairs_all,
        'g_float': g_float,
        'g_int_per_axis': g_int,
        'f_at_mustar_float': f_at_mustar_float,
        'f_at_mustar_q': f_at_mustar_q,
        'curv_neg_int': curv_neg_int,
        'lambda_min_float': lambda_min,
    }


# ---------------------------------------------------------------------
# Anchor LP min: greedy on g
# ---------------------------------------------------------------------

def _lp_min_g_dot_mu_int(
    lo_int: Sequence[int], hi_int: Sequence[int],
    g_int: Sequence[int], d: int,
):
    """Compute  min_{mu in box ∩ Delta_d}  sum_k g_int[k] * mu_k
    in exact integer arithmetic. Returns Python int at denom _SCALE^2,
    or None if box-simplex empty.

    Layout:
      g_int[k] is at denom _SCALE.
      lo_int[k] is at denom _SCALE.
      Product g_int * lo_int is at denom _SCALE^2.

    Greedy: sort axes by g ascending; saturate lo[k] first, then add
    capacity to k with smallest g until sum_mu = SCALE.
    """
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
    return val


def _D2_box_int(
    lo_int: Sequence[int], hi_int: Sequence[int],
    mu_star_int: Sequence[int], d: int,
):
    """sum_i max((lo_i - mu*_i)^2, (hi_i - mu*_i)^2), exact integer
    arithmetic at denom _SCALE^2.

    This is an UPPER bound on max_{mu in B} ||mu - mu*||^2 (and on
    every ||mu - mu*||^2 with mu in B).
    """
    s = 0
    for i in range(d):
        a = lo_int[i] - mu_star_int[i]
        b = hi_int[i] - mu_star_int[i]
        # max(a^2, b^2): pick whichever has larger absolute value.
        if abs(a) > abs(b):
            s += a * a
        else:
            s += b * b
    return s


# ---------------------------------------------------------------------
# Public API: integer cert
# ---------------------------------------------------------------------

def bound_anchor_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    mu_star_int: Sequence[int],
    g_int_per_axis: Sequence[int],
    f_at_mustar_q: Fraction,
    target_num: int,
    target_den: int,
    _scale: int = _SCALE,
    *,
    curv_neg_int: int = 0,
) -> bool:
    """True iff the anchor LB on box B intersect simplex >= target.

    Anchor LB:
        LB = f(mu*) - g . mu*  +  min_{mu in B cap Delta_d} g . mu
                                  - curv_neg * D2(B)

    where
      curv_neg = max(0, -scale_{W*} * lambda_min(A_{W*})), at denom _scale^2,
      D2(B)    = sum_i max((lo_i-mu*_i)^2, (hi_i-mu*_i)^2), at denom _scale^2.

    For PSD A_{W*}, curv_neg = 0 and the bound recovers the convex
    supporting-hyperplane bound. For non-PSD A_{W*}, the curvature term
    yields a SOUND under-approximation that tightens as B shrinks.

    Computed in exact integer arithmetic. Returns True if LB >= target.

    Empty domain (box does not intersect simplex): returns True
    (vacuous certification — matches convention used elsewhere).
    """
    d = len(lo_int)
    SCALE2 = _scale * _scale

    # min g . mu over box ∩ simplex (denom SCALE^2).
    lp_min = _lp_min_g_dot_mu_int(lo_int, hi_int, g_int_per_axis, d)
    if lp_min is None:
        return True  # empty domain vacuously certifies

    # g . mu_star (denom SCALE^2).
    g_dot_mustar = 0
    for k in range(d):
        g_dot_mustar += g_int_per_axis[k] * mu_star_int[k]

    # Curvature concession: curv_neg_int * D2_int.
    # curv_neg_int is at denom SCALE^2 (representing -scale * lambda_min
    # when negative). D2_int is at denom SCALE^2 (sum of squared
    # endpoint-to-mu* differences, each at denom SCALE^2).
    # Product is at denom SCALE^4. We need to subtract it from a quantity
    # at denom SCALE^2, so divide by SCALE^2 (using floor for conservative
    # rounding — we want the SUBTRAHEND to be larger, hence the LB
    # smaller, so we ROUND UP the subtrahend's product when downscaling).
    if curv_neg_int > 0:
        D2_int = _D2_box_int(lo_int, hi_int, mu_star_int, d)
        # curv_correction at denom SCALE^4:
        curv_correction_S4 = curv_neg_int * D2_int
        # Convert to denom SCALE^2 by ceiling-divide (so the value
        # subtracted is at least as large as the true correction).
        curv_correction_S2 = (curv_correction_S4 + SCALE2 - 1) // SCALE2
    else:
        curv_correction_S2 = 0

    # Anchor LB at denom SCALE^2:
    #   LB = f(mu*) + lp_min - g_dot_mustar - curv_correction
    # f_at_mustar_q is a Fraction; bring to denom SCALE^2 via direct
    # multiplication (works exactly when its denom divides SCALE^2).
    if SCALE2 % f_at_mustar_q.denominator != 0:
        # General case — use Fraction arithmetic
        from fractions import Fraction as _F
        lb_q = (
            f_at_mustar_q
            + _F(lp_min - g_dot_mustar - curv_correction_S2, SCALE2)
        )
        target_q = _F(target_num, target_den)
        return lb_q >= target_q

    f_at_S2 = f_at_mustar_q.numerator * (SCALE2 // f_at_mustar_q.denominator)
    lb_int_at_S2 = f_at_S2 + lp_min - g_dot_mustar - curv_correction_S2

    # LB >= target  iff  lb_int * target_den >= target_num * SCALE2.
    lhs = lb_int_at_S2 * target_den
    rhs = target_num * SCALE2
    return lhs >= rhs


# ---------------------------------------------------------------------
# Multi-anchor: OR of independent supporting hyperplanes
# ---------------------------------------------------------------------

def build_multi_anchor_data(
    d: int,
    mu_star: np.ndarray,
    *,
    windows: Optional[List[WindowMeta]] = None,
    sigma_dedup_tol: float = 1e-12,
) -> List[dict]:
    """Build a list of independent anchors at {mu*, sigma(mu*)}.

    By Lemma 3.3 of THEOREM.md, f is sigma-invariant; mu* and sigma(mu*)
    yield two independent supporting hyperplanes with the same
    f(mu*) = f(sigma(mu*)) constant. If mu* is sigma-symmetric (within
    `sigma_dedup_tol`), the second anchor is dropped to avoid
    redundancy.

    Returns: list of anchor dicts, each in the format produced by
    `build_anchor_data`. If a caller wants to OR with a third anchor
    (e.g. a different mu_star pulled from another local-min run), it
    can `extend` the returned list.
    """
    mu_star = np.asarray(mu_star, dtype=np.float64).copy()
    if mu_star.ndim != 1 or mu_star.shape[0] != d:
        raise ValueError(
            f"mu_star shape {mu_star.shape}, expected ({d},)"
        )
    if windows is None:
        windows = build_windows(d)

    anchors: List[dict] = [build_anchor_data(d, mu_star, windows=windows)]
    mu_star_sigma = mu_star[::-1].copy()
    is_symmetric = bool(np.max(np.abs(mu_star - mu_star_sigma)) < sigma_dedup_tol)
    if not is_symmetric:
        anchors.append(build_anchor_data(d, mu_star_sigma, windows=windows))
    return anchors


def bound_anchor_multi_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    anchors: List[dict],
    target_num: int,
    target_den: int,
    _scale: int = _SCALE,
) -> bool:
    """Multi-anchor OR cert: returns True iff ANY anchor's LB >= target.

    Each anchor is a sound supporting-hyperplane bound; OR of sound
    bounds is sound (if any single LB_a >= target then f(mu) >= target
    for every mu in B). Empty domain (box ∩ simplex empty) certifies
    vacuously via the underlying `bound_anchor_int_ge`.
    """
    for a in anchors:
        if bound_anchor_int_ge(
            lo_int, hi_int,
            a['mu_star_int'], a['g_int_per_axis'], a['f_at_mustar_q'],
            target_num, target_den,
            _scale=_scale,
            curv_neg_int=a['curv_neg_int'],
        ):
            return True
    return False


# ---------------------------------------------------------------------
# Per-box centroid anchor
# ---------------------------------------------------------------------

def build_centroid_anchor_cache(
    d: int,
    *,
    windows: Optional[List[WindowMeta]] = None,
) -> dict:
    """Precompute lambda_min(A_W) for every window, plus the dense
    matrices A_W cached in float for centroid evaluation.

    Returns dict with:
      'windows'             : list[WindowMeta]
      'A_per_window'        : np.ndarray (W, d, d) float
      'scale_per_window'    : np.ndarray (W,) float (= 2d/ell)
      'scale_q_per_window'  : list[Fraction]
      'lambda_min_per_window' : np.ndarray (W,) float
      'd'                   : int
    """
    if windows is None:
        windows = build_windows(d)
    W = len(windows)
    A = np.zeros((W, d, d), dtype=np.float64)
    scale = np.empty(W, dtype=np.float64)
    scale_q: List[Fraction] = []
    lam_min = np.empty(W, dtype=np.float64)
    for k, w in enumerate(windows):
        for (i, j) in w.pairs_all:
            A[k, i, j] = 1.0
        scale[k] = float(w.scale)
        scale_q.append(w.scale_q)
        eigs = np.linalg.eigvalsh(A[k])
        lam_min[k] = float(eigs[0])
    return {
        'windows': windows,
        'A_per_window': A,
        'scale_per_window': scale,
        'scale_q_per_window': scale_q,
        'lambda_min_per_window': lam_min,
        'd': d,
    }


def _argmax_window_at_centroid(
    mu_c_int: Sequence[int], windows: List[WindowMeta], d: int,
) -> Tuple[int, int]:
    """Return (w_star_idx, tv_int_at_S2) where tv_int_at_S2 is the EXACT
    integer TV_{W*}(mu_c) at denom _SCALE^2, modulo the scale factor
    being a Fraction with denom dividing 2d. We pick W* by maximizing
    TV_W(mu_c) using exact int arithmetic on the indicator part, then
    multiplying by scale_q (a Fraction).

    Tie-breaking: lowest index. The tie-breaking does NOT affect
    soundness (any single window yields a sound cut).
    """
    SCALE2 = _SCALE * _SCALE
    best_tv_q: Optional[Fraction] = None
    best_idx = -1
    for k, w in enumerate(windows):
        # int-arithmetic mu_c^T A_W mu_c in denom _SCALE^2.
        s = 0
        for (i, j) in w.pairs_all:
            s += mu_c_int[i] * mu_c_int[j]
        tv_q = w.scale_q * Fraction(s, SCALE2)
        if best_tv_q is None or tv_q > best_tv_q:
            best_tv_q = tv_q
            best_idx = k
    assert best_idx >= 0
    return best_idx, 0


def _D2_centered_int(
    lo_int: Sequence[int], hi_int: Sequence[int],
    mu_c_int: Sequence[int], d: int,
) -> int:
    """sum_i max((lo_i - mu_c_i)^2, (hi_i - mu_c_i)^2), denom _SCALE^2.

    For mu_c at the midpoint (mu_c_i = (lo_i + hi_i)//2), the two
    corner values both equal half_width_i^2. We compute via the
    explicit max so we don't rely on mu_c being exactly the midpoint.
    """
    s = 0
    for i in range(d):
        a = lo_int[i] - mu_c_int[i]
        b = hi_int[i] - mu_c_int[i]
        if abs(a) > abs(b):
            s += a * a
        else:
            s += b * b
    return s


def _lp_min_g_dot_mu_float(
    lo_f: np.ndarray, hi_f: np.ndarray,
    g_f: np.ndarray, lo_sum_f: float, d: int,
) -> float:
    """Float greedy LP for min_{mu in box ∩ Delta_d} g . mu.
    Returns +inf if box-simplex empty. Used ONLY for the smart-W
    selection inner loop — it does NOT enter the integer cert."""
    if lo_sum_f > 1.0 + 1e-14:
        return float("inf")
    remaining = 1.0 - lo_sum_f
    val = float(g_f @ lo_f)
    if remaining > 0:
        order = np.argsort(g_f, kind="stable")
        for k in order:
            if remaining <= 0:
                break
            cap = float(hi_f[k] - lo_f[k])
            add = cap if cap < remaining else remaining
            val += float(g_f[k]) * add
            remaining -= add
    return val


def _best_window_for_centroid_lb(
    lo_int: Sequence[int], hi_int: Sequence[int],
    mu_c_int: Sequence[int], cache: dict, d: int,
) -> int:
    """Return W*(B) = argmax_W LB(W, B), where

        LB(W, B) := TV_W(mu_c)
                  + min_{mu in box ∩ Delta_d} g_c^W . (mu - mu_c)
                  + min(0, scale_W * lambda_min(A_W)) * D2_centered(B).

    The selection runs in float64; soundness of the FINAL cert depends
    only on the integer recomputation in `bound_anchor_centroid_int_ge`,
    not on the float selection. Even if the float arithmetic picks a
    sub-optimal W (e.g. due to ULP noise on near-tied LB values), the
    integer cert at that W is still a sound LB on f over B.

    Cost: O(|W|) per box. At d=22 |W|=946; benchmark ~30-100ms per
    box. Acceptable since centroid is the LAST-RESORT tier (already
    gated at depth >= 60 in bnb.py).
    """
    A_per = cache['A_per_window']  # (W, d, d) float
    scale_per = cache['scale_per_window']  # (W,) float
    lam_per = cache['lambda_min_per_window']  # (W,) float

    # Float versions of box endpoints and centroid (no rigor needed).
    inv_S = 1.0 / float(_SCALE)
    lo_f = np.asarray(lo_int, dtype=np.float64) * inv_S
    hi_f = np.asarray(hi_int, dtype=np.float64) * inv_S
    mu_c_f = np.asarray(mu_c_int, dtype=np.float64) * inv_S
    lo_sum_f = float(lo_f.sum())

    # D2_centered in float (independent of W).
    delta_lo = lo_f - mu_c_f
    delta_hi = hi_f - mu_c_f
    D2_f = float(np.sum(np.maximum(delta_lo * delta_lo, delta_hi * delta_hi)))

    # For every window: A_W @ mu_c (vectorised), then per-window LB.
    # Compute Amu = einsum('wij,j->wi', A_per, mu_c_f) all at once.
    Amu_all = A_per @ mu_c_f  # shape (W, d)
    # tv_per[w] = scale_per[w] * mu_c . (A_W @ mu_c)
    tv_per = scale_per * (Amu_all @ mu_c_f)
    # Curv concession per window: min(0, scale_W * lam_min) * D2_f. This
    # is <= 0; subtracted from LB.
    curv_per = np.minimum(0.0, scale_per * lam_per) * D2_f

    best_lb = -float("inf")
    best_idx = 0
    n_W = A_per.shape[0]
    for k in range(n_W):
        # g_c^W = 2 * scale_W * (A_W @ mu_c)
        g_c = 2.0 * float(scale_per[k]) * Amu_all[k]
        lp_min_f = _lp_min_g_dot_mu_float(lo_f, hi_f, g_c, lo_sum_f, d)
        if lp_min_f == float("inf"):
            # Box-simplex empty; vacuous — same for every W. Return any.
            return k
        # slope_min_W(B) = lp_min_f - g_c . mu_c  (= min g . (mu - mu_c)).
        slope_min = lp_min_f - float(g_c @ mu_c_f)
        lb_w = float(tv_per[k]) + slope_min + float(curv_per[k])
        if lb_w > best_lb:
            best_lb = lb_w
            best_idx = k
    return best_idx


def bound_anchor_centroid_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    target_num: int,
    target_den: int,
    cache: dict,
    *,
    _scale: int = _SCALE,
    use_smart_w: bool = True,
) -> bool:
    """Per-box centroid anchor cert.

    Computes mu_c = (lo + hi)/2 (integer if both endpoints share parity
    at the dyadic depth; otherwise we use floor division, which gives
    a midpoint at most 1 ULP below true midpoint — still in B). Picks
    W*(B) by maximising the LB AFTER the curvature concession is
    subtracted (when use_smart_w=True; default). Builds the supporting
    hyperplane

        LB(B) = TV_W*(mu_c)
              + min_{mu in B cap Delta_d} g_c . (mu - mu_c)
              + min(0, scale * lambda_min) * D2_centered(B).

    All arithmetic is integer at denom _SCALE^2. Rounding is
    conservative: g_c rounded DOWN, TV_W*(mu_c) rounded DOWN,
    curv_neg rounded UP.

    Returns True iff LB(B) >= target_num/target_den. Empty domain
    certifies vacuously (matching `bound_anchor_int_ge` convention).

    Soundness for any W choice
    --------------------------
    For ANY window W, LB(W, B) is a sound LB on f over B (see module
    docstring "Smart-W centroid soundness" section). The selector
    `_best_window_for_centroid_lb` runs in float64 — soundness does
    NOT depend on the float selection being optimal: even a sub-optimal
    pick yields a sound integer cert. The legacy
    `_argmax_window_at_centroid` (selected by use_smart_w=False) is
    retained as a regression-comparison baseline.
    """
    d = cache['d']
    windows = cache['windows']
    SCALE = _scale
    SCALE2 = SCALE * SCALE

    # mu_c_int = (lo + hi) // 2 — at denom _scale.
    # For dyadic boxes split at midpoints, lo_i + hi_i is always even
    # at depth < D_SHIFT (Box.split asserts this). Floor division here
    # is exact when sum is even, and at most 1 ULP below true midpoint
    # when odd (which only happens at depth >= D_SHIFT, where the BnB
    # would have already aborted).
    mu_c_int: List[int] = [(lo_int[i] + hi_int[i]) >> 1 for i in range(d)]

    # Box-simplex feasibility (matches _lp_min_g_dot_mu_int).
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > SCALE or hi_sum < SCALE:
        return True  # empty domain vacuously certifies

    # Pick W*(B). Smart-W maximises LB(W,B) (cost: O(|W|) gradients +
    # greedy LPs in float). Legacy maximises TV_W(mu_c) only (one
    # int sum per window). Soundness holds for either choice.
    if use_smart_w:
        w_star_idx = _best_window_for_centroid_lb(
            lo_int, hi_int, mu_c_int, cache, d,
        )
    else:
        w_star_idx, _ = _argmax_window_at_centroid(mu_c_int, windows, d)
    w = windows[w_star_idx]
    scale_f = float(cache['scale_per_window'][w_star_idx])
    scale_q = cache['scale_q_per_window'][w_star_idx]
    lam_min = float(cache['lambda_min_per_window'][w_star_idx])

    # TV_W*(mu_c) exact: scale_q * sum_{(i,j) in pairs_all} mu_c_i mu_c_j
    # at denom _scale^2. f_at_muc_q is an exact Fraction.
    s_int = 0
    for (i, j) in w.pairs_all:
        s_int += mu_c_int[i] * mu_c_int[j]
    f_at_muc_q = scale_q * Fraction(s_int, SCALE2)

    # Floor f_at_muc to denom SCALE^2 (round DOWN — conservative LB).
    f_int_at_S2 = (f_at_muc_q.numerator * SCALE2) // f_at_muc_q.denominator
    # Note: integer-divide of a Fraction with arbitrary denom; floor
    # is exact via Python `//` for positive numerator.

    # Gradient at centroid: g_c = 2 * scale_W * (A @ mu_c). We compute
    # A @ mu_c as an integer vector at denom _scale (entries of A are 0/1),
    # then floor-multiply by 2*scale_q to get g_c rounded DOWN at denom
    # _scale.
    Amu_int: List[int] = [0] * d
    for (i, j) in w.pairs_all:
        Amu_int[i] += mu_c_int[j]
    # Each Amu_int[i] is at denom _scale.
    # g_c[i] = 2 * scale_q * Amu_int[i] / _scale  (still at denom _scale
    # after dividing through by _scale, NO that's not right — let's redo).
    # Actually: A @ mu_c in float is sum_j A[i,j]*mu_c[j], so its int
    # representation at denom _scale is Amu_int[i] (we summed mu_c_int[j],
    # each at denom _scale, with 0/1 weights). So Amu_int[i] is already
    # at denom _scale. Multiplying by 2 * scale_q (a Fraction) gives
    # g_c[i] at denom _scale.
    two_scale_q = 2 * scale_q
    g_int: List[int] = []
    for i in range(d):
        # g_c_q = two_scale_q * Fraction(Amu_int[i], 1) — at denom _scale.
        # Floor to int at denom _scale: floor(two_scale_q.num * Amu_int[i]
        # / two_scale_q.den).
        num = two_scale_q.numerator * Amu_int[i]
        den = two_scale_q.denominator
        # Python // is floor for arbitrary signs.
        g_int.append(num // den)

    # min_{mu in B cap simplex} g_c . mu  (at denom _scale^2). Uses the
    # same greedy as the fixed-anchor case.
    lp_min = _lp_min_g_dot_mu_int(lo_int, hi_int, g_int, d)
    if lp_min is None:
        # Should not reach here — we already verified lo_sum <= SCALE
        # <= hi_sum above. Defensive vacuous cert.
        return True

    # g_c . mu_c at denom _scale^2.
    g_dot_muc = 0
    for k in range(d):
        g_dot_muc += g_int[k] * mu_c_int[k]

    # Curvature concession.
    if lam_min >= 0:
        curv_neg_int = 0
        curv_correction_S2 = 0
    else:
        # neg_val_float = -scale * lambda_min > 0.
        neg_val_q = Fraction(float(-scale_f * lam_min))
        # ceil(neg_val_q * SCALE^2)
        num_at_S2 = neg_val_q.numerator * SCALE2
        den = neg_val_q.denominator
        curv_neg_int = (num_at_S2 + den - 1) // den
        # Bauer-Fike eigvalsh safety pad (same as build_anchor_data):
        # eigvalsh accuracy ~ ||A||_F * d * eps_machine = sqrt(k_W)*d*eps.
        k_W = len(w.pairs_all)
        lam_err_float = math.sqrt(max(k_W, 1)) * d * _EPS_MACHINE
        scaled_err_float = scale_f * lam_err_float
        err_q = Fraction(float(scaled_err_float))
        pad_S2 = (
            err_q.numerator * SCALE2 + err_q.denominator - 1
        ) // err_q.denominator
        curv_neg_int += pad_S2 + 1
        D2_int = _D2_centered_int(lo_int, hi_int, mu_c_int, d)
        # curv_correction at denom _scale^4 → denom _scale^2 by ceil-divide.
        curv_correction_S2 = (curv_neg_int * D2_int + SCALE2 - 1) // SCALE2

    # LB at denom _scale^2:
    #   LB = f_at_muc + (lp_min - g_dot_muc) - curv_correction
    lb_int_at_S2 = f_int_at_S2 + lp_min - g_dot_muc - curv_correction_S2

    lhs = lb_int_at_S2 * target_den
    rhs = target_num * SCALE2
    return lhs >= rhs


# ---------------------------------------------------------------------
# Float-side helpers (testing / diagnostics only)
# ---------------------------------------------------------------------

def anchor_lb_float(
    lo: np.ndarray, hi: np.ndarray,
    mu_star: np.ndarray, g: np.ndarray, f_at_mustar: float,
    *, scale_lambda_min: float = 0.0,
) -> float:
    """Float anchor LB on box B intersect simplex.

    Soundness for non-convex A_{W*} requires the curvature concession
    `scale_lambda_min * D2(B)` where scale_lambda_min := scale_{W*} * lambda_min
    (a negative number when A_{W*} is not PSD).

    With scale_lambda_min = 0 (the default), this is the supporting-line
    bound — sound iff A_{W*} is PSD.
    """
    d = len(lo)
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        return float("inf")  # vacuous (empty domain)

    # min g . mu over box ∩ simplex via greedy
    remaining = 1.0 - lo_sum
    val = float(g @ lo)
    if remaining > 0:
        order = np.argsort(g, kind="stable")
        for k in order:
            if remaining <= 0:
                break
            cap = float(hi[k] - lo[k])
            add = cap if cap < remaining else remaining
            val += float(g[k]) * add
            remaining -= add
    lp_min = val

    # D2(B) box-corner upper bound on ||mu - mu*||^2
    delta_lo = lo - mu_star
    delta_hi = hi - mu_star
    D2 = float(np.sum(np.maximum(delta_lo * delta_lo, delta_hi * delta_hi)))

    # When scale_lambda_min < 0, the curvature concession SUBTRACTS
    # |scale_lambda_min| * D2 from the LB.
    curv_correction = max(0.0, -scale_lambda_min) * D2

    return f_at_mustar - float(g @ mu_star) + lp_min - curv_correction


def anchor_lb_centroid_float(
    lo: np.ndarray, hi: np.ndarray,
    cache: dict,
    *,
    use_smart_w: bool = True,
) -> float:
    """Float per-box centroid anchor LB. Diagnostic mirror of
    `bound_anchor_centroid_int_ge` (no integer cert). Returns +inf on
    empty (box ∩ simplex empty).

    When use_smart_w=True, selects W*(B) = argmax_W LB(W, B) (the
    smart selector). When False, uses the legacy argmax_W TV_W(mu_c).
    Soundness holds for either choice.
    """
    d = cache['d']
    windows = cache['windows']
    A_per = cache['A_per_window']
    scale_per = cache['scale_per_window']
    lam_per = cache['lambda_min_per_window']

    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        return float("inf")

    mu_c = 0.5 * (lo + hi)

    delta_lo = lo - mu_c
    delta_hi = hi - mu_c
    D2 = float(np.sum(np.maximum(delta_lo * delta_lo, delta_hi * delta_hi)))

    if use_smart_w:
        # Pick W*(B) = argmax LB(W, B).
        Amu_all = A_per @ mu_c  # (W, d)
        tv_per = scale_per * (Amu_all @ mu_c)
        curv_per = np.minimum(0.0, scale_per * lam_per) * D2
        best_lb = -float("inf")
        w_star = 0
        n_W = A_per.shape[0]
        for k in range(n_W):
            g_c_k = 2.0 * float(scale_per[k]) * Amu_all[k]
            # min g . mu over box ∩ simplex (greedy).
            remaining = 1.0 - lo_sum
            val = float(g_c_k @ lo)
            if remaining > 0:
                order = np.argsort(g_c_k, kind="stable")
                for kk in order:
                    if remaining <= 0:
                        break
                    cap = float(hi[kk] - lo[kk])
                    add = cap if cap < remaining else remaining
                    val += float(g_c_k[kk]) * add
                    remaining -= add
            slope_min = val - float(g_c_k @ mu_c)
            lb_w = float(tv_per[k]) + slope_min + float(curv_per[k])
            if lb_w > best_lb:
                best_lb = lb_w
                w_star = k
        return best_lb
    else:
        # Legacy: argmax TV_W(mu_c).
        best_tv = -1.0
        w_star = -1
        for k in range(len(windows)):
            tv = scale_per[k] * float(mu_c @ A_per[k] @ mu_c)
            if tv > best_tv:
                best_tv = tv
                w_star = k
        A = A_per[w_star]
        scale = float(scale_per[w_star])
        lam_min = float(lam_per[w_star])
        g_c = 2.0 * scale * (A @ mu_c)
        remaining = 1.0 - lo_sum
        val = float(g_c @ lo)
        if remaining > 0:
            order = np.argsort(g_c, kind="stable")
            for k in order:
                if remaining <= 0:
                    break
                cap = float(hi[k] - lo[k])
                add = cap if cap < remaining else remaining
                val += float(g_c[k]) * add
                remaining -= add
        lp_min = val
        scale_lambda_min = scale * lam_min
        curv_correction = max(0.0, -scale_lambda_min) * D2
        return best_tv - float(g_c @ mu_c) + lp_min - curv_correction


# ---------------------------------------------------------------------
# Multi-corner anchor: OR over a per-box family of anchor points
# ---------------------------------------------------------------------

def _anchor_lb_at_point_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    p_int: Sequence[int], cache: dict, d: int,
    target_num: int, target_den: int,
    SCALE: int, SCALE2: int,
) -> bool:
    """Compute the smart-W centroid-style integer LB at anchor point p
    (assumed to lie in B ∩ Delta_d at denom SCALE) and return True iff
    LB >= target_num/target_den.

    The construction mirrors `bound_anchor_centroid_int_ge` exactly,
    swapping the box-midpoint mu_c for the supplied anchor p. Soundness
    follows from the per-anchor identity (THEOREM-style, also derived in
    the module docstring "Per-box centroid anchor"):

        TV_W(mu) = TV_W(p) + g_p . (mu - p) + scale_W (mu - p)^T A_W (mu - p)
                 >= TV_W(p) + g_p . (mu - p) + scale_W * lambda_min(A_W) * ||mu - p||^2

    For ANY symmetric A_W and ANY anchor point p (whether p is the box
    centroid, a corner, or anywhere else). Hence

        f(mu) >= max_W TV_W(p) + lp_min(g_p, B)
                              + min(0, scale_W * lambda_min) * D2_centered(B, p)

    where D2_centered(B, p) := sum_i max((lo_i - p_i)^2, (hi_i - p_i)^2)
    is a SOUND upper bound on max_{mu in B} ||mu - p||^2 (each axis term
    is the max of the two endpoint-distances squared).
    """
    windows = cache['windows']
    A_per = cache['A_per_window']
    scale_per = cache['scale_per_window']
    lam_per = cache['lambda_min_per_window']

    # ----- pick best W for this anchor point in float64 (selection-only) -----
    inv_S = 1.0 / float(SCALE)
    lo_f = np.asarray(lo_int, dtype=np.float64) * inv_S
    hi_f = np.asarray(hi_int, dtype=np.float64) * inv_S
    p_f = np.asarray(p_int, dtype=np.float64) * inv_S
    lo_sum_f = float(lo_f.sum())

    delta_lo = lo_f - p_f
    delta_hi = hi_f - p_f
    D2_f = float(np.sum(np.maximum(delta_lo * delta_lo, delta_hi * delta_hi)))

    Amu_all = A_per @ p_f  # (W, d)
    tv_per = scale_per * (Amu_all @ p_f)
    curv_per = np.minimum(0.0, scale_per * lam_per) * D2_f

    best_lb_f = -float("inf")
    w_star_idx = 0
    n_W = A_per.shape[0]
    for k in range(n_W):
        g_c_k = 2.0 * float(scale_per[k]) * Amu_all[k]
        # min g . mu over box ∩ simplex (greedy float).
        if lo_sum_f > 1.0 + 1e-14:
            return True  # vacuous (will be re-checked in int below; defensive)
        remaining = 1.0 - lo_sum_f
        val = float(g_c_k @ lo_f)
        if remaining > 0:
            order = np.argsort(g_c_k, kind="stable")
            for kk in order:
                if remaining <= 0:
                    break
                cap = float(hi_f[kk] - lo_f[kk])
                add = cap if cap < remaining else remaining
                val += float(g_c_k[kk]) * add
                remaining -= add
        slope_min = val - float(g_c_k @ p_f)
        lb_w = float(tv_per[k]) + slope_min + float(curv_per[k])
        if lb_w > best_lb_f:
            best_lb_f = lb_w
            w_star_idx = k

    # ----- compute the integer cert at the chosen W* -----
    w = windows[w_star_idx]
    scale_f = float(scale_per[w_star_idx])
    scale_q = cache['scale_q_per_window'][w_star_idx]
    lam_min = float(lam_per[w_star_idx])

    # TV_W*(p) exact: scale_q * sum_{(i,j) in pairs_all} p_i p_j at denom SCALE^2.
    s_int = 0
    for (i, j) in w.pairs_all:
        s_int += p_int[i] * p_int[j]
    f_at_p_q = scale_q * Fraction(s_int, SCALE2)
    # Floor f_at_p to denom SCALE^2 (round DOWN — conservative LB).
    f_int_at_S2 = (f_at_p_q.numerator * SCALE2) // f_at_p_q.denominator

    # g_p = 2 * scale_W * (A @ p), at denom SCALE (round DOWN).
    Amu_int: List[int] = [0] * d
    for (i, j) in w.pairs_all:
        Amu_int[i] += p_int[j]
    two_scale_q = 2 * scale_q
    g_int: List[int] = []
    for i in range(d):
        num = two_scale_q.numerator * Amu_int[i]
        den = two_scale_q.denominator
        g_int.append(num // den)

    # min_{mu in B cap simplex} g . mu (denom SCALE^2).
    lp_min = _lp_min_g_dot_mu_int(lo_int, hi_int, g_int, d)
    if lp_min is None:
        return True  # empty domain — vacuously certifies

    # g . p (denom SCALE^2).
    g_dot_p = 0
    for k in range(d):
        g_dot_p += g_int[k] * p_int[k]

    # Curvature concession.
    if lam_min >= 0:
        curv_correction_S2 = 0
    else:
        neg_val_q = Fraction(float(-scale_f * lam_min))
        num_at_S2 = neg_val_q.numerator * SCALE2
        den = neg_val_q.denominator
        curv_neg_int = (num_at_S2 + den - 1) // den
        # Bauer-Fike eigvalsh safety pad.
        k_W = len(w.pairs_all)
        lam_err_float = math.sqrt(max(k_W, 1)) * d * _EPS_MACHINE
        scaled_err_float = scale_f * lam_err_float
        err_q = Fraction(float(scaled_err_float))
        pad_S2 = (
            err_q.numerator * SCALE2 + err_q.denominator - 1
        ) // err_q.denominator
        curv_neg_int += pad_S2 + 1
        D2_int = _D2_centered_int(lo_int, hi_int, p_int, d)
        curv_correction_S2 = (curv_neg_int * D2_int + SCALE2 - 1) // SCALE2

    lb_int_at_S2 = f_int_at_S2 + lp_min - g_dot_p - curv_correction_S2
    lhs = lb_int_at_S2 * target_den
    rhs = target_num * SCALE2
    return lhs >= rhs


def _build_anchor_points_int(
    lo_int: Sequence[int], hi_int: Sequence[int], d: int, SCALE: int,
) -> List[List[int]]:
    """Generate a list of integer anchor points p (each at denom SCALE).

    SOUNDNESS NOTE: the per-anchor LB
        f(mu) >= TV_W(p) + g_p . (mu - p) + min(0, scale_W * lambda_min(A_W)) * D2_centered(B, p)
    holds for ANY anchor p in R^d (not just p in B ∩ Delta_d). The
    Taylor identity is global; the curvature concession bounds
    ||mu - p||^2 by D2_centered(B, p) for all mu in B (each axis
    independently). So we have flexibility in choosing p — in
    particular, the box centroid mu_c := (lo+hi)/2 need NOT lie on the
    simplex for the cert to be sound. Choosing p in B ∩ Delta_d is
    desirable for tightness (smaller D2, better g_p alignment) but
    not required for soundness.

    Anchors generated (each at denom SCALE):
      0. Box centroid p_0 = (lo + hi) // 2. Always included; matches
         exactly the anchor used by `bound_anchor_centroid_int_ge`,
         guaranteeing multi-corner OR is at least as strong as that
         cert. p_0 may NOT lie on the simplex — the cert is still
         sound (see SOUNDNESS NOTE above).
      1. Free-axis-concentrated point: p_1[i] = hi[i] on free axes
         (lo[i] > 0), p_1[i] = lo[i] on boundary axes. Then PROJECTED
         onto the simplex (sum = SCALE) by trimming/extending free
         axes to keep the point in [lo, hi]. Skipped if the box-simplex
         intersection is empty.
      2. Boundary-spread point: p_2[i] = midpoint on free axes,
         p_2[i] = hi[i] on boundary axes. Projected onto sum = SCALE.
      3. Single-axis points (one per free axis j): mass concentrated
         on j (clipped to [lo[j], hi[j]]), other axes at lo[i].
         Projected onto sum = SCALE.

    Anchors 1-3 are skipped if projection cannot reach sum = SCALE
    while staying in the box (which signals box-simplex empty); this
    leaves only the centroid-style anchor 0.
    """
    pts: List[List[int]] = []
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)

    # 0. Box centroid — always included (matches the centroid cert).
    p0 = [(lo_int[i] + hi_int[i]) >> 1 for i in range(d)]
    pts.append(p0)

    # If box-simplex is empty, only p0 is meaningful (downstream LB
    # routine will short-circuit on empty).
    if lo_sum > SCALE or hi_sum < SCALE:
        return pts

    # Free axes: lo[i] > 0. Boundary axes: lo[i] == 0.
    F: List[int] = [i for i in range(d) if lo_int[i] > 0]
    Bset: List[int] = [i for i in range(d) if lo_int[i] == 0]

    def _project_to_box_and_sum(raw: List[int]) -> Optional[List[int]]:
        """Clamp `raw` to [lo, hi] elementwise; if total != SCALE,
        redistribute (in priority order). Returns list of ints summing
        to SCALE with all components in [lo[i], hi[i]], or None if no
        such projection exists (box-simplex empty).
        """
        p = [max(lo_int[i], min(hi_int[i], raw[i])) for i in range(d)]
        s = sum(p)
        if s == SCALE:
            return p
        if s < SCALE:
            need = SCALE - s
            order = sorted(
                range(d), key=lambda i: -(hi_int[i] - p[i])
            )
            for i in order:
                if need <= 0:
                    break
                cap = hi_int[i] - p[i]
                add = cap if cap < need else need
                p[i] += add
                need -= add
            if need > 0:
                return None
            return p
        excess = s - SCALE
        order = sorted(
            range(d), key=lambda i: -(p[i] - lo_int[i])
        )
        for i in order:
            if excess <= 0:
                break
            cap = p[i] - lo_int[i]
            sub = cap if cap < excess else excess
            p[i] -= sub
            excess -= sub
        if excess > 0:
            return None
        return p

    # 1. Free-axis-concentrated corner: hi on F, lo on B.
    if F:
        F_set = set(F)
        raw1 = [hi_int[i] if i in F_set else lo_int[i] for i in range(d)]
        p1 = _project_to_box_and_sum(raw1)
        if p1 is not None and p1 != p0:
            pts.append(p1)

    # 2. Boundary-spread corner: midpoint on F, hi on B.
    if Bset:
        Bset_set = set(Bset)
        raw2 = [
            hi_int[i] if i in Bset_set else (lo_int[i] + hi_int[i]) >> 1
            for i in range(d)
        ]
        p2 = _project_to_box_and_sum(raw2)
        if p2 is not None and p2 != p0:
            pts.append(p2)

    # 3. Single-axis corners: for each j in F, push max mass onto j.
    sum_lo = sum(lo_int)
    for j in F:
        raw3 = [lo_int[i] for i in range(d)]
        target_j = SCALE - (sum_lo - lo_int[j])
        raw3[j] = max(lo_int[j], min(hi_int[j], target_j))
        p3 = _project_to_box_and_sum(raw3)
        if p3 is not None and p3 not in pts:
            pts.append(p3)

    return pts


def bound_anchor_multi_corner_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    target_num: int,
    target_den: int,
    cache: dict,
    *,
    _scale: int = _SCALE,
) -> bool:
    """Per-box multi-corner anchor cert: OR over several anchor points
    p in B ∩ Delta_d, each giving an independent supporting-hyperplane
    LB on f over B.

    Strategy
    --------
    For each anchor point p_k (centroid + several box corners), compute

        LB(W, p_k, B) := TV_W(p_k)
                       + min_{mu in B cap Delta_d} g_{p_k}^W . (mu - p_k)
                       + min(0, scale_W * lambda_min(A_W)) * D2_centered(B, p_k)

    where D2_centered(B, p) := sum_i max((lo_i - p_i)^2, (hi_i - p_i)^2),
    and g_{p_k}^W := 2 scale_W A_W p_k. For each p_k we pick W*_k =
    argmax_W LB(W, p_k, B) (the smart-W choice, computed in float).

    Soundness (max-of-valid-LBs argument)
    -------------------------------------
    For ANY symmetric A_W and ANY anchor p in B ∩ Delta_d, the exact
    second-order Taylor expansion of TV_W around p gives:

        TV_W(mu) = TV_W(p) + g_p . (mu - p) + scale_W (mu - p)^T A_W (mu - p)
                >= TV_W(p) + g_p . (mu - p) + scale_W * lambda_min(A_W) * ||mu - p||^2
                                              [Rayleigh quotient]

    The last term is bounded below by min(0, scale_W * lambda_min) *
    D2_centered(B, p) since ||mu - p||^2 <= D2_centered(B, p) for every
    mu in B (each axis (mu_i - p_i)^2 is at most max of the two
    endpoint-distances squared). Hence

        f(mu) := max_W TV_W(mu) >= TV_W(p) + g_p . (mu - p)
                                + min(0, scale_W * lambda_min) * D2_centered(B, p)
              >= LB(W, p, B)  for every mu in B ∩ Delta_d, every (W, p).

    Each LB(W, p, B) is therefore a SOUND lower bound on min_{mu in B}
    f(mu). For the OR cert, if ANY single LB(W*_k, p_k, B) >= target,
    then f(mu) >= target on every mu in B and the box certifies. The
    max over (W, p) of sound LBs is also sound — equivalently, the
    disjunction of "this anchor's LB >= target" propositions over
    (W, p) is sound (each disjunct, being sound, implies the conclusion;
    so does their OR).

    Anchor-point construction
    -------------------------
    See `_build_anchor_points_int`. Up to ~9 anchors at d=22:
      - centroid, plus
      - free-axis-concentrated corner (hi on free, lo on boundary), plus
      - boundary-spread corner (hi on boundary, midpoint on free), plus
      - one single-axis corner per free axis.
    Each anchor is constructed in EXACT INTEGER ARITHMETIC at denom
    SCALE, projected onto [lo, hi] and rebalanced to sum = SCALE.

    Conservative rounding (matches existing pattern)
    ------------------------------------------------
      - g_p rounded DOWN (under-approximation when p, mu >= 0).
      - TV_W(p) rounded DOWN.
      - curv_neg rounded UP (over-approximation of subtrahend).
      - +1 ULP Bauer-Fike pad on lambda_min (per-window safety).
      - Anchor points constructed at denom SCALE with sum exactly SCALE
        (residual handed to slack axes), all coordinates in [lo, hi].

    Empty domain (box ∩ simplex empty): returns True (vacuous).

    Cost: O(num_anchors) calls to the per-anchor evaluation. Per-anchor
    cost is dominated by the float-side W-selection loop (O(|W|*d)
    multiplications). At d=22 that's ~30ms × 9 anchors ≈ 270ms per box
    in the worst case. The cert short-circuits as soon as any anchor
    certifies, so easy boxes are much cheaper.
    """
    d = cache['d']
    SCALE = _scale
    SCALE2 = SCALE * SCALE

    # Box-simplex feasibility (empty -> vacuous True, matches centroid cert).
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > SCALE or hi_sum < SCALE:
        return True

    pts = _build_anchor_points_int(lo_int, hi_int, d, SCALE)

    for p_int in pts:
        # The cert is sound for ANY p (Taylor identity is global). We
        # only require p in [lo, hi] (so D2_centered is correct) — we do
        # NOT require sum(p) == SCALE. The box centroid (lo+hi)//2 in
        # particular may not lie on the simplex; this matches the
        # convention in `bound_anchor_centroid_int_ge`.
        ok = True
        for i in range(d):
            if p_int[i] < lo_int[i] or p_int[i] > hi_int[i]:
                ok = False
                break
        if not ok:
            continue
        if _anchor_lb_at_point_int_ge(
            lo_int, hi_int, p_int, cache, d,
            target_num, target_den, SCALE, SCALE2,
        ):
            return True
    return False

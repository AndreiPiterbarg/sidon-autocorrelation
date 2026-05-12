"""Multi-moment extension of the Matolcsi-Vinuesa (MV) dual bound.

Reference implementations:
    - `delsarte_dual/mv_bound.py`                   (MV single-moment reproduction)
    - `delsarte_dual/mv_construction_detailed.md`   (full MV derivation)
    - `delsarte_dual/multi_moment_derivation.md`    (Theorem 2 + Lemma 1 used below)

=================================================================================
CORRECTNESS NOTE ON THE UNIFORM BOUND ON |hat h(n)|     (mirrors Lemma 1 of
`multi_moment_derivation.md`)
=================================================================================

A key input to the multi-moment extension is the following *n-independent* bound:
for h >= 0 supported on [-1/2, 1/2], integrable to 1, with h <= M, one has

        |hat h(n)|  <=  M * sin(pi / M) / pi                          (*)

for EVERY integer n >= 1 — the bound is **independent of n**, and it is the
same value as MV's n = 1 bound (their Lemma 3.4).

Sketch of proof (bathtub / Chebyshev-Markov).
    |hat h(n)| = sup_t Re int h(x+t) exp(-2 pi i n x) dx
               <= sup_{tilde h} int_{-1/2}^{1/2} tilde h(x) cos(2 pi n x) dx
    over {tilde h : 0 <= tilde h <= M, int tilde h <= 1, supp subset [-1/2,1/2]}.
    By bathtub the optimum is M * 1_{A*}, with A* a super-level set of
    cos(2 pi n x) of measure 1/M.  On [-1/2, 1/2] the function cos(2 pi n x) has
    n peaks; the super-level set of measure 1/M is the union of n disjoint arcs
    of length 1/(n M) each, one around each peak (peaks shifted by the free
    translation t if needed, per M >= 1 so 1/(nM) <= 1/n, arcs fit).
    The integral evaluates to
        M * n * ( 2 sin(pi n * 1/(2 n M)) / (2 pi n) )
            = M * n * sin(pi/M) / (pi n)
            = M * sin(pi/M) / pi,
    which is (*); in particular it is INDEPENDENT of n.

Why the "obvious" generalisation  |hat h(n)| <= M sin(pi n / M) / (pi n)  is
WRONG: for n >= 2 and M <= 1.51 (current upper bound on C_{1a}), pi n / M >= pi
so sin(pi n / M) can be zero or negative, while |hat h(n)| is non-negative.
The error is using the n = 1 extremiser for general n; the correct extremiser
splits mass across all n peaks.  See section 2 of `multi_moment_derivation.md`
for the full discussion.

Since |hat h(n)| = z_n^2 where z_n = |hat f(n)|, (*) gives the uniform box
constraint used in the optimisation below:

        0  <=  z_n^2  <=  mu(M) := M * sin(pi/M) / pi     for all n >= 1.     (C)

=================================================================================
MASTER INEQUALITY (Theorem 2 of `multi_moment_derivation.md`)
=================================================================================

Let N = {1, ..., n_max}.  Parseval splits int (f*f) K exactly over N and the
tail; applying Cauchy-Schwarz *only to the tail* (using ||h||_2^2 <= M and
sum k_j^2 = K2) yields

   2/u + a  <=  M + 1 + 2 * sum_{n in N} z_n^2 k_n
                    + sqrt(M - 1 - 2 * sum_{n in N} z_n^4)
                      * sqrt(K2 - 1 - 2 * sum_{n in N} k_n^2)   (MM-10)

subject to constraint (C).  Setting n_max = 1 recovers MV's eq. (10) verbatim.

Positivity inside the square roots is required for (MM-10) to be an effective
constraint on M; if either radicand is negative the inequality is vacuous and
does not constrain M (in that case MV's argument yields no bound at that M and
we treat the test M as infeasible).

Lower bound procedure.  For each candidate M:
    1. Compute mu(M) = M * sin(pi/M) / pi; the box is [0, sqrt(mu(M))]^{n_max}
       for z = (z_1, ..., z_{n_max}).
    2. Maximise the RHS of (MM-10) over z in the box (the RHS is concave in
       y_n = z_n^2; see section 4 of `multi_moment_derivation.md`).
    3. If max RHS >= 2/u + a, then M is feasible.  The smallest feasible M is
       the multi-moment lower bound on C_{1a}.

We find this smallest feasible M by bisection on M.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import mpmath as mp
from mpmath import mpf

try:
    from .mv_bound import (
        MV_COEFFS_119,
        MV_DELTA,
        MV_U,
        MV_K2_BOUND_OVER_DELTA,
        MV_BOUND_FINAL,
        min_G_on_0_quarter,
        S1_sum,
        gain_parameter,
        k1_value,
    )
except ImportError:
    from mv_bound import (
        MV_COEFFS_119,
        MV_DELTA,
        MV_U,
        MV_K2_BOUND_OVER_DELTA,
        MV_BOUND_FINAL,
        min_G_on_0_quarter,
        S1_sum,
        gain_parameter,
        k1_value,
    )

# -----------------------------------------------------------------------------
# Precision
# -----------------------------------------------------------------------------
# 30 digits is ample for this scalar work; a_j coefficients are ~ 8 digits,
# k_n values are |J_0(pi n delta)|^2 which mpmath evaluates to full precision.
DPS = 30


# -----------------------------------------------------------------------------
# Fourier coefficients k_n of K (period-1 convention)
# -----------------------------------------------------------------------------

def kn_value(n: int, delta=MV_DELTA) -> mpf:
    """k_n = hat_K(n) = |J_0(pi n delta)|^2 (period-1 Fourier).

    Reference: mv_construction_detailed.md line 164
        "in period-1 convention hat K(1) = |J_0(pi delta)|^2".
    The same derivation (K(x) = (1/delta) eta(x/delta),
    eta = (2/pi)/sqrt(1-4x^2) on (-1/2, 1/2)) yields the generic identity
        hat K(n) = |J_0(pi n delta)|^2
    because eta is the arcsine density whose Fourier integral on its support
    equals J_0 at the frequency argument.
    """
    delta = mpf(delta)
    j0 = mp.besselj(0, mp.pi * mpf(n) * delta)
    return j0 * j0


def k_values(n_max: int, delta=MV_DELTA) -> List[mpf]:
    """Return [k_1, k_2, ..., k_{n_max}]."""
    return [kn_value(n, delta=delta) for n in range(1, n_max + 1)]


# -----------------------------------------------------------------------------
# The multi-moment master inequality RHS
# -----------------------------------------------------------------------------

def mu_of_M(M) -> mpf:
    """The uniform bathtub/Chebyshev-Markov upper bound on z_n^2:

        mu(M) = M * sin(pi/M) / pi    for all n >= 1                 (see Lemma 1)

    Note: for M in [1, 2], pi/M in [pi/2, pi], so sin(pi/M) is non-negative and
    mu(M) is well-defined and non-negative.  For M just above 1, mu(M) ~ 1/pi.
    """
    M = mpf(M)
    return M * mp.sin(mp.pi / M) / mp.pi


def _sum_zn2_kn(z_values: Sequence[mpf], k_values_: Sequence[mpf]) -> mpf:
    """Returns sum_n z_n^2 k_n."""
    total = mpf(0)
    for z, k in zip(z_values, k_values_):
        total += (mpf(z) ** 2) * mpf(k)
    return total


def _sum_zn4(z_values: Sequence[mpf]) -> mpf:
    """Returns sum_n z_n^4."""
    total = mpf(0)
    for z in z_values:
        zsq = mpf(z) ** 2
        total += zsq * zsq
    return total


def _sum_kn2(k_values_: Sequence[mpf]) -> mpf:
    """Returns sum_n k_n^2."""
    total = mpf(0)
    for k in k_values_:
        total += mpf(k) ** 2
    return total


def rhs_multimoment(M, z_values: Sequence, k_values_: Sequence, K2) -> mpf:
    """Compute the RHS of (MM-10):

        RHS = M + 1 + 2 sum z_n^2 k_n
               + sqrt(M - 1 - 2 sum z_n^4) * sqrt(K2 - 1 - 2 sum k_n^2)

    Parameters
    ----------
    M         : scalar ||f*f||_inf.
    z_values  : sequence of non-negative z_n = |hat f(n)|, length n_max.
    k_values_ : sequence of k_n = hat_K(n), length n_max.
    K2        : scalar ||K||_2^2 (e.g., 0.5747/delta).

    Positivity guard: if EITHER radicand is negative, we return -inf.  The
    rationale is that in the MV proof the sqrt step is a Cauchy-Schwarz upper
    bound on a non-negative tail, so negativity under the sqrt signals that the
    tail bound is vacuous and (MM-10) yields NO constraint on M; treating the
    returned RHS as -inf will then correctly declare that value of z infeasible
    in a MAXIMISATION setting (so the maximiser avoids infeasible z), while
    ensuring that no spurious complex contribution leaks into the real-valued
    optimisation.  See rhs_multimoment_feasible for the analogue that also
    checks whether a feasible z exists at all for the given M.
    """
    M = mpf(M)
    K2 = mpf(K2)
    z_values = [mpf(z) for z in z_values]
    k_values_ = [mpf(k) for k in k_values_]
    if len(z_values) != len(k_values_):
        raise ValueError("z_values and k_values_ must have the same length")

    sum_zk = _sum_zn2_kn(z_values, k_values_)
    sum_z4 = _sum_zn4(z_values)
    sum_k2 = _sum_kn2(k_values_)

    rad1 = M - 1 - 2 * sum_z4
    rad2 = K2 - 1 - 2 * sum_k2
    if rad1 < 0 or rad2 < 0:
        return mp.mpf("-inf")

    return M + 1 + 2 * sum_zk + mp.sqrt(rad1) * mp.sqrt(rad2)


# -----------------------------------------------------------------------------
# Maximising the RHS over the z-box
# -----------------------------------------------------------------------------
#
# With y_n := z_n^2, the RHS is
#     R(y) = M + 1 + 2 <k, y> + sqrt(M - 1 - 2 ||y||^2) * sqrt(K2 - 1 - 2 ||k||^2),
# a concave function of y (linear gain + concave composition of sqrt on concave
# affine) on the box y in [0, mu(M)]^{n_max}.  Interior critical point
# (Lagrangian section 4.1 of multi_moment_derivation.md):
#     y_n^star  =  lambda * k_n   for a common scalar lambda > 0,
# where lambda is determined by the scalar equation
#     4 * lambda^2 * sum k_n^2  =  (M - 1 - 2 ||y^star||^2) / (K2 - 1 - 2 ||k||^2) ... (*)
# i.e. plugging y_n^star = lambda k_n:
#     2 ||y^star||^2 = 2 lambda^2 ||k||^2,
#     so M - 1 - 2 lambda^2 ||k||^2 = 4 lambda^2 ||k||^2 * (K2 - 1 - 2 ||k||^2) / ???
# Simpler: from d R / d y_n = 2 k_n - 2 y_n * sqrt(K2 - 1 - 2||k||^2) / sqrt(M - 1 - 2||y||^2) = 0,
#     y_n = k_n * sqrt(M - 1 - 2 ||y||^2) / sqrt(K2 - 1 - 2 ||k||^2).
# Let S := sum k_n^2 (known).  Writing y_n = lambda k_n:
#     lambda^2 * S  -> ||y||^2 = lambda^2 S,
#     lambda = sqrt(M - 1 - 2 lambda^2 S) / sqrt(K2 - 1 - 2 S)
#     => lambda^2 (K2 - 1 - 2 S) = M - 1 - 2 lambda^2 S
#     => lambda^2 (K2 - 1 - 2 S + 2 S) = M - 1
#     => lambda^2 (K2 - 1) = M - 1
#     => lambda = sqrt( (M - 1) / (K2 - 1) ).
# This is the *same* lambda regardless of n_max.  Then y_n^star = lambda k_n,
# which must be clipped to [0, mu(M)].  Substituting back:
#     2 sum k_n y_n = 2 lambda S
#     ||y||^2 = lambda^2 S
#     M - 1 - 2 ||y||^2 = M - 1 - 2 lambda^2 S = M - 1 - 2 (M-1) S / (K2-1)
#         = (M - 1) (K2 - 1 - 2 S) / (K2 - 1)
#     sqrt(rad1) * sqrt(rad2) = sqrt((M-1)/(K2-1)) * (K2 - 1 - 2 S)
#         = lambda * (K2 - 1 - 2 S).
# So the interior-optimum RHS equals
#     R_int(M) = M + 1 + 2 lambda S + lambda (K2 - 1 - 2 S)
#             = M + 1 + lambda (K2 - 1)
#             = M + 1 + sqrt((M - 1)(K2 - 1)).
# This is EXACTLY MV's (MV-7) — i.e. without the z-refinement at all!  The
# interior critical point, unconstrained, reproduces (MV-7).  The GAIN over
# MV-7 comes entirely from the box constraint y_n <= mu(M): when some y_n^star
# = lambda k_n > mu(M) is clipped, we lose mass on the "spend" side (the
# radicand shrinks less) while we also cap the gain side (+2 k_n y_n is
# smaller).  Careful: clipping can *hurt* RHS, so the maximiser prefers
# interior points when feasible.
#
# Hence the multi-moment refinement is effective precisely when
#     y_n^star = lambda k_n  >  mu(M)   for some n.
# In MV's numerics: lambda = sqrt((M-1)/(K2-1)) ~ sqrt(0.275/3.164) ~ 0.295,
# k_1 ~ 0.896 (delta = 0.138, period-1), so lambda k_1 ~ 0.264 > mu(M) ~ 0.254
# — hence y_1 is clipped to mu(M), giving (MM-10) its n=1 refinement (i.e.,
# MV's eq. (10) with z_1 = 0.50426 ≈ sqrt(0.254)).
#
# For the code we do NOT rely on the closed form; we instead do a grid search
# over the z-box + local refinement, which is robust for arbitrary n_max and
# handles the box constraints correctly without case analysis.


def _grid_points(z_max: mpf, n_grid: int) -> List[mpf]:
    """Return n_grid equally spaced points in [0, z_max] (endpoints included)."""
    z_max = mpf(z_max)
    if n_grid < 2:
        return [z_max / 2] if n_grid == 1 else []
    step = z_max / (n_grid - 1)
    return [step * i for i in range(n_grid)]


def _iterate_box(n_max: int, n_grid: int, z_max: mpf):
    """Generator over the grid [0, z_max]^{n_max}.

    Yields lists of length n_max.  Size n_grid^{n_max}; callers should keep
    n_grid^{n_max} modest (e.g. n_max=5, n_grid=25 => 9.8M points).
    """
    axis = _grid_points(z_max, n_grid)
    if n_max <= 0:
        yield []
        return
    # Iterative Cartesian product to avoid deep recursion for n_max up to ~ 10.
    idxs = [0] * n_max
    while True:
        yield [axis[i] for i in idxs]
        # increment
        j = n_max - 1
        while j >= 0:
            idxs[j] += 1
            if idxs[j] < n_grid:
                break
            idxs[j] = 0
            j -= 1
        if j < 0:
            return


def _refine_local(
    M,
    z_init: Sequence[mpf],
    k_values_: Sequence[mpf],
    K2,
    z_max: mpf,
    n_rounds: int = 6,
    n_sub: int = 7,
    shrink: mpf = mpf("0.4"),
) -> Tuple[mpf, List[mpf]]:
    """Local coordinate-axis refinement around z_init.

    Each round we search each coordinate over a small interval of current
    halfwidth, pick the best on an `n_sub`-point grid, and shrink by `shrink`.

    Returns (best_rhs, best_z).
    """
    z_max = mpf(z_max)
    z = [mpf(zi) for zi in z_init]
    n_max = len(z)
    half = z_max / 4  # initial search halfwidth per axis
    best_rhs = rhs_multimoment(M, z, k_values_, K2)
    for _ in range(n_rounds):
        for i in range(n_max):
            lo = max(mpf(0), z[i] - half)
            hi = min(z_max, z[i] + half)
            if hi <= lo:
                continue
            step = (hi - lo) / (n_sub - 1)
            best_i = z[i]
            best_val = best_rhs
            for t in range(n_sub):
                z_try = lo + step * t
                z[i] = z_try
                val = rhs_multimoment(M, z, k_values_, K2)
                if val > best_val:
                    best_val = val
                    best_i = z_try
            z[i] = best_i
            best_rhs = best_val
        half = half * shrink
    return best_rhs, z


def max_rhs_over_z(
    M,
    k_values_: Sequence[mpf],
    K2,
    n_grid: int = None,       # kept for API compatibility; ignored
    n_refine_rounds: int = 0, # kept for API compatibility; ignored
) -> Tuple[mpf, List[mpf]]:
    """Maximise rhs_multimoment(M, z, k, K2) over z in [0, sqrt(mu(M))]^{n_max}.

    ANALYTIC KKT APPROACH (no grid).
    --------------------------------
    Let y_n = z_n^2 in [0, mu(M)], mu := M sin(pi/M)/pi (Lemma 2.14 box).
    The RHS expressed in y is
        R(y) = M + 1 + 2 <k, y> + sqrt(M - 1 - 2 ||y||^2) * sqrt(K2 - 1 - 2 ||k||^2).
    R is concave in y on its domain.  Its unique maximiser on [0, mu]^{n_max}
    is characterised by KKT: there exists c in {0, 1, ..., n_max} such that
    (after sorting k in decreasing order) the top c coordinates are clipped
    at y_n = mu and the remaining (n_max - c) are interior with y_n = lambda k_n,
    where
        lambda = sqrt( (M - 1 - 2 c mu^2) / (K2 - 1 - 2 S_C) ),
        S_C   := sum_{n in C} k_n^2      (C = clipped set, |C| = c).
    Self-consistency: lambda * k_sorted[c] <= mu (first interior ok) AND
                       lambda * k_sorted[c-1] >= mu (last clipped ok).
    We try all c in O(n_max) and keep the feasible max.  No grid, no refine.

    Returns (max_rhs, z_opt) where z_opt is sqrt(y_opt), length n_max.
    """
    M = mpf(M)
    K2 = mpf(K2)
    k_values_ = [mpf(k) for k in k_values_]
    n_max = len(k_values_)

    mu = mu_of_M(M)
    if mu <= 0:
        z_zero = [mpf(0)] * n_max
        return rhs_multimoment(M, z_zero, k_values_, K2), z_zero

    # Sort k in decreasing order; remember indices so we can un-permute at the end.
    order = sorted(range(n_max), key=lambda i: -float(k_values_[i]))
    k_sorted = [k_values_[i] for i in order]
    k_sorted_sq = [k * k for k in k_sorted]
    S_total = sum(k_sorted_sq)
    K2m1 = K2 - 1
    tail_inner = K2m1 - 2 * S_total
    if tail_inner < 0:
        # Tail sqrt radicand negative: no valid (M, z) pair; R = -inf everywhere.
        return mp.mpf("-inf"), [mpf(0)] * n_max

    best_rhs = mp.mpf("-inf")
    best_y_sorted: List[mpf] = [mpf(0)] * n_max

    # Cumulative S_C = sum_{i<c} k_sorted[i]^2
    S_C = mpf(0)
    for c in range(n_max + 1):
        if c > 0:
            S_C += k_sorted_sq[c - 1]
        rad_num = M - 1 - 2 * c * mu * mu
        if rad_num < 0:
            # Clipping c coords alone already violates rad1 >= 0.
            continue
        den = K2m1 - 2 * S_C
        if den <= 0:
            continue
        lam_sq = rad_num / den
        if lam_sq < 0:
            continue
        lam = mp.sqrt(lam_sq)

        # Self-consistency of the clipping pattern:
        # if c < n_max, require lam * k_sorted[c]  <= mu (first interior feasible);
        # if c > 0,     require lam * k_sorted[c-1] >= mu (last clipped belongs clipped).
        if c < n_max and lam * k_sorted[c] > mu:
            continue
        if c > 0 and lam * k_sorted[c - 1] < mu:
            continue

        # Build y in SORTED order
        y_sorted = [mu if i < c else lam * k_sorted[i] for i in range(n_max)]
        # ||y||^2 and <k, y> in sorted order
        y_sq_sum = mpf(0)
        ky_sum = mpf(0)
        for i in range(n_max):
            y_sq_sum += y_sorted[i] ** 2
            ky_sum += k_sorted[i] * y_sorted[i]
        rad1 = M - 1 - 2 * y_sq_sum
        if rad1 < 0:
            continue

        R = M + 1 + 2 * ky_sum + mp.sqrt(rad1) * mp.sqrt(tail_inner)
        if R > best_rhs:
            best_rhs = R
            best_y_sorted = y_sorted

    # Also robustly evaluate the two degenerate corners as fallbacks
    for y_corner in ([mpf(0)] * n_max, [mu] * n_max):
        y_sq_sum = sum(y * y for y in y_corner)
        rad1 = M - 1 - 2 * y_sq_sum
        if rad1 < 0:
            continue
        ky_sum = sum(k_sorted[i] * y_corner[i] for i in range(n_max))
        R = M + 1 + 2 * ky_sum + mp.sqrt(rad1) * mp.sqrt(tail_inner)
        if R > best_rhs:
            best_rhs = R
            best_y_sorted = list(y_corner)

    # Un-permute: z_opt[original_index] = sqrt(y_sorted[new_position])
    y_unsorted = [mpf(0)] * n_max
    for new_pos, orig_idx in enumerate(order):
        y_unsorted[orig_idx] = best_y_sorted[new_pos]
    z_opt = [mp.sqrt(y) if y > 0 else mpf(0) for y in y_unsorted]
    return best_rhs, z_opt


# -----------------------------------------------------------------------------
# Bisection on M to find the smallest feasible M under (MM-10)
# -----------------------------------------------------------------------------

def solve_for_M_multimoment(
    a_gain,
    delta=MV_DELTA,
    u=MV_U,
    K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
    n_max: int = 1,
    M_lo: mpf = None,
    M_hi: mpf = None,
    tol: mpf = mpf("1e-10"),
    max_iter: int = 120,
) -> mpf:
    """Solve (MM-10) for the smallest feasible M.

    We bisect on M in [M_lo, M_hi].  For each M, compute max RHS over z in the
    allowed box; M is feasible iff max RHS >= 2/u + a.  The function
    M  |->  max_z RHS(M, z) - (2/u + a) is continuous and increasing in M (RHS
    increases in M through the M+1 term and the sqrt(M-1-...) term; the z-box
    grows with M because mu(M) increases).  So bisection is valid.

    Returns the smallest feasible M, an mpf.
    """
    delta = mpf(delta)
    u = mpf(u)
    a = mpf(a_gain)
    K2 = mpf(K2_bound_over_delta) / delta
    target = 2 / u + a

    ks = k_values(n_max, delta=delta)

    if M_lo is None:
        # Strictly > 1 to avoid degenerate mu(M) boundary at M=1 (mu(1) = 0).
        M_lo = mpf("1.0001")
    if M_hi is None:
        # Upper bound well above current best upper bound 1.5029.
        M_hi = mpf("1.60")

    # Sanity: we expect M_hi to be feasible and M_lo to be infeasible.
    rhs_hi, _ = max_rhs_over_z(M_hi, ks, K2)
    if rhs_hi < target:
        raise ValueError(
            f"Upper bracket M_hi={M_hi} is INfeasible (max RHS {rhs_hi} < target "
            f"{target}); widen M_hi."
        )
    rhs_lo, _ = max_rhs_over_z(M_lo, ks, K2)
    if rhs_lo >= target:
        # M_lo is already feasible — the lower bound is below M_lo; narrow.
        return M_lo

    lo = mpf(M_lo)
    hi = mpf(M_hi)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rhs_mid, _ = max_rhs_over_z(mid, ks, K2)
        if rhs_mid >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi  # smallest known-feasible M


# -----------------------------------------------------------------------------
# Top-level: reproduce and extend MV
# -----------------------------------------------------------------------------

def reproduce_multimoment(
    n_max_values: Sequence[int] = (1, 2, 3, 5, 10),
    dps: int = DPS,
    n_grid_minG: int = 40001,
):
    """Plug in MV's 119 tabulated a_j, compute the gain a, then evaluate the
    multi-moment lower bound for each n_max in n_max_values.

    Prints a small table; returns a dict with all intermediate values.
    """
    mp.mp.dps = dps

    # --- MV common inputs ---
    min_G, argmin_G = min_G_on_0_quarter(n_grid=n_grid_minG)
    S1 = S1_sum()
    a_gain = gain_parameter(min_G, S1)
    K2 = MV_K2_BOUND_OVER_DELTA / MV_DELTA
    k1 = k1_value()

    print("=" * 72)
    print("Multi-moment refinement of the Matolcsi-Vinuesa bound")
    print("=" * 72)
    print(f"  MV common inputs (delta={MV_DELTA}, u={MV_U}):")
    print(f"    min_{{[0,1/4]}} G     = {float(min_G):+.8f}  at x = {argmin_G:.5f}")
    print(f"    S_1 (MV gain denom) = {float(S1):.8f}")
    print(f"    gain a              = {float(a_gain):.8f}")
    print(f"    k_1                 = {float(k1):.8f}")
    print(f"    K2 = 0.5747/delta   = {float(K2):.8f}")
    print(f"    2/u + a             = {float(2 / MV_U + a_gain):.8f}")
    print()
    print(f"  MV published bound (n_max=1, with z_1 refinement): {float(MV_BOUND_FINAL):.5f}")
    print()

    results = {
        "min_G": min_G,
        "argmin_G": argmin_G,
        "S1": S1,
        "gain_a": a_gain,
        "k1": k1,
        "K2": K2,
        "bounds_by_nmax": {},
        "k_values_by_nmax": {},
        "z_opt_by_nmax": {},
    }

    target = 2 / MV_U + a_gain

    # Header
    print(f"  {'n_max':>6s} {'M_lower':>14s}  {'k_n table':s}")
    print(f"  {'-'*6:>6s} {'-'*14:>14s}  {'-'*30:s}")

    # Widen bracket a little above 1.27 so bisection bracket is valid.
    for n_max in n_max_values:
        ks = k_values(n_max)
        # Verify feasibility bracket:
        M_star = solve_for_M_multimoment(
            a_gain=a_gain,
            n_max=n_max,
        )
        # Record z_opt at optimum:
        rhs_at_star, z_opt_at_star = max_rhs_over_z(M_star, ks, K2)
        results["bounds_by_nmax"][n_max] = M_star
        results["k_values_by_nmax"][n_max] = ks
        results["z_opt_by_nmax"][n_max] = z_opt_at_star
        kstr = ", ".join(f"{float(k):.4f}" for k in ks[:6]) + (
            ", ..." if len(ks) > 6 else ""
        )
        print(f"  {n_max:>6d} {float(M_star):>14.8f}  [{kstr}]")

    # Sanity check: n_max=1 should reproduce MV's 1.27481 to ~5 digits.
    if 1 in results["bounds_by_nmax"]:
        got = float(results["bounds_by_nmax"][1])
        exp = float(MV_BOUND_FINAL)
        diff = abs(got - exp)
        print()
        print(f"  Sanity check (n_max=1): computed {got:.6f} vs MV's {exp:.5f}")
        if diff < 5e-4:
            print(f"    OK — matches MV's published bound to within 5e-4.")
        else:
            print(f"    WARNING: mismatch |diff| = {diff:.2e} — CHECK INPUTS.")

    return results


if __name__ == "__main__":
    _ = reproduce_multimoment()

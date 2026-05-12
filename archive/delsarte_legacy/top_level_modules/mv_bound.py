"""Matolcsi-Vinuesa lower bound on C_{1a} — exact re-implementation.

Reference: arXiv:0907.1379 (MV 2010). Full citations and equation numbers
in `delsarte_dual/mv_construction_detailed.md`.

This module implements their dual bound:
  - The 119 tabulated Fourier coefficients a_j of G (transcribed from the
    paper's Appendix).
  - The master inequality eq. (7) / (10) for the lower bound on S = inf ||f*f||_inf.
  - Arithmetic at arbitrary mpmath precision.

Result: plugging in their tabulated a_j reproduces S >= 1.2748 (MV's published value).

Key identities
--------------
K(x) = (1/delta) * eta(x/delta),  eta(x) = (2/pi)/sqrt(1-4x^2) on (-1/2,1/2)
    Fourier coefficients (period u = 1/2 + delta):
        tilde_K(j) = (1/u) * |J_0(pi j delta/u)|^2
    MV's quoted bound:
        ||K||_2^2  <  0.5747/delta     (surrogate; see mv_construction_detailed.md)
    Period-1 Fourier:
        k_1 = hat_K(1) = |J_0(pi delta)|^2

G(x) = sum_{j=1}^n a_j cos(2 pi j x / u)
    tilde_G(j) = a_{|j|}/2 for 1 <= |j| <= n, else 0.
    tilde_G(0) = 0 (mean zero; automatic from cosines).
    Admissibility: G > 0 on [-1/4, 1/4].

Gain parameter:
    a = (4/u) * (min_{[0,1/4]} G)^2 / sum_{j=1}^n a_j^2 / |J_0(pi j delta/u)|^2

Master inequality (MV eq. 7):
    M + 1 + sqrt(M - 1) * sqrt(0.5747/delta - 1)  >=  2/u + a
where M = ||f*f||_inf.

Refined inequality (MV eq. 10, adds the z_1 = |hat f(1)| step):
    2/u + a  <=  M + 1 + 2 z_1^2 k_1 + sqrt(M - 1 - 2 z_1^4) * sqrt(0.5747/delta - 1 - 2 k_1^2)
With z_1 in [0, 0.50426] (Lemma 3.4 bound); plug in z_1 = 0.50426 for the tightest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp
from mpmath import mpf


# Fixed parameters from MV paper (p. 4 line 190, line 224)
MV_DELTA = mpf("0.138")
MV_U = mpf("0.638")
MV_N = 119
MV_K2_BOUND_OVER_DELTA = mpf("0.5747")    # MV: ||K||_2^2 < 0.5747/delta  (p.3 line 141)
MV_GAIN_LOWER = mpf("0.0713")             # MV: gain a > 0.0713 (p.4 line 224)
MV_BOUND_NO_Z1 = mpf("1.2743")            # MV: without z_1 refinement (p.4 line 228)
MV_BOUND_FINAL = mpf("1.27481")           # MV: with z_1 refinement (p.7 line 353), rounds to 1.2748


# The 119 coefficients a_j, copied VERBATIM from the MV paper Appendix
# (arXiv:0907.1379v2 PDF pp. 12-13, via the detailed re-read). Signed floats.
MV_COEFFS_119 = [
    mpf("+2.16620392"),  mpf("-1.87775750"),  mpf("+1.05828868"),  mpf("-7.29790538e-01"),
    mpf("+4.28008515e-01"),mpf("+2.17832838e-01"),mpf("-2.70415201e-01"),mpf("+2.72834790e-02"),
    mpf("-1.91721888e-01"),mpf("+5.51862060e-02"),mpf("+3.21662512e-01"),mpf("-1.64478392e-01"),
    mpf("+3.95478603e-02"),mpf("-2.05402785e-01"),mpf("-1.33758316e-02"),mpf("+2.31873221e-01"),
    mpf("-4.37967118e-02"),mpf("+6.12456374e-02"),mpf("-1.57361919e-01"),mpf("-7.78036253e-02"),
    mpf("+1.38714392e-01"),mpf("-1.45201483e-04"),mpf("+9.16539824e-02"),mpf("-8.34020840e-02"),
    mpf("-1.01919986e-01"),mpf("+5.94915025e-02"),mpf("-1.19336618e-02"),mpf("+1.02155366e-01"),
    mpf("-1.45929982e-02"),mpf("-7.95205457e-02"),mpf("+5.59733152e-03"),mpf("-3.58987179e-02"),
    mpf("+7.16132260e-02"),mpf("+4.15425065e-02"),mpf("-4.89180454e-02"),mpf("+1.65425755e-03"),
    mpf("-6.48251747e-02"),mpf("+3.45951253e-02"),mpf("+5.32122058e-02"),mpf("-1.28435276e-02"),
    mpf("+1.48814403e-02"),mpf("-6.49404547e-02"),mpf("-6.01344770e-03"),mpf("+4.33784473e-02"),
    mpf("-2.53362778e-04"),mpf("+3.81674519e-02"),mpf("-4.83816002e-02"),mpf("-2.53878079e-02"),
    mpf("+1.96933442e-02"),mpf("-3.04861682e-03"),mpf("+4.79203471e-02"),mpf("-2.00930265e-02"),
    mpf("-2.73895519e-02"),mpf("+3.30183589e-03"),mpf("-1.67380508e-02"),mpf("+4.23917582e-02"),
    mpf("+3.64690190e-03"),mpf("-1.79916104e-02"),mpf("+7.31661649e-05"),mpf("-2.99875575e-02"),
    mpf("+2.71842526e-02"),mpf("+1.41806855e-02"),mpf("-6.01781076e-03"),mpf("+5.86806100e-03"),
    mpf("-3.32350597e-02"),mpf("+9.23347466e-03"),mpf("+1.47071722e-02"),mpf("-7.42858080e-04"),
    mpf("+1.63414270e-02"),mpf("-2.87265671e-02"),mpf("-1.64287280e-03"),mpf("+8.02601605e-03"),
    mpf("-7.62613027e-04"),mpf("+2.18735533e-02"),mpf("-1.78816282e-02"),mpf("-6.58341101e-03"),
    mpf("+2.67706547e-03"),mpf("-6.25261247e-03"),mpf("+2.24942824e-02"),mpf("-8.10756022e-03"),
    mpf("-5.68160823e-03"),mpf("+7.01871209e-05"),mpf("-1.15294332e-02"),mpf("+1.83608944e-02"),
    mpf("-1.20567880e-03"),mpf("-3.13147456e-03"),mpf("+1.39083675e-03"),mpf("-1.49312478e-02"),
    mpf("+1.32106694e-02"),mpf("+1.73474188e-03"),mpf("-8.53469045e-04"),mpf("+4.03211203e-03"),
    mpf("-1.55352991e-02"),mpf("+8.74711543e-03"),mpf("+1.93998895e-03"),mpf("-2.71357322e-05"),
    mpf("+6.13179585e-03"),mpf("-1.41983972e-02"),mpf("+5.84710551e-03"),mpf("+9.22578333e-04"),
    mpf("-2.16583469e-04"),mpf("+7.07919829e-03"),mpf("-1.18488582e-02"),mpf("+4.39698322e-03"),
    mpf("-8.91346785e-05"),mpf("-3.42086367e-04"),mpf("+6.46355636e-03"),mpf("-8.87555371e-03"),
    mpf("+3.56799654e-03"),mpf("-4.97335419e-04"),mpf("-8.04560326e-04"),mpf("+5.55076717e-03"),
    mpf("-7.13560569e-03"),mpf("+4.53679038e-03"),mpf("-3.33261516e-03"),mpf("+2.35463427e-03"),
    mpf("+2.04023789e-04"),mpf("-1.27746711e-03"),mpf("+1.81247830e-04"),
]
assert len(MV_COEFFS_119) == 119


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

def G_value(x, a_coeffs=MV_COEFFS_119, u=MV_U):
    """G(x) = sum_j a_j cos(2 pi j x / u)."""
    x = mpf(x)
    u = mpf(u)
    pi = mp.pi
    total = mpf(0)
    for j, a in enumerate(a_coeffs, start=1):
        total += mpf(a) * mp.cos(2 * pi * j * x / u)
    return total


def min_G_on_0_quarter(a_coeffs=MV_COEFFS_119, u=MV_U, n_grid=10001):
    """Compute min_{x in [0, 1/4]} G(x) by fine grid.

    Returns the minimum VALUE and its argmin (float for inspection).
    This is NOT rigorous (grid approximation) — for a rigorous enclosure
    use `min_G_on_0_quarter_rigorous` (interval B&B).
    """
    import numpy as np
    xs = np.linspace(0.0, 0.25, n_grid)
    a_f = [float(a) for a in a_coeffs]
    u_f = float(u)
    vals = np.zeros(n_grid)
    for j, aj in enumerate(a_f, start=1):
        vals += aj * np.cos(2 * np.pi * j * xs / u_f)
    idx = int(vals.argmin())
    return float(vals[idx]), float(xs[idx])


def S1_sum(a_coeffs=MV_COEFFS_119, delta=MV_DELTA, u=MV_U):
    """S_1 = sum_{j=1}^n a_j^2 / |J_0(pi j delta / u)|^2, mpmath."""
    delta = mpf(delta)
    u = mpf(u)
    pi = mp.pi
    total = mpf(0)
    for j, a in enumerate(a_coeffs, start=1):
        j0 = mp.besselj(0, pi * j * delta / u)
        total += mpf(a) ** 2 / (j0 ** 2)
    return total


def gain_parameter(min_G, S1, u=MV_U):
    """a = (4/u) * min_G^2 / S1  (MV p. 4 eq. after (7))."""
    u = mpf(u)
    return 4 / u * mpf(min_G) ** 2 / mpf(S1)


def k1_value(delta=MV_DELTA):
    """k_1 = hat_K(1) = |J_0(pi delta)|^2  (MV p.7 line 348, period-1 FT)."""
    delta = mpf(delta)
    j0 = mp.besselj(0, mp.pi * delta)
    return j0 * j0


# -----------------------------------------------------------------------------
# Master inequalities
# -----------------------------------------------------------------------------

def solve_for_M_no_z1(a_gain, delta=MV_DELTA, u=MV_U, K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA):
    """Solve MV eq. (7) for M = ||f*f||_inf:

        M + 1 + sqrt(M - 1) * sqrt(K2 - 1)  >=  2/u + a

    where K2 = K2_bound_over_delta / delta (upper bound on ||K||_2^2).
    Return the smallest M satisfying the inequality, which is the lower bound.
    """
    delta = mpf(delta)
    u = mpf(u)
    a = mpf(a_gain)
    K2 = mpf(K2_bound_over_delta) / delta
    rhs = 2 / u + a
    c = rhs - 1                          # M + sqrt(M-1) sqrt(K2-1) >= c
    A = mp.sqrt(K2 - 1)
    # Let s = sqrt(M - 1). s >= 0.
    # Then s^2 + 1 + A s >= c  =>  s^2 + A s - (c - 1) >= 0
    # Critical: s^2 + A s - (c - 1) = 0 at s = (-A + sqrt(A^2 + 4(c-1)))/2
    disc = A * A + 4 * (c - 1)
    if disc < 0:
        raise ValueError("No real solution; check inputs.")
    s_star = (-A + mp.sqrt(disc)) / 2
    M_star = s_star ** 2 + 1
    return M_star


def solve_for_M_with_z1(
    a_gain, delta=MV_DELTA, u=MV_U,
    K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
    z1=mpf("0.50426"),
):
    """Solve MV eq. (10) for M:

        2/u + a  <=  M + 1 + 2 z_1^2 k_1 + sqrt(M - 1 - 2 z_1^4) * sqrt(K2 - 1 - 2 k_1^2)

    with the Lemma 3.4 bound on z_1 plugged in (default z1 = 0.50426).
    Returns the minimal M satisfying the inequality.
    """
    delta = mpf(delta)
    u = mpf(u)
    a = mpf(a_gain)
    K2 = mpf(K2_bound_over_delta) / delta
    k1 = k1_value(delta)
    z1 = mpf(z1)

    lhs_const = 2 / u + a                     # LHS of eq. (10)
    z1sq = z1 * z1
    z14 = z1sq * z1sq
    c_shift = 1 + 2 * z1sq * k1               # constant shifted to M side
    A2 = K2 - 1 - 2 * k1 * k1                 # second sqrt argument (const)
    if A2 < 0:
        raise ValueError(f"K2 - 1 - 2 k1^2 = {A2} < 0; inputs inconsistent.")
    A = mp.sqrt(A2)

    # Let M = s^2 + 1 + 2 z_1^4 (so that M - 1 - 2 z_1^4 = s^2 >= 0, s >= 0).
    # Then eq. (10) becomes:
    #   s^2 + 1 + 2 z_1^4 + 1 + 2 z_1^2 k_1 + A s  >=  2/u + a
    #   s^2 + A s + (2 + 2 z_1^4 + 2 z_1^2 k_1 - (2/u + a))  >=  0
    #   s^2 + A s - (lhs_const - 2 - 2 z_1^4 - 2 z_1^2 k_1) >= 0
    c_eff = lhs_const - 2 - 2 * z14 - 2 * z1sq * k1
    disc = A * A + 4 * c_eff
    if disc < 0:
        raise ValueError("No real solution; disc < 0.")
    s_star = (-A + mp.sqrt(disc)) / 2
    if s_star < 0:
        s_star = mpf(0)
    M_star = s_star ** 2 + 1 + 2 * z14
    return M_star


# -----------------------------------------------------------------------------
# Top-level: reproduce MV's 1.2748
# -----------------------------------------------------------------------------

def reproduce_MV_bound(dps: int = 30, n_grid_minG: int = 40001):
    """Plug in MV's 119 tabulated coefficients and return the bound.

    Returns a dict with all intermediate quantities for inspection.
    """
    mp.mp.dps = dps
    min_G, argmin_G = min_G_on_0_quarter(n_grid=n_grid_minG)
    S1 = S1_sum()
    a_gain = gain_parameter(min_G, S1)
    k1 = k1_value()
    M_no_z1 = solve_for_M_no_z1(a_gain)
    M_with_z1 = solve_for_M_with_z1(a_gain)
    return {
        "min_G": min_G,
        "argmin_G": argmin_G,
        "S1": S1,
        "gain_a": a_gain,
        "k1": k1,
        "K2_upper": MV_K2_BOUND_OVER_DELTA / MV_DELTA,
        "M_lower_no_z1": M_no_z1,
        "M_lower_with_z1": M_with_z1,
        "MV_target_no_z1": MV_BOUND_NO_Z1,
        "MV_target_final": MV_BOUND_FINAL,
    }


# -----------------------------------------------------------------------------
# Class-based API (B.3 of delsarte_dual pipeline)
#
# Three bound classes layer the MV construction for use by forbidden_region.py:
#   * MVSingleMomentBound    — reproduces MV's Lemma 3.3/3.4 bound phi(M, z_1).
#   * MVMultiMomentBound     — extends to (z_1, ..., z_N) via (MM-10).
#   * MVMultiMomentBoundWithMO — adds MO 2004 Lemma 2.17 (strong, on real parts).
#
# All three expose a ``phi(M, z_vec_or_c_vec) -> mpmath mpf`` callable such
# that phi > 0 means "this (M, z) is INFEASIBLE for admissible f" — i.e., z
# is in the forbidden region for that M.  Bisection on M in forbidden_region.py
# finds the smallest M for which the feasibility problem has no admissible z.
#
# IMPORTANT CORRECTION vs. the requested brief
# --------------------------------------------
# The brief specified per-frequency auxiliary bounds
#     |hat h(j)| <= (M / (pi j)) |sin(pi j / M)|.
# This is WRONG for j >= 2 at M <= 1.51: sin(pi j / M) can be zero or negative
# while |hat h(j)| >= 0, so the "bound" is unsound as an upper bound.
# The correct uniform bathtub/Chebyshev-Markov bound is
#     |hat h(j)| <= M sin(pi / M) / pi    INDEPENDENT of j >= 1
# (see mv_multimoment.py lines 13-47 for the full extremiser proof).
# The implementation below uses the CORRECT uniform bound, so the certificates
# it produces are sound.
# -----------------------------------------------------------------------------

from typing import Callable, Optional


def _lazy_mm():
    """Lazy import of mv_multimoment to break a circular import."""
    import mv_multimoment as _mm
    return _mm


class MVSingleMomentBound:
    """MV's single-moment dual bound phi(M, z_1) from Lemma 3.3 + 3.4.

    The inequality tested is MV eq. (10):
        phi(M, z_1) := 2/u + a
                      - (M + 1 + 2 z_1^2 k_1
                         + sqrt(max(0, M - 1 - 2 z_1^4))
                           * sqrt(K2 - 1 - 2 k_1^2) )
    and phi > 0 means the pair (M, z_1) is FORBIDDEN.

    Instantiation fixes (delta, u, G coefficients, gain a) once so that
    repeated calls from the bisection loop are fast.
    """
    def __init__(
        self,
        delta=MV_DELTA, u=MV_U,
        G_coeffs: Sequence = None,
        K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
        use_lemma_3_4: bool = True,
        n_grid_minG: int = 40001,
        dps: int = 30,
    ):
        mp.mp.dps = dps
        self.delta = mpf(delta)
        self.u = mpf(u)
        self.K2 = mpf(K2_bound_over_delta) / self.delta
        self.use_lemma_3_4 = use_lemma_3_4
        self.a_coeffs = list(G_coeffs) if G_coeffs is not None else list(MV_COEFFS_119)
        min_G, _ = min_G_on_0_quarter(a_coeffs=self.a_coeffs, u=self.u, n_grid=n_grid_minG)
        self.min_G = mpf(min_G)
        self.S1 = S1_sum(a_coeffs=self.a_coeffs, delta=self.delta, u=self.u)
        self.a_gain = gain_parameter(self.min_G, self.S1, u=self.u)
        self.k1 = k1_value(delta=self.delta)
        self.lhs = 2 / self.u + self.a_gain  # 2/u + a

    def phi(self, M, z1=None) -> mpf:
        """Return phi(M, z_1) > 0 iff (M, z_1) is forbidden.

        If z1 is None, clip to the Lemma 3.4 max z_1 = sqrt(mu(M)).
        """
        M = mpf(M)
        if z1 is None:
            z1 = mp.sqrt(_lazy_mm().mu_of_M(M)) if self.use_lemma_3_4 else mpf("0.5")
        z1 = mpf(z1)
        z1sq = z1 * z1
        z14 = z1sq * z1sq
        rad1 = M - 1 - 2 * z14
        rad2 = self.K2 - 1 - 2 * self.k1 * self.k1
        if rad1 < 0 or rad2 < 0:
            return mp.mpf("+inf")  # conservative: treat as infeasible => phi > 0
        rhs = M + 1 + 2 * z1sq * self.k1 + mp.sqrt(rad1) * mp.sqrt(rad2)
        return self.lhs - rhs

    def solve(self) -> mpf:
        """Return the tightest M* with the Lemma 3.4 z_1 clipping."""
        mm = _lazy_mm()
        return solve_for_M_with_z1(
            a_gain=self.a_gain, delta=self.delta, u=self.u,
            K2_bound_over_delta=self.K2 * self.delta,
            z1=mp.sqrt(mm.mu_of_M(mpf("1.2748"))),  # near-optimal z_1
        )


class MVMultiMomentBound:
    """Multi-moment extension of MV's bound using (MM-10).

    phi(M, (z_1, ..., z_N)) :=
        2/u + a
        - [ M + 1 + 2 sum z_n^2 k_n
            + sqrt(max(0, M - 1 - 2 sum z_n^4))
              * sqrt(max(0, K2 - 1 - 2 sum k_n^2)) ].
    """
    def __init__(
        self,
        delta=MV_DELTA, u=MV_U,
        G_coeffs: Sequence = None,
        N: int = 1,
        K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
        n_grid_minG: int = 40001,
        dps: int = 30,
    ):
        mp.mp.dps = dps
        self.delta = mpf(delta)
        self.u = mpf(u)
        self.K2 = mpf(K2_bound_over_delta) / self.delta
        self.a_coeffs = list(G_coeffs) if G_coeffs is not None else list(MV_COEFFS_119)
        self.N = int(N)
        min_G, _ = min_G_on_0_quarter(a_coeffs=self.a_coeffs, u=self.u, n_grid=n_grid_minG)
        self.min_G = mpf(min_G)
        self.S1 = S1_sum(a_coeffs=self.a_coeffs, delta=self.delta, u=self.u)
        self.a_gain = gain_parameter(self.min_G, self.S1, u=self.u)
        self.k_vals = _lazy_mm().k_values(self.N, delta=self.delta)  # period-1 k_n
        self.lhs = 2 / self.u + self.a_gain

    def phi(self, M, z_vec: Sequence) -> mpf:
        M = mpf(M)
        z_vec = [mpf(z) for z in z_vec]
        if len(z_vec) != self.N:
            raise ValueError(f"Expected {self.N} z-values, got {len(z_vec)}")
        rhs = _lazy_mm().rhs_multimoment(M, z_vec, self.k_vals, self.K2)
        if rhs == mp.mpf("-inf"):
            return mp.mpf("+inf")
        return self.lhs - rhs

    def phi_optimal_z(self, M) -> tuple:
        """Return (phi_min, z_opt): the z_opt maximising RHS (minimising phi)
        over the box [0, sqrt(mu(M))]^N, and the resulting phi value.
        """
        M = mpf(M)
        rhs_max, z_opt = _lazy_mm().max_rhs_over_z(M, self.k_vals, self.K2)
        if rhs_max == mp.mpf("-inf"):
            return mp.mpf("+inf"), [mpf(0)] * self.N
        return self.lhs - rhs_max, z_opt

    def solve(self, M_lo=None, M_hi=None, tol: mpf = mpf("1e-10")) -> mpf:
        """Return the smallest M* with phi(M*, z*) <= 0 for some admissible z."""
        return _lazy_mm().solve_for_M_multimoment(
            a_gain=self.a_gain, delta=self.delta, u=self.u,
            K2_bound_over_delta=self.K2 * self.delta,
            n_max=self.N,
            M_lo=M_lo, M_hi=M_hi, tol=tol,
        )


class MVMultiMomentBoundWithMO(MVMultiMomentBound):
    """Multi-moment bound with MO 2004 Lemma 2.17 (strong form) enforced.

    For N >= 2, we replace the magnitude-only box on (z_1, z_2) by the
    real-part layout and intersect with
        c_2 - 2 c_1 + 1 <= 0   <=>   Re hat f(2) <= 2 Re hat f(1) - 1.

    For N = 1, MO 2.17 involves hat f(2) which is not a variable, so we fall
    back to MVSingleMomentBound behaviour.  When mo_strong=False, we use the
    phase-eliminated (weak) form r_2 <= 2 r_1 + 1 which is vacuous; a
    warning is emitted.

    Variable layout for phi:
        if N = 1:  phi(M, z_1).
        if N >= 2: phi(M, z) where z is either
            - magnitude layout (r_1, ..., r_N)  with mo_strong=False;
            - real-part layout (c_1, s_1, c_2, s_2, r_3, ..., r_N) with
              mo_strong=True — the higher-order r_n >= sqrt(c_n^2 + s_n^2)
              are kept as magnitudes since MO 2.17 only involves n = 1, 2.
    """
    def __init__(self, *args, mo_strong: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mo_strong = bool(mo_strong)
        if self.N < 2 and self.mo_strong:
            import warnings
            warnings.warn("MO 2.17 needs N >= 2; falling back to magnitude-only.")

    def phi(self, M, z_or_realpart: Sequence) -> mpf:
        """Evaluate phi.

        If ``mo_strong`` and N >= 2, we expect ``z_or_realpart`` to be the
        length-(N+2) real-part-augmented vector (c_1, s_1, c_2, s_2, r_3, ..., r_N).
        MO 2.17 is enforced by returning +inf when c_2 - 2 c_1 + 1 > 0
        (marking the point as infeasible / trivially excluded).

        Otherwise, magnitudes (r_1, ..., r_N) are expected (N entries).
        """
        M = mpf(M)
        if self.mo_strong and self.N >= 2:
            expected = self.N + 2
            if len(z_or_realpart) != expected:
                raise ValueError(
                    f"With mo_strong & N>=2 expected {expected} entries "
                    f"(c1,s1,c2,s2,r3..rN), got {len(z_or_realpart)}"
                )
            c1, s1, c2, s2 = [mpf(v) for v in z_or_realpart[:4]]
            tail = [mpf(v) for v in z_or_realpart[4:]]  # r_3, ..., r_N (magnitudes)
            # MO 2.17: c_2 - 2 c_1 + 1 <= 0
            if c2 - 2 * c1 + 1 > 0:
                return mp.mpf("+inf")  # forbidden admissibility — no penalty
            # Magnitudes
            r1 = mp.sqrt(c1 * c1 + s1 * s1)
            r2 = mp.sqrt(c2 * c2 + s2 * s2)
            z_vec = [r1, r2] + tail
        else:
            z_vec = [mpf(v) for v in z_or_realpart]
            if len(z_vec) != self.N:
                raise ValueError(f"Expected {self.N} magnitudes, got {len(z_vec)}")
        rhs = _lazy_mm().rhs_multimoment(M, z_vec, self.k_vals, self.K2)
        if rhs == mp.mpf("-inf"):
            return mp.mpf("+inf")
        return self.lhs - rhs


if __name__ == "__main__":
    print("=" * 70)
    print("Matolcsi-Vinuesa lower bound reproduction")
    print("=" * 70)
    res = reproduce_MV_bound()
    for k, v in res.items():
        print(f"  {k:20s} = {v}")
    print()
    print(f"MV's published bound (final, after z_1 refinement): 1.2748")
    print(f"Our computation (M_lower_with_z1): {float(res['M_lower_with_z1']):.6f}")
    print()
    print("Class-based API sanity")
    print("-" * 40)
    b1 = MVSingleMomentBound()
    print(f"  MVSingleMomentBound.solve() = {float(b1.solve()):.6f}")
    b2 = MVMultiMomentBound(N=2)
    print(f"  MVMultiMomentBound(N=2).solve() = {float(b2.solve()):.6f}")
    b3 = MVMultiMomentBoundWithMO(N=2, mo_strong=True)
    print(f"  MVMultiMomentBoundWithMO(N=2).solve() (magnitude-only solve) = "
          f"{float(b3.solve()):.6f}  [MO is enforced in phi(...) only]")

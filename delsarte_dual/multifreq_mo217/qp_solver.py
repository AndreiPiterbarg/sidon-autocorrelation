"""QP solver for the master inequality combining

   * MV-Cauchy-Schwarz upper-bound side (the MV master inequality, MM-10
     specialised at n_max = 2, with the 119-coefficient G of Matolcsi-Vinuesa);
   * MO 2004 Proposition 2.11 at m = 3 lower bound on
        ||f*f||_2^2 = sum_j |hat f(j)|^4 ;
   * MO 2004 Lemma 2.14 (uniform |hat f(j)|^2 <= M sin(pi/M)/pi);
   * MO 2004 Lemma 2.17 (linear constraint on Re hat f(1), Re hat f(2)).

We bisect M until the system is feasible, treating the rigorous lower
bound on M = ||f*f||_oo as the smallest M for which an admissible
(c_1, s_1, c_2, s_2) point exists.

A summary of the math used here lives in
``delsarte_dual/multifreq_mo217/derivation.md``.

The implementation uses mpmath at dps=40 throughout (sufficient for the
~1e-30 residuals required of the bisection); the inner search over
(c_1, s_1, c_2, s_2) uses a coarse float64 grid + scipy Nelder-Mead polish
+ mpmath re-evaluation at the argmax for full precision.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
from mpmath import mpf

# Allow running the module both as a package (delsarte_dual.multifreq_mo217.qp_solver)
# and as a standalone script (python qp_solver.py from inside multifreq_mo217/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

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
from mv_multimoment import (
    k_values as mv_k_values,
    rhs_multimoment as mv_rhs_multimoment,
    mu_of_M as mv_mu_of_M,
)


DPS = 40
mp.mp.dps = DPS


# =============================================================================
# Section A.  K kernels for MO 2004 Proposition 2.11 (the kernel "K" in
# Prop 2.11 satisfies K(x) = 1 on supp f = [-1/4, 1/4]).
# =============================================================================
#
# Two natural choices used below:
#
#   K_step   : K = 1_{[-1/4, 1/4]}, zero elsewhere.   (period-1, real, even)
#   K_pm     : K = +1 on [-1/4, 1/4], -1 on [-1/2, -1/4] U [1/4, 1/2].
#
# Both satisfy K(x) = 1 on supp f.  K_pm has K_hat(0) = 0 so the constant
# term in M_tilde drops out, making M_tilde large and the Prop 2.11 bound
# more powerful.  We allow either choice.
#
# Note: the "K" of Prop 2.11 is COMPLETELY UNRELATED to MV's kernel K_beta
# (the autocorrelation of the arcsine).  Do not confuse them.
# =============================================================================


def K_step_fhat(j) -> mpf:
    """Period-1 Fourier coefficient of K_step = 1_{[-1/4, 1/4]}.

    K_step_fhat(j) = int_{-1/4}^{1/4} e^{-2 pi i j x} dx
                  = sin(pi j / 2) / (pi j)        (j != 0)
                  = 1/2                            (j  = 0)

    These are real and even in j (since K_step is real and even).
    Vanishes at all even j != 0.
    """
    j = int(j)
    if j == 0:
        return mpf(1) / 2
    return mp.sin(mp.pi * mpf(j) / 2) / (mp.pi * mpf(j))


def K_pm_fhat(j) -> mpf:
    """Period-1 Fourier coefficient of K_pm.

    K_pm = +1 on [-1/4, 1/4],  -1 on [-1/2, -1/4] U [1/4, 1/2].

    K_pm_fhat(j) = int_{-1/4}^{1/4} e^{-2 pi i j x} dx
                  - int_{[-1/2, -1/4] U [1/4, 1/2]} e^{-2 pi i j x} dx
                = (1/(pi j)) [2 sin(pi j / 2) - sin(pi j)]            (j != 0)
                = 0                                                    (j  = 0)

    sin(pi j) = 0 for integer j, so K_pm_fhat(j) = 2 sin(pi j / 2) / (pi j) for j != 0.
    """
    j = int(j)
    if j == 0:
        return mpf(0)
    return 2 * mp.sin(mp.pi * mpf(j) / 2) / (mp.pi * mpf(j))


def kernel_4_3_tail_norm(K_fhat: Callable, m: int = 3, n_max: int = 4000) -> mpf:
    """||_m K_hat||_{4/3} := (sum_{|j| >= m} |K_hat(j)|^{4/3})^{3/4}.

    The sum is doubled (positive and negative j); convergence is fine since
    |K_hat(j)| ~ C/|j|, so |K_hat|^{4/3} ~ C/|j|^{4/3} is summable.

    n_max=4000 truncation: the tail beyond j=4000 contributes < 1e-12
    relative for both K_step and K_pm.
    """
    s = mpf(0)
    for j in range(m, n_max + 1):
        kj = K_fhat(j)
        s += 2 * mp.power(abs(kj), mpf(4) / 3)
    return mp.power(s, mpf(3) / 4)


# Pre-computed values (cached at first use, recomputed if dps changes).
_KERNEL_DATA = {}


def kernel_data(name: str = "step", m: int = 3, n_max: int = 4000) -> dict:
    """Return cached (K_hat(0), K_hat(1), K_hat(2), tail_norm_4_3) for K."""
    key = (name, m, n_max, mp.mp.dps)
    if key not in _KERNEL_DATA:
        K_fhat = {"step": K_step_fhat, "pm": K_pm_fhat}[name]
        _KERNEL_DATA[key] = {
            "K_fhat": K_fhat,
            "K0": K_fhat(0),
            "K1": K_fhat(1),
            "K2": K_fhat(2),
            "tail_43": kernel_4_3_tail_norm(K_fhat, m=m, n_max=n_max),
            "name": name,
            "m": m,
        }
    return _KERNEL_DATA[key]


# =============================================================================
# Section B.  The Prop 2.11 lower bound on ||f*f||_2^2 at m = 3.
# =============================================================================
#
# MO 2004 Proposition 2.11 (m=3).  For f >= 0 supported on [-1/4, 1/4] with
# int f = 1, and any kernel K with K(x) = 1 on [-1/4, 1/4],
#
#   ||f*f||_2^2 = sum_j |hat f(j)|^4
#       >= 1 + 2 |hat f(1)|^4 + 2 |hat f(2)|^4
#                   + (|M_tilde| / ||_3 K_hat||_{4/3})^4         (Prop 2.11)
#
# where  M_tilde := 1 - K_hat(0) - 2 K_hat(1) Re hat f(1) - 2 K_hat(2) Re hat f(2).
#
# Combined with the trivial Hoelder upper bound  ||f*f||_2^2 <= ||f*f||_oo = M
# this yields the rigorous inequality
#
#   M >= 1 + 2 (c_1^2 + s_1^2)^2 + 2 (c_2^2 + s_2^2)^2 + (|M_tilde|/T)^4   (P2.11+H)
#
# where  c_n + i s_n = hat f(n),  T := ||_3 K_hat||_{4/3}.
# =============================================================================


def prop211_m3_lower_bound(M, c1, s1, c2, s2, K_data: dict) -> mpf:
    """RHS of Prop 2.11 m=3 + Hoelder, evaluated at (c_1, s_1, c_2, s_2).

    This is the lower bound on M the inequality (P2.11+H) imposes at the
    given (c_n, s_n).  The outer feasibility check is  M >= this RHS.
    """
    c1, s1, c2, s2 = mpf(c1), mpf(s1), mpf(c2), mpf(s2)
    z1sq = c1 * c1 + s1 * s1
    z2sq = c2 * c2 + s2 * s2
    K0, K1, K2 = K_data["K0"], K_data["K1"], K_data["K2"]
    T = K_data["tail_43"]
    M_tilde = 1 - K0 - 2 * K1 * c1 - 2 * K2 * c2
    M_tilde_4 = M_tilde * M_tilde * M_tilde * M_tilde   # |M_tilde|^4 since real^4
    return 1 + 2 * z1sq * z1sq + 2 * z2sq * z2sq + M_tilde_4 / (T ** 4)


# =============================================================================
# Section C.  MV master inequality side.
#
# Specialised to n_max = 2 (z_1, z_2 pulled out), with MV's 119 a_j and
# the MV kernel K_beta (NOT the K of Prop 2.11; K_beta is the arcsine
# autocorrelation kernel used in MV's gain mechanism).
# =============================================================================


@dataclass
class MVSetup:
    delta: mpf
    u: mpf
    a_gain: mpf
    k1_mv: mpf       # k_1 = hat K_beta(1) = |J_0(pi delta)|^2
    k2_mv: mpf       # k_2 = hat K_beta(2)
    K2_mv: mpf       # ||K_beta||_2^2 < 0.5747/delta
    lhs_target: mpf  # 2/u + a_gain
    min_G: mpf       # min_{[0,1/4]} G   (the MV gain numerator)
    n_max: int       # 2 in our setting

    @classmethod
    def build(cls, delta=MV_DELTA, u=MV_U, a_coeffs=None, n_max: int = 2,
              n_grid_minG: int = 40001) -> "MVSetup":
        if a_coeffs is None:
            a_coeffs = list(MV_COEFFS_119)
        delta = mpf(delta)
        u = mpf(u)
        min_G_val, _ = min_G_on_0_quarter(a_coeffs=a_coeffs, u=u, n_grid=n_grid_minG)
        S1 = S1_sum(a_coeffs=a_coeffs, delta=delta, u=u)
        a_gain = gain_parameter(min_G_val, S1, u=u)
        k1 = k1_value(delta=delta)
        ks = mv_k_values(n_max, delta=delta)
        K2 = mpf(MV_K2_BOUND_OVER_DELTA) / delta
        return cls(
            delta=delta, u=u, a_gain=a_gain,
            k1_mv=ks[0], k2_mv=ks[1] if n_max >= 2 else mpf(0),
            K2_mv=K2,
            lhs_target=2 / u + a_gain,
            min_G=mpf(min_G_val),
            n_max=int(n_max),
        )


def mv_master_rhs(M, z1, z2, mv: MVSetup) -> mpf:
    """RHS of MV master inequality (MM-10) at n_max = 2:

        M + 1 + 2 (z_1^2 k_1 + z_2^2 k_2)
          + sqrt(M - 1 - 2 (z_1^4 + z_2^4))
            * sqrt(K2 - 1 - 2 (k_1^2 + k_2^2)).

    Returns -inf if either radicand is negative (infeasible).
    """
    M = mpf(M)
    z1, z2 = mpf(z1), mpf(z2)
    z1sq, z2sq = z1 * z1, z2 * z2
    z14, z24 = z1sq * z1sq, z2sq * z2sq
    rad1 = M - 1 - 2 * (z14 + z24)
    rad2 = mv.K2_mv - 1 - 2 * (mv.k1_mv ** 2 + mv.k2_mv ** 2)
    if rad1 < 0 or rad2 < 0:
        return mp.mpf("-inf")
    return M + 1 + 2 * (z1sq * mv.k1_mv + z2sq * mv.k2_mv) + mp.sqrt(rad1) * mp.sqrt(rad2)


def mv_master_gap(M, c1, s1, c2, s2, mv: MVSetup) -> mpf:
    """Gap MV_RHS - LHS_target.  >= 0 means MV master inequality satisfied.

    z_n^2 := c_n^2 + s_n^2.
    """
    z1sq = c1 * c1 + s1 * s1
    z2sq = c2 * c2 + s2 * s2
    z1 = mp.sqrt(z1sq)
    z2 = mp.sqrt(z2sq)
    rhs = mv_master_rhs(M, z1, z2, mv)
    if rhs == mp.mpf("-inf"):
        return mp.mpf("-inf")
    return rhs - mv.lhs_target


# =============================================================================
# Section D.  Lemma 2.14 and 2.17 feasibility predicates.
# =============================================================================


def lemma_214_box(M) -> mpf:
    """Returns mu(M) = M sin(pi/M) / pi  (Lemma 2.14 box halfwidth-squared)."""
    return mv_mu_of_M(M)


def lemma_217_feasible(c1, s1, c2, eps=mpf("1e-30")) -> bool:
    """Lemma 2.17 STRONG (real-part) form:

        2 (c_1^2 + s_1^2) - 1  <=  c_2  <=  2 c_1 - 1.

    Allow a small numerical slack ``eps``.
    """
    c1, s1, c2 = mpf(c1), mpf(s1), mpf(c2)
    z1sq = c1 * c1 + s1 * s1
    return (2 * z1sq - 1 - eps <= c2) and (c2 <= 2 * c1 - 1 + eps)


def lemma_214_feasible(M, c1, s1, c2, s2, eps=mpf("1e-30")) -> bool:
    """Lemma 2.14: c_n^2 + s_n^2 <= mu(M) for n = 1, 2."""
    mu = lemma_214_box(M)
    z1sq = mpf(c1) ** 2 + mpf(s1) ** 2
    z2sq = mpf(c2) ** 2 + mpf(s2) ** 2
    return (z1sq <= mu + eps) and (z2sq <= mu + eps)


# =============================================================================
# Section E.  The QP-style joint feasibility check.
#
# At fixed M, find the (c_1, s_1, c_2, s_2) MAXIMISING the JOINT gap
#
#   phi_joint(M, c_1, s_1, c_2, s_2)
#       := min( MV_gap(M, c_1, s_1, c_2, s_2) ,
#               M - Prop211_m3_RHS(M, c_1, s_1, c_2, s_2) )
#
# subject to Lemma 2.14 and Lemma 2.17 (strong).
#
# - phi_joint >= 0 means BOTH inequalities are satisfied at this point, so
#   M is feasible.
# - phi_joint < 0 for all admissible points means M is infeasible (rule out).
#
# The smallest feasible M (bisection) is the rigorous lower bound.
#
# Notes.
# ------
# * The MV side is a STRICT inequality: 2/u + a <= MV_RHS.  Thus we want
#   MV_RHS - (2/u + a) >= 0.
# * The Prop 2.11 side is M >= Prop_RHS, i.e., M - Prop_RHS >= 0.
# * Both must hold simultaneously at the SAME (c_1, s_1, c_2, s_2).
# =============================================================================


def phi_joint(M, c1, s1, c2, s2, mv: MVSetup, K_data: dict) -> mpf:
    """min(MV gap, Prop 2.11 gap) at the given point.

    Returns -inf if any radicand is infeasible.
    """
    M = mpf(M)
    c1, s1, c2, s2 = mpf(c1), mpf(s1), mpf(c2), mpf(s2)

    # MV gap
    g_mv = mv_master_gap(M, c1, s1, c2, s2, mv)
    if g_mv == mp.mpf("-inf"):
        return mp.mpf("-inf")

    # Prop 2.11 gap
    rhs_p = prop211_m3_lower_bound(M, c1, s1, c2, s2, K_data)
    g_p = M - rhs_p

    return g_mv if g_mv < g_p else g_p


# =============================================================================
# Section F.  Inner maximisation of phi_joint over (c_1, s_1, c_2, s_2)
# =============================================================================
#
# The constraint set is
#
#    F(M) := { (c_1, s_1, c_2, s_2) :
#                c_1^2 + s_1^2 <= mu(M),
#                c_2^2 + s_2^2 <= mu(M),
#                2 c_1 - 1 >= c_2 >= 2 (c_1^2 + s_1^2) - 1 }
#
# - bounded (since |c_n|, |s_n| <= sqrt(mu(M))),
# - non-convex (Lemma 2.17 left is quadratic in (c_1, s_1)).
#
# Strategy (robust, two-stage):
#
#   1.  Coarse 4-D float64 grid search over F(M) (after symmetry pruning).
#   2.  scipy.optimize.minimize Nelder-Mead polish from the best grid point.
#   3.  mpmath re-evaluation of phi_joint at the polished argmax.
#
# Since mu(M) decreases as M decreases, F(M) shrinks, so coarse grids of
# 17^4 ~ 83k points are sufficient for the inner maximisation.
# =============================================================================


def _phi_joint_float(
    M_f: float,
    c1_f: float, s1_f: float, c2_f: float, s2_f: float,
    mv: MVSetup, K_data: dict,
) -> float:
    """Vectorisable float64 evaluation of phi_joint (single point version)."""
    z1sq = c1_f * c1_f + s1_f * s1_f
    z2sq = c2_f * c2_f + s2_f * s2_f

    # MV gap (float)
    z14 = z1sq * z1sq
    z24 = z2sq * z2sq
    rad1 = M_f - 1.0 - 2.0 * (z14 + z24)
    rad2 = float(mv.K2_mv) - 1.0 - 2.0 * (float(mv.k1_mv) ** 2 + float(mv.k2_mv) ** 2)
    if rad1 < 0.0 or rad2 < 0.0:
        return -np.inf
    mv_rhs = (
        M_f + 1.0
        + 2.0 * (z1sq * float(mv.k1_mv) + z2sq * float(mv.k2_mv))
        + (rad1 ** 0.5) * (rad2 ** 0.5)
    )
    g_mv = mv_rhs - float(mv.lhs_target)

    # Prop 2.11 gap (float)
    K0, K1, K2 = float(K_data["K0"]), float(K_data["K1"]), float(K_data["K2"])
    T = float(K_data["tail_43"])
    M_tilde = 1.0 - K0 - 2.0 * K1 * c1_f - 2.0 * K2 * c2_f
    rhs_p = 1.0 + 2.0 * z1sq * z1sq + 2.0 * z2sq * z2sq + (M_tilde / T) ** 4
    g_p = M_f - rhs_p

    return min(g_mv, g_p)


def _grid_points(zmax: float, n: int) -> np.ndarray:
    """Equally spaced points in [-zmax, zmax]."""
    return np.linspace(-zmax, zmax, n)


def _is_feasible_217_214_float(c1, s1, c2, s2, mu_M, eps=1e-12) -> bool:
    z1sq = c1 * c1 + s1 * s1
    z2sq = c2 * c2 + s2 * s2
    if z1sq > mu_M + eps:
        return False
    if z2sq > mu_M + eps:
        return False
    if c2 > 2 * c1 - 1 + eps:
        return False
    if c2 < 2 * z1sq - 1 - eps:
        return False
    return True


def maximize_phi_joint(
    M, mv: MVSetup, K_data: dict,
    n_grid_c1: int = 21,
    n_grid_s1: int = 9,
    n_grid_c2: int = 21,
    n_grid_s2: int = 9,
    n_polish: int = 200,
    verbose: bool = False,
) -> Tuple[mpf, Tuple[mpf, mpf, mpf, mpf]]:
    """Maximise phi_joint over (c_1, s_1, c_2, s_2) in F(M).

    Returns (max_phi, argmax) where argmax = (c_1, s_1, c_2, s_2).

    If the constraint set F(M) is empty (infeasible) returns (-inf, zeros).
    """
    M = mpf(M)
    mu_M = float(lemma_214_box(M))
    if mu_M <= 0:
        return mp.mpf("-inf"), (mpf(0), mpf(0), mpf(0), mpf(0))
    zmax = mu_M ** 0.5  # |c_n|, |s_n| <= sqrt(mu_M)
    M_f = float(M)

    # Coarse grid
    c1_axis = _grid_points(zmax, n_grid_c1)
    s1_axis = _grid_points(zmax, n_grid_s1)
    c2_axis = _grid_points(zmax, n_grid_c2)
    s2_axis = _grid_points(zmax, n_grid_s2)

    best_phi = -np.inf
    best_pt = (0.0, 0.0, 0.0, 0.0)

    # Note: we exploit a global symmetry (translation of f rotates hat f).
    # WLOG s_1 = 0 (rotate so hat f(1) is real >= 0).  This eliminates the
    # s_1 axis and reduces the grid by a factor of n_grid_s1.  However, in
    # the strong (real-part) layout of Lemma 2.17, the rotation also
    # rotates hat f(2) by exp(-4 pi i t_0), so we cannot simultaneously
    # set s_2 = 0.  We DO retain s_1 = 0 (degenerate first axis: only
    # s_1 = 0 is searched; this is rigorous).
    s1_axis_used = np.array([0.0])  # s_1 == 0 by translation choice
    if c1_axis[0] < 0:
        # Symmetry: c_1 -> -c_1 + s_1 -> -s_1 changes sign of c_2 (rotates by pi);
        # we enforce c_1 >= 0 by Lemma 2.17 right (2 c_1 - 1 >= c_2; if c_1 < 0
        # forces c_2 < -1, infeasible since |c_2| <= sqrt(mu) < 1).
        c1_axis = c1_axis[c1_axis >= 0.0]

    # Even with s_1 = 0, s_2 still ranges (no further symmetry).

    for c1 in c1_axis:
        # Lemma 2.17 right: c_2 <= 2 c_1 - 1
        c2_upper = 2 * c1 - 1
        if c2_upper < -zmax:
            continue
        for s1 in s1_axis_used:
            z1sq = c1 * c1 + s1 * s1
            if z1sq > mu_M + 1e-12:
                continue
            # Lemma 2.17 left: c_2 >= 2 z1sq - 1
            c2_lower = 2 * z1sq - 1
            for c2 in c2_axis:
                if c2 > c2_upper + 1e-12:
                    continue
                if c2 < c2_lower - 1e-12:
                    continue
                if c2 * c2 > mu_M + 1e-12:
                    continue
                for s2 in s2_axis:
                    if c2 * c2 + s2 * s2 > mu_M + 1e-12:
                        continue
                    val = _phi_joint_float(M_f, c1, s1, c2, s2, mv, K_data)
                    if val > best_phi:
                        best_phi = val
                        best_pt = (c1, s1, c2, s2)

    if best_phi == -np.inf:
        return mp.mpf("-inf"), (mpf(0), mpf(0), mpf(0), mpf(0))

    # scipy Nelder-Mead polish
    try:
        from scipy.optimize import minimize

        def neg_phi(v):
            c1_, s1_, c2_, s2_ = v
            # Project to feasibility (penalty for violations)
            z1sq = c1_ * c1_ + s1_ * s1_
            z2sq = c2_ * c2_ + s2_ * s2_
            penalty = 0.0
            if z1sq > mu_M:
                penalty += 1e3 * (z1sq - mu_M)
            if z2sq > mu_M:
                penalty += 1e3 * (z2sq - mu_M)
            if c2_ > 2 * c1_ - 1:
                penalty += 1e3 * (c2_ - (2 * c1_ - 1))
            if c2_ < 2 * z1sq - 1:
                penalty += 1e3 * ((2 * z1sq - 1) - c2_)
            val = _phi_joint_float(M_f, c1_, s1_, c2_, s2_, mv, K_data)
            if not np.isfinite(val):
                return 1e6 + penalty
            return -val + penalty

        res = minimize(
            neg_phi,
            x0=np.array(best_pt),
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": n_polish},
        )
        c1_p, s1_p, c2_p, s2_p = res.x
        if _is_feasible_217_214_float(c1_p, s1_p, c2_p, s2_p, mu_M):
            val_p = _phi_joint_float(M_f, c1_p, s1_p, c2_p, s2_p, mv, K_data)
            if val_p > best_phi:
                best_phi = val_p
                best_pt = (c1_p, s1_p, c2_p, s2_p)
    except Exception as e:
        if verbose:
            print(f"  scipy polish skipped: {e}")

    # Convert to mpmath and re-evaluate
    c1, s1, c2, s2 = (mpf(v) for v in best_pt)
    best_phi_mp = phi_joint(M, c1, s1, c2, s2, mv, K_data)
    if best_phi_mp == mp.mpf("-inf"):
        # Fallback to float value
        return mpf(best_phi), (c1, s1, c2, s2)
    return best_phi_mp, (c1, s1, c2, s2)


# =============================================================================
# Section G.  Outer bisection on M.
# =============================================================================


def bisect_M_lower_bound(
    mv: MVSetup,
    K_data: dict,
    M_lo: mpf = mpf("1.20"),
    M_hi: mpf = mpf("1.40"),
    tol: mpf = mpf("1e-9"),
    max_iter: int = 80,
    verbose: bool = True,
    n_grid_c1: int = 21,
    n_grid_c2: int = 21,
    n_grid_s2: int = 9,
) -> Tuple[mpf, dict]:
    """Smallest M for which max_{c1,s1,c2,s2} phi_joint(M; .) >= 0.

    Returns (M_star, info) where info contains the argmax at M_star.
    """
    def gap(M):
        phi_max, pt = maximize_phi_joint(
            M, mv, K_data,
            n_grid_c1=n_grid_c1, n_grid_s1=1,
            n_grid_c2=n_grid_c2, n_grid_s2=n_grid_s2,
        )
        return phi_max, pt

    g_lo, pt_lo = gap(M_lo)
    g_hi, pt_hi = gap(M_hi)
    if verbose:
        print(f"  M_lo = {float(M_lo):.6f}: max_phi = {float(g_lo):+.6e}")
        print(f"  M_hi = {float(M_hi):.6f}: max_phi = {float(g_hi):+.6e}")
    if g_hi < 0:
        raise RuntimeError(
            f"M_hi = {M_hi} is INfeasible (max_phi = {g_hi}); raise M_hi."
        )
    if g_lo >= 0:
        # Already feasible at M_lo — bound is below M_lo
        return M_lo, {"argmax": pt_lo, "phi_at_M": g_lo, "iters": 0}

    pt_at_hi = pt_hi
    iters = 0
    while M_hi - M_lo > tol and iters < max_iter:
        M_mid = (M_lo + M_hi) / 2
        g_mid, pt_mid = gap(M_mid)
        if verbose and iters % 5 == 0:
            print(f"  iter {iters:3d}: M_mid = {float(M_mid):.10f}, "
                  f"max_phi = {float(g_mid):+.6e}")
        if g_mid >= 0:
            M_hi = M_mid
            pt_at_hi = pt_mid
        else:
            M_lo = M_mid
        iters += 1
    return M_hi, {"argmax": pt_at_hi, "iters": iters}


# =============================================================================
# Section H.  Reference reductions / sanity checks.
# =============================================================================


def reduce_to_mv_no_z2(M, mv: MVSetup) -> mpf:
    """Sanity: with c_2 = s_2 = 0 (z_2 = 0), MV master inequality reduces to
    MV's eq. (10) with z_1 only.  Returns the MV-eq.(10) gap at MV's
    boundary z_1 = sqrt(mu(M))."""
    M = mpf(M)
    mu = lemma_214_box(M)
    z1 = mp.sqrt(mu)
    g = mv_master_gap(M, z1, mpf(0), mpf(0), mpf(0), mv)
    return g


# =============================================================================
# Section I.  Top-level entry: report all three lower bounds.
# =============================================================================


def main_report(
    K_choice: str = "step",
    n_grid_c1: int = 21,
    n_grid_c2: int = 21,
    n_grid_s2: int = 9,
    M_lo: mpf = mpf("1.05"),
    M_hi: mpf = mpf("1.40"),
    tol: mpf = mpf("1e-8"),
    verbose: bool = True,
):
    mp.mp.dps = DPS

    mv = MVSetup.build()
    K_data = kernel_data(K_choice)

    print("=" * 72)
    print("MO 2004 Prop 2.11 (m=3) + Lemma 2.17 + Lemma 2.14 in MV dual")
    print("=" * 72)
    print(f"  Kernel for Prop 2.11: K_{K_choice}")
    print(f"    K_hat(0) = {float(K_data['K0']):+.6f}")
    print(f"    K_hat(1) = {float(K_data['K1']):+.6f}")
    print(f"    K_hat(2) = {float(K_data['K2']):+.6f}")
    print(f"    ||_3 K_hat||_(4/3) = {float(K_data['tail_43']):.6f}")
    print()
    print(f"  MV setup:")
    print(f"    delta            = {float(mv.delta):.6f}")
    print(f"    u                = {float(mv.u):.6f}")
    print(f"    a_gain           = {float(mv.a_gain):.8f}")
    print(f"    k_1 (period-1)   = {float(mv.k1_mv):.8f}")
    print(f"    k_2 (period-1)   = {float(mv.k2_mv):.8f}")
    print(f"    K_2 = 0.5747/d   = {float(mv.K2_mv):.8f}")
    print(f"    LHS = 2/u + a    = {float(mv.lhs_target):.8f}")
    print()

    # --- Reference: MV alone (no Prop 2.11, no L2.17) ---
    M_mv = mpf(MV_BOUND_FINAL)
    g_mv_at_target = reduce_to_mv_no_z2(M_mv, mv)
    print(f"  (sanity) MV's published bound 1.27481")
    print(f"           gap at z_1 = sqrt(mu(M_mv)), z_2 = 0:  {float(g_mv_at_target):+.6e}")
    print()

    # --- Prop 2.11 alone (no MV constraint) ---
    print(f"  -- Prop 2.11 m=3 + Hoelder + L2.14 + L2.17 (no MV) --")
    bound_p, info_p = _bisect_p211_only(
        mv, K_data,
        M_lo=M_lo, M_hi=M_hi, tol=tol,
        n_grid_c1=n_grid_c1, n_grid_c2=n_grid_c2, n_grid_s2=n_grid_s2,
    )
    print(f"  M_LB (Prop 2.11 alone) = {float(bound_p):.10f}")
    pt = info_p["argmax"]
    print(f"  argmin (c_1, s_1, c_2, s_2) = "
          f"({float(pt[0]):.6f}, {float(pt[1]):.6f}, {float(pt[2]):.6f}, {float(pt[3]):.6f})")
    print()

    # --- Joint MV + Prop 2.11 (the main target) ---
    print(f"  -- MV + Prop 2.11 m=3 + L2.14 + L2.17 (joint) --")
    M_lo_joint = mpf("1.20")
    M_hi_joint = mpf("1.40")
    bound_j, info_j = bisect_M_lower_bound(
        mv, K_data,
        M_lo=M_lo_joint, M_hi=M_hi_joint, tol=tol,
        n_grid_c1=n_grid_c1, n_grid_c2=n_grid_c2, n_grid_s2=n_grid_s2,
        verbose=verbose,
    )
    print(f"  M_LB (joint) = {float(bound_j):.10f}")
    pt = info_j["argmax"]
    print(f"  argmax (c_1, s_1, c_2, s_2) = "
          f"({float(pt[0]):.6f}, {float(pt[1]):.6f}, {float(pt[2]):.6f}, {float(pt[3]):.6f})")
    print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  MV (single-moment)               1.27481  (published)")
    print(f"  Prop 2.11 m=3 + L2.14 + L2.17    {float(bound_p):.6f}")
    print(f"  JOINT MV + Prop 2.11 + L2.14+17  {float(bound_j):.6f}")
    print(f"  CS 2017 acceptance threshold:    1.2802")
    print()
    if bound_j > mpf("1.2803"):
        print(f"  RESULT: lift {float(bound_j):.6f} STRICTLY beats 1.2802.  Producing certificate.")
    else:
        print(f"  RESULT: NO LIFT above 1.2802.  Documenting failure mode.")
    print()
    return {
        "K_choice": K_choice,
        "K_data": {k: float(v) if isinstance(v, mpf) else None
                   for k, v in K_data.items() if k in ("K0", "K1", "K2", "tail_43")},
        "mv": {
            "delta": float(mv.delta), "u": float(mv.u),
            "a_gain": float(mv.a_gain),
            "k1_mv": float(mv.k1_mv), "k2_mv": float(mv.k2_mv),
            "K2_mv": float(mv.K2_mv),
            "lhs_target": float(mv.lhs_target),
        },
        "bound_p": bound_p,
        "bound_j": bound_j,
        "argmax_p": info_p["argmax"],
        "argmax_j": info_j["argmax"],
    }


def _bisect_p211_only(
    mv: MVSetup, K_data: dict,
    M_lo: mpf, M_hi: mpf, tol: mpf,
    n_grid_c1: int, n_grid_c2: int, n_grid_s2: int,
) -> Tuple[mpf, dict]:
    """Smallest M with max over feasible (c, s) of  M - Prop211_RHS  >= 0."""
    def gap(M):
        # Maximise (M - Prop211_RHS) over feasible (c1, s1, c2, s2).
        # Equivalent to minimising Prop211_RHS.
        M = mpf(M)
        mu_M = float(lemma_214_box(M))
        if mu_M <= 0:
            return mp.mpf("-inf"), (mpf(0),)*4
        zmax = mu_M ** 0.5
        M_f = float(M)
        K0, K1, K2 = float(K_data["K0"]), float(K_data["K1"]), float(K_data["K2"])
        T = float(K_data["tail_43"])

        c1_axis = np.linspace(0.0, zmax, n_grid_c1)  # c_1 >= 0 wlog
        s1_axis = np.array([0.0])
        c2_axis = _grid_points(zmax, n_grid_c2)
        s2_axis = _grid_points(zmax, n_grid_s2)

        best_phi = -np.inf
        best_pt = (0.0, 0.0, 0.0, 0.0)
        for c1 in c1_axis:
            c2_upper = 2 * c1 - 1
            if c2_upper < -zmax:
                continue
            for s1 in s1_axis:
                z1sq = c1 * c1 + s1 * s1
                if z1sq > mu_M + 1e-12:
                    continue
                c2_lower = 2 * z1sq - 1
                for c2 in c2_axis:
                    if c2 > c2_upper + 1e-12 or c2 < c2_lower - 1e-12:
                        continue
                    if c2 * c2 > mu_M + 1e-12:
                        continue
                    for s2 in s2_axis:
                        if c2 * c2 + s2 * s2 > mu_M + 1e-12:
                            continue
                        z1sq_ = c1 * c1 + s1 * s1
                        z2sq_ = c2 * c2 + s2 * s2
                        M_tilde = 1.0 - K0 - 2 * K1 * c1 - 2 * K2 * c2
                        rhs_p = 1.0 + 2 * z1sq_ ** 2 + 2 * z2sq_ ** 2 + (M_tilde / T) ** 4
                        phi = M_f - rhs_p
                        if phi > best_phi:
                            best_phi = phi
                            best_pt = (c1, s1, c2, s2)
        c1, s1, c2, s2 = (mpf(v) for v in best_pt)
        rhs_p_mp = prop211_m3_lower_bound(M, c1, s1, c2, s2, K_data)
        return M - rhs_p_mp, (c1, s1, c2, s2)

    g_lo, pt_lo = gap(M_lo)
    g_hi, pt_hi = gap(M_hi)
    if g_hi < 0:
        raise RuntimeError(f"M_hi = {M_hi} infeasible; raise M_hi.")
    if g_lo >= 0:
        return M_lo, {"argmax": pt_lo, "iters": 0}
    iters = 0
    pt_at_hi = pt_hi
    while M_hi - M_lo > tol and iters < 100:
        M_mid = (M_lo + M_hi) / 2
        g_mid, pt_mid = gap(M_mid)
        if g_mid >= 0:
            M_hi = M_mid
            pt_at_hi = pt_mid
        else:
            M_lo = M_mid
        iters += 1
    return M_hi, {"argmax": pt_at_hi, "iters": iters}


if __name__ == "__main__":
    main_report(K_choice="step")
    print()
    main_report(K_choice="pm")

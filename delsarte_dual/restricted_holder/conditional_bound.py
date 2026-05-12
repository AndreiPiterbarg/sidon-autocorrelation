"""Conditional lower bound on C_{1a} under a restricted form of MO 2004 Conj. 2.9.

Hyp_R(c, M_max): for every nonneg pdf f on [-1/4, 1/4] with ||f*f||_inf <= M_max,
                 ||f*f||_2^2 <= c * ||f*f||_inf * ||f*f||_1.

Conditional theorem (proved in derivation.md modulo Hyp_R):
   Hyp_R(c=log16/pi, M_max=M_target) implies C_{1a} >= M_target
                                              =  1.37842197377542... (40 dps).

The hypothesis is tight: M_max = M_target is the *minimum* M_max value under
which the proof's contradiction closes. The earlier draft used M_max = 1.51
("Sidon regime"), which is valid a fortiori but a strictly weaker conditional
statement.

The rigorous lower bound is M_optimal = inf over admissible z_1 of M_master(z_1),
where M_master(z_1) is the smallest M satisfying the modified MV master inequality
(MV eq. (10) with the C-S step's ||f*f||_2^2 replaced by c*M):

    2/u + a   <=   M + 1 + 2 z_1^2 k_1 + sqrt(c*M - 1 - 2 z_1^4)
                                       * sqrt(||K||_2^2 - 1 - 2 k_1^2).

For c = 1, the inf is at the Lemma 3.4 boundary z_1 = sqrt(M sin(pi/M)/pi) which
happens to evaluate to 0.50426 at M = M*(1) ~= 1.27484; this self-consistent
boundary value is MV's published recipe. For c < 1, the inf is at an *interior*
critical point z_1* < 0.50426 (the saddle of RHS w.r.t. z_1), giving the smaller
``conditional_bound_optimal`` value. The MV-recipe value for c < 1 (using z_1
clipped at the c = 1 boundary 0.50426 instead of the actual interior optimum)
is reported as ``conditional_bound_recipe`` for historical comparison only.

CLI:  python -m delsarte_dual.restricted_holder.conditional_bound
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
from mpmath import mpf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delsarte_dual.mv_bound import (  # noqa: E402
    MV_COEFFS_119, MV_DELTA, MV_U, MV_K2_BOUND_OVER_DELTA,
    min_G_on_0_quarter, S1_sum, gain_parameter, k1_value,
)


# MV's choice of z_1 (the sup of the Lemma 3.4 admissible range at M ~ 1.275;
# see MV p. 7 line 350). Used by the recipe-form bound.
MV_Z1 = mpf("0.50426")


# Decimal strings of MV's 119 tabulated cosine coefficients (transcribed
# verbatim from MV Appendix; see ``delsarte_dual/mv_construction_detailed.md``
# §4 and the matching list in ``mv_bound.py``). We hold them as strings so
# we can re-parse at the current dps for full precision; the imported
# ``MV_COEFFS_119`` is materialised at module-load dps (~15) and so is too
# coarse for 40-digit M(c).
MV_COEFFS_119_STR = [
    "+2.16620392",   "-1.87775750",   "+1.05828868",   "-7.29790538e-01",
    "+4.28008515e-01","+2.17832838e-01","-2.70415201e-01","+2.72834790e-02",
    "-1.91721888e-01","+5.51862060e-02","+3.21662512e-01","-1.64478392e-01",
    "+3.95478603e-02","-2.05402785e-01","-1.33758316e-02","+2.31873221e-01",
    "-4.37967118e-02","+6.12456374e-02","-1.57361919e-01","-7.78036253e-02",
    "+1.38714392e-01","-1.45201483e-04","+9.16539824e-02","-8.34020840e-02",
    "-1.01919986e-01","+5.94915025e-02","-1.19336618e-02","+1.02155366e-01",
    "-1.45929982e-02","-7.95205457e-02","+5.59733152e-03","-3.58987179e-02",
    "+7.16132260e-02","+4.15425065e-02","-4.89180454e-02","+1.65425755e-03",
    "-6.48251747e-02","+3.45951253e-02","+5.32122058e-02","-1.28435276e-02",
    "+1.48814403e-02","-6.49404547e-02","-6.01344770e-03","+4.33784473e-02",
    "-2.53362778e-04","+3.81674519e-02","-4.83816002e-02","-2.53878079e-02",
    "+1.96933442e-02","-3.04861682e-03","+4.79203471e-02","-2.00930265e-02",
    "-2.73895519e-02","+3.30183589e-03","-1.67380508e-02","+4.23917582e-02",
    "+3.64690190e-03","-1.79916104e-02","+7.31661649e-05","-2.99875575e-02",
    "+2.71842526e-02","+1.41806855e-02","-6.01781076e-03","+5.86806100e-03",
    "-3.32350597e-02","+9.23347466e-03","+1.47071722e-02","-7.42858080e-04",
    "+1.63414270e-02","-2.87265671e-02","-1.64287280e-03","+8.02601605e-03",
    "-7.62613027e-04","+2.18735533e-02","-1.78816282e-02","-6.58341101e-03",
    "+2.67706547e-03","-6.25261247e-03","+2.24942824e-02","-8.10756022e-03",
    "-5.68160823e-03","+7.01871209e-05","-1.15294332e-02","+1.83608944e-02",
    "-1.20567880e-03","-3.13147456e-03","+1.39083675e-03","-1.49312478e-02",
    "+1.32106694e-02","+1.73474188e-03","-8.53469045e-04","+4.03211203e-03",
    "-1.55352991e-02","+8.74711543e-03","+1.93998895e-03","-2.71357322e-05",
    "+6.13179585e-03","-1.41983972e-02","+5.84710551e-03","+9.22578333e-04",
    "-2.16583469e-04","+7.07919829e-03","-1.18488582e-02","+4.39698322e-03",
    "-8.91346785e-05","-3.42086367e-04","+6.46355636e-03","-8.87555371e-03",
    "+3.56799654e-03","-4.97335419e-04","-8.04560326e-04","+5.55076717e-03",
    "-7.13560569e-03","+4.53679038e-03","-3.33261516e-03","+2.35463427e-03",
    "+2.04023789e-04", "-1.27746711e-03","+1.81247830e-04",
]
assert len(MV_COEFFS_119_STR) == 119, len(MV_COEFFS_119_STR)


def _hi_prec_coeffs():
    """MV's 119 coefficients re-parsed at the current dps."""
    return [mp.mpf(s) for s in MV_COEFFS_119_STR]


def _hi_prec(x, default_str):
    """Return ``x`` re-parsed at the current dps if ``x`` matches the default.

    The module-level constants ``MV_DELTA``, ``MV_U``, ``MV_K2_BOUND_OVER_DELTA``
    are materialised at module-load precision (typically dps~15, 53 binary
    bits — float64 equivalent). To compute M(c) to 40 decimal digits we need
    the rationals 138/1000, 638/1000, 5747/10000 at full current precision,
    so we re-parse from the canonical decimal string when the caller passes
    the module default.
    """
    if isinstance(x, str):
        return mp.mpf(x)
    if x is None:
        return mp.mpf(default_str)
    if mp.almosteq(mpf(x), mp.mpf(default_str), abs_eps=mpf("1e-10")):
        return mp.mpf(default_str)
    return mpf(x)


def _resolve_c(c):
    """Re-evaluate c = log(16)/pi at the current dps if the caller passed a
    low-precision copy. Other c values are returned via mpf(c)."""
    c_input = mpf(c)
    log16_over_pi_hi = mp.log(16) / mp.pi
    if mp.almosteq(c_input, log16_over_pi_hi, abs_eps=mpf("1e-12")):
        return log16_over_pi_hi
    return c_input


def _ingredients(delta=MV_DELTA, u=MV_U, n_grid_minG: int = 400001):
    """Return (K2, a_gain, k1, u_hi) as mpmath mpfs at the current dps."""
    delta = _hi_prec(delta, "0.138")
    u = _hi_prec(u, "0.638")
    K2_bound = _hi_prec(MV_K2_BOUND_OVER_DELTA, "0.5747")
    K2 = K2_bound / delta
    coeffs = _hi_prec_coeffs()
    min_G, _ = min_G_on_0_quarter(a_coeffs=coeffs, u=u, n_grid=n_grid_minG)
    S1 = S1_sum(a_coeffs=coeffs, delta=delta, u=u)
    a_gain = gain_parameter(mpf(min_G), S1, u=u)
    k1 = k1_value(delta=delta)
    return K2, a_gain, k1, u


def _admissible_z1_max_sq(M, c, K2, k1):
    """Upper bound on z_1^2 = |\\hat f(1)|^2 admissible at M.

    Combines:
      Lemma 3.4 (MO 2004 Lemma 2.14):  z_1^2 <= M sin(pi/M) / pi.
      Hyp_R sqrt domain (so the modified eq. (10) is real-valued):
                                       z_1^4 <= (c*M - 1)/2.
    Both are necessary for f admissible under Hyp_R; the tighter one binds.
    """
    M = mpf(M); c = mpf(c)
    lemma34 = M * mp.sin(mp.pi / M) / mp.pi  # z_1^2 upper bound from Lemma 3.4
    hypR_arg = c * M - 1
    if hypR_arg <= 0:
        # Hyp_R is unsatisfiable at this M (would force ||f*f||_2^2 < 1 + 2 z_1^4
        # which contradicts Parseval); admissible set is empty.
        return mpf(0)
    hypR = mp.sqrt(hypR_arg / 2)             # z_1^2 upper bound from Hyp_R
    return min(lemma34, hypR)


def _master_rhs(M, c, z1, K2, a_gain, k1, u):
    """RHS of the modified master inequality (MV eq. (10) with c-substitution).

    Returns -inf if the radicand becomes negative (so the inequality cannot be
    interpreted as a real-valued bound at this (M, z_1) pair).
    """
    M = mpf(M); c = mpf(c); z1 = mpf(z1)
    z1sq = z1 * z1
    z14 = z1sq * z1sq
    A2 = K2 - 1 - 2 * k1 * k1
    rad = c * M - 1 - 2 * z14
    if rad < 0 or A2 < 0:
        return mpf("-inf")
    return M + 1 + 2 * z1sq * k1 + mp.sqrt(rad) * mp.sqrt(A2)


def modified_master_residual(M, c, z1=MV_Z1, delta=MV_DELTA, u=MV_U):
    """Residual LHS - RHS of the modified MV eq. (10).

    Equation (residual = 0 at the bound M_master(z_1)):
        2/u + a   ==   M + 1 + 2 z_1^2 k_1 + sqrt(c*M - 1 - 2 z_1^4)
                                            * sqrt(K_2 - 1 - 2 k_1^2).

    Sign convention: residual = LHS - RHS, so a *positive* residual means
    the master inequality fails (RHS < LHS) — there is no admissible f with
    these (M, z_1). A residual of zero is the boundary M_master(z_1).
    """
    K2, a_gain, k1, u_hi = _ingredients(delta=delta, u=u)
    z1 = _hi_prec(z1, "0.50426")
    c = _resolve_c(c)
    rhs = _master_rhs(M, c, z1, K2, a_gain, k1, u_hi)
    if rhs == mpf("-inf"):
        return mpf("+inf")
    lhs = 2 / u_hi + a_gain
    return lhs - rhs


def _solve_master_at_z1(c, z1, K2, a_gain, k1, u):
    """Solve the modified master equation for M at fixed z_1 (closed form).

    Returns the smallest M with RHS(M, z_1) = LHS = 2/u + a.
    """
    c = mpf(c); z1 = mpf(z1)
    A2 = K2 - 1 - 2 * k1 * k1
    if A2 < 0:
        raise ValueError(f"K2 - 1 - 2 k1^2 = {A2} < 0; inputs inconsistent.")
    A = mp.sqrt(A2)
    z1sq = z1 * z1
    z14 = z1sq * z1sq
    B = 2 / u + a_gain
    C = 2 * z1sq * k1
    rhs_const = c * (B - 1 - C) - 1 - 2 * z14
    disc = (c * A) ** 2 + 4 * rhs_const
    if disc < 0:
        raise ValueError(f"discriminant < 0: {disc}; inputs inconsistent.")
    s_star = (-c * A + mp.sqrt(disc)) / 2
    if s_star < 0:
        s_star = mpf(0)
    return (s_star * s_star + 1 + 2 * z14) / c


# -----------------------------------------------------------------------------
# Recipe form (MV's z_1 = 0.50426 fixed).
# Sub-optimal at c < 1 but reproduces MV's published 1.27481 at c = 1.
# -----------------------------------------------------------------------------
def conditional_bound_recipe(c, delta=MV_DELTA, u=MV_U, dps: int = 40,
                             z1=MV_Z1, n_grid_minG: int = 400001) -> mp.mpf:
    """Recipe-form bound: solve MV eq. (10) at z_1 = 0.50426 fixed.

    *This is NOT the rigorous lower bound on C_{1a} for c < 1.* It is the value
    obtained by following MV's recipe verbatim — using their z_1 = 0.50426 (the
    Lemma 3.4 boundary at M ~= 1.275) regardless of c. At c = 1 this recipe
    coincides with the rigorous inf-over-z_1 bound (because the interior
    critical point of RHS w.r.t. z_1 happens to lie *outside* the Lemma 3.4
    admissible range, so the boundary z_1 = 0.50426 IS the argmin of M_master).
    At c < 1 the interior critical point lies inside the admissible range, so
    z_1 = 0.50426 is sub-optimal and ``conditional_bound_recipe(c) >
    conditional_bound_optimal(c)``.

    Use ``conditional_bound_optimal`` for the rigorous lower bound.

    Returns
    -------
    M_recipe : mpmath mpf
        The unique M satisfying the modified eq. (10) with z_1 = 0.50426.
    """
    mp.mp.dps = dps
    delta = _hi_prec(delta, "0.138")
    u = _hi_prec(u, "0.638")
    K2_bound = _hi_prec(MV_K2_BOUND_OVER_DELTA, "0.5747")
    z1 = _hi_prec(z1, "0.50426")
    c = _resolve_c(c)
    K2 = K2_bound / delta
    coeffs = _hi_prec_coeffs()
    min_G, _ = min_G_on_0_quarter(a_coeffs=coeffs, u=u, n_grid=n_grid_minG)
    S1 = S1_sum(a_coeffs=coeffs, delta=delta, u=u)
    a_gain = gain_parameter(mpf(min_G), S1, u=u)
    k1 = k1_value(delta=delta)
    return _solve_master_at_z1(c, z1, K2, a_gain, k1, u)


# -----------------------------------------------------------------------------
# Optimal (rigorous) form: inf over z_1 of M_master(z_1).
# -----------------------------------------------------------------------------
def conditional_bound_optimal(c, delta=MV_DELTA, u=MV_U, dps: int = 40,
                              n_grid_minG: int = 400001,
                              M_max=None) -> mp.mpf:
    """Rigorous lower bound: inf over admissible z_1 of M_master(z_1).

    The function M_master: z_1 |--> M (with RHS(M, z_1) = LHS) is V-shaped:
    decreasing on [0, z_1*], increasing on [z_1*, z_max], where z_1* is the
    interior critical point of RHS(M, z_1) w.r.t. z_1 (also the argmax of
    RHS at fixed M, and the argmin of M_master). Closed form for z_1*:

        z_1*^4 = k_1^2 (c*M - 1) / (||K||_2^2 - 1).

    At z_1 = z_1*, the modified eq. (10) reduces to the *no-z_1 form*

        2/u + a = M + 1 + sqrt((c*M - 1) (||K||_2^2 - 1)),

    a quadratic in M with closed-form solution.

    The inf is at z_1* iff z_1* is admissible (Lemma 3.4 + Hyp_R sqrt domain);
    otherwise the inf is at the binding boundary, found by self-consistent
    iteration on z_1 = sqrt(z_1_max_sq(M)).

    Parameters
    ----------
    c : mpmath-compatible
        Holder-slack constant in Hyp_R.
    delta, u : mpmath-compatible
        MV's kernel parameters (default 0.138, 0.638).
    dps : int
        mpmath decimal precision.
    n_grid_minG : int
        Float-grid resolution for ``min_G_on_0_quarter``.
    M_max : mpmath-compatible or None
        Restriction-class threshold from Hyp_R(c, M_max). If None, defaults
        to the computed M_target itself — the *minimum* M_max under which
        the contradiction in the proof closes. Pass an explicit value > M_target
        to use a looser hypothesis (a fortiori); a value < M_target raises
        AssertionError because the proof would not close.

    Returns
    -------
    M_optimal : mpmath mpf
        The rigorous worst-case-z_1 lower bound on C_{1a} given Hyp_R(c, M_max).
    """
    mp.mp.dps = dps
    delta = _hi_prec(delta, "0.138")
    u = _hi_prec(u, "0.638")
    K2_bound = _hi_prec(MV_K2_BOUND_OVER_DELTA, "0.5747")
    c = _resolve_c(c)
    K2 = K2_bound / delta
    coeffs = _hi_prec_coeffs()
    min_G, _ = min_G_on_0_quarter(a_coeffs=coeffs, u=u, n_grid=n_grid_minG)
    S1 = S1_sum(a_coeffs=coeffs, delta=delta, u=u)
    a_gain = gain_parameter(mpf(min_G), S1, u=u)
    k1 = k1_value(delta=delta)

    # Step 1: try the interior critical point (no-z_1 form).
    # M^2 - (2E + cD) M + (D + E^2) = 0, with D = K2 - 1, E = 2/u + a - 1.
    D = K2 - 1
    E = 2 / u + a_gain - 1
    p = 2 * E + c * D
    q = D + E * E
    disc = p * p - 4 * q
    if disc < 0:
        raise ValueError(f"no-z_1 discriminant < 0: {disc}")
    M_interior = (p - mp.sqrt(disc)) / 2

    # Step 2: check whether z_1*(M_interior) is admissible. If yes, this is the inf.
    z1_star_sq = k1 * mp.sqrt((c * M_interior - 1) / D)
    z1_max_sq = _admissible_z1_max_sq(M_interior, c, K2, k1)
    if z1_star_sq <= z1_max_sq:
        M_target = M_interior
    else:
        # Step 3: interior critical point exceeds the admissible range; the inf is
        # at the binding boundary z_1 = sqrt(z_1_max_sq(M)). Self-consistent fixed
        # point (z_1_max depends on M, so we iterate).
        M = M_interior
        tol = mpf(10) ** (-(dps - 5))
        for _ in range(500):
            z1_max_sq_M = _admissible_z1_max_sq(M, c, K2, k1)
            z1 = mp.sqrt(z1_max_sq_M)
            M_new = _solve_master_at_z1(c, z1, K2, a_gain, k1, u)
            if abs(M - M_new) < tol:
                M = M_new
                break
            M = M_new
        M_target = M

    # Hyp_R closure check: the proof "assume f admissible with M < M_target;
    # then ||f*f||_inf < M_target <= M_max, so Hyp_R applies, contradiction"
    # only closes if M_max >= M_target. Default to the *minimum* M_max that
    # works, which is M_target itself.
    if M_max is None:
        M_max_resolved = M_target
    else:
        M_max_resolved = mpf(M_max)
    assert M_max_resolved >= M_target, (
        f"M_max={M_max_resolved} < M_target={M_target}; the conditional "
        f"theorem proof does not close (Hyp_R would not apply to f's with "
        f"||f*f||_inf in [M_max, M_target))."
    )
    return M_target


# -----------------------------------------------------------------------------
# Bisection cross-check: 1-D minimization of M_master over admissible z_1.
# -----------------------------------------------------------------------------
def conditional_bound_optimal_bisection(c, delta=MV_DELTA, u=MV_U, dps: int = 40,
                                        n_grid_minG: int = 400001,
                                        n_grid: int = 4001) -> tuple:
    """Cross-check: golden-section search over z_1 of M_master(z_1) on the
    admissible range. Returns (M_optimal, z_1_optimal). Used only to verify
    the closed-form ``conditional_bound_optimal`` agrees.
    """
    mp.mp.dps = dps
    delta = _hi_prec(delta, "0.138")
    u = _hi_prec(u, "0.638")
    K2_bound = _hi_prec(MV_K2_BOUND_OVER_DELTA, "0.5747")
    c = _resolve_c(c)
    K2 = K2_bound / delta
    coeffs = _hi_prec_coeffs()
    min_G, _ = min_G_on_0_quarter(a_coeffs=coeffs, u=u, n_grid=n_grid_minG)
    S1 = S1_sum(a_coeffs=coeffs, delta=delta, u=u)
    a_gain = gain_parameter(mpf(min_G), S1, u=u)
    k1 = k1_value(delta=delta)

    # Use the closed-form M for z_1_max bound to set the bracket.
    M_init = conditional_bound_optimal(c, delta=delta, u=u, dps=dps,
                                       n_grid_minG=n_grid_minG)
    z1_max_sq = _admissible_z1_max_sq(M_init, c, K2, k1)
    z1_max = mp.sqrt(z1_max_sq)

    # Coarse grid search to bracket the minimum.
    z_grid = [z1_max * mpf(i) / (n_grid - 1) for i in range(n_grid)]
    Ms = []
    for z in z_grid:
        try:
            Ms.append(_solve_master_at_z1(c, z, K2, a_gain, k1, u))
        except ValueError:
            Ms.append(mpf("+inf"))
    idx = min(range(n_grid), key=lambda i: Ms[i])

    # Refine via golden-section in a small bracket around the grid argmin.
    phi = (mp.sqrt(5) - 1) / 2
    lo_idx = max(0, idx - 1); hi_idx = min(n_grid - 1, idx + 1)
    a, b = z_grid[lo_idx], z_grid[hi_idx]
    f_eval = lambda z: _solve_master_at_z1(c, z, K2, a_gain, k1, u)
    c_pt = b - phi * (b - a)
    d_pt = a + phi * (b - a)
    fc = f_eval(c_pt); fd = f_eval(d_pt)
    tol = mpf(10) ** (-(dps - 5))
    for _ in range(500):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d_pt, fd = d_pt, c_pt, fc
            c_pt = b - phi * (b - a)
            fc = f_eval(c_pt)
        else:
            a, c_pt, fc = c_pt, d_pt, fd
            d_pt = a + phi * (b - a)
            fd = f_eval(d_pt)
    z_opt = (a + b) / 2
    M_opt = f_eval(z_opt)
    return M_opt, z_opt


# Kept as a back-compat alias (was the "main" function name in the previous
# revision); always returns the recipe value.
def conditional_bound(c, *args, **kwargs):
    """Deprecated alias of ``conditional_bound_recipe``.

    Provided for backward compatibility with the previous revision; raises
    no warning to avoid noise in test runs, but new code should use either
    ``conditional_bound_optimal`` (rigorous) or ``conditional_bound_recipe``
    (MV's z_1 = 0.50426 fixed; sub-optimal at c < 1).
    """
    return conditional_bound_recipe(c, *args, **kwargs)


def main():
    """CLI: print both M_recipe(c) and M_optimal(c) for c in a band around log16/pi."""
    mp.mp.dps = 50
    c_table = [
        mpf("0.90"), mpf("0.92"), mpf("0.94"), mpf("0.95"), mpf("0.96"),
        mpf("0.97"), mpf("0.98"), mpf("0.99"), mpf("1.0"),
    ]
    print("=" * 92)
    print("Conditional bound on C_{1a} under Hyp_R(c, M_max=1.51)")
    print("=" * 92)
    print(f"{'c':>8s}  {'M_recipe (z_1=0.50426)':>32s}  {'M_optimal (rigorous inf z_1)':>34s}")
    print("-" * 92)
    for c in c_table:
        Mr = conditional_bound_recipe(c, dps=50)
        Mo = conditional_bound_optimal(c, dps=50)
        print(f"{mp.nstr(c, 6):>8s}  {mp.nstr(Mr, 25):>32s}  {mp.nstr(Mo, 25):>34s}")
    print()
    c_mo = mp.log(16) / mp.pi
    print(f"c = log(16)/pi = {mp.nstr(c_mo, 50)}")
    Mr = conditional_bound_recipe(c_mo, dps=50)
    Mo = conditional_bound_optimal(c_mo, dps=50)
    print(f"M_recipe (sub-optimal):  {mp.nstr(Mr, 50)}")
    print(f"M_optimal (rigorous):    {mp.nstr(Mo, 50)}")
    print(f"Headline conditional bound: C_{{1a}} >= {mp.nstr(Mo, 30)} (under Hyp_R).")


if __name__ == "__main__":
    main()

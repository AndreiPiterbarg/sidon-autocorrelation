"""3-point Lasserre with L^2-type constraint via auto-correlation bound.

Key idea: replace the L^infty bound on rho^(2) (which excludes the natural
near-optimizers like MV's f ~ 1/x^{1/3}) with an L^2-type bound:

    ||f||_2^2 = R(0) where R(t) = (f * bar_f)(t) = integral f(x) f(x-t) dx.

R(0) can be approximated by integrating R against a bump peaked at 0.

In moment terms, R has moments r_a = sum_j C(a, j) (-1)^j g_{a-j, j}.
For a 1D bump q with int q = 1, peaked at 0:
    int q(t) R(t) dt = sum_a q_a r_a   (linear in g_{ab})
which approximates R(0) = ||f||_2^2 for sharp q.

Adding the constraint  int q R <= K   to the V3 SDP:
- For absolutely continuous f with ||f||_2^2 <= K, the constraint is satisfied.
- For singular pseudo-measures (e.g., diagonal): R(0) = infty, constraint violated.
- Crucially, MV's near-optimizer with ||f||_2^2 ~ 2.45 IS in this set.

Combined with the V3 3-point Lasserre framework, this might give meaningful
lifts that DO bound C_{1a} (since the unbounded-near-optimum is included).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from lasserre.threepoint_full import (
    enum_multi_indices, MomentVarMap, moment_matrix, localizer_matrix,
    BuildInfo,
)
from lasserre.threepoint_alternatives import (
    bump_objective_coefficients_rescaled, lebesgue_moments_1d,
    add_l_infty_constraint,
)


def autocorr_moments_from_g(g_lookup, max_a: int) -> List:
    """Compute r_a = int t^a R(t) dt as linear functional of g_{ab}.
    r_a = sum_j C(a, j) (-1)^j g_{a-j, j}.

    g_lookup is a callable g_lookup(a, b) -> CVXPY expr (scalar).
    """
    out = []
    for a in range(max_a + 1):
        terms = []
        for j in range(a + 1):
            coef = math.comb(a, j) * ((-1) ** j)
            terms.append(coef * g_lookup(a - j, j))
        if terms:
            out.append(cp.sum(cp.hstack(terms)))
        else:
            out.append(cp.Constant(0.0))
    return out


def make_g_lookup_from_map(g_map):
    """Wrap MomentVarMap so we can call as g_lookup(a, b)."""
    def lookup(a: int, b: int):
        return g_map.get((a, b))
    return lookup


def bump_polynomial_coeffs(N_q: int, support_half: float = 0.5) -> Dict[int, float]:
    """Compute MONOMIAL COEFFICIENTS q^{coef}_a of q_N(t) = (1 - (t/support_half)^2)^N_q / Z
    where Z = int_{-support_half}^{support_half} (1 - (t/support_half)^2)^N_q dt
    is the normalization making int q_N = 1.

    Then  q_N(t) = sum_a q^{coef}_a t^a  is the monomial expansion.

    int q_N(t) R(t) dt = sum_a q^{coef}_a * r_a  where r_a = int t^a R dt.

    These coefficients alternate in sign: q^{coef}_{2j} = C(N_q, j) (-1)^j (1/support_half^{2j}) / Z.
    """
    from scipy.special import beta as beta_fn
    # Normalization Z: int_{-h}^{h} (1 - (t/h)^2)^N dt
    # Substitute u = t/h: = h int_{-1}^1 (1-u^2)^N du = h * B(1/2, N+1)
    Z = support_half * beta_fn(0.5, N_q + 1)
    out = {}
    for j in range(N_q + 1):
        a = 2 * j
        # q^coef_a = C(N_q, j) * (-1)^j / (support_half^a) / Z
        out[a] = math.comb(N_q, j) * ((-1) ** j) / (support_half ** a) / Z
    return out


def add_l2_constraint(constraints: List, g_lookup, K_l2_rescaled: float, N_bump: int = 4,
                       max_g_deg: int = 8, R_support_half: float = 2.0) -> None:
    """Add the constraint int q_N(t) R(t) dt <= K_l2_rescaled where q_N is a bump on
    [-R_support_half, R_support_half] (matching the support of R = autocorrelation),
    centered at 0, normalized to int q_N = 1.  This approximates ||tilde_f||_2^2 = R(0) <= K.

    R_support_half = 2 in rescaled coords (since autocorrelation of f on [-1, 1] is on [-2, 2]).

    The bump  q_N(t) = (1 - (t/h)^2)^N / Z  with h = R_support_half is positive on (-h, h),
    vanishing at +/- h, peaked at 0 with q_N(0) = 1/Z.  As N_bump grows, the bump sharpens
    toward delta_0, giving int q R -> R(0).  However int q R <= sup R *only* if q >= 0 with
    int q = 1, which holds.  And R is a positive measure with R(0) finite for f in L^2.

    For uniform tilde_f = 1/2 on [-1, 1]:  R(t) = (1/4)(2 - |t|),  R(0) = 1/2.
    int q_N R dt should approach 0.5 as N_bump grows.
    """
    max_a = min(2 * N_bump, max_g_deg)
    r = autocorr_moments_from_g(g_lookup, max_a)
    q_coef = bump_polynomial_coeffs(N_bump, support_half=R_support_half)
    terms = []
    for a in range(max_a + 1):
        if a in q_coef and abs(q_coef[a]) > 1e-15:
            terms.append(q_coef[a] * r[a])
    expr = cp.sum(cp.hstack(terms))
    constraints.append(expr <= K_l2_rescaled)


def build_2pt_l2(k: int, N: int, *, epsilon: float = 0.1, K_l2: Optional[float] = None,
                  N_bump: int = 4) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """V3-style 2pt SDP with L^2-type constraint instead of L^infty."""
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}).")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    g_map = MomentVarMap(dim=2, max_deg=2 * k)
    m_map = MomentVarMap(dim=1, max_deg=2 * k)
    constraints = []
    constraints.append(g_map.get((0, 0)) == 1)
    constraints.append(m_map.get((0,)) == 1)
    for a in range(2 * k + 1):
        constraints.append(g_map.get((a, 0)) == m_map.get((a,)))

    block_sizes = []
    constraints.append(moment_matrix(g_map, k) >> 0); block_sizes.append((k+1)*(k+2)//2)
    if k >= 1:
        for axis in range(2):
            constraints.append(localizer_matrix(g_map, k, axis=axis, support_half=SUPPORT_HALF) >> 0)
            block_sizes.append(k*(k+1)//2)
    constraints.append(moment_matrix(m_map, k) >> 0); block_sizes.append(k+1)
    if k >= 1:
        constraints.append(localizer_matrix(m_map, k, axis=0, support_half=SUPPORT_HALF) >> 0)
        block_sizes.append(k)

    # Auto-correlation L^2 constraint: K_l2 is in ORIGINAL coords (||f||_2^2 in [-1/4, 1/4]).
    # In rescaled coords (v = 4u), tilde_f = (1/4) f(v/4), ||tilde_f||_2^2 = (1/4) ||f||_2^2.
    # The autocorrelation R is in original coords, so we need to track in rescaled.
    # In rescaled, the autocorrelation at 0 is ||tilde_f||_2^2 = K_l2 / 4.
    # But the auto-correlation moments r_a in rescaled coords... let me think.
    # Actually since g_{ab} are RESCALED moments, and r_a = sum C(a,j) (-1)^j g_{a-j,j},
    # the r_a are rescaled-coord moments of R = tilde_f * bar(tilde_f).
    # int q(t) R(t) dt in rescaled coords = sum q_a r_a, evaluated for tilde_f.
    # We want ||tilde_f||_2^2 <= K_l2 / 4 (rescaled), so: int q_N R <= K_l2 / 4.
    # We use bump on [-1, 1] (rescaled support of R: [-2, 2] but the part we care about is around 0).
    if K_l2 is not None:
        # In rescaled coords, R = tilde_f * bar(tilde_f) is on [-2, 2].
        # ||tilde_f||_2^2 = K_l2 / 4 (since rescaling u = 4v gives tilde_f = (1/4) f(v/4),
        # ||tilde_f||_2^2 = (1/4)^2 * 4 ||f||_2^2 = (1/4) ||f||_2^2 = K_l2 / 4 ).
        K_resc = K_l2 / 4.0
        add_l2_constraint(constraints, make_g_lookup_from_map(g_map),
                           K_l2_rescaled=K_resc, N_bump=N_bump, max_g_deg=2 * k,
                           R_support_half=2.0)

    # Bump-kernel objective in rescaled coords
    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = sum(c * g_map.get((a, b)) for (a, b), c in coeffs.items() if a + b <= 2 * k)
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=False,
        n_orbits_y=0, n_orbits_g=g_map.n_orbits(), n_orbits_m=m_map.n_orbits(),
        block_sizes=block_sizes, n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    return problem, info, {"g": g_map, "m": m_map}


def build_3pt_l2(k: int, N: int, *, epsilon: float = 0.1,
                  K_l2: Optional[float] = None, N_bump: int = 4,
                  l_inf_M_rho2_orig: Optional[float] = None,
                  l_inf_M_rho3_orig: Optional[float] = None) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """V3-style 3pt SDP with L^2 constraint (and optionally L^infty for comparison)."""
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}).")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    y_map = MomentVarMap(dim=3, max_deg=2 * k)
    constraints = []
    constraints.append(y_map.get((0, 0, 0)) == 1)
    block_sizes = []
    constraints.append(moment_matrix(y_map, k) >> 0); block_sizes.append((k+1)*(k+2)*(k+3)//6)
    if k >= 1:
        for axis in range(3):
            constraints.append(localizer_matrix(y_map, k, axis=axis, support_half=SUPPORT_HALF) >> 0)
            block_sizes.append(k*(k+1)*(k+2)//6)

    # Optional L^infty (V3 style)
    if l_inf_M_rho2_orig is not None:
        M_resc = l_inf_M_rho2_orig / 16.0
        Jmax = 2 * k
        J = lebesgue_moments_1d(Jmax, support_half=SUPPORT_HALF)
        def nu2_lookup(a, b):
            return M_resc * J[a] * J[b] - y_map.get((a, b, 0))
        b2 = enum_multi_indices(2, k)
        rows = []
        for i in range(len(b2)):
            row = []
            for j in range(len(b2)):
                aa = b2[i][0] + b2[j][0]; bb = b2[i][1] + b2[j][1]
                row.append(nu2_lookup(aa, bb))
            rows.append(row)
        constraints.append(cp.bmat(rows) >> 0); block_sizes.append(len(b2))
    if l_inf_M_rho3_orig is not None:
        M_resc = l_inf_M_rho3_orig / 64.0
        sizes = add_l_infty_constraint(constraints, y_map, k, M_density_rescaled=M_resc,
                                        support_half=SUPPORT_HALF)
        block_sizes.extend(sizes)

    # L^2 constraint via 2D marginal moments
    if K_l2 is not None:
        K_resc = K_l2 / 4.0
        def g_lookup(a, b):
            if a + b > 2 * k:
                return cp.Constant(0.0)
            return y_map.get((a, b, 0))
        add_l2_constraint(constraints, g_lookup, K_l2_rescaled=K_resc, N_bump=N_bump,
                           max_g_deg=2 * k, R_support_half=2.0)

    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = sum(c * y_map.get((a, b, 0)) for (a, b), c in coeffs.items() if a + b <= 2 * k)
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=True,
        n_orbits_y=y_map.n_orbits(), n_orbits_g=0, n_orbits_m=0,
        block_sizes=block_sizes, n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    return problem, info, {"y": y_map}

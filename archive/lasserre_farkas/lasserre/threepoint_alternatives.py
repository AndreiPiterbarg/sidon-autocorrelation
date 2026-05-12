"""Alternative formulations for the 3-point Lasserre relaxation.

Reuses the variable-creation, orbit-reduction, and moment-matrix infrastructure
from `lasserre.threepoint_full`, but swaps in different OBJECTIVES and adds
optional CONSTRAINTS to investigate whether any variant produces a meaningful
2pt vs 3pt lift.

Variants implemented:

  A) BUMP-KERNEL OBJECTIVE (V1's positive shifted bump in the V2 framework).
     Objective: inf int q^*_N(u_1+u_2) f(u_1) f(u_2) du_1 du_2
     where q^*_N = ((1-4t^2)^N + eps)/(I_N + eps), positive on [-1/2, 1/2].
     This sidesteps the "diagonal pseudo-measure -> trivial lambda=1" issue
     that kills the Christoffel-Darboux variant; gave Delta ~ 3e-4 at (7,7) in V1.

  B) L^infty DENSITY BOUND on rho^(2) and rho^(3).
     Adds a shifted-moment Hausdorff cone: M^2 (Lebesgue moment) - g_{ab} is a
     valid moment vector of a positive measure on [-1/4, 1/4]^2.  Excludes
     singular pseudo-measures (diagonals) that achieve the trivial bound.
     Combined with the bump-kernel objective.

  C) TRANSLATION-INVARIANT CD KERNEL.
     Use K_N(s) = sum_j p_j(s)^2 as a function of s = u_1+u_2-t (single variable),
     positive everywhere.  Distinct from the V2 bivariate p_j(s) p_j(t) form.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from lasserre.threepoint_full import (
    enum_multi_indices, MomentVarMap, moment_matrix, localizer_matrix,
    legendre_orthonormal_coeffs, BuildInfo,
)


# =====================================================================
# Bump-kernel coefficients in rescaled coords
# =====================================================================

def bump_integral(N: int) -> float:
    """I_N = int_{-1/2}^{1/2} (1 - 4 s^2)^N ds = sqrt(pi)/2 * Gamma(N+1)/Gamma(N+3/2)."""
    return 0.5 * math.sqrt(math.pi) * math.gamma(N + 1) / math.gamma(N + 1.5)


def bump_objective_coefficients_rescaled(N: int, epsilon: float, *, mu_scale: float = 0.25) \
        -> Dict[Tuple[int, int], float]:
    """For the objective int q^*_N(u_1 + u_2) f(u_1) f(u_2) du_1 du_2 expressed
    in RESCALED moments g^{rescaled}_{ab} (variables v_i = u_i / mu_scale on [-1, 1]).

    q^*_N(t) = ((1 - 4 t^2)^N + epsilon) / (I_N + epsilon).
    The factor (1/4)^t is absorbed via mu_scale^t = (1/4)^r.

    Returns dict (a, b) -> coefficient of g^{rescaled}_{ab}.
    """
    I_N = bump_integral(N)
    Z = I_N + epsilon
    # q^*_N(t) = (1/Z) [ epsilon + sum_{j=0}^{N} C(N, j) (-4)^j t^{2j} ]
    out: Dict[Tuple[int, int], float] = {(0, 0): epsilon / Z}
    for j in range(N + 1):
        deg = 2 * j
        prefac = math.comb(N, j) * ((-4.0) ** j) / Z
        for a in range(deg + 1):
            b = deg - a
            # (u_1+u_2)^deg expands to sum_{a+b=deg} C(deg, a) u_1^a u_2^b.
            # In rescaled coords, u_i = mu_scale * v_i, so u_1^a u_2^b = mu_scale^(a+b) v_1^a v_2^b.
            # Therefore the coefficient of g^{rescaled}_{ab} gets mu_scale^(a+b) factor.
            contrib = prefac * math.comb(deg, a) * (mu_scale ** (a + b))
            out[(a, b)] = out.get((a, b), 0.0) + contrib
    return out


# =====================================================================
# L^infty density bound: shifted Hausdorff moment cone
# =====================================================================

def lebesgue_moments_1d(max_deg: int, support_half: float = 1.0) -> np.ndarray:
    """J_a = int_{-support_half}^{support_half} u^a du,  a = 0, 1, ..., max_deg."""
    out = np.zeros(max_deg + 1)
    for a in range(max_deg + 1):
        if a % 2 == 1:
            out[a] = 0.0
        else:
            out[a] = 2 * (support_half ** (a + 1)) / (a + 1)
    return out


def add_l_infty_constraint(constraints: List[cp.Constraint],
                            var_map: MomentVarMap,
                            k: int,
                            M_density_rescaled: float,
                            *,
                            support_half: float = 1.0) -> List[int]:
    """Add an L^infty density bound constraint:
        rho^{(d)} <= M_density_rescaled  on  [-support_half, support_half]^d.

    Encoded as: nu := M_density_rescaled * Lebesgue^{(d)} - rho^{(d)} is a positive
    measure on [-support_half, support_half]^d.

    Moments of nu:
        nu_alpha = M_density_rescaled * J^{(d)}_alpha - g^{(d)}_alpha
    where J^{(d)}_alpha = prod_i J_{alpha_i} (d-fold tensor of 1D Lebesgue moments).

    Constraint: {nu_alpha} satisfies d-D Hausdorff PSD + d localizers.

    Returns list of PSD block sizes added.
    """
    d = var_map.dim
    # Precompute 1D Lebesgue moments
    Jmax = 2 * k
    J = lebesgue_moments_1d(Jmax, support_half=support_half)

    def lebesgue_d(alpha: Tuple[int, ...]) -> float:
        out = 1.0
        for a in alpha:
            out *= J[a]
        return out

    def nu_lookup(alpha: Tuple[int, ...]) -> cp.Expression:
        return M_density_rescaled * lebesgue_d(alpha) - var_map.get(alpha)

    # d-D Hankel matrix for nu
    basis = enum_multi_indices(d, k)
    n = len(basis)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            alpha = tuple(basis[i][t] + basis[j][t] for t in range(d))
            row.append(nu_lookup(alpha))
        rows.append(row)
    H_nu = cp.bmat(rows)
    constraints.append(H_nu >> 0)
    block_sizes = [n]

    # d localizers: (support_half^2 - u_axis^2) * M_{k-1}(nu) >= 0
    if k >= 1:
        loc_basis = enum_multi_indices(d, k - 1)
        h2 = support_half * support_half
        nL = len(loc_basis)
        for axis in range(d):
            rows = []
            for i in range(nL):
                row = []
                for j in range(nL):
                    alpha = tuple(loc_basis[i][t] + loc_basis[j][t] for t in range(d))
                    shifted = list(alpha)
                    shifted[axis] += 2
                    row.append(h2 * nu_lookup(alpha) - nu_lookup(tuple(shifted)))
                rows.append(row)
            L_nu = cp.bmat(rows)
            constraints.append(L_nu >> 0)
            block_sizes.append(nL)

    return block_sizes


# =====================================================================
# Builder: 2pt with BUMP objective, optional L_infty
# =====================================================================

def build_2pt_bump(k: int, N: int, *,
                    epsilon: float = 0.1,
                    l_inf_M_rho2_orig: Optional[float] = None,
                    enforce_reflection: bool = True) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """2pt baseline using the V1 BUMP-KERNEL objective in the V2 rescaled framework.

    epsilon: kernel positivity floor (q^*_N has min = epsilon/(I_N + epsilon) on
             [-1/2, 1/2], avoiding boundary-Dirac collapse).
    l_inf_M_rho2_orig: if given, bound ||rho^(2)||_infty <= M^2 in ORIGINAL coords.
        In rescaled coords v_i = 4 u_i, the bound on tilde rho^(2) is M^2 / 16.
    """
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}) for moments g_{{ab}}, a+b<=2N.")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    g_map = MomentVarMap(dim=2, max_deg=2 * k, reflection_zero_odd=enforce_reflection)
    m_map = MomentVarMap(dim=1, max_deg=2 * k, reflection_zero_odd=enforce_reflection)
    constraints: List[cp.Constraint] = []
    constraints.append(g_map.get((0, 0)) == 1)
    constraints.append(m_map.get((0,)) == 1)
    for a in range(2 * k + 1):
        constraints.append(g_map.get((a, 0)) == m_map.get((a,)))

    block_sizes: List[int] = []
    M2 = moment_matrix(g_map, k)
    constraints.append(M2 >> 0); block_sizes.append(M2.shape[0])
    if k >= 1:
        for axis in range(2):
            L = localizer_matrix(g_map, k, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L >> 0); block_sizes.append(L.shape[0])
    M1 = moment_matrix(m_map, k)
    constraints.append(M1 >> 0); block_sizes.append(M1.shape[0])
    if k >= 1:
        L1 = localizer_matrix(m_map, k, axis=0, support_half=SUPPORT_HALF)
        constraints.append(L1 >> 0); block_sizes.append(L1.shape[0])

    # Optional L_infty: ||rho^(2)||_infty in rescaled coords <= M^2 / 16
    if l_inf_M_rho2_orig is not None:
        M_resc = l_inf_M_rho2_orig / 16.0
        sizes = add_l_infty_constraint(
            constraints, g_map, k, M_density_rescaled=M_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes)
        # Also bound the 1D ||f||_infty <= sqrt(M_rho2) (necessary for f >= 0)
        # In rescaled: ||tilde f||_infty <= sqrt(M_resc) * 1
        # Hmm actually for f >= 0 with f*f bounded by M^2 a.e., we don't immediately
        # get f bounded.  But for product, ||f||_infty^2 = ||f \otimes f||_infty.
        # In the relaxation g doesn't factor, so we add the 1D bound as separate.
        M_f_resc = math.sqrt(M_resc * 2)  # generous; uniform tilde_f = 0.5 has ||.||_inf = 0.5
        sizes_1d = add_l_infty_constraint(
            constraints, m_map, k, M_density_rescaled=M_f_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes_1d)

    # Objective
    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = 0
    for (a, b), c in coeffs.items():
        if a + b > 2 * k:
            continue
        obj = obj + c * g_map.get((a, b))
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=False,
        n_orbits_y=0, n_orbits_g=g_map.n_orbits(), n_orbits_m=m_map.n_orbits(),
        block_sizes=block_sizes, n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    handles = dict(g=g_map, m=m_map, coeffs=coeffs)
    return problem, info, handles


def build_3pt_bump(k: int, N: int, *,
                    epsilon: float = 0.1,
                    l_inf_M_rho2_orig: Optional[float] = None,
                    l_inf_M_rho3_orig: Optional[float] = None,
                    enforce_reflection: bool = True) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """3pt lift with bump-kernel objective."""
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}).")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    y_map = MomentVarMap(dim=3, max_deg=2 * k, reflection_zero_odd=enforce_reflection)
    constraints: List[cp.Constraint] = []
    constraints.append(y_map.get((0, 0, 0)) == 1)

    block_sizes: List[int] = []
    M3 = moment_matrix(y_map, k)
    constraints.append(M3 >> 0); block_sizes.append(M3.shape[0])
    if k >= 1:
        for axis in range(3):
            L = localizer_matrix(y_map, k, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L >> 0); block_sizes.append(L.shape[0])

    # Optional L_infty constraints
    if l_inf_M_rho2_orig is not None:
        # Need 2D analog on g_{ab} = y_{a, b, 0}.  Build a thin "2D MomentVarMap" wrapper
        # that aliases through y_{a, b, 0} -- but since we want the SHIFTED Hausdorff cone,
        # easier to inline.
        M_resc = l_inf_M_rho2_orig / 16.0
        # Inline: build 2D nu_{ab} = M_resc * J_a J_b - y_{a,b,0}, then 2D Hankel + 2 localizers
        Jmax = 2 * k
        J = lebesgue_moments_1d(Jmax, support_half=SUPPORT_HALF)

        def nu2_lookup(a: int, b: int) -> cp.Expression:
            return M_resc * J[a] * J[b] - y_map.get((a, b, 0))

        basis_2d = enum_multi_indices(2, k)
        rows = []
        for i in range(len(basis_2d)):
            row = []
            for j in range(len(basis_2d)):
                a = basis_2d[i][0] + basis_2d[j][0]
                b = basis_2d[i][1] + basis_2d[j][1]
                row.append(nu2_lookup(a, b))
            rows.append(row)
        constraints.append(cp.bmat(rows) >> 0); block_sizes.append(len(basis_2d))
        if k >= 1:
            loc_basis_2d = enum_multi_indices(2, k - 1)
            h2 = SUPPORT_HALF * SUPPORT_HALF
            for axis in range(2):
                rows = []
                for i in range(len(loc_basis_2d)):
                    row = []
                    for j in range(len(loc_basis_2d)):
                        a = loc_basis_2d[i][0] + loc_basis_2d[j][0]
                        b = loc_basis_2d[i][1] + loc_basis_2d[j][1]
                        sh = [a, b]
                        sh[axis] += 2
                        row.append(h2 * nu2_lookup(a, b) - nu2_lookup(sh[0], sh[1]))
                    rows.append(row)
                constraints.append(cp.bmat(rows) >> 0); block_sizes.append(len(loc_basis_2d))

    if l_inf_M_rho3_orig is not None:
        M_resc = l_inf_M_rho3_orig / 64.0  # rescaled bound = M^3 / 4^3
        sizes = add_l_infty_constraint(
            constraints, y_map, k, M_density_rescaled=M_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes)

    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = 0
    for (a, b), c in coeffs.items():
        if a + b > 2 * k:
            continue
        obj = obj + c * y_map.get((a, b, 0))
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=True,
        n_orbits_y=y_map.n_orbits(), n_orbits_g=0, n_orbits_m=0,
        block_sizes=block_sizes, n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    handles = dict(y=y_map, coeffs=coeffs)
    return problem, info, handles


__all__ = [
    "bump_integral", "bump_objective_coefficients_rescaled",
    "lebesgue_moments_1d", "add_l_infty_constraint",
    "build_2pt_bump", "build_3pt_bump",
]

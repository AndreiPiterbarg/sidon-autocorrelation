"""4-point Lasserre lift with L^2 bound on rho^(2).

Goal: encode ||rho^(2)||_2^2 <= K via the 4-point moments z_{abcd}, where
rho^(2) = f \\otimes f and z represents an exchangeable 4-point measure with
2-marginal coupled to g_{ab}.

The L^2 on rho^(2) is encoded as a bump-test of the 2D autocorrelation
(rho^(2) * \\bar rho^(2)) at the origin (0, 0), which equals ||rho^(2)||_2^2
for absolutely continuous rho^(2).

In 4-point moments:
    rho^(2) * \\bar rho^(2)(t_1, t_2) = int rho^(2)(u_1, u_2) rho^(2)(u_1-t_1, u_2-t_2) du.
    int t_1^a t_2^b (rho^(2) * \\bar rho^(2)) dt = sum_{j_1, j_2} C(a, j_1) C(b, j_2) (-1)^{j_1+j_2} z_{a-j_1, b-j_2, j_1, j_2}.

For a separable bump  q(t_1, t_2) = q_1(t_1) q_2(t_2)  on [-2, 2]^2:
    int q (rho^(2) * \\bar rho^(2)) ~ ||rho^(2)||_2^2  for sharp q.

Implementation:
- 4-point moment block z_{abcd}, a+b+c+d <= 2k.
- 4D PSD on z + 4 box localizers.
- Marginalization: z_{ab00} = g_{ab} (or symmetric variants).
- L^2 bump test (linear constraint on z).

This module provides the 4-point lift in the V2 framework with bump-kernel
objective (similar to V3's threepoint_alternatives) plus the L^2 constraint.
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
)
from lasserre.threepoint_l2 import (
    bump_polynomial_coeffs, autocorr_moments_from_g,
)


def s4_canonical(idx: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Canonical S_4 representative: sort decreasing.
    Note: actual exchangeability may be smaller than S_4 (e.g., (1,2)<->(3,4) only).
    For our use (rho^(2) acting like a single object with 2 internal coords):
    the natural symmetry is: swap the 2 ordered pairs (1,2) and (3,4),
    plus within each pair, exchangeability of (1,2) and (3,4) separately.
    For simplicity, use full S_4 here; larger orbit reduction at variable level.
    """
    return tuple(sorted(idx, reverse=True))


class FourPointMap(MomentVarMap):
    """4-point moment map z_{abcd} on [-1/4, 1/4]^4, with S_4 x Z/2 orbit reduction."""
    def __init__(self, max_deg: int, *, reflection_zero_odd: bool = True):
        self.dim = 4
        self.max_deg = max_deg
        self.reflection_zero_odd = reflection_zero_odd
        self._var_by_canon = {}
        for alpha in enum_multi_indices(4, max_deg):
            if reflection_zero_odd and (sum(alpha) % 2 == 1):
                continue
            canon = s4_canonical(alpha)
            if canon not in self._var_by_canon:
                self._var_by_canon[canon] = cp.Variable(name=f"z{canon}")

    def _canonical(self, alpha):
        return s4_canonical(alpha)

    def get(self, alpha):
        if len(alpha) != 4:
            raise ValueError(f"alpha must be 4-tuple, got {alpha}")
        if any(a < 0 for a in alpha):
            raise ValueError(f"negative components in {alpha}")
        if sum(alpha) > self.max_deg:
            raise ValueError(f"alpha {alpha} exceeds max_deg {self.max_deg}")
        if self.reflection_zero_odd and (sum(alpha) % 2 == 1):
            return cp.Constant(0.0)
        canon = s4_canonical(alpha)
        return self._var_by_canon[canon]


def add_l2_rho2_constraint(constraints: List, z_map: FourPointMap, k: int,
                            K_l2_rescaled: float, N_bump_per_axis: int = 2,
                            R2_support_half: float = 2.0) -> int:
    """Add the L^2 bound on rho^(2):
        ||tilde_rho^(2)||_2^2 <= K_l2_rescaled.

    Encoded via bump test of the 2D autocorrelation (rho^(2) * bar(rho^(2))) at origin.

    The 2D autocorrelation moments rho_{ab} = sum_{j1,j2} C(a, j1) C(b, j2) (-1)^{j1+j2} z_{a-j1, b-j2, j1, j2}.
    The 2D bump q(t_1, t_2) = q_1(t_1) q_1(t_2) (separable, peaked at (0,0)).
    int q rho^(2)*bar rho^(2) dt = sum_{a, b} q^coef_a * q^coef_b * rho_{a, b}.

    Returns the maximum degree used.
    """
    h = R2_support_half
    q_coef = bump_polynomial_coeffs(N_bump_per_axis, support_half=h)
    max_a = 2 * N_bump_per_axis
    max_b = 2 * N_bump_per_axis
    if max_a + max_b > 2 * k:
        max_a = 2 * k // 2
        max_b = 2 * k // 2
    # Compute rho_{ab} = int t_1^a t_2^b (rho^(2) * bar rho^(2)) dt
    # = sum_{j1, j2} C(a, j1) C(b, j2) (-1)^{j1+j2} z_{a-j1, b-j2, j1, j2}
    terms = []
    for a in range(max_a + 1):
        if a not in q_coef:
            continue
        for b in range(max_b + 1):
            if b not in q_coef:
                continue
            qa = q_coef[a]
            qb = q_coef[b]
            if abs(qa * qb) < 1e-15:
                continue
            # Sum z_{a-j1, b-j2, j1, j2} weighted
            inner_terms = []
            for j1 in range(a + 1):
                for j2 in range(b + 1):
                    coef = math.comb(a, j1) * math.comb(b, j2) * ((-1) ** (j1 + j2))
                    if abs(coef) < 1e-15:
                        continue
                    inner_terms.append(coef * z_map.get((a - j1, b - j2, j1, j2)))
            if inner_terms:
                terms.append(qa * qb * cp.sum(cp.hstack(inner_terms)))
    if terms:
        expr = cp.sum(cp.hstack(terms))
        constraints.append(expr <= K_l2_rescaled)
    return max(max_a, max_b)


def build_4pt_l2(k: int, N: int, *, epsilon: float = 0.1,
                  K_l2_rho2: Optional[float] = None,
                  N_bump_l2: int = 2) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """4-point Lasserre with L^2 bound on rho^(2).

    Variables: z_{abcd} for a+b+c+d <= 2k (with S_4 x Z/2 orbit dedup).
    g_{ab} = z_{ab00} (2-marginal).
    m_a = z_{a000} (1-marginal).
    """
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}).")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    z_map = FourPointMap(max_deg=2 * k)
    constraints = []
    constraints.append(z_map.get((0, 0, 0, 0)) == 1)

    block_sizes = []
    # 4D PSD on z
    M4 = moment_matrix(z_map, k)
    constraints.append(M4 >> 0); block_sizes.append(M4.shape[0])
    if k >= 1:
        # 4 box localizers
        for axis in range(4):
            L = localizer_matrix(z_map, k, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L >> 0); block_sizes.append(L.shape[0])

    # L^2 on rho^(2) constraint
    if K_l2_rho2 is not None:
        # In rescaled coords, ||tilde_rho^(2)||_2^2 = K_l2_rho2 / 16  (since rho^(2)_orig = 16 * tilde)
        # Actually: rho^(2)_orig(u_1, u_2) = ?. With v_i = 4 u_i:
        #   tilde_rho^(2)(v_1, v_2) = (1/16) rho^(2)(v_1/4, v_2/4)
        #   ||tilde_rho^(2)||_2^2 = int tilde_rho^(2)^2 = (1/256) int rho^(2)(v/4)^2 dv = (1/256) * 16 ||rho^(2)||_2^2 = (1/16) ||rho^(2)||_2^2
        K_resc = K_l2_rho2 / 16.0
        add_l2_rho2_constraint(constraints, z_map, k, K_l2_rescaled=K_resc,
                                N_bump_per_axis=N_bump_l2, R2_support_half=2.0)

    # Objective: bump-kernel integral on rho^(2) (= g_{ab} = z_{ab00})
    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = sum(c * z_map.get((a, b, 0, 0)) for (a, b), c in coeffs.items() if a + b <= 2 * k)
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=True,  # using "with_3pt" loosely as "with lift"
        n_orbits_y=len(z_map._var_by_canon), n_orbits_g=0, n_orbits_m=0,
        block_sizes=block_sizes, n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    return problem, info, {"z": z_map}

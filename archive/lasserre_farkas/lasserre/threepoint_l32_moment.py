"""L^{3/2}-constrained moment-Lasserre SDP for the Sidon constant C_{1a}.

================================================================================
MATHEMATICAL FORMULATION
================================================================================

We extend V3 (`threepoint_full.py` + `threepoint_alternatives.py`) by replacing
the L^infty density bound with an L^{3/2} bound.

Goal: lower-bound

    C_{1a}^{(B)} := inf  sup_{|t| <= 1/2} (f * f)(t)
                  s.t.  f >= 0,  supp f subset [-1/4, 1/4],  int f = 1,
                        ||f||_{3/2} <= B

The L^{3/2} constraint is non-polynomial in moments of f; we encode it via a
multi-density auxiliary g and the polynomial inequality g^2 >= f^3.

Key observation:   ||f||_{3/2}^{3/2} = int f^{3/2}.
If g >= 0 satisfies g^2 >= f^3 pointwise, then g >= f^{3/2} (both nonnegative),
so int g >= int f^{3/2} = ||f||_{3/2}^{3/2}.  Thus

    int g <= B^{3/2}    ==>    ||f||_{3/2} <= B.

================================================================================
JOINT MOMENT-LASSERRE ENCODING
================================================================================

Variables:
  m_a := int x^a f(x) dx,  a = 0, ..., 2k          (1D f-moments)
  z_a := int x^a g(x) dx,  a = 0, ..., 2k          (1D g-moments)
  mu_{a, j, k} := int x^a f(x)^j g(x)^k dx         (JOINT moments)
        for (a, j, k) needed to populate the Putinar matrices below.

Linkage (linear equalities):
  mu_{a, 0, 0} = J_a := int_{-1/4}^{1/4} x^a dx     (Lebesgue moments, constants)
  mu_{a, 1, 0} = m_a
  mu_{a, 0, 1} = z_a

Constraints:

  (C1) f >= 0 supported on [-1/4, 1/4]:  Hausdorff PSD on m  (already in V3).
  (C2) g >= 0 supported on [-1/4, 1/4]:  Hausdorff PSD on z  (NEW).
       Encoded as 1D moment matrix M_k(z) PSD + 1D box localizer
       ((1/4)^2 - x^2) M_{k-1}(z) PSD.
  (C3) Normalization:  m_0 = 1.
  (C4) L^{3/2} bound:  z_0 <= B^{3/2}   (linear).
  (C5) JOINT MOMENT MATRIX PSD (the consistency constraint between f, g, mu):
       Pick a basis B_J = {x^a f^j g^k}_{(a,j,k) in I_J}.  The Hankel matrix
       indexed by (alpha, alpha') in I_J^2 with entries
            H[alpha, alpha'] = mu_{a + a', j + j', k + k'}
       is PSD.
  (C6) PUTINAR LOCALIZER for  g^2 - f^3 >= 0:  1D Hausdorff cone on
            nu_a := mu_{a, 0, 2} - mu_{a, 3, 0},  a = 0, ..., 2 K_L
       i.e. M_{K_L}(nu) PSD + box localizer ((1/4)^2 - x^2) M_{K_L-1}(nu) PSD.

  (C7) [Optional] Same as (C6) but with f and g squared in the basis cross-products
       to certify e.g. g >= 0 jointly (already covered by (C2)).

We also add Putinar f-Hausdorff (mu_{., 1, .} PSD on a 1D basis) and g-Hausdorff
(mu_{., ., 1} PSD on 1D basis) conditions: these are subsumed by (C5) in the
joint matrix where we include rows/cols indexed by elements with j > 0 and k > 0.

================================================================================
SCOPE / HONESTY
================================================================================

This is the SIMPLEST sound encoding I could find.  In particular:

  * The joint moment matrix in (C5) at HIGH order would be expensive.  We
    truncate to a small basis I_J, which preserves SOUNDNESS (linear
    relaxation of the PSD cone) but may not be tight.
  * The Putinar localizer (C6) is at level K_L, similarly truncated.
  * If the relaxation is too loose, the SDP will produce unrealistically
    small lambda (collapse to ~ 1, like V2).
  * If the joint PSD is the only thing linking m and z, and the link is weak,
    we may see lambda essentially independent of B.

We report HONESTLY whether the encoding produces meaningful values.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
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


# =====================================================================
# Joint moment variable container
# =====================================================================

class JointMomentMap:
    """Container for joint moments mu_{a, j, k} = int x^a f^j g^k dx, where
    we work in RESCALED coords v in [-1, 1] (so 'x' here is the rescaled v;
    the support_half is 1).

    Reflection symmetry:  f(-x) = f(x), g(-x) = g(x)  imply  mu_{a, j, k} = 0
    for a odd (since x^a integrand has odd symmetry, even f^j g^k).

    Ranges:
        a in [0, max_a]
        j in [0, max_j]
        k in [0, max_k]
        sum a + j + k restricted to <= max_total only at variable-creation time;
        a CVXPY variable is allocated for each requested (a, j, k) on demand.
    """
    def __init__(self, max_a: int, max_j: int, max_k: int,
                 *, reflection_zero_odd_a: bool = True):
        self.max_a = max_a
        self.max_j = max_j
        self.max_k = max_k
        self.reflection_zero_odd_a = reflection_zero_odd_a
        self._var: Dict[Tuple[int, int, int], cp.Expression] = {}

    def get(self, a: int, j: int, k: int) -> cp.Expression:
        if a < 0 or j < 0 or k < 0:
            raise ValueError(f"Negative joint index ({a},{j},{k})")
        if a > self.max_a or j > self.max_j or k > self.max_k:
            raise ValueError(
                f"Joint index ({a},{j},{k}) exceeds limits "
                f"(max_a={self.max_a}, max_j={self.max_j}, max_k={self.max_k})")
        if self.reflection_zero_odd_a and (a % 2 == 1):
            return cp.Constant(0.0)
        key = (a, j, k)
        if key not in self._var:
            self._var[key] = cp.Variable(name=f"mu_{a}_{j}_{k}")
        return self._var[key]

    def n_orbits(self) -> int:
        return len(self._var)

    def free_variables(self) -> List[cp.Variable]:
        return list(self._var.values())


# =====================================================================
# Add L^{3/2} constraint via multi-density Putinar
# =====================================================================

def add_l32_constraint(
    constraints: List[cp.Constraint],
    *,
    f_moments: Dict[int, cp.Expression],   # mapping a -> m_a
    k: int,                                 # max f-moment degree available is 2k
    B: float,                               # L^{3/2} bound: ||f||_{3/2} <= B
    K_L: Optional[int] = None,              # Putinar localizer order for g^2 >= f^3
    K_joint: Optional[int] = None,          # joint moment matrix order in 'a'
    support_half: float = 1.0,              # rescaled support; default 1
    mu_scale: float = 0.25,                 # scale: working coord v = u / mu_scale
) -> Tuple[Dict, List[int]]:
    """Add an L^{3/2} density-norm bound using the multi-density Putinar encoding.

    Adds:
      - Auxiliary g-moment variables z_a, a = 0..2k
      - Joint moments mu_{a, j, k}
      - Hausdorff PSD on z (g >= 0 on [-mu_scale, mu_scale])
      - Joint moment matrix PSD (consistency)
      - Putinar localizer PSD on g^2 - f^3
      - Linear linking equations: mu_{a,0,0} = J_a, mu_{a,1,0} = m_a, mu_{a,0,1} = z_a
      - Linear bound: int g dx <= B^{3/2}, i.e., (mu_scale) * z_0 <= B^{3/2}
        (since dx_orig = mu_scale * dv where v is the rescaled coord)

    All 'x' moments below are in RESCALED coords v in [-1, 1].  To recover
    the L^{3/2} bound in ORIGINAL coords, recall:
        g_orig(u) du = g_resc(v) (mu_scale) dv  if g is treated as a density.
    Equivalent: int_orig g(u) du = mu_scale * (rescaled int).
    Set the linkage so that 'g' here represents the density in the same coord
    system as 'f' (rescaled).  Then ||f||_{3/2}^{3/2} (orig) = mu_scale * ||~f||
    relations follow.

    For simplicity, we work entirely in rescaled coords from here.  The B value
    passed in is interpreted in ORIGINAL coords (||f_orig||_{3/2} <= B), so we
    convert: in rescaled coords, ||~f||_{3/2}^{3/2} = (1/mu_scale^{1/2}) *
    ||f_orig||_{3/2}^{3/2} (since ~f(v) = mu_scale * f(mu_scale v) for prob-norm).

    Actually let's be precise.  Working def:  f_orig: [-1/4, 1/4] -> R_>=0,
    int f_orig = 1.   Rescaled  ~f(v) := mu_scale * f_orig(mu_scale * v),
    so int ~f dv = int mu_scale * f_orig(mu_scale v) dv = int f_orig(u) du = 1.
    Then ||~f||_{3/2}^{3/2} = int ~f^{3/2} dv = int (mu_scale)^{3/2} f_orig^{3/2}(u) (du / mu_scale)
                            = mu_scale^{1/2} ||f_orig||_{3/2}^{3/2}.
    So  ||f_orig||_{3/2} = ||~f||_{3/2} / mu_scale^{1/3}.
    The bound  ||f_orig||_{3/2} <= B  becomes  ||~f||_{3/2} <= B * mu_scale^{1/3}.
    We bound int ~g dv in rescaled coords: since ~g >= ~f^{3/2}, we get
         int ~g dv >= int ~f^{3/2} dv = ||~f||_{3/2}^{3/2} <= B^{3/2} * mu_scale^{1/2}.
    Therefore the constraint  z_0 = int ~g dv <= B_resc := B^{3/2} * mu_scale^{1/2}.

    Returns (handles, block_sizes) where handles include the JointMomentMap and
    z-moment map; block_sizes lists the PSD block sizes added.
    """
    if K_L is None:
        K_L = max(1, k - 2)  # localizer order; needs joint moments up to a = 2*K_L
    if K_joint is None:
        K_joint = max(1, k - 2)  # joint matrix order in 'a'

    # Joint moment max indices:
    # - For C5 (joint matrix): basis (a, j, k) with a in 0..K_joint, j in 0..2, k in 0..1.
    #   Cross product entry has a' in 0..2K_joint, j' in 0..4, k' in 0..2.
    # - For C6 (Putinar g^2 - f^3): need a in 0..2*K_L for j=3,k=0 and j=0,k=2 entries.
    max_a_needed = max(2 * K_joint, 2 * K_L + 2)  # +2 for box localizer
    max_j_needed = 4
    max_k_needed = 2

    j_map = JointMomentMap(max_a_needed, max_j_needed, max_k_needed)

    # 1D g-moment map (z) on max_deg = 2k
    z_map = MomentVarMap(dim=1, max_deg=2 * k, reflection_zero_odd=True,
                         enforce_normalization=False)

    block_sizes: List[int] = []

    # ---- (C2) g >= 0: Hausdorff PSD on z ----
    Mz = moment_matrix(z_map, k)
    constraints.append(Mz >> 0)
    block_sizes.append(Mz.shape[0])
    if k >= 1:
        Lz = localizer_matrix(z_map, k, axis=0, support_half=support_half)
        constraints.append(Lz >> 0)
        block_sizes.append(Lz.shape[0])

    # ---- (C4) L^{3/2} bound: z_0 <= B^{3/2} * mu_scale^{1/2} ----
    B_resc = (B ** 1.5) * (mu_scale ** 0.5)
    constraints.append(z_map.get((0,)) <= B_resc)

    # ---- Linkage:  mu_{a, 0, 0} = J_a  (Lebesgue moments in rescaled coords)
    # In rescaled coord v in [-support_half, support_half]:  J_a = 2 * support_half^{a+1}/(a+1) for a even, 0 else.
    Jvals = lebesgue_moments_1d(max_a_needed, support_half=support_half)

    # mu_{a, 0, 0} is just J_a.  Setting them as linear equalities is wasteful;
    # instead we treat mu_{a,0,0} as constants by adding equality constraints
    # only for the indices that actually get queried.

    # Linkage:  mu_{a, 1, 0} = m_a   and   mu_{a, 0, 1} = z_a
    # Iterate over all even a we will need.
    for a in range(max_a_needed + 1):
        if a % 2 == 1:
            continue  # both sides are 0 by reflection
        # mu_{a, 0, 0} = J_a
        # The joint variable mu_{a, 0, 0} is a CVXPY var (created on first .get).
        # We set it equal to the constant J_a.
        constraints.append(j_map.get(a, 0, 0) == Jvals[a])
        # mu_{a, 1, 0} = m_a   (only if m_a is provided)
        if a in f_moments:
            constraints.append(j_map.get(a, 1, 0) == f_moments[a])
        # mu_{a, 0, 1} = z_a
        if a <= 2 * k:
            constraints.append(j_map.get(a, 0, 1) == z_map.get((a,)))

    # ---- (C5) JOINT MOMENT MATRIX PSD ----
    # Basis: B_J = {(a, j, k) : 0 <= a <= K_joint, j in {0,1,2}, k in {0,1}}.
    # Reflection: a odd entries are zero, so we keep them in the basis but
    # entries involving them combine to zero (handled by JointMomentMap.get).
    # Filter: only need a even ... but for the matrix to be square, keep all.
    basis_J: List[Tuple[int, int, int]] = []
    for a in range(K_joint + 1):
        for j in range(3):  # j in {0, 1, 2}
            for kk in range(2):  # k in {0, 1}
                basis_J.append((a, j, kk))
    nJ = len(basis_J)
    rows: List[List[cp.Expression]] = []
    for i in range(nJ):
        row: List[cp.Expression] = []
        ai, ji, ki = basis_J[i]
        for jj in range(nJ):
            ap, jp, kp = basis_J[jj]
            row.append(j_map.get(ai + ap, ji + jp, ki + kp))
        rows.append(row)
    H_J = cp.bmat(rows)
    constraints.append(H_J >> 0)
    block_sizes.append(nJ)

    # ---- (C6) Putinar localizer:  g^2 - f^3 >= 0 ----
    # nu_a := mu_{a, 0, 2} - mu_{a, 3, 0}.   1D moments on x in [-support_half, support_half].
    # Hankel PSD on basis {1, x, ..., x^{K_L}}, i.e., M_{K_L}(nu) PSD with size K_L+1.
    nu_max = 2 * K_L + 2  # need a up to 2*K_L for Hankel and 2*K_L+2 for box localizer
    if nu_max > max_a_needed:
        raise RuntimeError(f"Internal: K_L={K_L} too large for max_a_needed={max_a_needed}")

    def nu(a: int) -> cp.Expression:
        if a > max_a_needed:
            return cp.Constant(0.0)
        return j_map.get(a, 0, 2) - j_map.get(a, 3, 0)

    # Hankel
    sizeL = K_L + 1
    rows = [[nu(i + jj) for jj in range(sizeL)] for i in range(sizeL)]
    H_nu = cp.bmat(rows)
    constraints.append(H_nu >> 0)
    block_sizes.append(sizeL)
    # Box localizer ( support_half^2 - x^2 ) * M_{K_L-1}(nu)
    if K_L >= 1:
        sizeLloc = K_L
        h2 = support_half * support_half
        rows = []
        for i in range(sizeLloc):
            row = []
            for jj in range(sizeLloc):
                row.append(h2 * nu(i + jj) - nu(i + jj + 2))
            rows.append(row)
        L_nu = cp.bmat(rows)
        constraints.append(L_nu >> 0)
        block_sizes.append(sizeLloc)

    handles = dict(j_map=j_map, z_map=z_map, K_L=K_L, K_joint=K_joint,
                   B=B, B_resc=B_resc, Jvals=Jvals)
    return handles, block_sizes


# =====================================================================
# 2pt builder with L^{3/2}
# =====================================================================

def build_2pt_l32(k: int, N: int, *,
                  B: float,
                  epsilon: float = 0.1,
                  l_inf_M_rho2_orig: Optional[float] = None,
                  K_L: Optional[int] = None,
                  K_joint: Optional[int] = None,
                  enforce_reflection: bool = True
                  ) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """2-point baseline with L^{3/2} constraint.

    B: L^{3/2} bound on f (in ORIGINAL coords [-1/4, 1/4]).
    epsilon: bump-kernel positivity floor.
    l_inf_M_rho2_orig: optionally ALSO impose L^infty bound on rho^(2) = f * f.
    K_L, K_joint: orders for the L^{3/2} encoding (defaults to max(1, k-2)).
    """
    if 2 * k < 2 * N:
        raise ValueError(f"Need 2k >= 2N (have k={k}, N={N}).")
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

    # Optional L_infty (additive)
    if l_inf_M_rho2_orig is not None:
        M_resc = l_inf_M_rho2_orig / 16.0
        sizes = add_l_infty_constraint(
            constraints, g_map, k, M_density_rescaled=M_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes)
        M_f_resc = math.sqrt(M_resc * 2)
        sizes_1d = add_l_infty_constraint(
            constraints, m_map, k, M_density_rescaled=M_f_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes_1d)

    # L^{3/2} via multi-density Putinar
    f_moments_dict = {}
    for a in range(2 * k + 1):
        if a % 2 == 0:  # only even, by reflection
            f_moments_dict[a] = m_map.get((a,))
    l32_handles, l32_block_sizes = add_l32_constraint(
        constraints,
        f_moments=f_moments_dict,
        k=k, B=B, K_L=K_L, K_joint=K_joint,
        support_half=SUPPORT_HALF, mu_scale=MU_SCALE,
    )
    block_sizes.extend(l32_block_sizes)

    # Bump-kernel objective in rescaled coords.
    coeffs = bump_objective_coefficients_rescaled(N, epsilon, mu_scale=MU_SCALE)
    obj = 0
    for (a, b), c in coeffs.items():
        if a + b > 2 * k:
            continue
        obj = obj + c * g_map.get((a, b))
    problem = cp.Problem(cp.Minimize(obj), constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=False,
        n_orbits_y=0,
        n_orbits_g=g_map.n_orbits(),
        n_orbits_m=m_map.n_orbits(),
        block_sizes=block_sizes,
        n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    handles = dict(g=g_map, m=m_map, coeffs=coeffs, l32=l32_handles)
    return problem, info, handles


# =====================================================================
# 3pt builder with L^{3/2}
# =====================================================================

def build_3pt_l32(k: int, N: int, *,
                  B: float,
                  epsilon: float = 0.1,
                  l_inf_M_rho2_orig: Optional[float] = None,
                  l_inf_M_rho3_orig: Optional[float] = None,
                  K_L: Optional[int] = None,
                  K_joint: Optional[int] = None,
                  enforce_reflection: bool = True
                  ) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """3-point lift with L^{3/2} constraint."""
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

    # Optional L_infty constraints (inlined; see threepoint_alternatives.build_3pt_bump)
    if l_inf_M_rho2_orig is not None:
        M_resc = l_inf_M_rho2_orig / 16.0
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
        M_resc = l_inf_M_rho3_orig / 64.0
        sizes = add_l_infty_constraint(
            constraints, y_map, k, M_density_rescaled=M_resc,
            support_half=SUPPORT_HALF)
        block_sizes.extend(sizes)

    # L^{3/2}: link via m_a = y_{a, 0, 0}
    f_moments_dict = {}
    for a in range(2 * k + 1):
        if a % 2 == 0:
            f_moments_dict[a] = y_map.get((a, 0, 0))
    l32_handles, l32_block_sizes = add_l32_constraint(
        constraints,
        f_moments=f_moments_dict,
        k=k, B=B, K_L=K_L, K_joint=K_joint,
        support_half=SUPPORT_HALF, mu_scale=MU_SCALE,
    )
    block_sizes.extend(l32_block_sizes)

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
    handles = dict(y=y_map, coeffs=coeffs, l32=l32_handles)
    return problem, info, handles


__all__ = [
    "JointMomentMap",
    "add_l32_constraint",
    "build_2pt_l32",
    "build_3pt_l32",
]

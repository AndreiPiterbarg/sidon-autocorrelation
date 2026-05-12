"""Continuous 2-point vs 3-point Lasserre relaxation for the Sidon constant C_{1a}.

Implements the formulation from delsarte_dual/sdp_hierarchy_design.md §3:
inf over positive measures mu on [-1/4, 1/4] with int mu = 1 of
    L_N(mu) := int q_N(x + y) d(mu otimes mu)(x, y)
where q_N(t) = (1 - 4 t^2)^N / I_N is a centered polynomial bump on
[-1/2, 1/2], int q_N = 1, q_N >= 0.  Hence L_N(mu) <= ||f * f||_infty
for any feasible f, so the optimal value is a rigorous lower bound on
C_{1a} = inf_f ||f*f||_infty.

The 2-point relaxation has variables m_a (1D moments of mu) and g_{ab}
(2D moments of mu otimes mu) with:
    g_{a0} = m_a, g_{ab} = g_{ba},
    1D moment matrix on m PSD + Hausdorff localizer (1/16 - x^2),
    2D moment matrix on g PSD + 2 box localizers.
The 3-point lift adds y_{abc} (3D moments) with:
    y_{ab0} = g_{ab} (so y_{a00} = m_a),
    y_{abc} symmetric under S_3,
    3D moment matrix on y PSD + 3 box localizers.

The objective is
    L_N = sum_{a+b <= 2N, a+b even} c_{ab} g_{ab},
    c_{ab} = (1/I_N) * C(N, (a+b)/2) * (-4)^{(a+b)/2} * C(a+b, a).
Need k >= N so that g_{ab} for a+b <= 2N exists in the relaxation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import cvxpy as cp


def bump_integral(N: int) -> float:
    """I_N = int_{-1/2}^{1/2} (1 - 4 s^2)^N ds = sqrt(pi)/2 * Gamma(N+1)/Gamma(N+3/2)."""
    return 0.5 * math.sqrt(math.pi) * math.gamma(N + 1) / math.gamma(N + 1.5)


def bump_coeffs_2d(N: int, epsilon_shift: float = 0.0) -> Dict[Tuple[int, int], float]:
    """Return c_{ab} such that int q^*_N(x+y) F(x,y) dxdy = sum c_{ab} g_{ab}.

    q^*_N(t) = ((1-4t^2)^N + epsilon_shift) / Z   on [-1/2, 1/2],
        Z = I_N + epsilon_shift,   so int q^*_N = 1.

    With epsilon_shift > 0 the kernel is BOUNDED BELOW by epsilon_shift / Z > 0
    on the support, which prevents the moment relaxation from collapsing to a
    trivial zero via boundary-Dirac pseudo-measures (q_N has zeros at +/-1/2,
    which a Dirac at +/-1/4 in mu would exploit).

    q^*_N(x+y) = (1/Z) [ sum_{j=0}^{N} C(N,j) (-4)^j (x+y)^{2j} + epsilon_shift ].
    """
    I_N = bump_integral(N)
    Z = I_N + epsilon_shift
    out: Dict[Tuple[int, int], float] = {(0, 0): epsilon_shift / Z}
    for j in range(N + 1):
        deg = 2 * j
        prefac = math.comb(N, j) * ((-4.0) ** j) / Z
        for a in range(deg + 1):
            b = deg - a
            out[(a, b)] = out.get((a, b), 0.0) + prefac * math.comb(deg, a)
    return out


def enum_multi_indices(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    """All alpha in N^d with |alpha| <= max_deg."""
    if d == 0:
        return [tuple()]
    out: List[Tuple[int, ...]] = []
    if d == 1:
        return [(a,) for a in range(max_deg + 1)]
    for a in range(max_deg + 1):
        for tail in enum_multi_indices(d - 1, max_deg - a):
            out.append((a,) + tail)
    return out


def basis_for_psd(d: int, k: int) -> List[Tuple[int, ...]]:
    """Monomial basis indices for moment matrix of order k in d variables."""
    return enum_multi_indices(d, k)


@dataclass
class BuildInfo:
    k: int
    N: int
    with_3pt: bool
    n_vars_m: int
    n_vars_g: int
    n_vars_y: int
    block_sizes: List[int] = field(default_factory=list)  # PSD block sizes
    n_eq_constraints: int = 0
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    peak_mem_mb: float = 0.0
    status: str = ""
    objective: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)


def _moment_matrix(var_lookup, basis, d):
    """Build the moment matrix M[a,b] = y_{basis[a] + basis[b]} as a CVXPY expression."""
    n = len(basis)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            idx = tuple(basis[i][t] + basis[j][t] for t in range(d))
            row.append(var_lookup[idx])
        rows.append(row)
    return cp.bmat(rows)


def _localizer_matrix(var_lookup, basis, d, axis, support_half):
    """Build localizing matrix for (support_half^2 - x_{axis}^2) >= 0.

    Entry (i, j) = support_half^2 * y_{a+b} - y_{a+b+2*e_axis}.
    """
    h2 = support_half * support_half
    n = len(basis)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            base = tuple(basis[i][t] + basis[j][t] for t in range(d))
            shifted = list(base)
            shifted[axis] += 2
            shifted_t = tuple(shifted)
            row.append(h2 * var_lookup[base] - var_lookup[shifted_t])
        rows.append(row)
    return cp.bmat(rows)


def _make_var_lookup_1d(k: int) -> Tuple[Dict[Tuple[int, ...], cp.Variable], List[Tuple[int, ...]]]:
    """1D moment variables m_0, ..., m_{2k}.  m_0 fixed to 1 separately."""
    indices = [(a,) for a in range(2 * k + 1)]
    vars_ = {a: cp.Variable() for a in indices}
    return vars_, indices


def _make_var_lookup_2d(k: int) -> Tuple[Dict[Tuple[int, int], cp.Variable], List[Tuple[int, int]]]:
    """2D moment variables g_{ab} for a+b <= 2k, with g_{ab} = g_{ba}."""
    seen: Dict[Tuple[int, int], cp.Variable] = {}
    indices: List[Tuple[int, int]] = []
    for a in range(2 * k + 1):
        for b in range(2 * k + 1 - a):
            indices.append((a, b))
            canon = tuple(sorted((a, b), reverse=True))  # symmetric: g_{ab}=g_{ba}
            if canon not in seen:
                seen[canon] = cp.Variable()
    # full lookup: any (a,b) -> canonical variable
    lookup = {(a, b): seen[tuple(sorted((a, b), reverse=True))] for (a, b) in indices}
    return lookup, indices


def _make_var_lookup_3d(k: int) -> Tuple[Dict[Tuple[int, int, int], cp.Variable], List[Tuple[int, int, int]]]:
    """3D moment variables y_{abc} for a+b+c <= 2k, with full S_3 symmetry."""
    seen: Dict[Tuple[int, int, int], cp.Variable] = {}
    indices: List[Tuple[int, int, int]] = []
    for a in range(2 * k + 1):
        for b in range(2 * k + 1 - a):
            for c in range(2 * k + 1 - a - b):
                indices.append((a, b, c))
                canon = tuple(sorted((a, b, c), reverse=True))
                if canon not in seen:
                    seen[canon] = cp.Variable()
    lookup = {(a, b, c): seen[tuple(sorted((a, b, c), reverse=True))] for (a, b, c) in indices}
    return lookup, indices


def build_2pt_sdp(k: int, N: int, *, enforce_reflection: bool = True, epsilon_shift: float = 0.0) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """Build the 2-point Lasserre relaxation. Returns (problem, info, handles).

    enforce_reflection: if True, restrict to f(x) = f(-x) (reflection symmetry).
    Standard rearrangement reduction in Sidon: WLOG f is symmetric and unimodal,
    so this doesn't lose generality and rules out boundary-Dirac extremizers
    that exploit q_N's zeros at t = +/- 1/2.
    """
    if k < N:
        raise ValueError(f"Need k >= N (k={k}, N={N}) so all g_{{ab}} with a+b<=2N exist.")
    t0 = time.time()
    SUPPORT_HALF = 0.25  # f supported on [-1/4, 1/4]

    m_lookup, _ = _make_var_lookup_1d(k)
    g_lookup, _ = _make_var_lookup_2d(k)

    constraints = []
    # Normalisation
    constraints.append(m_lookup[(0,)] == 1)
    constraints.append(g_lookup[(0, 0)] == 1)
    # Marginal coupling g_{a0} = m_a
    for a in range(2 * k + 1):
        constraints.append(g_lookup[(a, 0)] == m_lookup[(a,)])
    if enforce_reflection:
        # f(x) = f(-x) => m_a = 0 for odd a, g_{ab} = 0 for odd a+b
        for a in range(1, 2 * k + 1, 2):
            constraints.append(m_lookup[(a,)] == 0)
        for (a, b) in list(g_lookup.keys()):
            if (a + b) % 2 == 1:
                constraints.append(g_lookup[(a, b)] == 0)

    # 1D moment matrix M_k(m) PSD
    basis_1d = basis_for_psd(1, k)
    M1 = _moment_matrix(m_lookup, basis_1d, 1)
    constraints.append(M1 >> 0)
    block_sizes = [len(basis_1d)]

    # 1D localiser M_{k-1}((1/16 - x^2) m) PSD
    if k >= 1:
        loc_basis_1d = basis_for_psd(1, k - 1)
        L1 = _localizer_matrix(m_lookup, loc_basis_1d, 1, axis=0, support_half=SUPPORT_HALF)
        constraints.append(L1 >> 0)
        block_sizes.append(len(loc_basis_1d))

    # 2D moment matrix
    basis_2d = basis_for_psd(2, k)
    M2 = _moment_matrix(g_lookup, basis_2d, 2)
    constraints.append(M2 >> 0)
    block_sizes.append(len(basis_2d))

    # 2D localisers (one per axis)
    if k >= 1:
        loc_basis_2d = basis_for_psd(2, k - 1)
        for axis in range(2):
            L2 = _localizer_matrix(g_lookup, loc_basis_2d, 2, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L2 >> 0)
            block_sizes.append(len(loc_basis_2d))

    # Objective: L_N = sum c_{ab} g_{ab}
    coeffs = bump_coeffs_2d(N, epsilon_shift=epsilon_shift)
    obj_expr = 0
    for (a, b), c in coeffs.items():
        if a + b > 2 * k:
            continue
        obj_expr = obj_expr + c * g_lookup[(a, b)]

    problem = cp.Problem(cp.Minimize(obj_expr), constraints)
    info = BuildInfo(
        k=k, N=N, with_3pt=False,
        n_vars_m=2 * k + 1,
        n_vars_g=len([k1 for k1 in g_lookup if g_lookup[k1].id == g_lookup[k1].id]),
        n_vars_y=0,
        block_sizes=block_sizes,
        n_eq_constraints=len([c for c in constraints if c.is_dcp() and not isinstance(c, cp.constraints.PSD)]),
        build_seconds=time.time() - t0,
    )
    handles = dict(m=m_lookup, g=g_lookup, basis_1d=basis_1d, basis_2d=basis_2d, coeffs=coeffs)
    return problem, info, handles


def build_3pt_sdp(k: int, N: int, *, enforce_reflection: bool = True, epsilon_shift: float = 0.0) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """Build the 3-point Lasserre relaxation = 2-point + 3D block + couplings."""
    if k < N:
        raise ValueError(f"Need k >= N (k={k}, N={N}) so all g_{{ab}} with a+b<=2N exist.")
    t0 = time.time()
    SUPPORT_HALF = 0.25

    m_lookup, _ = _make_var_lookup_1d(k)
    g_lookup, _ = _make_var_lookup_2d(k)
    y_lookup, _ = _make_var_lookup_3d(k)

    constraints = []
    constraints.append(m_lookup[(0,)] == 1)
    constraints.append(g_lookup[(0, 0)] == 1)
    constraints.append(y_lookup[(0, 0, 0)] == 1)
    for a in range(2 * k + 1):
        constraints.append(g_lookup[(a, 0)] == m_lookup[(a,)])
    for a in range(2 * k + 1):
        for b in range(2 * k + 1 - a):
            constraints.append(y_lookup[(a, b, 0)] == g_lookup[(a, b)])
    if enforce_reflection:
        for a in range(1, 2 * k + 1, 2):
            constraints.append(m_lookup[(a,)] == 0)
        for (a, b) in list(g_lookup.keys()):
            if (a + b) % 2 == 1:
                constraints.append(g_lookup[(a, b)] == 0)
        for (a, b, c) in list(y_lookup.keys()):
            if (a + b + c) % 2 == 1:
                constraints.append(y_lookup[(a, b, c)] == 0)

    # 1D blocks
    basis_1d = basis_for_psd(1, k)
    M1 = _moment_matrix(m_lookup, basis_1d, 1)
    constraints.append(M1 >> 0)
    block_sizes = [len(basis_1d)]
    if k >= 1:
        loc_basis_1d = basis_for_psd(1, k - 1)
        L1 = _localizer_matrix(m_lookup, loc_basis_1d, 1, axis=0, support_half=SUPPORT_HALF)
        constraints.append(L1 >> 0)
        block_sizes.append(len(loc_basis_1d))

    # 2D blocks
    basis_2d = basis_for_psd(2, k)
    M2 = _moment_matrix(g_lookup, basis_2d, 2)
    constraints.append(M2 >> 0)
    block_sizes.append(len(basis_2d))
    if k >= 1:
        loc_basis_2d = basis_for_psd(2, k - 1)
        for axis in range(2):
            L2 = _localizer_matrix(g_lookup, loc_basis_2d, 2, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L2 >> 0)
            block_sizes.append(len(loc_basis_2d))

    # 3D blocks
    basis_3d = basis_for_psd(3, k)
    M3 = _moment_matrix(y_lookup, basis_3d, 3)
    constraints.append(M3 >> 0)
    block_sizes.append(len(basis_3d))
    if k >= 1:
        loc_basis_3d = basis_for_psd(3, k - 1)
        for axis in range(3):
            L3 = _localizer_matrix(y_lookup, loc_basis_3d, 3, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L3 >> 0)
            block_sizes.append(len(loc_basis_3d))

    # Objective uses g (same as 2-point)
    coeffs = bump_coeffs_2d(N, epsilon_shift=epsilon_shift)
    obj_expr = 0
    for (a, b), c in coeffs.items():
        if a + b > 2 * k:
            continue
        obj_expr = obj_expr + c * g_lookup[(a, b)]

    problem = cp.Problem(cp.Minimize(obj_expr), constraints)
    info = BuildInfo(
        k=k, N=N, with_3pt=True,
        n_vars_m=2 * k + 1,
        n_vars_g=len(set(id(v) for v in g_lookup.values())),
        n_vars_y=len(set(id(v) for v in y_lookup.values())),
        block_sizes=block_sizes,
        build_seconds=time.time() - t0,
    )
    handles = dict(m=m_lookup, g=g_lookup, y=y_lookup,
                   basis_1d=basis_1d, basis_2d=basis_2d, basis_3d=basis_3d,
                   coeffs=coeffs)
    return problem, info, handles


def solve(problem: cp.Problem, info: BuildInfo, *, solver: str = "MOSEK", verbose: bool = False) -> BuildInfo:
    """Solve the SDP and populate solve_seconds / peak_mem_mb / status / objective in info."""
    import psutil
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.time()
    try:
        problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        info.solve_seconds = time.time() - t0
        info.status = f"ERROR: {type(exc).__name__}: {exc}"
        info.peak_mem_mb = (proc.memory_info().rss - rss_before) / 1e6
        return info
    info.solve_seconds = time.time() - t0
    info.status = problem.status
    info.objective = float(problem.value) if problem.value is not None else None
    info.peak_mem_mb = (proc.memory_info().rss - rss_before) / 1e6
    return info


__all__ = [
    "bump_integral", "bump_coeffs_2d",
    "build_2pt_sdp", "build_3pt_sdp", "solve",
    "BuildInfo",
]

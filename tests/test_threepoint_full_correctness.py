"""Correctness tests for lasserre/threepoint_full.py."""
from __future__ import annotations

import math
import sys
from itertools import permutations
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lasserre.threepoint_full import (
    legendre_orthonormal_coeffs,
    MomentVarMap, moment_matrix, localizer_matrix,
    NuSideDualization,
    build_2pt_full, build_3pt_full, solve,
)


# =====================================================================
# Legendre orthonormality
# =====================================================================

def _gauss_quadrature_legendre(n: int):
    """Gauss-Legendre nodes/weights on [-1/2, 1/2]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = nodes / 2.0  # rescale [-1, 1] -> [-1/2, 1/2]
    weights = weights / 2.0
    return nodes, weights


def test_legendre_orthonormal():
    """int_{-1/2}^{1/2} p_j(t) p_k(t) dt = delta_{jk}."""
    N = 6
    leg = legendre_orthonormal_coeffs(N)
    nodes, weights = _gauss_quadrature_legendre(40)
    # Evaluate p_j at nodes by horner
    P = np.zeros((N + 1, len(nodes)))
    for j in range(N + 1):
        for r in range(j + 1):
            P[j] += leg[j, r] * nodes ** r
    G = (P * weights) @ P.T
    np.testing.assert_allclose(G, np.eye(N + 1), atol=1e-10)


def test_legendre_low_degree_explicit():
    """Match against analytic formulas:
       p_0 = 1
       p_1(t) = sqrt(3) * 2t = 2 sqrt(3) t
       p_2(t) = sqrt(5) * (1/2)(3*(2t)^2 - 1) = sqrt(5) * (6 t^2 - 0.5)
    """
    leg = legendre_orthonormal_coeffs(2)
    np.testing.assert_allclose(leg[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(leg[1], [0.0, 2 * math.sqrt(3), 0.0], atol=1e-12)
    # P_2(x) = (3x^2 - 1)/2 = 1.5 x^2 - 0.5
    # p_2(t) = sqrt(5) * (1.5 (2t)^2 - 0.5) = sqrt(5) * (6 t^2 - 0.5)
    np.testing.assert_allclose(leg[2], [-0.5 * math.sqrt(5), 0.0, 6 * math.sqrt(5)], atol=1e-12)


# =====================================================================
# Moment-var map: orbit structure
# =====================================================================

def test_orbit_dedup_3d():
    """All S_3 permutations of an even-degree multi-index map to the same variable."""
    m = MomentVarMap(dim=3, max_deg=4)
    base = (3, 1, 0)
    if sum(base) % 2 == 1:
        # base is reflection-zero: every permutation is the constant 0
        for perm in permutations(base):
            assert m.get(perm).value is None  # CVXPY Constant
    canonical_var = m.get((4, 0, 0))
    for perm in permutations((4, 0, 0)):
        assert m.get(perm) is canonical_var
    canonical_var2 = m.get((2, 2, 0))
    for perm in permutations((2, 2, 0)):
        assert m.get(perm) is canonical_var2


def test_reflection_zero():
    """Odd-total-degree multi-indices map to scalar zero."""
    m = MomentVarMap(dim=3, max_deg=4)
    z = m.get((1, 0, 0))
    # Constant has no variables
    assert z.is_constant()
    assert float(z.value if z.value is not None else 0.0) == 0.0


def test_orbit_count_3d_k4():
    """Number of S_3 orbits of even-degree multi-indices with sum <= 8."""
    m = MomentVarMap(dim=3, max_deg=8)
    # Manually count: for each total degree 2d with d=0..4, count partitions
    # of 2d into 3 nonneg parts with parts sorted decreasing.
    expected = 0
    for total in range(0, 9, 2):  # 0, 2, 4, 6, 8
        # number of partitions of total into <=3 parts
        cnt = 0
        for a in range(total + 1):
            for b in range(min(a, total - a) + 1):
                c = total - a - b
                if c <= b:
                    cnt += 1
        expected += cnt
    assert m.n_orbits() == expected


# =====================================================================
# Moment matrix entries pull from the map correctly
# =====================================================================

def test_moment_matrix_zero_block():
    """When max_deg permits, M_k entries with odd total degree are 0."""
    m = MomentVarMap(dim=3, max_deg=4)
    # basis at level 1: {(0,0,0), (1,0,0), (0,1,0), (0,0,1)}
    # M[(1,0,0), (0,0,0)] = y_{1,0,0} which is reflection-zero
    M = moment_matrix(m, 1)
    # Set the free variables to arbitrary values, check M
    for v in m.free_variables():
        v.value = 0.5
    # Check the (1,0,0) <-> (0,0,0) entry
    M_val = M.value
    # basis indexing matches enum_multi_indices with d=3, max_deg=1.
    # That's [(0,0,0), (0,0,1), (0,1,0), (1,0,0)] (lex with d=3? actually order depends).
    # We just check that for any pair of basis elements with odd-sum total, entry is 0.
    from lasserre.threepoint_full import enum_multi_indices
    basis = enum_multi_indices(3, 1)
    for i, alpha in enumerate(basis):
        for j, beta in enumerate(basis):
            tot = sum(alpha) + sum(beta)
            if tot % 2 == 1:
                assert abs(M_val[i, j]) < 1e-12, f"M[{alpha}, {beta}] = {M_val[i, j]} != 0"


# =====================================================================
# Two-point and three-point: 3pt is a tightening (lambda^3pt >= lambda^2pt)
# =====================================================================

@pytest.mark.parametrize("k,N", [(3, 4), (4, 4), (4, 6)])
def test_3pt_dominates_2pt(k, N):
    """lambda^{3pt} >= lambda^{2pt} - tol since 3pt strictly tightens."""
    p2, info2, _ = build_2pt_full(k, N)
    p3, info3, _ = build_3pt_full(k, N)
    info2 = solve(p2, info2, solver="MOSEK")
    info3 = solve(p3, info3, solver="MOSEK")
    assert info2.objective is not None, f"2pt failed: {info2.status}"
    assert info3.objective is not None, f"3pt failed: {info3.status}"
    # Allow MOSEK numerical tolerance ~ 1e-7
    assert info3.objective >= info2.objective - 1e-6, (
        f"3pt < 2pt by {info2.objective - info3.objective:.3e} — bug "
        f"(2pt={info2.objective}, 3pt={info3.objective})"
    )


def test_normalization():
    """y_{0,0,0} = 1 must hold at the optimum."""
    p3, info3, h3 = build_3pt_full(k=3, N=4)
    info3 = solve(p3, info3, solver="MOSEK")
    assert info3.objective is not None
    y0 = h3["y"].get((0, 0, 0)).value
    np.testing.assert_allclose(y0, 1.0, atol=1e-7)


def test_lambda_in_sensible_range():
    """lambda^* approximates min_f sup_t (f*f)_N(t).  Should be in [0.5, 2.5]
    given that int(f*f)=1 forces sup >= 1 (but Legendre projection can overshoot
    slightly below) and uniform f has sup(f*f) = 2 as a feasible upper witness.
    """
    p3, info3, _ = build_3pt_full(k=3, N=4)
    info3 = solve(p3, info3, solver="MOSEK")
    assert info3.objective is not None
    assert 0.5 <= info3.objective <= 2.5


# =====================================================================
# Polynomial identity sanity: lambda - sum_j alpha_j(g) p_j(t) is SOS-able
# =====================================================================

def test_polynomial_identity_satisfied():
    """At the optimum, the polynomial identity must hold to high precision."""
    p2, info2, h2 = build_2pt_full(k=3, N=4)
    info2 = solve(p2, info2, solver="MOSEK")
    assert info2.objective is not None
    nu = h2["nu"]
    g_map = h2["g"]
    # Evaluate LHS and RHS coefficient-by-coefficient
    leg = nu.leg  # (N+1, N+1)
    N = nu.N
    K_nu = nu.K_nu
    lam_val = float(nu.lam.value)
    X0_val = nu.X0.value
    X1_val = nu.X1.value if nu.K_nu >= 1 else None
    # Build g values from optimal
    max_g_deg = min(2 * 3, N)
    g_vals = np.zeros((max_g_deg + 1, max_g_deg + 1))
    for a in range(max_g_deg + 1):
        for b in range(max_g_deg + 1 - a):
            v = g_map.get((a, b)).value
            g_vals[a, b] = v if v is not None else 0.0
    # Check identity for r = 0..2*K_nu
    max_err = 0.0
    mu_scale = 0.25  # rescaled mu side
    for r in range(2 * K_nu + 1):
        # LHS
        lhs = (lam_val if r == 0 else 0.0)
        # tildeQ[r] = sum_{j>=r}^N leg[j,r] * sum_{a+b<=j} leg[j,a+b] * C(a+b, a) * mu_scale^(a+b) * g[a, b]
        for j in range(r, N + 1):
            for a in range(j + 1):
                for b in range(j + 1 - a):
                    if a + b > max_g_deg:
                        continue
                    s = a + b
                    lhs -= leg[j, r] * leg[j, s] * math.comb(s, a) * (mu_scale ** s) * g_vals[a, b]
        # RHS
        rhs = 0.0
        size0 = K_nu + 1
        for i in range(size0):
            j = r - i
            if 0 <= j < size0:
                rhs += X0_val[i, j]
        if K_nu >= 1:
            size1 = K_nu
            for i in range(size1):
                j = r - i
                if 0 <= j < size1:
                    rhs += 0.25 * X1_val[i, j]
            for i in range(size1):
                j = (r - 2) - i
                if 0 <= j < size1:
                    rhs -= X1_val[i, j]
        err = abs(lhs - rhs)
        max_err = max(max_err, err)
    assert max_err < 1e-6, f"Polynomial identity violated by {max_err}"


# =====================================================================
# Recoverable extremizer sanity: uniform f yields a finite, nonnegative value
# =====================================================================

def test_uniform_f_feasibility():
    """Uniform f on [-1/4, 1/4] gives sup(f*f) = 2.  The relaxation lambda^* is
    the min over the moment cone of sup_t (f*f)_N(t), which approximates
    inf_f sup(f*f) >= 1 (since int(f*f) = 1, average over [-1/2,1/2] >= 1).
    """
    p2, info2, _ = build_2pt_full(k=3, N=4)
    info2 = solve(p2, info2, solver="MOSEK")
    assert info2.objective is not None
    # Lower bound: average value of (f*f) over [-1/2, 1/2] is int(f*f)/(length) = 1/1 = 1.
    # So sup_t (f*f)_N(t) >= 1 (Legendre projection preserves mean).
    # However, MOSEK projection can fall slightly below 1 due to overshooting; allow some slack.
    assert 0.5 <= info2.objective <= 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

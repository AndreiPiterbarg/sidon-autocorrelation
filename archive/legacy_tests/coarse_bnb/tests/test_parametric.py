"""Unit tests for the `parametric` package.

Run with:
    pytest -x tests/test_parametric.py

Or directly:
    python -m pytest tests/test_parametric.py -v

Note on achievable bounds
-------------------------
The pointwise SOS certificate
    p(x+y) - tilde_lam = sigma_0 + (1/16 - x^2) sigma_1 + (1/16 - y^2) sigma_2
over (x, y) in [-1/4, 1/4]^2 with int p = 1 and p >= 0 on [-1/2, 1/2] is
UPPER-BOUNDED by min_{u in [-1/2, 1/2]} p(u), which is <= 1 (Dirac-limit
argument on admissible densities).  Hence `solve_outer_sdp(...)` in this
formulation will return <= 1 at any (L, N).

Reproducing Matolcsi-Vinuesa's 1.262 bound requires either
    (a) exploiting f (x) f independence / rank-1 structure, or
    (b) using Fourier-side constraints (positive-definiteness of f*f),
neither of which is captured by the pointwise SOS certificate alone.

The tests below therefore assert the TRIVIAL bound (1.0) is reproduced, and
mark the MV reproduction as `xfail` for clarity.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from parametric.chebyshev_duality import (
    bivariate_basis,
    bivariate_pair_map,
    build_Kl_table,
    chebyshev_monomial_coefs,
    cheb_expand_to_monomial_vector,
    integrate_Tl_2t,
)
from parametric.primal_qp import solve_inner_qp, evaluate_p_on_grid
from parametric.outer_sdp import solve_outer_sdp
from parametric.certify import (
    certify_outer_sdp,
    reconstruct_univ_p,
    round_matrix,
    psd_check_fmpq,
)


# ---------------------------------------------------------------------------
# chebyshev_duality algebra
# ---------------------------------------------------------------------------

def test_chebyshev_monomial_T2():
    c = chebyshev_monomial_coefs(5)
    # T_2(t) = 2 t^2 - 1.
    assert c[2][0] == -1
    assert c[2][1] == 0
    assert c[2][2] == 2
    # T_4(t) = 8 t^4 - 8 t^2 + 1.
    assert c[4][0] == 1
    assert c[4][2] == -8
    assert c[4][4] == 8


def test_integrate_Tl_2t():
    # int_{-1/2}^{1/2} T_0(2t) dt = 1.
    # int T_2(2t) dt = int (8 t^2 - 1) dt = 8/12 - 1 = -1/3.
    # int T_4(2t) dt = int (128 t^4 - 32 t^2 + 1) dt = 128/80 - 32/12 + 1
    #                = 8/5 - 8/3 + 1 = (24 - 40 + 15)/15 = -1/15.
    vals = integrate_Tl_2t(5)
    assert vals[0] == Fraction(1)
    assert vals[1] == Fraction(0)
    assert vals[2] == Fraction(-1, 3)
    assert vals[3] == Fraction(0)
    assert vals[4] == Fraction(-1, 15)


def test_build_Kl_table_T2():
    # T_2(2(x+y)) = 8(x+y)^2 - 1 = 8 x^2 + 16 xy + 8 y^2 - 1.
    K = build_Kl_table(5)
    assert K[2][0][0] == -1
    assert K[2][2][0] == 8
    assert K[2][0][2] == 8
    assert K[2][1][1] == 16


def test_bivariate_basis_and_pairs():
    basis = bivariate_basis(2)
    assert len(basis) == 6
    # Pairs summing to (1, 1): only (a=0,b=1)+(a=1,b=0) and (a=1,b=0)+(a=0,b=1).
    pm = bivariate_pair_map(bivariate_basis(1))
    basis1 = bivariate_basis(1)
    idx = {ab: i for i, ab in enumerate(basis1)}
    expected = {(idx[(1, 0)], idx[(0, 1)]), (idx[(0, 1)], idx[(1, 0)])}
    assert set(pm[(1, 1)]) == expected


def test_monomial_expansion_of_cheb():
    # p(t) = T_2(2t) => monomials [-1, 0, 8, 0, 0] at L=5.
    m = [Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0)]
    mono = cheb_expand_to_monomial_vector(m)
    assert mono[0] == -1
    assert mono[1] == 0
    assert mono[2] == 8
    assert mono[3] == 0
    assert mono[4] == 0


# ---------------------------------------------------------------------------
# inner QP (bivariate moment relaxation)
# ---------------------------------------------------------------------------

def test_inner_qp_trivial_constant_p():
    # p(t) = 1  =>  lambda = 1.
    m = np.array([1.0])
    out = solve_inner_qp(m, N=0, solver="CLARABEL")
    assert out["status"] == "optimal"
    assert abs(out["lambda_N"] - 1.0) < 1e-6


def test_inner_qp_requires_2N_ge_Lm1():
    # L=3, N=0 should raise because 2*0 < 3-1.
    with pytest.raises(ValueError):
        solve_inner_qp(np.array([1.0, 0.0, 0.0]), N=0)


def test_inner_qp_matches_min_on_box():
    # p(t) = 24 t^2 - 2 = 1 + 3 T_2(2t) - 3 + ...  Let's just use the monomial.
    #   m = [1, 0, 3]  =>  p(t) = 1 + 3 (8 t^2 - 1) = 24 t^2 - 2.
    # Over admissible f, inf_f int p (f*f) = inf_f E_{f x f}[24(x+y)^2 - 2]
    #   = 24 * 2 * E[X^2] - 2.  Minimized when X concentrated at 0: inf -> -2.
    m = np.array([1.0, 0.0, 3.0])
    out = solve_inner_qp(m, N=1, solver="CLARABEL")
    assert out["status"] == "optimal"
    assert abs(out["lambda_N"] - (-2.0)) < 1e-4


def test_inner_qp_higher_N_non_worse():
    # Increasing N can only tighten (SDP relaxation is monotonically tighter).
    m = np.array([1.0, 0.0, 3.0, 0.0, 0.0])
    b1 = solve_inner_qp(m, N=2, solver="CLARABEL")["lambda_N"]
    b2 = solve_inner_qp(m, N=3, solver="CLARABEL")["lambda_N"]
    # b2 >= b1 - small tolerance.
    assert b2 + 1e-5 >= b1


# ---------------------------------------------------------------------------
# outer SDP
# ---------------------------------------------------------------------------

def test_outer_sdp_trivial_L1():
    # Any N gives bound = 1.
    out = solve_outer_sdp(L=1, N=1, solver="CLARABEL")
    assert out["status"] == "optimal"
    assert abs(out["bound"] - 1.0) < 1e-5


def test_outer_sdp_L3_N2_returns_valid_bound():
    out = solve_outer_sdp(L=3, N=2, solver="CLARABEL")
    assert out["status"] == "optimal"
    assert out["bound"] is not None
    # Bound theoretically <= 1 for this pointwise formulation.
    assert out["bound"] <= 1.0 + 1e-5


def test_outer_sdp_respects_int_p_eq_1():
    out = solve_outer_sdp(L=5, N=3, solver="CLARABEL")
    # int p = sum_l m_l * int T_l(2t).
    m = out["m"]
    int_T = [float(v) for v in integrate_Tl_2t(5)]
    intp = float(np.dot(m, int_T))
    assert abs(intp - 1.0) < 1e-5


@pytest.mark.xfail(
    reason=(
        "The pointwise SOS certificate in outer_sdp.py is theoretically capped "
        "at min_u p(u) <= 1 and cannot reproduce the Matolcsi-Vinuesa 1.262 "
        "bound without adding Fourier-side / independence constraints."
    ),
    strict=False,
)
def test_outer_sdp_beats_MV_1_262():  # pragma: no cover (xfail)
    out = solve_outer_sdp(L=11, N=6, solver="CLARABEL")
    assert out["bound"] > 1.262


# ---------------------------------------------------------------------------
# Certification (rational verification)
# ---------------------------------------------------------------------------

def test_certify_roundtrip_on_L1_N1():
    sol = solve_outer_sdp(L=1, N=1, solver="CLARABEL")
    cert = certify_outer_sdp(sol, bits=50, verify_psd=True)
    # int p should be exactly 1 after rounding 1.0 to a dyadic rational.
    assert abs(float(cert["int_p_cert"]) - 1.0) < 1e-10
    assert 0.99 <= float(cert["lam_cert"]) <= 1.0 + 1e-3


def test_certify_psd_check_on_known_PSD():
    # I + diag(1, 2) is PSD.
    M = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
    assert psd_check_fmpq(M)


def test_certify_psd_check_rejects_indefinite():
    # [[1, 2], [2, 1]] has det = 1 - 4 = -3 < 0.
    M = [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]]
    assert not psd_check_fmpq(M)


def test_certify_reconstruction_univariate():
    # Q0p = I_2  =>  sigma_0_p(t) = 1 + t^2.
    # Q1p = [[1]] =>  sigma_1_p(t) = 1.
    # (1 - 4 t^2) * sigma_1_p = 1 - 4 t^2.
    # Total p(t) = 1 + t^2 + 1 - 4 t^2 = 2 - 3 t^2 -> [2, 0, -3, 0, 0] at L=5.
    Q0p = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    Q1p = [[Fraction(1)]]
    p = reconstruct_univ_p(Q0p, Q1p, L=5)
    assert p[0] == 2
    assert p[1] == 0
    assert p[2] == -3
    assert p[3] == 0
    assert p[4] == 0


def test_round_matrix_symmetry():
    M = np.array([[1.0, 0.5001], [0.4999, 2.0]])
    R = round_matrix(M, bits=20)
    # Symmetric after averaging.
    assert R[0][1] == R[1][0]


# ---------------------------------------------------------------------------
# sanity:  evaluate_p_on_grid
# ---------------------------------------------------------------------------

def test_evaluate_p_on_grid_constant():
    res = evaluate_p_on_grid(np.array([1.0]))
    assert abs(res["min"] - 1.0) < 1e-12
    assert abs(res["max"] - 1.0) < 1e-12


def test_evaluate_p_on_grid_T2():
    # p(t) = 1 + T_2(2t) = 1 + 8 t^2 - 1 = 8 t^2 >= 0, max at t = +/- 1/2 is 2.
    res = evaluate_p_on_grid(np.array([1.0, 0.0, 1.0]))
    assert abs(res["min"]) < 1e-10
    assert abs(res["max"] - 2.0) < 1e-3

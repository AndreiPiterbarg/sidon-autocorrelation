"""C.3 tests for `chebyshev_dual.bilinear_kernel`.

Covers:
  - Jacobi-Anger Chebyshev series numerically recovers cos/sin within the
    rigorous truncation bound.
  - A_0 = e_0 e_0^T and A_l for l >= 1 is rank <= 2.
  - A(r) is symmetric and linear in r.
  - For r = (1, 0, ..., 0) (p = 1, uniform measure on T) and c with c_0 = 1,
    the bilinear form c^T A(r) c equals 1 (the trivial bound).
  - For uniform f on [-1/4, 1/4] and a small test r, c^T A(r) c agrees
    (within truncation bound) with  integral integral f(x) f(y) p(x+y) dx dy
    computed by scipy quadrature.
  - Truncation bound is positive, decays as N grows, and dominates the
    observed pointwise error.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from flint import arb, arb_mat, ctx, fmpq

from chebyshev_dual.cheb_hankel import (
    chebyshev_moments_of_density,
    fmpq_mat_to_numpy,
)
from chebyshev_dual.bilinear_kernel import (
    arb_mat_to_numpy_mid,
    build_jacobi_anger_tables,
    build_kernel_matrix_A,
    evaluate_cheb_series_float,
    jacobi_anger_chebyshev_coeffs,
    jacobi_anger_truncation_infty,
    kernel_truncation_error_bound,
    omega_of,
)


def _arb_mid_f(a: arb) -> float:
    return float(a.mid())


def _arb_upper_f(a: arb) -> float:
    return float(a.upper())


# ---------------------------------------------------------------------
# Jacobi-Anger pointwise recovery
# ---------------------------------------------------------------------

@pytest.mark.parametrize("l", [1, 2, 4, 8])
@pytest.mark.parametrize("N", [16, 32])
def test_jacobi_anger_chebyshev_recovers_cos_sin(l: int, N: int) -> None:
    """Sum alpha_{l, k} T_k(y) reproduces cos(omega_l y) within truncation bound."""
    old_prec = ctx.prec
    ctx.prec = 128
    try:
        K = 2 * N
        alpha, beta = jacobi_anger_chebyshev_coeffs(l, K)
        # Joint truncation bound (L^inf on y in [-1, 1]).
        tau = _arb_upper_f(jacobi_anger_truncation_infty(l, K))

        ys = np.linspace(-0.99, 0.99, 21)
        for y in ys:
            cos_true = math.cos(math.pi * l * y / 2.0)
            sin_true = math.sin(math.pi * l * y / 2.0)
            cos_approx = evaluate_cheb_series_float(alpha, float(y))
            sin_approx = evaluate_cheb_series_float(beta, float(y))
            # Each individual error is bounded by the joint tau; use that as slack.
            assert abs(cos_approx - cos_true) <= tau + 1e-12, (
                f"l={l} N={N} y={y}: cos err {abs(cos_approx - cos_true):.3e} vs tau {tau:.3e}"
            )
            assert abs(sin_approx - sin_true) <= tau + 1e-12, (
                f"l={l} N={N} y={y}: sin err {abs(sin_approx - sin_true):.3e} vs tau {tau:.3e}"
            )
    finally:
        ctx.prec = old_prec


def test_l0_alpha_is_unit_beta_zero() -> None:
    """l=0 gives cos(0)=1, sin(0)=0."""
    alpha, beta = jacobi_anger_chebyshev_coeffs(0, 10)
    assert _arb_mid_f(alpha[0]) == 1.0
    for k in range(1, 11):
        assert _arb_mid_f(alpha[k]) == 0.0
        assert _arb_mid_f(beta[k]) == 0.0


def test_truncation_bound_decays_with_N() -> None:
    """For fixed l, the truncation bound decreases monotonically as N grows."""
    ctx.prec = 128
    l = 4
    taus = []
    for N in [8, 16, 24, 32]:
        taus.append(_arb_upper_f(jacobi_anger_truncation_infty(l, 2 * N)))
    for i in range(len(taus) - 1):
        assert taus[i] >= taus[i + 1], f"tau not monotone: {taus}"
    # And the largest N gives a tiny bound at l=4.
    assert taus[-1] < 1e-20


def test_truncation_bound_nonnegative() -> None:
    for l in [1, 3, 7, 16]:
        for N in [16, 32]:
            tau = jacobi_anger_truncation_infty(l, 2 * N)
            assert _arb_upper_f(tau) >= 0.0


# ---------------------------------------------------------------------
# A_l basis matrices
# ---------------------------------------------------------------------

def test_A0_is_e0_outer_e0() -> None:
    ctx.prec = 128
    tables = build_jacobi_anger_tables(N=8, D=4)
    A0 = arb_mat_to_numpy_mid(tables.A_basis[0])
    expected = np.zeros((17, 17))
    expected[0, 0] = 1.0
    assert np.allclose(A0, expected, atol=1e-14)


def test_Al_rank_at_most_two() -> None:
    """A_l = 2 (alpha alpha^T - beta beta^T) has rank <= 2 for l >= 1."""
    ctx.prec = 128
    tables = build_jacobi_anger_tables(N=12, D=6)
    for l in range(1, 7):
        Al = arb_mat_to_numpy_mid(tables.A_basis[l])
        Al_sym = 0.5 * (Al + Al.T)
        svals = np.linalg.svd(Al_sym, compute_uv=False)
        # The third singular value should be at machine noise.
        assert svals[2] < max(svals[0], svals[1]) * 1e-12 + 1e-14, (
            f"A_l for l={l}: singular values {svals[:4]}"
        )


def test_kernel_A_symmetric() -> None:
    ctx.prec = 128
    tables = build_jacobi_anger_tables(N=10, D=5)
    r = [arb('0.7'), arb('-0.2'), arb('0.3'), arb(0), arb('0.05'), arb('-0.1')]
    A = arb_mat_to_numpy_mid(build_kernel_matrix_A(r, tables))
    assert np.allclose(A, A.T, atol=1e-14)


def test_kernel_A_linear_in_r() -> None:
    ctx.prec = 128
    tables = build_jacobi_anger_tables(N=8, D=4)
    r1 = [arb('0.3'), arb('0.1'), arb('-0.2'), arb('0.05'), arb(0)]
    r2 = [arb('0.4'), arb('-0.15'), arb('0.25'), arb(0), arb('0.1')]
    r_sum = [r1[l] + r2[l] for l in range(len(r1))]

    A1 = arb_mat_to_numpy_mid(build_kernel_matrix_A(r1, tables))
    A2 = arb_mat_to_numpy_mid(build_kernel_matrix_A(r2, tables))
    A_sum = arb_mat_to_numpy_mid(build_kernel_matrix_A(r_sum, tables))
    assert np.allclose(A1 + A2, A_sum, atol=1e-14)


# ---------------------------------------------------------------------
# Bilinear form sanity
# ---------------------------------------------------------------------

def _quad_form_arb(c_floats: list, A: arb_mat) -> float:
    n = A.nrows()
    s = arb(0)
    for i in range(n):
        for j in range(n):
            s = s + arb(c_floats[i]) * A[i, j] * arb(c_floats[j])
    return _arb_mid_f(s)


def test_uniform_r_matches_trivial_bound() -> None:
    """r = (1, 0, 0, ...) gives p = 1, so integral integral f f p = (int f)^2 = 1
    for any probability f.  Check c^T A c = 1 for any c with c_0 = 1."""
    ctx.prec = 128
    N, D = 12, 6
    tables = build_jacobi_anger_tables(N=N, D=D)
    r = [arb(1)] + [arb(0)] * D
    A = build_kernel_matrix_A(r, tables)

    # Try several c vectors with c_0 = 1.
    rng = np.random.default_rng(42)
    for trial in range(5):
        c = rng.standard_normal(2 * N + 1)
        c[0] = 1.0
        q = _quad_form_arb(list(c), A)
        assert abs(q - 1.0) < 1e-14, f"trial {trial}: q = {q}"


def test_bilinear_form_matches_direct_quadrature_uniform_f() -> None:
    """For f = uniform on [-1/4, 1/4] and a chosen r, c^T A(r) c agrees with
    scipy-quadrature integral integral f(x) f(y) p(x+y) dx dy within the rigorous
    truncation bound."""
    from scipy import integrate
    ctx.prec = 128

    N, D = 20, 8
    # Uniform f: c from closed form.
    def uniform_cheb_closed(max_k: int):
        out = []
        for k in range(max_k + 1):
            if k == 0:
                out.append(1.0)
            elif k % 2 == 1:
                out.append(0.0)
            else:
                out.append(-1.0 / (k * k - 1))
        return out
    c = uniform_cheb_closed(2 * N)

    # Chosen r: small magnitude so p stays close to 1.
    r_floats = [1.0, 0.1, -0.05, 0.02, 0.0, 0.01, 0.0, -0.01, 0.0]
    r_arb = [arb.from_float(float(x)) if hasattr(arb, 'from_float') else arb(float(x)) for x in r_floats]
    # Fall back to arb via repr if from_float not present.
    r_arb = [arb(repr(float(x))) for x in r_floats]

    tables = build_jacobi_anger_tables(N=N, D=D)
    A = build_kernel_matrix_A(r_arb, tables)
    q = _quad_form_arb(c, A)

    # Direct quadrature.
    f_dens = 2.0  # uniform f(x) = 2 on [-1/4, 1/4]
    def p_of_t(t):
        val = r_floats[0]
        for l in range(1, len(r_floats)):
            val += 2.0 * r_floats[l] * math.cos(2.0 * math.pi * l * t)
        return val

    # integral integral f(x) f(y) p(x+y) dx dy  on [-1/4,1/4]^2
    def inner(x):
        val, _ = integrate.quad(
            lambda y: f_dens * f_dens * p_of_t(x + y),
            -0.25, 0.25, epsabs=1e-12, epsrel=1e-12,
        )
        return val
    exact, _ = integrate.quad(inner, -0.25, 0.25, epsabs=1e-12, epsrel=1e-10)

    # Truncation bound for uniform f is bounded by the r-specific kernel bound.
    tau = _arb_upper_f(kernel_truncation_error_bound(r_arb, N, D))
    diff = abs(q - exact)
    # The bound must dominate the observed difference (mod numerical quadrature noise).
    assert diff <= tau + 1e-8, f"diff {diff:.3e} > tau {tau:.3e}"
    # And the bound itself is small at this N, D.
    assert tau < 1e-20, f"tau too loose: {tau}"


def test_jacobi_anger_tables_dimensions() -> None:
    ctx.prec = 128
    N, D = 6, 3
    tables = build_jacobi_anger_tables(N=N, D=D)
    assert tables.N == N
    assert tables.D == D
    assert len(tables.alphas) == D + 1
    assert len(tables.betas) == D + 1
    assert len(tables.A_basis) == D + 1
    for l in range(D + 1):
        assert len(tables.alphas[l]) == 2 * N + 1
        assert len(tables.betas[l]) == 2 * N + 1
        assert tables.A_basis[l].nrows() == 2 * N + 1
        assert tables.A_basis[l].ncols() == 2 * N + 1

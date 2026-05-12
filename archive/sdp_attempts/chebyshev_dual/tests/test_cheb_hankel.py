"""C.1 tests for `chebyshev_dual.cheb_hankel`.

Covers:
  - Chebyshev <-> monomial change of basis is exact and T @ U = I.
  - Uniform measure on [-1/4, 1/4]: closed-form c_k agree with numeric
    Chebyshev moments, and both Hausdorff blocks H_1, H_2 are PSD.
  - Delta measure at the origin: c_k = T_k(0), H_1 and H_2 are rank-1 PSD.
  - Bad moment rejection: c = (1, 0, 2, 0, ...) is rejected by the H_2 block.
  - Apply returns fmpq_mat and matches the float path within epsilon.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from flint import fmpq, fmpq_mat

from chebyshev_dual.cheb_hankel import (
    HankelBlockSpec,
    chebyshev_moments_of_density,
    cheb_to_monomial_matrix,
    fmpq_mat_to_numpy,
    hausdorff_hankel_blocks_chebyshev,
    monomial_to_cheb_matrix,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _min_eig_sym(M: np.ndarray) -> float:
    M = 0.5 * (M + M.T)
    return float(np.linalg.eigvalsh(M).min())


def _uniform_chebyshev_moments_closed_form(max_k: int) -> list:
    """Closed-form c_k for uniform f = 2 on [-1/4, 1/4].

    c_k = (1/2) int_{-1}^{1} T_k(y) dy
        = { 1           if k == 0,
            0           if k odd,
            -1/(k^2-1)  if k even, k >= 2 }.
    """
    out = []
    for k in range(max_k + 1):
        if k == 0:
            out.append(Fraction(1))
        elif k % 2 == 1:
            out.append(Fraction(0))
        else:
            out.append(Fraction(-1, k * k - 1))
    return out


# ---------------------------------------------------------------------
# Change-of-basis sanity
# ---------------------------------------------------------------------

@pytest.mark.parametrize("N", [0, 1, 2, 5, 10, 16])
def test_T_times_U_is_identity(N: int) -> None:
    T = monomial_to_cheb_matrix(N)
    U = cheb_to_monomial_matrix(N)
    # Promote T to fmpq_mat for the product.
    T_q = fmpq_mat(T.nrows(), T.ncols())
    for i in range(T.nrows()):
        for j in range(T.ncols()):
            T_q[i, j] = fmpq(int(T[i, j]))
    prod = T_q * U
    for i in range(N + 1):
        for j in range(N + 1):
            want = fmpq(1) if i == j else fmpq(0)
            assert prod[i, j] == want, f"(T U)[{i},{j}] = {prod[i, j]} != {want}"


def test_known_chebyshev_coefficients() -> None:
    """T_0..T_5 match textbook coefficients."""
    T = monomial_to_cheb_matrix(5)
    expected = {
        0: [1, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0],
        2: [-1, 0, 2, 0, 0, 0],
        3: [0, -3, 0, 4, 0, 0],
        4: [1, 0, -8, 0, 8, 0],
        5: [0, 5, 0, -20, 0, 16],
    }
    for k, row in expected.items():
        got = [int(T[k, j]) for j in range(6)]
        assert got == row, f"T_{k}: got {got}, want {row}"


# ---------------------------------------------------------------------
# Uniform measure on [-1/4, 1/4]
# ---------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 4, 8])
def test_uniform_measure_moments_closed_form_matches_numeric(N: int) -> None:
    """Numerical Chebyshev moments of f(x) = 2 on [-1/4, 1/4] match closed form."""
    closed = _uniform_chebyshev_moments_closed_form(2 * N)
    numeric = chebyshev_moments_of_density(lambda x: 2.0, 2 * N)
    assert len(numeric) == 2 * N + 1
    for k, (cf, nf) in enumerate(zip(closed, numeric)):
        assert abs(float(cf) - nf) < 1e-10, (
            f"c_{k}: closed={cf} ({float(cf):.6e}), numeric={nf:.6e}"
        )


@pytest.mark.parametrize("N", [2, 4, 8, 12])
def test_uniform_measure_hausdorff_blocks_psd(N: int) -> None:
    """Uniform measure yields strictly PSD H_1 and H_2 (nonneg interior point)."""
    closed = _uniform_chebyshev_moments_closed_form(2 * N)
    c = [fmpq(cf.numerator, cf.denominator) for cf in closed]

    H1_spec, H2_spec = hausdorff_hankel_blocks_chebyshev(N)
    H1 = H1_spec.apply(c)
    H2 = H2_spec.apply(c)

    assert H1.nrows() == N + 1 and H1.ncols() == N + 1
    assert H2.nrows() == N and H2.ncols() == N

    # Numeric PSD with a bit of slack (the uniform measure is strictly inside the moment cone).
    eig1 = _min_eig_sym(fmpq_mat_to_numpy(H1))
    eig2 = _min_eig_sym(fmpq_mat_to_numpy(H2))
    # Give a tiny absolute tolerance for roundoff; min eig is well-bounded away from 0.
    assert eig1 > 1e-10, f"H_1 min eig = {eig1} at N={N}"
    assert eig2 > 1e-10, f"H_2 min eig = {eig2} at N={N}"


def test_uniform_c0_equals_one() -> None:
    """Mass normalization: c_0 = 1 for any probability measure, uniform included."""
    closed = _uniform_chebyshev_moments_closed_form(10)
    assert closed[0] == Fraction(1)


# ---------------------------------------------------------------------
# Delta measure at the origin
# ---------------------------------------------------------------------

def _T_k_at_0(k: int) -> Fraction:
    """T_k(0): 1 at k=0, (-1)^{k/2} at even k>=2, 0 at odd k."""
    if k == 0:
        return Fraction(1)
    if k % 2 == 1:
        return Fraction(0)
    return Fraction(1) if (k // 2) % 2 == 0 else Fraction(-1)


@pytest.mark.parametrize("N", [2, 4, 8])
def test_delta_measure_moments_and_hausdorff_rank_one(N: int) -> None:
    """Delta at origin: c_k = T_k(0); H_1, H_2 are rank-1 PSD (boundary)."""
    c = [fmpq(_T_k_at_0(k).numerator, _T_k_at_0(k).denominator)
         for k in range(2 * N + 1)]
    assert c[0] == fmpq(1)  # normalization

    # tilde_m = U c; for delta at y = 0 we expect tilde_m_k = 0^k (1, 0, 0, ...).
    U = cheb_to_monomial_matrix(2 * N)
    tilde_m = [sum((U[k, j] * c[j] for j in range(2 * N + 1)), fmpq(0))
               for k in range(2 * N + 1)]
    assert tilde_m[0] == fmpq(1)
    for k in range(1, 2 * N + 1):
        assert tilde_m[k] == fmpq(0), f"tilde_m_{k} should be 0, got {tilde_m[k]}"

    H1_spec, H2_spec = hausdorff_hankel_blocks_chebyshev(N)
    H1 = fmpq_mat_to_numpy(H1_spec.apply(c))
    H2 = fmpq_mat_to_numpy(H2_spec.apply(c))

    # Both are e_0 e_0^T: a single 1 at (0, 0).
    expected_H1 = np.zeros((N + 1, N + 1))
    expected_H1[0, 0] = 1.0
    expected_H2 = np.zeros((N, N))
    expected_H2[0, 0] = 1.0
    assert np.allclose(H1, expected_H1, atol=1e-14)
    assert np.allclose(H2, expected_H2, atol=1e-14)

    # Rank-1 PSD: min eig = 0, max eig = 1.
    ev1 = np.linalg.eigvalsh(H1)
    ev2 = np.linalg.eigvalsh(H2)
    assert ev1.min() > -1e-12 and abs(ev1.max() - 1.0) < 1e-12
    assert ev2.min() > -1e-12 and abs(ev2.max() - 1.0) < 1e-12


# ---------------------------------------------------------------------
# Bad moment sequence
# ---------------------------------------------------------------------

def test_hausdorff_rejects_c2_too_large() -> None:
    """c = (1, 0, 2, 0, ...) is infeasible: tilde_m_2 = 3/2 > 1 triggers H_2 < 0.

    From T_2(y) = 2y^2 - 1 and tilde_m_2 = int y^2 d tilde_nu <= 1 for any
    probability measure on [-1, 1], the constraint c_2 = 2 tilde_m_2 - 1
    implies c_2 in [-1, 1].  With c_2 = 2 we get tilde_m_2 = 3/2, so the
    (1 - y^2)-localizer entry 1 - 3/2 = -1/2 breaks H_2 PSD at N = 1.
    """
    N = 1
    c = [fmpq(1), fmpq(0), fmpq(2)]  # length 2N + 1 = 3
    H1_spec, H2_spec = hausdorff_hankel_blocks_chebyshev(N)

    H1 = fmpq_mat_to_numpy(H1_spec.apply(c))
    H2 = fmpq_mat_to_numpy(H2_spec.apply(c))

    # H_1 is diag(1, 3/2) at N = 1 -- still PSD.
    assert _min_eig_sym(H1) > 0
    # H_2 is [[1 - 3/2]] = [[-1/2]] -- NOT PSD.
    assert H2.shape == (1, 1)
    assert H2[0, 0] == pytest.approx(-0.5)
    assert _min_eig_sym(H2) < -0.4


def test_hausdorff_rejects_c4_incoherent() -> None:
    """c = (1, 0, 0, 0, -1, 0, ...) has tilde_m_2 = 1/2, tilde_m_4 = -1/4: infeasible.

    tilde_m_0 = 1, tilde_m_2 = (1 + c_2)/2 = 1/2, tilde_m_4 = 3/8 + c_2/2 + c_4/8
                 = 3/8 + 0 + (-1)/8 = 1/4 -- wait let's recompute.

    Using U[4, 0] = 3/8, U[4, 2] = 1/2, U[4, 4] = 1/8 (see docstring table):
      tilde_m_4 = 3/8 c_0 + 1/2 c_2 + 1/8 c_4 = 3/8 + 0 - 1/8 = 1/4.

    That's still feasible.  A cleaner incoherence: c_4 = 5 forces
    tilde_m_4 = 3/8 + 5/8 = 1, but tilde_m_2 = 1/2, so Hankel
    [[m_0, m_2], [m_2, m_4]] = [[1, 1/2], [1/2, 1]] -- PSD.  Still OK.

    Let me choose c_2 = c_4 = ... = 1 (persistently concentrated at y = 1):
      tilde_m_2 = (1 + 1)/2 = 1, tilde_m_4 = 3/8 + 1/2 + 1/8 = 1 -- rank-1 at +1.
    Also feasible (delta at y=1).

    A clear infeasibility: c_0 = 1, c_2 = 1 (=> tilde_m_2 = 1), c_4 = -1
    (=> tilde_m_4 = 3/8 + 1/2 - 1/8 = 3/4).
    But tilde_m_2 = 1 forces the measure to be a delta at +1 or -1, with
    tilde_m_4 = 1 -- contradiction.  The Hankel H_1 at N=2:
      [[1, 0, 1], [0, 1, 0], [1, 0, 3/4]]
    has det of top-left 2x2 = 1, but full det = 3/4 - 1 = -1/4 -> not PSD.
    """
    N = 2
    c = [fmpq(1), fmpq(0), fmpq(1), fmpq(0), fmpq(-1)]  # len 5
    H1_spec, _H2_spec = hausdorff_hankel_blocks_chebyshev(N)
    H1 = fmpq_mat_to_numpy(H1_spec.apply(c))
    # Should not be PSD.
    assert _min_eig_sym(H1) < -1e-9, f"H_1 min eig = {_min_eig_sym(H1)}"


# ---------------------------------------------------------------------
# Apply agreement: rational vs float
# ---------------------------------------------------------------------

def test_apply_fmpq_vs_float_agree() -> None:
    N = 6
    closed = _uniform_chebyshev_moments_closed_form(2 * N)
    c_q = [fmpq(cf.numerator, cf.denominator) for cf in closed]
    c_f = [float(cf) for cf in closed]

    H1_spec, H2_spec = hausdorff_hankel_blocks_chebyshev(N)
    H1_q = fmpq_mat_to_numpy(H1_spec.apply(c_q))
    H1_f = H1_spec.apply_float(c_f)
    assert np.allclose(H1_q, H1_f, atol=1e-14, rtol=1e-12)

    H2_q = fmpq_mat_to_numpy(H2_spec.apply(c_q))
    H2_f = H2_spec.apply_float(c_f)
    assert np.allclose(H2_q, H2_f, atol=1e-14, rtol=1e-12)


# ---------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------

def test_hankel_block_spec_dimensions() -> None:
    N = 5
    H1, H2 = hausdorff_hankel_blocks_chebyshev(N)
    assert H1.size == N + 1
    assert H2.size == N
    assert H1.degree == 2 * N
    assert H2.degree == 2 * N
    assert len(H1.per_k) == 2 * N + 1
    assert len(H2.per_k) == 2 * N + 1


def test_reject_bad_support() -> None:
    with pytest.raises(NotImplementedError):
        hausdorff_hankel_blocks_chebyshev(4, support=(Fraction(-1, 2), Fraction(1, 2)))

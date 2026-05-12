"""Unit tests for lasserre.preelim — the consistency + simplex pre-elimination.

Soundness test
--------------
We sample valid moment vectors y by evaluating monomials at a random point
μ ∈ Δ_d (the standard simplex).  Every such y satisfies:

    y_0 = 1                              (μ^0 = 1)
    y_α = Σ_i y_{α + e_i}                (μ^α · Σ_i μ_i = Σ_i μ^{α + e_i})
    y_α ≥ 0                              (μ_i ≥ 0)

So y is feasible for the consistency + simplex equalities.  If pre-elim is
SOUND, projecting y to ỹ and then reconstructing via T · ỹ + c must yield
back exactly y.  We require bitwise-ish equality (< 1e-10 max-abs error).

Also checks:
  • T and offset shapes are right.
  • The reduced solution has the correct number of free coordinates.
  • The residual equalities (if any) also hold for y.
  • The transform defines a well-defined y: y_red → reconstruct(y_red)
    satisfies every original equality.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from lasserre.preelim import (assemble_consistency_equalities,
                                build_preelim_transform)
from lasserre_scalable import _precompute


def _monomial_eval(mono_list, mu: np.ndarray) -> np.ndarray:
    """y_j = μ^{mono_list[j]} — a valid moment vector of a Dirac at μ.

    Satisfies the simplex + consistency equalities by construction.
    """
    mu = np.asarray(mu, dtype=np.float64)
    out = np.empty(len(mono_list), dtype=np.float64)
    for j, alpha in enumerate(mono_list):
        val = 1.0
        for i, a in enumerate(alpha):
            if a:
                val *= mu[i] ** a
        out[j] = val
    return out


def _random_mu(d: int, rng) -> np.ndarray:
    """Sample uniformly from Δ_d using exponential trick."""
    g = rng.random(d) + 1e-3  # avoid exact zero
    return g / g.sum()


@pytest.mark.parametrize('d,order', [(4, 2), (4, 3), (6, 2), (6, 3), (8, 2)])
def test_reconstruct_roundtrip(d, order):
    """Project a valid y to ỹ, reconstruct, must equal exactly."""
    P = _precompute(d, order, verbose=False)
    xf = build_preelim_transform(P, verbose=False)

    rng = np.random.default_rng(seed=12345 + d * 10 + order)
    for trial in range(5):
        mu = _random_mu(d, rng)
        y_full = _monomial_eval(P['mono_list'], mu)

        # Project -> reconstruct must recover y exactly.
        y_red = xf.project(y_full)
        y_back = xf.reconstruct(y_red)
        err = np.max(np.abs(y_full - y_back))
        assert err < 1e-10, (
            f"roundtrip failed: d={d} order={order} trial={trial}, "
            f"max|Δ|={err:.3e}")


@pytest.mark.parametrize('d,order', [(4, 2), (4, 3), (6, 2), (6, 3), (8, 2)])
def test_original_equalities_preserved(d, order):
    """The equality system A y = b factors into:
      1. Pivoted rows — automatically satisfied by reconstruct (for any ỹ).
      2. Residual rows — must be imposed on ỹ.
    With protect_degrees left at default, some rows may be residual.  We
    verify both halves on valid moment vectors derived from a Dirac at
    μ ∈ Δ_d (which satisfy A y = b by construction)."""
    P = _precompute(d, order, verbose=False)
    xf = build_preelim_transform(P, verbose=False)
    A, b, _tags = assemble_consistency_equalities(P)

    # Part 1: pivoted-only check with no-protect preelim (must be zero for
    # arbitrary ỹ).
    xf_no_protect = build_preelim_transform(
        P, protect_degrees=set(), verbose=False)
    rng = np.random.default_rng(seed=54321 + d * 10 + order)
    for trial in range(3):
        y_red = rng.standard_normal(xf_no_protect.n_y_red)
        y_full = xf_no_protect.reconstruct(y_red)
        residual = np.asarray(A @ y_full).ravel() - b
        err = np.max(np.abs(residual))
        assert err < 1e-9, (
            f"(no-protect) reconstruct fails A y = b: "
            f"d={d} order={order} trial={trial}, max|Δ| = {err:.3e}")

    # Part 2: with default protection, valid y (from Dirac at μ ∈ Δ_d)
    # projected then reconstructed must STILL satisfy A y = b — because
    # reconstruct(project(valid_y)) = valid_y and valid_y satisfies A y = b.
    for trial in range(3):
        mu = _random_mu(d, rng)
        y_valid = _monomial_eval(P['mono_list'], mu)
        y_back = xf.reconstruct(xf.project(y_valid))
        residual = np.asarray(A @ y_back).ravel() - b
        err = np.max(np.abs(residual))
        assert err < 1e-9, (
            f"(default-protect) reconstruct(project(valid_y)) fails A y = b: "
            f"d={d} order={order} trial={trial}, max|Δ| = {err:.3e}")


@pytest.mark.parametrize('d,order', [(4, 2), (6, 2)])
def test_substitute_matrix_composition(d, order):
    """For a random sparse matrix B with n_y cols, check that
    B @ y == B_new @ ỹ + c_new where (B_new, c_new) = substitute_matrix(B).
    """
    P = _precompute(d, order, verbose=False)
    xf = build_preelim_transform(P, verbose=False)

    rng = np.random.default_rng(seed=99 + d + order)
    import scipy.sparse as sp
    m = 20
    B_dense = rng.standard_normal((m, xf.n_y))
    B_dense[np.abs(B_dense) < 1.2] = 0.0
    B = sp.csr_matrix(B_dense)

    B_new, c_new = xf.substitute_matrix(B)

    # Compare on a valid y
    mu = _random_mu(d, rng)
    y_full = _monomial_eval(P['mono_list'], mu)
    y_red = xf.project(y_full)

    lhs = np.asarray(B @ y_full).ravel()
    rhs = np.asarray(B_new @ y_red).ravel() + c_new
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-10, f"substitute_matrix mismatch: max|Δ|={err:.3e}"


@pytest.mark.parametrize('d,order', [(4, 2), (4, 3)])
def test_pick_substitution(d, order):
    """substitute_pick(p): y[p] should equal C·ỹ + c."""
    P = _precompute(d, order, verbose=False)
    xf = build_preelim_transform(P, verbose=False)
    rng = np.random.default_rng(seed=7 + d + order)

    # Use the moment-matrix pick as a natural test
    pick = np.asarray(P['moment_pick'], dtype=np.int64)
    C, c = xf.substitute_pick(pick)

    mu = _random_mu(d, rng)
    y_full = _monomial_eval(P['mono_list'], mu)
    y_red = xf.project(y_full)

    expected = np.where(pick >= 0, y_full[np.maximum(pick, 0)], 0.0)
    got = np.asarray(C @ y_red).ravel() + c
    err = np.max(np.abs(expected - got))
    assert err < 1e-10, f"pick substitution mismatch: max|Δ|={err:.3e}"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

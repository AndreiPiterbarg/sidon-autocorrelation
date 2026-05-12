"""C.2 tests for `chebyshev_dual.toeplitz_dual`.

Covers:
  - Trace map  r_l = <S_l, Q>  with S_0 = I and S_l = (J_l + J_l^T)/2.
  - Rank-1 Q = q q^T from q = (1, 1, ..., 1)/sqrt(D+1) is the Fejer kernel,
    giving r_l = 1 - l/(D+1); feasibility confirmed.
  - r = (1, 1) is REJECTED (spec counter-example to the Caratheodory cone).
  - Round-trip Q -> r -> feasibility-recovery.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from flint import fmpq, fmpq_mat

from chebyshev_dual.toeplitz_dual import (
    apply_trace_maps,
    fejer_riesz_factorize,
    fejer_riesz_trace_maps,
    is_fejer_riesz_feasible_numeric,
    trace_maps_as_numpy,
)


def _f(q):
    """Coerce fmpq / fmpz / int / float to float (python-flint 0.8 has no float(fmpq))."""
    if isinstance(q, fmpq):
        return float(int(q.p)) / float(int(q.q))
    return float(q)


# ---------------------------------------------------------------------
# Trace maps
# ---------------------------------------------------------------------

def test_S_0_is_identity() -> None:
    for D in [0, 1, 2, 5, 10]:
        S = fejer_riesz_trace_maps(D)[0]
        for i in range(D + 1):
            for j in range(D + 1):
                expected = fmpq(1) if i == j else fmpq(0)
                assert S[i, j] == expected


def test_S_l_symmetric_one_half_offdiagonal() -> None:
    D = 5
    maps = fejer_riesz_trace_maps(D)
    for l in range(1, D + 1):
        S = maps[l]
        # Symmetric
        for i in range(D + 1):
            for j in range(D + 1):
                assert S[i, j] == S[j, i], f"S_{l} not symmetric at ({i},{j})"
        # Entry (i, j) is 1/2 iff |i - j| == l AND both indices in range
        for i in range(D + 1):
            for j in range(D + 1):
                if abs(i - j) == l:
                    assert S[i, j] == fmpq(1, 2), f"S_{l}[{i},{j}] wrong"
                else:
                    assert S[i, j] == fmpq(0)


# ---------------------------------------------------------------------
# Apply trace maps
# ---------------------------------------------------------------------

def test_apply_trace_maps_identity_Q_gives_r0_is_trace() -> None:
    D = 4
    Q = fmpq_mat(D + 1, D + 1)
    for k in range(D + 1):
        Q[k, k] = fmpq(1)
    r = apply_trace_maps(Q, D=D)
    assert r[0] == fmpq(D + 1)  # tr(I_{D+1}) = D+1
    for l in range(1, D + 1):
        assert r[l] == fmpq(0)


def test_apply_trace_maps_rank1_all_ones_gives_fejer_kernel() -> None:
    """q = (1, 1, ..., 1), Q = q q^T:  r_l = (D+1-l).

    Normalizing by tr(Q) = D+1 gives the Fejer kernel  r_l = 1 - l/(D+1).
    """
    D = 6
    Q = fmpq_mat(D + 1, D + 1)
    for i in range(D + 1):
        for j in range(D + 1):
            Q[i, j] = fmpq(1)
    r = apply_trace_maps(Q, D=D)
    for l in range(D + 1):
        assert r[l] == fmpq(D + 1 - l), f"r_{l} = {r[l]}, want {D + 1 - l}"


def test_trace_maps_as_numpy_matches_fmpq_version() -> None:
    D = 5
    maps_q = fejer_riesz_trace_maps(D)
    maps_np = trace_maps_as_numpy(D)
    for l in range(D + 1):
        for i in range(D + 1):
            for j in range(D + 1):
                sij = maps_q[l][i, j]
                expected = float(int(sij.p)) / float(int(sij.q))
                assert maps_np[l, i, j] == expected


# ---------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------

def test_feasibility_fejer_kernel_accepted() -> None:
    """Fejer kernel r_l = 1 - l/(D+1) is feasible (certified PSD Q from q=1/sqrt(D+1)*ones)."""
    for D in [2, 4, 8]:
        r = [1.0 - l / (D + 1) for l in range(D + 1)]
        ok, Q = is_fejer_riesz_feasible_numeric(r, tol=1e-7)
        assert ok, f"D={D}: Fejer kernel should be feasible"
        # Double-check: r recovered from returned Q.
        r_back = [_f(v) for v in apply_trace_maps(Q, D=D)]
        for l in range(D + 1):
            assert abs(r_back[l] - r[l]) < 1e-6, (
                f"D={D}, l={l}: r_back = {r_back[l]}, want {r[l]}"
            )


def test_feasibility_rejects_caratheodory_only_sequence() -> None:
    """r = (1, 1) is PSD-Toeplitz (Caratheodory) but NOT Fejer-Riesz feasible,
    because p(t) = 1 + 2 cos(2 pi t) is negative at t = 1/2."""
    r = [1.0, 1.0]
    ok, Q = is_fejer_riesz_feasible_numeric(r, tol=1e-7)
    assert not ok, "r = (1, 1) should be rejected by the Fejer-Riesz SOS cone"


def test_feasibility_rejects_negative_r0() -> None:
    """r_0 < 0 violates tr(Q) = r_0 >= 0 for PSD Q."""
    r = [-0.5, 0.1]
    ok, _ = is_fejer_riesz_feasible_numeric(r, tol=1e-7)
    assert not ok


def test_feasibility_accepts_cos_plus_one() -> None:
    """p(t) = 1 + cos(2 pi t) = (1/2)|1 + e^{2 pi i t}|^2 >= 0;
    coefficients (r_0, r_1) = (1, 1/2)."""
    r = [1.0, 0.5]
    ok, Q = is_fejer_riesz_feasible_numeric(r, tol=1e-7)
    assert ok
    # The closed-form Q is [[1/2, 1/2], [1/2, 1/2]] (rank 1).
    r_back = [_f(v) for v in apply_trace_maps(Q, D=1)]
    assert abs(r_back[0] - 1.0) < 1e-6
    assert abs(r_back[1] - 0.5) < 1e-6


# ---------------------------------------------------------------------
# Spectral factorization
# ---------------------------------------------------------------------

def test_factorize_rank_one_Q() -> None:
    """q_cols recovered from Q = (1, 1, 1) (1, 1, 1)^T agrees with q up to sign."""
    q_true = np.array([1.0, 2.0, 0.5, -1.5])
    Q = np.outer(q_true, q_true)
    fact = fejer_riesz_factorize(Q)
    assert fact.q_columns.shape == (4, 1)
    q_rec = fact.q_columns[:, 0]
    # agreement up to sign
    if np.dot(q_rec, q_true) < 0:
        q_rec = -q_rec
    assert np.allclose(q_rec, q_true, atol=1e-9)
    assert fact.residual < 1e-12


def test_factorize_preserves_trace_maps() -> None:
    """Random PSD Q factorized as sum q_k q_k^T gives back same r through apply_trace_maps."""
    rng = np.random.default_rng(7)
    D = 5
    A = rng.standard_normal((D + 1, D + 1))
    Q = A @ A.T  # PSD
    r_from_Q = [_f(v) for v in apply_trace_maps(Q, D=D)]
    fact = fejer_riesz_factorize(Q)
    # Reconstruct from q_cols
    Q_rec = fact.q_columns @ fact.q_columns.T
    r_from_rec = [_f(v) for v in apply_trace_maps(Q_rec, D=D)]
    for l in range(D + 1):
        assert abs(r_from_Q[l] - r_from_rec[l]) < 1e-9

"""C.4 tests for `chebyshev_dual.sdp_driver`.

End-to-end SDP solve checks:
  - Small-case (N, D) solve reaches an `optimal` / `optimal_inaccurate` status and
    returns a valid tau, r, Q, Y1, Y2 with the diagnosed eigenvalues non-negative
    (up to solver slack).
  - tau >= 1.0 at all tested (N, D), since the trivial measure r = (1, 0, ..., 0)
    is always feasible and gives  int (f*f) = 1.  (The bound is weak at small D.)
  - tr(Q) = 1 to solver tolerance (the r_0 = 1 normalization).
  - The SDP tensors agree with the phase-1/2 building blocks on a spot check.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("cvxpy")
from chebyshev_dual.sdp_driver import (
    SDPCoefficientTensors,
    SolveResult,
    build_sdp_tensors,
    solve_chebyshev_dual_sdp,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# Tensor spot checks
# ---------------------------------------------------------------------

def test_build_sdp_tensors_shape_and_symmetry() -> None:
    t = build_sdp_tensors(N=4, D=3, prec=128)
    assert t.n == 9
    assert t.A_basis_np.shape == (4, 9, 9)
    assert t.A1_per_k.shape == (9, 5, 5)
    assert t.A2_per_k.shape == (9, 4, 4)
    assert t.S_tensor.shape == (4, 4, 4)
    # Symmetry of A_basis matrices
    for l in range(4):
        assert np.allclose(t.A_basis_np[l], t.A_basis_np[l].T, atol=1e-12)
    # A_0 = e_0 e_0^T
    E00 = np.zeros((9, 9))
    E00[0, 0] = 1.0
    assert np.allclose(t.A_basis_np[0], E00, atol=1e-14)
    # S_0 = I
    assert np.allclose(t.S_tensor[0], np.eye(4), atol=1e-14)


# ---------------------------------------------------------------------
# End-to-end solve
# ---------------------------------------------------------------------

@pytest.mark.parametrize("N,D", [(4, 2), (6, 3), (8, 4)])
def test_sdp_solve_small_reaches_optimum_and_bounds_above_one(N: int, D: int) -> None:
    res = solve_chebyshev_dual_sdp(N=N, D=D, solver="CLARABEL")
    assert res.status in ("optimal", "optimal_inaccurate")
    # Trivial bound: tau >= 1 is always feasible via r = (1, 0, ..., 0) and Y_i = 0.
    # Numerical solver returns  ~= 1  within a few ulps; allow slack.
    assert res.tau >= 1.0 - 1e-4, f"tau = {res.tau} below trivial bound"
    # Sanity on dimensions
    assert res.r.shape == (D + 1,)
    assert res.Q.shape == (D + 1, D + 1)
    assert res.Y1.shape == (N + 1, N + 1)
    if N >= 1:
        assert res.Y2.shape == (N, N)
    # Normalization  tr(Q) = 1
    assert abs(float(np.trace(res.Q)) - 1.0) < 1e-5
    # Primal eigenvalue slacks (PSD cones): should be approximately non-negative.
    # CLARABEL returns optimal_inaccurate with ~1e-5 primal eig slop; accept it.
    # The certified pipeline (phase 5) performs a rigorous PSD shift.
    assert res.Q_min_eig > -1e-4
    assert res.Y1_min_eig > -1e-4
    if N >= 1:
        assert res.Y2_min_eig > -1e-4
    # Truncation bound must be non-negative and small at these sizes.
    assert res.truncation_bound >= 0.0
    assert res.truncation_bound < 1e-4


def test_sdp_solve_returns_fejer_riesz_feasible_r() -> None:
    """The returned r must be in the Fejer-Riesz SOS cone, i.e., there exists
    a PSD Q* with the reported r.  Check by re-evaluating the trace maps on the
    returned Q*."""
    from chebyshev_dual.toeplitz_dual import apply_trace_maps
    from flint import fmpq

    res = solve_chebyshev_dual_sdp(N=6, D=3, solver="CLARABEL")
    r_from_Q = apply_trace_maps(res.Q, D=3)
    r_from_Q_float = np.array(
        [float(int(v.p)) / float(int(v.q)) for v in r_from_Q]
    )
    # Agreement to solver tolerance
    assert np.allclose(r_from_Q_float, res.r, atol=1e-4)


def test_reuse_tensors_across_solves() -> None:
    """Tensors can be built once and reused for repeated solves (e.g., in a warm
    start context).  Verify reproducibility."""
    tensors = build_sdp_tensors(N=4, D=2, prec=128)
    res_a = solve_chebyshev_dual_sdp(N=4, D=2, tensors=tensors)
    res_b = solve_chebyshev_dual_sdp(N=4, D=2, tensors=tensors)
    assert abs(res_a.tau - res_b.tau) < 1e-6

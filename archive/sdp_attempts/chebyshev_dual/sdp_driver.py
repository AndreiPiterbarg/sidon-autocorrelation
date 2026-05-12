"""Single-SDP solver for the Chebyshev-dual Lasserre relaxation of C_{1a}.

SDP formulation
===============

We maximize a certified lower bound  tau <= C_{1a}  via the combined SDP
(the SOS dual of the Shor-lifted primal  min tr(A(r) Z)  over  Z >> 0,
H_i(Z e_0) >> 0,  Z_{00} = 1, coupled with  r in Fejer-Riesz cone,  r_0 = 1):

    max  tau
    s.t.  A(r)  -  tau * E_00  -  (1/2) ( v_1(Y_1) e_0^T + e_0 v_1(Y_1)^T )
                                -  (1/2) ( v_2(Y_2) e_0^T + e_0 v_2(Y_2)^T )  >>  0
          Q >> 0,   tr(Q) = 1,         r_l = <S_l, Q>      (Fejer-Riesz cone)
          Y_1 >> 0   (size N+1),       Y_2 >> 0   (size N)

Here
  * A(r) = sum_{l=0}^{D} r_l A_l   is the Jacobi-Anger kernel matrix
    (A_0 = e_0 e_0^T;  A_l = 2 (alpha_l alpha_l^T - beta_l beta_l^T) for l >= 1).
  * v_i(Y_i) in R^{2N+1}, [v_i(Y_i)]_k = < Y_i, A_i^{(k)} >, is the adjoint of the
    Hankel map c |-> H_i(c) = sum_k c_k A_i^{(k)}.
  * E_00, e_0 are the standard (2N+1)-dim anchors at the c_0 = 1 slot.

All constant coefficients are built in exact rationals / arbs upstream, then
collapsed to float64 midpoints for the solver.  The certified pipeline
(Phase 5) re-checks the solve in fmpq + arb arithmetic.

Spec deviation (carried from earlier phases)
--------------------------------------------
The top-level prompt writes the collapse as "A(r) - tau E_00 - L_1^*(Y_1) - L_2^*(Y_2) = 0"
as a matrix equation.  That is type-inconsistent (L_i^* produces vectors in
R^{2N+1}, not matrices).  The correct Lasserre-style SOS dual is the LMI above
with symmetrized rank-<=2 corrections  (v e_0^T + e_0 v^T)/2.  We also use the
Fejer-Riesz SOS cone (Q >> 0 + trace maps), not the spec's Caratheodory
Toeplitz R(r) >> 0 (which would make the bound unsound).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np

from chebyshev_dual.bilinear_kernel import (
    JacobiAngerTables,
    arb_mat_to_numpy_mid,
    build_jacobi_anger_tables,
    kernel_truncation_error_bound,
)
from chebyshev_dual.cheb_hankel import (
    HankelBlockSpec,
    fmpq_mat_to_numpy,
    hausdorff_hankel_blocks_chebyshev,
)
from chebyshev_dual.toeplitz_dual import trace_maps_as_numpy


# ---------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------

@dataclass
class SolveResult:
    """Output of `solve_chebyshev_dual_sdp`."""
    N: int
    D: int
    tau: float
    r: np.ndarray            # shape (D+1,)
    Q: np.ndarray            # shape (D+1, D+1)
    Y1: np.ndarray           # shape (N+1, N+1)
    Y2: np.ndarray           # shape (N, N)     (empty when N == 0)
    truncation_bound: float  # arb-upper-bound on |eps_trunc| at r = r*
    status: str
    solve_time_s: float
    solver: str
    # Derived diagnostics:
    lmi_residual_fro: float  # ||A(r) - tau E00 - 1/2 corrections||_F
    Q_min_eig: float
    Y1_min_eig: float
    Y2_min_eig: float
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Precomputation (float materialization of coefficient tensors)
# ---------------------------------------------------------------------

@dataclass
class SDPCoefficientTensors:
    """Float64 views of the constant coefficient tensors used in the SDP."""
    N: int
    D: int
    n: int                    # = 2N + 1  (side of A(r))
    A_basis_np: np.ndarray    # (D+1, n, n): A_basis_np[l] = A_l (float midpoint)
    A1_per_k: np.ndarray      # (n, N+1, N+1): per-c_k matrix for H_1
    A2_per_k: np.ndarray      # (n, N, N):     per-c_k matrix for H_2
    S_tensor: np.ndarray      # (D+1, D+1, D+1): Fejer-Riesz trace maps
    truncation_pointwise: np.ndarray  # (D+1,) arb upper bounds, l-th entry is
                                      # the sup|R^(c)_l| + sup|R^(s)_l| bound
    tables: JacobiAngerTables


def build_sdp_tensors(N: int, D: int, prec: int = 128) -> SDPCoefficientTensors:
    """Materialize all constant data for the SDP at (N, D), at arb precision `prec`."""
    tables = build_jacobi_anger_tables(N=N, D=D, prec=prec)
    H1_spec, H2_spec = hausdorff_hankel_blocks_chebyshev(N)

    n = 2 * N + 1
    A_basis_np = np.zeros((D + 1, n, n), dtype=np.float64)
    for l in range(D + 1):
        A_basis_np[l] = arb_mat_to_numpy_mid(tables.A_basis[l])
        # Symmetrize against midpoint-roundoff asymmetry.
        A_basis_np[l] = 0.5 * (A_basis_np[l] + A_basis_np[l].T)

    A1_per_k = np.zeros((n, N + 1, N + 1), dtype=np.float64)
    for k in range(n):
        A1_per_k[k] = fmpq_mat_to_numpy(H1_spec.per_k[k])
        A1_per_k[k] = 0.5 * (A1_per_k[k] + A1_per_k[k].T)

    h2 = max(N, 0)
    A2_per_k = np.zeros((n, h2, h2), dtype=np.float64)
    for k in range(n):
        A2_per_k[k] = fmpq_mat_to_numpy(H2_spec.per_k[k])
        A2_per_k[k] = 0.5 * (A2_per_k[k] + A2_per_k[k].T)

    S_tensor = trace_maps_as_numpy(D)  # (D+1, D+1, D+1)

    # Per-l pointwise truncation (cos + sin tails).  Used only for reporting
    # and for building the scalar truncation_bound at r = r*.
    from chebyshev_dual.bilinear_kernel import jacobi_anger_truncation_infty
    trunc = np.zeros(D + 1, dtype=np.float64)
    for l in range(D + 1):
        if l == 0:
            trunc[l] = 0.0
        else:
            trunc[l] = float(jacobi_anger_truncation_infty(l, 2 * N).upper())

    return SDPCoefficientTensors(
        N=N, D=D, n=n,
        A_basis_np=A_basis_np,
        A1_per_k=A1_per_k,
        A2_per_k=A2_per_k,
        S_tensor=S_tensor,
        truncation_pointwise=trunc,
        tables=tables,
    )


# ---------------------------------------------------------------------
# The SDP
# ---------------------------------------------------------------------

def solve_chebyshev_dual_sdp(
    N: int,
    D: int,
    *,
    solver: str = "CLARABEL",
    prec: int = 128,
    verbose: bool = False,
    tensors: Optional[SDPCoefficientTensors] = None,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> SolveResult:
    """Assemble and solve the combined SDP.  Returns the optimal tau and
    primal/dual variables (Y_1, Y_2, Q, r).  Caller is responsible for any
    downstream rigorous certification (Phase 5)."""
    import cvxpy as cp

    if tensors is None:
        tensors = build_sdp_tensors(N=N, D=D, prec=prec)
    assert tensors.N == N and tensors.D == D

    n = tensors.n
    # --- Variables ------------------------------------------------------
    tau = cp.Variable()
    Q = cp.Variable((D + 1, D + 1), symmetric=True)
    Y1 = cp.Variable((N + 1, N + 1), symmetric=True)
    Y2: Optional[cp.Variable]
    if N >= 1:
        Y2 = cp.Variable((N, N), symmetric=True)
    else:
        Y2 = None

    # --- r(Q) -----------------------------------------------------------
    # r_l = sum_{i, j} S_tensor[l, i, j] Q[i, j]
    r_exprs = [cp.sum(cp.multiply(tensors.S_tensor[l], Q)) for l in range(D + 1)]

    # --- A(r) -----------------------------------------------------------
    # A(r) = sum_l r_l * A_basis[l]
    A_of_r = sum(r_exprs[l] * tensors.A_basis_np[l] for l in range(D + 1))

    # --- v_1(Y_1), v_2(Y_2): vectors in R^n -----------------------------
    # v_i[k] = <Y_i, A_i^{(k)}>
    v1_entries = [cp.sum(cp.multiply(tensors.A1_per_k[k], Y1)) for k in range(n)]
    v1 = cp.reshape(cp.hstack(v1_entries), (n, 1), order="C")

    if Y2 is not None:
        v2_entries = [cp.sum(cp.multiply(tensors.A2_per_k[k], Y2)) for k in range(n)]
        v2 = cp.reshape(cp.hstack(v2_entries), (n, 1), order="C")
    else:
        v2 = None

    # --- Correction and LMI ---------------------------------------------
    e0_col = np.zeros((n, 1))
    e0_col[0, 0] = 1.0
    E00 = np.zeros((n, n))
    E00[0, 0] = 1.0

    corr = 0.5 * (v1 @ e0_col.T + e0_col @ v1.T)
    if v2 is not None:
        corr = corr + 0.5 * (v2 @ e0_col.T + e0_col @ v2.T)

    LMI = A_of_r - tau * E00 - corr

    constraints: List[Any] = [
        Q >> 0,
        Y1 >> 0,
        cp.trace(Q) == 1.0,
        LMI >> 0,
    ]
    if Y2 is not None:
        constraints.append(Y2 >> 0)

    problem = cp.Problem(cp.Maximize(tau), constraints)

    # --- Solve ----------------------------------------------------------
    t0 = time.time()
    opts: Dict[str, Any] = dict(verbose=verbose)
    if solver_opts:
        opts.update(solver_opts)
    try:
        problem.solve(solver=solver, **opts)
    except Exception as exc:
        raise RuntimeError(f"SDP solve failed: {exc}") from exc
    solve_time = time.time() - t0

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"SDP did not reach optimum (status={problem.status}, "
            f"solver={solver}, N={N}, D={D})"
        )

    # --- Extract primal ---------------------------------------------------
    tau_opt = float(tau.value)
    Q_opt = np.asarray(Q.value, dtype=np.float64)
    Y1_opt = np.asarray(Y1.value, dtype=np.float64)
    Y2_opt = (
        np.asarray(Y2.value, dtype=np.float64)
        if Y2 is not None else np.zeros((0, 0), dtype=np.float64)
    )
    r_opt = np.array([float(e.value) for e in r_exprs], dtype=np.float64)

    # --- Diagnostics ------------------------------------------------------
    # Recompute correction and LMI residual directly in numpy for sanity.
    v1_val = np.array([
        float(np.sum(tensors.A1_per_k[k] * Y1_opt)) for k in range(n)
    ]).reshape(n, 1)
    if Y2 is not None:
        v2_val = np.array([
            float(np.sum(tensors.A2_per_k[k] * Y2_opt)) for k in range(n)
        ]).reshape(n, 1)
    else:
        v2_val = np.zeros((n, 1))
    corr_val = 0.5 * (v1_val @ e0_col.T + e0_col @ v1_val.T
                      + v2_val @ e0_col.T + e0_col @ v2_val.T)
    A_r_val = sum(r_opt[l] * tensors.A_basis_np[l] for l in range(D + 1))
    LMI_val = A_r_val - tau_opt * E00 - corr_val
    LMI_val = 0.5 * (LMI_val + LMI_val.T)
    lmi_fro = float(np.linalg.norm(LMI_val, ord="fro"))
    lmi_min_eig = float(np.linalg.eigvalsh(LMI_val).min())

    # Evaluate truncation error at r_opt (using arb-certified per-l bounds).
    from flint import arb
    r_arb = [arb(float(r_opt[l])) for l in range(D + 1)]
    trunc_arb = kernel_truncation_error_bound(r_arb, N, D)
    trunc_upper = float(trunc_arb.upper())

    Q_eig = float(np.linalg.eigvalsh(0.5 * (Q_opt + Q_opt.T)).min())
    Y1_eig = float(np.linalg.eigvalsh(0.5 * (Y1_opt + Y1_opt.T)).min())
    Y2_eig = (
        float(np.linalg.eigvalsh(0.5 * (Y2_opt + Y2_opt.T)).min())
        if Y2_opt.size > 0 else 0.0
    )

    return SolveResult(
        N=N, D=D,
        tau=tau_opt,
        r=r_opt, Q=Q_opt, Y1=Y1_opt, Y2=Y2_opt,
        truncation_bound=trunc_upper,
        status=problem.status,
        solve_time_s=solve_time,
        solver=solver,
        lmi_residual_fro=lmi_fro,
        Q_min_eig=Q_eig,
        Y1_min_eig=Y1_eig,
        Y2_min_eig=Y2_eig,
        extra={
            "lmi_min_eig": lmi_min_eig,
            "problem_value": float(problem.value),
        },
    )


__all__ = [
    "SolveResult",
    "SDPCoefficientTensors",
    "build_sdp_tensors",
    "solve_chebyshev_dual_sdp",
]

"""Quadratic-programming optimiser for the Matolcsi-Vinuesa multiplier.

For a Bochner-admissible kernel ``K``, the trigonometric polynomial

    ``G(x) = sum_{j=1}^{N} a_j cos(2 pi j x / u)``

is selected to minimise

    ``S_1(a; K) = sum_{j=1}^{N} a_j^2 / w_j(K)``,
    ``w_j(K)   = hat K(j/u)``,

subject to the semi-infinite constraint ``G(x) >= 1`` on ``[0, 1/4]``;
see the writeup, Section 2.  Smaller ``S_1`` yields a larger gain
``a = (4/u) m_G^2 / S_1`` and therefore a stronger lower bound on
``C_{1a}``.

The semi-infinite constraint is discretised to ``n_grid`` evenly spaced
points on ``[0, 1/4]``; the resulting QP is solved with cvxpy
(preferring MOSEK, then CLARABEL, then SCS, then ECOS).  The downstream
pipeline re-establishes rigour by certifying ``min G >= m_G`` over the
true continuum via a Taylor branch-and-bound (``grid_bound.G_min``).

After solving, the optimal float coefficients are rounded to exact
``fmpq`` with a fixed denominator so subsequent rigorous evaluations are
exact-rational.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from flint import arb, fmpq

from .kernels import Kernel


@dataclass
class QPResult:
    """Output of :func:`solve_qp_for_kernel`."""

    a_opt_float: np.ndarray
    a_opt_fmpq: List[fmpq]
    S1_float: float
    min_G_grid_float: float
    solver: str
    status: str
    n: int
    n_grid: int


def solve_qp_for_kernel(
    kernel: Kernel,
    n: int = 200,
    u: fmpq = fmpq(638, 1000),
    n_grid: int = 5001,
    prec_bits_weights: int = 128,
    fmpq_denom: int = 10**8,
    verbose: bool = False,
) -> QPResult:
    """Solve the semi-infinite QP for ``kernel`` and return rounded coeffs.

    Parameters
    ----------
    kernel:
        Bochner-admissible kernel; ``kernel.K_tilde_real`` is queried at
        ``xi = j / u`` for ``j = 1, ..., n``.
    n:
        Number of cosine modes.
    u:
        Period parameter, equal to ``1/2 + delta_1``.
    n_grid:
        Discretisation density on ``[0, 1/4]``.
    prec_bits_weights:
        Precision used when evaluating the QP weights ``w_j``.
    fmpq_denom:
        Denominator for rounding the optimal coefficients to ``fmpq``.
    """
    u_f = float(u.p) / float(u.q)

    # QP weights w_j = hat K(j/u) (arb -> float).
    w = np.zeros(n)
    for j in range(1, n + 1):
        xi = arb(fmpq(j)) / arb(u)
        w_arb = kernel.K_tilde_real(xi, prec_bits=prec_bits_weights)
        w[j - 1] = float(w_arb.mid())

    if w.min() <= 0:
        # A non-positive weight signals a Bochner violation at that
        # frequency.  Clamp to a tiny positive value (heavily penalising
        # that coefficient in the objective) so the QP stays well-posed.
        tiny = 1e-12
        w = np.where(w > tiny, w, tiny)
        if verbose:
            print("  warning: clamped a non-positive QP weight to eps")

    # Constraint matrix: B[i, j-1] = cos(2 pi j x_i / u) for i = 0..n_grid-1.
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, n))
    for j in range(1, n + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)

    import cvxpy as cp

    a_var = cp.Variable(n)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a_var))))
    cons = [B @ a_var >= 1.0]
    prob = cp.Problem(obj, cons)

    solvers: List[tuple] = []
    try:
        import mosek  # noqa: F401

        solvers.append(("MOSEK", {}))
    except ImportError:
        pass
    solvers += [("CLARABEL", {}), ("SCS", {}), ("ECOS", {})]

    final_solver = None
    final_status = None
    failures: List[tuple] = []
    for solver_name, opts in solvers:
        try:
            prob.solve(solver=solver_name, verbose=False, **opts)
        except Exception as exc:
            failures.append((solver_name, str(exc)[:80]))
            continue
        if a_var.value is not None and prob.status in (
            "optimal",
            "optimal_inaccurate",
        ):
            final_solver = solver_name
            final_status = prob.status
            break

    if a_var.value is None:
        raise RuntimeError(
            f"QP failed for kernel {kernel.name}; "
            f"tried {[s for s, _ in solvers]}; errors: {failures}"
        )

    a_opt_float = np.asarray(a_var.value).flatten()
    S1_float = float(np.sum((a_opt_float ** 2) / w))
    min_G_grid = float((B @ a_opt_float).min())

    a_fmpq = [
        fmpq(int(round(a * fmpq_denom)), fmpq_denom) for a in a_opt_float
    ]

    if verbose:
        print(
            f"  {kernel.name}: solver={final_solver} status={final_status} "
            f"S_1={S1_float:.5f} min G on grid={min_G_grid:.6f}"
        )

    return QPResult(
        a_opt_float=a_opt_float,
        a_opt_fmpq=a_fmpq,
        S1_float=S1_float,
        min_G_grid_float=min_G_grid,
        solver=final_solver,
        status=final_status,
        n=n,
        n_grid=n_grid,
    )


__all__ = ["solve_qp_for_kernel", "QPResult"]

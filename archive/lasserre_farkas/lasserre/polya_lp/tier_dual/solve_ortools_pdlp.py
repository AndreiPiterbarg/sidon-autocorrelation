"""Google PDLP (ortools) on the Sidon dual / epigraph LP.

This is the production-grade PDLP — same algorithm class as cuOpt's
GPU PDLP, but on CPU. If it converges on the epigraph dual, the
reformulation path is validated and the only remaining question is
whether GPU acceleration (cuOpt / cuPDLPx) gives extra wall speedup.

API: solve_with_ortools_pdlp(build_eq, build_ub, c, bounds, tol, ...)
The function works on either DualBuildResult or DualEpiBuildResult.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time

import numpy as np
from scipy import sparse as sp

from ortools.pdlp.python import pdlp as _pdlp
from ortools.pdlp import solvers_pb2

from lasserre.polya_lp.tier_dual.solve_dual import DualSolveResult


_INF = np.inf  # ortools recognizes np.inf / -np.inf as "no bound"


def _build_qp_from_lp(
    A_eq: sp.csr_matrix,
    b_eq: np.ndarray,
    A_ub: Optional[sp.csr_matrix],
    b_ub: Optional[np.ndarray],
    c: np.ndarray,
    bounds: list,
) -> _pdlp.QuadraticProgram:
    """Pack our LP into ortools.pdlp's QuadraticProgram (zero quadratic part).

    ortools.pdlp uses the standard form
        min  c^T x  +  0.5 x^T Q x
        s.t. cl <= A x <= cu
             l  <= x  <= u
    Equality is encoded as cl = cu = b. Inequality A_ub x <= b_ub becomes
    cl = -inf, cu = b_ub. Stack them.
    """
    n_vars = c.shape[0]
    if A_ub is not None and A_ub.shape[0] > 0:
        A_full = sp.vstack([A_eq, A_ub], format="csr")
        cl = np.concatenate([b_eq, np.full(A_ub.shape[0], -_INF)])
        cu = np.concatenate([b_eq, b_ub])
    else:
        A_full = A_eq.tocsr()
        cl = b_eq.copy()
        cu = b_eq.copy()

    qp = _pdlp.QuadraticProgram()
    n_rows = A_full.shape[0]
    qp.resize_and_initialize(n_vars, n_rows)
    qp.objective_vector = c.astype(np.float64)

    # Variable bounds
    var_lo = np.empty(n_vars, dtype=np.float64)
    var_hi = np.empty(n_vars, dtype=np.float64)
    for j, (lo, hi) in enumerate(bounds):
        var_lo[j] = -_INF if lo is None else float(lo)
        var_hi[j] = _INF if hi is None else float(hi)
    qp.variable_lower_bounds = var_lo
    qp.variable_upper_bounds = var_hi

    # Constraint bounds
    qp.constraint_lower_bounds = cl.astype(np.float64)
    qp.constraint_upper_bounds = cu.astype(np.float64)

    # Constraint matrix: pdlp wants Eigen sparse via .constraint_matrix
    # The descriptor is "Data descriptor" so likely a setter for scipy/Eigen
    # equivalent. Looking at ortools examples: assign a scipy sparse matrix.
    qp.constraint_matrix = A_full.astype(np.float64)
    return qp


def _make_params(
    tol: float = 1e-6,
    iter_limit: int = 100000,
    time_sec_limit: float = 600.0,
    verbosity: int = 0,
    num_threads: int = 0,   # 0 = auto
    use_feasibility_polishing: bool = True,
) -> "solvers_pb2.PrimalDualHybridGradientParams":
    p = solvers_pb2.PrimalDualHybridGradientParams()
    tc = p.termination_criteria
    tc.eps_optimal_absolute = tol
    tc.eps_optimal_relative = tol
    tc.iteration_limit = iter_limit
    tc.time_sec_limit = time_sec_limit
    # Feasibility polishing requires the legacy primal-residual path; if
    # we want it on, also turn OFF
    # handle_some_primal_gradients_on_finite_bounds_as_residuals.
    if use_feasibility_polishing:
        p.use_feasibility_polishing = True
        p.handle_some_primal_gradients_on_finite_bounds_as_residuals = False
    p.verbosity_level = verbosity
    if num_threads > 0:
        p.num_threads = num_threads
    return p


def solve_dual_ortools_pdlp(
    build,
    tol: float = 1e-6,
    iter_limit: int = 100000,
    time_sec_limit: float = 600.0,
    verbosity: int = 0,
    num_threads: int = 0,
    is_epigraph: bool = True,
) -> DualSolveResult:
    """Solve a dual or epigraph-dual LP via Google PDLP (CPU).

    is_epigraph: True if `build` is a DualEpiBuildResult (objective is +tau,
        and alpha = obj at optimum).  False if it is a DualBuildResult
        (objective is -y_simplex, and alpha = obj  ALSO at optimum --
        because we minimize -y_simplex and obj_min = -y_simplex_max =
        +primal_alpha).
    Either way, alpha = obj_min when the build produces obj = primal_alpha.
    """
    t0 = time.time()
    A_ub = getattr(build, "A_ub", None)
    b_ub = getattr(build, "b_ub", None)
    qp = _build_qp_from_lp(build.A_eq, build.b_eq, A_ub, b_ub,
                           build.c, build.bounds)
    params = _make_params(
        tol=tol, iter_limit=iter_limit, time_sec_limit=time_sec_limit,
        verbosity=verbosity, num_threads=num_threads,
    )
    result = _pdlp.primal_dual_hybrid_gradient(qp, params)

    wall = time.time() - t0
    sol_status_int = int(result.solve_log.termination_reason)
    # Use enum names from the protobuf
    from ortools.pdlp import solve_log_pb2
    name_map = {v.number: v.name for v in
                solve_log_pb2.TerminationReason.DESCRIPTOR.values}
    status_name = name_map.get(sol_status_int, str(sol_status_int))

    converged = (status_name in ("TERMINATION_REASON_OPTIMAL",))

    x = np.asarray(result.primal_solution, dtype=np.float64)
    y = np.asarray(result.dual_solution, dtype=np.float64)

    # If the solver bailed out before producing a solution, return what
    # we have without attempting to compute alpha.
    if x.size == 0:
        return DualSolveResult(
            status=status_name, alpha=None, x=None, y_eq=None, y_ub=None,
            wall_s=wall, backend="ortools_pdlp",
            raw_status=status_name, converged=False,
            primal_res=float("inf"), dual_res=float("inf"), kkt=float("inf"),
        )

    # Recover alpha
    obj = float(np.dot(build.c, x))
    if is_epigraph:
        # epigraph form: c = e_{tau_idx}, alpha = obj.
        alpha = obj
    else:
        # standard dual form: c = -e_{y_simplex_idx}; obj = -y_simplex_opt
        # = +primal_alpha (since y_simplex_opt = -primal_alpha).
        alpha = obj  # same identity

    # KKT-ish residuals from solve log
    kkt = 0.0
    primal_res = 0.0
    dual_res = 0.0
    if result.solve_log.solution_stats:
        st = result.solve_log.solution_stats
        primal_res = st.convergence_information[0].l_inf_primal_residual if st.convergence_information else 0.0
        dual_res = st.convergence_information[0].l_inf_dual_residual if st.convergence_information else 0.0
        kkt = max(primal_res, dual_res)

    n_ub = A_ub.shape[0] if A_ub is not None else 0
    return DualSolveResult(
        status=("OPTIMAL" if converged else status_name),
        alpha=alpha, x=x,
        y_eq=y[n_ub:] if n_ub > 0 else y,
        y_ub=y[:n_ub] if n_ub > 0 else None,
        wall_s=wall, backend="ortools_pdlp",
        raw_status=status_name, converged=converged,
        primal_res=primal_res, dual_res=dual_res, kkt=kkt,
    )

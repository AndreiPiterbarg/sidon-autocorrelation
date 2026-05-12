"""NVIDIA cuOpt PDLP solver for the Sidon dual / dual-epigraph LPs.

Targets cuOpt 26.02 API (cuopt.linear_programming):
    lp.DataModel        : sets the LP data
    lp.SolverSettings   : configured via set_parameter(name, value)
    lp.Solve(dm, ss)    : returns lp.Solution
    Solution.get_primal_solution() / get_termination_status() / etc.

Method ids (from SolverSettings.toDict default):
    0 = Concurrent (default; runs PDLP + dual simplex + barrier)
    1 = PDLP only
    2 = Dual Simplex
    3 = Barrier (IPM)

PDLP solver modes (set via 'pdlp_solver_mode'):
    0..4 = Stable1, Stable2, Stable3, Methodical1, Fast1
    Default: 4 (Fast1).  Stable3 is "best balance" per docs.

Install on the pod:
    pip install --extra-index-url https://pypi.nvidia.com cuopt-cu12   # CUDA 12.x
    pip install --extra-index-url https://pypi.nvidia.com cuopt-cu13   # CUDA 13.x
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import traceback

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.tier_dual.solve_dual import DualSolveResult


# =====================================================================
# Detect cuOpt availability
# =====================================================================

CUOPT_AVAILABLE = False
CUOPT_IMPORT_ERROR: Optional[str] = None
_CUOPT_API: Optional[str] = None
_lp = None  # the cuopt.linear_programming module

try:
    from cuopt import linear_programming as _lp
    CUOPT_AVAILABLE = True
    _CUOPT_API = "cuopt.linear_programming (26.x)"
except Exception as e:  # noqa
    CUOPT_IMPORT_ERROR = (
        f"cuopt.linear_programming import failed: {e!r}\n"
        f"Install:\n"
        f"  pip install --extra-index-url https://pypi.nvidia.com cuopt-cu12  "
        f"# CUDA 12.x"
    )


def _check_available():
    if not CUOPT_AVAILABLE:
        raise RuntimeError(
            f"cuOpt is not available on this machine.\n{CUOPT_IMPORT_ERROR}"
        )


# =====================================================================
# Stack our LP into a cuOpt DataModel
# =====================================================================

def _stack_constraints(build) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Pack equalities first, then inequalities. Returns (A, row_lo, row_hi).

    Equalities  : row_lo = row_hi = b_eq
    Inequalities: row_lo = -inf,  row_hi = b_ub
    """
    A_eq = build.A_eq
    A_ub = getattr(build, "A_ub", None)
    b_eq = build.b_eq
    b_ub = getattr(build, "b_ub", None)

    if A_ub is not None and A_ub.shape[0] > 0:
        A = sp.vstack([A_eq, A_ub], format="csr")
        row_lo = np.concatenate([b_eq, np.full(A_ub.shape[0], -np.inf)])
        row_hi = np.concatenate([b_eq, b_ub])
    else:
        A = A_eq.tocsr()
        row_lo = b_eq.copy()
        row_hi = b_eq.copy()
    return A, row_lo.astype(np.float64), row_hi.astype(np.float64)


def _var_bounds(build) -> Tuple[np.ndarray, np.ndarray]:
    n = build.n_vars
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    for j, (l, u) in enumerate(build.bounds):
        lo[j] = -np.inf if l is None else float(l)
        hi[j] = +np.inf if u is None else float(u)
    return lo, hi


# =====================================================================
# Build DataModel
# =====================================================================

def build_data_model(build):
    """Construct a cuopt.linear_programming.DataModel from our build."""
    _check_available()
    A, row_lo, row_hi = _stack_constraints(build)
    var_lo, var_hi = _var_bounds(build)
    A_csr = A.astype(np.float64).tocsr()

    dm = _lp.DataModel()
    dm.set_csr_constraint_matrix(
        A_csr.data.astype(np.float64),
        A_csr.indices.astype(np.int32),
        A_csr.indptr.astype(np.int32),
    )
    dm.set_constraint_lower_bounds(row_lo)
    dm.set_constraint_upper_bounds(row_hi)
    dm.set_variable_lower_bounds(var_lo)
    dm.set_variable_upper_bounds(var_hi)
    dm.set_objective_coefficients(build.c.astype(np.float64))
    dm.set_objective_offset(0.0)
    dm.set_objective_scaling_factor(1.0)
    dm.set_maximize(False)   # we minimize
    return dm


# =====================================================================
# Solve
# =====================================================================

def solve_dual_cuopt(
    build,
    *,
    tol: float = 1e-6,
    iter_limit: int = 200000,
    time_sec_limit: float = 1800.0,
    method: int = 1,                        # 1 = PDLP only
    pdlp_solver_mode: int = 2,              # 2 = Stable3 (best balance per docs)
    log_level: int = 1,
    is_epigraph: bool = True,
    crossover: bool = False,
    primal_infeasible_tol: float = 0.0,
    dual_infeasible_tol: float = 0.0,
) -> DualSolveResult:
    """Solve the LP via cuOpt PDLP on GPU.

    method:
        0 = Concurrent (PDLP + simplex + barrier in parallel; GPU+CPU)
        1 = PDLP only (this is what we typically want)
        2 = Dual Simplex (GPU)
        3 = Barrier (IPM)
    """
    _check_available()
    t0 = time.time()

    dm = build_data_model(build)

    ss = _lp.SolverSettings()
    ss.set_optimality_tolerance(float(tol))
    # Per-side tolerances (overrides the symmetric optimality_tolerance)
    ss.set_parameter("absolute_primal_tolerance", float(tol))
    ss.set_parameter("relative_primal_tolerance", float(tol))
    ss.set_parameter("absolute_dual_tolerance", float(tol))
    ss.set_parameter("relative_dual_tolerance", float(tol))
    ss.set_parameter("absolute_gap_tolerance", float(tol))
    ss.set_parameter("relative_gap_tolerance", float(tol))
    ss.set_parameter("iteration_limit", int(iter_limit))
    ss.set_parameter("time_limit", float(time_sec_limit))
    ss.set_parameter("method", int(method))
    ss.set_parameter("pdlp_solver_mode", int(pdlp_solver_mode))
    ss.set_parameter("log_to_console", bool(log_level >= 1))
    ss.set_parameter("crossover", bool(crossover))
    ss.set_parameter("num_gpus", 1)

    if primal_infeasible_tol > 0:
        ss.set_parameter("primal_infeasible_tolerance", float(primal_infeasible_tol))
    if dual_infeasible_tol > 0:
        ss.set_parameter("dual_infeasible_tolerance", float(dual_infeasible_tol))

    solver_t0 = time.time()
    sol = _lp.Solve(dm, ss)
    wall_solve = time.time() - solver_t0
    wall = time.time() - t0

    # Extract result
    try:
        x = np.asarray(sol.get_primal_solution(), dtype=np.float64)
    except Exception:
        x = None
    try:
        y = np.asarray(sol.get_dual_solution(), dtype=np.float64)
    except Exception:
        y = None
    try:
        status = sol.get_termination_status()
    except Exception:
        status = "UNKNOWN"
    try:
        term_reason = sol.get_termination_reason()
    except Exception:
        term_reason = ""
    try:
        primal_obj = float(sol.get_primal_objective())
    except Exception:
        primal_obj = None
    try:
        lp_stats = sol.get_lp_stats()
    except Exception:
        lp_stats = None

    n_ub = (build.A_ub.shape[0] if getattr(build, "A_ub", None) is not None
            else 0)

    if x is None or x.size == 0:
        return DualSolveResult(
            status=str(status), alpha=None, x=None, y_eq=None, y_ub=None,
            wall_s=wall, backend="cuopt_pdlp",
            raw_status=f"{status} ({term_reason})", converged=False,
        )

    # Convert: alpha = c^T x  (we minimize tau directly, alpha = obj)
    alpha = primal_obj if primal_obj is not None else float(np.dot(build.c, x))

    converged = "OPT" in str(status).upper() or "OPTIMAL" in str(status).upper()
    primal_res = 0.0
    dual_res = 0.0
    if lp_stats:
        # Try common attributes
        for attr in ("primal_residual", "l2_primal_residual",
                     "l_inf_primal_residual"):
            try:
                primal_res = float(getattr(lp_stats, attr, 0.0))
                break
            except Exception:
                continue
        for attr in ("dual_residual", "l2_dual_residual",
                     "l_inf_dual_residual"):
            try:
                dual_res = float(getattr(lp_stats, attr, 0.0))
                break
            except Exception:
                continue

    return DualSolveResult(
        status="OPTIMAL" if converged else str(status),
        alpha=alpha, x=x,
        y_eq=y[n_ub:] if (y is not None and n_ub > 0) else (y if n_ub == 0 else None),
        y_ub=y[:n_ub] if (y is not None and n_ub > 0) else None,
        wall_s=wall, backend="cuopt_pdlp",
        raw_status=f"{status} ({term_reason})",
        converged=converged,
        primal_res=primal_res, dual_res=dual_res,
        kkt=max(primal_res, dual_res),
    )


# =====================================================================
# Auto-fallback solve: cuOpt -> ortools -> mosek
# =====================================================================

def solve_dual_auto(
    build,
    *,
    tol: float = 1e-6,
    iter_limit: int = 200000,
    time_sec_limit: float = 1800.0,
    log_level: int = 1,
    is_epigraph: bool = True,
    prefer: str = "cuopt",
) -> DualSolveResult:
    """Try the requested backend, fall back to others if it fails."""
    if prefer == "cuopt":
        order = ["cuopt", "ortools", "mosek"]
    elif prefer == "ortools":
        order = ["ortools", "cuopt", "mosek"]
    else:
        order = ["mosek", "cuopt", "ortools"]

    for backend in order:
        try:
            if backend == "cuopt" and CUOPT_AVAILABLE:
                return solve_dual_cuopt(
                    build, tol=tol, iter_limit=iter_limit,
                    time_sec_limit=time_sec_limit,
                    log_level=log_level, is_epigraph=is_epigraph,
                )
            if backend == "ortools":
                from lasserre.polya_lp.tier_dual.solve_ortools_pdlp import (
                    solve_dual_ortools_pdlp,
                )
                return solve_dual_ortools_pdlp(
                    build, tol=tol, iter_limit=iter_limit,
                    time_sec_limit=time_sec_limit,
                    verbosity=log_level, is_epigraph=is_epigraph,
                )
            if backend == "mosek":
                from lasserre.polya_lp.tier_dual.build_dual_epi import (
                    solve_epi_mosek,
                )
                if is_epigraph:
                    return solve_epi_mosek(build, tol=tol,
                                            verbose=log_level >= 1)
                from lasserre.polya_lp.tier_dual.solve_dual import (
                    solve_dual_mosek,
                )
                return solve_dual_mosek(build, tol=tol,
                                        verbose=log_level >= 1)
        except Exception:
            continue

    return DualSolveResult(
        status="ALL_BACKENDS_FAILED",
        alpha=None, x=None, wall_s=0.0, backend="none",
        raw_status="all backends failed", converged=False,
    )

"""Solvers for the dual LP. Two backends:

  solve_dual_mosek(build, tol=1e-9) : ground-truth MOSEK IPM. Used for
                                       soundness verification.
  solve_dual_pdlp(build, ...)        : tier4.pdlp_robust on the dual.
                                       The structural payoff: 1 free var
                                       instead of n_q+1 in the primal.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.tier_dual.build_dual import DualBuildResult


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------

@dataclass
class DualSolveResult:
    status: str
    alpha: Optional[float]                      # = -obj = +y_simplex_optimal
    x: Optional[np.ndarray]                     # full primal of the dual LP
    y_eq: Optional[np.ndarray] = None           # dual of equality block
    y_ub: Optional[np.ndarray] = None           # dual of inequality block (>= 0)
    wall_s: float = 0.0
    backend: str = ""
    raw_status: object = None
    primal_res: float = 0.0
    dual_res: float = 0.0
    kkt: float = 0.0
    converged: bool = False


# =====================================================================
# MOSEK
# =====================================================================

def solve_dual_mosek(build: DualBuildResult, tol: float = 1e-9,
                     verbose: bool = False) -> DualSolveResult:
    """MOSEK IPM on the dual."""
    import mosek
    t0 = time.time()

    # Stack constraints. MOSEK packs ALL constraints in one matrix; we put
    # the inequalities first (rows [0, n_ub)) then the equalities
    # (rows [n_ub, n_ub+n_eq)).
    n_vars = build.n_vars
    A_ub = build.A_ub
    A_eq = build.A_eq
    n_ub = A_ub.shape[0]
    n_eq = A_eq.shape[0]
    if n_ub > 0:
        A_full = sp.vstack([A_ub, A_eq], format="csr")
    else:
        A_full = A_eq.tocsr()
    n_rows = A_full.shape[0]

    with mosek.Env() as env, env.Task() as task:
        if verbose:
            task.set_Stream(mosek.streamtype.log,
                            lambda msg: print(msg, end="", flush=True))
        task.appendvars(n_vars)
        task.appendcons(n_rows)

        for j, (lo, hi) in enumerate(build.bounds):
            if lo is None and hi is None:
                bk = mosek.boundkey.fr; lb = ub = 0.0
            elif lo is None:
                bk = mosek.boundkey.up; lb = 0.0; ub = float(hi)
            elif hi is None:
                bk = mosek.boundkey.lo; lb = float(lo); ub = 0.0
            elif lo == hi:
                bk = mosek.boundkey.fx; lb = ub = float(lo)
            else:
                bk = mosek.boundkey.ra; lb = float(lo); ub = float(hi)
            task.putvarbound(j, bk, lb, ub)
            task.putcj(j, float(build.c[j]))

        # Inequality rows: A_ub x <= b_ub
        for i in range(n_ub):
            task.putconbound(i, mosek.boundkey.up, 0.0, float(build.b_ub[i]))
        # Equality rows: A_eq x = b_eq
        for i in range(n_eq):
            task.putconbound(n_ub + i, mosek.boundkey.fx,
                             float(build.b_eq[i]), float(build.b_eq[i]))

        A_coo = A_full.tocoo()
        task.putaijlist(
            A_coo.row.astype(np.int64).tolist(),
            A_coo.col.astype(np.int64).tolist(),
            A_coo.data.astype(np.float64).tolist(),
        )

        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        task.putintparam(mosek.iparam.num_threads, 0)
        task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, tol)
        task.putdouparam(mosek.dparam.intpnt_tol_pfeas, tol)
        task.putdouparam(mosek.dparam.intpnt_tol_dfeas, tol)
        task.optimize()

        sol_status = task.getsolsta(mosek.soltype.itr)
        wall = time.time() - t0
        if sol_status != mosek.solsta.optimal:
            return DualSolveResult(
                status=f"OTHER({sol_status})", alpha=None, x=None,
                wall_s=wall, backend="mosek_dual_ipm",
                raw_status=sol_status, converged=False,
            )

        xx = np.zeros(n_vars); task.getxx(mosek.soltype.itr, xx)
        yy = np.zeros(n_rows); task.gety(mosek.soltype.itr, yy)
        y_ub_part = yy[:n_ub] if n_ub > 0 else None
        y_eq_part = yy[n_ub:]
        obj = task.getprimalobj(mosek.soltype.itr)
        # We minimize  c^T x = -y_simplex  in the dual.
        # By strong LP duality, max y_simplex = primal_min_obj = -primal_alpha,
        # so y_simplex_opt = -primal_alpha < 0, and
        #   obj_min = -y_simplex_opt = -(-primal_alpha) = +primal_alpha.
        # Therefore alpha_dual = +obj.
        alpha = float(obj)

        return DualSolveResult(
            status="OPTIMAL", alpha=alpha, x=xx,
            y_eq=y_eq_part, y_ub=y_ub_part,
            wall_s=wall, backend="mosek_dual_ipm",
            raw_status=sol_status, converged=True,
            primal_res=0.0, dual_res=0.0, kkt=0.0,
        )


# =====================================================================
# HiGHS (fallback / cross-check)
# =====================================================================

def solve_dual_highs(build: DualBuildResult,
                     method: str = "ipm",
                     tol: float = 1e-7,
                     verbose: bool = False) -> DualSolveResult:
    """HiGHS solver (IPM or simplex)."""
    import highspy
    t0 = time.time()

    h = highspy.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("solver", method)
    if method == "ipm":
        h.setOptionValue("ipm_optimality_tolerance", float(tol))
        h.setOptionValue("primal_feasibility_tolerance", float(tol))
        h.setOptionValue("dual_feasibility_tolerance", float(tol))
        h.setOptionValue("run_crossover", "off")

    inf = highspy.kHighsInf
    n_vars = build.n_vars
    A_ub = build.A_ub
    A_eq = build.A_eq
    n_ub = A_ub.shape[0]
    n_eq = A_eq.shape[0]
    if n_ub > 0:
        A = sp.vstack([A_ub, A_eq], format="csr")
        row_lo = np.concatenate([np.full(n_ub, -inf), build.b_eq.copy()])
        row_hi = np.concatenate([build.b_ub.copy(), build.b_eq.copy()])
    else:
        A = A_eq.tocsr()
        row_lo = build.b_eq.copy()
        row_hi = build.b_eq.copy()

    A_csc = A.tocsc()
    col_lo = np.empty(n_vars)
    col_hi = np.empty(n_vars)
    for j, (lo, hi) in enumerate(build.bounds):
        col_lo[j] = -inf if lo is None else lo
        col_hi[j] = inf if hi is None else hi

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = build.c.copy()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = row_lo
    lp.row_upper_ = row_hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_csc.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMinimize

    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    wall = time.time() - t0

    if status != highspy.HighsModelStatus.kOptimal:
        return DualSolveResult(
            status=f"OTHER({status})", alpha=None, x=None,
            wall_s=wall, backend=f"highs_dual_{method}",
            raw_status=status, converged=False,
        )

    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    y = np.asarray(sol.row_dual)
    info = h.getInfo()
    # Same sign convention as MOSEK path: alpha = +obj (we minimize -y_simplex).
    alpha = float(info.objective_function_value)
    return DualSolveResult(
        status="OPTIMAL", alpha=alpha, x=x,
        y_eq=y[n_ub:] if n_ub > 0 else y,
        y_ub=y[:n_ub] if n_ub > 0 else None,
        wall_s=wall, backend=f"highs_dual_{method}",
        raw_status=status, converged=True,
    )


# =====================================================================
# PDLP via tier4.pdlp_robust
# =====================================================================

def solve_dual_pdlp(
    build: DualBuildResult,
    max_outer: int = 200,
    max_inner: int = 1000,
    tol: float = 1e-6,
    free_var_box: float = 50.0,
    use_halpern: bool = False,
    verbose: bool = False,
    log_every: int = 5,
) -> DualSolveResult:
    """Run tier4.pdlp_robust restarted PDHG on the dual LP.

    This is the structural payoff of the reformulation: only y_simplex
    is free, so the alpha-drift failure mode of the primal does NOT
    apply here.
    """
    from lasserre.polya_lp.tier4.pdlp_robust import (
        build_gpu_lp, pdlp_solve, unscale,
    )
    import torch

    t0 = time.time()
    lp_gpu, scaling = build_gpu_lp(
        build.A_eq, build.b_eq, build.c, build.bounds,
        A_ub=build.A_ub, b_ub=build.b_ub,
        free_var_box=free_var_box,
    )
    res = pdlp_solve(
        lp_gpu, max_outer=max_outer, max_inner=max_inner, tol=tol,
        use_halpern=use_halpern,
        log_every=log_every, print_log=verbose,
    )

    # Unscale to original LP coordinates
    x_orig, y_orig = unscale(res.x, res.y, scaling)
    x_np = x_orig.detach().cpu().numpy()
    y_np = y_orig.detach().cpu().numpy()

    # Recover alpha. At dual optimum y_simplex = -primal_alpha, so
    # alpha_dual = -y_simplex_opt. In ORIGINAL coords: y_simplex = x_np[y_simplex_idx].
    alpha = -float(x_np[build.y_simplex_idx])

    n_ub = build.A_ub.shape[0]
    return DualSolveResult(
        status="OPTIMAL" if res.converged else "NOT_CONVERGED",
        alpha=alpha, x=x_np,
        y_eq=y_np[n_ub:] if n_ub > 0 else y_np,
        y_ub=y_np[:n_ub] if n_ub > 0 else None,
        wall_s=time.time() - t0,
        backend="pdlp_dual_robust",
        primal_res=res.primal_res, dual_res=res.dual_res, kkt=res.kkt,
        converged=res.converged,
    )

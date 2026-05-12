"""Coarse LP solve for Tier-4 active-set identification.

The Tier-4 workflow needs a CHEAP, low-precision LP solve to identify
the optimal active set: which lambda_W > 0 and which c_beta > 0. This
module exposes a single entry point coarse_solve(build, tol) and two
backends:

  backend="highs_ipm" (default, WORKING): HiGHS interior-point at low
    tolerance. Fast (typically 5-20x faster than MOSEK at 1e-9 on the
    same LP). Reliable convergence on every Polya-LP shape we've tested.

  backend="pdhg_gpu" (PLACEHOLDER): GPU restarted Halpern-PDHG via
    tier4.pdlp_robust. Currently does NOT converge on this LP class
    because the free `alpha` variable (objective coefficient -1) drifts
    to the artificial projection box much faster than the dual y can
    react via the |beta|=0 coupling row. The cuPDLP-CSC fix (diagonal
    preconditioning + low-precision warmstart) is the natural next
    step but is research-grade. Marked as GPU swap-in target; not used
    by the default Tier-4 driver.

Returned object exposes:
  alpha   -- coarse value of -obj
  x, y    -- primal / dual vectors (for active-set extraction)
  slx, sux -- reduced costs at lower / upper bounds (when available)
  primal_res, dual_res, kkt -- numerical quality
  wall_s, backend
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import BuildResult


@dataclass
class CoarseResult:
    alpha: Optional[float]
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    slx: Optional[np.ndarray]
    sux: Optional[np.ndarray]
    primal_res: float
    dual_res: float
    kkt: float
    wall_s: float
    backend: str
    converged: bool
    raw_status: object = None


def _stack_constraints(build: BuildResult):
    """Return (A, b, lb, ub) where A x = b is the equality stack and
    inequality rows (when present) are appended as A_ub <= b_ub.
    For coarse solving we keep them inequality-typed and let HiGHS handle.
    """
    A_eq = build.A_eq
    b_eq = build.b_eq
    A_ub = getattr(build, "A_ub", None)
    b_ub = getattr(build, "b_ub", None)
    return A_eq, b_eq, A_ub, b_ub


def coarse_solve(build: BuildResult, tol: float = 1e-4,
                 backend: str = "highs_ipm",
                 verbose: bool = False) -> CoarseResult:
    """Solve LP at coarse tolerance for active-set identification.

    tol: relative KKT tolerance target. 1e-4 is the standard PDLP/cuOpt
    setting for active-set identification (Applegate et al. 2023).

    Backends (sorted by typical speed at our scale, fastest first):
      * "mosek_simplex"  : MOSEK dual simplex.  Often the fastest on
                           sparse equality LPs because it terminates as
                           soon as a basic optimum is found.
      * "mosek_ipm_low"  : MOSEK IPM at relaxed tolerance.  Saves the
                           final precision-tightening iterations.
      * "highs_simplex"  : HiGHS dual simplex.
      * "highs_ipm"      : HiGHS interior-point at coarse tol.
      * "pdhg_gpu"       : tier4.pdlp_robust GPU PDHG (NOT converging
                           on this LP; placeholder).
    """
    if backend == "highs_ipm":
        return _coarse_highs_ipm(build, tol=tol, verbose=verbose)
    elif backend == "highs_simplex":
        return _coarse_highs_simplex(build, verbose=verbose)
    elif backend == "mosek_ipm_low":
        return _coarse_mosek_ipm_low(build, tol=tol, verbose=verbose)
    elif backend == "mosek_simplex":
        return _coarse_mosek_simplex(build, verbose=verbose)
    elif backend == "pdhg_gpu":
        return _coarse_pdhg_gpu(build, tol=tol, verbose=verbose)
    else:
        raise ValueError(f"Unknown backend {backend!r}")


# ---------------------------------------------------------------------
# HiGHS-IPM backend (PRIMARY)
# ---------------------------------------------------------------------

def _coarse_highs_ipm(build: BuildResult, tol: float, verbose: bool) -> CoarseResult:
    import highspy

    t0 = time.time()
    A_eq, b_eq, A_ub, b_ub = _stack_constraints(build)
    n_vars = build.n_vars

    h = highspy.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("solver", "ipm")
    # Coarse tolerances
    h.setOptionValue("ipm_optimality_tolerance", float(tol))
    h.setOptionValue("primal_feasibility_tolerance", float(tol))
    h.setOptionValue("dual_feasibility_tolerance", float(tol))
    # Skip crossover for IPM-only output
    h.setOptionValue("run_crossover", "off")

    inf = highspy.kHighsInf

    # Stack equalities + inequalities
    if A_ub is not None and A_ub.shape[0] > 0:
        A = sp.vstack([A_eq, A_ub], format="csr")
        b_lo = np.concatenate([b_eq, np.full(A_ub.shape[0], -inf)])
        b_hi = np.concatenate([b_eq, b_ub])
    else:
        A = A_eq.tocsr()
        b_lo = b_eq.copy()
        b_hi = b_eq.copy()

    n_rows = A.shape[0]
    A_csc = A.tocsc()

    col_lo = np.empty(n_vars)
    col_hi = np.empty(n_vars)
    for j, (lo, hi) in enumerate(build.bounds):
        col_lo[j] = -inf if lo is None else lo
        col_hi[j] = inf if hi is None else hi

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_rows
    lp.col_cost_ = build.c.copy()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = b_lo
    lp.row_upper_ = b_hi
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
        return CoarseResult(
            alpha=None, x=None, y=None, slx=None, sux=None,
            primal_res=float("inf"), dual_res=float("inf"), kkt=float("inf"),
            wall_s=wall, backend="highs_ipm",
            converged=False, raw_status=status,
        )

    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    y = np.asarray(sol.row_dual)
    info = h.getInfo()
    obj = info.objective_function_value
    alpha = -float(obj)
    # Residuals (normalized)
    Ax = A @ x
    primal_res = float(np.linalg.norm(np.maximum(b_lo - Ax, 0)
                                      + np.maximum(Ax - b_hi, 0)))
    rc = build.c - A.T @ y
    # reduced cost violation: rc must satisfy sign condition w.r.t. bounds
    rc_viol = np.zeros_like(rc)
    has_lo = ~np.isinf(col_lo)
    has_up = ~np.isinf(col_hi)
    free = (~has_lo) & (~has_up)
    only_lo = has_lo & (~has_up)
    only_up = has_up & (~has_lo)
    rc_viol[free] = np.abs(rc[free])
    rc_viol[only_lo] = np.maximum(0.0, -rc[only_lo])
    rc_viol[only_up] = np.maximum(0.0, rc[only_up])
    dual_res = float(np.linalg.norm(rc_viol))
    norm_b = max(1.0, np.abs(np.where(np.isinf(b_lo), b_hi, b_lo)).max())
    norm_c = max(1.0, np.abs(build.c).max())
    kkt = max(primal_res / (1 + norm_b), dual_res / (1 + norm_c))

    # HiGHS reduced cost at variable bounds: col_dual contains rc; if rc>0
    # the variable is at its lower bound (slx > 0); if rc<0 it is at upper.
    col_dual = np.asarray(sol.col_dual)
    slx = np.maximum(col_dual, 0.0)
    sux = np.maximum(-col_dual, 0.0)

    return CoarseResult(
        alpha=alpha, x=x, y=y, slx=slx, sux=sux,
        primal_res=primal_res, dual_res=dual_res, kkt=kkt,
        wall_s=wall, backend="highs_ipm",
        converged=True, raw_status=status,
    )


# ---------------------------------------------------------------------
# HiGHS simplex backend
# ---------------------------------------------------------------------

def _coarse_highs_simplex(build: BuildResult, verbose: bool) -> CoarseResult:
    import highspy

    t0 = time.time()
    A_eq, b_eq, A_ub, b_ub = _stack_constraints(build)
    n_vars = build.n_vars

    h = highspy.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("solver", "simplex")
    inf = highspy.kHighsInf

    if A_ub is not None and A_ub.shape[0] > 0:
        A = sp.vstack([A_eq, A_ub], format="csr")
        b_lo = np.concatenate([b_eq, np.full(A_ub.shape[0], -inf)])
        b_hi = np.concatenate([b_eq, b_ub])
    else:
        A = A_eq.tocsr()
        b_lo = b_eq.copy()
        b_hi = b_eq.copy()

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
    lp.row_lower_ = b_lo
    lp.row_upper_ = b_hi
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
        return CoarseResult(
            alpha=None, x=None, y=None, slx=None, sux=None,
            primal_res=float("inf"), dual_res=float("inf"), kkt=float("inf"),
            wall_s=wall, backend="highs_simplex",
            converged=False, raw_status=status,
        )

    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    y = np.asarray(sol.row_dual)
    info = h.getInfo()
    alpha = -float(info.objective_function_value)
    return CoarseResult(
        alpha=alpha, x=x, y=y, slx=None, sux=None,
        primal_res=0.0, dual_res=0.0, kkt=1e-12,  # simplex hits exact basic optimum
        wall_s=wall, backend="highs_simplex",
        converged=True, raw_status=status,
    )


# ---------------------------------------------------------------------
# MOSEK IPM low-tolerance backend
# ---------------------------------------------------------------------

def _coarse_mosek_ipm_low(build: BuildResult, tol: float, verbose: bool) -> CoarseResult:
    """MOSEK IPM at relaxed tolerance (skip the final precision iterations).

    Reuses the empirically-best MOSEK options from solve.py but with
    intpnt_tol_rel_gap, intpnt_tol_pfeas, intpnt_tol_dfeas all relaxed
    to `tol` (typically 1e-5 to 1e-7).
    """
    import mosek

    t0 = time.time()
    A_eq, b_eq, A_ub, b_ub = _stack_constraints(build)
    n_vars = build.n_vars

    if A_ub is not None and A_ub.shape[0] > 0:
        A = sp.vstack([A_ub, A_eq], format="csr")
        n_ub = A_ub.shape[0]
        n_eq_only = A_eq.shape[0]
    else:
        A = A_eq.tocsr()
        n_ub = 0
        n_eq_only = A_eq.shape[0]
    n_rows = A.shape[0]
    A_csc = A.tocsc()

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
        for i in range(n_ub):
            task.putconbound(i, mosek.boundkey.up, 0.0, float(b_ub[i]))
        for i in range(n_eq_only):
            task.putconbound(n_ub + i, mosek.boundkey.fx,
                             float(b_eq[i]), float(b_eq[i]))
        A_coo = A.tocoo()
        task.putaijlist(
            A_coo.row.astype(np.int64).tolist(),
            A_coo.col.astype(np.int64).tolist(),
            A_coo.data.astype(np.float64).tolist(),
        )
        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
        task.putintparam(mosek.iparam.intpnt_solve_form, mosek.solveform.dual)
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
        task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
        task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, 1)
        task.putintparam(mosek.iparam.presolve_eliminator_max_fill, 5)
        task.putintparam(mosek.iparam.intpnt_order_method,
                         mosek.orderingtype.force_graphpar)
        task.putintparam(mosek.iparam.num_threads, 0)
        task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, tol)
        task.putdouparam(mosek.dparam.intpnt_tol_pfeas, tol)
        task.putdouparam(mosek.dparam.intpnt_tol_dfeas, tol)
        # Cap iterations -- coarse should be quick. Fall back to whatever
        # we have if MOSEK doesn't finish.
        task.putintparam(mosek.iparam.intpnt_max_iterations, 60)

        task.optimize()
        sol_status = task.getsolsta(mosek.soltype.itr)
        wall = time.time() - t0

        if sol_status not in (mosek.solsta.optimal,
                              mosek.solsta.unknown):  # unknown = max iter; we can still extract
            return CoarseResult(
                alpha=None, x=None, y=None, slx=None, sux=None,
                primal_res=float("inf"), dual_res=float("inf"), kkt=float("inf"),
                wall_s=wall, backend="mosek_ipm_low",
                converged=False, raw_status=sol_status,
            )
        xx = np.zeros(n_vars); task.getxx(mosek.soltype.itr, xx)
        yy = np.zeros(n_rows); task.gety(mosek.soltype.itr, yy)
        obj = task.getprimalobj(mosek.soltype.itr)
        alpha = -float(obj)
        # quick KKT estimate
        rc = build.c - A.T @ yy
        kkt_est = max(np.linalg.norm(A @ xx - np.concatenate(
            [b_ub if A_ub is not None and A_ub.shape[0] > 0 else np.array([]), b_eq])) / max(1, np.abs(np.concatenate([b_ub if A_ub is not None and A_ub.shape[0] > 0 else np.array([]), b_eq])).max()),
                      np.linalg.norm(rc) / max(1, np.abs(build.c).max()))
        return CoarseResult(
            alpha=alpha, x=xx, y=yy, slx=None, sux=None,
            primal_res=0.0, dual_res=0.0, kkt=float(kkt_est),
            wall_s=wall, backend="mosek_ipm_low",
            converged=(sol_status == mosek.solsta.optimal),
            raw_status=sol_status,
        )


# ---------------------------------------------------------------------
# MOSEK simplex backend
# ---------------------------------------------------------------------

def _coarse_mosek_simplex(build: BuildResult, verbose: bool) -> CoarseResult:
    """MOSEK dual simplex. Often the fastest on small sparse equality LPs."""
    import mosek

    t0 = time.time()
    A_eq, b_eq, A_ub, b_ub = _stack_constraints(build)
    n_vars = build.n_vars

    if A_ub is not None and A_ub.shape[0] > 0:
        A = sp.vstack([A_ub, A_eq], format="csr")
        n_ub = A_ub.shape[0]
        n_eq_only = A_eq.shape[0]
    else:
        A = A_eq.tocsr()
        n_ub = 0
        n_eq_only = A_eq.shape[0]
    n_rows = A.shape[0]

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
        for i in range(n_ub):
            task.putconbound(i, mosek.boundkey.up, 0.0, float(b_ub[i]))
        for i in range(n_eq_only):
            task.putconbound(n_ub + i, mosek.boundkey.fx,
                             float(b_eq[i]), float(b_eq[i]))
        A_coo = A.tocoo()
        task.putaijlist(
            A_coo.row.astype(np.int64).tolist(),
            A_coo.col.astype(np.int64).tolist(),
            A_coo.data.astype(np.float64).tolist(),
        )
        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.dual_simplex)
        task.putintparam(mosek.iparam.num_threads, 0)
        task.optimize()
        sol_status = task.getsolsta(mosek.soltype.bas)
        wall = time.time() - t0

        if sol_status != mosek.solsta.optimal:
            return CoarseResult(
                alpha=None, x=None, y=None, slx=None, sux=None,
                primal_res=float("inf"), dual_res=float("inf"), kkt=float("inf"),
                wall_s=wall, backend="mosek_simplex",
                converged=False, raw_status=sol_status,
            )
        xx = np.zeros(n_vars); task.getxx(mosek.soltype.bas, xx)
        yy = np.zeros(n_rows); task.gety(mosek.soltype.bas, yy)
        obj = task.getprimalobj(mosek.soltype.bas)
        alpha = -float(obj)
        return CoarseResult(
            alpha=alpha, x=xx, y=yy, slx=None, sux=None,
            primal_res=0.0, dual_res=0.0, kkt=1e-14,  # simplex hits exact basic optimum
            wall_s=wall, backend="mosek_simplex",
            converged=True, raw_status=sol_status,
        )


# ---------------------------------------------------------------------
# GPU PDHG backend (PLACEHOLDER -- does NOT converge on this LP class)
# ---------------------------------------------------------------------

def _coarse_pdhg_gpu(build: BuildResult, tol: float, verbose: bool) -> CoarseResult:
    """GPU PDHG via tier4.pdlp_robust.

    KNOWN LIMITATION: free `alpha` drifts to the artificial box; this
    backend will return a non-converged result on the standard
    Polya-Handelman LP. Kept here as the swap-in target for a future
    cuPDLP-CSC-style implementation with diagonal preconditioning.
    """
    from lasserre.polya_lp.tier4.pdlp_robust import solve_buildresult
    t0 = time.time()
    res, scaling, alpha_pdlp, x_orig, y_orig = solve_buildresult(
        build, max_outer=80, max_inner=2000, tol=tol,
        free_var_box=50.0, use_halpern=False,
        log_every=10, print_log=verbose,
    )
    wall = time.time() - t0
    x_np = x_orig.detach().cpu().numpy()
    y_np = y_orig.detach().cpu().numpy()
    return CoarseResult(
        alpha=alpha_pdlp, x=x_np, y=y_np, slx=None, sux=None,
        primal_res=res.primal_res, dual_res=res.dual_res, kkt=res.kkt,
        wall_s=wall, backend="pdhg_gpu",
        converged=res.converged, raw_status=None,
    )

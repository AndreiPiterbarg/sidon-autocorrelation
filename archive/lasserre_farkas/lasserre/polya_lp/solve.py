"""LP solver wrapper. HiGHS primary, scipy fallback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import BuildResult


@dataclass
class SolveResult:
    status: str                  # "OPTIMAL" / "INFEASIBLE" / "UNBOUNDED" / "OTHER"
    alpha: Optional[float]       # the certified lower bound (= maximized alpha)
    x: Optional[np.ndarray]
    solver: str
    wall_s: float
    raw_status: object = None
    # CRITICAL for Farkas rigorization: dual values + reduced costs.
    # Without these we cannot produce a rigorous certificate from a numerical solve.
    y: Optional[np.ndarray] = None     # dual on equalities
    slx: Optional[np.ndarray] = None   # reduced cost on lower bounds (>= 0)
    sux: Optional[np.ndarray] = None   # reduced cost on upper bounds (>= 0)


def solve_lp(build: BuildResult, solver: str = "auto", verbose: bool = False,
             method: str = "auto", tol: float = 1e-9) -> SolveResult:
    """Solve the LP described by build.

    solver: "auto" (try MOSEK IPM first, fall back to HiGHS),
            "mosek" (force MOSEK IPM), "highs" (highspy), "scipy".
    method: "auto", "simplex", "ipm" (interior-point).

    DEFAULT: MOSEK IPM with the optimal options for Pólya/Handelman LPs
    (presolve eliminator ON, basis OFF, lindep OFF). Per agent benchmarks,
    expected 50-200x speedup over HiGHS dual simplex.
    """
    t0 = time.time()

    A_eq = build.A_eq
    b_eq = build.b_eq
    c = build.c
    bounds = build.bounds

    if solver in ("auto", "mosek"):
        try:
            return _solve_with_mosek_ipm(build, verbose=verbose, t0=t0, tol=tol)
        except Exception as e:
            if solver == "mosek":
                raise
            if verbose:
                print(f"  MOSEK failed ({e}); falling back to highspy")

    if solver in ("auto", "highs"):
        try:
            return _solve_highspy(build, verbose=verbose, t0=t0, method=method)
        except Exception as e:
            if verbose:
                print(f"  highspy failed ({e}); falling back to scipy.optimize.linprog")

    # Fallback: scipy.optimize.linprog with method='highs'
    from scipy.optimize import linprog
    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": verbose, "presolve": True},
    )
    wall = time.time() - t0
    if res.success:
        return SolveResult(
            status="OPTIMAL",
            alpha=float(-res.fun),  # we minimized -alpha
            x=res.x,
            solver="scipy.linprog/highs",
            wall_s=wall,
            raw_status=res.status,
        )
    if res.status == 2:  # infeasible
        return SolveResult("INFEASIBLE", None, None, "scipy.linprog/highs", wall, res.status)
    if res.status == 3:  # unbounded
        return SolveResult("UNBOUNDED", None, None, "scipy.linprog/highs", wall, res.status)
    return SolveResult("OTHER", None, None, "scipy.linprog/highs", wall, res.status)


def _solve_with_mosek_ipm(build: BuildResult, verbose: bool, t0: float,
                          tol: float = 1e-9) -> SolveResult:
    """MOSEK IPM with the empirically-best options for Pólya LPs.

    Per agent research (Mittelmann benchmarks, MOSEK docs):
      - intpnt_basis = NEVER: skip basis identification (30-60% time savings)
      - intpnt_solve_form = DUAL: better-conditioned KKT for our LP
      - presolve_eliminator = ON: substitute free q vars (HUGE win)
      - presolve_lindep = OFF: we know rows are independent
      - num_threads = 0: use all cores for Cholesky
    """
    import mosek
    n_vars = build.n_vars

    # Build combined constraint system for MOSEK: equality + inequality.
    # MOSEK can handle both via boundkey per row.
    has_ub = (build.A_ub is not None and build.A_ub.shape[0] > 0)
    if has_ub:
        # Stack inequality rows + equality rows
        A_combined = sp.vstack([build.A_ub, build.A_eq], format="csr")
        n_ub = build.A_ub.shape[0]
        n_eq_only = build.A_eq.shape[0]
    else:
        A_combined = build.A_eq
        n_ub = 0
        n_eq_only = build.A_eq.shape[0]
    n_rows = A_combined.shape[0]
    A_csc = A_combined.tocsc()

    with mosek.Env() as env:
        with env.Task() as task:
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

            # Inequality rows: -INF <= row <= b_ub[i]
            for i in range(n_ub):
                task.putconbound(i, mosek.boundkey.up,
                                 0.0, float(build.b_ub[i]))
            # Equality rows: row == b_eq[i]
            for i in range(n_eq_only):
                task.putconbound(n_ub + i, mosek.boundkey.fx,
                                 float(build.b_eq[i]), float(build.b_eq[i]))

            A_coo = A_combined.tocoo()
            task.putaijlist(
                A_coo.row.astype(np.int64).tolist(),
                A_coo.col.astype(np.int64).tolist(),
                A_coo.data.astype(np.float64).tolist(),
            )

            task.putobjsense(mosek.objsense.minimize)

            # OPTIMAL OPTIONS for Polya/Handelman LP (MOSEK 11.x):
            # - intpnt: force IPM (avoid simplex degeneracy)
            # - solve_form=DUAL: better-conditioned KKT
            # - intpnt_basis=NEVER: skip crossover (30-60% time saved)
            # - presolve_lindep_use=OFF: rows are independent by construction
            # - presolve_eliminator_max_num_tries=1: ONE pass only.
            #   The q-vars form a chain (q_beta depends on q_{beta-e_j}),
            #   so cascading elimination densifies A catastrophically -- one
            #   substitution creates fill that propagates. Empirical evidence:
            #   d=16 R=12 had factor nnz inflate 950x (813K -> 776M) under the
            #   prior cascade. Limiting to 1 pass should chop factor 3-8x.
            # - presolve_eliminator_max_fill=5: cap densification per pivot.
            # - intpnt_order_method=force_graphpar: nested-dissection (METIS-
            #   style) ordering. Beats AMD by 2-5x on banded-cluster LPs like
            #   ours where every monomial beta touches every shift beta-e_j.
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            task.putintparam(mosek.iparam.intpnt_solve_form, mosek.solveform.dual)
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
            task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
            task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, 1)
            task.putintparam(mosek.iparam.presolve_eliminator_max_fill, 5)
            # ORDERING: was force_graphpar (METIS nested-dissection). At d=16
            # R=28 with 30M-row chain-coupled LP, METIS sat single-threaded
            # for 1h+ without finishing -- nested dissection is pathological
            # on long chain-like adjacency. Switching to appminloc (approximate
            # min-degree) which is single-threaded but linear in nnz and well
            # suited to chain sparsity. Kept commented for revert if needed.
            # task.putintparam(mosek.iparam.intpnt_order_method,
            #                  mosek.orderingtype.force_graphpar)
            task.putintparam(mosek.iparam.intpnt_order_method,
                             mosek.orderingtype.appminloc)
            task.putintparam(mosek.iparam.num_threads, 0)
            task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, tol)
            task.putdouparam(mosek.dparam.intpnt_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_tol_dfeas, tol)

            # MAXIMUM diagnostic logging. Every log_* MOSEK accepts, set to
            # the highest value the param will take. Useful when we don't yet
            # know which serial phase between "Presolve terminated" and the
            # IPM iteration table is the bottleneck.
            task.putintparam(mosek.iparam.log, 10)
            for pname, val in [
                ('log_presolve', 4),       # presolve detail
                ('log_intpnt', 4),         # IPM detail
                ('log_order', 10),         # ordering detail (try high)
                ('log_factor', 10),        # factorization detail
                ('log_storage', 4),        # storage / memory
                ('log_optimizer', 4),      # general optimizer progress
                ('log_response', 4),       # response codes
                ('log_expand', 4),         # presolve expand
                ('log_local_info', 4),     # local info
                ('log_ana_pro', 4),        # problem analyzer
                ('log_check_convexity', 4),
                ('log_include_summary', 4),
                ('log_infeas_ana', 4),
                ('log_sensitivity', 4),
                ('log_sensitivity_opt', 4),
                ('log_bi', 4),
                ('log_cut_second_opt', 4),
                ('log_feas_repair', 4),
                ('log_sim', 4),
            ]:
                p = getattr(mosek.iparam, pname, None)
                if p is None:
                    continue
                try: task.putintparam(p, val)
                except Exception: pass

            # Have MOSEK also write a per-iteration log file directly to disk
            # (independent of the Python stream callback). Path is provided
            # via an env var so it's easy to change between runs.
            try:
                lpath = os.environ.get('MOSEK_LOG_FILE')
                if lpath:
                    task.linkfiletostream(mosek.streamtype.log, lpath, 0)
            except Exception:
                pass

            task.optimize()

            wall = time.time() - t0
            sol_status = task.getsolsta(mosek.soltype.itr)
            soltype = mosek.soltype.itr

            if sol_status == mosek.solsta.optimal:
                xx = np.zeros(n_vars)
                task.getxx(soltype, xx)
                # Extract dual values for Farkas rigorization
                yy = np.zeros(n_rows)
                task.gety(soltype, yy)
                slx = np.zeros(n_vars)
                task.getslx(soltype, slx)
                sux = np.zeros(n_vars)
                task.getsux(soltype, sux)
                obj = task.getprimalobj(soltype)
                alpha = -float(obj)
                return SolveResult("OPTIMAL", alpha, xx, "mosek_ipm",
                                   wall, sol_status,
                                   y=yy, slx=slx, sux=sux)
            if sol_status == mosek.solsta.prim_infeas_cer:
                return SolveResult("INFEASIBLE", None, None, "mosek_ipm",
                                   wall, sol_status)
            if sol_status == mosek.solsta.dual_infeas_cer:
                return SolveResult("UNBOUNDED", None, None, "mosek_ipm",
                                   wall, sol_status)
            return SolveResult(f"OTHER({sol_status})", None, None,
                               "mosek_ipm", wall, sol_status)


def _solve_highspy(build: BuildResult, verbose: bool, t0: float,
                   method: str = "auto") -> SolveResult:
    """Direct highspy interface (faster, scales further than scipy)."""
    import highspy

    h = highspy.Highs()
    if not verbose:
        h.silent()
    if method == "simplex":
        h.setOptionValue("solver", "simplex")
    elif method == "ipm":
        h.setOptionValue("solver", "ipm")
    # else "auto": let HiGHS decide

    A_eq = build.A_eq.tocsc()
    n_vars = build.n_vars
    n_eq = A_eq.shape[0]

    inf = highspy.kHighsInf

    # Add columns
    col_lo = np.empty(n_vars)
    col_hi = np.empty(n_vars)
    for i, (lo, hi) in enumerate(build.bounds):
        col_lo[i] = -inf if lo is None else lo
        col_hi[i] = inf if hi is None else hi

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_eq
    lp.col_cost_ = build.c.copy()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = build.b_eq.copy()
    lp.row_upper_ = build.b_eq.copy()  # equality
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A_eq.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_eq.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_eq.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMinimize  # we minimize -alpha

    h.passModel(lp)
    h.run()

    status = h.getModelStatus()
    wall = time.time() - t0

    if status == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        # Objective stored as the model objective; we want alpha = -obj
        info = h.getInfo()
        obj = info.objective_function_value
        alpha = -float(obj)
        return SolveResult("OPTIMAL", alpha, x, "highspy", wall, status)

    if status == highspy.HighsModelStatus.kInfeasible:
        return SolveResult("INFEASIBLE", None, None, "highspy", wall, status)
    if status in (highspy.HighsModelStatus.kUnbounded,
                  highspy.HighsModelStatus.kUnboundedOrInfeasible):
        return SolveResult("UNBOUNDED", None, None, "highspy", wall, status)
    return SolveResult(f"OTHER({status})", None, None, "highspy", wall, status)

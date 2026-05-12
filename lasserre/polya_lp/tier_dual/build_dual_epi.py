"""Epigraph form of the dual LP -- ZERO free variables.

The standard dual (build_dual.py) had 1 free variable y_simplex. PDLP
still failed on it because y_simplex has objective gradient -1 and the
dual feedback through n_W inequality rows is too weak to prevent drift.

EPIGRAPH SUBSTITUTION:
  In the standard dual,
      max y_simplex
      s.t. y_β >= 0,  y_0 = 1,  moment recursion,
           sum_b y_β coeff_W(β) + y_simplex <= 0  for all W
  We have at the optimum  y_simplex = -max_W (sum_b y_β coeff_W(β)).
  Setting  tau := -y_simplex, the LP becomes

      min tau
      s.t. y_β >= 0,  tau >= 0,
           y_0 = 1, moment recursion,
           sum_b y_β coeff_W(β) - tau <= 0  for all W

  - alpha = tau_optimal (NO sign flip; obj is exactly tau).
  - All variables now BOUNDED below by 0 (no free vars).
  - The moment-measure bound  y_β = E[mu^β] <= 1 on the simplex, so
    we can ALSO put y_β <= 1 as an upper bound, making every variable
    a finite box. PDLP-friendly.

Box bounds we apply:
  y_β   in [0, 1]   (moment of probability measure on the unit simplex)
  tau   in [0, T_MAX] where T_MAX is loose (default 10).

Why [0, 1] is sound:
  At the optimum, y_β = int mu^β dmu on a probability measure dmu on the
  simplex {mu >= 0, sum mu = 1}. Each mu_i in [0, 1], so mu^β <=1 hence
  the integral <= 1. The bound is tight only at extremal measures.

Why tau in [0, T_MAX] is sound:
  tau = max_W (sum_β y_β coeff_W(β)). Each coeff_W(β) <= max_ij M_W[i,j],
  and M_W[i,j] <= 2d/2 = d (for our normalized window matrices). So
  tau <= d * (sum_β y_β) <= d * (number of monos in support) which is
  finite. T_MAX = 10 is conservative for d <= 100.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.poly import enum_monomials_le, index_map
from lasserre.polya_lp.tier_dual.build_dual import _coeff_W_pairs


@dataclass
class DualEpiBuildResult:
    """Epigraph dual LP. Variable layout: x = [y_beta (n_le_R), tau (1)]."""
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    c: np.ndarray
    bounds: List[Tuple[Optional[float], Optional[float]]]
    n_vars: int
    y_idx: slice
    tau_idx: int
    monos_le_R: List[Tuple[int, ...]]
    monos_le_Rm1: List[Tuple[int, ...]]
    beta_to_idx: dict
    n_W: int
    n_q_recursion_rows: int
    d: int
    R: int
    build_wall_s: float

    @property
    def n_eq(self) -> int:
        return self.A_eq.shape[0]

    @property
    def n_ub(self) -> int:
        return self.A_ub.shape[0]


def build_dual_epi_lp(
    d: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    y_upper: float = 1.0,
    tau_upper: float = 10.0,
    verbose: bool = False,
) -> DualEpiBuildResult:
    """Build the epigraph dual LP.

    Args:
        y_upper   : upper bound on each y_beta (default 1.0; sound by
                    moment-measure bound on the simplex).
        tau_upper : upper bound on tau (default 10.0; conservative for
                    Sidon problem where alpha < 2 always).
    """
    t0 = time.time()
    n_W = len(M_mats)
    monos_le_R = enum_monomials_le(d, R)
    n_le_R = len(monos_le_R)
    beta_to_idx = index_map(monos_le_R)
    monos_le_Rm1 = enum_monomials_le(d, R - 1) if R >= 1 else []
    n_q = len(monos_le_Rm1)

    tau_idx = n_le_R
    n_vars = n_le_R + 1

    if verbose:
        print(f"  Epi-dual LP: n_vars={n_vars} (all BOUNDED), "
              f"n_eq={1+n_q}, n_ub={n_W}", flush=True)

    # ---- Equalities ----
    eq_rows: List[int] = []
    eq_cols: List[int] = []
    eq_vals: List[float] = []
    eq_rhs: List[float] = []
    zero_beta = tuple([0] * d)

    # y_0 = 1
    row_idx = 0
    eq_rows.append(row_idx)
    eq_cols.append(beta_to_idx[zero_beta])
    eq_vals.append(1.0)
    eq_rhs.append(1.0)
    row_idx += 1

    # Moment recursion: y_K = sum_j y_{K+e_j} for K with |K| <= R-1
    for K in monos_le_Rm1:
        eq_rows.append(row_idx)
        eq_cols.append(beta_to_idx[K])
        eq_vals.append(1.0)
        for j in range(d):
            shifted = list(K)
            shifted[j] += 1
            if sum(shifted) > R:
                continue
            shifted_t = tuple(shifted)
            j_idx = beta_to_idx.get(shifted_t)
            if j_idx is not None:
                eq_rows.append(row_idx)
                eq_cols.append(j_idx)
                eq_vals.append(-1.0)
        eq_rhs.append(0.0)
        row_idx += 1

    n_eq_rows = row_idx
    A_eq = sp.csr_matrix(
        (np.asarray(eq_vals, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(n_eq_rows, n_vars),
    )
    b_eq = np.asarray(eq_rhs, dtype=np.float64)

    # ---- Inequalities ----
    # For each W:  sum_b y_beta coeff_W(beta) - tau <= 0
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    ub_vals: List[float] = []
    for w, M_W in enumerate(M_mats):
        pairs = _coeff_W_pairs(np.asarray(M_W, dtype=np.float64))
        for beta, val in pairs:
            j = beta_to_idx.get(beta)
            if j is None:
                continue
            ub_rows.append(w)
            ub_cols.append(j)
            ub_vals.append(val)
        # -tau
        ub_rows.append(w)
        ub_cols.append(tau_idx)
        ub_vals.append(-1.0)

    A_ub = sp.csr_matrix(
        (np.asarray(ub_vals, dtype=np.float64),
         (np.asarray(ub_rows, dtype=np.int64),
          np.asarray(ub_cols, dtype=np.int64))),
        shape=(n_W, n_vars),
    )
    b_ub = np.zeros(n_W, dtype=np.float64)

    # ---- Objective: min tau ----
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[tau_idx] = 1.0

    # ---- Bounds: ALL bounded ----
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for _ in range(n_le_R):
        bounds.append((0.0, y_upper))
    bounds.append((0.0, tau_upper))

    return DualEpiBuildResult(
        A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        c=c_obj, bounds=bounds,
        n_vars=n_vars, y_idx=slice(0, n_le_R), tau_idx=tau_idx,
        monos_le_R=monos_le_R, monos_le_Rm1=monos_le_Rm1,
        beta_to_idx=beta_to_idx,
        n_W=n_W, n_q_recursion_rows=n_q,
        d=d, R=R,
        build_wall_s=time.time() - t0,
    )


def summarize_epi(b: DualEpiBuildResult) -> str:
    return (f"Epi-Dual LP at d={b.d} R={b.R}: "
            f"n_vars={b.n_vars} (ALL bounded, 0 free), "
            f"n_eq={b.n_eq} (1 + {b.n_q_recursion_rows} recursion), "
            f"n_ub={b.n_ub}, "
            f"nnz_eq={b.A_eq.nnz}, nnz_ub={b.A_ub.nnz}, "
            f"build={b.build_wall_s*1000:.1f}ms")


# =====================================================================
# Solvers
# =====================================================================

def solve_epi_mosek(build: DualEpiBuildResult, tol: float = 1e-9,
                    verbose: bool = False):
    """MOSEK on the epigraph dual."""
    from lasserre.polya_lp.tier_dual.solve_dual import DualSolveResult
    import mosek

    t0 = time.time()
    A_ub = build.A_ub; A_eq = build.A_eq
    n_ub = A_ub.shape[0]; n_eq = A_eq.shape[0]
    if n_ub > 0:
        A_full = sp.vstack([A_ub, A_eq], format="csr")
    else:
        A_full = A_eq.tocsr()
    n_rows = A_full.shape[0]
    n_vars = build.n_vars

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
            task.putconbound(i, mosek.boundkey.up, 0.0, float(build.b_ub[i]))
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
                wall_s=wall, backend="mosek_epi",
                raw_status=sol_status, converged=False,
            )
        xx = np.zeros(n_vars); task.getxx(mosek.soltype.itr, xx)
        yy = np.zeros(n_rows); task.gety(mosek.soltype.itr, yy)
        obj = task.getprimalobj(mosek.soltype.itr)
        # alpha = tau_opt = obj (we minimize tau directly)
        alpha = float(obj)
        return DualSolveResult(
            status="OPTIMAL", alpha=alpha, x=xx,
            y_eq=yy[n_ub:], y_ub=yy[:n_ub] if n_ub > 0 else None,
            wall_s=wall, backend="mosek_epi",
            raw_status=sol_status, converged=True,
        )


def solve_epi_pdlp(
    build: DualEpiBuildResult,
    max_outer: int = 200, max_inner: int = 1000,
    tol: float = 1e-6, free_var_box: float = 50.0,
    use_halpern: bool = False, verbose: bool = False, log_every: int = 5,
):
    """PDLP on the epigraph dual.

    With ZERO free variables, PDHG should converge cleanly. Returns the
    same DualSolveResult shape as solve_dual_pdlp.
    """
    from lasserre.polya_lp.tier_dual.solve_dual import DualSolveResult
    from lasserre.polya_lp.tier4.pdlp_robust import (
        build_gpu_lp, pdlp_solve, unscale,
    )

    t0 = time.time()
    lp_gpu, scaling = build_gpu_lp(
        build.A_eq, build.b_eq, build.c, build.bounds,
        A_ub=build.A_ub, b_ub=build.b_ub,
        free_var_box=free_var_box,  # irrelevant since no free vars
    )
    res = pdlp_solve(
        lp_gpu, max_outer=max_outer, max_inner=max_inner, tol=tol,
        use_halpern=use_halpern, log_every=log_every, print_log=verbose,
    )
    x_orig, y_orig = unscale(res.x, res.y, scaling)
    x_np = x_orig.detach().cpu().numpy()
    y_np = y_orig.detach().cpu().numpy()
    # alpha = tau_optimal (no sign flip)
    alpha = float(x_np[build.tau_idx])

    n_ub = build.A_ub.shape[0]
    return DualSolveResult(
        status="OPTIMAL" if res.converged else "NOT_CONVERGED",
        alpha=alpha, x=x_np,
        y_eq=y_np[n_ub:] if n_ub > 0 else y_np,
        y_ub=y_np[:n_ub] if n_ub > 0 else None,
        wall_s=time.time() - t0, backend="pdlp_epi",
        primal_res=res.primal_res, dual_res=res.dual_res, kkt=res.kkt,
        converged=res.converged,
    )

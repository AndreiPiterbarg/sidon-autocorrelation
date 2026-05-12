"""Joint Lasserre SDP via bisection on t (the epigraph variable).

The Lasserre relaxation of val(d) = min_mu max_W mu^T M_W mu is:

    L_k(d) = min t
             s.t. t * M_{k-1}(y) - M_{k-1}(q_W * y) succeq 0   forall W
                  M_k(y) succeq 0                              (moment)
                  M_{k-1}(mu_i y) succeq 0   for each i        (simplex mu_i>=0)
                  Ay = b                                        (y_0=1 + consistency)

where q_W(mu) = mu^T M_W mu. The window-localizing constraint t*M_{k-1}(y)
is BILINEAR in (t, y), so MOSEK Fusion cannot express it as a Variable*Variable.

BISECTION on t turns each feasibility check into a linear SDP:
    find y s.t. t_fixed * M_{k-1}(y) - M_{k-1}(q_W y) succeq 0, base constraints.

If feasible → L_k(d) <= t_fixed.  If infeasible → L_k(d) > t_fixed.
Bisect until gap < tol.

At the converged upper bracket t_hi (a FEASIBLE t value), the optimal dual
S_W for each window is a PSD matrix. The (0,0) entry of S_W (after proper
normalization) gives the effective lambda_W — the convex combination over
windows that attains t_hi.

This module returns lam_W (float) and t_hi (float), which the safe_certify
pipeline can then use to produce a rigorous rational certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import numpy as np
from scipy import sparse as sp

from lasserre.precompute import _precompute


@dataclass
class BisectResult:
    d: int
    order: int
    t_hi: float                  # feasible upper bracket
    t_lo: float                  # infeasible lower bracket
    lam_win: np.ndarray          # (n_win,) effective weights, sum=1
    mu_A: np.ndarray             # base eq duals at t_hi
    S_base: List[np.ndarray]     # base PSD duals (moment + mu_i loc) at t_hi
    S_win: List[np.ndarray]      # window-localizing PSD duals at t_hi
    y: np.ndarray                # primal moments at t_hi
    n_bisect_steps: int
    total_time: float


def _build_feasibility_model_at_t(P: dict, t_val: float):
    """Build MOSEK Fusion model for feasibility check at fixed t.

    Returns the Model and references to constraints for dual extraction.
    """
    import mosek
    from mosek.fusion import (Model, Domain, Expr, Matrix, ObjectiveSense)

    d = P['d']
    n_y = P['n_y']
    n_loc = P['n_loc']
    n_basis = P['n_basis']
    t_pick = P['t_pick']
    ab_eiej_idx = P['ab_eiej_idx']
    ab_flat = P['ab_flat']
    M_mats = P['M_mats']
    windows = P['windows']
    n_win = len(windows)
    moment_pick = P['moment_pick']
    loc_picks = P['loc_picks']
    idx = P['idx']

    M = Model('feasibility_at_t')
    y = M.variable('y', n_y, Domain.unbounded())

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    M.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency constraints
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_cons = 0
    for r in range(len(P['consist_mono'])):
        ai = int(consist_idx[r])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[r]
        has_child = False
        for ci in range(d):
            ch = int(child_idx[ci])
            if ch >= 0:
                c_rows.append(n_cons)
                c_cols.append(ch)
                c_vals.append(1.0)
                has_child = True
        if has_child:
            c_rows.append(n_cons)
            c_cols.append(ai)
            c_vals.append(-1.0)
            n_cons += 1
    if n_cons > 0:
        C_cons = Matrix.sparse(n_cons, n_y, c_rows, c_cols, c_vals)
        cons_con = M.constraint('consist', Expr.mul(C_cons, y),
                                 Domain.equalsTo([0.0] * n_cons))
    else:
        cons_con = None

    # Base PSD: moment block M_k(y)
    mom_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    mom_con = M.constraint('moment', mom_mat, Domain.inPSDCone(n_basis))

    # Localizing PSD: M_{k-1}(mu_i * y) for each i
    loc_cons = []
    for i in range(d):
        lm = Expr.reshape(y.pick(list(loc_picks[i])), n_loc, n_loc)
        loc_cons.append(
            M.constraint(f'loc_{i}', lm, Domain.inPSDCone(n_loc)))

    # Window-localizing PSD: t_val * M_{k-1}(y) - M_{k-1}(q_W y) succeq 0
    # = t_val * y[t_pick].reshape(n_loc,n_loc) - Cw @ y
    # NOTE: t_val is a scalar (float), so t_val * y[t_pick] is linear in y.
    t_times_y_picked = Expr.mul(t_val, y.pick(t_pick))  # length n_loc^2 expr
    flat_size = n_loc * n_loc

    win_cons = []
    for w in range(n_win):
        Mw = M_mats[w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0 or ab_eiej_idx is None:
            Lw_flat = t_times_y_picked
        else:
            y_idx = ab_eiej_idx[:, :, nz_i, nz_j]
            valid = y_idx >= 0
            if not np.any(valid):
                Lw_flat = t_times_y_picked
            else:
                ab_exp = np.broadcast_to(ab_flat[:, :, None], y_idx.shape)
                mw_vals = Mw[nz_i, nz_j]
                mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)
                rows = ab_exp[valid].ravel().tolist()
                cols = y_idx[valid].ravel().tolist()
                vals = mw_exp[valid].ravel().tolist()
                Cw = Matrix.sparse(flat_size, n_y, rows, cols, vals)
                Lw_flat = Expr.sub(t_times_y_picked, Expr.mul(Cw, y))
        Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
        win_cons.append(
            M.constraint(f'wpsd_{w}', Lw_mat, Domain.inPSDCone(n_loc)))

    # Objective: arbitrary (feasibility). Minimize 0 or use a dummy.
    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    return {
        'model': M,
        'y': y,
        'eq_con_consist': cons_con,
        'mom_con': mom_con,
        'loc_cons': loc_cons,
        'win_cons': win_cons,
    }


def bisect_joint_sdp(d: int, order: int,
                      t_lo: float = 1.0,
                      t_hi: float = 1.5,
                      tol: float = 1e-5,
                      max_iters: int = 30,
                      verbose: bool = False,
                      nthreads: int = 8) -> BisectResult:
    """Bisection on t until we bracket L_k(d) to within tol."""
    import mosek
    from mosek.fusion import (SolutionStatus, ProblemStatus)

    t_start = time.time()
    P = _precompute(d, order, verbose=verbose)

    # First: probe at t = t_hi. If infeasible → raise t_hi.
    # Then: bisect.
    def check_feasible(tv: float):
        build = _build_feasibility_model_at_t(P, tv)
        M = build['model']
        try:
            M.setSolverParam('numThreads', nthreads)
            M.setSolverParam('intpntCoTolPfeas', 1e-9)
            M.setSolverParam('intpntCoTolDfeas', 1e-9)
            M.setSolverParam('intpntCoTolRelGap', 1e-9)
            M.solve()
            ps = M.getProblemStatus()
            ss = M.getPrimalSolutionStatus()
            feasible = (ps == ProblemStatus.PrimalAndDualFeasible
                        and ss == SolutionStatus.Optimal)
            if feasible:
                # Extract duals at this feasible t
                y_val = np.array(build['y'].level(), dtype=np.float64)
                mu_A_list = []
                if build['eq_con_consist'] is not None:
                    mu_A_list = list(build['eq_con_consist'].dual())
                # y_0 equation: the dual is at the "y0" constraint;
                # we'll just pad a 0 at position 0 and let safe_certify
                # figure it out. Actually we include it:
                # The correct mu_A has length = 1 + n_cons.
                # But `cons_con` only covers consistency, not y0=1.
                # For safe_certify we need the full mu_A matching build_sdp.A.
                # We return two arrays and let caller merge.
                mom_S = np.array(build['mom_con'].dual(),
                                 dtype=np.float64).reshape(
                    P['n_basis'], P['n_basis'])
                loc_S = [np.array(c.dual(), dtype=np.float64).reshape(
                    P['n_loc'], P['n_loc']) for c in build['loc_cons']]
                win_S = [np.array(c.dual(), dtype=np.float64).reshape(
                    P['n_loc'], P['n_loc']) for c in build['win_cons']]
                return True, {'y': y_val,
                              'mu_A_consist': np.array(mu_A_list),
                              'S_mom': mom_S,
                              'S_loc': loc_S,
                              'S_win': win_S}
            return False, None
        finally:
            M.dispose()

    # Probe the upper bracket
    if verbose:
        print(f'[bisect] d={d} order={order}, initial bracket '
              f'[{t_lo:.6f}, {t_hi:.6f}]', flush=True)

    feas, data_hi = check_feasible(t_hi)
    if not feas:
        # Expand upper bracket until feasible
        for _ in range(6):
            t_hi *= 1.2
            if verbose:
                print(f'  t_hi infeasible, expanding to {t_hi:.6f}', flush=True)
            feas, data_hi = check_feasible(t_hi)
            if feas:
                break
        else:
            raise RuntimeError(f'Could not find feasible upper bracket up to {t_hi}')

    # Bisect
    n_steps = 0
    for step in range(max_iters):
        if t_hi - t_lo < tol:
            break
        mid = 0.5 * (t_lo + t_hi)
        if verbose:
            print(f'  [{step+1}/{max_iters}] probe t={mid:.8f}', flush=True)
        feas_mid, data_mid = check_feasible(mid)
        n_steps += 1
        if feas_mid:
            t_hi = mid
            data_hi = data_mid
            if verbose:
                print(f'     FEAS   → t_hi={t_hi:.8f}', flush=True)
        else:
            t_lo = mid
            if verbose:
                print(f'     INFEAS → t_lo={t_lo:.8f}', flush=True)

    # Extract lam_W from window-PSD duals at t_hi
    # For each W, lam_W ≈ S_W[0, 0] (the scalar component along the constant y_0=1)
    S_win = data_hi['S_win']
    lam_W = np.array([float(S[0, 0]) for S in S_win], dtype=np.float64)
    lam_W = np.maximum(lam_W, 0.0)
    s = lam_W.sum()
    if s > 0:
        lam_W = lam_W / s

    return BisectResult(
        d=d, order=order,
        t_hi=t_hi, t_lo=t_lo,
        lam_win=lam_W,
        mu_A=data_hi['mu_A_consist'],
        S_base=[data_hi['S_mom']] + data_hi['S_loc'],
        S_win=S_win,
        y=data_hi['y'],
        n_bisect_steps=n_steps,
        total_time=time.time() - t_start,
    )

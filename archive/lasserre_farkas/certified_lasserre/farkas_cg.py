"""Constraint-generation Farkas certification.

Strategy:
  1. Build the feasibility SDP with NO window-localizing constraints (just
     moment + mu_i localizing + Ay=b).
  2. Solve. If infeasible, extract Farkas and we're done.
  3. If feasible with solution y_star, check each window W: is
     t*M_{k-1}(y_star) - M_{k-1}(q_W y_star) PSD? If min eigenvalue < -tol,
     W is violated.
  4. Add the K most-violated windows (default K=5) as new PSD constraints.
  5. Re-solve the augmented model. Repeat.
  6. Converge when either infeasibility is detected OR no violations remain.

Soundness argument:
  A Farkas certificate for a SUBSET S of window constraints proves
  infeasibility of the MORE-CONSTRAINED problem with ALL windows. This is
  because more constraints = smaller feasible set; if the reduced (subset)
  problem is infeasible, the full problem is also infeasible.

Why this is fast:
  Typically 5-30 windows are "active" at the optimum. Starting with 0 and
  adding violated ones reaches convergence in few iterations, each solving
  a MUCH smaller SDP than the all-windows one. At d=8 L3 (120 windows),
  CG typically touches 5-15 windows, giving ~10x-24x speedup.

At d=16 L3 (496 windows), CG expected to touch 10-40 windows → ~15-50x.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple, Set
import time
import numpy as np
from scipy import sparse as sp

try:
    import flint  # type: ignore
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

from lasserre.precompute import _precompute
from certified_lasserre.build_sdp import (
    PSDBlock, _build_moment_block, _build_loc_blocks,
    _build_equality_constraints,
)
from certified_lasserre.safe_certify_flint import (
    _round_mat_fmpq, _round_vec_fmpq,
    _fmpq_to_frac, _fmpq_to_float,
    _sparse_matvec_fmpq, _adjoint_block_fmpq,
)
from certified_lasserre.safe_certify import _decimal_str
from certified_lasserre.farkas_certify import (
    _adj_t_fmpq, _adj_qW_exact_fmpq,
    FarkasCertificate, FarkasCertifyResult,
)


def _check_window_violations_float(y_vals: np.ndarray, t_val: float,
                                    P: dict, active_set: Set[int],
                                    tol: float = 1e-6):
    """Return list of (w_idx, min_eig_float) for non-active windows with
    significant violations. Sorted by severity.
    """
    n_loc = P['n_loc']
    ab_eiej_idx = P['ab_eiej_idx']
    if n_loc == 0 or ab_eiej_idx is None:
        return []
    t_pick_arr = np.asarray(P['t_pick'], dtype=np.int64)
    L_t = t_val * y_vals[t_pick_arr].reshape(n_loc, n_loc)
    safe_idx = np.clip(ab_eiej_idx, 0, len(y_vals) - 1)
    y_abij = y_vals[safe_idx]
    y_abij[ab_eiej_idx < 0] = 0.0

    violations = []
    nontrivial = P['nontrivial_windows']
    for w in nontrivial:
        if w in active_set:
            continue
        Mw = P['M_mats'][w]
        L_q = np.einsum('ij,abij->ab', Mw, y_abij)
        L_w = 0.5 * (L_t - L_q + (L_t - L_q).T)
        eigmin = float(np.linalg.eigvalsh(L_w)[0])
        if eigmin < -tol:
            violations.append((w, eigmin))
    violations.sort(key=lambda x: x[1])
    return violations


def _build_cg_base_model(P: dict, t_val: float):
    """Base model with moment + mu_i loc + equality constraints.
    Window-localizing PSDs are added incrementally via `_add_window_to_model`.
    """
    from mosek.fusion import (Model, Domain, Expr, Matrix, ObjectiveSense)

    d = P['d']
    n_y = P['n_y']
    n_loc = P['n_loc']
    n_basis = P['n_basis']
    moment_pick = P['moment_pick']
    loc_picks = P['loc_picks']
    idx = P['idx']

    M = Model('farkas_cg')
    y = M.variable('y', n_y, Domain.unbounded())

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    y0_con = M.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency
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
    cons_con = None
    if n_cons > 0:
        C_cons = Matrix.sparse(n_cons, n_y, c_rows, c_cols, c_vals)
        cons_con = M.constraint('consist', Expr.mul(C_cons, y),
                                 Domain.equalsTo([0.0] * n_cons))

    # Moment + mu_i localizing PSDs
    mom_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    mom_con = M.constraint('moment', mom_mat, Domain.inPSDCone(n_basis))
    loc_cons = []
    for i in range(d):
        lm = Expr.reshape(y.pick(list(loc_picks[i])), n_loc, n_loc)
        loc_cons.append(
            M.constraint(f'loc_{i}', lm, Domain.inPSDCone(n_loc)))

    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    return {
        'model': M,
        'y': y,
        'y0_con': y0_con,
        'cons_con': cons_con,
        'mom_con': mom_con,
        'loc_cons': loc_cons,
        't_val': t_val,
        'win_cons': {},        # dict: window_idx -> Constraint
    }


def _add_window_to_model(build: dict, P: dict, w: int):
    """Add window-localizing PSD constraint for window w to an existing model."""
    from mosek.fusion import (Expr, Matrix, Domain)
    M = build['model']
    y = build['y']
    t_val = build['t_val']
    n_loc = P['n_loc']
    n_y = P['n_y']
    t_pick = P['t_pick']
    ab_eiej_idx = P['ab_eiej_idx']
    ab_flat = P['ab_flat']
    M_mats = P['M_mats']
    flat_size = n_loc * n_loc

    t_times_y_picked = Expr.mul(t_val, y.pick(t_pick))
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
    c = M.constraint(f'wpsd_{w}', Lw_mat, Domain.inPSDCone(n_loc))
    build['win_cons'][w] = c


@dataclass
class CGResult:
    t_test: float
    status: str  # 'INFEAS' / 'FEAS' / 'ERROR'
    active_windows: List[int]
    n_iters: int
    solve_time: float
    build: Optional[dict]  # model + constraints for dual extraction
    mu_A_consist: Optional[np.ndarray]
    mu_A_y0: float
    S_mom: Optional[np.ndarray]
    S_loc: Optional[List[np.ndarray]]
    S_win_dict: dict  # window idx -> S_W float matrix


def run_cg_at_t(P: dict, t_val: float,
                 max_iters: int = 30,
                 violate_tol: float = 1e-6,
                 n_add_per_iter: int = 5,
                 initial_windows: Optional[List[int]] = None,
                 nthreads: int = 8,
                 verbose: bool = True) -> CGResult:
    """Run constraint generation at fixed t. Returns CGResult with
    INFEAS (Farkas certified) or FEAS (found feasible y) status.
    """
    from mosek.fusion import (SolutionStatus, ProblemStatus, AccSolutionStatus)

    if verbose:
        print(f'[cg] d={P["d"]} order={P["order"]} t={t_val:.6f}', flush=True)

    build = _build_cg_base_model(P, t_val)
    M = build['model']
    M.setSolverParam('numThreads', nthreads)
    M.setSolverParam('intpntCoTolPfeas', 1e-8)
    M.setSolverParam('intpntCoTolDfeas', 1e-8)
    M.setSolverParam('intpntCoTolRelGap', 1e-8)
    M.setSolverParam('intpntCoTolInfeas', 1e-8)
    M.acceptedSolutionStatus(AccSolutionStatus.Anything)

    if initial_windows:
        for w in initial_windows:
            _add_window_to_model(build, P, w)

    t_start = time.time()
    n_iters = 0
    for iter_k in range(max_iters):
        n_iters = iter_k + 1
        t_solve0 = time.time()
        M.solve()
        t_solve = time.time() - t_solve0
        ps = M.getProblemStatus()
        ss = M.getPrimalSolutionStatus()
        if verbose:
            active_count = len(build['win_cons'])
            print(f'  [{iter_k+1}] active={active_count:3d} solve={t_solve:.1f}s '
                  f'status={ps}/{ss}', flush=True)

        if ps == ProblemStatus.PrimalInfeasible:
            # INFEASIBILITY certified — return duals
            mu_A_y0 = float(build['y0_con'].dual()[0])
            if build['cons_con'] is not None:
                mu_A_cons = np.array(build['cons_con'].dual(), dtype=np.float64)
            else:
                mu_A_cons = np.zeros(0)
            S_mom = np.array(build['mom_con'].dual(),
                              dtype=np.float64).reshape(P['n_basis'], P['n_basis'])
            S_mom = 0.5 * (S_mom + S_mom.T)
            S_loc = []
            for c in build['loc_cons']:
                S = np.array(c.dual(), dtype=np.float64).reshape(P['n_loc'], P['n_loc'])
                S_loc.append(0.5 * (S + S.T))
            S_win_dict = {}
            for w, c in build['win_cons'].items():
                S = np.array(c.dual(), dtype=np.float64).reshape(P['n_loc'], P['n_loc'])
                S_win_dict[w] = 0.5 * (S + S.T)
            active_list = list(build['win_cons'].keys())
            if verbose:
                print(f'  INFEASIBILITY detected after {n_iters} iters '
                      f'with {len(active_list)} active windows, '
                      f'mu_A[0]={mu_A_y0:.3e}', flush=True)
            return CGResult(
                t_test=t_val, status='INFEAS',
                active_windows=active_list, n_iters=n_iters,
                solve_time=time.time() - t_start,
                build=build, mu_A_consist=mu_A_cons,
                mu_A_y0=mu_A_y0, S_mom=S_mom, S_loc=S_loc,
                S_win_dict=S_win_dict,
            )

        if ps != ProblemStatus.PrimalAndDualFeasible or ss != SolutionStatus.Optimal:
            # Unclear status — let's check primal anyway
            pass

        # Get primal y to check violations
        try:
            y_val = np.array(build['y'].level(), dtype=np.float64)
        except Exception as e:
            if verbose:
                print(f'  failed to get primal: {e}', flush=True)
            break

        # Check violations
        active_set = set(build['win_cons'].keys())
        violations = _check_window_violations_float(
            y_val, t_val, P, active_set, tol=violate_tol)
        if not violations:
            if verbose:
                print(f'  no violations — FEASIBLE at t={t_val}', flush=True)
            active_list = list(build['win_cons'].keys())
            return CGResult(
                t_test=t_val, status='FEAS',
                active_windows=active_list, n_iters=n_iters,
                solve_time=time.time() - t_start,
                build=None, mu_A_consist=None, mu_A_y0=0.0,
                S_mom=None, S_loc=None, S_win_dict={},
            )
        # Add most-violated windows
        to_add = [w for w, _ in violations[:n_add_per_iter]]
        if verbose:
            most_viol = violations[0][1] if violations else 0.0
            print(f'    {len(violations)} violating windows, adding '
                  f'{len(to_add)} (most: {most_viol:+.3e})', flush=True)
        for w in to_add:
            _add_window_to_model(build, P, w)

    # Exceeded max_iters
    if verbose:
        print(f'  max_iters reached without resolving status', flush=True)
    return CGResult(
        t_test=t_val, status='ERROR',
        active_windows=list(build['win_cons'].keys()),
        n_iters=n_iters, solve_time=time.time() - t_start,
        build=None, mu_A_consist=None, mu_A_y0=0.0,
        S_mom=None, S_loc=None, S_win_dict={},
    )


def farkas_certify_cg(d: int, order: int, t_test: float,
                       max_cg_iters: int = 30,
                       violate_tol: float = 1e-6,
                       n_add_per_iter: int = 5,
                       max_denom_S: int = 10**12,
                       max_denom_mu: int = 10**12,
                       eig_margin: float = 1e-10,
                       nthreads: int = 8,
                       verbose: bool = True,
                       ) -> Tuple[FarkasCertifyResult, Optional[FarkasCertificate]]:
    """Like farkas_certify_at but using constraint generation for speed."""
    if not _HAS_FLINT:
        raise ImportError("python-flint required")

    t_start = time.time()
    P = _precompute(d, order, verbose=False)

    if verbose:
        print(f'[farkas_certify_cg] d={d} order={order} t={t_test:.9f}', flush=True)

    cg = run_cg_at_t(
        P, t_test,
        max_iters=max_cg_iters,
        violate_tol=violate_tol,
        n_add_per_iter=n_add_per_iter,
        nthreads=nthreads,
        verbose=verbose,
    )

    if cg.status != 'INFEAS':
        notes = f'CG returned {cg.status} after {cg.n_iters} iters'
        return FarkasCertifyResult(
            d=d, order=order,
            t_test=Fraction(0), t_test_float=t_test,
            lb_rig=Fraction(0), lb_rig_decimal='n/a',
            status=cg.status,
            mu0_float=cg.mu_A_y0, residual_l1_float=0.0,
            safety_margin_float=0.0,
            moment_l1_bound=Fraction(2 * order + 1),
            solver_time=cg.solve_time, round_time=0.0,
            total_time=time.time() - t_start,
            notes=notes,
        ), None

    # Infeasibility: run rational verification on active windows
    if verbose:
        print(f'  rational verification on {len(cg.active_windows)} '
              f'active windows...', flush=True)

    n_y = P['n_y']
    n_loc = P['n_loc']
    n_basis = P['n_basis']
    windows = P['windows']
    M_mats = P['M_mats']
    t_pick_np = np.asarray(P['t_pick'], dtype=np.int64)
    ab_eiej_idx = P['ab_eiej_idx']
    A_full, b_full, _ = _build_equality_constraints(P)

    t0 = time.time()
    # Build mu_A_full = [y0_dual, consistency_duals]
    mu_A_full = np.concatenate([[cg.mu_A_y0], cg.mu_A_consist])
    mu_A_fmpq = _round_vec_fmpq(mu_A_full, max_denom_mu)
    t_test_frac = Fraction(float(t_test)).limit_denominator(10**12)
    t_test_fmpq = flint.fmpq(t_test_frac.numerator, t_test_frac.denominator)

    # Rational Cholesky with margin
    def _chol_with_margin(S_float, max_denom, margin):
        S_sym = 0.5 * (S_float + S_float.T)
        w, V = np.linalg.eigh(S_sym)
        w_pos = np.maximum(w, 0.0) + float(margin)
        L_float = V * np.sqrt(w_pos)[None, :]
        return _round_mat_fmpq(L_float, max_denom)

    # Base blocks
    L_mom = _chol_with_margin(cg.S_mom, max_denom_S, eig_margin)
    S_mom_fmpq = L_mom * L_mom.transpose()
    L_loc_list = [_chol_with_margin(S, max_denom_S, eig_margin) for S in cg.S_loc]
    S_loc_fmpq = [L * L.transpose() for L in L_loc_list]

    # Active windows only
    S_win_fmpq = {}
    for w, S_float in cg.S_win_dict.items():
        L_fm = _chol_with_margin(S_float, max_denom_S, eig_margin)
        S_win_fmpq[w] = L_fm * L_fm.transpose()

    if verbose:
        print(f'  chol done in {time.time()-t0:.1f}s', flush=True)

    # Residual
    t_res = time.time()
    r = [flint.fmpq(0) for _ in range(n_y)]
    AT_mu = _sparse_matvec_fmpq(A_full.T.tocsr(), mu_A_fmpq)
    for k in range(n_y):
        r[k] += AT_mu[k]

    base_blocks: List[PSDBlock] = [_build_moment_block(P)]
    base_blocks.extend(_build_loc_blocks(P))
    base_S_fmpq = [S_mom_fmpq] + S_loc_fmpq
    for blk, S_fm in zip(base_blocks, base_S_fmpq):
        adj = _adjoint_block_fmpq(blk, S_fm)
        for k in range(n_y):
            r[k] += adj[k]

    # Window-localizing contributions — parallel when many active
    try:
        from certified_lasserre.parallel_adj import parallel_window_residuals
        n_active = len(S_win_fmpq)
        use_parallel = n_active >= 5
    except ImportError:
        use_parallel = False
    if use_parallel:
        import os
        n_workers = min(16, os.cpu_count() or 4, n_active)
        if verbose:
            print(f'  computing {n_active} window residuals in parallel '
                  f'({n_workers} workers)...', flush=True)
        win_contrib = parallel_window_residuals(
            S_win_fmpq, windows, P, t_test_fmpq, d, n_workers=n_workers,
        )
        for k in range(n_y):
            r[k] += win_contrib[k]
    else:
        for w, S_fm in S_win_fmpq.items():
            ell, s_lo = windows[w]
            coeff = flint.fmpq(2 * d, ell)
            adj_t = _adj_t_fmpq(S_fm, t_pick_np, n_y)
            for k in range(n_y):
                r[k] += t_test_fmpq * adj_t[k]
            if ab_eiej_idx is None:
                continue
            Mw_support = (M_mats[w] != 0).astype(np.int64)
            adj_q = _adj_qW_exact_fmpq(S_fm, ab_eiej_idx, Mw_support, coeff, n_y)
            for k in range(n_y):
                r[k] -= adj_q[k]

    res_l1 = flint.fmpq(0)
    for v in r:
        res_l1 += abs(v)
    res_l1_float = _fmpq_to_float(res_l1)

    mu0 = mu_A_fmpq[0]
    mu0_float = _fmpq_to_float(mu0)

    M_y = Fraction(2 * order + 1)
    M_y_fmpq = flint.fmpq(M_y.numerator, M_y.denominator)
    safety_threshold = res_l1 * M_y_fmpq
    safety_margin = mu0 - safety_threshold
    safety_margin_float = _fmpq_to_float(safety_margin)
    infeasibility_certified = safety_margin > 0
    res_time = time.time() - t_res

    total_time = time.time() - t_start
    if verbose:
        print(f'  residual time={res_time:.1f}s', flush=True)
        print(f'  mu0={mu0_float:.3e}  ||r||_1={res_l1_float:.3e}  '
              f'margin={safety_margin_float:+.3e}', flush=True)
        if infeasibility_certified:
            print(f'  FARKAS CERTIFIED: val(d) > t={_decimal_str(t_test_frac, 15)}',
                  flush=True)
        else:
            print(f'  Farkas FAILED: margin <= 0', flush=True)

    lb_rig_frac = t_test_frac if infeasibility_certified else Fraction(0)
    result = FarkasCertifyResult(
        d=d, order=order,
        t_test=t_test_frac, t_test_float=t_test,
        lb_rig=lb_rig_frac,
        lb_rig_decimal=(_decimal_str(lb_rig_frac, 15) if lb_rig_frac != 0 else 'not_certified'),
        status='CERTIFIED' if infeasibility_certified else 'NOT_CERTIFIED',
        mu0_float=mu0_float,
        residual_l1_float=res_l1_float,
        safety_margin_float=safety_margin_float,
        moment_l1_bound=M_y,
        solver_time=cg.solve_time,
        round_time=total_time - cg.solve_time,
        total_time=total_time,
        notes=f'CG: {cg.n_iters} iters, {len(cg.active_windows)} active windows',
    )
    return result, None

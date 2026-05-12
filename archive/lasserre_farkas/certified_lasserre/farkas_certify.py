"""Farkas-style rigorous certification of val(d) >= t_lo via SDP infeasibility.

The Lasserre L_k relaxation of val(d) = min_mu max_W mu^T M_W mu is:
    min t
    s.t. L_W(y, t) := t*M_{k-1}(y) - M_{k-1}(q_W y) succeq 0   for each W
         F_0(y) := M_k(y)            succeq 0                   (moment)
         F_i(y) := M_{k-1}(mu_i y)   succeq 0   for i=1..d     (simplex mu_i>=0)
         Ay = b                                                  (y_0=1 + consistency)

For fixed t, this is a feasibility SDP in y. If INFEASIBLE at t = t_test, then
L_k(d) > t_test, hence val(d) > t_test.

Farkas infeasibility certificate: exists dual multipliers
    mu_A in R^{n_eq}  (multipliers on Ay=b)
    S_j succeq 0      (multipliers on F_j(y) succeq 0)
    S_W succeq 0      (multipliers on L_W(y, t_test) succeq 0)
such that:
    (1)  [stationarity in y]
         A^T mu_A  +  sum_j adj(F_j)(S_j)  +  sum_W adj_W(S_W; t_test)  =  0  in R^{n_y}
    (2)  [strict cost inequality]
         b^T mu_A  +  sum_j <F_j(0), S_j>  +  sum_W <L_W(0; t_test), S_W>  >  0
         Since F_j(0)=0 and L_W(0; t_test)=0, this reduces to:
             mu_A[0]  >  0

where adj_W(S_W; t)[alpha]
    = <S_W, d L_W / d y_alpha at t>
    = t * adj_t(S_W)[alpha]  -  adj_qW(S_W)[alpha]

with
    adj_t(S_W)[alpha]  = sum_{a,b: t_pick[ab]=alpha} S_W[a,b]
    adj_qW(S_W)[alpha] = sum_{a,b,i,j: ab_eiej_idx[a,b,i,j]=alpha} M_W[i,j] * S_W[a,b]

SAFE BOUND (when the residual r = lhs of (1) is not exactly zero in rationals):
For any primal-feasible y with Ay=b and F_j(y), L_W(y) PSD, we have
    0 <= b^T mu_A - <r, y>  +  (nonneg terms)
hence
    mu_A[0] >= <r, y> >= -||r||_1 * M_y
where M_y = (2*order + 1) is an upper bound on ||y||_1 for probability moments.

Infeasibility follows if  mu_A[0] - ||r||_1 * M_y > 0.

Rigorous lower bound on val(d):
    if we certify infeasibility at t_test, then val(d) > t_test.

To report a rational lower bound t_cert ≈ val(d) - epsilon, pick t_test slightly below
the optimal L_k(d) (found by float bisection) and verify Farkas with safety margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple
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


def _adj_t_fmpq(S_fmpq_mat, t_pick_np: np.ndarray, n_y: int) -> list:
    """adj_t(S)[alpha] = sum_{a,b: t_pick[ab]=alpha} S[a,b].

    S is fmpq_mat (n_loc, n_loc). t_pick is np.int array of length n_loc^2.
    """
    nl = S_fmpq_mat.nrows()
    out = [flint.fmpq(0) for _ in range(n_y)]
    # Materialize S entries once, then iterate in Python with list access
    t_pick_list = t_pick_np.tolist()
    nsq = nl * nl
    for k in range(nsq):
        alpha = t_pick_list[k]
        if alpha < 0:
            continue
        a = k // nl
        b = k - a * nl
        out[alpha] += S_fmpq_mat[a, b]
    return out


def _adj_qW_fmpq(S_fmpq_mat, ab_eiej_idx: np.ndarray,
                 M_W: np.ndarray, n_y: int) -> list:
    """adj_qW(S_W)[alpha] = sum_{a,b,i,j} M_W[i,j] * S_W[a,b]
                           where ab_eiej_idx[a,b,i,j] = alpha.
    """
    nl = S_fmpq_mat.nrows()
    d = M_W.shape[0]
    out = [flint.fmpq(0) for _ in range(n_y)]
    # M_W is d x d with values that are fmpq (from coefficient 2d/ell * 0/1)
    # Strategy: for each nonzero (i, j) of M_W, loop over (a, b).
    nz_i, nz_j = np.nonzero(M_W)
    if len(nz_i) == 0:
        return out
    for idx_ij in range(len(nz_i)):
        i = int(nz_i[idx_ij])
        j = int(nz_j[idx_ij])
        m_val = M_W[i, j]  # python float
        # Convert to fmpq exactly: M_W entries are (2d/ell) * integer_0_or_1
        # The caller should supply M_W with rational-friendly values.
        m_fmpq = flint.fmpq(int(round(m_val * 10**9)), 10**9)  # approx
        # Better: for exact rationals, pass ell to compute 2d/ell as fmpq.
        # We do this here using an approximation — caller uses exact version below.
        for a in range(nl):
            for b in range(nl):
                alpha = int(ab_eiej_idx[a, b, i, j])
                if alpha < 0:
                    continue
                out[alpha] += m_fmpq * S_fmpq_mat[a, b]
    return out


def _adj_qW_exact_fmpq(S_fmpq_mat, ab_eiej_idx: np.ndarray,
                        M_W_support: np.ndarray, coeff_fmpq,
                        n_y: int) -> list:
    """adj_qW(S_W)[alpha] with M_W = coeff * M_W_support (0/1 indicator).

    M_W_support is (d,d) 0/1 int. coeff_fmpq is fmpq (= 2d/ell).

    Optimizations:
      1. Materialize S_fmpq_mat entries as a flat Python list once (C-level
         fmpq creates; avoids per-access overhead).
      2. Convert alpha/a/b numpy arrays to Python lists (avoid int() in hot loop).
      3. Flat-index S lookup: S_flat[a*nl + b] instead of S_mat[a, b].
    """
    nl = S_fmpq_mat.nrows()
    nz_i, nz_j = np.nonzero(M_W_support)
    if len(nz_i) == 0:
        return [flint.fmpq(0) for _ in range(n_y)]
    idx_slice = ab_eiej_idx[:, :, nz_i, nz_j]  # (nl, nl, K)
    valid_mask = idx_slice >= 0
    if not valid_mask.any():
        return [flint.fmpq(0) for _ in range(n_y)]

    # Pre-materialize S_fmpq_mat entries in a flat list (indexed by a*nl+b)
    S_flat = [None] * (nl * nl)
    for a in range(nl):
        for b in range(nl):
            S_flat[a * nl + b] = S_fmpq_mat[a, b]

    # Broadcast ab index (a*nl+b) without materializing a, b arrays
    ab_arr = (np.arange(nl)[:, None, None] * nl + np.arange(nl)[None, :, None])
    ab_arr = np.broadcast_to(ab_arr, idx_slice.shape)
    alpha_flat = idx_slice[valid_mask].ravel()
    ab_flat = ab_arr[valid_mask].ravel()

    # Convert to Python lists for fast loop access
    alpha_list = alpha_flat.tolist()
    ab_list = ab_flat.tolist()
    N = len(alpha_list)

    accum = [flint.fmpq(0) for _ in range(n_y)]
    for k in range(N):
        accum[alpha_list[k]] += S_flat[ab_list[k]]

    out = [flint.fmpq(0) for _ in range(n_y)]
    for alpha in range(n_y):
        a_val = accum[alpha]
        if a_val != 0:
            out[alpha] = coeff_fmpq * a_val
    return out


def _sparse_matvec_fmpq_from_AT(A_csr: sp.csr_matrix, v_fmpq: list) -> list:
    """A^T @ v where A is n_eq x n_y and v is fmpq list of len n_eq.
    Returns list of len n_y.
    """
    return _sparse_matvec_fmpq(A_csr.T.tocsr(), v_fmpq)


# =====================================================================
# Full Farkas feasibility SDP build + solve at fixed t_test
# =====================================================================

def _build_feasibility_at_t(P: dict, t_val: float,
                             cons_tol: float = 1e-9):
    """Build the fixed-t feasibility SDP. Returns (Model, handles).

    Identical to joint_bisect._build_feasibility_model_at_t but with
    references organized for dual extraction.
    """
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

    M = Model('farkas_feas')
    y = M.variable('y', n_y, Domain.unbounded())

    # y_0 = 1
    zero = tuple(0 for _ in range(d))
    y0_con = M.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

    # Consistency
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    c_rows, c_cols, c_vals = [], [], []
    n_cons = 0
    consist_order = []  # which consist_mono index each row corresponds to
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
            consist_order.append(r)
            n_cons += 1
    cons_con = None
    if n_cons > 0:
        C_cons = Matrix.sparse(n_cons, n_y, c_rows, c_cols, c_vals)
        cons_con = M.constraint('consist', Expr.mul(C_cons, y),
                                 Domain.equalsTo([0.0] * n_cons))

    # Base PSD: moment + mu_i loc
    mom_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
    mom_con = M.constraint('moment', mom_mat, Domain.inPSDCone(n_basis))

    loc_cons = []
    for i in range(d):
        lm = Expr.reshape(y.pick(list(loc_picks[i])), n_loc, n_loc)
        loc_cons.append(
            M.constraint(f'loc_{i}', lm, Domain.inPSDCone(n_loc)))

    # Window-localizing PSD
    t_times_y_picked = Expr.mul(t_val, y.pick(t_pick))
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

    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

    return {
        'model': M,
        'y': y,
        'y0_con': y0_con,
        'cons_con': cons_con,
        'mom_con': mom_con,
        'loc_cons': loc_cons,
        'win_cons': win_cons,
        'n_cons': n_cons,
    }


@dataclass
class FarkasCertificate:
    d: int
    order: int
    t_test: Fraction             # the t value at which infeasibility is certified
    mu_A: np.ndarray             # (n_eq,) Fractions, with mu_A[0] corresponding to y_0=1
    S_mom: np.ndarray            # (n_basis, n_basis) Fractions
    S_loc: List[np.ndarray]      # d matrices (n_loc, n_loc) Fractions
    S_win: List[np.ndarray]      # n_win matrices (n_loc, n_loc) Fractions
    residual_abs_sum: Fraction
    moment_l1_bound: Fraction
    mu0: Fraction                # mu_A[0]
    safety_margin: Fraction      # = mu0 - residual_abs_sum * moment_l1_bound  > 0 required
    infeasibility_certified: bool


@dataclass
class FarkasCertifyResult:
    d: int
    order: int
    t_test: Fraction
    t_test_float: float
    lb_rig: Fraction             # a rational < t_test provably < val(d). Set to t_test (- eps) if certified.
    lb_rig_decimal: str
    status: str
    mu0_float: float
    residual_l1_float: float
    safety_margin_float: float
    moment_l1_bound: Fraction
    solver_time: float
    round_time: float
    total_time: float
    notes: str = ''


def farkas_certify_at(d: int, order: int, t_test: float,
                       max_denom_S: int = 10**12,
                       max_denom_mu: int = 10**12,
                       eig_margin: float = 1e-10,
                       nthreads: int = 16,
                       verbose: bool = True,
                       use_fast_residual: bool = True,
                       fast_D_L: int = 10**4,
                       fast_use_bignum: bool = False,
                       _fast_residual_cache: Optional[dict] = None,
                       _precompute_cache: Optional[dict] = None,
                       ) -> Tuple[FarkasCertifyResult, Optional[FarkasCertificate]]:
    """Try to produce a rigorous Farkas infeasibility certificate at t=t_test.

    If the feasibility SDP is INFEASIBLE at t_test, we extract the dual
    certificate (mu_A, S_j, S_W), round to rationals with a PSD margin, and
    verify the stationarity + strict positivity conditions.

    Returns (result, cert_or_None).
    """
    if not _HAS_FLINT:
        raise ImportError("python-flint required")

    import mosek
    from mosek.fusion import (SolutionStatus, ProblemStatus, AccSolutionStatus)

    t_start = time.time()
    if _precompute_cache is not None and (d, order) in _precompute_cache:
        P = _precompute_cache[(d, order)]
    else:
        P = _precompute(d, order, verbose=False)
        if _precompute_cache is not None:
            _precompute_cache[(d, order)] = P

    n_y = P['n_y']
    n_loc = P['n_loc']
    n_basis = P['n_basis']
    windows = P['windows']
    M_mats = P['M_mats']
    t_pick_np = np.asarray(P['t_pick'], dtype=np.int64)
    ab_eiej_idx = P['ab_eiej_idx']

    if verbose:
        print(f'[farkas_certify_at] d={d} order={order} '
              f't_test={t_test:.9f}')
        print(f'  n_y={n_y} n_loc={n_loc} n_basis={n_basis} '
              f'n_win={len(windows)}', flush=True)

    # Build & solve
    t0 = time.time()
    build = _build_feasibility_at_t(P, t_test)
    M = build['model']
    try:
        M.setSolverParam('numThreads', nthreads)
        # Looser tolerances for better infeasibility detection; rounding compensates.
        M.setSolverParam('intpntCoTolPfeas', 1e-8)
        M.setSolverParam('intpntCoTolDfeas', 1e-8)
        M.setSolverParam('intpntCoTolRelGap', 1e-8)
        M.setSolverParam('intpntCoTolInfeas', 1e-8)
        # Accept certificate solutions (required when problem is infeasible)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()
        ps = M.getProblemStatus()
        ss = M.getPrimalSolutionStatus()
        dss = M.getDualSolutionStatus()
        solver_time = time.time() - t0
        if verbose:
            print(f'  solver: primal_status={ss}, dual_status={dss}, '
                  f'problem_status={ps}, time={solver_time:.2f}s')

        # Check infeasibility status
        is_primal_infeasible = (ps == ProblemStatus.PrimalInfeasible)
        is_feasible = (ps == ProblemStatus.PrimalAndDualFeasible and
                       ss == SolutionStatus.Optimal)

        if not is_primal_infeasible:
            if is_feasible:
                return FarkasCertifyResult(
                    d=d, order=order,
                    t_test=Fraction(0), t_test_float=t_test,
                    lb_rig=Fraction(0),
                    lb_rig_decimal='n/a',
                    status='feasible_at_t_test',
                    mu0_float=0.0, residual_l1_float=0.0, safety_margin_float=0.0,
                    moment_l1_bound=Fraction(2 * order + 1),
                    solver_time=solver_time, round_time=0.0,
                    total_time=time.time() - t_start,
                    notes='SDP feasible at t_test — try larger t_test or '
                          'lower order.'
                ), None
            else:
                return FarkasCertifyResult(
                    d=d, order=order,
                    t_test=Fraction(0), t_test_float=t_test,
                    lb_rig=Fraction(0),
                    lb_rig_decimal='n/a',
                    status=f'other:{ps}/{ss}',
                    mu0_float=0.0, residual_l1_float=0.0, safety_margin_float=0.0,
                    moment_l1_bound=Fraction(2 * order + 1),
                    solver_time=solver_time, round_time=0.0,
                    total_time=time.time() - t_start,
                    notes='Unexpected solver status.'
                ), None

        # Extract Farkas dual (from the infeasibility certificate)
        mu_A_y0 = float(build['y0_con'].dual()[0])
        if build['cons_con'] is not None:
            mu_A_cons = np.array(build['cons_con'].dual(), dtype=np.float64)
        else:
            mu_A_cons = np.zeros(0)
        mu_A_full = np.concatenate([[mu_A_y0], mu_A_cons])

        S_mom_float = np.array(build['mom_con'].dual(),
                                dtype=np.float64).reshape(n_basis, n_basis)
        S_mom_float = 0.5 * (S_mom_float + S_mom_float.T)

        S_loc_float = []
        for c in build['loc_cons']:
            S = np.array(c.dual(), dtype=np.float64).reshape(n_loc, n_loc)
            S_loc_float.append(0.5 * (S + S.T))

        S_win_float = []
        for c in build['win_cons']:
            S = np.array(c.dual(), dtype=np.float64).reshape(n_loc, n_loc)
            S_win_float.append(0.5 * (S + S.T))

        # Identify ACTIVE windows: those with significant trace (dual
        # nonzero). Inactive windows have S_W ~ 0 and contribute zero.
        win_trace = np.array([float(np.trace(S)) for S in S_win_float])
        mu_A_abs = abs(mu_A_y0)
        # Active threshold: trace > 1e-4 * max_trace (heuristic)
        max_trace = max(win_trace.max(), 1e-30)
        active_windows = np.where(win_trace > 1e-6 * max_trace)[0]
        if verbose:
            print(f'  Farkas certificate extracted:')
            print(f'    mu_A[0] = {mu_A_y0:.6e}')
            print(f'    S_mom min_eig = {float(np.linalg.eigvalsh(S_mom_float)[0]):+.3e}')
            print(f'    S_loc min_eigs = {[float(np.linalg.eigvalsh(S)[0]) for S in S_loc_float[:3]]} ...')
            print(f'    active windows: {len(active_windows)}/{len(S_win_float)} '
                  f'(trace max={max_trace:.2e})')
    finally:
        M.dispose()

    # === Rational rounding + exact Farkas verification ===
    t0 = time.time()

    # Build A (equality constraints) so we can compute A^T @ mu_A
    A_full, b_full, eq_names = _build_equality_constraints(P)

    # t_test as fmpq (used by both paths)
    t_test_frac = Fraction(float(t_test)).limit_denominator(10**12)
    t_test_fmpq = flint.fmpq(t_test_frac.numerator, t_test_frac.denominator)

    # Fast path branches here and returns early (see the use_fast_residual block below).
    # The slow path continues with limit_denominator-based fmpq rounding.
    if not use_fast_residual:
        # Round mu_A to fmpq via limit_denominator (slow)
        mu_A_fmpq = _round_vec_fmpq(mu_A_full, max_denom_mu)
    else:
        mu_A_fmpq = None  # computed inside the fast block

    # Rational Cholesky with PSD margin — BATCH eigendecomposition
    # (numpy batches multiple eigh calls via np.linalg.eigh on 3D array)
    def _chol_with_margin(S_float, max_denom, margin):
        S_sym = 0.5 * (S_float + S_float.T)
        w, V = np.linalg.eigh(S_sym)
        w_pos = np.maximum(w, 0.0) + float(margin)
        L_float = V * np.sqrt(w_pos)[None, :]
        return _round_mat_fmpq(L_float, max_denom)

    def _chol_batch(S_list_float, max_denom, margin):
        """Batch eigendecompose a list of S matrices, then round to fmpq in parallel."""
        if not S_list_float:
            return []
        # Stack into (N, n, n) if all same size
        sizes = set(S.shape for S in S_list_float)
        if len(sizes) == 1:
            S_stack = np.stack(S_list_float, axis=0)
            S_stack = 0.5 * (S_stack + S_stack.transpose(0, 2, 1))
            w_stack, V_stack = np.linalg.eigh(S_stack)   # batched LAPACK call
            w_pos_stack = np.maximum(w_stack, 0.0) + float(margin)
            L_stack = V_stack * np.sqrt(w_pos_stack)[:, None, :]
            return [_round_mat_fmpq(L_stack[k], max_denom) for k in range(L_stack.shape[0])]
        else:
            return [_chol_with_margin(S, max_denom, margin) for S in S_list_float]

    if verbose:
        print(f'  rounding...', flush=True)

    # =================================================================
    # FAST PATH: fixed-denominator int64 residual.
    # =================================================================
    if use_fast_residual:
        from certified_lasserre.fast_residual import (
            build_residual_precomp, compute_residual_fast,
            chol_round_int_product, chol_round_bignum_product,
            round_vec_fixed_denom,
        )
        D_L = int(fast_D_L)
        D_S = D_L * D_L
        D_W = D_L * D_L
        D_mu = int(max_denom_mu)
        _chol_fn = chol_round_bignum_product if fast_use_bignum else chol_round_int_product

        # Round everything in one pass (vectorized).
        S_mom_num, _ = _chol_fn(S_mom_float, eig_margin, D_L)
        S_loc_num = [_chol_fn(S, eig_margin, D_L)[0]
                     for S in S_loc_float]
        S_win_num: List[Optional[np.ndarray]] = [None] * len(S_win_float)
        active_list = active_windows.tolist()
        for w in active_list:
            Sn, _ = _chol_fn(S_win_float[w], eig_margin, D_L)
            S_win_num[w] = Sn
        mu_A_num = round_vec_fixed_denom(mu_A_full, D_mu)
        mu_A_fmpq = [flint.fmpq(int(x), D_mu) for x in mu_A_num.tolist()]

        if verbose:
            n_active_actual = sum(1 for S in S_win_num if S is not None)
            print(f'  rounded {n_active_actual} active window duals '
                  f'(fast: fixed D_L={D_L})', flush=True)
            print(f'  computing rational residual (fast path)...', flush=True)

        # Precompute scatter indices once per (d, order).  Cached across probes
        # via the _fast_residual_cache dict the caller optionally passes in.
        base_blocks: List[PSDBlock] = [_build_moment_block(P)]
        base_blocks.extend(_build_loc_blocks(P))
        cache_key = (d, order)
        if _fast_residual_cache is not None and cache_key in _fast_residual_cache:
            pre = _fast_residual_cache[cache_key]
        else:
            pre = build_residual_precomp(P, base_blocks)
            if _fast_residual_cache is not None:
                _fast_residual_cache[cache_key] = pre

        r = compute_residual_fast(
            pre=pre,
            A_csr=A_full,
            mu_A_num=mu_A_num, D_mu=D_mu,
            base_S_num=[S_mom_num] + S_loc_num, D_S=D_S,
            win_S_num=S_win_num, D_W=D_W,
            t_test_fmpq=t_test_fmpq,
        )
    else:
        # =============================================================
        # SLOW PATH: original limit_denominator + fmpq loops.  Kept for
        # validation; slower by ~10x at d=4 order=3 per benchmark.
        # =============================================================
        L_mom = _chol_with_margin(S_mom_float, max_denom_S, eig_margin)
        S_mom_fmpq = L_mom * L_mom.transpose()

        L_loc_list = [_chol_with_margin(S, max_denom_S, eig_margin) for S in S_loc_float]
        S_loc_fmpq = [L * L.transpose() for L in L_loc_list]

        # Only round ACTIVE windows (inactive ones have S_W=0 exactly -> skip).
        # Batch eigendecompose active windows together (LAPACK batched).
        active_set = set(active_windows.tolist())
        active_S_list = [S_win_float[w] for w in active_windows]
        active_L_list = _chol_batch(active_S_list, max_denom_S, eig_margin)
        L_win_list = [None] * len(S_win_float)
        S_win_fmpq = [None] * len(S_win_float)
        for k, w in enumerate(active_windows.tolist()):
            L_fm = active_L_list[k]
            L_win_list[w] = L_fm
            S_win_fmpq[w] = L_fm * L_fm.transpose()
        if verbose:
            n_active_actual = sum(1 for S in S_win_fmpq if S is not None)
            print(f'  rounded {n_active_actual} active window duals', flush=True)

        if verbose:
            print(f'  computing rational residual (slow path)...', flush=True)

        # Residual in R^{n_y}:
        r = [flint.fmpq(0) for _ in range(n_y)]

        # A^T mu_A
        AT_mu = _sparse_matvec_fmpq(A_full.T.tocsr(), mu_A_fmpq)
        for k in range(n_y):
            r[k] += AT_mu[k]

        # adj(F_0)(S_mom) and adj(F_i)(S_loc_i)
        base_blocks: List[PSDBlock] = [_build_moment_block(P)]
        base_blocks.extend(_build_loc_blocks(P))
        base_S_fmpq = [S_mom_fmpq] + S_loc_fmpq
        for blk, S_fm in zip(base_blocks, base_S_fmpq):
            adj = _adjoint_block_fmpq(blk, S_fm)
            for k in range(n_y):
                r[k] += adj[k]

        # Window-localizing contributions (skip inactive windows)
        d_ = d
        for w, S_fm in enumerate(S_win_fmpq):
            if S_fm is None:
                continue  # inactive window
            ell, s_lo = windows[w]
            coeff = flint.fmpq(2 * d_, ell)

            # adj_t(S_W) * t_test
            adj_t = _adj_t_fmpq(S_fm, t_pick_np, n_y)
            for k in range(n_y):
                r[k] += t_test_fmpq * adj_t[k]

            # - adj_qW(S_W)
            if ab_eiej_idx is None:
                continue
            Mw_support = (M_mats[w] != 0).astype(np.int64)
            adj_q = _adj_qW_exact_fmpq(S_fm, ab_eiej_idx, Mw_support, coeff, n_y)
            for k in range(n_y):
                r[k] -= adj_q[k]

    # ||r||_1 (common to both paths)
    res_l1 = flint.fmpq(0)
    for v in r:
        res_l1 += abs(v)
    res_l1_float = _fmpq_to_float(res_l1)

    # mu_A[0]
    mu0 = mu_A_fmpq[0]
    mu0_float = _fmpq_to_float(mu0)

    # Safety: mu0 > ||r||_1 * M_y?
    M_y = Fraction(2 * order + 1)
    M_y_fmpq = flint.fmpq(M_y.numerator, M_y.denominator)
    safety_threshold = res_l1 * M_y_fmpq
    safety_margin = mu0 - safety_threshold
    safety_margin_float = _fmpq_to_float(safety_margin)
    infeasibility_certified = safety_margin > 0

    round_time = time.time() - t0

    if verbose:
        print(f'  mu0 = {mu0_float:.6e}')
        print(f'  ||r||_1 = {res_l1_float:.3e}')
        print(f'  safety_threshold = ||r||_1 * (2k+1) = {_fmpq_to_float(safety_threshold):.3e}')
        print(f'  safety_margin = mu0 - thresh = {safety_margin_float:.6e}')
        if infeasibility_certified:
            print(f'  FARKAS CERTIFIED: val(d) > t_test = {_decimal_str(t_test_frac, 15)}')
        else:
            print(f'  Farkas FAILED: margin <= 0. Need tighter rounding or bigger t_test.')

    total_time = time.time() - t_start

    if infeasibility_certified:
        lb_rig_frac = t_test_frac
    else:
        lb_rig_frac = Fraction(0)

    # Pack (skip storing S_win full matrix for space reasons)
    cert = FarkasCertificate(
        d=d, order=order,
        t_test=t_test_frac,
        mu_A=np.array([_fmpq_to_frac(v) for v in mu_A_fmpq], dtype=object),
        S_mom=np.array([[_fmpq_to_frac(S_mom_fmpq[i, j])
                         for j in range(n_basis)]
                        for i in range(n_basis)], dtype=object)
                if False else np.empty(0, dtype=object),
        S_loc=[], S_win=[],
        residual_abs_sum=_fmpq_to_frac(res_l1),
        moment_l1_bound=M_y,
        mu0=_fmpq_to_frac(mu0),
        safety_margin=_fmpq_to_frac(safety_margin),
        infeasibility_certified=infeasibility_certified,
    )
    result = FarkasCertifyResult(
        d=d, order=order,
        t_test=t_test_frac,
        t_test_float=t_test,
        lb_rig=lb_rig_frac,
        lb_rig_decimal=(_decimal_str(lb_rig_frac, 15) if lb_rig_frac != 0 else 'not_certified'),
        status='CERTIFIED' if infeasibility_certified else 'NOT_CERTIFIED',
        mu0_float=mu0_float,
        residual_l1_float=res_l1_float,
        safety_margin_float=safety_margin_float,
        moment_l1_bound=M_y,
        solver_time=solver_time,
        round_time=round_time,
        total_time=total_time,
    )
    return result, cert


def farkas_certify_bisect(d: int, order: int,
                           t_lo: float = 1.0, t_hi: float = 1.5,
                           tol: float = 1e-5, max_bisect: int = 30,
                           max_denom_S: int = 10**9,
                           max_denom_mu: int = 10**10,
                           eig_margin: float = 1e-9,
                           nthreads: int = 8,
                           verbose: bool = True,
                           use_fast_residual: bool = True,
                           fast_D_L: int = 10**4,
                           fast_use_bignum: bool = False,
                           ) -> FarkasCertifyResult:
    """Bisect to find the largest t such that Farkas infeasibility is
    certifiable, returning a rigorous lower bound on val(d).
    """
    best_result = None
    best_lb = Fraction(0)
    probe_lo, probe_hi = t_lo, t_hi
    # Reuse the (d, order) scatter cache and precompute cache across all probes.
    fast_cache: dict = {} if use_fast_residual else None
    pc_cache: dict = {}
    for step in range(max_bisect):
        t_try = 0.5 * (probe_lo + probe_hi)
        if verbose:
            print(f'\n[bisect {step+1}/{max_bisect}] probe t={t_try:.8f} '
                  f'(bracket=[{probe_lo:.6f}, {probe_hi:.6f}])', flush=True)
        res, cert = farkas_certify_at(
            d=d, order=order, t_test=t_try,
            max_denom_S=max_denom_S, max_denom_mu=max_denom_mu,
            eig_margin=eig_margin, nthreads=nthreads, verbose=False,
            use_fast_residual=use_fast_residual, fast_D_L=fast_D_L,
            fast_use_bignum=fast_use_bignum,
            _fast_residual_cache=fast_cache,
            _precompute_cache=pc_cache,
        )
        if verbose:
            print(f'  -> {res.status}, mu0={res.mu0_float:.3e}, '
                  f'||r||_1={res.residual_l1_float:.3e}, '
                  f'margin={res.safety_margin_float:+.3e}')
        if res.status == 'CERTIFIED':
            best_result = res
            best_lb = res.lb_rig
            probe_lo = t_try  # can push higher
        else:
            probe_hi = t_try  # drop back
        if probe_hi - probe_lo < tol:
            break

    if best_result is None:
        raise RuntimeError(f'Could not certify any t in [{t_lo}, {t_hi}]')
    return best_result

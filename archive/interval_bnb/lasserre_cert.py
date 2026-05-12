"""Lasserre SDP leaf-box certifier.

For a box B = [lo, hi] in R^d and target t, we want to prove
    min_{mu in B cap Delta_d}  max_W  mu^T M_W mu  >=  t.

The Lasserre order-2 SDP relaxation builds pseudo-moments y_alpha for
|alpha| <= 4 over the variables mu_0, ..., mu_{d-1}, and constructs

    M_2(y)  (moment matrix, size binom(d+2, 2) = (d+1)(d+2)/2 )
    L_i(y)  = M_1((mu_i - lo_i) y)     (box-low localizing)
    U_i(y)  = M_1((hi_i - mu_i) y)     (box-high localizing)

All required to be PSD. The LP bounds y_0 = 1, sum y_i = 1 (simplex),
y_alpha >= 0 (mu^alpha nonneg for alpha in R_+^d), y_alpha = sum_i
y_{alpha + e_i} (consistency from sum mu = 1).

For each window W we add the window constraint

    u  >=  sum_{(i,j) in pairs_all(W)}  scale_W  y_{e_i + e_j},

and the SDP minimises u. Weak duality: optimal u is a LOWER bound on
the true min over the box's mu. If u >= t, box CERTIFIES.

Order 1 gives a trivial LB = 1 in practice. Order 2 is the smallest
useful order. At d=16 the M_2 cone is 153 x 153. We use Clarabel for
a vendor-free stack.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _build_monomials(d: int, deg: int) -> List[Tuple[int, ...]]:
    """All monomials in d variables of total degree <= deg, as sorted tuples
    of variable indices (length = actual degree, no zeros). Index 0 = 1.
    """
    out = [()]
    for k in range(1, deg + 1):
        for comb in combinations_with_replacement(range(d), k):
            out.append(tuple(comb))
    return out


def _alpha_of(mon: Tuple[int, ...], d: int) -> Tuple[int, ...]:
    """Convert a monomial (sorted-var-tuple) to multi-index (alpha_0..alpha_{d-1})."""
    alpha = [0] * d
    for v in mon:
        alpha[v] += 1
    return tuple(alpha)


def _index_of_alpha(idx: Dict[Tuple[int, ...], int], alpha: Tuple[int, ...]) -> int:
    return idx[alpha]


def lasserre_box_lb_float(
    lo: np.ndarray, hi: np.ndarray,
    windows, d: int,
    order: int = 2,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> float:
    """Solve the Lasserre L_order SDP for the box. Returns LB on
    min_mu max_W mu^T M_W mu, or -inf on solver failure.
    """
    import cvxpy as cp

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    # --- Build monomial index up to degree 2*order ---
    max_deg = 2 * order
    monos = _build_monomials(d, max_deg)
    # Unique alpha -> index in y. Multiple monomials map to same alpha.
    alpha_to_idx: Dict[Tuple[int, ...], int] = {}
    for mon in monos:
        a = _alpha_of(mon, d)
        if a not in alpha_to_idx:
            alpha_to_idx[a] = len(alpha_to_idx)
    n_y = len(alpha_to_idx)

    # y[alpha_to_idx[alpha]] represents E[mu^alpha].
    y = cp.Variable(n_y)

    # Basis for M_order: monomials of degree <= order.
    basis_order = [mon for mon in monos if len(mon) <= order]
    alphas_basis = [_alpha_of(m, d) for m in basis_order]
    B = len(basis_order)  # basis size

    def add_alpha(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x + z for x, z in zip(a, b))

    # M_order(y): B x B symmetric matrix with [i,j] = y_{basis_i + basis_j}.
    M_mat = []
    for i in range(B):
        row = []
        for j in range(B):
            a = add_alpha(alphas_basis[i], alphas_basis[j])
            row.append(y[alpha_to_idx[a]])
        M_mat.append(row)
    M = cp.bmat(M_mat)

    # Localizing for (mu_k - lo_k): order (order - 1) basis, size B1.
    basis_loc = [mon for mon in monos if len(mon) <= order - 1]
    alphas_loc = [_alpha_of(m, d) for m in basis_loc]
    B1 = len(basis_loc)

    def build_loc_matrix(var_k: int, coef_lo: float) -> cp.Expression:
        """M_{order-1}((mu_k - coef_lo) * y)."""
        rows = []
        e_k = [0] * d
        e_k[var_k] = 1
        e_k = tuple(e_k)
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_alpha(alphas_loc[i], alphas_loc[j])
                # Entry: y_{base + e_k} - coef_lo * y_{base}
                base_idx = alpha_to_idx[base]
                # base+e_k may exceed max_deg; skip constraint in that case
                a_plus = add_alpha(base, e_k)
                if sum(a_plus) > max_deg:
                    # Cannot enforce; drop localizing at this entry.
                    # Use coef_lo * y[base_idx] and treat entry as no-op (0).
                    # Minimal fallback: set entry to 0 (SDP ignores this row).
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(y[alpha_to_idx[a_plus]] - coef_lo * y[base_idx])
            rows.append(rr)
        return cp.bmat(rows)

    def build_upper_loc_matrix(var_k: int, coef_hi: float) -> cp.Expression:
        """M_{order-1}((coef_hi - mu_k) * y)."""
        rows = []
        e_k = [0] * d
        e_k[var_k] = 1
        e_k = tuple(e_k)
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_alpha(alphas_loc[i], alphas_loc[j])
                base_idx = alpha_to_idx[base]
                a_plus = add_alpha(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(coef_hi * y[base_idx] - y[alpha_to_idx[a_plus]])
            rows.append(rr)
        return cp.bmat(rows)

    cons = []
    # y_0 = 1
    cons += [y[alpha_to_idx[(0,) * d]] == 1]
    # Non-negativity: all y_alpha >= 0 (since mu >= 0).
    cons += [y >= 0]
    # Moment matrix PSD.
    cons += [M >> 0]
    # Simplex: sum_i y_{e_i} = 1.
    def e_i(i):
        v = [0] * d
        v[i] = 1
        return tuple(v)
    cons += [cp.sum([y[alpha_to_idx[e_i(i)]] for i in range(d)]) == 1]
    # Consistency from sum mu = 1 (gives y_alpha = sum_i y_{alpha + e_i}) for |alpha| < max_deg.
    for alpha, ai in alpha_to_idx.items():
        s = sum(alpha)
        if s >= max_deg:
            continue
        # sum_i y_{alpha + e_i} == y_alpha
        children = []
        for i in range(d):
            a2 = list(alpha)
            a2[i] += 1
            a2 = tuple(a2)
            if a2 in alpha_to_idx:
                children.append(y[alpha_to_idx[a2]])
        if children:
            cons += [cp.sum(children) == y[ai]]
    # Box localizing: for each k, M_{order-1}((mu_k - lo_k)*y) >= 0 AND
    # M_{order-1}((hi_k - mu_k)*y) >= 0.
    for k in range(d):
        lo_k = float(lo[k])
        hi_k = float(hi[k])
        cons += [build_loc_matrix(k, lo_k) >> 0]
        cons += [build_upper_loc_matrix(k, hi_k) >> 0]

    # Window: u >= sum scale_W * y_{e_i + e_j} for each W.
    u = cp.Variable()
    for w in windows:
        if len(w.pairs_all) == 0:
            continue
        terms = []
        for (i, j) in w.pairs_all:
            a = [0] * d
            a[i] += 1
            a[j] += 1
            terms.append(y[alpha_to_idx[tuple(a)]])
        cons += [u >= float(w.scale) * cp.sum(terms)]

    prob = cp.Problem(cp.Minimize(u), cons)
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"[sdp] exception: {e}")
        return float("-inf")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        if verbose:
            print(f"[sdp] non-optimal status: {prob.status}")
        return float("-inf")
    return float(u.value)


def lasserre_box_certifies(
    lo: np.ndarray, hi: np.ndarray,
    windows, d: int,
    target_q: Fraction,
    *,
    order: int = 2,
    rel_tol: float = 1e-6,
    solver: str = "CLARABEL",
) -> bool:
    """True iff the L_order SDP LB on the box is >= target_q * (1 + rel_tol)."""
    lb = lasserre_box_lb_float(lo, hi, windows, d, order=order, solver=solver)
    if lb == float("-inf"):
        return False
    target_f = float(target_q)
    return lb >= target_f * (1.0 + rel_tol)


# ---------------------------------------------------------------------
# Residual-aware variant for publication-rigor escalation.
#
# Bypasses CVXPY's high-level solve() to call Clarabel's DefaultSolver
# directly on the canonicalised conic data so that we can read the
# native KKT residuals (r_prim, r_dual) and the dual-objective value
# (obj_val_dual) needed for cushion-based rigorous certificates.
#
# See bound_sdp_escalation.py for the consuming routine and the full
# soundness derivation.
# ---------------------------------------------------------------------


def lasserre_box_lb_float_with_residuals(
    lo: np.ndarray, hi: np.ndarray,
    windows, d: int,
    *,
    order: int = 2,
    tol_gap_abs: float = 1e-9,
    tol_gap_rel: float = 1e-9,
    tol_feas: float = 1e-9,
    time_limit_s: float = 5.0,
    max_iter: int = 200,
    verbose: bool = False,
) -> dict:
    """Solve the Lasserre order-`order` box SDP and return primal/dual
    objective values plus Clarabel's native KKT residuals.

    Returns dict with keys:
      'status'        : str — Clarabel termination status
                        ('Solved' | 'AlmostSolved' | 'PrimalInfeasible'
                         | 'DualInfeasible' | 'AlmostPrimalInfeasible'
                         | 'AlmostDualInfeasible' | 'MaxIterations'
                         | 'MaxTime' | 'NumericalError'
                         | 'InsufficientProgress' | 'CallbackTerminated'
                         | 'PRESOLVE_INFEASIBLE' | 'BUILD_FAILED'
                         | 'EMPTY' on early-exit before solve attempt)
      'obj_val'       : float — primal objective at termination (NaN if no soln)
      'obj_val_dual'  : float — dual objective at termination (NaN if no soln)
      'r_prim'        : float — Clarabel native primal residual
                        (relative, KKT-style)
      'r_dual'        : float — Clarabel native dual residual
                        (relative, KKT-style)
      'duality_gap'   : float — abs(obj_val - obj_val_dual)
      'iterations'    : int
      'solve_time'    : float
      'is_feasible_status' : bool — True iff status indicates a usable
                                    solution (Solved / AlmostSolved)
      'n_y'           : int — number of pseudo-moment vars in the SDP
      'n_psd_blocks'  : int — number of PSD cones (moment + 2d localizing)

    For the consuming "is u_dual_LB >= target?" cert: we use obj_val_dual
    (lower bound on the SDP optimum by weak duality, when the dual is
    feasible) minus a cushion of max(100 * max(r_prim, r_dual,
    duality_gap_abs), 1e-6).

    Soundness: for the `min u` SDP, obj_val_dual is the dual function
    value at termination. By weak duality, if the DUAL is feasible then
    obj_val_dual <= primal_opt. Clarabel's r_dual measures stationarity
    residual (dual feasibility violation in the relative-KKT sense). A
    cushion of 100x the largest residual conservatively absorbs both
    the dual-feasibility violation and float64 finite-precision noise.

    Empty box (sum(lo) > 1 or sum(hi) < 1) returns status 'EMPTY' and
    obj_val_dual = +inf (vacuous; caller may treat as certifying).
    """
    import cvxpy as cp
    import scipy.sparse as sp
    import clarabel

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    # Empty-box short-circuit (vacuous cert convention used by callers).
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-12 or hi_sum < 1.0 - 1e-12:
        return {
            'status': 'EMPTY',
            'obj_val': float('inf'),
            'obj_val_dual': float('inf'),
            'r_prim': 0.0,
            'r_dual': 0.0,
            'duality_gap': 0.0,
            'iterations': 0,
            'solve_time': 0.0,
            'is_feasible_status': True,
            'n_y': 0,
            'n_psd_blocks': 0,
        }

    # ---- Build the same CVXPY problem as lasserre_box_lb_float ----
    max_deg = 2 * order
    monos = _build_monomials(d, max_deg)
    alpha_to_idx: Dict[Tuple[int, ...], int] = {}
    for mon in monos:
        a = _alpha_of(mon, d)
        if a not in alpha_to_idx:
            alpha_to_idx[a] = len(alpha_to_idx)
    n_y = len(alpha_to_idx)

    y = cp.Variable(n_y)
    basis_order = [mon for mon in monos if len(mon) <= order]
    alphas_basis = [_alpha_of(m, d) for m in basis_order]
    B = len(basis_order)

    def add_alpha(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x + z for x, z in zip(a, b))

    M_mat = []
    for i in range(B):
        row = []
        for j in range(B):
            a = add_alpha(alphas_basis[i], alphas_basis[j])
            row.append(y[alpha_to_idx[a]])
        M_mat.append(row)
    M = cp.bmat(M_mat)

    basis_loc = [mon for mon in monos if len(mon) <= order - 1]
    alphas_loc = [_alpha_of(m, d) for m in basis_loc]
    B1 = len(basis_loc)

    def build_loc_matrix(var_k: int, coef_lo: float) -> 'cp.Expression':
        rows = []
        e_k = [0] * d
        e_k[var_k] = 1
        e_k = tuple(e_k)
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_alpha(alphas_loc[i], alphas_loc[j])
                base_idx = alpha_to_idx[base]
                a_plus = add_alpha(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(y[alpha_to_idx[a_plus]] - coef_lo * y[base_idx])
            rows.append(rr)
        return cp.bmat(rows)

    def build_upper_loc_matrix(var_k: int, coef_hi: float) -> 'cp.Expression':
        rows = []
        e_k = [0] * d
        e_k[var_k] = 1
        e_k = tuple(e_k)
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_alpha(alphas_loc[i], alphas_loc[j])
                base_idx = alpha_to_idx[base]
                a_plus = add_alpha(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(coef_hi * y[base_idx] - y[alpha_to_idx[a_plus]])
            rows.append(rr)
        return cp.bmat(rows)

    cons = []
    cons += [y[alpha_to_idx[(0,) * d]] == 1]
    cons += [y >= 0]
    cons += [M >> 0]

    def e_i(i: int) -> Tuple[int, ...]:
        v = [0] * d
        v[i] = 1
        return tuple(v)

    cons += [cp.sum([y[alpha_to_idx[e_i(i)]] for i in range(d)]) == 1]
    for alpha, ai in alpha_to_idx.items():
        s = sum(alpha)
        if s >= max_deg:
            continue
        children = []
        for i in range(d):
            a2 = list(alpha)
            a2[i] += 1
            a2 = tuple(a2)
            if a2 in alpha_to_idx:
                children.append(y[alpha_to_idx[a2]])
        if children:
            cons += [cp.sum(children) == y[ai]]
    n_psd_blocks = 1  # M
    for k in range(d):
        lo_k = float(lo[k])
        hi_k = float(hi[k])
        cons += [build_loc_matrix(k, lo_k) >> 0]
        cons += [build_upper_loc_matrix(k, hi_k) >> 0]
        n_psd_blocks += 2

    u = cp.Variable()
    for w in windows:
        if len(w.pairs_all) == 0:
            continue
        terms = []
        for (i, j) in w.pairs_all:
            a = [0] * d
            a[i] += 1
            a[j] += 1
            terms.append(y[alpha_to_idx[tuple(a)]])
        cons += [u >= float(w.scale) * cp.sum(terms)]

    prob = cp.Problem(cp.Minimize(u), cons)

    # Canonicalize to conic form, then call Clarabel directly so we can
    # read native KKT residuals.
    try:
        data, _chain, _inverse_data = prob.get_problem_data(cp.CLARABEL)
    except Exception as e:
        if verbose:
            print(f"[sdp-residual] canon exception: {e}")
        return {
            'status': 'BUILD_FAILED',
            'obj_val': float('nan'),
            'obj_val_dual': float('nan'),
            'r_prim': float('inf'),
            'r_dual': float('inf'),
            'duality_gap': float('inf'),
            'iterations': 0,
            'solve_time': 0.0,
            'is_feasible_status': False,
            'n_y': n_y,
            'n_psd_blocks': n_psd_blocks,
        }

    A = data['A']
    b = data['b']
    q = data['c']
    P = data.get('P')
    if P is None:
        P = sp.csc_matrix((q.size, q.size))
    P = sp.triu(P).tocsc()
    dims = data['dims']

    # Build clarabel cones from cvxpy dims (order matters: zero,
    # nonneg, soc, psd, exp, p3d).
    cones = []
    if dims.zero > 0:
        cones.append(clarabel.ZeroConeT(int(dims.zero)))
    if dims.nonneg > 0:
        cones.append(clarabel.NonnegativeConeT(int(dims.nonneg)))
    for sd in dims.soc:
        cones.append(clarabel.SecondOrderConeT(int(sd)))
    for sd in dims.psd:
        cones.append(clarabel.PSDTriangleConeT(int(sd)))
    for _ in range(int(dims.exp)):
        cones.append(clarabel.ExponentialConeT())
    for sd in dims.p3d:
        cones.append(clarabel.PowerConeT(float(sd)))

    sett = clarabel.DefaultSettings()
    sett.verbose = bool(verbose)
    sett.tol_gap_abs = float(tol_gap_abs)
    sett.tol_gap_rel = float(tol_gap_rel)
    sett.tol_feas = float(tol_feas)
    sett.tol_infeas_abs = float(tol_feas)
    sett.tol_infeas_rel = float(tol_feas)
    sett.time_limit = float(time_limit_s)
    sett.max_iter = int(max_iter)

    try:
        solver = clarabel.DefaultSolver(P, q, A, b, cones, sett)
        sol = solver.solve()
    except Exception as e:
        if verbose:
            print(f"[sdp-residual] solve exception: {e}")
        return {
            'status': f'EXCEPTION:{type(e).__name__}',
            'obj_val': float('nan'),
            'obj_val_dual': float('nan'),
            'r_prim': float('inf'),
            'r_dual': float('inf'),
            'duality_gap': float('inf'),
            'iterations': 0,
            'solve_time': 0.0,
            'is_feasible_status': False,
            'n_y': n_y,
            'n_psd_blocks': n_psd_blocks,
        }

    status_str = str(sol.status)
    is_feas = status_str in ('Solved', 'AlmostSolved')
    primal_obj = float(sol.obj_val)
    # Clarabel exposes obj_val and obj_val_dual; cvxpy sense already
    # encodes minimisation, so the offset is zero.
    dual_obj = float(sol.obj_val_dual)
    r_prim = float(sol.r_prim)
    r_dual = float(sol.r_dual)
    gap = abs(primal_obj - dual_obj) if (
        np.isfinite(primal_obj) and np.isfinite(dual_obj)) else float('inf')

    return {
        'status': status_str,
        'obj_val': primal_obj,
        'obj_val_dual': dual_obj,
        'r_prim': r_prim,
        'r_dual': r_dual,
        'duality_gap': gap,
        'iterations': int(sol.iterations),
        'solve_time': float(sol.solve_time),
        'is_feasible_status': is_feas,
        'n_y': n_y,
        'n_psd_blocks': n_psd_blocks,
    }

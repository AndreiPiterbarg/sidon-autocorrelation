"""L benchmark: per-composition Lasserre/Shor SDP for cell pruning.

VERDICT (this file's premise, formally derived):
=================================================
Q is an LP relaxation of a quadratic min-max. Variant L tightens this with
SDP at the per-composition cell level — feasibility test of a Shor (= order-1
Lasserre + RLT) or true order-2 Lasserre relaxation.

DERIVATION (verified):
======================
Composition c (integer, length d, sum 4n*m). Let x_i := m * a_i so that the
cell becomes (using b = c/m, |a-b|_oo <= 1/m, a >= 0, sum a = 4n):

    x in R^d, max(0, c_i - 1) <= x_i <= c_i + 1, sum x = 4n*m.

Per window W = (ell, s_lo) define the symmetric indicator A_W with
A_W[i,j] = 1 iff (i+j) in [s_lo, s_lo + ell - 2]. Then:

    m^2 * TV_W(a) = (1/(4n*ell)) * x^T A_W x.

Composition c PRUNES when, for every x in the cell, there exists W with
m^2 * TV_W(a) > c_target * m^2. Equivalently, the system

    (S):  x in cell
          x^T A_W x <= 4n*ell*c_target*m^2     for ALL W

is INFEASIBLE.  L uses the SDP relaxation of (S) via moment matrices.

Lift via X = x x^T. Then x^T A_W x = Tr(A_W X). PSD relaxation:

    M_1(y) := [[1, x^T], [x, X]] >= 0   (Shor / order-1 Lasserre)

with RLT cuts on (x_i - lo_i)(x_j - lo_j) >= 0 etc., box bounds lo <= x <= hi,
sum x = 4n*m, X[i,i] in [lo_i^2, hi_i^2], and (the key) for ALL W:

    Tr(A_W X) <= 4n*ell*c_target*m^2.

If the SDP returns INFEASIBLE (Farkas certificate), composition c is L-pruned.

SOUNDNESS:
==========
PSD relaxation is a SUPERSET of the true integer/continuous feasible set
(every feasible (x, X = x x^T) is PSD-feasible). So SDP-INFEASIBLE
=> original-INFEASIBLE => composition pruned. One-sided test, never wrong.

We require Farkas/dual-infeasibility certificates from the solver to claim
infeasibility (avoids numerical false-positives at near-boundary cases).
For MOSEK: status PRIMAL_INFEASIBLE_CER or DUAL_INFEASIBLE_CER required.
For Clarabel: status PrimalInfeasible required.

ORDER-2 LASSERRE (option):
==========================
Same as `interval_bnb/lasserre_cert.py:lasserre_box_lb_float` but at the
PER-COMPOSITION level — basis monos up to degree 2, M_2 PSD block of size
binom(d+2, 2). Localizing for box constraints. Window constraints become

    sum_{(i,j) pair in W} y_{e_i + e_j}  <=  4n*ell*c_target*m^2 / (mass_const)

We test feasibility (no objective) — INFEASIBLE iff cell is L2-pruned.

Order-2 dim grows: d=10 -> M_2 is 66x66; d=14 -> 120x120. Heavy but works.
"""
from __future__ import annotations
import os, sys, time, json, argparse
from itertools import combinations
from typing import Optional
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs


# ----------------------------------------------------------------------
# Solver detection
# ----------------------------------------------------------------------
def _detect_solver(prefer: str = 'auto') -> str:
    """Return solver name: 'MOSEK', 'CLARABEL', or 'SCS' (last resort)."""
    try:
        import cvxpy as cp
        avail = set(cp.installed_solvers())
    except ImportError:
        return 'NONE'
    if prefer.upper() in avail:
        return prefer.upper()
    if 'MOSEK' in avail:
        try:
            import mosek
            env = mosek.Env()
            env.checkoutlicense(mosek.feature.pton)
            return 'MOSEK'
        except Exception:
            pass
    if 'CLARABEL' in avail:
        return 'CLARABEL'
    if 'SCS' in avail:
        return 'SCS'
    return 'NONE'


# ----------------------------------------------------------------------
# Window matrices
# ----------------------------------------------------------------------
def _build_A_matrices(d, windows):
    """Build symmetric indicator A_W (d x d) for each window W = (ell, s_lo).

    A_W[i,j] = 1 iff s_lo <= i + j <= s_lo + ell - 2.
    """
    A_list = []
    for (ell, s_lo) in windows:
        s_hi = s_lo + ell - 2
        A = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    A[i, j] = 1.0
        A_list.append(A)
    return A_list


# ----------------------------------------------------------------------
# Order-1 Shor SDP feasibility (with RLT)
# ----------------------------------------------------------------------
def _shor_feasibility(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                       solver='MOSEK', tol=1e-9, eps_margin=1e-9, verbose=False):
    """Test SDP feasibility of: x in cell, Tr(A_W X) <= thr_W for all W.

    Returns True if the SDP proves INFEASIBLE (composition L-pruned).
    Sound: SDP is a relaxation, so SDP-infeasible => original-infeasible.

    cell:
      lo <= x <= hi  (lo = max(0, c-1), hi = c+1)
      sum x = 4n*m
      X[i,i] in [lo_i^2, hi_i^2]
      RLT cuts on X[i,j]
      [[1, x^T],[x, X]] >> 0

    threshold (m^2 units, integer-conv units): each window
      Tr(A_W X) <= 4n*ell*c_target*m^2
    """
    import cvxpy as cp

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    # Moment matrix Y = [[1, x^T], [x, X]] >= 0
    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    # Diagonal: x_i^2 ranges
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        # Also tighter scalar McCormick (linear): X[i,i] >= 2*lo_i*x_i - lo_i^2
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    # RLT off-diagonal
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # Window constraints
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    # Feasibility: minimize 0
    prob = cp.Problem(cp.Minimize(0), cons)

    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=verbose,
                        tol_feas=tol, tol_gap_abs=tol, tol_gap_rel=tol,
                        max_iter=300)
        elif solver == 'SCS':
            prob.solve(solver='SCS', verbose=verbose,
                        eps=tol, max_iters=20000)
        else:
            return False, prob.status if hasattr(prob, 'status') else 'NO_SOLVER'
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    status = prob.status
    # Soundness: claim "pruned" only on certified infeasibility.
    # cvxpy maps:
    #   MOSEK: PRIMAL_INFEASIBLE_CER -> "infeasible"
    #          DUAL_INFEASIBLE_CER   -> "unbounded"
    #   Clarabel: PrimalInfeasible -> "infeasible"
    # We accept "infeasible" (and "infeasible_inaccurate" only if user opts in).
    # Critical: do NOT accept "unbounded" — that would indicate the dual is
    # infeasible, which for our minimize-0 problem means the primal has no
    # finite minimum, which only happens with a numerical problem here.
    if status == 'infeasible':
        return True, status
    return False, status


# ----------------------------------------------------------------------
# Order-2 Lasserre SDP feasibility (true Lasserre, dim = O(d^2/2))
# ----------------------------------------------------------------------
from itertools import combinations_with_replacement

def _build_monomials(d, deg):
    """All sorted-tuple monomials in d vars of total degree <= deg (deg counts
    duplicates). Index 0 = empty tuple = constant 1."""
    out = [()]
    for k in range(1, deg + 1):
        for comb in combinations_with_replacement(range(d), k):
            out.append(tuple(comb))
    return out


def _alpha_of(mon, d):
    a = [0] * d
    for v in mon:
        a[v] += 1
    return tuple(a)


def _lasserre2_feasibility(c_int, lo, hi, windows, n_half, m, c_target,
                            solver='MOSEK', tol=1e-9, eps_margin=1e-9,
                            verbose=False):
    """Order-2 Lasserre SDP feasibility test on the cell.

    Variables: pseudo-moments y_alpha for |alpha| <= 4 over x_0..x_{d-1}.
    PSD: M_2(y) (size B=binom(d+2,2)) and 2d localizing M_1's.
    Equality: y_0 = 1 (we lift x to projective form by normalizing scale),
              sum_i y_{e_i} = 4n*m   (== sum constraint).
    Box: lo_i <= x_i <= hi_i (encoded via 2d localizing)
    Window: sum_{(i,j) in W} y_{e_i + e_j} <= 4n*ell*c_target*m^2

    INFEASIBLE => composition L2-pruned.
    """
    import cvxpy as cp

    d = len(c_int)
    max_deg = 4
    monos = _build_monomials(d, max_deg)
    alpha_to_idx = {}
    for mn in monos:
        a = _alpha_of(mn, d)
        if a not in alpha_to_idx:
            alpha_to_idx[a] = len(alpha_to_idx)
    n_y = len(alpha_to_idx)
    y = cp.Variable(n_y)

    basis2 = [mn for mn in monos if len(mn) <= 2]
    alphas2 = [_alpha_of(mn, d) for mn in basis2]
    B2 = len(basis2)

    def add_a(a, b):
        return tuple(x + z for x, z in zip(a, b))

    M_rows = []
    for i in range(B2):
        row = []
        for j in range(B2):
            a = add_a(alphas2[i], alphas2[j])
            row.append(y[alpha_to_idx[a]])
        M_rows.append(row)
    M = cp.bmat(M_rows)

    basis1 = [mn for mn in monos if len(mn) <= 1]
    alphas1 = [_alpha_of(mn, d) for mn in basis1]
    B1 = len(basis1)

    def loc_low(k, lo_k):
        e_k = [0] * d
        e_k[k] = 1
        e_k = tuple(e_k)
        rows = []
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_a(alphas1[i], alphas1[j])
                a_plus = add_a(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(y[alpha_to_idx[a_plus]] - lo_k * y[alpha_to_idx[base]])
            rows.append(rr)
        return cp.bmat(rows)

    def loc_high(k, hi_k):
        e_k = [0] * d
        e_k[k] = 1
        e_k = tuple(e_k)
        rows = []
        for i in range(B1):
            rr = []
            for j in range(B1):
                base = add_a(alphas1[i], alphas1[j])
                a_plus = add_a(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(hi_k * y[alpha_to_idx[base]] - y[alpha_to_idx[a_plus]])
            rows.append(rr)
        return cp.bmat(rows)

    cons = []
    cons += [y[alpha_to_idx[(0,) * d]] == 1.0]
    cons += [y >= 0]   # all monomial integrals are nonneg since x >= 0
    cons += [M >> 0]

    # sum_i y_{e_i} = 4n*m
    nm = float(4 * n_half * m)
    e_idx = []
    for i in range(d):
        e = [0] * d
        e[i] = 1
        e_idx.append(alpha_to_idx[tuple(e)])
    cons += [cp.sum([y[k] for k in e_idx]) == nm]

    # Box localizing
    for k in range(d):
        cons += [loc_low(k, float(lo[k])) >> 0]
        cons += [loc_high(k, float(hi[k])) >> 0]

    # Window constraints: sum_{(i,j) in W ordered pair} y_{e_i + e_j} <= thr
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m
    for (ell, s_lo) in windows:
        s_hi = s_lo + ell - 2
        terms = []
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    a = [0] * d
                    a[i] += 1
                    a[j] += 1
                    terms.append(y[alpha_to_idx[tuple(a)]])
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.sum(terms) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=verbose,
                        tol_feas=tol, tol_gap_abs=tol, tol_gap_rel=tol,
                        max_iter=400)
        elif solver == 'SCS':
            prob.solve(solver='SCS', verbose=verbose, eps=tol, max_iters=30000)
        else:
            return False, 'NO_SOLVER'
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'
    return prob.status == 'infeasible', prob.status


# ----------------------------------------------------------------------
# Per-composition L-prune (with optional order)
# ----------------------------------------------------------------------
def _make_cell(c_int, m):
    """Cell box: lo = max(0, c-1), hi = c+1 (in x = m*a units)."""
    c = np.asarray(c_int, dtype=np.float64)
    lo = np.maximum(0.0, c - 1.0)
    hi = c + 1.0
    return lo, hi


def prune_L_one(c_int, A_mats, windows, n_half, m, c_target,
                 solver='MOSEK', order=1, tol=1e-9, eps_margin=1e-9,
                 verbose=False):
    """Run SDP for one composition. Returns (pruned: bool, status_str)."""
    lo, hi = _make_cell(c_int, m)
    if order == 1:
        return _shor_feasibility(c_int, lo, hi, A_mats, windows, n_half, m,
                                   c_target, solver=solver, tol=tol,
                                   eps_margin=eps_margin, verbose=verbose)
    elif order == 2:
        return _lasserre2_feasibility(c_int, lo, hi, windows, n_half, m,
                                        c_target, solver=solver, tol=tol,
                                        eps_margin=eps_margin, verbose=verbose)
    else:
        raise ValueError(f"Unknown order: {order}")


# ----------------------------------------------------------------------
# Soundness audit: enumerate δ-cell at low resolution, check infeasibility.
# ----------------------------------------------------------------------
def _audit_one(c_int, A_mats, windows, n_half, m, c_target, n_audit_grid=3):
    """For a small grid of δ in cell (|δ|_∞ <= 1, sum δ = 0, δ <= c), check
    whether ANY δ has all m^2*TV_W(b-δ/m) <= c_target*m^2.

    If found: composition is FEASIBLE (was incorrectly L-pruned -> SOUNDNESS BUG).
    Returns (False, None) if no counter-example; else (True, witness_x).
    """
    d = len(c_int)
    # Use δ ∈ {-1, 0, +1}^d ∩ {sum=0, δ_i <= c_i}; same x = c - δ in cell.
    if n_audit_grid == 3:
        levels = (-1, 0, 1)
    elif n_audit_grid == 5:
        levels = (-1, -0.5, 0, 0.5, 1)
    else:
        levels = (-1, 0, 1)

    cs_m2 = c_target * m * m
    n_d = float(n_half)

    if d > 8:
        # Too many to enumerate; sample randomly.
        rng = np.random.default_rng(0)
        for _ in range(2000):
            delta = rng.choice(levels, size=d)
            if abs(np.sum(delta)) > 1e-9:
                continue
            x = np.asarray(c_int, dtype=np.float64) - delta
            if np.any(x < 0):
                continue
            ok = True
            for A_mat, (ell, _) in zip(A_mats, windows):
                v = float(x @ A_mat @ x)
                thr = 4.0 * n_d * ell * cs_m2
                if v > thr + 1e-9:
                    ok = False
                    break
            if ok:
                return True, x
        return False, None
    else:
        from itertools import product
        for delta in product(levels, repeat=d):
            if abs(sum(delta)) > 1e-9:
                continue
            x = np.asarray(c_int, dtype=np.float64) - np.asarray(delta)
            if np.any(x < 0):
                continue
            ok = True
            for A_mat, (ell, _) in zip(A_mats, windows):
                v = float(x @ A_mat @ x)
                thr = 4.0 * n_d * ell * cs_m2
                if v > thr + 1e-9:
                    ok = False
                    break
            if ok:
                return True, x
        return False, None


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run(n_half, m, c_target, batch_size=200_000, solver='auto', order=1,
         max_l_per_run=None, audit_size=20, verbose=True, eps_margin=1e-9,
         tol=1e-9):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)
    actual_solver = _detect_solver(solver if solver != 'auto' else 'MOSEK')

    if verbose:
        print(f"\n=== L-bench: n_half={n_half}, m={m}, c_target={c_target}, "
              f"order={order}, solver={actual_solver} ===")
        print(f"d={d}, S_full=4nm={S_full}, palindromic half_sum=2nm={S_half}, "
              f"total palindromic comps={n_total_half:,}")

    # Build LP/SDP scaffolding
    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"  n_win = {n_win}, n_sigma = {len(sigmas)}")

    # JIT warmup
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    n_processed = 0
    pruned_F = surv_F_n = 0
    pruned_Q = surv_Q_n = 0
    pruned_L = surv_L_n = 0
    extra_Q_over_F = 0
    extra_L_over_Q = 0
    bug_L_minus_F = 0   # L-survivor that is NOT F-survivor (looser; means L bug)
    bug_L_unsound = 0   # L-pruned but enumeration finds witness (UNSOUND!)
    t_F = t_Q = t_L = 0.0
    l_solve_times = []
    l_status_counter = {}

    f_surv_total_idx = []  # list of (batch_idx, c_int) for soundness audit

    t0 = time.time()
    n_l_done = 0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)

        tf = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - tf
        pruned_F += int(np.sum(~sF))
        surv_F_n += int(np.sum(sF))

        f_surv_idx = np.where(sF)[0]
        sQ_on_F = np.ones(len(f_surv_idx), dtype=bool)

        tq = time.time()
        for k, idx in enumerate(f_surv_idx):
            c_int = batch[idx]
            if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target, margin=eps_margin):
                sQ_on_F[k] = False
        t_Q += time.time() - tq
        n_extra_Q = int(np.sum(~sQ_on_F))
        extra_Q_over_F += n_extra_Q
        surv_Q_n += int(np.sum(sQ_on_F))
        pruned_Q += int(np.sum(~sF)) + n_extra_Q

        # L: run on Q-survivors only (L can only be tighter than Q in theory,
        # but we verify the theory by soundness audit later).
        q_surv_idx = f_surv_idx[sQ_on_F]
        sL_on_Q = np.ones(len(q_surv_idx), dtype=bool)

        if max_l_per_run is not None and n_l_done + len(q_surv_idx) > max_l_per_run:
            n_run_l = max(0, max_l_per_run - n_l_done)
            q_surv_idx = q_surv_idx[:n_run_l]
            sL_on_Q = sL_on_Q[:n_run_l]

        tl = time.time()
        for k, idx in enumerate(q_surv_idx):
            c_int = batch[idx]
            t_one = time.time()
            pruned, status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                            c_target, solver=actual_solver,
                                            order=order, tol=tol,
                                            eps_margin=eps_margin)
            t_one = time.time() - t_one
            l_solve_times.append(t_one)
            l_status_counter[status] = l_status_counter.get(status, 0) + 1
            if pruned:
                sL_on_Q[k] = False
            n_l_done += 1
        t_L += time.time() - tl

        n_extra_L = int(np.sum(~sL_on_Q))
        extra_L_over_Q += n_extra_L
        surv_L_n += int(np.sum(sL_on_Q))
        pruned_L += int(np.sum(~sF)) + n_extra_Q + n_extra_L

        # Save F-survivors c-vectors for audit
        for idx in f_surv_idx[:5]:
            f_surv_total_idx.append(batch[idx].copy())

    elapsed = time.time() - t0

    # Stats
    if l_solve_times:
        l_med = float(np.median(l_solve_times))
        l_max = float(np.max(l_solve_times))
        l_p95 = float(np.percentile(l_solve_times, 95))
    else:
        l_med = l_max = l_p95 = 0.0

    print(f"\n--- L vs Q vs F ---")
    print(f"  total processed:      {n_processed:,}")
    print(f"  F survivors:          {surv_F_n:,}  ({100*surv_F_n/max(1,n_processed):.2f}%)")
    print(f"  Q survivors:          {surv_Q_n:,}  ({100*surv_Q_n/max(1,n_processed):.2f}%)")
    print(f"  L survivors:          {surv_L_n:,}  ({100*surv_L_n/max(1,n_processed):.2f}%)")
    if surv_F_n > 0:
        print(f"  Q extra over F:       {extra_Q_over_F:,} ({100*extra_Q_over_F/max(1,surv_F_n):.2f}% of F)")
    if surv_Q_n > 0 or extra_L_over_Q > 0:
        print(f"  L extra over Q:       {extra_L_over_Q:,} ({100*extra_L_over_Q/max(1,surv_Q_n+extra_L_over_Q):.2f}% of Q-surv)")
    print(f"  L solve t  med/p95/max: {l_med*1000:.1f}/{l_p95*1000:.1f}/{l_max*1000:.1f} ms"
          f"  on {len(l_solve_times)} SDPs")
    print(f"  L statuses: {l_status_counter}")
    print(f"  Wall: F {t_F:.2f}s + Q {t_Q:.2f}s + L {t_L:.2f}s = {elapsed:.2f}s")

    # Soundness audit on a sample of L-pruned compositions: try enumeration
    # at coarse grid; if a witness exists, L is unsound.
    if extra_L_over_Q > 0:
        print(f"\n--- L-soundness audit (enumerate cell at coarse grid) ---")
        # Just re-process a small batch and audit L-pruned ones
        n_aud = 0
        n_aud_target = audit_size
        for half_batch in generate_compositions_batched(n_half, S_half,
                                                          batch_size=batch_size):
            batch = np.empty((len(half_batch), d), dtype=np.int32)
            batch[:, :n_half] = half_batch
            batch[:, n_half:] = half_batch[:, ::-1]
            sF = prune_F(batch, n_half, m, c_target)
            f_surv_idx = np.where(sF)[0]
            for idx in f_surv_idx:
                if n_aud >= n_aud_target:
                    break
                c_int = batch[idx]
                # Q-pruned?
                if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, margin=eps_margin):
                    continue
                # L-pruned?
                pruned_l, _ = prune_L_one(c_int, A_mats, windows, n_half, m,
                                             c_target, solver=actual_solver,
                                             order=order, tol=tol,
                                             eps_margin=eps_margin)
                if not pruned_l:
                    continue
                # Now audit: any δ in coarse {-1,0,1}^d with sum=0 and x=c-δ in box,
                # all windows below thr?
                feas, witness = _audit_one(c_int, A_mats, windows, n_half, m,
                                              c_target, n_audit_grid=3)
                n_aud += 1
                if feas:
                    bug_L_unsound += 1
                    print(f"   *** UNSOUND: c={c_int.tolist()}, witness x={witness}")
            if n_aud >= n_aud_target:
                break
        print(f"  Audited {n_aud} L-pruned comps; {bug_L_unsound} unsound -> "
              f"{'PASS' if bug_L_unsound == 0 else 'FAIL'}")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'order': order, 'solver': actual_solver, 'n_win': n_win,
        'n_processed': n_processed,
        'surv_F': surv_F_n, 'surv_Q': surv_Q_n, 'surv_L': surv_L_n,
        'pruned_F': pruned_F, 'pruned_Q': pruned_Q, 'pruned_L': pruned_L,
        'extra_Q_over_F': extra_Q_over_F,
        'extra_L_over_Q': extra_L_over_Q,
        'bug_L_unsound': bug_L_unsound,
        't_F': t_F, 't_Q': t_Q, 't_L': t_L,
        'l_solve_t_med_ms': l_med * 1000,
        'l_solve_t_p95_ms': l_p95 * 1000,
        'l_solve_t_max_ms': l_max * 1000,
        'l_n_solves': len(l_solve_times),
        'l_status_counter': l_status_counter,
        'elapsed': elapsed,
    }


# ----------------------------------------------------------------------
# Sanity: tiny verification
# ----------------------------------------------------------------------
def _sanity_tests(solver='MOSEK', order=1):
    """Verify L on hand-built cases: (a) clearly pruned, (b) clearly not."""
    print(f"\n=== L sanity (solver={solver}, order={order}) ===")

    # Case 1: trivial — composition c with very large concentration on bin 0.
    # For (n=2, m=10, c_target=1.20), c=(40,0,0,0) sum=40=4nm. cell allows
    # δ in {|δ|≤1, sum=0, δ_0 in [-1, 1] ... but x_0 = c_0 - δ_0 in [39,41]}
    # only constraint is sum = 0; x_0 large => x_0^2 ≈ 1600, m^2*TV(W={s=0}, ell=2)
    # = (1/(4*2*2)) * x_0^2 = 100 = m^2 = 100. So always = 100 = m^2 * 1.0.
    # This is BELOW c_target * m^2 = 120, so cell does NOT certify > 1.20 — composition
    # is NOT L-pruned.
    n_half, m, c_target = 2, 10, 1.20
    d = 2 * n_half
    windows, _ = _build_windows(d)
    A_mats = _build_A_matrices(d, windows)

    c_test = np.array([40, 0, 0, 0], dtype=np.int32)
    # but sum = 40, n_half=2, m=10, sum should be 4*2*10=80. So adjust:
    c_test = np.array([20, 20, 20, 20], dtype=np.int32)  # sum 80
    pruned, status = prune_L_one(c_test, A_mats, windows, n_half, m, c_target,
                                    solver=solver, order=order)
    print(f"  Uniform c=(20,20,20,20) (n=2,m=10,c=1.20): pruned={pruned} status={status}")

    # Case 2: very imbalanced -> x has a big spike, TV at center = high
    c_test2 = np.array([40, 0, 0, 40], dtype=np.int32)
    pruned, status = prune_L_one(c_test2, A_mats, windows, n_half, m, c_target,
                                    solver=solver, order=order)
    print(f"  c=(40,0,0,40) (extreme corners): pruned={pruned} status={status}")

    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--solver', type=str, default='auto')
    ap.add_argument('--order', type=int, default=1, choices=[1, 2])
    ap.add_argument('--max_l', type=int, default=None,
                     help='Cap L SDPs total (smoke test)')
    ap.add_argument('--audit', type=int, default=10)
    ap.add_argument('--out', type=str, default='_L_results.json')
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()

    if args.sanity:
        actual = _detect_solver(args.solver)
        _sanity_tests(solver=actual, order=args.order)
        sys.exit(0)

    results = []
    if args.sweep:
        configs = [
            (3, 10, 1.28),  # F=172
            (4, 10, 1.28),  # F=1014, Q=964
            (5, 5, 1.28),   # F=558, Q=240
            (6, 5, 1.28),   # F=7 — L should kill all? (cascade-killer)
        ]
        for nh, m, c in configs:
            try:
                r = run(nh, m, c, batch_size=args.batch, solver=args.solver,
                         order=args.order, max_l_per_run=args.max_l,
                         audit_size=args.audit, tol=args.tol)
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  *** ERROR in ({nh},{m},{c}): {e}")
                results.append({'n_half': nh, 'm': m, 'c_target': c,
                                'error': str(e)})
    else:
        r = run(args.n_half, args.m, args.c_target, batch_size=args.batch,
                 solver=args.solver, order=args.order,
                 max_l_per_run=args.max_l, audit_size=args.audit, tol=args.tol)
        results.append(r)

    with open(args.out, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nWrote {args.out}")

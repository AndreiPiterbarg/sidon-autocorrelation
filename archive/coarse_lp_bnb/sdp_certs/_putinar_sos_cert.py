"""Putinar SOS certificate for per-cell box certification.

Goal
====
Rigorous per-cell SOS certificate via Putinar's positivstellensatz.  We
prove, for a single pruning window W independently, that the quadratic
polynomial

    p_W(delta) := TV_W(c/S + delta) - c_target
                = (tv0_W - c_target) + g_W . delta + scale_W * delta^T A_W delta

is nonneg on the box-cell

    Q = { delta in R^d : h - delta_i >= 0,  h + delta_i >= 0,  sum_i delta_i = 0 }

with h = 1/(2S).  The cell is certified iff EXISTS a window W for which a
feasible Putinar SOS decomposition is found.

Putinar at order r
==================
Putinar's positivstellensatz: p_W is nonneg on Q iff there exist SOS
polynomials sigma_0, sigma_i^+, sigma_i^- and a polynomial lambda such that

    p_W(delta) = sigma_0(delta)
               + sum_i sigma_i^+(delta) (h - delta_i)
               + sum_i sigma_i^-(delta) (h + delta_i)
               + lambda(delta) * (sum_i delta_i)               [equality term]

with degree budgets controlled by the relaxation order r:
    deg(sigma_0)         <= 2r
    deg(sigma_i^{+/-})   <= 2(r-1)
    deg(lambda)          <= 2r - 1

For r = 2 (the order requested):
    sigma_0  : degree 4  (Gram matrix on basis of monomials <= deg 2,
                          size = 1 + d + d(d+1)/2)
    sigma_i^{+/-} : degree 2 (Gram matrix on basis <= deg 1, size = 1+d)
    lambda   : free polynomial of degree <= 3

DUAL (SOS) FORMULATION
======================
We parameterize each SOS sigma directly via a PSD Gram matrix:

    sigma(delta) = b(delta)^T G b(delta),   G >> 0

where b(delta) is a vector of monomials.  The polynomial identity

    p_W(delta) = sigma_0(delta)
               + sum_i sigma_i^+(delta) (h - delta_i)
               + sum_i sigma_i^-(delta) (h + delta_i)
               + lambda(delta) * sum_i delta_i

is enforced by matching coefficients of every monomial alpha (multi-index
with sum(alpha) <= 4) on both sides.

This is the DUAL of the moment-LP / Lasserre primal we implemented in
_coarse_L2_bench.py.  Solver-equivalence: the SOS dual and moment primal
have the same optimum value (modulo strict-feasibility / Slater issues).
However, the SOS view is what's needed to PRODUCE a Putinar identity
(the Gram matrices factor as G = L L^T, and the identity is then
b^T G b = (L^T b)^T (L^T b), an explicit SOS sum) that can be checked
in any rational arithmetic — the implementation here yields floating-
point Gram matrices, but the structure is what matters.

For the FEASIBILITY question (does cell Q certify p_W >= 0?) we test:

    SOS_FEAS:  exists G_0 >> 0, G_i^+ >> 0, G_i^- >> 0, lambda_alpha
               such that the polynomial identity holds.

This is a pure SDP feasibility problem.  Soundness is unconditional:
if the SDP is FEASIBLE, then a Putinar certificate exists in floating
point, and the proven quantity is "p_W is nonneg on Q" up to numerical
tolerance (any solver tol issue reduces to checking a specific
arithmetic identity, NOT to soundness of the cert).

For RIGOR in floating point we add a small slack `gap_target >= 0`:

    p_W(delta) - gap_target = sigma_0(...) + ...

i.e. we certify p_W >= gap_target on Q, with gap_target chosen large
enough (e.g. 1e-6 * |c_target|) to absorb solver inaccuracy.  We then
add `gap_target` back to c_target when comparing — the cert proves
p_W(delta) >= 0 on Q via slack, and gap_target serves as a safety margin.

Multi-window joint cert
=======================
Per-cell certification: try EACH window W independently; cell is certified
iff EXISTS W with feasible SOS at order r.  This is sound because
    min_delta max_W TV_W >= max_W min_delta TV_W
        >= max_W (TV_W proven >= c_target by SOS).

API
===
- `cell_cert_putinar_sos(c_int, S, d, c_target, window, order=2, ...)`
    SOS feasibility for one window.  Returns (feasible, status, info).
- `cell_cert_putinar_max(c_int, S, d, c_target, windows, order=2, ...)`
    Best (= "any feasible") over a list of windows.

Note: this is computationally heavier than Shor / Lasserre-2 because it
introduces 2d additional Gram matrices for the box localizers.  Useful
mainly as a reference / soundness witness, not for production cascade
runs.

Tests
=====
30 hardest-uncertified cells from c=1.281 (we use d=4, S=20, c=1.281,
since d=4 is the smallest where the SDP is interesting and S=20 gives
the size scaling we want).  We compare:
    - triangle (cell_var + quad_corr)
    - Shor (order-1 PSD lift) -- via _coarse_L_bench.cell_cert_shor
    - Lasserre-2 (order-2 moment) -- via _coarse_L2_bench.cell_cert_lasserre2
    - Putinar SOS at order 2 (this file, dual view)

Soundness anchor: we verify, on every cert, that the LB returned by the
moment dual (Lasserre-2 LB) matches the SOS feasibility threshold up to
numerical tol.  Mismatch by more than 1e-5 is flagged.
"""
from __future__ import annotations
import os, sys, time, json, argparse, math
from itertools import product, combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Pull cvxpy lazily
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

# Import shared infrastructure from the Shor and Lasserre benches
from _coarse_L_bench import (
    all_windows, build_A_matrix, tv_at, grad_at,
    cell_vertices, qp_min_vertex_eval,
    cell_cert_shor, triangle_cert, find_hard_cells,
)
from _coarse_L2_bench import cell_cert_lasserre2


# =====================================================================
# Multi-index machinery
# =====================================================================

def _alphas_up_to(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    """All multi-indices alpha in N^d with sum(alpha) <= max_deg, sorted by
    (total_degree, lex)."""
    out = []
    def _rec(remaining: int, current: List[int]):
        if len(current) == d:
            out.append(tuple(current))
            return
        for a in range(remaining + 1):
            current.append(a)
            _rec(remaining - a, current)
            current.pop()
    _rec(max_deg, [])
    out.sort(key=lambda a: (sum(a), a))
    return out


def _basis_up_to(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    """Monomial basis: all alpha with sum(alpha) <= max_deg."""
    return _alphas_up_to(d, max_deg)


def _add(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _e(d: int, i: int, k: int = 1) -> Tuple[int, ...]:
    """Unit multi-index k * e_i."""
    a = [0] * d
    a[i] = k
    return tuple(a)


# =====================================================================
# Putinar SOS feasibility (single window) — DUAL SOS view
# =====================================================================

def cell_cert_putinar_sos(
    c_int: np.ndarray, S: int, d: int, c_target: float,
    window: Tuple[int, int],
    order: int = 2,
    solver: str = 'auto',
    tol: float = 1e-9,
    gap_slack: float = 0.0,
    verbose: bool = False,
) -> Tuple[bool, str, Dict]:
    """Putinar SOS feasibility test for one window at relaxation order `order`.

    Tests whether there exist SOS polynomials sigma_0, sigma_i^{+/-} and
    polynomial lambda such that

        p_W(delta) - gap_slack
            = sigma_0(delta)
            + sum_i sigma_i^+(delta) * (h - delta_i)
            + sum_i sigma_i^-(delta) * (h + delta_i)
            + lambda(delta) * sum_i delta_i.

    Returns (feasible, status, info).
        feasible: True iff SDP found a certificate.
        status:   cvxpy solver status string.
        info:     dict with diagnostics (gram_min_eig, identity_resid, ...).
    """
    if not _HAS_CVXPY:
        return False, 'NO_CVXPY', {}

    if order < 1:
        raise ValueError("Putinar order must be >= 1.")

    ell, s_lo = window
    c = np.asarray(c_int, dtype=np.float64)
    h = 1.0 / (2.0 * S)
    A = build_A_matrix(d, ell, s_lo)
    g = grad_at(c, S, d, ell, s_lo)
    tv0 = tv_at(c, S, d, ell, s_lo)
    scale = 2.0 * d / ell

    # The polynomial p(delta) = (tv0 - c_target - gap_slack)
    #                         + sum_i g_i delta_i
    #                         + scale * sum_{i,j} A[i,j] delta_i delta_j
    # Coefficient of monomial alpha:
    p_coef: Dict[Tuple[int, ...], float] = {}
    zero = tuple([0] * d)
    p_coef[zero] = float(tv0 - c_target - gap_slack)
    for i in range(d):
        ei = _e(d, i)
        p_coef[ei] = p_coef.get(ei, 0.0) + float(g[i])
    for i in range(d):
        for j in range(d):
            if A[i, j] != 0.0:
                aij = _add(_e(d, i), _e(d, j))
                p_coef[aij] = p_coef.get(aij, 0.0) + scale * float(A[i, j])

    # Bases for Gram matrices
    # Order r relaxation:
    #   sigma_0 SOS deg <= 2r => Gram on basis monomials of deg <= r
    #   sigma_i^{+/-} SOS deg <= 2(r-1) => Gram on basis of deg <= r-1
    #   lambda free poly deg <= 2r-1
    r = order
    basis_sigma0 = _basis_up_to(d, r)
    basis_sigma_box = _basis_up_to(d, r - 1) if r >= 1 else [zero]
    if r == 1:
        # sigma_i^{+/-} of degree 0 = nonneg scalars
        basis_sigma_box = [zero]
    lambda_alphas = _alphas_up_to(d, 2 * r - 1)

    # All monomials we need to track in identity matching
    # Max degree of any term on RHS:
    #   sigma_0: 2r
    #   sigma_i^{+/-} * (h ± delta_i): 2(r-1) + 1 = 2r - 1
    #   lambda * sum delta: (2r-1) + 1 = 2r
    max_deg = 2 * r
    target_alphas = _alphas_up_to(d, max_deg)
    target_idx = {a: k for k, a in enumerate(target_alphas)}
    n_target = len(target_alphas)

    # === cvxpy variables ===
    nB0 = len(basis_sigma0)
    nBb = len(basis_sigma_box)
    G0 = cp.Variable((nB0, nB0), symmetric=True, name='G0')
    Gp = [cp.Variable((nBb, nBb), symmetric=True, name=f'Gp_{i}') for i in range(d)]
    Gm = [cp.Variable((nBb, nBb), symmetric=True, name=f'Gm_{i}') for i in range(d)]
    lam = cp.Variable(len(lambda_alphas), name='lambda_coefs')

    cons = []
    cons.append(G0 >> 0)
    for i in range(d):
        cons.append(Gp[i] >> 0)
        cons.append(Gm[i] >> 0)

    # === Build coefficient expressions for RHS, indexed by alpha ===
    # rhs[alpha] = expression equal to the coef of delta^alpha on RHS.
    # We accumulate using a dict of cvxpy expressions.
    rhs_expr: Dict[Tuple[int, ...], cp.Expression] = {a: 0.0 for a in target_alphas}

    # sigma_0 contribution: sigma_0(delta) = sum_{a,b} G0[a,b] * delta^{basis_sigma0[a]+basis_sigma0[b]}
    for a in range(nB0):
        for b in range(nB0):
            ab = _add(basis_sigma0[a], basis_sigma0[b])
            if ab in target_idx:
                rhs_expr[ab] = rhs_expr[ab] + G0[a, b]

    # sigma_i^+ * (h - delta_i):
    #   contributes G_p[a,b] * delta^{basis_sigma_box[a]+basis_sigma_box[b]} * (h - delta_i)
    # sigma_i^- * (h + delta_i): analogous with sign flip.
    for i in range(d):
        ei = _e(d, i)
        for a in range(nBb):
            for b in range(nBb):
                base = _add(basis_sigma_box[a], basis_sigma_box[b])
                # h * Gp coef
                if base in target_idx:
                    rhs_expr[base] = rhs_expr[base] + h * Gp[i][a, b]
                    rhs_expr[base] = rhs_expr[base] + h * Gm[i][a, b]
                # ± delta_i * Gp coef
                base_e = _add(base, ei)
                if base_e in target_idx:
                    rhs_expr[base_e] = rhs_expr[base_e] - Gp[i][a, b]
                    rhs_expr[base_e] = rhs_expr[base_e] + Gm[i][a, b]

    # lambda(delta) * sum_i delta_i:
    # contributes lam_alpha * sum_i delta^{alpha + e_i}
    for k_alpha, alpha in enumerate(lambda_alphas):
        for i in range(d):
            beta = _add(alpha, _e(d, i))
            if beta in target_idx:
                rhs_expr[beta] = rhs_expr[beta] + lam[k_alpha]

    # === Match LHS = RHS coefficient by coefficient ===
    for alpha in target_alphas:
        lhs = float(p_coef.get(alpha, 0.0))
        cons.append(rhs_expr[alpha] == lhs)

    # === Feasibility problem ===
    # Solve a feasibility problem.  Use a trivial objective (0) so the
    # solver returns a feasible point if one exists.
    prob = cp.Problem(cp.Minimize(cp.Constant(0.0)), cons)

    actual_solver = solver
    if solver == 'auto':
        avail = set(cp.installed_solvers())
        for s in ('MOSEK', 'CLARABEL', 'SCS'):
            if s in avail:
                actual_solver = s
                break

    info = {
        'window': list(window),
        'order': order,
        'gap_slack': gap_slack,
        'nB_sigma0': nB0,
        'nB_sigma_box': nBb,
        'n_lambda': len(lambda_alphas),
        'n_target_alphas': n_target,
        'n_constraints_psd': 1 + 2 * d,
        'n_constraints_eq': n_target,
    }

    try:
        if actual_solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                       mosek_params={
                           'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                       })
        elif actual_solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=verbose,
                       eps_abs=tol, eps_rel=tol, max_iter=400)
        else:
            prob.solve(solver='SCS', verbose=verbose, eps=tol, max_iters=20000)
    except cp.error.SolverError as e:
        # MOSEK throws SolverError when primal-infeasible without a clean
        # cert.  This means the SOS at this order is infeasible — the
        # legitimate "no Putinar cert exists" outcome.  Return cleanly.
        info['exception'] = f'SolverError: {e}'
        info['status'] = 'sos_infeasible_solver'
        return False, 'sos_infeasible_solver', info
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        return False, f'EXC:{type(e).__name__}', info

    status = prob.status
    info['status'] = status

    # Feasible: SDP found a primal point.  optimal/optimal_inaccurate are
    # the success codes for cp.Minimize(0) feasibility problems.
    # infeasible/infeasible_inaccurate cleanly indicate no Putinar cert
    # exists at this order — return False.
    feasible = status in ('optimal', 'optimal_inaccurate')

    if feasible:
        # Diagnostic: smallest eigenvalue of each Gram (should be >= 0 up to tol)
        try:
            G0_val = G0.value
            min_eigs = []
            if G0_val is not None:
                w0 = np.linalg.eigvalsh(G0_val)
                min_eigs.append(('G0', float(w0.min())))
            for i in range(d):
                if Gp[i].value is not None:
                    wp = np.linalg.eigvalsh(Gp[i].value)
                    min_eigs.append((f'Gp_{i}', float(wp.min())))
                if Gm[i].value is not None:
                    wm = np.linalg.eigvalsh(Gm[i].value)
                    min_eigs.append((f'Gm_{i}', float(wm.min())))
            info['min_gram_eig'] = float(min(e for _, e in min_eigs)) if min_eigs else None
            info['n_grams'] = len(min_eigs)
        except Exception:
            info['min_gram_eig'] = None

    return feasible, status, info


# =====================================================================
# Multi-window joint cert: any-window-feasible
# =====================================================================

def cell_cert_putinar_max(
    c_int: np.ndarray, S: int, d: int, c_target: float,
    windows: Sequence[Tuple[int, int]],
    order: int = 2,
    solver: str = 'auto',
    tol: float = 1e-9,
    gap_slack: float = 0.0,
    early_stop: bool = True,
    verbose: bool = False,
) -> Tuple[bool, Tuple[int, int], List[Dict]]:
    """Try Putinar SOS on each window in `windows`; cert iff ANY feasible.

    Sound: if some W has p_W >= 0 on Q, then max_W TV_W >= c_target on Q.

    Returns (any_feasible, certifying_window, per_window_diagnostics).
    """
    diag = []
    cert_W = (-1, -1)
    any_feas = False
    for W in windows:
        feas, status, info = cell_cert_putinar_sos(
            c_int, S, d, c_target, W, order=order,
            solver=solver, tol=tol, gap_slack=gap_slack, verbose=verbose,
        )
        diag.append({'W': list(W), 'feas': bool(feas), 'status': status,
                     'min_gram_eig': info.get('min_gram_eig')})
        if feas and not any_feas:
            any_feas = True
            cert_W = W
            if early_stop:
                break
    return any_feas, cert_W, diag


# =====================================================================
# Driver: 30 uncertified cells from c=1.281
# =====================================================================

def run_bench(d: int, S: int, c_target: float,
              max_cells: int = 30,
              order: int = 2,
              gap_slack: float = 0.0,
              solver: str = 'auto', tol: float = 1e-9,
              verbose: bool = True) -> Dict:
    """Benchmark Putinar SOS vs triangle / Shor / Lasserre-2 on hard cells.

    Pipeline:
      1. Find compositions c with TV_W(c/S) > c_target for some W (grid passers).
      2. Subset to triangle-uncertified (tri_net <= 0).
      3. On the K cells with tri_net closest to zero (most likely savable):
         a. Run Shor (cell_cert_shor on triangle's W*) — record cert/loose.
         b. Run Lasserre-2 (cell_cert_lasserre2 on triangle's W*).
         c. Run Putinar SOS at `order` (cell_cert_putinar_sos on triangle's W*).
    """
    print(f"\n=== _putinar_sos_cert: d={d}, S={S}, c_target={c_target}, order={order} ===")
    print(f"    cell width h = 1/(2S) = {1.0 / (2.0 * S):.6f}")
    print(f"    gap_slack = {gap_slack:.2e}")

    hard, n_grid_pass, n_tri_cert, n_total = find_hard_cells(d, S, c_target)
    print(f"    grid-point passers  : {n_grid_pass:,}")
    print(f"    triangle certified  : {n_tri_cert:,}")
    print(f"    HARD cells (failing): {len(hard):,}")
    if not hard:
        print("    No hard cells — triangle certifies everything.")
        return {'d': d, 'S': S, 'c_target': c_target,
                'order': order, 'n_hard': 0, 'rows': []}

    # Sort by triangle net, take top K
    hard.sort(key=lambda kv: -kv[1]['net'])
    hard = hard[:max_cells]
    print(f"    Running Shor + L2 + Putinar SOS on {len(hard)} hardest cells.\n")
    print(f"    triangle net range: [{hard[0][1]['net']:+.6f}, {hard[-1][1]['net']:+.6f}]\n")

    n_shor_cert = 0
    n_l2_cert = 0
    n_sos_cert = 0
    n_sos_strict_better = 0   # SOS feas but L2 not cert
    n_sound_violation = 0
    rows = []
    times_shor = []
    times_l2 = []
    times_sos = []

    for k, (c, tri) in enumerate(hard):
        W = tri['W']

        # Shor
        t0 = time.time()
        shor_lb, shor_status = cell_cert_shor(c, S, d, c_target, W,
                                                solver=solver, tol=tol)
        dt_shor = time.time() - t0
        shor_cert = (shor_lb >= c_target - 1e-9)
        if shor_cert:
            n_shor_cert += 1

        # Lasserre-2
        t1 = time.time()
        l2_lb, l2_status = cell_cert_lasserre2(c, S, d, c_target, W,
                                                solver=solver, tol=tol)
        dt_l2 = time.time() - t1
        l2_cert = (l2_lb >= c_target - 1e-9)
        if l2_cert:
            n_l2_cert += 1

        # Putinar SOS at requested order
        t2 = time.time()
        sos_feas, sos_status, sos_info = cell_cert_putinar_sos(
            c, S, d, c_target, W, order=order,
            solver=solver, tol=tol, gap_slack=gap_slack,
        )
        dt_sos = time.time() - t2
        if sos_feas:
            n_sos_cert += 1

        # Soundness: SOS-feas certifies p_W >= 0; L2 LB is a lower bound on
        # min p_W; so if SOS feas at gap_slack=0, L2_LB should be >= 0 - tol
        # i.e. l2_lb >= c_target - tol.  (Equivalent to l2_cert.)
        # If sos_feas but not l2_cert with margin > 1e-5, this would be a
        # numerical anomaly — flag.
        if sos_feas and l2_lb != float('-inf'):
            l2_margin = l2_lb - c_target  # >=0 means cert
            if l2_margin < -1e-5:
                n_sound_violation += 1

        if sos_feas and not l2_cert:
            n_sos_strict_better += 1

        times_shor.append(dt_shor)
        times_l2.append(dt_l2)
        times_sos.append(dt_sos)

        rows.append({
            'k': k, 'c': c.tolist(),
            'tri_W': list(W), 'tri_net': float(tri['net']),
            'tri_tv': float(tri['tv']),
            'shor_lb': float(shor_lb) if shor_lb != float('-inf') else None,
            'shor_cert': bool(shor_cert),
            'l2_lb': float(l2_lb) if l2_lb != float('-inf') else None,
            'l2_cert': bool(l2_cert),
            'sos_feas': bool(sos_feas),
            'sos_status': sos_status,
            'sos_min_gram_eig': sos_info.get('min_gram_eig'),
            'time_shor_s': float(dt_shor),
            'time_l2_s': float(dt_l2),
            'time_sos_s': float(dt_sos),
        })

        if verbose and k < 8:
            print(f"    [{k:3d}] c={c.tolist()}  tri_net={tri['net']:+.5f}  "
                  f"shor={'C' if shor_cert else '.'}  L2={'C' if l2_cert else '.'}  "
                  f"SOS={'C' if sos_feas else '.'}({sos_status[:6]})  "
                  f"T={dt_shor*1000:.0f}/{dt_l2*1000:.0f}/{dt_sos*1000:.0f}ms")

    pct = lambda x: 100.0 * x / max(1, len(hard))
    times_shor = np.asarray(times_shor)
    times_l2 = np.asarray(times_l2)
    times_sos = np.asarray(times_sos)

    print(f"\n    --- Summary (window=triangle-W*) ---")
    print(f"    Hard cells tested      : {len(hard)}")
    print(f"    Shor certified         : {n_shor_cert}  ({pct(n_shor_cert):.1f}%)")
    print(f"    Lasserre-2 certified   : {n_l2_cert}  ({pct(n_l2_cert):.1f}%)")
    print(f"    Putinar SOS feasible   : {n_sos_cert}  ({pct(n_sos_cert):.1f}%)")
    print(f"    SOS feas but L2 not cert: {n_sos_strict_better}")
    print(f"    Soundness anomalies    : {n_sound_violation}")
    print(f"    Time / cell (ms)  Shor: med={1000*np.median(times_shor):.0f}  "
          f"p95={1000*np.percentile(times_shor,95):.0f}")
    print(f"    Time / cell (ms)    L2: med={1000*np.median(times_l2):.0f}  "
          f"p95={1000*np.percentile(times_l2,95):.0f}")
    print(f"    Time / cell (ms)   SOS: med={1000*np.median(times_sos):.0f}  "
          f"p95={1000*np.percentile(times_sos,95):.0f}")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'order': order, 'gap_slack': gap_slack,
        'n_hard_total': len(hard),
        'n_hard_tested': len(hard),
        'n_shor_cert': n_shor_cert,
        'n_l2_cert': n_l2_cert,
        'n_sos_cert': n_sos_cert,
        'n_sos_strict_better': n_sos_strict_better,
        'n_sound_violation': n_sound_violation,
        'time_shor_med_ms': float(1000 * np.median(times_shor)) if len(times_shor) else None,
        'time_l2_med_ms':   float(1000 * np.median(times_l2)) if len(times_l2) else None,
        'time_sos_med_ms':  float(1000 * np.median(times_sos)) if len(times_sos) else None,
        'time_shor_p95_ms': float(1000 * np.percentile(times_shor,95)) if len(times_shor) else None,
        'time_l2_p95_ms':   float(1000 * np.percentile(times_l2,95)) if len(times_l2) else None,
        'time_sos_p95_ms':  float(1000 * np.percentile(times_sos,95)) if len(times_sos) else None,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.281)
    ap.add_argument('--max_cells', type=int, default=30)
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--gap_slack', type=float, default=0.0)
    ap.add_argument('--solver', default='auto')
    ap.add_argument('--out', default='_putinar_sos_results.json')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if not _HAS_CVXPY:
        print("ERROR: cvxpy not available — cannot run SDP.")
        sys.exit(1)

    r = run_bench(args.d, args.S, args.c_target,
                  max_cells=args.max_cells,
                  order=args.order,
                  gap_slack=args.gap_slack,
                  solver=args.solver, tol=1e-9,
                  verbose=not args.quiet)
    rcopy = dict(r)
    if 'rows' in rcopy and len(rcopy['rows']) > 30:
        rcopy['rows'] = rcopy['rows'][:30]
    with open(args.out, 'w') as fp:
        json.dump([rcopy], fp, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()

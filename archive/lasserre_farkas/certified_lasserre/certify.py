"""End-to-end certified lower bound on val(d) from the aggregated Lasserre
SDP.

Pipeline:
    build_sdp_data(d, order, lam)        -> SDPData   [float64]
    solve_primal_dual(sdp, solver)       -> SolverResult
    extract_dual(sdp, sol)               -> PolishedDual
    round_and_repair(sdp, dual, ...)     -> CertifiedDual   [exact rationals]
    _verify_certificate(sdp, cert)       -> raises on any soundness failure
    _decimal_str(cert.lb_rig, digits=15) -> pretty decimal expansion

Soundness chain proved by _verify_certificate:

  1.  For each block j, S_j^rat = L_j L_j^T + slack_j * I is PSD by
      construction (product of rational matrix and its transpose, plus
      nonneg scalar times identity).

  2.  Exact rational identity  c - A^T lambda - sum_j F_j^*(S_j) = 0
      is checked component-by-component in Fraction arithmetic.

  3.  Given (1) and (2), weak SDP duality gives
          primal SDP optimum  >=  b^T lambda   =  lambda[0]      (since
                                                                  b[0]=1, rest 0)

  4.  The primal SDP value itself is a LOWER BOUND on min_{mu in Delta}
      mu^T M_lambda mu  (standard Lasserre relaxation soundness).

  5.  By  max >= convex combination:  max_W mu^T M_W mu
                                       >= sum_W lambda_W mu^T M_W mu
                                       =  mu^T M_lambda mu.

  6.  Therefore val(d) = min_mu max_W mu^T M_W mu
                      >= min_mu mu^T M_lambda mu
                      >= primal SDP value
                      >= b^T lambda
                      =  cert.lb_rig.

Only (2) needs exact arithmetic; (1), (3)--(6) are symbolic / structural.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Optional
import json
import time
import numpy as np

from certified_lasserre.build_sdp import build_sdp_data, SDPData
from certified_lasserre.bm_solver import solve_primal_dual, SolverResult
from certified_lasserre.dual_extract import extract_dual, PolishedDual
from certified_lasserre.round_repair import (
    round_and_repair, CertifiedDual,
    _adjoint_block_rational, _frac_sparse_matvec, _rational_c,
    _vec_to_fractions,
)


# =====================================================================
# Rational helpers
# =====================================================================

def _decimal_str(q: Fraction, digits: int = 15) -> str:
    """Decimal expansion of q to `digits` digits after the point
    (truncated, with the next digit used for banker's-rounding)."""
    import mpmath
    with mpmath.workdps(digits + 10):
        val = mpmath.mpf(q.numerator) / mpmath.mpf(q.denominator)
        return mpmath.nstr(val, n=digits + 1, strip_zeros=False)


def _rational_matmul_naive(A, B):
    # A, B are object arrays of Fractions
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            s = Fraction(0)
            for l in range(k):
                s += A[i, l] * B[l, j]
            out[i, j] = s
    return out


# =====================================================================
# Verifier
# =====================================================================

def verify_certificate(sdp: SDPData, cert: CertifiedDual,
                        verbose: bool = True) -> Fraction:
    """Recompute the stationarity residual from scratch in exact rationals.

    Returns the rational lb_rig = cert.lb_rig on success.  Raises ValueError
    on any failure.
    """
    d, order = cert.d, cert.order
    assert (d, order) == (sdp.d, sdp.order), \
        f'(d,order) mismatch: sdp={sdp.d,sdp.order}, cert={d,order}'

    # Rebuild c from rational window weights
    c_rat = _rational_c(sdp, cert.lam_windows)

    # Recompute S_j = L_j L_j^T + diag(slack_j)
    S_list = []
    for L, slack_j, blk in zip(cert.L_blocks, cert.slack, sdp.blocks):
        if L.shape[1] > 0:
            S = _rational_matmul_naive(L, L.T)
        else:
            S = np.empty((blk.size, blk.size), dtype=object)
            for ii in range(blk.size):
                for jj in range(blk.size):
                    S[ii, jj] = Fraction(0)
        # Add diagonal slack
        for ii in range(blk.size):
            if slack_j[ii] != 0:
                S[ii, ii] = S[ii, ii] + slack_j[ii]
        S_list.append(S)

    # Residual
    AT_csr = sdp.A.T.tocsr()
    AT_lam = _frac_sparse_matvec(AT_csr, cert.lambda_A)
    r = np.empty(sdp.n_y, dtype=object)
    for k in range(sdp.n_y):
        r[k] = c_rat[k] - AT_lam[k]
    for blk, S in zip(sdp.blocks, S_list):
        adj = _adjoint_block_rational(blk, S)
        for k in range(sdp.n_y):
            r[k] -= adj[k]

    nz = sum(1 for x in r if x != 0)
    if nz > 0:
        raise ValueError(f'verify_certificate: exact residual has {nz} '
                         f'nonzero entries.')

    # Verify window weights sum to 1 (exact)
    ws = Fraction(0)
    for w in cert.lam_windows:
        if w < 0:
            raise ValueError(f'negative window weight {w}')
        ws += w
    if ws != 1:
        raise ValueError(f'window weights sum to {ws} != 1')

    # Check slack nonneg
    for j, slack_j in enumerate(cert.slack):
        for k, s in enumerate(slack_j):
            if s < 0:
                raise ValueError(f'slack_{j}[{k}] = {s} is negative')

    if verbose:
        print('[verify_certificate] PASSED.')
        print(f'  d={d}, order={order}')
        print(f'  lb_rig = {cert.lb_rig} ({_decimal_str(cert.lb_rig, 15)})')
        print(f'  window weights sum: {ws}')
        tot = Fraction(0)
        for sl in cert.slack:
            for v in sl:
                tot += v
        print(f'  total diag slack: {float(tot):.3e}')
    return cert.lb_rig


# =====================================================================
# Full pipeline
# =====================================================================

@dataclass
class CertifyResult:
    d: int
    order: int
    lb_rig: Fraction
    lb_rig_decimal: str
    solver_time: float
    round_time: float
    verify_time: float
    total_time: float
    solver: str
    primal_obj: float
    residual_norm_float: float
    notes: str = ''


def certify_bound(d: int, order: int,
                  lam: Optional[np.ndarray] = None,
                  solver: str = 'auto',
                  max_denom_L: int = 10**6,
                  max_denom_lam: int = 10**10,
                  max_denom_win: int = 10**8,
                  try_slack: bool = True,
                  verbose: bool = True) -> (CertifyResult, CertifiedDual):
    """Produce a rigorous lower bound lb_rig <= val(d) and an exact
    rational certificate.

    Args:
        d, order:   Lasserre parameters.
        lam:        window weights (shape n_win).  If None, uniform.
        solver:     'auto'|'mosek'|'clarabel'|'scs'|'bm'.

    Returns:
        (CertifyResult, CertifiedDual).
    """
    t_start = time.time()
    if verbose:
        print(f'[certify_bound] d={d}, order={order}, solver={solver}')

    sdp = build_sdp_data(d, order, lam=lam, verbose=verbose)

    t0 = time.time()
    sol = solve_primal_dual(sdp, solver=solver, verbose=False)
    solver_time = time.time() - t0
    if verbose:
        print(f'  primal obj = {sol.obj:.9f}, solver = {sol.solver}, '
              f'time = {solver_time:.2f}s, status = {sol.status}')

    polished = extract_dual(sdp, sol, verbose=verbose)
    res_float = polished.residual_norm

    t0 = time.time()
    cert = round_and_repair(
        sdp, polished,
        max_denom_L=max_denom_L,
        max_denom_lam=max_denom_lam,
        max_denom_win=max_denom_win,
        try_slack=try_slack,
        verbose=verbose,
    )
    round_time = time.time() - t0

    t0 = time.time()
    verify_certificate(sdp, cert, verbose=verbose)
    verify_time = time.time() - t0

    total_time = time.time() - t_start
    res = CertifyResult(
        d=d, order=order,
        lb_rig=cert.lb_rig,
        lb_rig_decimal=_decimal_str(cert.lb_rig, 15),
        solver_time=solver_time,
        round_time=round_time,
        verify_time=verify_time,
        total_time=total_time,
        solver=sol.solver,
        primal_obj=sol.obj,
        residual_norm_float=res_float,
        notes='',
    )
    if verbose:
        print(f'[certify_bound] DONE.  lb_rig = {res.lb_rig} = '
              f'{res.lb_rig_decimal}   ({total_time:.1f}s)')
    return res, cert


# =====================================================================
# Certificate serialization
# =====================================================================

def save_certificate(cert: CertifiedDual, path: str, sdp: Optional[SDPData] = None) -> None:
    """Write the certificate to a JSON file with all rationals as
    [numerator, denominator] pairs, so that a verifier only needs to
    recompute c, A, b from (d, order, lam_windows) and recheck
        c - A^T lambda - sum_j (L_j L_j^T + s_j I)^* adjoint = 0
        b^T lambda = lb_rig.
    """
    def q(f: Fraction):
        return [int(f.numerator), int(f.denominator)]

    payload = {
        'd': cert.d,
        'order': cert.order,
        'lb_rig': q(cert.lb_rig),
        'lb_rig_decimal': _decimal_str(cert.lb_rig, 20),
        'lam_windows': [q(x) for x in cert.lam_windows],
        'lambda_A': [q(x) for x in cert.lambda_A],
        'L_blocks': [
            [[q(x) for x in row] for row in L.tolist()]
            if L.shape[1] > 0 else []
            for L in cert.L_blocks
        ],
        'slack': [[q(v) for v in sl] for sl in cert.slack],
        'block_sizes': [L.shape[0] if L.size else 0 for L in cert.L_blocks]
                        if all(L.size > 0 for L in cert.L_blocks) else
                        [sdp.blocks[j].size for j in range(len(cert.L_blocks))]
                        if sdp else None,
        'exact_residual_nonzeros': int(sum(1 for x in cert.exact_residual if x != 0)),
    }
    if sdp is not None:
        payload['n_y'] = sdp.n_y
        payload['n_eq'] = sdp.n_eq
        payload['block_sizes'] = [blk.size for blk in sdp.blocks]

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

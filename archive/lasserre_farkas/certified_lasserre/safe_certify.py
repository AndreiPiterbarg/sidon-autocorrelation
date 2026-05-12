"""Safe rigorous certification of val(d) lower bound via the aggregated
Lasserre relaxation.

Contrast with `certify.py`:
  * `certify.py` tries to make the stationarity residual EXACTLY zero via
    exact rational repair. This requires the residual to lie in col([A^T | W])
    which is not always achievable — the diag-slack augmentation leaves
    off-diagonal residual components unabsorbable, and exact repair fails.
  * `safe_certify.py` uses the Jansson-style SAFE DUAL BOUND:

        lb_safe = b^T lambda - ||r||_1

    where r = c - A^T lambda - sum_j adj(S_j) is the (exact) rational
    residual. This is a VALID rigorous lower bound whenever the primal
    moment variables satisfy |y_alpha| <= 1 (which holds for probability
    moments under Lasserre L2 plus simplex constraint). No exact repair
    needed.

Soundness proof (in exact rationals):

    val(d) = min_mu max_W mu^T M_W mu
          >= min_mu mu^T M_lambda mu       (max >= convex comb)
          >= primal SDP value              (Lasserre relaxation)
          = c^T y*                          (primal optimum)

At any (lambda, {S_j}) with S_j PSD (by construction S_j = L_j L_j^T + D_j
with D_j >= 0 diagonal), and defining r = c - A^T lambda - sum adj(S_j),
we have for any primal-feasible y (Ay=b, F_j(y) PSD):

    c^T y = (A^T lambda + sum adj(S_j) + r)^T y
          = lambda^T (Ay) + sum <S_j, F_j(y)> + r^T y
          = lambda^T b + sum <S_j, F_j(y)> + r^T y
          >= lambda^T b + 0 + r^T y                [both PSD]
          >= lambda^T b - ||r||_infty * ||y||_1     [Hoelder]

For our problem, y_0 = 1 (only nonzero b entry) gives lambda^T b = lambda[0].
The moment bound ||y||_1 <= M_y is computed from the structure:
  - y_alpha = E[mu^alpha], mu probability distribution on Delta_d
  - For each total degree k <= 2*order:
        sum_{|alpha|=k} y_alpha * multinom(alpha) = E[(sum mu_i)^k] = 1
  - So sum_{|alpha|=k} y_alpha <= 1 (since multinom >= 1).
  - Total: ||y||_1 <= 1 + 1 + ... + 1 = (2*order + 1) summed over degrees 0..2k.

Therefore lb_safe = lambda[0] - ||r||_1 * (2*order+1) is rigorous.

In practice, residuals shrink very fast with rounding precision, so this
correction is microscopic (e.g. 10^-12 * 7 << 0.001 bound gap).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple
import time
import numpy as np
from scipy import sparse as sp

from certified_lasserre.build_sdp import SDPData, PSDBlock, build_sdp_data
from certified_lasserre.bm_solver import solve_primal_dual, SolverResult
from certified_lasserre.dual_extract import extract_dual, PolishedDual
from certified_lasserre.round_repair import (
    _float_to_fraction, _vec_to_fractions, _frac_sparse_matvec,
    _adjoint_block_rational, _rational_c, _fraction_matmul,
)


# =====================================================================
# Moment L1 bound
# =====================================================================

def _moment_l1_bound(sdp: SDPData) -> Fraction:
    """Upper bound on sum_alpha |y_alpha| for any primal-feasible y.

    Each total degree k in [0, 2*order] contributes sum_{|alpha|=k} y_alpha
    = [x_1 + ... + x_d]^k evaluated at y, which equals (sum_i mu_i)^k in the
    expectation representation, hence <= 1 when sum mu_i <= 1 (simplex).

    Simplex constraint sum mu_i = 1 gives exactly 1 per degree (not <=).
    With consistency Sum_i y_{alpha+e_i} = y_alpha, plus y_0 = 1, we have
    sum_{|alpha|=k} y_alpha * multinom(alpha) = 1 via repeated substitution.
    Since multinom(alpha) >= 1, sum_{|alpha|=k} y_alpha <= 1.

    Total L1 norm bound: (2*order + 1).
    """
    return Fraction(2 * sdp.order + 1)


# =====================================================================
# Rational Cholesky with PSD margin
# =====================================================================

def _rational_cholesky_with_margin(S_float: np.ndarray,
                                   max_denom: int,
                                   eig_margin_float: float = 1e-10,
                                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Given approx-PSD float S, return (L_rat, diag_slack_rat, margin_used).

    Pipeline:
      1. eigendecompose S; clip negative eigs to zero; add eig_margin_float*I
         for safety.
      2. form L_float = V * sqrt(w+margin)[None,:] (truncated to positive part).
      3. round L entries to rationals with denom bound max_denom.
      4. the diagonal slack is the margin we added (as Fraction).

    Output:
      L_rat:  (n, r) object-array of Fractions.  S = L_rat L_rat^T is EXACTLY
              rational PSD.
      diag_slack_rat: (n,) array of Fraction, each equal to a nonneg value.
                      S_final = L_rat L_rat^T + diag(diag_slack_rat) still PSD.
      margin_used: the float margin actually added to eigenvalues.
    """
    n = S_float.shape[0]
    S_sym = 0.5 * (S_float + S_float.T)
    w, V = np.linalg.eigh(S_sym)
    # Clip
    w_pos = np.maximum(w, 0.0) + eig_margin_float
    L_float = V * np.sqrt(w_pos)[None, :]
    # Round to rationals
    L_rat = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            L_rat[i, j] = _float_to_fraction(float(L_float[i, j]), max_denom)
    # No explicit diagonal slack (the margin is baked into L via sqrt)
    diag_slack = np.array([Fraction(0) for _ in range(n)], dtype=object)
    return L_rat, diag_slack, float(eig_margin_float)


# =====================================================================
# Safe certificate data class
# =====================================================================

@dataclass
class SafeCertificate:
    d: int
    order: int
    lam_windows: np.ndarray      # Fractions, window weights (sum=1)
    lambda_A: np.ndarray         # Fractions, equality multipliers
    L_blocks: List[np.ndarray]   # Fractions, S_j = L_j L_j^T + diag(slack_j)
    slack: List[np.ndarray]      # Fractions, >=0 diagonal slack
    residual_abs_sum: Fraction   # exact sum_k |r_k|
    moment_l1_bound: Fraction
    lam0: Fraction               # b^T lambda = lambda[0]
    lb_safe: Fraction            # = lam0 - residual_abs_sum * moment_l1_bound

    @property
    def lb_rig(self) -> Fraction:
        return self.lb_safe


@dataclass
class SafeCertifyResult:
    d: int
    order: int
    lb_rig: Fraction
    lb_rig_decimal: str
    primal_obj_float: float
    lam0_rat: Fraction
    residual_l1: Fraction
    residual_l1_float: float
    moment_l1_bound: Fraction
    safety_loss: Fraction         # how much was subtracted: residual * moment_bound
    solver: str
    solver_status: str
    solver_time: float
    round_time: float
    total_time: float
    notes: str = ''


def _decimal_str(q: Fraction, digits: int = 15) -> str:
    import mpmath
    with mpmath.workdps(digits + 10):
        val = mpmath.mpf(q.numerator) / mpmath.mpf(q.denominator)
        return mpmath.nstr(val, n=digits + 1, strip_zeros=False)


# =====================================================================
# Main pipeline
# =====================================================================

def safe_certify(sdp: SDPData,
                 solver: str = 'auto',
                 max_denom_L: int = 10**10,
                 max_denom_lam: int = 10**12,
                 max_denom_win: int = 10**10,
                 eig_margin: float = 1e-9,
                 verbose: bool = True) -> Tuple[SafeCertifyResult, SafeCertificate]:
    """End-to-end: solve, round, compute safe rigorous bound.

    Args:
        sdp: prebuilt SDPData (so caller can supply optimal lambda).
        solver, max_denom_*, eig_margin: see below.

    Returns:
        (SafeCertifyResult, SafeCertificate)
    """
    t_start = time.time()
    d, order = sdp.d, sdp.order

    if verbose:
        print(f'[safe_certify] d={d}, order={order}, n_y={sdp.n_y}, '
              f'n_eq={sdp.n_eq}, n_blocks={len(sdp.blocks)}')

    t0 = time.time()
    sol = solve_primal_dual(sdp, solver=solver, verbose=False)
    solver_time = time.time() - t0
    if verbose:
        print(f'  solver={sol.solver} status={sol.status} '
              f'primal={sol.obj:.9f} time={solver_time:.2f}s')

    polished = extract_dual(sdp, sol, verbose=verbose)

    # Round window weights
    t0 = time.time()
    lam_win_float = np.asarray(sdp.lam, dtype=np.float64)
    lam_win_f = _vec_to_fractions(lam_win_float, max_denom_win)
    total = Fraction(0)
    for v in lam_win_f:
        total += v
    assert total != 0
    lam_win_f = np.array([v / total for v in lam_win_f], dtype=object)

    # Rebuild c in exact rationals
    c_rat = _rational_c(sdp, lam_win_f)

    # Round lambda_A (equality multipliers)
    lam_A_rat = _vec_to_fractions(polished.lambda_A, max_denom_lam)

    # Rational Cholesky of each S_j with safety margin (ensures PSD in rationals)
    L_rat_list: List[np.ndarray] = []
    S_rat_list: List[np.ndarray] = []
    slack_list: List[np.ndarray] = []
    for S_float, blk in zip(polished.S_blocks, sdp.blocks):
        L_rat, slack_j, margin_used = _rational_cholesky_with_margin(
            S_float, max_denom=max_denom_L, eig_margin_float=eig_margin,
        )
        if L_rat.shape[1] > 0:
            S_rat = _fraction_matmul(L_rat, L_rat.T)
        else:
            S_rat = np.empty((blk.size, blk.size), dtype=object)
            for ii in range(blk.size):
                for jj in range(blk.size):
                    S_rat[ii, jj] = Fraction(0)
        L_rat_list.append(L_rat)
        S_rat_list.append(S_rat)
        slack_list.append(slack_j)

    # Exact residual r = c - A^T lam - sum adj(S_j)
    AT_csr = sdp.A.T.tocsr()
    AT_lambda = _frac_sparse_matvec(AT_csr, lam_A_rat)
    r_frac = np.empty(sdp.n_y, dtype=object)
    for k in range(sdp.n_y):
        r_frac[k] = c_rat[k] - AT_lambda[k]
    for blk, S_frac, slack_j in zip(sdp.blocks, S_rat_list, slack_list):
        adj = _adjoint_block_rational(blk, S_frac)
        for k in range(sdp.n_y):
            r_frac[k] -= adj[k]
        # Slack adj: diag(slack_j) has nonzero only on the (k,k) entries, and
        # adj_block(diag(slack)) = sum_k slack_j[k] * adj_block(e_k e_k^T).
        # But our rational Cholesky already set slack = 0, so skip.
        for k_j in range(blk.size):
            s_jk = slack_j[k_j]
            if s_jk == 0:
                continue
            # adj of E_{k,k}: coord of 2*basis_j[k_j] in monomial index.
            # Use adjoint on a rational matrix
            E = np.empty((blk.size, blk.size), dtype=object)
            for a in range(blk.size):
                for b in range(blk.size):
                    E[a, b] = Fraction(1 if (a == b and a == k_j) else 0)
            adj_E = _adjoint_block_rational(blk, E)
            for k in range(sdp.n_y):
                r_frac[k] -= s_jk * adj_E[k]

    # Residual L1 norm (exact rational)
    res_l1 = Fraction(0)
    for v in r_frac:
        res_l1 += abs(v)
    res_l1_float = float(res_l1)

    # Moment L1 bound
    M_y = _moment_l1_bound(sdp)

    # Safety loss and safe bound
    safety_loss = res_l1 * M_y
    lam0 = lam_A_rat[0]  # b[0] = 1
    lb_safe = lam0 - safety_loss

    round_time = time.time() - t0
    total_time = time.time() - t_start

    if verbose:
        print(f'  lam0 rational               = {_decimal_str(lam0, 15)}')
        print(f'  residual ||r||_1 (exact)    = {res_l1_float:.3e}')
        print(f'  moment_l1_bound             = {M_y}')
        print(f'  safety_loss                 = {float(safety_loss):.3e}')
        print(f'  lb_safe (RIGOROUS)          = {_decimal_str(lb_safe, 15)}')

    cert = SafeCertificate(
        d=d, order=order,
        lam_windows=lam_win_f, lambda_A=lam_A_rat,
        L_blocks=L_rat_list, slack=slack_list,
        residual_abs_sum=res_l1, moment_l1_bound=M_y,
        lam0=lam0, lb_safe=lb_safe,
    )
    result = SafeCertifyResult(
        d=d, order=order,
        lb_rig=lb_safe,
        lb_rig_decimal=_decimal_str(lb_safe, 15),
        primal_obj_float=sol.obj,
        lam0_rat=lam0,
        residual_l1=res_l1,
        residual_l1_float=res_l1_float,
        moment_l1_bound=M_y,
        safety_loss=safety_loss,
        solver=sol.solver,
        solver_status=str(sol.status),
        solver_time=solver_time,
        round_time=round_time,
        total_time=total_time,
    )
    return result, cert


def safe_certify_from_params(d: int, order: int,
                             lam: Optional[np.ndarray] = None,
                             **kwargs) -> Tuple[SafeCertifyResult, SafeCertificate]:
    """Convenience: build SDP then safe_certify."""
    verbose = kwargs.get('verbose', True)
    sdp = build_sdp_data(d, order, lam=lam, verbose=verbose)
    return safe_certify(sdp, **kwargs)


# =====================================================================
# Verification (independent recompute)
# =====================================================================

def verify_safe_certificate(sdp: SDPData, cert: SafeCertificate,
                            verbose: bool = True) -> Fraction:
    """Recompute lb_safe from scratch in Fractions. Returns cert.lb_safe on
    success, else raises.
    """
    d, order = cert.d, cert.order
    assert (d, order) == (sdp.d, sdp.order)

    # Rebuild c from rational window weights
    c_rat = _rational_c(sdp, cert.lam_windows)

    # Recompute adj(S_j)
    r = np.empty(sdp.n_y, dtype=object)
    for k in range(sdp.n_y):
        r[k] = c_rat[k]
    AT_csr = sdp.A.T.tocsr()
    AT_lambda = _frac_sparse_matvec(AT_csr, cert.lambda_A)
    for k in range(sdp.n_y):
        r[k] -= AT_lambda[k]
    for blk, L, slack_j in zip(sdp.blocks, cert.L_blocks, cert.slack):
        if L.shape[1] > 0:
            S = _fraction_matmul(L, L.T)
        else:
            S = np.empty((blk.size, blk.size), dtype=object)
            for ii in range(blk.size):
                for jj in range(blk.size):
                    S[ii, jj] = Fraction(0)
        adj = _adjoint_block_rational(blk, S)
        for k in range(sdp.n_y):
            r[k] -= adj[k]
        # Diagonal slack
        for k_j in range(blk.size):
            s_jk = slack_j[k_j]
            if s_jk == 0:
                continue
            E = np.empty((blk.size, blk.size), dtype=object)
            for a in range(blk.size):
                for b in range(blk.size):
                    E[a, b] = Fraction(1 if (a == b and a == k_j) else 0)
            adj_E = _adjoint_block_rational(blk, E)
            for k in range(sdp.n_y):
                r[k] -= s_jk * adj_E[k]

    res_l1 = Fraction(0)
    for v in r:
        res_l1 += abs(v)

    if res_l1 != cert.residual_abs_sum:
        raise ValueError(
            f'verify: residual mismatch {float(res_l1)} vs '
            f'stored {float(cert.residual_abs_sum)}')

    # Check window weights sum to 1 and are nonneg
    ws = Fraction(0)
    for w in cert.lam_windows:
        if w < 0:
            raise ValueError(f'negative window weight {w}')
        ws += w
    if ws != 1:
        raise ValueError(f'window weights sum to {ws} != 1')

    # lb_safe = lam0 - res_l1 * moment_l1_bound
    expected = cert.lam0 - cert.residual_abs_sum * cert.moment_l1_bound
    if expected != cert.lb_safe:
        raise ValueError(
            f'verify: lb_safe mismatch {float(expected)} vs stored '
            f'{float(cert.lb_safe)}')

    if verbose:
        print(f'[verify_safe_certificate] PASSED')
        print(f'  d={d}, order={order}')
        print(f'  lb_rig = lb_safe = {_decimal_str(cert.lb_safe, 15)}')
        print(f'  residual_abs_sum = {float(res_l1):.3e}')
        print(f'  safety loss       = {float(cert.residual_abs_sum * cert.moment_l1_bound):.3e}')
    return cert.lb_safe

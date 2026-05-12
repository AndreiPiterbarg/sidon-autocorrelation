"""Tier-1 SCHUR ELIMINATION of the q variables in the Polya/Handelman LP.

==============================================================================
MATHEMATICAL DERIVATION
==============================================================================

The original LP, per multi-index beta with |beta| <= R:

    sum_W lambda_W coeff_W(beta)
        - alpha * delta_{beta=0}
        + q_beta - sum_j q_{beta - e_j}
        - c_beta
      = 0

where q_K is defined for |K| <= R-1 (q_K = 0 outside that range), and
coeff_W(beta) is the coefficient of mu^beta in mu^T M_W mu, nonzero only
for |beta| = 2.

Define r_beta := sum_W lambda_W coeff_W(beta) - alpha * delta_{beta=0}.
"Group A" rows (|beta| <= R-1) include q_beta and so DEFINE q_beta:

    q_beta = sum_j q_{beta - e_j} + c_beta - r_beta

By induction in degree, this closed form follows:

    q_beta = sum_{gamma <= beta} mult(beta - gamma) (c_gamma - r_gamma)

where mult(tau) = (|tau|)! / prod_i tau_i! is the multinomial coefficient,
and "<=" is the componentwise ordering on multi-indices.

"Group B" rows (|beta| = R) have q_beta == 0 (since |beta| > R-1) and become

    -sum_j q_{beta - e_j} = c_beta - r_beta

i.e.

    c_beta - r_beta = -sum_j q_{beta - e_j}
    c_beta = r_beta - sum_j q_{beta - e_j}.

Substituting the closed form:

    sum_j q_{beta - e_j}
      = sum_j sum_{gamma <= beta - e_j} mult(beta - e_j - gamma) (c_gamma - r_gamma)
      = sum_{gamma < beta} N_c(beta, gamma) (c_gamma - r_gamma)

where the j-sum collapses (proof: by direct manipulation of the multinomial
identity, the sum over j with beta_j > gamma_j of mult(beta - e_j - gamma)
equals mult(beta - gamma) for any gamma <= beta with gamma != beta).

Therefore Group B becomes

    c_beta - sum_{gamma < beta} mult(beta - gamma) c_gamma
        = r_beta - sum_{gamma < beta} mult(beta - gamma) r_gamma

which expands the RHS using r_gamma = sum_W lambda_W coeff_W(gamma)
- alpha delta_{gamma=0}:

    c_beta - sum_{gamma < beta} mult(beta - gamma) c_gamma
        - sum_W lambda_W L_W(beta) - alpha mult(beta) = 0

where

    L_W(beta) = coeff_W(beta) - sum_{gamma < beta} mult(beta - gamma) coeff_W(gamma)
              = -sum_{gamma <= beta, |gamma|=2, gamma != beta} mult(beta - gamma) coeff_W(gamma)
                + coeff_W(beta) [nonzero only if |beta|=2 = R]

For the typical |beta| = R >= 3 case, coeff_W(beta) = 0, so:

    L_W(beta) = - sum_{gamma <= beta, |gamma|=2} mult(beta - gamma) coeff_W(gamma)

==============================================================================
ELIMINATED LP
==============================================================================

Variables:
    alpha            (free, 1)
    lambda_W         (>= 0, n_W; sum lambda_W = 1)
    c_beta           (>= 0, |beta| <= R; n_le_R total)

Equality constraints:
    [Group B, |beta|=R]:  c_beta - sum_{gamma<beta} mult(beta-gamma) c_gamma
                          + sum_W lambda_W L_W(beta) - alpha mult(beta) = 0
    [Simplex]:            sum_W lambda_W = 1

Objective: minimize -alpha   (equivalent to maximize alpha)

Soundness: this is a STRICT REFORMULATION of the original LP (q is
algebraically determined by the recurrence and substituted). The optimal
alpha matches the original LP exactly. No relaxation, no approximation.

==============================================================================
NUMERICAL NOTES
==============================================================================

For (d_eff, R) = (8, 12), max multinomial is mult((2,2,2,2,1,1,1,1)) =
12!/(2!^4 1!^4) = 29,937,600. So coefficient dynamic range is roughly
1e0 to 3e7. Within MOSEK's auto-scaling capability (1e-6 to 1e6 typical).

For (d_eff, R) = (8, 27) target, max mult is unbounded by combinatorial
explosion. We expect ~10^15 to 10^25 dynamic range. Will need rescaling.
For now, focus on R<=15 where MOSEK can handle.

==============================================================================
IMPLEMENTATION
==============================================================================

Iterating over all (beta, gamma) pairs with gamma <= beta and |beta|=R is
expensive: total pairs = sum_k C(d-1+k, d-1) C(d-1+R-k, d-1) ≈ tens of
millions at d=8, R=12. We vectorize via NumPy: for each gamma in Sigma_R,
generate all tau with |tau| = R - |gamma| (yielding beta = gamma + tau),
compute mult(tau), and emit COO triplets in batch.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict
import math
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import BuildOptions, BuildResult
from lasserre.polya_lp.poly import enum_monomials_le, enum_monomials_eq


# =====================================================================
# Helpers
# =====================================================================

def _factorial_table(R: int) -> np.ndarray:
    """Precompute factorials up to R. Returns (R+1,) float64 array."""
    f = np.empty(R + 1, dtype=np.float64)
    f[0] = 1.0
    for k in range(1, R + 1):
        f[k] = f[k-1] * k
    return f


def _build_void_lookup(monos: List[Tuple[int, ...]], d: int) -> Dict[bytes, int]:
    """Map monomial bytes -> row index. Same trick as build._batch_lookup."""
    if not monos:
        return {}
    arr = np.asarray(monos, dtype=np.int8)
    arr = np.ascontiguousarray(arr)
    void_dt = np.dtype((np.void, d * 1))
    void = arr.view(void_dt).ravel()
    return {bytes(v): i for i, v in enumerate(void.tolist())}


def _multinomial_vec(tau_arr: np.ndarray, fact: np.ndarray) -> np.ndarray:
    """Multinomial coefficient over rows of (N, d) int array tau_arr.

    mult(tau) = (sum tau)! / prod_i tau_i!
    Returns float64 (N,).
    """
    if tau_arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    sums = tau_arr.sum(axis=1, dtype=np.int64)            # (N,)
    # numerator
    num = fact[sums]                                       # (N,)
    # denominator: prod fact[tau_i]
    # gather fact at each tau entry, multiply across columns
    den = np.prod(fact[tau_arr.astype(np.int64)], axis=1)  # (N,)
    return num / den


# =====================================================================
# Coefficient tables
# =====================================================================

def _build_lambda_coef_block(
    d: int,
    R: int,
    M_mats: Sequence[np.ndarray],
    monos_eq_R: List[Tuple[int, ...]],
    fact: np.ndarray,
) -> sp.csr_matrix:
    """Build the (n_eq_R, n_W) lambda block of the Schur LP.

    Group B equation: sum_{gamma <= beta} mult(beta-gamma) c_gamma
                      + alpha mult(beta) - sum_W lambda_W K_W(beta) = 0
    where K_W(beta) = sum_{gamma <= beta, |gamma|=2} mult(beta-gamma) coeff_W(gamma).

    The COEFFICIENT of lambda_W in the Group B row at beta is therefore
    -K_W(beta) (negative; lambda_W moves to the same side as c).

    coeff_W(2*e_i)     = M_W[i, i]
    coeff_W(e_i + e_j) = 2 * M_W[i, j]   (i < j; M_W symmetric)
    coeff_W(gamma) = 0 if |gamma| != 2.

    Vectorized: for each (i, j) pair (gamma = e_i + e_j or 2*e_i),
    compute mult(beta - gamma) for each beta with gamma <= beta, then
    add -mult(beta - gamma) * coeff_W(gamma) into entry (beta_idx, w).
    """
    n_eq_R = len(monos_eq_R)
    n_W = len(M_mats)
    if n_W == 0 or n_eq_R == 0:
        return sp.csr_matrix((n_eq_R, n_W))

    M_stack = np.stack([np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)
    # M_stack shape (n_W, d, d).

    beta_arr = np.asarray(monos_eq_R, dtype=np.int8)       # (n_eq_R, d)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    w_arange = np.arange(n_W, dtype=np.int64)

    # ----- diagonal gammas: gamma = 2*e_i (|gamma|=2) -----
    for i in range(d):
        # gamma = 2*e_i is in monos_le_R iff beta has beta_i >= 2.
        mask = (beta_arr[:, i] >= 2)
        if not mask.any():
            continue
        sub_betas = beta_arr[mask]                          # (k, d)
        sub_idx = np.flatnonzero(mask)
        tau = sub_betas.copy()
        tau[:, i] -= 2
        tau = tau.astype(np.int64)
        mult = _multinomial_vec(tau, fact)                  # (k,)
        # coef_W(gamma) = M_W[i, i]
        coef_M = M_stack[:, i, i]                           # (n_W,)
        k = sub_idx.size
        beta_grid = np.broadcast_to(sub_idx[:, None], (k, n_W)).reshape(-1)
        w_grid = np.broadcast_to(w_arange[None, :], (k, n_W)).reshape(-1)
        # contribution to lambda coef in row beta: -mult(beta-gamma) * coeff_W(gamma)
        v = (-mult)[:, None] * coef_M[None, :]              # (k, n_W)
        v_flat = v.reshape(-1)
        keep = v_flat != 0
        rows.append(beta_grid[keep])
        cols.append(w_grid[keep])
        vals.append(v_flat[keep])

    # ----- cross gammas: gamma = e_i + e_j with i < j (|gamma|=2) -----
    for i in range(d):
        for j in range(i + 1, d):
            mask = (beta_arr[:, i] >= 1) & (beta_arr[:, j] >= 1)
            if not mask.any():
                continue
            sub_betas = beta_arr[mask]
            sub_idx = np.flatnonzero(mask)
            tau = sub_betas.copy()
            tau[:, i] -= 1
            tau[:, j] -= 1
            tau = tau.astype(np.int64)
            mult = _multinomial_vec(tau, fact)
            # coef_W(gamma) = 2 * M_W[i, j]
            coef_M = 2.0 * M_stack[:, i, j]                  # (n_W,)
            k = sub_idx.size
            beta_grid = np.broadcast_to(sub_idx[:, None], (k, n_W)).reshape(-1)
            w_grid = np.broadcast_to(w_arange[None, :], (k, n_W)).reshape(-1)
            v = (-mult)[:, None] * coef_M[None, :]
            v_flat = v.reshape(-1)
            keep = v_flat != 0
            rows.append(beta_grid[keep])
            cols.append(w_grid[keep])
            vals.append(v_flat[keep])

    # Note: for |gamma|=2 with gamma == beta (only possible if R == 2), the
    # mult(0) = 1 case is already correctly included above (tau is the zero
    # vector; mult is 1; v contribution is -1 * coeff_W(beta)). No special
    # case needed.

    if rows:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_vals = np.concatenate(vals)
    else:
        all_rows = np.zeros(0, dtype=np.int64)
        all_cols = np.zeros(0, dtype=np.int64)
        all_vals = np.zeros(0, dtype=np.float64)

    A = sp.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_eq_R, n_W),
    )
    A.sum_duplicates()
    return A


def _build_c_coef_block(
    d: int,
    R: int,
    monos_le_R: List[Tuple[int, ...]],
    monos_eq_R: List[Tuple[int, ...]],
    fact: np.ndarray,
) -> sp.csr_matrix:
    """Build the (n_eq_R, n_le_R) block where entry (beta_idx, gamma_idx) is
    +mult(beta - gamma) if gamma <= beta else 0.

    Recall the Group B equation (after q-elimination):
       sum_{gamma <= beta} mult(beta - gamma) c_gamma + alpha mult(beta)
           - sum_W lambda_W K_W(beta) = 0

    so the c-coefficient is +mult(beta-gamma) for ALL gamma <= beta
    (including gamma == beta, where mult(0) = 1).

    Strategy: iterate over GAMMA in Sigma_R. For each gamma, generate all
    tau with |tau| = R - |gamma|, tau >= 0. Then beta = gamma + tau, and
    we emit triplet (beta_idx, gamma_idx, +mult(tau)).
    """
    n_eq_R = len(monos_eq_R)
    n_le_R = len(monos_le_R)

    # Lookup beta -> row index in monos_eq_R
    eq_R_lookup = _build_void_lookup(monos_eq_R, d)

    # Group gamma by degree to vectorize tau generation per-degree
    # Pre-enumerate Delta_k = {tau in N^d : |tau| = k} for k = 0..R
    delta_by_k: Dict[int, np.ndarray] = {}
    for k in range(R + 1):
        delta_k = enum_monomials_eq(d, k)
        if delta_k:
            delta_by_k[k] = np.asarray(delta_k, dtype=np.int8)
        else:
            delta_by_k[k] = np.zeros((0, d), dtype=np.int8)

    rows_chunks: List[np.ndarray] = []
    cols_chunks: List[np.ndarray] = []
    vals_chunks: List[np.ndarray] = []

    void_dt = np.dtype((np.void, d * 1))

    monos_le_R_arr = np.asarray(monos_le_R, dtype=np.int8)

    for gamma_idx, gamma in enumerate(monos_le_R):
        gamma_deg = int(sum(gamma))
        k = R - gamma_deg
        if k < 0:
            continue
        delta_k = delta_by_k[k]                            # (n_tau, d)
        n_tau = delta_k.shape[0]
        if n_tau == 0:
            continue
        gamma_arr = np.asarray(gamma, dtype=np.int8)
        # beta = gamma + tau
        beta_arr = (delta_k.astype(np.int64) + gamma_arr.astype(np.int64))
        beta_arr = beta_arr.astype(np.int8)
        beta_arr = np.ascontiguousarray(beta_arr)
        beta_void = beta_arr.view(void_dt).ravel()
        # Lookup beta index in monos_eq_R
        beta_indices = np.fromiter(
            (eq_R_lookup.get(bytes(v), -1) for v in beta_void.tolist()),
            dtype=np.int64, count=n_tau,
        )
        valid = beta_indices >= 0
        if not valid.any():
            continue
        beta_indices = beta_indices[valid]
        valid_tau = delta_k[valid].astype(np.int64)

        # mult(tau) for each valid tau (mult(0) = 1)
        mult = _multinomial_vec(valid_tau, fact)           # (m,)
        # Coef = +mult for ALL tau (including tau == 0 -> mult = 1).
        coefs = mult

        rows_chunks.append(beta_indices.copy())
        cols_chunks.append(np.full(beta_indices.size, gamma_idx, dtype=np.int64))
        vals_chunks.append(coefs.copy())

    if rows_chunks:
        all_rows = np.concatenate(rows_chunks)
        all_cols = np.concatenate(cols_chunks)
        all_vals = np.concatenate(vals_chunks)
    else:
        all_rows = np.zeros(0, dtype=np.int64)
        all_cols = np.zeros(0, dtype=np.int64)
        all_vals = np.zeros(0, dtype=np.float64)

    return sp.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_eq_R, n_le_R),
    )


def _build_alpha_coef_col(
    R: int,
    monos_eq_R: List[Tuple[int, ...]],
    fact: np.ndarray,
) -> np.ndarray:
    """Coefficient on alpha in each Group B row: +mult(beta).

    Derivation: the Group B equation is
        sum_{gamma <= beta} mult(beta-gamma) c_gamma
            + alpha mult(beta) - sum_W lambda_W K_W(beta) = 0
    coming from substituting r_gamma = (lambda part) - alpha delta_{gamma=0}.
    The alpha contribution from the recurrence sum is
       -sum_{gamma <= beta} mult(beta-gamma) * (-alpha delta_{gamma=0})
       = +alpha * mult(beta-0)  (only gamma=0 contributes)
       = +alpha * mult(beta).
    Moving to LHS = 0 form preserves the +mult(beta) sign on alpha.
    """
    if not monos_eq_R:
        return np.zeros(0, dtype=np.float64)
    beta_arr = np.asarray(monos_eq_R, dtype=np.int64)
    return _multinomial_vec(beta_arr, fact)


# =====================================================================
# Top-level builder
# =====================================================================

def build_handelman_lp_schur(
    d: int,
    M_mats: Sequence[np.ndarray],
    options: BuildOptions,
) -> BuildResult:
    """Construct the Schur-eliminated Polya/Handelman LP.

    The eliminated LP has:
      - Variables: alpha (1), lambda_W (n_W), c_beta (n_le_R)  [no q]
      - Equalities: |beta|=R rows (n_eq_R) + 1 simplex
      - Bounds: alpha free, lambda >= 0, c >= 0

    Equivalent to the original Polya LP (max alpha) under q substitution.
    """
    t0 = time.time()
    R = options.R
    n_vars_poly = d

    if R < 2:
        raise ValueError("Schur-elim requires R >= 2 (need degree-R Group B rows).")

    # Enumerate
    monos_le_R = enum_monomials_le(d, R)
    n_le_R = len(monos_le_R)
    monos_eq_R = enum_monomials_eq(d, R)
    n_eq_R = len(monos_eq_R)
    n_W = len(M_mats)
    if options.fixed_lambda is not None:
        # Schur-elim with fixed lambda is supported; aggregates lambda block to RHS.
        # For now only support free lambda.
        raise NotImplementedError("Schur-elim with fixed lambda not implemented yet.")
    n_lambda_var = n_W

    # Variable layout: alpha | lambda_W | c_beta
    alpha_idx = 0
    lambda_start = 1
    lambda_stop = 1 + n_lambda_var
    c_start = lambda_stop
    c_stop = c_start + n_le_R
    n_vars = c_stop

    if options.verbose:
        print(f"  [Schur] LP layout: alpha=1, lambda={n_lambda_var}, "
              f"c={n_le_R}; total n_vars={n_vars}", flush=True)
        print(f"  [Schur] Group B rows: {n_eq_R}; plus 1 simplex.",
              flush=True)

    # Factorial table
    fact = _factorial_table(R)

    # ----- Build A_eq (Group B + simplex) -----
    t_block = time.time()
    A_c = _build_c_coef_block(d, R, monos_le_R, monos_eq_R, fact)  # (n_eq_R, n_le_R)
    if options.verbose:
        print(f"  [Schur] c-block: nnz={A_c.nnz}, "
              f"t={time.time()-t_block:.2f}s", flush=True)
    t_block = time.time()
    A_lam = _build_lambda_coef_block(d, R, M_mats, monos_eq_R, fact)  # (n_eq_R, n_W)
    if options.verbose:
        print(f"  [Schur] lambda-block: nnz={A_lam.nnz}, "
              f"t={time.time()-t_block:.2f}s", flush=True)
    alpha_col = _build_alpha_coef_col(R, monos_eq_R, fact)              # (n_eq_R,)

    # Assemble Group B as horizontal stack: [alpha_col | A_lam | A_c]
    alpha_block = sp.csr_matrix(alpha_col.reshape(-1, 1))               # (n_eq_R, 1)
    A_groupB = sp.hstack([alpha_block, A_lam, A_c], format="csr")       # (n_eq_R, n_vars)

    # Simplex row: sum lambda = 1
    sim_row = sp.csr_matrix(
        (np.ones(n_W),
         (np.zeros(n_W, dtype=np.int64),
          np.arange(lambda_start, lambda_stop, dtype=np.int64))),
        shape=(1, n_vars),
    )
    A_eq = sp.vstack([A_groupB, sim_row], format="csr")
    b_eq = np.concatenate([np.zeros(n_eq_R), np.array([1.0])])

    # ----- Bounds -----
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    fb = options.free_var_box
    bounds.append((-fb, fb) if fb else (None, None))   # alpha
    for _ in range(n_lambda_var):
        bounds.append((0.0, None))                     # lambda >= 0
    for _ in range(n_le_R):
        bounds.append((0.0, None))                     # c >= 0

    # ----- Objective: minimize -alpha -----
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[alpha_idx] = -1.0

    return BuildResult(
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=None,
        b_ub=None,
        c=c_obj,
        bounds=bounds,
        n_vars=n_vars,
        alpha_idx=alpha_idx,
        lambda_idx=slice(lambda_start, lambda_stop),
        q_idx=slice(c_start, c_start),                  # empty (no q)
        c_idx=slice(c_start, c_stop),
        monos_le_R=monos_le_R,
        monos_le_Rm1=[],                                # no q vars
        n_windows=n_W,
        fixed_lambda=None,
        options=options,
        build_wall_s=time.time() - t0,
        n_nonzero_A=A_eq.nnz,
        c_slacks_eliminated=False,
    )

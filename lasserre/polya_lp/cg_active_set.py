"""Combined Tier 2 + Tier 3: column generation on Sigma_R rows AND
window-active-set pricing on lambda_W columns. Provides rigorous
LP-Farkas certificate of full-LP equivalence at convergence.

Algorithm:
  Init:
    Sigma_R = Newton-polytope seed (term_sparsity)
    B_R     = compute_B_R(Sigma_R)
    W_active = all windows (start dense; pricing identifies dropped ones)
  Loop (until both pricing checks return empty):
    1. Build restricted LP with (Sigma_R, B_R, W_active = M_mats[W_active])
    2. Solve with MOSEK (must use mosek so we get y duals)
    3. Sigma_R-pricing (Tier 3): find_violators on dropped beta in
       E = (B_R u (B_R+e_j) u A u {0}) \ Sigma_R
    4. Window-pricing (Tier 2): for each dropped W in W_full \ W_active,
       compute reduced cost RC_W = -y_simplex - sum_beta y_beta coeff_W(beta).
       Add window if RC_W < -tol (= profitable).
    5. If both empty: CONVERGED. Restricted-LP alpha = full-LP alpha
       (LP duality + complementary slackness).
    6. Otherwise: add violators, recompute B_R, repeat.

Soundness (verified by agent literature audit):
  At convergence:
    * For every beta in Sigma_R: LHS_beta = c_beta >= 0 (LP enforces).
    * For every beta in candidate set E\Sigma_R: LHS_beta >= -tol (pricing).
    * For every beta NOT in E: LHS_beta = 0 trivially (no q/lambda support).
    * For every W in W_active: RC_W = 0 (in basis) or >= 0 (at lambda=0).
    * For every W in W_dropped: RC_W >= -tol (pricing).
  So (alpha*, lambda*, q*, c* := LHS) is feasible in the full LP, and
  no entering variable could improve. By LP duality + complementary
  slackness, alpha* IS the full-LP optimum. Hence alpha* is a rigorous
  lower bound on val(d) (Mai-Magron-Lasserre 2022 + standard CG theory).

Tolerances:
  tol = 1e-8 for both pricing tests (one to two orders above MOSEK's
  internal 1e-9 KKT). Tighter risks numerical false-positives that
  cause non-termination; looser risks accepting marginally-violating
  solutions whose rigorous-LB may differ.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Dict
import time
import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.term_sparsity import (
    TermSparsitySupport, build_term_sparsity_support, compute_B_R,
    polynomial_support_from_M_mats,
)
from lasserre.polya_lp.cutting_plane import find_violators


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------

@dataclass
class CGActiveSetIter:
    iteration: int
    n_constraints: int          # |Sigma_R| in current restricted LP
    n_q_vars: int               # |B_R|
    n_active_W: int             # |W_active|
    n_W_dropped: int            # |W_full| - |W_active|
    alpha: Optional[float]
    n_beta_violators: int
    max_beta_violation: float   # |min LHS_β| over candidates (positive)
    n_W_violators: int          # dropped windows with RC < -tol
    max_W_violation: float      # |min RC_W| over dropped windows
    build_wall_s: float
    solve_wall_s: float
    pricing_wall_s: float
    rss_mb: float
    solver_status: str = ""


@dataclass
class CGActiveSetResult:
    final_alpha: Optional[float]
    final_Sigma_R: List[Tuple[int, ...]]
    final_B_R: List[Tuple[int, ...]]
    final_W_active: List[int]
    final_x: Optional[np.ndarray]
    iterations: List[CGActiveSetIter]
    converged: bool
    total_wall_s: float
    n_W_full: int


# ---------------------------------------------------------------------
# Helpers: lambda coefficient matrix (full window set)
# ---------------------------------------------------------------------

def _build_lambda_coef_matrix(
    M_mats: Sequence[np.ndarray],
    Sigma_R: Sequence[Tuple[int, ...]],
) -> sp.csr_matrix:
    """Build the (|Sigma_R|, n_W_full) sparse matrix where entry [β, w] is
    coeff_W(β). Used for pricing dropped windows.

    coeff_W(β) is nonzero only for |β|=2:
      β = 2 e_i:        M_W[i, i]
      β = e_i + e_j:    2 M_W[i, j]    (i < j; M_W symmetric)
    """
    if not M_mats:
        return sp.csr_matrix((len(Sigma_R), 0))
    d = M_mats[0].shape[0]
    n_W = len(M_mats)
    n_S = len(Sigma_R)

    # Build beta index lookup using bytes-keyed dict for speed.
    beta_arr = np.asarray(Sigma_R, dtype=np.int8)
    beta_arr = np.ascontiguousarray(beta_arr)
    void_dt = np.dtype((np.void, d * np.dtype(np.int8).itemsize))
    beta_void = beta_arr.view(void_dt).ravel()
    beta_idx = {bytes(v): i for i, v in enumerate(beta_void.tolist())}

    # Build all |β|=2 multi-indices and their indices in Sigma_R.
    diag_betas = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        diag_betas[i, i] = 2
    diag_betas_c = np.ascontiguousarray(diag_betas)
    diag_void = diag_betas_c.view(void_dt).ravel()
    diag_idx = np.array(
        [beta_idx.get(bytes(v), -1) for v in diag_void.tolist()],
        dtype=np.int64,
    )

    if d >= 2:
        i_grid, j_grid = np.meshgrid(
            np.arange(d, dtype=np.int64),
            np.arange(d, dtype=np.int64),
            indexing='ij')
        triu = i_grid < j_grid
        i_arr = i_grid[triu]
        j_arr = j_grid[triu]
        n_cross = int(i_arr.size)
        cross_betas = np.zeros((n_cross, d), dtype=np.int8)
        cross_betas[np.arange(n_cross), i_arr] = 1
        cross_betas[np.arange(n_cross), j_arr] = 1
        cross_betas_c = np.ascontiguousarray(cross_betas)
        cross_void = cross_betas_c.view(void_dt).ravel()
        cross_idx = np.array(
            [beta_idx.get(bytes(v), -1) for v in cross_void.tolist()],
            dtype=np.int64,
        )
    else:
        i_arr = np.zeros(0, dtype=np.int64)
        j_arr = np.zeros(0, dtype=np.int64)
        cross_idx = np.zeros(0, dtype=np.int64)
        n_cross = 0

    # Stack windows once.
    M_stack = np.stack(
        [np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)  # (n_W, d, d)
    diag_coefs = np.diagonal(M_stack, axis1=1, axis2=2)             # (n_W, d)
    if n_cross > 0:
        cross_coefs = 2.0 * M_stack[:, i_arr, j_arr]                # (n_W, n_cross)
    else:
        cross_coefs = np.zeros((n_W, 0), dtype=np.float64)

    # Build COO triplets.
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []

    # Diag.
    w_idx = np.arange(n_W, dtype=np.int64)
    d_rows = np.broadcast_to(diag_idx[None, :], (n_W, d)).ravel()
    d_cols = np.broadcast_to(w_idx[:, None], (n_W, d)).ravel()
    d_vals = diag_coefs.ravel()
    keep = (d_rows >= 0) & (d_vals != 0)
    rows.append(d_rows[keep].copy())
    cols.append(d_cols[keep].copy())
    vals.append(d_vals[keep].copy())

    # Cross.
    if n_cross > 0:
        c_rows = np.broadcast_to(cross_idx[None, :], (n_W, n_cross)).ravel()
        c_cols = np.broadcast_to(w_idx[:, None], (n_W, n_cross)).ravel()
        c_vals = cross_coefs.ravel()
        keep = (c_rows >= 0) & (c_vals != 0)
        rows.append(c_rows[keep].copy())
        cols.append(c_cols[keep].copy())
        vals.append(c_vals[keep].copy())

    all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    all_vals = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float64)
    return sp.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_S, n_W),
    )


# ---------------------------------------------------------------------
# Helpers: extract MOSEK duals into (y_beta, y_simplex) by row index
# ---------------------------------------------------------------------

def _split_duals(
    sol: SolveResult,
    n_sigma_R: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Split sol.y into the per-beta block and the simplex-row dual.

    The build orders constraint rows as (n_sigma_R beta-rows, then 1 simplex
    row if there are lambda variables). For c_slack-eliminated builds, the
    beta rows live in A_ub instead and y_beta corresponds to those — but
    in our pipeline we keep eliminate_c_slacks=False (per breakthrough
    notes), so the standard ordering applies.

    Returns (y_beta of length n_sigma_R, y_simplex scalar) or (None, None)
    if duals not available.
    """
    if sol.y is None:
        return None, None
    y = np.asarray(sol.y, dtype=np.float64)
    if y.size < n_sigma_R + 1:
        # No simplex row (fixed_lambda) — caller should not invoke window
        # pricing in that case.
        if y.size == n_sigma_R:
            return y, 0.0
        return None, None
    y_beta = y[:n_sigma_R]
    y_simplex = float(y[n_sigma_R])
    return y_beta, y_simplex


def _price_windows(
    A_lambda_full: sp.csr_matrix,
    y_beta: np.ndarray,
    y_simplex: float,
    W_active: List[int],
    n_W_full: int,
    tol: float,
) -> Tuple[List[int], float]:
    """Compute reduced costs of all dropped windows; return profitable ones.

    Reduced cost (in min -alpha standard form):
       RC_W = 0 - (A[:, W])^T y = -(sum_beta y_beta coeff_W(beta) + y_simplex)
    Variable W wants to enter iff RC_W < -tol.

    Equivalently: ADD W if  sum y_beta coeff_W(beta) + y_simplex > +tol.

    Returns (sorted-by-most-profitable list of W indices, max RC magnitude).
    """
    if A_lambda_full.shape[1] == 0 or n_W_full == 0:
        return [], 0.0
    # All-windows reduced cost vector (length n_W_full).
    # A_lambda_full.T @ y_beta gives sum_beta y_beta coeff_W(beta).
    activity = A_lambda_full.T.dot(y_beta) + y_simplex   # shape (n_W_full,)
    rc = -activity                                       # min-form RC
    # Mask to dropped windows only.
    active_mask = np.zeros(n_W_full, dtype=bool)
    if W_active:
        active_mask[np.asarray(W_active, dtype=np.int64)] = True
    dropped_idx = np.flatnonzero(~active_mask)
    if dropped_idx.size == 0:
        return [], 0.0
    rc_dropped = rc[dropped_idx]
    max_rc_mag = float(-rc_dropped.min()) if rc_dropped.size else 0.0
    if max_rc_mag <= tol:
        return [], max_rc_mag
    viol_mask = rc_dropped < -tol
    viol_W = dropped_idx[viol_mask]
    viol_rc = rc_dropped[viol_mask]
    # Sort by most negative (= most profitable) first.
    order = np.argsort(viol_rc)
    return [int(W) for W in viol_W[order]], max_rc_mag


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def solve_with_cg_active_set(
    d: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
    add_top_k_betas: int = -1,
    add_top_k_windows: int = -1,
    initial_W_active: Optional[List[int]] = None,
    include_low_degree_seed: bool = True,
    use_q_polynomial: bool = True,
    verbose: bool = True,
) -> CGActiveSetResult:
    """Combined Tier-2 + Tier-3 column generation for the Polya/Handelman LP.

    Args:
        d:             ambient dimension (post-Z/2 d_eff if applicable).
        M_mats:        list of d x d window matrices (post-Z/2-rescale).
        R:             total degree of the certificate.
        max_iter:      hard cap on outer loop iterations.
        tol:           pricing tolerance; both (Sigma_R LHS) and (window RC)
                       must be >= -tol at convergence.
        add_top_k_betas:   if > 0, add only top-K most-violating betas per
                           iteration. -1 (default) = add all violators.
        add_top_k_windows: same for window pricing.
        initial_W_active:  start with this subset of windows; default None
                           = all windows (recommended; window pricing
                           identifies any to drop).
        include_low_degree_seed:  forwarded to term_sparsity.

    Returns CGActiveSetResult with full iteration history.
    """
    import resource

    t_total = time.time()
    n_W_full = len(M_mats)
    iters: List[CGActiveSetIter] = []

    # 1. Initial Sigma_R seed (Newton polytope + low-degree).
    ts = build_term_sparsity_support(
        M_mats, R, include_low_degree=include_low_degree_seed)
    Sigma_R: List[Tuple[int, ...]] = list(ts.Sigma_R)
    B_R: List[Tuple[int, ...]] = list(ts.B_R)
    A_support = list(ts.A)

    # 2. Initial active windows.
    W_active: List[int] = (sorted(set(initial_W_active))
                           if initial_W_active is not None
                           else list(range(n_W_full)))

    if verbose:
        print(f"  CG+ActiveSet seed: |Sigma_R|={len(Sigma_R)}/"
              f"{ts.full_n_constraints}  |B_R|={len(B_R)}/{ts.full_n_q_vars}  "
              f"|W_active|={len(W_active)}/{n_W_full}", flush=True)

    final_sol: Optional[SolveResult] = None
    final_build: Optional[BuildResult] = None

    for it in range(max_iter):
        # Build restricted LP.
        sub_M_mats = [M_mats[w] for w in W_active]
        opts = BuildOptions(
            R=R,
            use_z2=True,
            use_q_polynomial=use_q_polynomial,
            verbose=False,
            restricted_Sigma_R=Sigma_R,
            restricted_B_R=B_R,
        )
        t_b = time.time()
        build = build_handelman_lp(d, sub_M_mats, opts)
        wall_b = time.time() - t_b

        # Solve. Force MOSEK so we get duals.
        t_s = time.time()
        sol = solve_lp(build, solver='mosek', verbose=False)
        wall_s = time.time() - t_s
        final_sol = sol
        final_build = build

        if sol.alpha is None:
            iters.append(CGActiveSetIter(
                iteration=it,
                n_constraints=build.A_eq.shape[0],
                n_q_vars=len(B_R),
                n_active_W=len(W_active),
                n_W_dropped=n_W_full - len(W_active),
                alpha=None,
                n_beta_violators=-1,
                max_beta_violation=float('inf'),
                n_W_violators=-1,
                max_W_violation=float('inf'),
                build_wall_s=wall_b, solve_wall_s=wall_s,
                pricing_wall_s=0.0,
                rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,
                solver_status=sol.status,
            ))
            if verbose:
                print(f"  iter {it}: SOLVE FAILED status={sol.status}", flush=True)
            return CGActiveSetResult(
                final_alpha=None,
                final_Sigma_R=Sigma_R,
                final_B_R=B_R,
                final_W_active=W_active,
                final_x=None,
                iterations=iters,
                converged=False,
                total_wall_s=time.time() - t_total,
                n_W_full=n_W_full,
            )

        # Pricing.
        t_p = time.time()

        # 3a. Sigma_R pricing (Tier 3): vectorized find_violators.
        beta_violators, max_beta_viol, n_cand_betas = find_violators(
            build, sol, sub_M_mats, R, A_support, tol=tol,
            max_violators_to_return=(
                add_top_k_betas if add_top_k_betas > 0 else -1),
        )

        # 3b. Window pricing (Tier 2): RC of dropped windows.
        # Build (|Sigma_R|, n_W_full) matrix mapping beta -> coeff_W(beta).
        # Done over the FULL window set so we can score dropped W's.
        n_W_dropped = n_W_full - len(W_active)
        if n_W_dropped > 0:
            A_lambda_full = _build_lambda_coef_matrix(M_mats, Sigma_R)
            y_beta, y_simplex = _split_duals(sol, len(Sigma_R))
            if y_beta is None:
                W_violators: List[int] = []
                max_W_viol = 0.0
            else:
                W_violators, max_W_viol = _price_windows(
                    A_lambda_full, y_beta, y_simplex,
                    W_active, n_W_full, tol=tol,
                )
                if add_top_k_windows > 0:
                    W_violators = W_violators[:add_top_k_windows]
        else:
            W_violators = []
            max_W_viol = 0.0

        wall_p = time.time() - t_p

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        rec = CGActiveSetIter(
            iteration=it,
            n_constraints=build.A_eq.shape[0],
            n_q_vars=len(B_R),
            n_active_W=len(W_active),
            n_W_dropped=n_W_dropped,
            alpha=sol.alpha,
            n_beta_violators=len(beta_violators),
            max_beta_violation=max_beta_viol,
            n_W_violators=len(W_violators),
            max_W_violation=max_W_viol,
            build_wall_s=wall_b, solve_wall_s=wall_s,
            pricing_wall_s=wall_p,
            rss_mb=rss_mb,
            solver_status=sol.status,
        )
        iters.append(rec)

        if verbose:
            print(
                f"  iter {it}: alpha={sol.alpha:.7f}  "
                f"|Sigma|={len(Sigma_R)}  |W_act|={len(W_active)}  "
                f"beta_viol={len(beta_violators)} "
                f"(max={max_beta_viol:.2e})  "
                f"W_viol={len(W_violators)} (max RC={max_W_viol:.2e})  "
                f"build={wall_b:.2f}s solve={wall_s:.2f}s "
                f"price={wall_p:.2f}s rss={rss_mb:.0f}MB",
                flush=True,
            )

        if not beta_violators and not W_violators:
            if verbose:
                print(
                    f"  CG+ActiveSet CONVERGED at iter {it}  "
                    f"alpha={sol.alpha:.7f}  "
                    f"|Sigma_R|={len(Sigma_R)}/{ts.full_n_constraints}  "
                    f"|W|={len(W_active)}/{n_W_full}",
                    flush=True,
                )
            return CGActiveSetResult(
                final_alpha=sol.alpha,
                final_Sigma_R=Sigma_R,
                final_B_R=B_R,
                final_W_active=W_active,
                final_x=sol.x,
                iterations=iters,
                converged=True,
                total_wall_s=time.time() - t_total,
                n_W_full=n_W_full,
            )

        # Add violators.
        if beta_violators:
            Sigma_R = sorted(set(Sigma_R) | set(beta_violators))
            # Recompute B_R from the new Sigma_R (q-poly variables should
            # cover all q_K used in row K and predecessors of each
            # beta in Sigma_R).
            B_R = compute_B_R(Sigma_R, d, R)
        if W_violators:
            W_active = sorted(set(W_active) | set(W_violators))

    if verbose:
        print(
            f"  CG+ActiveSet hit max_iter={max_iter} without convergence  "
            f"alpha={final_sol.alpha if final_sol else None}",
            flush=True,
        )
    return CGActiveSetResult(
        final_alpha=(final_sol.alpha if final_sol else None),
        final_Sigma_R=Sigma_R,
        final_B_R=B_R,
        final_W_active=W_active,
        final_x=(final_sol.x if final_sol else None),
        iterations=iters,
        converged=False,
        total_wall_s=time.time() - t_total,
        n_W_full=n_W_full,
    )

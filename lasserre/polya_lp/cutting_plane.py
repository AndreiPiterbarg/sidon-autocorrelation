"""Column-generation / cutting-plane orchestrator for the Pólya-Handelman LP.

Why CG and not pure Newton-polytope: agent 1's lemma in AUDIT.md fails
for our inhomogeneous LP because the q multiplier introduces extension
constraints c_beta = q_beta - sum q_{beta-e_j} that need not be >= 0 by
construction. Soundness must be enforced empirically: solve the
restricted LP, check the FULL set for violations, add violators, repeat.

Algorithm:
  1. Build initial Sigma_R^{(0)} from term_sparsity.build_term_sparsity_support.
  2. Solve the restricted LP -> (alpha*, lambda*, q*_K for K in B_R).
  3. Extend q to all K (q_K := 0 for K not in B_R), and compute
       c_beta = sum_W lambda*_W coeff_W(beta) - alpha* * delta_{beta=0}
                + q_beta - sum_j q_{beta-e_j}
     for every beta with |beta| <= R, beta not in Sigma_R.
     The set of "candidate violator" betas E is bounded by:
       E = B_R cup (B_R + e_j for j) cup A cup {0}
     all other beta have c_beta = 0 trivially.
  4. Violators V = {beta in E\Sigma_R : c_beta < -tol}. If empty, RETURN.
  5. Sigma_R^{(k+1)} = Sigma_R^{(k)} cup V; recompute B_R; goto 2.

At convergence, the restricted-LP optimum equals the full-LP optimum
(by construction: every full-LP constraint is satisfied).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Set
import time
import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp, _coeff_W_vector,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.term_sparsity import (
    TermSparsitySupport, build_term_sparsity_support, compute_B_R,
    polynomial_support_from_M_mats,
)
from lasserre.polya_lp.poly import index_map


@dataclass
class CGIterRecord:
    iteration: int
    n_constraints: int
    n_q_vars: int
    n_lambda_vars: int
    alpha: Optional[float]
    n_violators: int
    max_violation: float
    build_wall_s: float
    solve_wall_s: float
    check_wall_s: float
    solver_status: str = ""


@dataclass
class CGResult:
    final_alpha: Optional[float]
    final_support: TermSparsitySupport
    final_x: Optional[np.ndarray]
    iterations: List[CGIterRecord]
    converged: bool
    total_wall_s: float


# ---------------------------------------------------------------------
# Violator enumeration
# ---------------------------------------------------------------------

def _candidate_violator_set(
    Sigma_R: Sequence[Tuple[int, ...]],
    B_R: Sequence[Tuple[int, ...]],
    A: Sequence[Tuple[int, ...]],
    d: int,
    R: int,
) -> List[Tuple[int, ...]]:
    """E = B_R cup (B_R + e_j) cup A cup {0}, all with |.| <= R, MINUS Sigma_R.

    Any beta not in E has c_beta = 0 in the full-LP extension, hence
    cannot be a violator. So we only need to check betas in E.
    """
    Sigma_set = set(Sigma_R)
    E: Set[Tuple[int, ...]] = set()
    zero = tuple([0] * d)
    if zero not in Sigma_set:
        E.add(zero)
    for K in B_R:
        if sum(K) <= R and K not in Sigma_set:
            E.add(K)
        for j in range(d):
            shift = list(K)
            shift[j] += 1
            shift_t = tuple(shift)
            if sum(shift_t) <= R and shift_t not in Sigma_set:
                E.add(shift_t)
    for a in A:
        if sum(a) <= R and a not in Sigma_set:
            E.add(a)
    return sorted(E)


def find_violators(
    build: BuildResult,
    sol: SolveResult,
    M_mats: Sequence[np.ndarray],
    R: int,
    A: Sequence[Tuple[int, ...]],
    tol: float = 1e-8,
    max_violators_to_return: int = -1,
) -> Tuple[List[Tuple[int, ...]], float, int]:
    """Compute c_beta for every candidate beta not in Sigma_R; return violators.

    Vectorized implementation: stacks candidate betas into a single (N, d)
    int8 array and computes c_beta for all candidates with batched numpy
    ops. Lambda contribution uses a pre-aggregated (d, d) diag/cross table
    (sum_W lam_W M_W); q contributions use np.void byte-views for batched
    dict lookups (same trick as build._batch_lookup). Targets ~50-100x
    speedup vs. per-tuple Python loops at N=100K.

    Returns (violators, max_violation_magnitude, n_candidates_checked).
    Sorted by most-violating first.
    """
    if sol.x is None or sol.alpha is None:
        return [], 0.0, 0

    d = len(build.monos_le_R[0])
    alpha = float(sol.alpha)

    # Extract LP solution
    lam = (sol.x[build.lambda_idx] if build.fixed_lambda is None
           else build.fixed_lambda)
    lam = np.asarray(lam, dtype=np.float64)
    q_vec = (sol.x[build.q_idx]
             if build.options.use_q_polynomial else np.zeros(0))
    q_vec = np.asarray(q_vec, dtype=np.float64)

    # ---------------------------------------------------------------
    # Build E (candidate set); stack as (N, d) int8 array.
    # ---------------------------------------------------------------
    E = _candidate_violator_set(
        build.monos_le_R, build.monos_le_Rm1, A, d, R)
    n_candidates = len(E)
    if n_candidates == 0:
        return [], 0.0, 0
    beta_arr = np.asarray(E, dtype=np.int8)  # (N, d)
    N = beta_arr.shape[0]

    # ---------------------------------------------------------------
    # Pre-aggregate lambda * coeff_W into a single (d, d) table:
    #   M_agg[i, j] = sum_W lam_W * M_W[i, j]
    # Then for |beta|=2:
    #   beta = 2*e_i      -> coeff = M_agg[i, i]
    #   beta = e_i + e_j  -> coeff = 2 * M_agg[i, j]   (i < j; M_agg sym)
    # All other betas pick up zero lambda mass (coeff_W is purely deg-2).
    # ---------------------------------------------------------------
    if len(M_mats) > 0:
        M_stack = np.stack(
            [np.asarray(M, dtype=np.float64) for M in M_mats], axis=0)
        M_agg = np.tensordot(lam, M_stack, axes=(0, 0))  # (d, d)
        M_diag = np.diagonal(M_agg).copy()               # (d,)
    else:
        M_agg = np.zeros((d, d), dtype=np.float64)
        M_diag = np.zeros(d, dtype=np.float64)

    # ---------------------------------------------------------------
    # Classify each candidate beta by support pattern.
    # ---------------------------------------------------------------
    deg = beta_arr.sum(axis=1, dtype=np.int64)         # (N,)
    nnz = (beta_arr != 0).sum(axis=1, dtype=np.int64)  # (N,)

    c = np.zeros(N, dtype=np.float64)

    # alpha: -alpha at beta=0
    zero_mask = (deg == 0)
    if zero_mask.any():
        c[zero_mask] -= alpha

    # Lambda diag: 2*e_i  ->  M_agg[i, i]
    diag_mask = (deg == 2) & (nnz == 1)
    if diag_mask.any():
        # single nonzero index per row
        i_of_diag = beta_arr[diag_mask].argmax(axis=1)
        c[diag_mask] += M_diag[i_of_diag]

    # Lambda cross: e_i + e_j  ->  2 * M_agg[i, j]
    cross_mask = (deg == 2) & (nnz == 2)
    if cross_mask.any():
        rows = beta_arr[cross_mask]                    # (Nc, d)
        # extract the two distinct nonzero positions (sorted ascending)
        nz_pos = np.argsort(-rows, axis=1, kind='stable')[:, :2]
        nz_pos.sort(axis=1)
        i_idx, j_idx = nz_pos[:, 0], nz_pos[:, 1]
        c[cross_mask] += 2.0 * M_agg[i_idx, j_idx]

    # ---------------------------------------------------------------
    # q lookup: bytes-keyed dict over B_R, values from q_vec.
    # Batched lookup via np.void byte-view (matches build._batch_lookup).
    # ---------------------------------------------------------------
    void_dt = np.dtype((np.void, d * np.dtype(np.int8).itemsize))

    if q_vec.size > 0 and len(build.monos_le_Rm1) > 0:
        B_arr = np.ascontiguousarray(
            np.asarray(build.monos_le_Rm1, dtype=np.int8))
        B_void = B_arr.view(void_dt).ravel()
        q_lookup = {bytes(v): float(q_vec[i])
                    for i, v in enumerate(B_void.tolist())}
    else:
        q_lookup = {}

    if q_lookup:
        # Build ONE big batch of all queries:
        #   row 0..N-1     : beta itself (sign +1)
        #   row N..2N-1    : beta - e_0 (sign -1, only where beta[0] > 0)
        #   row 2N..3N-1   : beta - e_1 (sign -1, only where beta[1] > 0)
        #   ...etc
        # We mark "invalid" rows (beta[j] == 0) by leaving them at the
        # original beta — the dict lookup may hit, but we mask them out
        # before applying. To make masking cheap, we set those rows to a
        # sentinel (the negated beta) so the bytes are guaranteed not to
        # collide with any real key. Actually simpler: build only the
        # valid rows and re-scatter via index arrays.
        sign_chunks: List[np.ndarray] = []
        idx_chunks: List[np.ndarray] = []
        query_chunks: List[np.ndarray] = []

        # +q_beta over all candidates
        idx_chunks.append(np.arange(N, dtype=np.int64))
        sign_chunks.append(np.ones(N, dtype=np.float64))
        query_chunks.append(beta_arr)

        # -q_{beta - e_j} for each j, only where beta[:, j] > 0
        for j in range(d):
            valid_idx = np.flatnonzero(beta_arr[:, j] > 0)
            if valid_idx.size == 0:
                continue
            sub = beta_arr[valid_idx].copy()
            sub[:, j] -= 1
            idx_chunks.append(valid_idx)
            sign_chunks.append(-np.ones(valid_idx.size, dtype=np.float64))
            query_chunks.append(sub)

        all_queries = np.ascontiguousarray(
            np.concatenate(query_chunks, axis=0))
        all_idx = np.concatenate(idx_chunks)
        all_sign = np.concatenate(sign_chunks)
        all_void = all_queries.view(void_dt).ravel()
        # Single fused dict-lookup loop (instead of d+1 separate ones).
        all_vals = np.fromiter(
            (q_lookup.get(bytes(v), 0.0) for v in all_void.tolist()),
            dtype=np.float64, count=all_void.size,
        )
        all_vals *= all_sign
        # Scatter-add by candidate index.
        np.add.at(c, all_idx, all_vals)

    # ---------------------------------------------------------------
    # Identify violators and sort by most-negative first.
    # ---------------------------------------------------------------
    max_violation = float(c.min()) if N else 0.0
    max_viol_mag = abs(max_violation) if max_violation < 0 else 0.0

    viol_mask = c < -tol
    if not viol_mask.any():
        return [], max_viol_mag, n_candidates

    viol_betas = beta_arr[viol_mask]
    viol_vals = c[viol_mask]
    order = np.argsort(viol_vals)               # most-negative first
    viol_betas = viol_betas[order]
    if max_violators_to_return > 0:
        viol_betas = viol_betas[:max_violators_to_return]

    out = [tuple(int(x) for x in row) for row in viol_betas]
    return out, max_viol_mag, n_candidates


# ---------------------------------------------------------------------
# Main CG orchestrator
# ---------------------------------------------------------------------

def solve_with_cg(
    d: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    max_iter: int = 20,
    tol: float = 1e-8,
    add_top_k: int = -1,
    include_low_degree_seed: bool = True,
    solver_method: str = "auto",
    verbose: bool = True,
) -> CGResult:
    """Orchestrate column generation for the Handelman LP."""
    t_total = time.time()
    iters: List[CGIterRecord] = []

    # Initial seed
    ts = build_term_sparsity_support(M_mats, R, include_low_degree=include_low_degree_seed)
    A = ts.A

    if verbose:
        print(f"  CG seed: |Sigma_R|={ts.n_constraints}/{ts.full_n_constraints}, "
              f"|B_R|={ts.n_q_vars}/{ts.full_n_q_vars}, |A|={len(A)}", flush=True)

    final_sol: Optional[SolveResult] = None
    final_build: Optional[BuildResult] = None

    for it in range(max_iter):
        # Build restricted LP
        opts = BuildOptions(
            R=R,
            use_z2=True,
            use_q_polynomial=True,
            verbose=False,
            restricted_Sigma_R=ts.Sigma_R,
            restricted_B_R=ts.B_R,
        )
        t_b = time.time()
        build = build_handelman_lp(d, M_mats, opts)
        wall_b = time.time() - t_b

        # Solve
        t_s = time.time()
        sol = solve_lp(build, method=solver_method, verbose=False)
        wall_s = time.time() - t_s
        final_sol = sol
        final_build = build

        if sol.alpha is None:
            iters.append(CGIterRecord(
                iteration=it, n_constraints=build.A_eq.shape[0],
                n_q_vars=ts.n_q_vars, n_lambda_vars=len(M_mats),
                alpha=None, n_violators=-1, max_violation=float('inf'),
                build_wall_s=wall_b, solve_wall_s=wall_s, check_wall_s=0.0,
                solver_status=sol.status,
            ))
            if verbose:
                print(f"  iter {it}: SOLVE FAILED status={sol.status}; "
                      f"build={wall_b:.2f}s solve={wall_s:.2f}s", flush=True)
            return CGResult(
                final_alpha=None, final_support=ts, final_x=None,
                iterations=iters, converged=False,
                total_wall_s=time.time() - t_total,
            )

        # Check for violators
        t_c = time.time()
        violators, max_viol, n_cand = find_violators(
            build, sol, M_mats, R, A, tol=tol,
            max_violators_to_return=(add_top_k if add_top_k > 0 else -1))
        wall_c = time.time() - t_c

        rec = CGIterRecord(
            iteration=it, n_constraints=build.A_eq.shape[0],
            n_q_vars=ts.n_q_vars, n_lambda_vars=len(M_mats),
            alpha=sol.alpha, n_violators=len(violators),
            max_violation=max_viol,
            build_wall_s=wall_b, solve_wall_s=wall_s, check_wall_s=wall_c,
            solver_status=sol.status,
        )
        iters.append(rec)

        if verbose:
            print(f"  iter {it}: alpha={sol.alpha:.6f} "
                  f"|Sigma|={build.A_eq.shape[0]} |B|={ts.n_q_vars} "
                  f"violators={len(violators)}/{n_cand} "
                  f"max_viol={max_viol:.2e}  "
                  f"build={wall_b:.2f}s solve={wall_s:.2f}s "
                  f"check={wall_c:.2f}s", flush=True)

        if not violators:
            if verbose:
                print(f"  CG converged at iter {it}, alpha={sol.alpha:.6f}", flush=True)
            return CGResult(
                final_alpha=sol.alpha, final_support=ts, final_x=sol.x,
                iterations=iters, converged=True,
                total_wall_s=time.time() - t_total,
            )

        # Add violators to Sigma_R
        new_Sigma = sorted(set(ts.Sigma_R) | set(violators))
        new_B = compute_B_R(new_Sigma, d, R)
        ts = TermSparsitySupport(
            Sigma_R=new_Sigma,
            B_R=new_B,
            A=A,
            n_var_dim=d,
            R=R,
            seed_iter=ts.seed_iter + 1,
            n_constraints=len(new_Sigma),
            n_q_vars=len(new_B),
            full_n_constraints=ts.full_n_constraints,
            full_n_q_vars=ts.full_n_q_vars,
            notes=f"CG iter {it+1}",
        )

    if verbose:
        print(f"  CG max_iter reached without convergence", flush=True)
    return CGResult(
        final_alpha=(final_sol.alpha if final_sol else None),
        final_support=ts,
        final_x=(final_sol.x if final_sol else None),
        iterations=iters,
        converged=False,
        total_wall_s=time.time() - t_total,
    )

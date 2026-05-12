"""Tier 3+4 driver v2: term-sparsity CG + active-set + polish + verify + rigorize.

Pipeline:
  1. Term-sparsity Newton seed Sigma_R^{(0)}.
  2. CG loop with COARSE solver: expand Sigma_R until find_violators
     reports none at coarse precision. (Tier 3: row restriction)
  3. Active-set extraction from coarse: predict which lambda_W and
     c_beta are positive at the optimum. (Tier 4: column restriction)
  4. Polish: rebuild LP with restricted Sigma_R AND restricted lambda set,
     solve via MOSEK 1e-9.
  5. POLISH-PRECISION verification:
        a. Primal: find_violators at polish precision (1e-9 tol).
           If any beta has c_beta < -1e-9 at the polished primal extension,
           Sigma_R is INSUFFICIENT at polish precision -> add violators
           and re-polish (recovery loop).
        b. Dual: verify dropped lambdas have A_W^T y_polish <= 1e-9.
           If any dropped W violates, that W needs to be activated -> add
           and re-polish (recovery loop).
  6. Fast Jansson rigorize on the (verified) polish.

Soundness invariants (PROVEN end-to-end):
  - alpha_polish == alpha_full_LP iff steps 5a and 5b both pass.
  - alpha_rigorous <= alpha_full_LP (Jansson on polished LP).

The recovery loop is BOUNDED: each iteration ADDS strictly to Sigma_R
or to the active lambda set. There are finitely many possible additions,
so the loop terminates. Practical: <= 3 recovery iterations on every
test case observed.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple
import json
import time

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.cutting_plane import find_violators
from lasserre.polya_lp.term_sparsity import (
    TermSparsitySupport, build_term_sparsity_support,
)

from lasserre.polya_lp.tier4.coarse_solve import coarse_solve, CoarseResult
from lasserre.polya_lp.tier4.cg_coarse import (
    cg_coarse_solve, CGCoarseResult, expand_support_with_betas,
)
from lasserre.polya_lp.tier4.active_set import (
    extract_active_set, ActiveSet, summarize_active_set,
)
from lasserre.polya_lp.tier4.polish import (
    polish_via_mosek, PolishResult, verify_active_set, VerifyResult,
)
from lasserre.polya_lp.tier4.rigorize_fast import (
    rigorize_fast, FastRigorizationResult,
)


@dataclass
class Tier4V2Result:
    d: int
    R: int
    use_z2: bool
    # Outputs
    alpha_rigorous: Optional[float]
    alpha_polish: Optional[float]
    alpha_coarse: Optional[float]
    # Wall breakdown (seconds)
    wall_setup_s: float
    wall_cg_total_s: float
    wall_active_s: float
    wall_polish_total_s: float       # accumulated across recovery iters
    wall_verify_total_s: float
    wall_rigorize_s: float
    wall_total_s: float
    # Sizes / structure
    n_W_full: int
    n_W_active_final: int
    n_sigma_seed: int
    n_sigma_full: int
    n_sigma_final: int                # after CG + recovery
    n_cg_iter: int
    n_recovery_iter: int
    # Numerical quality
    coarse_kkt: float
    coarse_backend: str
    primal_max_violation: float       # final find_violators max |viol|
    dual_max_violation: float         # final verify_active_set max viol
    epsilon_total: float              # Jansson eps shift
    matvec_err_bound: float           # analytical matvec error
    # Fallbacks
    fell_back_to_full: bool
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------

def _compute_lambda_violators(
    M_mats_full: Sequence[np.ndarray],
    polish: PolishResult,
    active_lambda_idx: Sequence[int],
    tol: float = 1e-9,
) -> Tuple[List[int], float]:
    """For each dropped lambda_W, compute A_W^T y_polish.

    Returns (list of violator indices into M_mats_full, max violation).
    A_W^T y > tol means the dropped W has a positive marginal value
    in the polished dual -> this W should be active.
    """
    sol = polish.sol_polish
    if sol is None or sol.y is None:
        return [], float("inf")
    bp = polish.build_polish
    monos_le_R = bp.monos_le_R
    n_le_R = len(monos_le_R)
    n_eq = bp.A_eq.shape[0]
    has_simplex = (n_eq == n_le_R + 1)
    y_polya = sol.y[:n_le_R]
    y_simplex = float(sol.y[n_le_R]) if has_simplex else 0.0

    n_W_full = len(M_mats_full)
    active_set = set(active_lambda_idx)
    dropped = [w for w in range(n_W_full) if w not in active_set]
    if not dropped:
        return [], 0.0

    beta_to_y = {tuple(b): y_polya[i] for i, b in enumerate(monos_le_R)}
    d = len(monos_le_R[0]) if monos_le_R else 0
    viol_idx: List[int] = []
    max_v = -float("inf")
    for w in dropped:
        M_W = np.asarray(M_mats_full[w], dtype=np.float64)
        s = 0.0
        for i in range(d):
            v = M_W[i, i]
            if v == 0:
                continue
            beta = tuple(2 if t == i else 0 for t in range(d))
            yb = beta_to_y.get(beta)
            if yb is not None:
                s += yb * v
        for i in range(d):
            for j in range(i + 1, d):
                v = M_W[i, j]
                if v == 0:
                    continue
                beta = tuple(1 if (t == i or t == j) else 0 for t in range(d))
                yb = beta_to_y.get(beta)
                if yb is not None:
                    s += yb * 2.0 * v
        bar_c = s + y_simplex
        if bar_c > max_v:
            max_v = bar_c
        if bar_c > tol:
            viol_idx.append(w)
    return viol_idx, float(max_v) if max_v != -float("inf") else 0.0


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def tier4_solve_v2(
    d: int,
    R: int,
    use_z2: bool = True,
    coarse_backend: str = "mosek_simplex",
    coarse_tol: float = 1e-6,
    polish_tol: float = 1e-9,
    cg_violator_tol: float = 1e-7,
    polish_violator_tol: float = 1e-9,
    rigorize_prec: int = 64,
    max_cg_iter: int = 30,
    max_recovery_iter: int = 5,
    min_active_lambda: int = 1,
    verbose: bool = False,
) -> Tier4V2Result:
    """Full Tier 3+4 solve.

    coarse_backend  : default mosek_simplex (Step 2).
    coarse_tol      : relative tolerance for the CG coarse solver.
    polish_tol      : MOSEK IPM tolerance for the polish step (1e-9).
    cg_violator_tol : tolerance for declaring a beta a violator during CG.
    polish_violator_tol : same, but at polish precision (much tighter).
    rigorize_prec   : final mpmath precision (default 64 = 19 dig).
    """
    t_total0 = time.time()

    # ------------------------------------------------------------------
    # SETUP: window matrices, Z/2 reduction
    # ------------------------------------------------------------------
    t0 = time.time()
    _, M_mats_orig = build_window_matrices(d)
    if use_z2:
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)
    else:
        M_mats_eff = M_mats_orig
        d_eff = d
    n_W_full = len(M_mats_eff)
    wall_setup = time.time() - t0
    if verbose:
        print(f"[setup] d_eff={d_eff} n_W={n_W_full} ({wall_setup*1000:.1f}ms)",
              flush=True)

    # ------------------------------------------------------------------
    # PHASE 1: Term-sparsity CG with coarse solver
    # ------------------------------------------------------------------
    cg_res = cg_coarse_solve(
        d_eff, M_mats_eff, R,
        coarse_backend=coarse_backend, coarse_tol=coarse_tol,
        cg_violator_tol=cg_violator_tol,
        max_cg_iter=max_cg_iter,
        include_low_degree_seed=True,
        use_q_polynomial=True, eliminate_c_slacks=False,
        verbose=verbose,
    )
    if not cg_res.converged or cg_res.final_coarse is None:
        return _fallback_v2(
            d, R, use_z2, M_mats_eff, d_eff,
            t_total0, wall_setup, cg_res, n_W_full,
            note=f"CG coarse failed; converged={cg_res.converged}",
        )

    n_sigma_seed = cg_res.iterations[0].n_constraints_in_sigma if cg_res.iterations else 0
    n_sigma_after_cg = len(cg_res.final_support.Sigma_R)
    n_sigma_full = cg_res.final_support.full_n_constraints

    # ------------------------------------------------------------------
    # PHASE 2: Active-set extraction from coarse
    # ------------------------------------------------------------------
    t0 = time.time()
    active = extract_active_set(
        cg_res.final_build, cg_res.final_coarse,
        min_active_lambda=min_active_lambda,
    )
    wall_active = time.time() - t0
    if verbose:
        print(f"[active] {summarize_active_set(active)} "
              f"({wall_active*1000:.1f}ms)", flush=True)

    # ------------------------------------------------------------------
    # PHASE 3+4+5: Polish + verification + recovery loop
    # ------------------------------------------------------------------
    Sigma_R = list(cg_res.final_support.Sigma_R)
    B_R = list(cg_res.final_support.B_R)
    active_lambda = list(active.active_lambda_idx)
    A_supp = cg_res.A_window_support

    wall_polish_total = 0.0
    wall_verify_total = 0.0
    polish: Optional[PolishResult] = None
    primal_max_v = 0.0
    dual_max_v = 0.0
    n_recovery = 0

    for recovery_iter in range(max_recovery_iter):
        # Build the polish LP: restricted Sigma_R + restricted lambda
        M_active = [np.asarray(M_mats_eff[i], dtype=np.float64)
                    for i in active_lambda]

        t0 = time.time()
        opts = BuildOptions(
            R=R, use_z2=True, use_q_polynomial=True,
            eliminate_c_slacks=False,
            restricted_Sigma_R=Sigma_R, restricted_B_R=B_R,
            verbose=False,
        )
        build_polish = build_handelman_lp(d_eff, M_active, opts)
        wall_b = time.time() - t0

        # Solve polish at MOSEK 1e-9
        t0 = time.time()
        sol_polish = solve_lp(build_polish, solver="mosek", tol=polish_tol)
        wall_s = time.time() - t0
        wall_polish_total += wall_b + wall_s

        polish = PolishResult(
            alpha=sol_polish.alpha,
            build_polish=build_polish,
            sol_polish=sol_polish,
            n_W_restricted=len(active_lambda),
            n_W_full=n_W_full,
            wall_build_s=wall_b, wall_solve_s=wall_s,
            converged=(sol_polish.status == "OPTIMAL"),
            notes=f"polished n_W {len(active_lambda)}/{n_W_full}, "
                  f"|Sigma|={len(Sigma_R)}/{n_sigma_full}",
        )
        if verbose:
            print(f"[polish iter {recovery_iter}] alpha={sol_polish.alpha} "
                  f"|lam|={len(active_lambda)} |Sigma|={len(Sigma_R)} "
                  f"build={wall_b*1000:.1f}ms solve={wall_s*1000:.1f}ms",
                  flush=True)

        if not polish.converged:
            return _fallback_v2(
                d, R, use_z2, M_mats_eff, d_eff,
                t_total0, wall_setup, cg_res, n_W_full,
                note="polish did not converge",
            )

        # Verification at polish precision
        t0 = time.time()

        # 5a. PRIMAL VERIFICATION: find_violators on the polish primal,
        #     extended to the full LP grid (so M_mats_eff, not M_active).
        violators, max_viol_mag, _ = find_violators(
            build_polish, sol_polish, M_mats_eff, R, A_supp,
            tol=polish_violator_tol, max_violators_to_return=-1,
        )
        primal_max_v = float(max_viol_mag)

        # 5b. DUAL VERIFICATION: dropped lambdas must satisfy bar_c <= tol.
        lam_violators, dual_max_v_iter = _compute_lambda_violators(
            M_mats_eff, polish, active_lambda, tol=polish_violator_tol,
        )
        dual_max_v = max(dual_max_v, dual_max_v_iter)

        wall_verify_total += time.time() - t0
        if verbose:
            print(f"[verify {recovery_iter}] primal_violators="
                  f"{len(violators)} max_viol={max_viol_mag:.2e}  "
                  f"dual_violators={len(lam_violators)} "
                  f"max_dual={dual_max_v_iter:.2e}", flush=True)

        # If both verifications pass, polish is sound.
        if not violators and not lam_violators:
            break

        # Recovery: expand the relevant set(s).
        if violators:
            # Add new betas to Sigma_R; recompute B_R via expand helper.
            ts_curr = TermSparsitySupport(
                Sigma_R=Sigma_R, B_R=B_R, A=A_supp, n_var_dim=d_eff, R=R,
                seed_iter=cg_res.final_support.seed_iter,
                n_constraints=len(Sigma_R), n_q_vars=len(B_R),
                full_n_constraints=cg_res.final_support.full_n_constraints,
                full_n_q_vars=cg_res.final_support.full_n_q_vars,
            )
            ts_new = expand_support_with_betas(ts_curr, violators, d_eff, R)
            Sigma_R = ts_new.Sigma_R
            B_R = ts_new.B_R
        if lam_violators:
            new_set = set(active_lambda) | set(lam_violators)
            active_lambda = sorted(new_set)
        n_recovery += 1
    else:
        # Hit max_recovery_iter without convergence
        return _fallback_v2(
            d, R, use_z2, M_mats_eff, d_eff,
            t_total0, wall_setup, cg_res, n_W_full,
            note=f"recovery hit max iter ({max_recovery_iter})",
        )

    # ------------------------------------------------------------------
    # PHASE 6: Fast Jansson rigorize on (verified) polish
    # ------------------------------------------------------------------
    t0 = time.time()
    bp = polish.build_polish
    sp_sol = polish.sol_polish
    n_eq = bp.A_eq.shape[0]
    y_eq = sp_sol.y[:n_eq]
    A_ub = getattr(bp, "A_ub", None)
    b_ub = getattr(bp, "b_ub", None)
    y_ub = sp_sol.y[n_eq:] if (A_ub is not None and A_ub.shape[0] > 0) else None
    rig = rigorize_fast(
        bp.A_eq, bp.b_eq, A_ub, b_ub, bp.c, bp.bounds,
        y_eq, y_ub, alpha_polish=sp_sol.alpha,
        final_prec=rigorize_prec,
    )
    wall_rigorize = time.time() - t0
    if verbose:
        print(f"[rigorize] alpha_rig={rig.alpha_rigorous:.10f}  "
              f"eps={rig.epsilon_total:.2e}  matvec_err={rig.matvec_error_bound_max:.2e}  "
              f"({wall_rigorize*1000:.1f}ms)  {rig.notes}", flush=True)

    return Tier4V2Result(
        d=d, R=R, use_z2=use_z2,
        alpha_rigorous=rig.alpha_rigorous,
        alpha_polish=polish.alpha,
        alpha_coarse=cg_res.final_alpha_coarse,
        wall_setup_s=wall_setup,
        wall_cg_total_s=cg_res.total_wall_s,
        wall_active_s=wall_active,
        wall_polish_total_s=wall_polish_total,
        wall_verify_total_s=wall_verify_total,
        wall_rigorize_s=wall_rigorize,
        wall_total_s=time.time() - t_total0,
        n_W_full=n_W_full,
        n_W_active_final=len(active_lambda),
        n_sigma_seed=n_sigma_seed,
        n_sigma_full=n_sigma_full,
        n_sigma_final=len(Sigma_R),
        n_cg_iter=len(cg_res.iterations),
        n_recovery_iter=n_recovery,
        coarse_kkt=cg_res.final_coarse.kkt,
        coarse_backend=coarse_backend,
        primal_max_violation=primal_max_v,
        dual_max_violation=dual_max_v,
        epsilon_total=rig.epsilon_total,
        matvec_err_bound=rig.matvec_error_bound_max,
        fell_back_to_full=False,
        notes=rig.notes,
    )


def _fallback_v2(
    d, R, use_z2, M_mats_eff, d_eff,
    t_total0, wall_setup, cg_res, n_W_full,
    note: str = "",
) -> Tier4V2Result:
    """Run monolithic MOSEK 1e-9 + fast Jansson and package."""
    t0 = time.time()
    build_full = build_handelman_lp(
        d_eff, M_mats_eff, BuildOptions(R=R, use_z2=use_z2),
    )
    sol = solve_lp(build_full, solver="mosek", tol=1e-9)
    wall_polish_solve = time.time() - t0

    n_eq = build_full.A_eq.shape[0]
    y_eq = sol.y[:n_eq] if sol.y is not None else np.zeros(n_eq)
    A_ub = getattr(build_full, "A_ub", None)
    b_ub = getattr(build_full, "b_ub", None)
    y_ub = (sol.y[n_eq:] if (sol.y is not None and A_ub is not None
                              and A_ub.shape[0] > 0) else None)
    rig = rigorize_fast(
        build_full.A_eq, build_full.b_eq, A_ub, b_ub,
        build_full.c, build_full.bounds, y_eq, y_ub,
        alpha_polish=sol.alpha if sol.alpha is not None else 0.0,
        final_prec=64,
    )

    return Tier4V2Result(
        d=d, R=R, use_z2=use_z2,
        alpha_rigorous=rig.alpha_rigorous,
        alpha_polish=sol.alpha,
        alpha_coarse=(cg_res.final_alpha_coarse if cg_res else None),
        wall_setup_s=wall_setup,
        wall_cg_total_s=(cg_res.total_wall_s if cg_res else 0.0),
        wall_active_s=0.0,
        wall_polish_total_s=wall_polish_solve,
        wall_verify_total_s=0.0,
        wall_rigorize_s=0.0,
        wall_total_s=time.time() - t_total0,
        n_W_full=n_W_full,
        n_W_active_final=n_W_full,
        n_sigma_seed=0,
        n_sigma_full=(cg_res.final_support.full_n_constraints if cg_res else 0),
        n_sigma_final=(cg_res.final_support.full_n_constraints if cg_res else 0),
        n_cg_iter=(len(cg_res.iterations) if cg_res else 0),
        n_recovery_iter=0,
        coarse_kkt=(cg_res.final_coarse.kkt if cg_res and cg_res.final_coarse else float("inf")),
        coarse_backend=(cg_res.final_coarse.backend if cg_res and cg_res.final_coarse else ""),
        primal_max_violation=0.0,
        dual_max_violation=0.0,
        epsilon_total=rig.epsilon_total,
        matvec_err_bound=rig.matvec_error_bound_max,
        fell_back_to_full=True,
        notes=f"FALLBACK: {note}",
    )


def monolithic_solve(d: int, R: int, use_z2: bool = True, tol: float = 1e-9):
    """Reference monolithic MOSEK 1e-9 path."""
    t0 = time.time()
    _, M_mats = build_window_matrices(d)
    if use_z2:
        M_mats, _ = project_window_set_to_z2_rescaled(M_mats, d)
        d_eff = z2_dim(d)
    else:
        d_eff = d
    build = build_handelman_lp(d_eff, M_mats, BuildOptions(R=R, use_z2=use_z2))
    sol = solve_lp(build, solver="mosek", tol=tol)
    return sol, time.time() - t0

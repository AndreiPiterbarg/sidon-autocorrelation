"""End-to-end Tier-4 driver:

  coarse LP solve  ->  active-set extraction  ->  MOSEK polish on
  reduced LP  ->  Jansson rigorous LB.

Entry point:
    tier4_solve(d, R, **kwargs) -> Tier4Result

The standard "monolithic MOSEK" path is also exposed for benchmarking:
    monolithic_solve(d, R) -> SolveResult.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import time

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)

from lasserre.polya_lp.tier4.coarse_solve import coarse_solve, CoarseResult
from lasserre.polya_lp.tier4.active_set import (
    extract_active_set, summarize_active_set, ActiveSet,
)
from lasserre.polya_lp.tier4.polish import (
    polish_via_mosek, verify_active_set, PolishResult, VerifyResult,
)
from lasserre.polya_lp.tier4.rigorize import (
    rigorize_from_polish, RigorizationResult,
)


@dataclass
class Tier4Result:
    d: int
    R: int
    use_z2: bool
    # Outputs
    alpha_rigorous: Optional[float]
    alpha_polish: Optional[float]
    alpha_coarse: Optional[float]
    # Wall-time breakdown (seconds)
    wall_setup_s: float
    wall_coarse_s: float
    wall_active_s: float
    wall_polish_build_s: float
    wall_polish_solve_s: float
    wall_verify_s: float
    wall_rigorize_s: float
    wall_total_s: float
    # Active set
    n_W_full: int
    n_W_active: int
    n_cbeta_full: int
    n_cbeta_active: int
    tol_active: float
    # Verification
    verify_max_violation: float
    verify_n_violators: int
    fell_back_to_full: bool
    # Coarse / polish quality
    coarse_kkt: float
    coarse_backend: str
    epsilon_shift: float
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def tier4_solve(
    d: int,
    R: int,
    use_z2: bool = True,
    coarse_backend: str = "highs_ipm",
    coarse_tol: float = 1e-5,
    polish_tol: float = 1e-9,
    rigorize_prec: int = 200,
    verify_tol: float = 1e-7,
    verbose: bool = False,
    auto_fallback_on_violation: bool = False,
) -> Tier4Result:
    """Run Tier-4 on a (d, R) instance.

    coarse_backend  : 'highs_ipm' (default, working) or 'pdhg_gpu' (placeholder).
    coarse_tol      : KKT tolerance for the coarse solve. 1e-5 reliably
                      identifies the active set on Polya-LPs we've tested.
    polish_tol      : MOSEK IPM tolerance for the reduced LP (default 1e-9).
    rigorize_prec   : mpmath precision for Jansson directed-down rounding.
    verify_tol      : reduced-cost violation tolerance for the dropped windows.
                      If exceeded, the active-set was wrong and we either
                      fall back to monolithic (auto_fallback_on_violation=True)
                      or report the violation in the result.
    """
    t_total0 = time.time()

    # --- Setup: build window matrices, optional Z/2 ---------------------
    t0 = time.time()
    _, M_mats_orig = build_window_matrices(d)
    if use_z2:
        M_mats_eff, _counts = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)
    else:
        M_mats_eff = M_mats_orig
        d_eff = d
    n_W_full = len(M_mats_eff)

    # Build the FULL LP for the coarse solve
    build_full = build_handelman_lp(
        d_eff, M_mats_eff, BuildOptions(R=R, use_z2=use_z2),
    )
    wall_setup = time.time() - t0
    if verbose:
        print(f"[setup] d_eff={d_eff} n_W={n_W_full} n_eq={build_full.A_eq.shape[0]} "
              f"n_vars={build_full.n_vars} nnz={build_full.A_eq.nnz} "
              f"({wall_setup*1000:.1f}ms)", flush=True)

    # --- Coarse LP solve ------------------------------------------------
    t0 = time.time()
    coarse: CoarseResult = coarse_solve(
        build_full, tol=coarse_tol, backend=coarse_backend, verbose=verbose,
    )
    wall_coarse = time.time() - t0
    if verbose:
        print(f"[coarse] backend={coarse.backend} alpha={coarse.alpha} "
              f"kkt={coarse.kkt:.2e} ({wall_coarse*1000:.1f}ms)", flush=True)
    if not coarse.converged:
        # Fall back to monolithic
        return _fallback_monolithic(
            d, R, use_z2, build_full, M_mats_eff, d_eff,
            t_total0, wall_setup, wall_coarse, coarse,
            note="coarse did not converge; fell back",
        )

    # --- Active-set extraction ------------------------------------------
    t0 = time.time()
    active: ActiveSet = extract_active_set(build_full, coarse)
    wall_active = time.time() - t0
    if verbose:
        print(f"[active] {summarize_active_set(active)} "
              f"({wall_active*1000:.1f}ms)", flush=True)

    # --- MOSEK polish on restricted LP ----------------------------------
    polish: PolishResult = polish_via_mosek(
        M_mats_eff, d_eff, R, active,
        use_q_polynomial=True, eliminate_c_slacks=False,
        tol=polish_tol, verbose=verbose,
    )
    if verbose:
        print(f"[polish] alpha={polish.alpha} n_W={polish.n_W_restricted}/"
              f"{polish.n_W_full} build={polish.wall_build_s*1000:.1f}ms "
              f"solve={polish.wall_solve_s*1000:.1f}ms", flush=True)

    fell_back = False
    if not polish.converged:
        return _fallback_monolithic(
            d, R, use_z2, build_full, M_mats_eff, d_eff,
            t_total0, wall_setup, wall_coarse, coarse, active=active,
            note="polish did not converge; fell back",
        )

    # --- Verify active-set was correct ----------------------------------
    t0 = time.time()
    verify: VerifyResult = verify_active_set(
        M_mats_eff, polish, active, tol=verify_tol,
    )
    wall_verify = time.time() - t0
    if verbose:
        print(f"[verify] max_violation={verify.max_violation:.3e}  "
              f"n_violators={verify.n_violators} ({wall_verify*1000:.1f}ms)",
              flush=True)
    if verify.max_violation > verify_tol:
        if auto_fallback_on_violation:
            return _fallback_monolithic(
                d, R, use_z2, build_full, M_mats_eff, d_eff,
                t_total0, wall_setup, wall_coarse, coarse,
                active=active, polish=polish, verify=verify,
                note=f"active-set verification failed (max_v={verify.max_violation:.3e}); "
                     "fell back to monolithic",
            )

    # --- Jansson rigorize -----------------------------------------------
    t0 = time.time()
    rig: RigorizationResult = rigorize_from_polish(polish, prec=rigorize_prec)
    wall_rigorize = time.time() - t0
    if verbose:
        print(f"[rigorize] alpha_rig={rig.alpha_rigorous:.10f}  "
              f"epsilon={rig.epsilon_shift:.3e} ({wall_rigorize*1000:.1f}ms)  "
              f"{rig.notes}", flush=True)

    wall_total = time.time() - t_total0
    return Tier4Result(
        d=d, R=R, use_z2=use_z2,
        alpha_rigorous=rig.alpha_rigorous,
        alpha_polish=polish.alpha,
        alpha_coarse=coarse.alpha,
        wall_setup_s=wall_setup,
        wall_coarse_s=wall_coarse,
        wall_active_s=wall_active,
        wall_polish_build_s=polish.wall_build_s,
        wall_polish_solve_s=polish.wall_solve_s,
        wall_verify_s=wall_verify,
        wall_rigorize_s=wall_rigorize,
        wall_total_s=wall_total,
        n_W_full=n_W_full,
        n_W_active=polish.n_W_restricted,
        n_cbeta_full=active.n_cbeta_total,
        n_cbeta_active=len(active.active_cbeta_idx),
        tol_active=active.tol_active,
        verify_max_violation=verify.max_violation,
        verify_n_violators=verify.n_violators,
        fell_back_to_full=False,
        coarse_kkt=coarse.kkt,
        coarse_backend=coarse.backend,
        epsilon_shift=rig.epsilon_shift,
        notes=rig.notes,
    )


def monolithic_solve(d: int, R: int, use_z2: bool = True,
                     tol: float = 1e-9, verbose: bool = False):
    """Reference: the standard MOSEK 1e-9 path.  Returns (alpha, wall_s)."""
    t0 = time.time()
    _, M_mats = build_window_matrices(d)
    if use_z2:
        M_mats, _ = project_window_set_to_z2_rescaled(M_mats, d)
        d_eff = z2_dim(d)
    else:
        d_eff = d
    build = build_handelman_lp(
        d_eff, M_mats, BuildOptions(R=R, use_z2=use_z2),
    )
    sol = solve_lp(build, solver="mosek", tol=tol, verbose=verbose)
    return sol, time.time() - t0


# ---------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------

def _fallback_monolithic(
    d, R, use_z2, build_full, M_mats_eff, d_eff,
    t_total0, wall_setup, wall_coarse, coarse,
    active=None, polish=None, verify=None, note="",
) -> Tier4Result:
    """Run the monolithic MOSEK 1e-9 solve and package as a Tier4Result."""
    t0 = time.time()
    sol = solve_lp(build_full, solver="mosek", tol=1e-9)
    wall_polish_solve = time.time() - t0

    alpha_polish = sol.alpha
    # Apply Jansson on the full LP solution
    try:
        from lasserre.polya_lp.tier4.rigorize import rigorize_lp_lb
        n_eq = build_full.A_eq.shape[0]
        y_eq = sol.y[:n_eq]
        A_ub = getattr(build_full, "A_ub", None)
        b_ub = getattr(build_full, "b_ub", None)
        y_ub = sol.y[n_eq:] if (A_ub is not None and A_ub.shape[0] > 0) else None
        rig = rigorize_lp_lb(
            build_full.A_eq, build_full.b_eq, A_ub, b_ub,
            build_full.c, build_full.bounds, y_eq, y_ub,
            alpha_polish=alpha_polish, prec=200,
        )
        alpha_rig = rig.alpha_rigorous
        eps_shift = rig.epsilon_shift
    except Exception:
        alpha_rig = None
        eps_shift = float("nan")

    wall_total = time.time() - t_total0
    return Tier4Result(
        d=d, R=R, use_z2=use_z2,
        alpha_rigorous=alpha_rig,
        alpha_polish=alpha_polish,
        alpha_coarse=coarse.alpha,
        wall_setup_s=wall_setup,
        wall_coarse_s=wall_coarse,
        wall_active_s=0.0,
        wall_polish_build_s=0.0,
        wall_polish_solve_s=wall_polish_solve,
        wall_verify_s=0.0,
        wall_rigorize_s=0.0,
        wall_total_s=wall_total,
        n_W_full=len(M_mats_eff),
        n_W_active=len(M_mats_eff),
        n_cbeta_full=(build_full.c_idx.stop - build_full.c_idx.start),
        n_cbeta_active=(build_full.c_idx.stop - build_full.c_idx.start),
        tol_active=float("nan"),
        verify_max_violation=0.0,
        verify_n_violators=0,
        fell_back_to_full=True,
        coarse_kkt=coarse.kkt,
        coarse_backend=coarse.backend,
        epsilon_shift=eps_shift,
        notes=f"FALLBACK: {note}",
    )

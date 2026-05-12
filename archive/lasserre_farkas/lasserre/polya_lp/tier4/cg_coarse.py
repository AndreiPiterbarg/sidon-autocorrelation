"""Cutting-plane + coarse-solve loop for Tier-4.

Composes term-sparsity row restriction (Tier 3) with a fast coarse LP
solver. Iteratively expands Sigma_R until no candidate beta has
c_beta < -tol in the full-LP extension at the current primal.

Invariant (PROVEN by find_violators's correctness):
  At convergence, alpha(restricted LP) = alpha(full LP).
  This holds because the full-LP rows omitted from the restricted LP
  have c_beta >= 0 in the extension, which is exactly the LP slack
  condition. So the restricted primal extends to a feasible full-LP
  primal at the same objective value.

We use the FAST coarse solver (MOSEK simplex by default) so that each
CG iteration is cheap. Active-set extraction and polish happen AFTER
this loop, in driver_v2.

Reused infrastructure (UNCHANGED, READ-ONLY):
  lasserre/polya_lp/term_sparsity.py    : Newton seed, expand_support, B_R
  lasserre/polya_lp/cutting_plane.py    : find_violators (vectorized)
  lasserre/polya_lp/build.py            : build_handelman_lp with restricted args
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import time

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp,
)
from lasserre.polya_lp.term_sparsity import (
    TermSparsitySupport, build_term_sparsity_support, expand_support,
    polynomial_support_from_M_mats,
)
from lasserre.polya_lp.cutting_plane import find_violators

from lasserre.polya_lp.tier4.coarse_solve import coarse_solve, CoarseResult


@dataclass
class CGCoarseRecord:
    iteration: int
    n_constraints_in_sigma: int
    n_q_vars_in_B: int
    alpha_coarse: Optional[float]
    n_violators_found: int
    max_violation: float
    n_candidates_checked: int
    coarse_kkt: float
    build_wall_s: float
    coarse_wall_s: float
    violator_check_wall_s: float
    coarse_status: str = ""


@dataclass
class CGCoarseResult:
    converged: bool
    final_alpha_coarse: Optional[float]
    final_support: TermSparsitySupport
    final_build: Optional[BuildResult]    # the LP at convergence
    final_coarse: Optional[CoarseResult]  # coarse solution at convergence
    A_window_support: List[Tuple[int, ...]]  # union supp(M_W); needed by find_violators
    iterations: List[CGCoarseRecord] = field(default_factory=list)
    total_wall_s: float = 0.0


def cg_coarse_solve(
    d_eff: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    coarse_backend: str = "mosek_simplex",
    coarse_tol: float = 1e-6,
    cg_violator_tol: float = 1e-7,
    max_cg_iter: int = 30,
    add_top_k: int = -1,
    include_low_degree_seed: bool = True,
    use_q_polynomial: bool = True,
    eliminate_c_slacks: bool = False,
    expand_radius: int = 1,
    verbose: bool = False,
) -> CGCoarseResult:
    """Run cutting-plane with the chosen coarse backend.

    Convergence: when find_violators returns no beta with c_beta <
    -cg_violator_tol at the coarse primal extended to the full-LP grid.

    For correctness, cg_violator_tol must be SMALLER (tighter) than
    coarse_tol: if the LP is solved to KKT ~ coarse_tol, the c_beta
    values can be off by up to coarse_tol. So we should use
    cg_violator_tol <= coarse_tol; when polish at 1e-9 follows, we'll
    recheck violators at 1e-9 in driver_v2.

    Returns final result; caller must run polish + verification afterwards.
    """
    t_total = time.time()
    iters: List[CGCoarseRecord] = []

    # Initial Newton-polytope seed
    ts = build_term_sparsity_support(
        M_mats, R, include_low_degree=include_low_degree_seed,
    )
    A_supp = ts.A

    if verbose:
        print(f"  [CG seed] |Sigma_R|={ts.n_constraints}/{ts.full_n_constraints}  "
              f"|B_R|={ts.n_q_vars}/{ts.full_n_q_vars}  |A|={len(A_supp)}", flush=True)

    final_build: Optional[BuildResult] = None
    final_coarse: Optional[CoarseResult] = None

    for it in range(max_cg_iter):
        opts = BuildOptions(
            R=R, use_z2=True,
            use_q_polynomial=use_q_polynomial,
            eliminate_c_slacks=eliminate_c_slacks,
            restricted_Sigma_R=ts.Sigma_R,
            restricted_B_R=ts.B_R,
            verbose=False,
        )
        t0 = time.time()
        build = build_handelman_lp(d_eff, M_mats, opts)
        wall_b = time.time() - t0

        t0 = time.time()
        coarse = coarse_solve(build, tol=coarse_tol, backend=coarse_backend,
                              verbose=False)
        wall_c = time.time() - t0

        if not coarse.converged:
            iters.append(CGCoarseRecord(
                iteration=it, n_constraints_in_sigma=len(ts.Sigma_R),
                n_q_vars_in_B=len(ts.B_R), alpha_coarse=None,
                n_violators_found=-1, max_violation=float("inf"),
                n_candidates_checked=0, coarse_kkt=coarse.kkt,
                build_wall_s=wall_b, coarse_wall_s=wall_c,
                violator_check_wall_s=0.0,
                coarse_status=f"FAILED_{coarse.raw_status}",
            ))
            if verbose:
                print(f"  [CG iter {it}] coarse FAILED status={coarse.raw_status}",
                      flush=True)
            return CGCoarseResult(
                converged=False, final_alpha_coarse=None,
                final_support=ts, final_build=build, final_coarse=coarse,
                A_window_support=A_supp,
                iterations=iters, total_wall_s=time.time() - t_total,
            )

        # Wrap coarse into a SolveResult-shaped object that find_violators expects
        # (it reads .x and .alpha).
        from lasserre.polya_lp.solve import SolveResult as _SR
        sol_for_check = _SR(
            status="OPTIMAL", alpha=coarse.alpha, x=coarse.x,
            solver=coarse.backend, wall_s=coarse.wall_s,
        )

        t0 = time.time()
        violators, max_viol, n_cand = find_violators(
            build, sol_for_check, M_mats, R, A_supp,
            tol=cg_violator_tol,
            max_violators_to_return=(add_top_k if add_top_k > 0 else -1),
        )
        wall_v = time.time() - t0

        rec = CGCoarseRecord(
            iteration=it, n_constraints_in_sigma=len(ts.Sigma_R),
            n_q_vars_in_B=len(ts.B_R), alpha_coarse=coarse.alpha,
            n_violators_found=len(violators), max_violation=max_viol,
            n_candidates_checked=n_cand, coarse_kkt=coarse.kkt,
            build_wall_s=wall_b, coarse_wall_s=wall_c,
            violator_check_wall_s=wall_v,
            coarse_status="OPTIMAL",
        )
        iters.append(rec)

        if verbose:
            print(f"  [CG iter {it}] alpha={coarse.alpha:.7f}  "
                  f"|Sigma|={len(ts.Sigma_R)}/{ts.full_n_constraints}  "
                  f"violators={len(violators)}/{n_cand}  "
                  f"max_viol={max_viol:.2e}  "
                  f"build={wall_b*1000:.0f}ms  coarse={wall_c*1000:.0f}ms  "
                  f"check={wall_v*1000:.0f}ms",
                  flush=True)

        final_build = build
        final_coarse = coarse

        if not violators:
            return CGCoarseResult(
                converged=True, final_alpha_coarse=coarse.alpha,
                final_support=ts, final_build=build, final_coarse=coarse,
                A_window_support=A_supp,
                iterations=iters, total_wall_s=time.time() - t_total,
            )

        # Expand Sigma_R with the violators
        ts = expand_support(ts, violators, expand_radius=expand_radius)

    # Hit max_cg_iter without converging
    if verbose:
        print(f"  [CG] max_iter={max_cg_iter} reached without convergence",
              flush=True)
    return CGCoarseResult(
        converged=False, final_alpha_coarse=(coarse.alpha if final_coarse else None),
        final_support=ts, final_build=final_build, final_coarse=final_coarse,
        A_window_support=A_supp,
        iterations=iters, total_wall_s=time.time() - t_total,
    )


def expand_support_with_betas(
    ts: TermSparsitySupport,
    new_betas: List[Tuple[int, ...]],
    d: int,
    R: int,
) -> TermSparsitySupport:
    """Add specific betas to Sigma_R (used by polish-time verification recovery).

    This is similar to expand_support but adds the EXACT violator betas
    rather than their lattice neighbors. It's the minimal expansion that
    absorbs the violation.
    """
    Sigma_set = set(ts.Sigma_R)
    for b in new_betas:
        if sum(b) <= R:
            Sigma_set.add(tuple(b))
    Sigma_list = sorted(Sigma_set)
    from lasserre.polya_lp.term_sparsity import compute_B_R
    B_list = compute_B_R(Sigma_list, d, R)
    return TermSparsitySupport(
        Sigma_R=Sigma_list, B_R=B_list, A=ts.A, n_var_dim=d, R=R,
        seed_iter=ts.seed_iter + 1,
        n_constraints=len(Sigma_list), n_q_vars=len(B_list),
        full_n_constraints=ts.full_n_constraints,
        full_n_q_vars=ts.full_n_q_vars,
        notes=f"Polish-time expansion (+{len(new_betas)})",
    )

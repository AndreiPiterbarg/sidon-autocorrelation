"""Multi-stage CG + active-set pipeline.

Newton-polytope sparsity gives ZERO row reduction for the Sidon LP (the
support A covers all (i,j) pairs after Z/2, so A oplus Delta_{R-2} = full
Sigma_R). So Tier 3 alone is a no-op. The win has to come from Tier 2.

Multi-stage strategy:
  1. Solve full LP at R_seed (small, fast). Extract active windows
     A_W^(0) = {W : lambda_W > tol}. Typically this is much smaller than
     |W_full| even for the Sidon LP.
  2. Optionally include neighbors of A_W^(0) (window-index distance <= k).
  3. Run solve_with_cg_active_set at R_target with initial_W_active = A_W^(0).
     The pricing step adds any wrongly-dropped windows. At convergence,
     restricted-LP alpha = full-LP alpha.

Soundness: CG+activeSet pricing recovers full-LP optimum at convergence
(see cg_active_set.py docstring). Multi-stage is a *seeding* heuristic;
correctness is unaffected.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import time
import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.cg_active_set import (
    CGActiveSetResult, solve_with_cg_active_set,
)


@dataclass
class MultiStageResult:
    seed_alpha: Optional[float]
    seed_R: int
    seed_active_W: List[int]
    seed_wall_s: float
    target_alpha: Optional[float]
    target_R: int
    target_result: CGActiveSetResult
    total_wall_s: float


def extract_active_windows(
    sol: SolveResult,
    n_W: int,
    lambda_idx: slice,
    tol: float = 1e-9,
) -> List[int]:
    """Return indices of windows with lambda > tol at the LP optimum."""
    if sol.x is None:
        return list(range(n_W))
    lam = sol.x[lambda_idx]
    return [w for w, v in enumerate(lam) if v > tol]


def solve_multistage(
    d: int,
    M_mats: Sequence[np.ndarray],
    R_target: int,
    *,
    R_seed: int = 8,
    tol: float = 1e-8,
    active_lambda_tol: float = 1e-9,
    expand_neighbors: int = 0,
    max_iter_per_stage: int = 30,
    use_q_polynomial: bool = True,
    verbose: bool = True,
) -> MultiStageResult:
    """Multi-stage solve: seed W at low R, then CG+activeSet at R_target.

    Args:
        R_seed: solve full LP at this R first (cheap; identifies which
                windows have lambda_W > tol).
        active_lambda_tol: threshold on lambda for "active".
        expand_neighbors: include +/- k window indices around each active
                          one (cheap insurance).
    """
    t_total = time.time()
    n_W = len(M_mats)

    # Stage 1: full LP at R_seed.
    if verbose:
        print(f"  [stage 1] full LP at R={R_seed}, n_W={n_W}", flush=True)
    opts = BuildOptions(
        R=R_seed, use_z2=True, eliminate_c_slacks=False,
        use_q_polynomial=use_q_polynomial,
    )
    t0 = time.time()
    build = build_handelman_lp(d, M_mats, opts)
    sol = solve_lp(build, solver='mosek', verbose=False)
    seed_wall = time.time() - t0
    if sol.alpha is None:
        raise RuntimeError(
            f"Stage 1 (R={R_seed}) failed: status={sol.status}")

    active = extract_active_windows(
        sol, n_W, build.lambda_idx, tol=active_lambda_tol)

    # Optional: expand by neighbors.
    if expand_neighbors > 0:
        idx_set = set(active)
        for w in active:
            for off in range(-expand_neighbors, expand_neighbors + 1):
                nb = w + off
                if 0 <= nb < n_W:
                    idx_set.add(nb)
        active = sorted(idx_set)
    else:
        active = sorted(active)

    if verbose:
        print(
            f"  [stage 1] alpha={sol.alpha:.7f}  "
            f"|active|={len(active)}/{n_W}  wall={seed_wall:.1f}s",
            flush=True,
        )

    # Stage 2: CG+activeSet at R_target with seeded W.
    if verbose:
        print(f"  [stage 2] CG+activeSet at R={R_target} with "
              f"|W_init|={len(active)}", flush=True)
    cgas = solve_with_cg_active_set(
        d=d, M_mats=M_mats, R=R_target,
        max_iter=max_iter_per_stage, tol=tol,
        initial_W_active=active,
        verbose=verbose,
    )

    return MultiStageResult(
        seed_alpha=sol.alpha,
        seed_R=R_seed,
        seed_active_W=active,
        seed_wall_s=seed_wall,
        target_alpha=cgas.final_alpha,
        target_R=R_target,
        target_result=cgas,
        total_wall_s=time.time() - t_total,
    )

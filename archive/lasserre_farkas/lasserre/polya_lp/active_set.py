"""Active-set window selection: after solving the LP, identify which windows
have nonzero lambda and use only those for subsequent solves at higher R.

Rationale: at the optimum, only a small subset of windows have lambda_W > 0
(typically 5-20 of the ~d^2 windows). Restricting to active windows shrinks
the LP without losing optimality (provided the active set is stable).

Usage pattern:
  1. Solve at R=R0 with all windows -> get active set S = {W : lambda_W > tol}.
  2. Solve at R=R1 > R0 with windows restricted to S (plus optional neighbors).
  3. Iterate until alpha stops increasing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import time
import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)


def extract_active_windows(
    build: BuildResult,
    sol: SolveResult,
    tol: float = 1e-9,
) -> List[int]:
    """Return indices of windows with lambda > tol at the LP optimum."""
    if sol.x is None or build.fixed_lambda is not None:
        return list(range(build.n_windows))
    lam = sol.x[build.lambda_idx]
    return [w for w, v in enumerate(lam) if v > tol]


@dataclass
class ActiveSetResult:
    R_history: List[int]
    alpha_history: List[float]
    active_size_history: List[int]
    wall_history: List[float]
    final_alpha: float
    final_active_indices: List[int]
    total_wall_s: float


def active_set_solve(
    d: int,
    R_schedule: List[int],
    use_z2: bool = True,
    use_q_polynomial: bool = True,
    initial_R: Optional[int] = None,
    growth: int = 0,
    solver: str = "highs",
    verbose: bool = True,
) -> ActiveSetResult:
    """Solve the LP at increasing R, restricting to active windows after R0.

    growth: at each step, also include the `growth` windows nearest to active
    ones (helpful if the active set is unstable across R).
    """
    t_total = time.time()
    _, M_mats_orig = build_window_matrices(d)
    if use_z2:
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)
    else:
        M_mats_eff = M_mats_orig
        d_eff = d
    n_W_full = len(M_mats_eff)

    R_hist: List[int] = []
    alpha_hist: List[float] = []
    size_hist: List[int] = []
    wall_hist: List[float] = []

    # Step 1: full LP at the lowest R to seed the active set
    R0 = initial_R if initial_R is not None else R_schedule[0]
    if verbose:
        print(f"  [seed] full LP at R={R0}, n_W={n_W_full}", flush=True)
    opts = BuildOptions(R=R0, use_z2=use_z2, use_q_polynomial=use_q_polynomial)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    sol = solve_lp(build, solver=solver)
    wall = time.time() - t0
    if sol.alpha is None:
        raise RuntimeError(f"Seed solve failed: {sol.status}")
    R_hist.append(R0); alpha_hist.append(sol.alpha)
    size_hist.append(n_W_full); wall_hist.append(wall)

    active_indices = extract_active_windows(build, sol)
    if verbose:
        print(f"  [seed] alpha={sol.alpha:.6f}, active windows: {len(active_indices)}/{n_W_full}, wall={wall:.1f}s", flush=True)

    # Subsequent R values: restrict to active set
    for R in R_schedule:
        if R == R0:
            continue
        # Optionally extend active set with `growth` neighbors (closest by index)
        idx_set = set(active_indices)
        if growth > 0:
            for i in list(active_indices):
                for d_off in range(-growth, growth + 1):
                    nb = i + d_off
                    if 0 <= nb < n_W_full:
                        idx_set.add(nb)
        sub_indices = sorted(idx_set)
        sub_M_mats = [M_mats_eff[i] for i in sub_indices]

        if verbose:
            print(f"  [step] R={R}, |active|={len(sub_indices)}/{n_W_full}", flush=True)
        opts = BuildOptions(R=R, use_z2=use_z2, use_q_polynomial=use_q_polynomial)
        t0 = time.time()
        build = build_handelman_lp(d_eff, sub_M_mats, opts)
        sol = solve_lp(build, solver=solver)
        wall = time.time() - t0
        if sol.alpha is None:
            if verbose:
                print(f"  [step] R={R} solve failed: {sol.status}")
            continue
        # Map active indices back to full numbering
        local_active = extract_active_windows(build, sol)
        active_indices = [sub_indices[i] for i in local_active]
        R_hist.append(R); alpha_hist.append(sol.alpha)
        size_hist.append(len(sub_indices)); wall_hist.append(wall)
        if verbose:
            print(f"  [step] R={R} alpha={sol.alpha:.6f}, |active|={len(active_indices)}, wall={wall:.1f}s", flush=True)

    return ActiveSetResult(
        R_history=R_hist,
        alpha_history=alpha_hist,
        active_size_history=size_hist,
        wall_history=wall_hist,
        final_alpha=alpha_hist[-1],
        final_active_indices=active_indices,
        total_wall_s=time.time() - t_total,
    )

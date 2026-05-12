"""MOSEK polish on the active-window-restricted LP.

Builds a smaller LP using ONLY the active windows (those with lambda_W>0
predicted by tier4.active_set), then solves with MOSEK IPM at 1e-9.

Soundness: restricting the LP to a subset of windows is equivalent to
fixing the inactive lambda_W to 0. This is sound IF the active-set
prediction is correct: the polished alpha equals the true LP optimum.

Verification (recommended): after polish, check the dropped windows
have non-positive reduced cost in the restricted LP's dual. If any
dropped window has positive reduced cost, the active set was wrong and
we must add it back. Implemented in `verify_active_set`.

NOTE on c-slacks: we KEEP all c_beta slack variables in the polish LP
(do not restrict rows). This is conservative: it preserves all original
constraints. If the user wants further row pruning, layer in
term_sparsity / Newton-polytope (that's Tier-3 territory).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import time
import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, BuildResult, build_handelman_lp,
)
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.tier4.active_set import ActiveSet


@dataclass
class PolishResult:
    alpha: Optional[float]
    build_polish: Optional[BuildResult]
    sol_polish: Optional[SolveResult]
    n_W_restricted: int
    n_W_full: int
    wall_build_s: float
    wall_solve_s: float
    converged: bool
    notes: str = ""
    verify: Optional["VerifyResult"] = None


def polish_via_mosek(
    M_mats_full: Sequence[np.ndarray],
    d_eff: int,
    R: int,
    active: ActiveSet,
    use_q_polynomial: bool = True,
    eliminate_c_slacks: bool = False,
    tol: float = 1e-9,
    verbose: bool = False,
) -> PolishResult:
    """Polish: build the LP restricted to active windows, solve with MOSEK.

    M_mats_full: the FULL window-matrix list (after Z/2 projection). The
                 active.active_lambda_idx points into this list.
    d_eff      : effective ambient dimension (Z/2-reduced).
    R          : Polya degree.

    Returns the polished alpha and the underlying SolveResult (with
    duals for downstream Jansson rigorization).
    """
    if len(active.active_lambda_idx) == 0:
        raise ValueError("active set has no lambda windows; cannot polish")

    M_active = [np.asarray(M_mats_full[i], dtype=np.float64)
                for i in active.active_lambda_idx]

    t0 = time.time()
    build_polish = build_handelman_lp(
        d_eff, M_active,
        BuildOptions(
            R=R, use_z2=True,
            use_q_polynomial=use_q_polynomial,
            eliminate_c_slacks=eliminate_c_slacks,
            verbose=verbose,
        ),
    )
    wall_build = time.time() - t0

    t0 = time.time()
    sol = solve_lp(build_polish, solver="mosek", tol=tol, verbose=verbose)
    wall_solve = time.time() - t0

    return PolishResult(
        alpha=sol.alpha,
        build_polish=build_polish,
        sol_polish=sol,
        n_W_restricted=len(M_active),
        n_W_full=len(M_mats_full),
        wall_build_s=wall_build,
        wall_solve_s=wall_solve,
        converged=(sol.status == "OPTIMAL"),
        notes=f"polished n_W {len(M_active)}/{len(M_mats_full)}",
    )


# ---------------------------------------------------------------------
# Verification: check that dropped windows had non-positive reduced cost
# ---------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Active-set verification.

    bar_c_W : reduced costs of DROPPED windows in the polished LP, where
              bar_c_W = sum_beta y_beta * coeff_W(beta) - simplex_dual.
              If max(bar_c_W) <= tol, all dropped windows had no incentive
              to be active and the polished alpha equals the full LP
              optimum.
    max_violation : max over dropped windows of bar_c_W (positive = bad).
    n_violators   : how many dropped windows would prefer to be active.
    """
    max_violation: float
    n_violators: int
    bar_c_dropped: np.ndarray
    dropped_indices: np.ndarray


def verify_active_set(
    M_mats_full: Sequence[np.ndarray],
    polish: PolishResult,
    active: ActiveSet,
    tol: float = 1e-7,
) -> VerifyResult:
    """Verify that polished dual is feasible for the full LP's dropped columns.

    For variable lambda_W (lo=0, hi=None) the dual feasibility condition
    is A_W^T y <= c_W = 0, i.e.,
        sum_{beta with |beta|=2} y*_beta * coeff_W(beta)  +  y*_simplex  <=  0
    for every column W. The polished LP only enforces this for active W;
    we check the dropped W's manually.

    NOTE: a violator here means the polished y* is not a valid dual for
    the full LP at the polished primal. The polished primal is still the
    optimal of the full LP when the active set is right (degeneracy
    aside). So a positive max_violation does NOT necessarily mean the
    active set is wrong -- it means the polished y* needs adjustment
    before being used in Jansson rigorization on the full LP.
    """
    sol = polish.sol_polish
    if sol is None or sol.y is None:
        return VerifyResult(max_violation=float("inf"), n_violators=-1,
                            bar_c_dropped=np.zeros(0),
                            dropped_indices=np.zeros(0, dtype=np.int64))

    build_polish = polish.build_polish
    monos_le_R = build_polish.monos_le_R
    n_W_full = len(M_mats_full)
    active_set = set(active.active_lambda_idx)
    dropped_idx = np.array([w for w in range(n_W_full) if w not in active_set],
                           dtype=np.int64)
    if dropped_idx.size == 0:
        return VerifyResult(max_violation=0.0, n_violators=0,
                            bar_c_dropped=np.zeros(0),
                            dropped_indices=dropped_idx)

    # The simplex constraint sum_W lambda = 1 in build.py is encoded as a
    # SEPARATE row with RHS = 1; let me verify by inspecting the build.
    # In lasserre/polya_lp/build.py, the LP rows are: n_le_R Polya rows
    # (per |beta| <= R), THEN one simplex row (when n_lambda_var > 0).
    # So y has length n_le_R + 1 in the unrestricted form; the LAST
    # entry is the simplex dual.
    n_le_R = len(monos_le_R)
    n_eq = build_polish.A_eq.shape[0]
    has_simplex_row = (n_eq == n_le_R + 1)
    if has_simplex_row:
        y_polya = sol.y[:n_le_R]
        y_simplex = float(sol.y[n_le_R])
    else:
        y_polya = sol.y[:n_le_R]
        y_simplex = 0.0

    # Compute A_W^T y for each dropped W; we want this <= 0 (rc >= 0).
    # Violator = A_W^T y > tol  (positive sum means y_polished is not
    # dual-feasible on the dropped column).
    bar_c = np.zeros(dropped_idx.size, dtype=np.float64)
    beta_to_y = {tuple(b): y_polya[i] for i, b in enumerate(monos_le_R)}
    d = len(monos_le_R[0]) if monos_le_R else 0
    for k, w in enumerate(dropped_idx):
        M_W = np.asarray(M_mats_full[w], dtype=np.float64)
        s = 0.0
        for i in range(d):
            beta = tuple(2 if t == i else 0 for t in range(d))
            y_b = beta_to_y.get(beta)
            if y_b is not None and M_W[i, i] != 0:
                s += y_b * M_W[i, i]
        for i in range(d):
            for j in range(i + 1, d):
                v = M_W[i, j]
                if v == 0:
                    continue
                beta = tuple(1 if (t == i or t == j) else 0 for t in range(d))
                y_b = beta_to_y.get(beta)
                if y_b is not None:
                    s += y_b * 2.0 * v
        bar_c[k] = s + y_simplex

    max_v = float(bar_c.max())
    n_viol = int((bar_c > tol).sum())
    return VerifyResult(
        max_violation=max_v, n_violators=n_viol,
        bar_c_dropped=bar_c, dropped_indices=dropped_idx,
    )

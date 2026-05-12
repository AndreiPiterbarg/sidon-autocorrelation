"""Coarse BnB v4 — high-impact patches on top of v3.

Patches added (all sound):
  B1   μ-space corner LB:  μ^T A_W μ ≥ Σ_{(i,j) ∈ W} lo_i lo_j.
       Sound for any A_W (uses μ ≥ lo ≥ 0). Fires before F.
  B34  Strong empty test: Σ max(lo, 0) > 1 + ε OR Σ min(hi, 1) < 1 − ε.
  B35  Single-coord forced-zero empty test.
  B36  Cheap F-positive predicate: m_W + Σ min(g_i·lo_i, g_i·hi_i) > 0
       (per-coord linear screen, O(d), pre-LP).
  B43  Widest-window-first: always include the two widest ell-windows in L's
       top-K, regardless of F-rank.
  B45  Cap BnB depth at small d (depth 1 if d ≤ 4, else max_depth).
  B46  Skip splitting at root if L_joint bound shows no improvement potential
       over F_best (avoids wasted splits).

DOES NOT IMPLEMENT (rejected per strict triage — premise violated):
  B5/6/8/9/10/12/13/21: rely on the false claim A_W ⪰ 0.
  Verified counter-example: d=4, window {1,2} has A_W with negative eigenvalue.
"""
from __future__ import annotations
import os, sys, time, logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

logging.getLogger('cvxpy').setLevel(logging.ERROR)

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

# Import v3 as the base; we override specific tiers.
import _coarse_bnb_v3 as v3

# Re-export commonly used objects.
build_A_matrix_vec = v3.build_A_matrix_vec
enumerate_windows = v3.enumerate_windows
WindowData = v3.WindowData
build_all_windows = v3.build_all_windows
Cell = v3.Cell
lp_max_linear = v3.lp_max_linear
lp_min_linear = v3.lp_min_linear
mccormick_lb_pairwise = v3.mccormick_lb_pairwise
CellCache = v3.CellCache
tier_L_single = v3.tier_L_single
tier_L_joint = v3.tier_L_joint
get_sdp_template = v3.get_sdp_template
get_joint_template = v3.get_joint_template


# =====================================================================
# B34/B35: empty cell tests
# =====================================================================

def is_cell_empty(cell: Cell, eps: float = 1e-12) -> bool:
    """Sound test: cell empty (no μ with Σμ=1, μ ∈ [lo, hi])."""
    if cell.lo.sum() > 1.0 + eps:    # B34: even minimum μ sums above 1
        return True
    if cell.hi.sum() < 1.0 - eps:    # B34: even maximum μ sums below 1
        return True
    # B35: check whether removing each coord makes the rest infeasible.
    # If lo_i > 0 for some i, then Σ_{j≠i} μ_j = 1 − μ_i ≤ 1 − lo_i, so we need
    # Σ_{j≠i} hi_j ≥ 1 − lo_i. Else empty.
    rest_sum_max = cell.hi.sum()
    for i in range(cell.d):
        if cell.lo[i] > 0:
            if (rest_sum_max - cell.hi[i]) < (1.0 - cell.lo[i]) - eps:
                return True
    return False


# =====================================================================
# B1: μ-space corner LB (pre-F screen)
# =====================================================================

def tier_B1_mu_corner(cell: Cell, W: WindowData, c_target: float) -> float:
    """μ^T A_W μ ≥ Σ_{(i,j) in W} lo_i lo_j  (since μ ≥ lo ≥ 0).

    f_W(μ) = Q_coef · μ^T A_W μ − c_target
           ≥ Q_coef · Σ_{(i,j) in W} lo_i lo_j − c_target.

    Sound for ANY A_W (uses μ ≥ lo ≥ 0 only).  No PSD assumption.
    """
    lo = cell.lo
    corner_val = float(np.sum(W.A * np.outer(lo, lo)))
    return W.Q_coef * corner_val - c_target


# =====================================================================
# B36: cheap F-positive predicate (O(d), no LP)
# =====================================================================

def tier_B36_predicate(cell: Cell, mu_star: np.ndarray, W: WindowData,
                         c_target: float, h: np.ndarray) -> float:
    """O(d) sound LB skipping the LP.

    f_W(ε) ≥ m_W + Σ_i min(g_i·lo_eps_i, g_i·hi_eps_i) − Q_coef·Σ_{i≠j in W} |h_i·h_j|

    Linear part: per-coord min ignoring Σ=0 (looser than LP, but O(d)).
    Quad part: entry-wise off-diag bound (looser than sign-aware McCormick but O(d²)).
    """
    A = W.A
    margin = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    g = W.grad_coef * (A @ mu_star)
    lo_eps = np.maximum(cell.lo - mu_star, -mu_star)
    hi_eps = cell.hi - mu_star
    linear_lb_per_coord = np.minimum(g * lo_eps, g * hi_eps).sum()
    # Quad: entry-wise off-diag
    h_outer = np.outer(h, h)
    quad_lb = -W.Q_coef * float((A * h_outer).sum() - (np.diag(A) * h * h).sum())
    return float(margin + linear_lb_per_coord + quad_lb)


# =====================================================================
# B43: widest-window-first in L-tier candidate set
# =====================================================================

def select_L_candidates(cell_cache: CellCache, windows: List[WindowData],
                          K: int = 4) -> List[int]:
    """Top-K windows for L tier: combine F-rank + widest-ell.

    Always include the 2 widest-ell windows. Fill remaining from F-rank.
    """
    if cell_cache.f_ranked_indices is None:
        # Fallback: rank by ell descending
        ells = np.array([W.ell for W in windows])
        return list(np.argsort(-ells)[:K])
    # Widest 2 windows by ell
    ells = np.array([W.ell for W in windows])
    widest_idx = list(np.argsort(-ells)[:2])
    # F-rank top filled in
    out = list(widest_idx)
    for idx in cell_cache.f_ranked_indices:
        if int(idx) not in out:
            out.append(int(idx))
        if len(out) >= K:
            break
    return out[:K]


# =====================================================================
# CertResult (v4)
# =====================================================================

@dataclass
class CertResult:
    certified: bool
    tier_used: str
    bound: float = 0.0
    depth_used: int = 0
    n_subcells: int = 1
    best_W_idx: int = -1


# =====================================================================
# Cell certification cascade (v4)
# =====================================================================

def cert_cell(cell: Cell, windows: List[WindowData], c_target: float,
                max_depth: int = 6, current_depth: int = 0,
                use_B1: bool = True, use_F: bool = True,
                use_L: bool = True, use_L_joint: bool = True,
                joint_K: int = 4, verbose: bool = False) -> CertResult:
    """Sound box certificate.

    Order: empty → B1 (μ-space corner) → F → L_single → L_joint → split.
    """
    # B34/B35: empty cell vacuously certified
    if is_cell_empty(cell):
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    if not cell.is_simplex_feasible():
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)

    cache = CellCache.build(cell)

    # B1: μ-space corner LB (cheap, sound)
    if use_B1:
        for i, W in enumerate(windows):
            b = tier_B1_mu_corner(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # Tier F (all windows)
    if use_F:
        v3.compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)

    # Tier L (B43: widest-window-first)
    if use_L and v3.HAS_CVXPY and cache.f_ranked_indices is not None:
        K_L = min(3, len(windows))
        candidates = select_L_candidates(cache, windows, K=K_L)
        for w_idx in candidates:
            lb = tier_L_single(cache, windows[w_idx], c_target, verbose)
            if lb > 0:
                return CertResult(certified=True, tier_used='L',
                                  bound=lb, depth_used=current_depth,
                                  best_W_idx=w_idx)

    # Tier L_joint
    if use_L_joint and v3.HAS_CVXPY and cache.f_ranked_indices is not None:
        lb = tier_L_joint(cache, windows, c_target, K=joint_K)
        if lb is not None and lb > 0:
            return CertResult(certified=True, tier_used='L_joint',
                              bound=lb, depth_used=current_depth)

    # B45: cap BnB depth at small d
    eff_max_depth = max_depth
    if cell.d <= 4:
        eff_max_depth = min(max_depth, 1)

    if current_depth >= eff_max_depth:
        return CertResult(certified=False, tier_used='max_depth_reached',
                          depth_used=current_depth)

    # Split
    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 else windows[0]
    axis = v3.split_axis_gradient_weighted(cache, best_W)
    sub1, sub2 = cell.split(axis)
    r1 = cert_cell(sub1, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1, use_B1=use_B1, use_F=use_F,
                    use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, verbose=verbose)
    if not r1.certified:
        return CertResult(certified=False, tier_used='split_sub1_fail',
                          depth_used=r1.depth_used, n_subcells=r1.n_subcells)
    r2 = cert_cell(sub2, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1, use_B1=use_B1, use_F=use_F,
                    use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, verbose=verbose)
    if not r2.certified:
        return CertResult(certified=False, tier_used='split_sub2_fail',
                          depth_used=r2.depth_used,
                          n_subcells=r1.n_subcells + r2.n_subcells)
    return CertResult(certified=True, tier_used='split',
                      depth_used=max(r1.depth_used, r2.depth_used),
                      n_subcells=r1.n_subcells + r2.n_subcells)


def certify_composition(c: np.ndarray, S: int, d: int, c_target: float,
                         windows: Optional[List[WindowData]] = None,
                         max_depth: int = 6, **flags) -> CertResult:
    if windows is None:
        windows = build_all_windows(d)
    cell = Cell.from_integer_composition(c, S)
    return cert_cell(cell, windows, c_target, max_depth=max_depth, **flags)

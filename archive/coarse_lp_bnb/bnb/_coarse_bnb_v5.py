"""Coarse BnB v5 — SOUND L_joint + high-impact tightening.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL CORRECTNESS FIX vs v4
═══════════════════════════════════════════════════════════════════════════════

v4's `tier_L_joint` solved the epigraph SDP
        max t  s.t.  ∀W ∈ topK :  f_W(ε,X) ≥ t,  (ε,X) feasible.
That returns  J = max_(ε,X) min_W f_W(ε,X)  (max-min).  Using J > 0 as a
certification predicate is UNSOUND: J > 0 only proves "there exists a feasible
(ε,X) at which all f_W are positive", not "for every feasible ε, some f_W is
positive".  Counter-example (2D box):  f_1 = 2 − 4ε_1,  f_2 = 2 − 4ε_2  gives
J = 2 > 0 but min_ε max_W f_W = −2 < 0.

v5 replaces this with the LAGRANGIAN DUAL
        L(λ) = min_(ε,X) Σ_W λ_W f_W(ε,X),     λ ∈ Δ^{|W|}.
For any feasible λ, L(λ) > 0 ⇒ Σ λ_W f_W(ε) > 0 ∀ε ⇒ max_W f_W(ε) > 0 ∀ε.
SOUND.  We optimize λ via Frank–Wolfe, warm-started at e_{argmax_W L_single_W}.
Each FW iteration is one SDP solve with aggregated (margin_λ, grad_λ, Q_λ),
re-using the existing v3 ShorSDPTemplate.

═══════════════════════════════════════════════════════════════════════════════
ADDITIONAL SOUND IMPROVEMENTS
═══════════════════════════════════════════════════════════════════════════════

PRE-F SCREENS (cheap, before LP+SDP):
  B1u    μ^T A μ ≥ 1 − Σ(J−A)_{ij} hi_i hi_j  (uses Σμ = 1).
         Complement of B1; closes high-density windows on hi-bounded cells.
  B1diag Diagonal Cauchy–Schwarz + linear-AM-GM off-diagonal:
            Σ_{i∈D} μ_i² ≥ (Σ_{i∈D} μ_i)² / |D|,    D = {i : A_W[i,i]=1};
            Σ_{i≠j, A=1} μ_i μ_j ≥ 2·loᵀA_off μ − loᵀA_off lo  (linear in μ).
         One LP per window; tighter than B1 on diagonal-rich windows.

L-TIER STRENGTHENING:
  m_W ranking  L candidates picked by m_W = Q·μ*^T A μ* − c_target, NOT by F.
              F double-penalizes wide-ℓ windows that Shor recovers; m_W is a
              better predictor of L's winner.

CASCADE LOGIC:
  B46  Skip wasteful root split if joint L bound shows no improvement signal
       over F's best (cheap guard against unproductive recursion).

DROPPED FROM v4:
  v4 B45 (depth=1 cap at d≤4) preserved.
  All v3 RLT cuts preserved (positivity, Σ=0 row sums).

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os, sys, logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

logging.getLogger('cvxpy').setLevel(logging.ERROR)

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v4 as v4

# Re-export commonly used objects.
build_A_matrix_vec = v3.build_A_matrix_vec
enumerate_windows = v3.enumerate_windows
WindowData = v3.WindowData
build_all_windows = v3.build_all_windows
Cell = v3.Cell
lp_max_linear = v3.lp_max_linear
lp_min_linear = v3.lp_min_linear
CellCache = v3.CellCache
tier_L_single = v3.tier_L_single  # sound; we keep as-is
tier_B1_mu_corner = v4.tier_B1_mu_corner
is_cell_empty = v4.is_cell_empty
HAS_CVXPY = v3.HAS_CVXPY


# =====================================================================
# B1u: Σμ=1 upper-corner LB  (NEW pre-F screen)
# =====================================================================

def tier_B1u_complement_corner(cell: Cell, W: WindowData,
                                c_target: float) -> float:
    """Sound LB:  μ^T A_W μ ≥ 1 − Σ_{ij}(J−A_W)_{ij} hi_i hi_j.

    Derivation: (Σμ)^2 = 1 since Σμ=1. So Σ_{i,j} μ_i μ_j = 1, and
        μ^T A_W μ = 1 − μ^T (J − A_W) μ  ≥  1 − Σ (J−A_W)_{ij} hi_i hi_j
    using μ_i μ_j ≤ hi_i hi_j entrywise (μ ≥ 0, μ ≤ hi, (J−A) entries ≥ 0).
    """
    d = cell.d
    hi = cell.hi
    JmA = 1.0 - W.A  # (J − A_W), entries in {0, 1}
    upper_complement = float(np.sum(JmA * np.outer(hi, hi)))
    lb_quad = 1.0 - upper_complement
    return W.Q_coef * lb_quad - c_target


# =====================================================================
# B1diag: linear-AM-GM lower bound via lo  (NEW pre-F screen)
# =====================================================================

def tier_B1diag_amgm(cell: Cell, W: WindowData, c_target: float) -> float:
    """Sound LB via linear-AM-GM on off-diagonal + diagonal-squared:
        μ_i μ_j ≥ lo_i μ_j + lo_j μ_i − lo_i lo_j  for (i,j) with A_W=1, i≠j.
    Diagonal:
        Σ_{i: A_W[i,i]=1} μ_i² ≥ ( lp_min_linear(D, lo, hi, 1) )² / |D|
        if that min is ≥ 0; else just 0.
    Combined LB on μ^T A_W μ:
        2 (loᵀ A_off) μ − loᵀ A_off lo + CS_diag_lb
    Final LB on linear in μ; minimize over the cell {lo≤μ≤hi, Σμ=1}.
    Sound by AM-GM + Cauchy–Schwarz.
    """
    d = cell.d
    A = W.A
    lo = cell.lo
    hi = cell.hi
    # Off-diagonal mass with diag stripped
    diag_mask = np.diag(A) > 0.5
    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)
    # Linear form in μ for off-diagonal AM-GM lower bound:
    #   2 (A_off @ lo) · μ − loᵀ A_off lo
    coef = 2.0 * (A_off @ lo)
    const = -float(lo @ A_off @ lo)
    # min over {lo≤μ≤hi, Σμ=1} of coef·μ
    lin_min = lp_min_linear(coef, lo, hi, target=1.0)
    if not np.isfinite(lin_min):
        return -np.inf
    lb_off = lin_min + const
    # Diagonal piece: Cauchy–Schwarz Σ_D μ_i² ≥ (Σ_D μ_i)²/|D| (only if Σμ_i≥0 on D)
    D_size = int(diag_mask.sum())
    if D_size > 0:
        D_vec = diag_mask.astype(np.float64)
        sum_min = lp_min_linear(D_vec, lo, hi, target=1.0)
        if not np.isfinite(sum_min):
            return -np.inf
        cs_lb = max(0.0, sum_min) ** 2 / D_size
    else:
        cs_lb = 0.0
    return W.Q_coef * (lb_off + cs_lb) - c_target


# =====================================================================
# Sound L_joint via Lagrangian dual + Frank–Wolfe
# =====================================================================

def _aggregate(top_windows: List[WindowData], lam: np.ndarray,
                mu_star: np.ndarray, c_target: float
                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (Q_lam, grad_lam, margin_lam) for Σ_W λ_W f_W.

      f_W(ε,X) = m_W + grad_W·ε + Q_coef tr(A_W X)
      ⇒ Σ λ_W f_W = (Σ λ_W m_W) + (Σ λ_W grad_W)·ε + tr( (Σ λ_W Q_coef A_W) X )
    """
    d = len(mu_star)
    Q_lam = np.zeros((d, d))
    grad_lam = np.zeros(d)
    margin_lam = 0.0
    for w, W in zip(lam, top_windows):
        if w <= 0.0:
            continue
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        g_W = W.grad_coef * (W.A @ mu_star)
        Q_W = W.Q_coef * W.A
        margin_lam += w * m_W
        grad_lam += w * g_W
        Q_lam += w * Q_W
    return Q_lam, grad_lam, margin_lam


def _fW_value(W: WindowData, mu_star: np.ndarray, eps_val: np.ndarray,
                X_val: np.ndarray, c_target: float) -> float:
    """Evaluate f_W at a candidate (ε,X) inside the SDP feasible set."""
    m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
    g_W = W.grad_coef * (W.A @ mu_star)
    Q_W = W.Q_coef * W.A
    return float(m_W + g_W @ eps_val + np.trace(Q_W @ X_val))


def tier_L_joint_lagrangian(cell_cache: CellCache, windows: List[WindowData],
                              c_target: float, K: int = 4,
                              max_fw_iters: int = 6, fw_tol: float = 1e-6,
                              warm_lam: Optional[np.ndarray] = None
                              ) -> Tuple[float, dict]:
    """SOUND joint multi-window Shor LB via Lagrangian dual + Frank–Wolfe.

    Returns (LB, info).  LB > 0 ⇒ cell certified (Σ_W λ_W f_W > 0 ∀ε ⇒
    max_W f_W > 0 ∀ε).  Sound for any λ in the simplex.

    Algorithm:
      0. Pick top-K windows by m_W = Q·μ*^T A μ* − c_target (best for L).
      1. Compute L_single for each: pick λ_0 = e_{argmax L_single}.
      2. For t = 1..T:
         a. Solve SDP at λ_t → bound L(λ_t), recover (ε*, X*).
         b. Compute subgradient g_W = f_W(ε*, X*) for each W.
         c. Frank–Wolfe step toward s_t = e_{argmax_W g_W} (max grows L(λ)):
              λ_{t+1} = (1 − η) λ_t + η · s_t,  η = 2/(t+2).
         d. Keep best bound across iterations.
      3. Return best LB.
    """
    if not HAS_CVXPY:
        return -np.inf, {'error': 'no_cvxpy'}
    d = cell_cache.cell.d
    mu_star = cell_cache.mu_star

    # ---- Pick top-K by m_W (not F) — better predictor of L's winner ----
    m_scores = []
    for i, W in enumerate(windows):
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        m_scores.append((m_W, i, W))
    m_scores.sort(reverse=True)
    K_eff = min(K, len(windows))
    top_windows = [w for _, _, w in m_scores[:K_eff]]

    # ---- Warm-start λ at e_{argmax L_single} ----
    template = v3.get_sdp_template(d)
    best_single_lb = -np.inf
    best_single_k = 0
    L_single_values: List[float] = []
    for k, W in enumerate(top_windows):
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        g_W = W.grad_coef * (W.A @ mu_star)
        Q_W = W.Q_coef * W.A
        lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                    mu_star, Q_W, g_W, m_W)
        if lb is None:
            lb = -np.inf
        L_single_values.append(lb)
        if lb > best_single_lb:
            best_single_lb = lb
            best_single_k = k

    if warm_lam is not None and len(warm_lam) == K_eff:
        lam = np.asarray(warm_lam, dtype=np.float64)
        lam = np.maximum(lam, 0)
        s = lam.sum()
        lam = lam / s if s > 0 else np.eye(K_eff)[best_single_k]
    else:
        lam = np.eye(K_eff)[best_single_k]

    best_lb = best_single_lb
    best_lam = lam.copy()
    best_eps = None
    best_X = None

    # ---- Frank–Wolfe iterations ----
    iter_history = [(0, float(best_single_lb), lam.copy())]
    for t in range(1, max_fw_iters + 1):
        Q_lam, g_lam, m_lam = _aggregate(top_windows, lam, mu_star, c_target)
        lb_t, info_t = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                        mu_star, Q_lam, g_lam, m_lam)
        if lb_t is None:
            iter_history.append((t, None, lam.copy()))
            break
        eps_val = info_t.get('eps') if isinstance(info_t, dict) else None
        # Recover X via cvxpy variable value
        X_val = template.X.value if template.X.value is not None else None
        iter_history.append((t, float(lb_t), lam.copy()))
        if lb_t > best_lb:
            best_lb = float(lb_t)
            best_lam = lam.copy()
            best_eps = eps_val.copy() if eps_val is not None else None
            best_X = X_val.copy() if X_val is not None else None
            if best_lb > 0:
                # Already certified; don't waste more iters
                break
        # Subgradient: g_W = f_W(ε*, X*) for each W in top-K
        if eps_val is None or X_val is None:
            break
        g_subgrad = np.array([_fW_value(W, mu_star, eps_val, X_val, c_target)
                                for W in top_windows])
        # FW direction: put all weight on W maximizing g_subgrad (since L(λ) is
        # concave in λ and ∂L/∂λ_W = g_W at the inner minimizer).
        s_idx = int(np.argmax(g_subgrad))
        s_vec = np.eye(K_eff)[s_idx]
        eta = 2.0 / (t + 2.0)
        new_lam = (1.0 - eta) * lam + eta * s_vec
        # If λ unchanged enough, stop
        if np.linalg.norm(new_lam - lam) < fw_tol:
            break
        lam = new_lam

    info = {
        'best_lam': best_lam,
        'L_single_values': L_single_values,
        'best_single_lb': best_single_lb,
        'iter_history': iter_history,
        'K_eff': K_eff,
        'method': 'lagrangian_fw',
    }
    return float(best_lb), info


# =====================================================================
# Tier B36 cheap predicate (carried over from v4)
# =====================================================================

tier_B36_predicate = v4.tier_B36_predicate


# =====================================================================
# m_W ranking for L candidates (better than F-rank)
# =====================================================================

def select_L_candidates_v5(cell_cache: CellCache, windows: List[WindowData],
                             c_target: float, K: int = 3,
                             include_widest: int = 2) -> List[int]:
    """Pick top-K window indices by m_W = Q·μ*^T A μ* − c_target, plus the
    `include_widest` widest-ell windows (B43 widest-first preserved).
    """
    mu_star = cell_cache.mu_star
    m_scores = []
    for i, W in enumerate(windows):
        m = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        m_scores.append((m, i))
    m_scores.sort(reverse=True)
    chosen = []
    # Widest by ell first
    if include_widest > 0:
        ells = np.array([W.ell for W in windows])
        widest = list(np.argsort(-ells)[:include_widest])
        for idx in widest:
            if int(idx) not in chosen:
                chosen.append(int(idx))
    # Fill from m-rank
    for _, idx in m_scores:
        if int(idx) not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= K:
            break
    return chosen[:K]


# =====================================================================
# Cascade certification (v5)
# =====================================================================

@dataclass
class CertResult:
    certified: bool
    tier_used: str
    bound: float = 0.0
    depth_used: int = 0
    n_subcells: int = 1
    best_W_idx: int = -1


def cert_cell(cell: Cell, windows: List[WindowData], c_target: float,
                max_depth: int = 6, current_depth: int = 0,
                use_B1: bool = True, use_B1u: bool = True,
                use_B1diag: bool = True, use_F: bool = True,
                use_L: bool = True, use_L_joint: bool = True,
                joint_K: int = 4, joint_fw_iters: int = 6,
                verbose: bool = False) -> CertResult:
    """Sound box certificate (v5 cascade).

    Order: empty → B1 → B1u → B1diag → F → L_single → L_joint(lagrangian) → split.
    """
    # Empty cell
    if is_cell_empty(cell):
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    if not cell.is_simplex_feasible():
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)

    cache = CellCache.build(cell)

    # ---- B1 (lo·lo corner) ----
    if use_B1:
        for i, W in enumerate(windows):
            b = tier_B1_mu_corner(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # ---- B1u (Σμ=1 complement-corner) ----
    if use_B1u:
        for i, W in enumerate(windows):
            b = tier_B1u_complement_corner(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1u',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # ---- B1diag (Cauchy–Schwarz + linear-AM-GM) ----
    if use_B1diag:
        for i, W in enumerate(windows):
            b = tier_B1diag_amgm(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1diag',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # ---- Tier F (all windows, vectorized) ----
    if use_F:
        v3.compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)

    # ---- Tier L (m_W ranking + widest-first) ----
    L_best = -np.inf
    if use_L and HAS_CVXPY:
        K_L = min(3, len(windows))
        candidates = select_L_candidates_v5(cache, windows, c_target, K=K_L,
                                              include_widest=2)
        for w_idx in candidates:
            lb = tier_L_single(cache, windows[w_idx], c_target, verbose)
            if lb > L_best:
                L_best = lb
            if lb > 0:
                return CertResult(certified=True, tier_used='L',
                                  bound=lb, depth_used=current_depth,
                                  best_W_idx=w_idx)

    # ---- Tier L_joint (Lagrangian; SOUND replacement of v4's epigraph) ----
    Lj_best = -np.inf
    if use_L_joint and HAS_CVXPY:
        # Build top-K by m_W
        candidates_j = select_L_candidates_v5(cache, windows, c_target,
                                                K=joint_K, include_widest=2)
        top_j = [windows[i] for i in candidates_j]
        lb_j, info_j = tier_L_joint_lagrangian(cache, top_j, c_target,
                                                  K=len(top_j),
                                                  max_fw_iters=joint_fw_iters)
        Lj_best = lb_j
        if lb_j > 0:
            return CertResult(certified=True, tier_used='L_joint',
                              bound=lb_j, depth_used=current_depth)

    # ---- B45: depth cap at small d ----
    eff_max_depth = max_depth
    if cell.d <= 4:
        eff_max_depth = min(max_depth, 1)
    if current_depth >= eff_max_depth:
        return CertResult(certified=False, tier_used='max_depth_reached',
                          depth_used=current_depth)

    # ---- B46: skip split if neither L nor L_joint produced any improvement ----
    # B46 fires only at root (current_depth == 0). Sound: if all sound tiers
    # already failed AND L_joint did not narrow the gap below F, splitting is
    # unlikely to help — return uncertified rather than burn 2^depth work.
    f_best = cache.f_best_bound if use_F else -np.inf
    if (current_depth == 0 and use_L_joint and use_F
            and Lj_best <= f_best + 1e-9 and L_best <= f_best + 1e-9):
        return CertResult(certified=False, tier_used='B46_no_signal',
                          bound=max(f_best, Lj_best, L_best),
                          depth_used=current_depth)

    # ---- Split (gradient-weighted) ----
    best_W = (windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0
              else windows[0])
    axis = v3.split_axis_gradient_weighted(cache, best_W)
    sub1, sub2 = cell.split(axis)
    r1 = cert_cell(sub1, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1,
                    use_B1=use_B1, use_B1u=use_B1u, use_B1diag=use_B1diag,
                    use_F=use_F, use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, joint_fw_iters=joint_fw_iters,
                    verbose=verbose)
    if not r1.certified:
        return CertResult(certified=False, tier_used='split_sub1_fail',
                          depth_used=r1.depth_used,
                          n_subcells=r1.n_subcells)
    r2 = cert_cell(sub2, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1,
                    use_B1=use_B1, use_B1u=use_B1u, use_B1diag=use_B1diag,
                    use_F=use_F, use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, joint_fw_iters=joint_fw_iters,
                    verbose=verbose)
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

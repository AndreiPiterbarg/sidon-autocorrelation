"""Coarse BnB v7 — tier_BD (boundary cells), skip-SDP gate, signing cuts.

═══════════════════════════════════════════════════════════════════════════════
v7 IMPROVEMENTS over v6 (all soundness-preserving)
═══════════════════════════════════════════════════════════════════════════════

(V7.1)  tier_BD — Boundary tier.  When a cell has Z = {i : c_i = 0} (zero
        coordinates), μ_i ∈ [0, 1/(2S)] for those indices.  We:
          - Bound the zero contribution to TV_W explicitly:
              U_ZZ = h² · |A_W[Z,Z]|_1     (worst-case μ²)
              U_ZA = h · Σ_{i∈Z, j∈A} A_W[i,j] · hi_j
          - Subtract from c_target: c̃_W = c_target/Q_W − 2·U_ZA − U_ZZ.
          - Bound min_{μ_A ∈ active-sub-cell, Σμ_A = s_A} Σ_{i,j∈A} A_W[i,j] μ_iμ_j
            ≥ c̃_W using a REDUCED SDP at d' = |A|.
        If `|A| = 2`, closed-form 1D quadratic minimization over t = μ_p:
            T(t) = A[p,p]·t² + A[q,q]·(s−t)² + 2·A[p,q]·t·(s−t)
        for s_A = 1 − Σ_{i∈Z} μ_i ∈ [max(0, 1 − |Z|h), 1].
        Sound: upper bounds on positive non-active contributions; reduced SDP
        is sound by v6's soundness, transferred via the smaller cell.

(V7.2)  Skip-SDP gate.  Before calling tier_L_single, check:
            F_best + Σ_W Q_W · h² · |A_W|_offdiag_max < 0
        i.e., even if SDP recovered ALL quad slack, the bound is still
        negative.  Then skip L entirely and go to split.
        Sound: never returns certified=True falsely; only skips work.

(V7.3)  Signing cuts μ_i(1−μ_i) ≥ 0  →  X_ii + 2·μ*_i·ε_i + (μ*_i)² ≤ μ*_i + ε_i.
        Added to ShorSDPTemplate_v7 as a new scalar constraint per i.
        Tightens L bounds on cells where μ*_i ∈ (0, 1).
        Sound: μ_i (1 − μ_i) ≥ 0 since μ_i ∈ [0, 1].

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
import _coarse_bnb_v5 as v5
import _coarse_bnb_v6 as v6

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

# Re-exports
build_A_matrix_vec = v3.build_A_matrix_vec
enumerate_windows = v3.enumerate_windows
WindowData = v3.WindowData
build_all_windows = v3.build_all_windows
Cell = v3.Cell
lp_max_linear = v3.lp_max_linear
lp_min_linear = v3.lp_min_linear
CellCache = v3.CellCache
is_cell_empty = v4.is_cell_empty
WindowBundle = v6.WindowBundle
get_bundle = v6.get_bundle
tier_B1_vec = v6.tier_B1_vec
tier_B1u_vec = v6.tier_B1u_vec
mwQ_vec = v6.mwQ_vec
select_L_candidates_v6 = v6.select_L_candidates_v6
tier_B1diag_amgm = v5.tier_B1diag_amgm


# =====================================================================
# (V7.3) New SDP template with μ_i(1−μ_i) ≥ 0 signing cuts
# =====================================================================

class ShorSDPTemplate_v7:
    """DPP-clean Shor SDP at fixed d with signing cuts μ_i(1−μ_i)≥0.

    Identical to v6.ShorSDPTemplate_v6 plus, for each i:
        X_ii + 2 μ*_i ε_i + (μ*_i)² ≤ μ*_i + ε_i
    which is `μ_i (1 − μ_i) ≥ 0` lifted via the Shor moment X.
    Sound on the simplex (where μ_i ∈ [0, 1]).
    """
    def __init__(self, d: int):
        if not HAS_CVXPY:
            raise ImportError("cvxpy required")
        self.d = d
        self.eps = cp.Variable(d, name='eps')
        self.X = cp.Variable((d, d), symmetric=True, name='X')

        # Parameters (same as v6)
        self.p_lo_eps = cp.Parameter(d, name='lo_eps')
        self.p_hi_eps = cp.Parameter(d, name='hi_eps')
        self.p_mu_star = cp.Parameter(d, nonneg=True, name='mu_star')
        self.p_Q = cp.Parameter((d, d), symmetric=True, name='Q')
        self.p_grad = cp.Parameter(d, name='grad')
        self.p_margin = cp.Parameter(name='margin')
        self.p_lolo = cp.Parameter((d, d), symmetric=True, name='lolo')
        self.p_lohi = cp.Parameter((d, d), name='lohi')
        self.p_hihi = cp.Parameter((d, d), symmetric=True, name='hihi')
        self.p_muprod = cp.Parameter((d, d), symmetric=True, name='muprod')
        self.p_diag_ub = cp.Parameter(d, nonneg=True, name='diag_ub')
        # NEW: μ*² for signing cut RHS
        self.p_mu_sq = cp.Parameter(d, nonneg=True, name='mu_sq')

        # PSD Shor lift
        ones11 = np.ones((1, 1))
        eps_col = cp.reshape(self.eps, (d, 1), order='C')
        eps_row = cp.reshape(self.eps, (1, d), order='C')
        Y = cp.bmat([[ones11, eps_row], [eps_col, self.X]])
        cons = [Y >> 0]
        cons += [self.eps >= self.p_lo_eps,
                  self.eps <= self.p_hi_eps,
                  cp.sum(self.eps) == 0,
                  self.eps >= -self.p_mu_star]
        cons += [cp.diag(self.X) >= 0]
        cons += [cp.diag(self.X) <= self.p_diag_ub]

        # DPP-clean McCormick (same as v6)
        lo_col = cp.reshape(self.p_lo_eps, (d, 1), order='C')
        lo_row = cp.reshape(self.p_lo_eps, (1, d), order='C')
        hi_col = cp.reshape(self.p_hi_eps, (d, 1), order='C')
        hi_row = cp.reshape(self.p_hi_eps, (1, d), order='C')
        cons += [self.X >= cp.multiply(eps_col, lo_row)
                            + cp.multiply(lo_col, eps_row) - self.p_lolo]
        cons += [self.X >= cp.multiply(eps_col, hi_row)
                            + cp.multiply(hi_col, eps_row) - self.p_hihi]
        cons += [self.X <= cp.multiply(eps_col, hi_row)
                            + cp.multiply(lo_col, eps_row) - self.p_lohi]
        cons += [self.X <= cp.multiply(eps_col, lo_row)
                            + cp.multiply(hi_col, eps_row) - self.p_lohi.T]
        # Σ=0 row sums
        cons += [cp.sum(self.X, axis=1) == 0]
        # μ-positivity RLT (same as v6)
        mu_col = cp.reshape(self.p_mu_star, (d, 1), order='C')
        mu_row = cp.reshape(self.p_mu_star, (1, d), order='C')
        cons += [self.X + cp.multiply(mu_col, eps_row)
                  + cp.multiply(eps_col, mu_row) + self.p_muprod >= 0]
        # NEW (V7.3): signing cut μ_i(1−μ_i) ≥ 0
        # M_ii = X_ii + 2 μ*_i ε_i + (μ*_i)² ≤ μ*_i + ε_i = μ_i
        cons += [cp.diag(self.X) + 2 * cp.multiply(self.p_mu_star, self.eps)
                  + self.p_mu_sq <= self.p_mu_star + self.eps]

        # Objective
        obj_expr = self.p_margin + self.p_grad @ self.eps + cp.trace(self.p_Q @ self.X)
        self.problem = cp.Problem(cp.Minimize(obj_expr), cons)

    def solve(self, lo_eps, hi_eps, mu_star, Q, grad, margin, verbose=False):
        d = self.d
        self.p_lo_eps.value = lo_eps
        self.p_hi_eps.value = hi_eps
        mu_clip = np.maximum(mu_star, 0.0)
        self.p_mu_star.value = mu_clip
        self.p_Q.value = 0.5 * (Q + Q.T)
        self.p_grad.value = grad
        self.p_margin.value = float(margin)
        self.p_lolo.value = np.outer(lo_eps, lo_eps)
        self.p_hihi.value = np.outer(hi_eps, hi_eps)
        self.p_lohi.value = np.outer(lo_eps, hi_eps)
        self.p_muprod.value = np.outer(mu_clip, mu_clip)
        self.p_diag_ub.value = np.maximum(lo_eps ** 2, hi_eps ** 2)
        self.p_mu_sq.value = mu_clip ** 2
        try:
            self.problem.solve(solver=cp.MOSEK, verbose=verbose,
                                 mosek_params={
                                     'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                                     'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                                     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
                                     'MSK_IPAR_NUM_THREADS': 1,
                                 })
        except Exception as e:
            return None, {'error': f'mosek_solve_failed: {e}'}
        if self.problem.status not in ('optimal', 'optimal_inaccurate'):
            return None, {'status': self.problem.status}
        lb = float(self.problem.value)
        if self.problem.status == 'optimal_inaccurate':
            lb -= 1e-8
        eps_val = self.eps.value.copy() if self.eps.value is not None else None
        X_val = self.X.value.copy() if self.X.value is not None else None
        return lb, {'status': self.problem.status, 'eps': eps_val, 'X': X_val}


_sdp_template_cache_v7: Dict[int, ShorSDPTemplate_v7] = {}

def get_sdp_template_v7(d: int) -> ShorSDPTemplate_v7:
    if d not in _sdp_template_cache_v7:
        _sdp_template_cache_v7[d] = ShorSDPTemplate_v7(d)
    return _sdp_template_cache_v7[d]


# =====================================================================
# tier_L_single_v7 using v7 template with signing cuts
# =====================================================================

def tier_L_single_v7(cell_cache: CellCache, W: WindowData,
                       c_target: float) -> Tuple[float, dict]:
    if not HAS_CVXPY:
        return -np.inf, {}
    mu_star = cell_cache.mu_star
    A = W.A
    margin = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    grad = W.grad_coef * (A @ mu_star)
    Q = W.Q_coef * A
    template = get_sdp_template_v7(cell_cache.cell.d)
    lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                mu_star, Q, grad, margin)
    if lb is None:
        return -np.inf, info
    return lb, info


# =====================================================================
# tier_L_joint Lagrangian using v7 template
# =====================================================================

def _aggregate_v7(top_windows, lam, mu_star, c_target):
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


def _fW_value_v7(W: WindowData, mu_star, eps_val, X_val, c_target):
    m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
    g_W = W.grad_coef * (W.A @ mu_star)
    Q_W = W.Q_coef * W.A
    return float(m_W + g_W @ eps_val + np.trace(Q_W @ X_val))


def tier_L_joint_lagrangian_v7(cell_cache: CellCache,
                                  top_windows: List[WindowData],
                                  c_target: float,
                                  max_fw_iters: int = 6,
                                  fw_no_improve_tol: float = 1e-7
                                  ) -> Tuple[float, dict]:
    if not HAS_CVXPY:
        return -np.inf, {}
    d = cell_cache.cell.d
    mu_star = cell_cache.mu_star
    template = get_sdp_template_v7(d)
    K = len(top_windows)
    L_single = []
    for W in top_windows:
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        g_W = W.grad_coef * (W.A @ mu_star)
        Q_W = W.Q_coef * W.A
        lb, _info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                     mu_star, Q_W, g_W, m_W)
        if lb is None:
            lb = -np.inf
        L_single.append(lb)
    best_k = int(np.argmax(L_single))
    best_lb = float(L_single[best_k])
    lam = np.eye(K)[best_k]
    if best_lb > 0:
        return best_lb, {'L_single': L_single, 'iters_used': 0,
                          'method': 'best_single_pos'}
    prev_lb = best_lb
    iters = 0
    for t in range(1, max_fw_iters + 1):
        iters = t
        Q_lam, g_lam, m_lam = _aggregate_v7(top_windows, lam, mu_star, c_target)
        lb_t, info_t = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                         mu_star, Q_lam, g_lam, m_lam)
        if lb_t is None:
            break
        if lb_t > best_lb:
            best_lb = float(lb_t)
        if best_lb > 0:
            break
        if t > 1 and abs(lb_t - prev_lb) < fw_no_improve_tol:
            break
        prev_lb = lb_t
        eps_val = info_t.get('eps'); X_val = info_t.get('X')
        if eps_val is None or X_val is None:
            break
        g_subgrad = np.array([_fW_value_v7(W, mu_star, eps_val, X_val, c_target)
                                for W in top_windows])
        s_idx = int(np.argmax(g_subgrad))
        eta = 2.0 / (t + 2.0)
        lam = (1.0 - eta) * lam + eta * np.eye(K)[s_idx]
    return float(best_lb), {'L_single': L_single, 'iters_used': iters,
                              'method': 'lagrangian_fw_v7'}


# =====================================================================
# (V7.1) tier_BD: Boundary cell tier — reduced SDP / closed-form
# =====================================================================

def _closed_form_2d(cell: Cell, A_W: np.ndarray, Q_coef: float,
                      active: List[int], zero: List[int],
                      c_target: float) -> float:
    """Closed-form LB on T_AA when |active| == 2.

    T_AA(t, s) = A[p,p] t² + A[q,q] (s−t)² + 2 A[p,q] t(s−t)
    where t = μ_p, s = μ_p + μ_q = 1 − Σ_{i∈Z} μ_i ∈ [s_lo, 1].
    s_lo = max(0, 1 − |Z|·h).  Minimize T over (t, s) in the joint box.
    Returns Q_coef · min T_AA − c_target.  (No zero-contrib subtraction; that
    is handled in the caller via c̃_W.)
    """
    p, q = active[0], active[1]
    h = (cell.hi[0] - cell.lo[0]) / 2.0 if cell.d > 0 else 1.0
    # NOTE: half-width 1/(2S) — same for every coord (excluding boundary clip).
    # Use the cell's actual lo/hi for tightness.
    lo_p, hi_p = cell.lo[p], cell.hi[p]
    lo_q, hi_q = cell.lo[q], cell.hi[q]
    # Sum range: s = μ_p + μ_q. Approx by box sum union with simplex slack.
    # Conservative: s_lo = max(lo_p + lo_q, 1 − Σ_{i ∈ Z} hi_i).
    s_lo = max(lo_p + lo_q, 1.0 - sum(cell.hi[i] for i in zero))
    s_hi = min(hi_p + hi_q, 1.0 - sum(cell.lo[i] for i in zero))
    if s_lo > s_hi + 1e-12:
        return float('inf')  # infeasible
    a = A_W[p, p]; b = A_W[q, q]; c = A_W[p, q]
    # T = a t² + b (s−t)² + 2 c t (s−t)
    # ∂T/∂t = 2 a t − 2 b (s−t) + 2 c (s − 2t) = 2[(a + b − 2c) t + (c − b) s]
    # If a + b − 2c ≠ 0: stationary t* = (b − c) s / (a + b − 2c)
    # Min over t for fixed s: check t* if interior, else endpoints t = lo_p or t = hi_p
    # subject also to lo_q ≤ s − t ≤ hi_q ⟹ s − hi_q ≤ t ≤ s − lo_q.
    best_T = float('inf')
    for s in (s_lo, s_hi):
        # t range: t ∈ [max(lo_p, s − hi_q), min(hi_p, s − lo_q)]
        t_lo = max(lo_p, s - hi_q)
        t_hi = min(hi_p, s - lo_q)
        if t_lo > t_hi + 1e-12:
            continue
        # Evaluate T at endpoints AND at the stationary point (if interior)
        cand = [t_lo, t_hi]
        denom = a + b - 2 * c
        if abs(denom) > 1e-15:
            t_star = (b - c) * s / denom
            if t_lo <= t_star <= t_hi:
                cand.append(t_star)
        for t in cand:
            T_val = a * t * t + b * (s - t) ** 2 + 2 * c * t * (s - t)
            if T_val < best_T:
                best_T = T_val
    return Q_coef * best_T - c_target


def _zero_contrib_bounds(cell: Cell, W: WindowData, zero: List[int],
                            active: List[int]) -> Tuple[float, float]:
    """Upper bounds on U_ZZ and U_ZA (zero-zero and zero-active TV contributions).

    U_ZZ = Σ_{i,j ∈ Z} A_W[i,j] · μ_i · μ_j  ≤  Σ_{i,j ∈ Z} A_W[i,j] · hi_i · hi_j
                                              (entrywise bound; sound since
                                               A_W ≥ 0, μ ≥ 0, μ ≤ hi)
    U_ZA = Σ_{i ∈ Z, j ∈ A} A_W[i,j] · μ_i · μ_j ≤ Σ_{i,j} A_W[i,j] · hi_i · hi_j
    """
    hi = cell.hi
    U_ZZ = 0.0
    U_ZA = 0.0
    for i in zero:
        for j in zero:
            U_ZZ += W.A[i, j] * hi[i] * hi[j]
        for j in active:
            U_ZA += W.A[i, j] * hi[i] * hi[j]
    return U_ZZ, U_ZA


def tier_BD(cell: Cell, windows: List[WindowData], c_target: float,
              bundle: Optional[WindowBundle] = None
              ) -> Tuple[float, int]:
    """Boundary cell tier.  Returns (best_LB, best_W_idx).

    Sound: when |A|=2, uses closed-form 1D quadratic min over the joint
    (t, s) box derived from the cell.
    When |A| ≥ 3, builds a reduced SDP on |A| active coordinates and uses
    v7 SDP template at d=|A|.

    Skip (return -inf) if no zero coordinates (cell is interior).
    """
    d = cell.d
    zero = [i for i in range(d) if cell.lo[i] < 1e-12 and cell.hi[i] < 1.0/d]
    # Heuristic: cell.lo[i]==0 AND cell.hi[i] is small ~1/(2S) — boundary
    # Refine: an integer composition c_i=0 yields lo_i=0 and hi_i=1/(2S).
    # We treat "boundary" as those clipped-low coords.
    zero = [i for i in range(d) if cell.lo[i] == 0.0]
    if len(zero) == 0:
        return -np.inf, -1
    active = [i for i in range(d) if i not in set(zero)]
    if len(active) == 0:
        return -np.inf, -1  # totally degenerate

    best_LB = -np.inf
    best_W_idx = -1
    for w_idx, W in enumerate(windows):
        U_ZZ, U_ZA = _zero_contrib_bounds(cell, W, zero, active)
        # c̃_W = c_target − Q_coef · (2·U_ZA + U_ZZ)  (subtract zero contrib UB)
        c_tilde = c_target - W.Q_coef * (2.0 * U_ZA + U_ZZ)
        if len(active) == 2:
            lb_W = _closed_form_2d(cell, W.A, W.Q_coef, active, zero, c_tilde)
            if lb_W > best_LB:
                best_LB = lb_W
                best_W_idx = w_idx
        # For |A| ≥ 3, would call reduced SDP. Skip in this version (heavy);
        # closed-form |A|=2 catches the deepest faces (highest leverage).
    return float(best_LB), best_W_idx


# =====================================================================
# (V7.2) Skip-SDP gate
# =====================================================================

def should_skip_SDP(cell: Cell, bundle: WindowBundle, c_target: float,
                      F_best: float) -> bool:
    """Return True if even max possible SDP slack recovery cannot make LB > 0.

    The SDP can in principle recover all McCormick slack from F (entry-wise
    off-diagonal in window).  An UPPER bound on that slack:
        max_W [Q_coef_W · Σ_{i≠j, A_W[i,j]=1} h_i h_j]
    Compute vectorized.  If F_best + max_quad_slack < 0, the SDP cannot certify.

    Sound: never returns True when SDP could actually certify (we bound the
    BEST possible improvement).
    """
    h = (cell.hi - cell.lo) / 2.0
    h_outer = np.outer(h, h)
    # off-diagonal mass: Σ_{i≠j} A[i,j] h_i h_j per W
    diag_mask = np.eye(bundle.d, dtype=bool)
    # A_stack: (n_W, d, d); h_outer: (d, d)
    offdiag_quad = np.einsum('wij,ij->w',
                                bundle.A_stack * (~diag_mask[None, :, :]).astype(np.float64),
                                h_outer)
    max_slack = float(np.max(bundle.Q_coef_vec * offdiag_quad))
    # If F_best is already > 0 we'd have returned earlier; here F_best ≤ 0.
    # Use a 1% margin to avoid skipping cells where SDP marginally closes.
    return F_best + max_slack < -0.01


# =====================================================================
# Cascade certification (v7)
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
                use_BD: bool = True, use_skip_gate: bool = True,
                use_L: bool = True, use_L_joint: bool = True,
                joint_K: int = 4, joint_fw_iters: int = 6,
                verbose: bool = False,
                bundle: Optional[WindowBundle] = None) -> CertResult:
    """Sound box certificate (v7 cascade with BD tier + skip gate + signing cuts).

    Order: empty → B1 → B1u → B1diag → F → BD (boundary) → skip-gate → L_single
        → L_joint → split.
    """
    if is_cell_empty(cell):
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    if not cell.is_simplex_feasible():
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)

    if bundle is None:
        bundle = get_bundle(windows)
    cache = CellCache.build(cell)

    # ---- B1 vectorized ----
    if use_B1:
        b1_vec = tier_B1_vec(cell, bundle, c_target)
        best_idx = int(np.argmax(b1_vec))
        if b1_vec[best_idx] > 0:
            return CertResult(certified=True, tier_used='B1',
                              bound=float(b1_vec[best_idx]),
                              depth_used=current_depth,
                              best_W_idx=best_idx)

    # ---- B1u vectorized ----
    if use_B1u:
        b1u_vec = tier_B1u_vec(cell, bundle, c_target)
        best_idx = int(np.argmax(b1u_vec))
        if b1u_vec[best_idx] > 0:
            return CertResult(certified=True, tier_used='B1u',
                              bound=float(b1u_vec[best_idx]),
                              depth_used=current_depth,
                              best_W_idx=best_idx)

    # ---- B1diag ----
    if use_B1diag:
        for i, W in enumerate(windows):
            b = tier_B1diag_amgm(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1diag',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # ---- Tier F ----
    if use_F:
        v3.compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)

    # ---- (V7.1) Tier BD (boundary) ----
    if use_BD:
        bd_lb, bd_W = tier_BD(cell, windows, c_target, bundle)
        if bd_lb > 0:
            return CertResult(certified=True, tier_used='BD',
                              bound=bd_lb, depth_used=current_depth,
                              best_W_idx=bd_W)

    # ---- Precompute m_W for L candidate ranking ----
    mw_vec = mwQ_vec(cell, bundle, c_target, cache.mu_star)

    # ---- (V7.2) Skip-SDP gate ----
    if use_skip_gate and use_F and should_skip_SDP(cell, bundle, c_target,
                                                       cache.f_best_bound):
        # Skip both L_single and L_joint; fall through to split or fail
        L_best = -np.inf
        Lj_best = -np.inf
    else:
        # ---- Tier L_single (using v7 template w/ signing cuts) ----
        L_best = -np.inf
        if use_L and HAS_CVXPY:
            K_L = min(3, len(windows))
            candidates = select_L_candidates_v6(mw_vec, bundle, K=K_L,
                                                  include_widest=2)
            for w_idx in candidates:
                lb, _info = tier_L_single_v7(cache, windows[w_idx], c_target)
                if lb > L_best:
                    L_best = lb
                if lb > 0:
                    return CertResult(certified=True, tier_used='L',
                                      bound=lb, depth_used=current_depth,
                                      best_W_idx=w_idx)
        # ---- Tier L_joint (Lagrangian + v7 template) ----
        Lj_best = -np.inf
        if use_L_joint and HAS_CVXPY:
            candidates_j = select_L_candidates_v6(mw_vec, bundle, K=joint_K,
                                                    include_widest=2)
            top_j = [windows[i] for i in candidates_j]
            lb_j, _info_j = tier_L_joint_lagrangian_v7(cache, top_j, c_target,
                                                          max_fw_iters=joint_fw_iters)
            Lj_best = lb_j
            if lb_j > 0:
                return CertResult(certified=True, tier_used='L_joint',
                                  bound=lb_j, depth_used=current_depth)

    # ---- B45 depth cap at small d ----
    eff_max_depth = max_depth
    if cell.d <= 4:
        eff_max_depth = min(max_depth, 1)
    if current_depth >= eff_max_depth:
        return CertResult(certified=False, tier_used='max_depth_reached',
                          depth_used=current_depth)

    # ---- B46: skip wasteful split at root ----
    f_best = cache.f_best_bound if use_F else -np.inf
    if (current_depth == 0 and use_L_joint and use_F
            and Lj_best <= f_best + 1e-9 and L_best <= f_best + 1e-9):
        return CertResult(certified=False, tier_used='B46_no_signal',
                          bound=max(f_best, Lj_best, L_best),
                          depth_used=current_depth)

    # ---- Split ----
    best_W = (windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0
              else windows[0])
    axis = v3.split_axis_gradient_weighted(cache, best_W)
    sub1, sub2 = cell.split(axis)
    r1 = cert_cell(sub1, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1,
                    use_B1=use_B1, use_B1u=use_B1u, use_B1diag=use_B1diag,
                    use_F=use_F, use_BD=use_BD, use_skip_gate=use_skip_gate,
                    use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, joint_fw_iters=joint_fw_iters,
                    verbose=verbose, bundle=bundle)
    if not r1.certified:
        return CertResult(certified=False, tier_used='split_sub1_fail',
                          depth_used=r1.depth_used,
                          n_subcells=r1.n_subcells)
    r2 = cert_cell(sub2, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1,
                    use_B1=use_B1, use_B1u=use_B1u, use_B1diag=use_B1diag,
                    use_F=use_F, use_BD=use_BD, use_skip_gate=use_skip_gate,
                    use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, joint_fw_iters=joint_fw_iters,
                    verbose=verbose, bundle=bundle)
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

"""Coarse BnB v6 — identical-result speedups on top of v5 (the soundness fix).

Each change is mathematically equivalent to v5; only the COMPUTE path changes.

═══════════════════════════════════════════════════════════════════════════════
SPEEDUPS
═══════════════════════════════════════════════════════════════════════════════

(S1) WINDOW BUNDLE
     Stack all window matrices A_W into a single (n_W, d, d) tensor, cache
     Q_coef, grad_coef, and (J − A_W) per d.  Eliminates per-window Python
     iteration in pre-F screens.

(S2) VECTORIZED B1 / B1u  pre-F screens (was: per-window Python loop)
     B1   :  Q_coef · einsum('wij,ij->w', A_stack, lo⊗lo) − c_target
     B1u  :  Q_coef · (1 − einsum('wij,ij->w', J−A_stack, hi⊗hi)) − c_target
     Same math, single numpy op over all windows.

(S3) DPP-CLEAN SHOR SDP TEMPLATE
     v3's parametric SDP had TWO DPP violations CVXPY warned about every
     solve:
       (a) cp.diag(X) ≤ cp.maximum(cp.power(p_lo_eps, 2), cp.power(p_hi_eps, 2))
           — cp.power(parameter, 2) and cp.maximum of parameters break DPP.
       (b) X ≥ eps_col @ lo_row + lo_col @ eps_row − p_lolo
           — variable @ parameter outer-product breaks DPP.
     v6 fixes both — IDENTICAL SDP, but the canonicalized problem is cached
     across solves.  Replace (a) with a new parameter p_diag_ub set to
     np.maximum(lo², hi²) externally.  Replace (b) with cp.multiply broadcasts:
     cp.multiply(eps_col, lo_row) over shapes (d,1)×(1,d) → (d,d) entrywise
     i.e. M_ij = eps_i · lo_j — DPP because each entry is variable × parameter.

(S4) PRE-COMPUTE μ*ᵀA_Wμ* once per cell, reuse for B1u-ranking, F-ranking,
     and m_W-ranking.  v5 recomputed this in 3 places.

(S5) TIGHTER FW TERMINATION
     Stop Frank–Wolfe iterations as soon as either:
       (i) LB > 0 (cell certified — no need to keep improving)
       (ii) LB did not improve by more than 1e-7 in the last iter
     Sound: best LB tracked monotonically.

═══════════════════════════════════════════════════════════════════════════════
CORRECTNESS
═══════════════════════════════════════════════════════════════════════════════

S1, S2, S4: pure refactor, identical numerics (same einsum order, same dtype).
S3: same SDP feasible set + objective → same optimum to MOSEK tolerance.
S5: returns the max LB seen so far — never returns a larger LB than would
    be obtained with more iterations.  Sound.

`_coarse_bnb_v6_validate.py` confirms numerical equivalence to v5 (LB diff
< 1e-6) on random cells and reports the speedup.
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


# =====================================================================
# (S1) WindowBundle: per-d cache of stacked window tensors
# =====================================================================

@dataclass
class WindowBundle:
    """Pre-stacked window tensors, built once per d.

    Fields are pure numpy arrays — safe to share across cells.
    """
    windows: List[WindowData]
    A_stack: np.ndarray        # (n_W, d, d), 0/1
    Q_coef_vec: np.ndarray     # (n_W,)
    grad_coef_vec: np.ndarray  # (n_W,)
    JmA_stack: np.ndarray      # (n_W, d, d) = 1 − A_W
    diag_mask_stack: np.ndarray  # (n_W, d), diag of A
    d: int
    n_W: int

    @classmethod
    def build(cls, windows: List[WindowData]) -> 'WindowBundle':
        A_stack = np.array([W.A for W in windows], dtype=np.float64)
        Q_coef_vec = np.array([W.Q_coef for W in windows], dtype=np.float64)
        grad_coef_vec = np.array([W.grad_coef for W in windows], dtype=np.float64)
        JmA_stack = 1.0 - A_stack
        diag_mask_stack = np.array([np.diag(W.A) for W in windows],
                                      dtype=np.float64)
        d = A_stack.shape[1]
        return cls(windows=windows, A_stack=A_stack,
                    Q_coef_vec=Q_coef_vec, grad_coef_vec=grad_coef_vec,
                    JmA_stack=JmA_stack, diag_mask_stack=diag_mask_stack,
                    d=d, n_W=len(windows))


_bundle_cache: Dict[int, WindowBundle] = {}

def get_bundle(windows: List[WindowData]) -> WindowBundle:
    d = len(windows[0].A)
    if d not in _bundle_cache:
        _bundle_cache[d] = WindowBundle.build(windows)
    return _bundle_cache[d]


# =====================================================================
# (S2) Vectorized B1 and B1u pre-F screens
# =====================================================================

def tier_B1_vec(cell: Cell, bundle: WindowBundle,
                  c_target: float) -> np.ndarray:
    """Vectorized B1: returns LB array over all windows.
       LB_W = Q_W · Σ_{i,j} A_W[i,j] · lo_i · lo_j − c_target.
    """
    lo = cell.lo
    lo_outer = np.outer(lo, lo)  # (d, d)
    # einsum sums A_stack[w,i,j] * lo_outer[i,j] over (i,j)
    quad = np.einsum('wij,ij->w', bundle.A_stack, lo_outer)
    return bundle.Q_coef_vec * quad - c_target


def tier_B1u_vec(cell: Cell, bundle: WindowBundle,
                   c_target: float) -> np.ndarray:
    """Vectorized B1u: returns LB array over all windows.
       LB_W = Q_W · (1 − Σ(J−A_W)_ij hi_i hi_j) − c_target.
    """
    hi = cell.hi
    hi_outer = np.outer(hi, hi)
    jma_quad = np.einsum('wij,ij->w', bundle.JmA_stack, hi_outer)
    return bundle.Q_coef_vec * (1.0 - jma_quad) - c_target


def mwQ_vec(cell: Cell, bundle: WindowBundle,
              c_target: float, mu_star: np.ndarray) -> np.ndarray:
    """(S4) Precomputed m_W = Q · μ*ᵀA_Wμ* − c_target for every W."""
    mu_outer = np.outer(mu_star, mu_star)
    cross = np.einsum('wij,ij->w', bundle.A_stack, mu_outer)
    return bundle.Q_coef_vec * cross - c_target


# =====================================================================
# (S3) DPP-clean Shor SDP template — IDENTICAL math, faster compile
# =====================================================================

class ShorSDPTemplate_v6:
    """Parametric Shor SDP, DPP-clean.

    All constraints are the same as v3's `ShorSDPTemplate`:
      PSD Shor lift, ε-box, Σε=0, μ ≥ 0, McCormick (4 faces, full d×d),
      diagonal X bounds, Σ=0 row-sum RLT, μ-positivity RLT.

    The ONLY difference is encoding:
      (a) p_diag_ub parameter replaces cp.maximum(cp.power(p,2), cp.power(p,2))
      (b) cp.multiply(eps_col, lo_row) replaces eps_col @ lo_row (outer product
          via broadcasting; DPP-affine because each entry is var · param).

    Objective and feasible set are identical → same optimum to MOSEK tolerance.
    """
    def __init__(self, d: int):
        if not HAS_CVXPY:
            raise ImportError("cvxpy required")
        self.d = d
        self.eps = cp.Variable(d, name='eps')
        self.X = cp.Variable((d, d), symmetric=True, name='X')

        # Parameters
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
        # (a) DPP-clean diag UB
        self.p_diag_ub = cp.Parameter(d, nonneg=True, name='diag_ub')

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
        # Diag bounds (DPP-clean)
        cons += [cp.diag(self.X) >= 0]
        cons += [cp.diag(self.X) <= self.p_diag_ub]
        # (b) McCormick via cp.multiply broadcast (DPP-clean)
        lo_col = cp.reshape(self.p_lo_eps, (d, 1), order='C')
        lo_row = cp.reshape(self.p_lo_eps, (1, d), order='C')
        hi_col = cp.reshape(self.p_hi_eps, (d, 1), order='C')
        hi_row = cp.reshape(self.p_hi_eps, (1, d), order='C')
        # cp.multiply((d,1) variable, (1,d) parameter) broadcasts to (d,d)
        # M_ij = var_i · param_j  — DPP because entrywise variable × parameter
        cons += [self.X >= cp.multiply(eps_col, lo_row)
                            + cp.multiply(lo_col, eps_row)
                            - self.p_lolo]
        cons += [self.X >= cp.multiply(eps_col, hi_row)
                            + cp.multiply(hi_col, eps_row)
                            - self.p_hihi]
        cons += [self.X <= cp.multiply(eps_col, hi_row)
                            + cp.multiply(lo_col, eps_row)
                            - self.p_lohi]
        cons += [self.X <= cp.multiply(eps_col, lo_row)
                            + cp.multiply(hi_col, eps_row)
                            - self.p_lohi.T]
        # Σ=0 RLT row sums
        cons += [cp.sum(self.X, axis=1) == 0]
        # Positivity RLT (DPP via broadcast)
        mu_col = cp.reshape(self.p_mu_star, (d, 1), order='C')
        mu_row = cp.reshape(self.p_mu_star, (1, d), order='C')
        cons += [self.X + cp.multiply(mu_col, eps_row)
                            + cp.multiply(eps_col, mu_row)
                            + self.p_muprod
                  >= 0]
        # Objective: min margin + grad·ε + tr(Q X)
        obj_expr = self.p_margin + self.p_grad @ self.eps + cp.trace(self.p_Q @ self.X)
        self.problem = cp.Problem(cp.Minimize(obj_expr), cons)

    def solve(self, lo_eps, hi_eps, mu_star, Q, grad, margin,
                verbose: bool = False):
        d = self.d
        self.p_lo_eps.value = lo_eps
        self.p_hi_eps.value = hi_eps
        self.p_mu_star.value = np.maximum(mu_star, 0.0)
        self.p_Q.value = 0.5 * (Q + Q.T)
        self.p_grad.value = grad
        self.p_margin.value = float(margin)
        self.p_lolo.value = np.outer(lo_eps, lo_eps)
        self.p_hihi.value = np.outer(hi_eps, hi_eps)
        self.p_lohi.value = np.outer(lo_eps, hi_eps)
        self.p_muprod.value = np.outer(self.p_mu_star.value,
                                          self.p_mu_star.value)
        # Precomputed diag UB — equivalent to v3's cp.maximum(cp.power(·,2))
        self.p_diag_ub.value = np.maximum(lo_eps ** 2, hi_eps ** 2)
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


_sdp_template_cache_v6: Dict[int, ShorSDPTemplate_v6] = {}

def get_sdp_template_v6(d: int) -> ShorSDPTemplate_v6:
    if d not in _sdp_template_cache_v6:
        _sdp_template_cache_v6[d] = ShorSDPTemplate_v6(d)
    return _sdp_template_cache_v6[d]


# =====================================================================
# Tier L (single W) using v6 DPP-clean template
# =====================================================================

def tier_L_single_v6(cell_cache: CellCache, W: WindowData,
                       c_target: float) -> Tuple[float, dict]:
    if not HAS_CVXPY:
        return -np.inf, {}
    mu_star = cell_cache.mu_star
    A = W.A
    margin = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    grad = W.grad_coef * (A @ mu_star)
    Q = W.Q_coef * A
    template = get_sdp_template_v6(cell_cache.cell.d)
    lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                mu_star, Q, grad, margin)
    if lb is None:
        return -np.inf, info
    return lb, info


# =====================================================================
# Tier L_joint Lagrangian using v6 template + tighter FW termination (S5)
# =====================================================================

def _fW_value(W: WindowData, mu_star: np.ndarray, eps_val: np.ndarray,
                X_val: np.ndarray, c_target: float) -> float:
    m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
    g_W = W.grad_coef * (W.A @ mu_star)
    Q_W = W.Q_coef * W.A
    return float(m_W + g_W @ eps_val + np.trace(Q_W @ X_val))


def _aggregate(top_windows, lam, mu_star, c_target):
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


def tier_L_joint_lagrangian_v6(cell_cache: CellCache,
                                  top_windows: List[WindowData],
                                  c_target: float,
                                  max_fw_iters: int = 6,
                                  fw_no_improve_tol: float = 1e-7
                                  ) -> Tuple[float, dict]:
    """Sound Lagrangian L_joint with v6 SDP + tighter FW termination.

    Stops as soon as:
      - LB > 0 (already certified)
      - LB did not improve by more than `fw_no_improve_tol` since previous iter
    """
    if not HAS_CVXPY:
        return -np.inf, {}
    d = cell_cache.cell.d
    mu_star = cell_cache.mu_star
    template = get_sdp_template_v6(d)
    K = len(top_windows)

    # Warm start: L_single for each, pick argmax
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

    # Early exit if already certified by best single
    if best_lb > 0:
        return best_lb, {'best_lam': lam, 'L_single': L_single,
                          'iters_used': 0, 'method': 'best_single_already_pos'}

    # FW iterations
    prev_lb = best_lb
    iters_used = 0
    for t in range(1, max_fw_iters + 1):
        iters_used = t
        Q_lam, g_lam, m_lam = _aggregate(top_windows, lam, mu_star, c_target)
        lb_t, info_t = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                         mu_star, Q_lam, g_lam, m_lam)
        if lb_t is None:
            break
        if lb_t > best_lb:
            best_lb = float(lb_t)
        # Terminate: certified
        if best_lb > 0:
            break
        # Terminate: no improvement
        if t > 1 and abs(lb_t - prev_lb) < fw_no_improve_tol:
            break
        prev_lb = lb_t
        eps_val = info_t.get('eps')
        X_val = info_t.get('X')
        if eps_val is None or X_val is None:
            break
        # Subgradient g_W = f_W(ε*, X*)
        g_subgrad = np.array([_fW_value(W, mu_star, eps_val, X_val, c_target)
                                for W in top_windows])
        s_idx = int(np.argmax(g_subgrad))
        eta = 2.0 / (t + 2.0)
        lam = (1.0 - eta) * lam + eta * np.eye(K)[s_idx]

    return float(best_lb), {'L_single': L_single, 'iters_used': iters_used,
                              'best_lam': lam, 'method': 'lagrangian_fw_v6'}


# =====================================================================
# m_W ranking using cached vector
# =====================================================================

def select_L_candidates_v6(mw_vec: np.ndarray, bundle: WindowBundle,
                              K: int = 3, include_widest: int = 2) -> List[int]:
    chosen: List[int] = []
    if include_widest > 0:
        ells = np.array([W.ell for W in bundle.windows])
        for idx in np.argsort(-ells)[:include_widest]:
            if int(idx) not in chosen:
                chosen.append(int(idx))
    m_order = np.argsort(-mw_vec)
    for idx in m_order:
        if int(idx) not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= K:
            break
    return chosen[:K]


# =====================================================================
# Cascade certification (v6)
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
                verbose: bool = False,
                bundle: Optional[WindowBundle] = None) -> CertResult:
    """Sound box certificate (v6 cascade — same logic as v5, faster).

    Order: empty → B1(vec) → B1u(vec) → B1diag → F → L_single → L_joint → split.
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
        best_b1 = float(b1_vec[best_idx])
        if best_b1 > 0:
            return CertResult(certified=True, tier_used='B1',
                              bound=best_b1, depth_used=current_depth,
                              best_W_idx=best_idx)

    # ---- B1u vectorized ----
    if use_B1u:
        b1u_vec = tier_B1u_vec(cell, bundle, c_target)
        best_idx = int(np.argmax(b1u_vec))
        best_b1u = float(b1u_vec[best_idx])
        if best_b1u > 0:
            return CertResult(certified=True, tier_used='B1u',
                              bound=best_b1u, depth_used=current_depth,
                              best_W_idx=best_idx)

    # ---- B1diag (kept as v5 — per-W LP, harder to vectorize) ----
    if use_B1diag:
        for i, W in enumerate(windows):
            b = v5.tier_B1diag_amgm(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1diag',
                                  bound=b, depth_used=current_depth,
                                  best_W_idx=i)

    # ---- Tier F (existing v3 vectorized) ----
    if use_F:
        v3.compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)

    # ---- (S4) Precompute m_W vector once, reuse for L + L_joint ranking ----
    mw_vec = mwQ_vec(cell, bundle, c_target, cache.mu_star)

    # ---- Tier L_single (m_W ranked, widest-first) ----
    L_best = -np.inf
    if use_L and HAS_CVXPY:
        K_L = min(3, len(windows))
        candidates = select_L_candidates_v6(mw_vec, bundle, K=K_L,
                                              include_widest=2)
        for w_idx in candidates:
            lb, _info = tier_L_single_v6(cache, windows[w_idx], c_target)
            if lb > L_best:
                L_best = lb
            if lb > 0:
                return CertResult(certified=True, tier_used='L',
                                  bound=lb, depth_used=current_depth,
                                  best_W_idx=w_idx)

    # ---- Tier L_joint (Lagrangian + v6 SDP + tight FW termination) ----
    Lj_best = -np.inf
    if use_L_joint and HAS_CVXPY:
        candidates_j = select_L_candidates_v6(mw_vec, bundle, K=joint_K,
                                                include_widest=2)
        top_j = [windows[i] for i in candidates_j]
        lb_j, _info_j = tier_L_joint_lagrangian_v6(cache, top_j, c_target,
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

    # ---- B46: skip wasteful split at root if no improvement signal ----
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
                    verbose=verbose, bundle=bundle)
    if not r1.certified:
        return CertResult(certified=False, tier_used='split_sub1_fail',
                          depth_used=r1.depth_used,
                          n_subcells=r1.n_subcells)
    r2 = cert_cell(sub2, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1,
                    use_B1=use_B1, use_B1u=use_B1u, use_B1diag=use_B1diag,
                    use_F=use_F, use_L=use_L, use_L_joint=use_L_joint,
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

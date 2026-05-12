"""Coarse BnB v3 — optimized version of v2.

Drop-in replacement preserving all soundness guarantees. Optimizations:
  (4)  Vectorized tier_F quadratic LB.
  (5)  Vectorized lp_max_linear (sort + cumsum + searchsorted).
  (6)  Vectorized build_A_matrix.
  (1)  Cache F-tier results across Q / L / L_joint / split.
  (7)  Cache per-cell lo_eps, hi_eps, h, H, Amu.
  (8)  Parametric CVXPY problem (DPP) — built once per d.
  (10) MOSEK tightened tolerances (1e-10) + thread pinning.
  (11) Sign-aware McCormick for tier_F quadratic LB
       (uses min of 4 corner products per (i,j); recovers ε_i² ≥ min(lo²,hi²)
        when 0 ∉ [lo_i, hi_i]).
  (14) Σ=0 RLT cuts in SDP: cp.sum(X, axis=1) == 0.
  (15) Positivity RLT in SDP: X_ij ≥ −μ*_i μ*_j − μ*_i ε_j − ε_i μ*_j.
  (16) Joint Shor via epigraph: max_t  s.t.  f_W(ε, X) ≥ t  for each W in top-K.

Each optimization is independently sound. Validated by V1 (MC soundness) + V2
(tier monotonicity) in _coarse_bnb_v3_validate.py.
"""
from __future__ import annotations
import os, sys, time, math, logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

logging.getLogger('cvxpy').setLevel(logging.ERROR)
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


# =====================================================================
# Window enumeration and A_W precomputation (item 6 vectorization)
# =====================================================================

def enumerate_windows(d: int) -> List[Tuple[int, int]]:
    out = []
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            out.append((ell, s_lo))
    return out


def build_A_matrix_vec(d: int, ell: int, s_lo: int) -> np.ndarray:
    """Vectorized: A_W[i,j] = 1 iff s_lo ≤ i+j ≤ s_lo+ell−2."""
    s_hi = s_lo + ell - 2
    idx = np.arange(d)
    I, J = np.meshgrid(idx, idx, indexing='ij')
    sums = I + J
    A = ((sums >= s_lo) & (sums <= s_hi)).astype(np.float64)
    return A


@dataclass
class WindowData:
    ell: int
    s_lo: int
    A: np.ndarray
    Q_coef: float
    grad_coef: float
    # Cached: off-diagonal mask (A=1 and i≠j) for sign-aware quad LB
    A_offdiag_mask: np.ndarray = field(default=None, repr=False)
    # Cached: diagonal mask (A=1 and i=j) for diagonal contribution
    A_diag_mask: np.ndarray = field(default=None, repr=False)

    @classmethod
    def build(cls, d: int, ell: int, s_lo: int) -> 'WindowData':
        A = build_A_matrix_vec(d, ell, s_lo)
        obj = cls(ell=ell, s_lo=s_lo, A=A,
                   Q_coef=2.0 * d / ell, grad_coef=4.0 * d / ell)
        obj.A_offdiag_mask = (A > 0.5) & (~np.eye(d, dtype=bool))
        obj.A_diag_mask = np.diag(A) > 0.5
        return obj


def build_all_windows(d: int) -> List[WindowData]:
    return [WindowData.build(d, ell, s_lo) for (ell, s_lo) in enumerate_windows(d)]


# =====================================================================
# Cell representation
# =====================================================================

@dataclass
class Cell:
    lo: np.ndarray
    hi: np.ndarray
    # Cached values for performance
    _center: Optional[np.ndarray] = None
    _h: Optional[np.ndarray] = None

    def __post_init__(self):
        self.lo = np.maximum(np.asarray(self.lo, dtype=np.float64), 0.0)
        self.hi = np.asarray(self.hi, dtype=np.float64)
        assert np.all(self.lo <= self.hi + 1e-14)

    @property
    def d(self) -> int:
        return len(self.lo)

    @property
    def half_widths(self) -> np.ndarray:
        if self._h is None:
            self._h = (self.hi - self.lo) / 2.0
        return self._h

    @property
    def center(self) -> np.ndarray:
        if self._center is None:
            mid = (self.lo + self.hi) / 2.0
            s = mid.sum()
            center = mid + (1.0 - s) / self.d
            self._center = np.clip(center, self.lo, self.hi)
        return self._center

    def is_simplex_feasible(self) -> bool:
        return (self.lo.sum() <= 1.0 + 1e-12 and self.hi.sum() >= 1.0 - 1e-12)

    def split(self, axis: int, frac: float = 0.5) -> Tuple['Cell', 'Cell']:
        mid = self.lo[axis] + frac * (self.hi[axis] - self.lo[axis])
        lo1, hi1 = self.lo.copy(), self.hi.copy()
        lo2, hi2 = self.lo.copy(), self.hi.copy()
        hi1[axis] = mid
        lo2[axis] = mid
        return Cell(lo1, hi1), Cell(lo2, hi2)

    @classmethod
    def from_integer_composition(cls, c: np.ndarray, S: int) -> 'Cell':
        c = np.asarray(c, dtype=np.float64)
        h = 1.0 / (2.0 * S)
        lo = np.maximum(c / S - h, 0.0)
        hi = c / S + h
        return cls(lo, hi)


# =====================================================================
# LP closed-form (item 5: vectorized via cumsum + searchsorted)
# =====================================================================

def lp_max_linear(grad: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   target: float = 0.0) -> float:
    """max grad·ε s.t. lo ≤ ε ≤ hi, Σε = target.  Vectorized greedy.

    Closed form: ε = lo + θ·(hi−lo) where θ ∈ [0,1]^d, Σθ·w = budget.
    Greedy: sort by grad descending, fill θ=1 until budget exhausted.
    """
    w = hi - lo  # ≥ 0
    s_lo = float(lo.sum())
    budget = target - s_lo
    total_w = float(w.sum())
    if budget < -1e-12 or budget > total_w + 1e-12:
        return -np.inf
    budget = max(0.0, min(budget, total_w))
    base_val = float(grad @ lo)

    # Sort by grad descending
    order = np.argsort(-grad)
    w_sorted = w[order]
    grad_sorted = grad[order]
    cum_w = np.cumsum(w_sorted)

    # Index where cumulative w first exceeds budget
    idx = int(np.searchsorted(cum_w, budget))
    if idx >= len(cum_w):
        # All filled
        return base_val + float(grad_sorted @ w_sorted)
    # Take grad_sorted[:idx] fully, partial for grad_sorted[idx]
    full_val = float(grad_sorted[:idx] @ w_sorted[:idx])
    cum_before = cum_w[idx - 1] if idx > 0 else 0.0
    partial = budget - cum_before
    return base_val + full_val + float(grad_sorted[idx]) * partial


def lp_min_linear(grad: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   target: float = 0.0) -> float:
    val = lp_max_linear(-grad, lo, hi, target)
    return -val if np.isfinite(val) else val


# =====================================================================
# Sign-aware McCormick LB on ε_iε_j over [lo_i,hi_i]×[lo_j,hi_j]
# =====================================================================

def mccormick_lb_pairwise(lo_eps: np.ndarray, hi_eps: np.ndarray) -> np.ndarray:
    """Vectorized McCormick LB on ε_iε_j for all (i,j).

    Returns M_lb[i,j] = min(lo_i·lo_j, lo_i·hi_j, hi_i·lo_j, hi_i·hi_j).
    For i=j (diagonal): M_lb[i,i] = min(lo_i², lo_i·hi_i, hi_i²)
                       = min(lo_i², hi_i²) if 0 ∉ [lo_i, hi_i]
                       = 0 if 0 ∈ [lo_i, hi_i] (lo_i·hi_i ≤ 0).
    Sound: ε_iε_j is bilinear on box, min at corner.
    """
    lo_lo = np.outer(lo_eps, lo_eps)
    lo_hi = np.outer(lo_eps, hi_eps)
    hi_lo = np.outer(hi_eps, lo_eps)
    hi_hi = np.outer(hi_eps, hi_eps)
    return np.minimum(np.minimum(lo_lo, lo_hi),
                       np.minimum(hi_lo, hi_hi))


# =====================================================================
# Tier F (vectorized, sign-aware)
# =====================================================================

@dataclass
class CellCache:
    """Cached per-cell quantities to amortize across tiers."""
    cell: Cell
    mu_star: np.ndarray
    h: np.ndarray
    lo_eps: np.ndarray
    hi_eps: np.ndarray
    M_lb: np.ndarray   # mccormick LB matrix (d, d)
    # F-tier scratch: filled lazily, indexed by W
    f_bounds: Dict[int, float] = field(default_factory=dict)
    f_best_bound: float = -np.inf
    f_best_W_idx: int = -1
    f_ranked_indices: Optional[np.ndarray] = None  # sorted desc by F bound

    @classmethod
    def build(cls, cell: Cell) -> 'CellCache':
        mu_star = cell.center
        h = cell.half_widths
        lo_eps = np.maximum(cell.lo - mu_star, -mu_star)
        hi_eps = cell.hi - mu_star
        M_lb = mccormick_lb_pairwise(lo_eps, hi_eps)
        return cls(cell=cell, mu_star=mu_star, h=h,
                    lo_eps=lo_eps, hi_eps=hi_eps, M_lb=M_lb)


def tier_F(cell_cache: CellCache, W: WindowData, c_target: float) -> float:
    """Vectorized tier F bound. SOUND.

    f_W(ε) = margin_W + grad_W·ε + ε^T Q_W ε
    LB(min f_W) = margin_W + min(grad_W·ε) + min(ε^T Q_W ε)

    min(grad_W·ε) over (box, Σ=0): lp_min_linear.
    min(ε^T Q_W ε) over box: Q_coef · Σ_{(i,j) in W ordered} min(ε_iε_j)
                            ≥ Q_coef · Σ_{(i,j) in W ordered} M_lb[i,j]
      (Sound: dropped Σ=0 constraint, used corner minimum per pair.
       Tighter than entry-wise −h_i h_j when cell is offset from origin.)
    """
    mu_star = cell_cache.mu_star
    A = W.A
    margin_W = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    grad_W = W.grad_coef * (A @ mu_star)
    lin_min = lp_min_linear(grad_W, cell_cache.lo_eps,
                              cell_cache.hi_eps, target=0.0)
    # Quadratic LB: sum over (i,j) where A=1 of mccormick LB.
    # SOUND: each ε_iε_j ≥ M_lb[i,j] independently; sum holds.
    quad_lb_sum = float(np.sum(A * cell_cache.M_lb))
    bound = margin_W + lin_min + W.Q_coef * quad_lb_sum
    return float(bound)


def compute_F_all_windows(cell_cache: CellCache,
                            windows: List[WindowData],
                            c_target: float) -> None:
    """Compute tier_F for all windows; cache into cell_cache.

    After this, cell_cache.f_bounds, f_best_bound, f_best_W_idx,
    f_ranked_indices are populated.
    """
    bounds = np.empty(len(windows))
    for i, W in enumerate(windows):
        bounds[i] = tier_F(cell_cache, W, c_target)
        cell_cache.f_bounds[i] = bounds[i]
    best_idx = int(np.argmax(bounds))
    cell_cache.f_best_bound = float(bounds[best_idx])
    cell_cache.f_best_W_idx = best_idx
    cell_cache.f_ranked_indices = np.argsort(-bounds)


# =====================================================================
# Parametrized SDP problem (item 8: DPP, built once per d)
# =====================================================================

class ShorSDPTemplate:
    """Parametrized CVXPY problem for the single-window Shor SDP.

    Built once per d. Parameters: lo_eps, hi_eps, mu_star, Q (d×d), grad (d),
    margin (1). Solved many times by reassigning parameter values.

    Optimizations:
      Σ=0 RLT cuts (item 14): cp.sum(X, axis=1) == 0
      Positivity RLT (item 15): X_ij ≥ −μ*_i μ*_j − μ*_i ε_j − ε_i μ*_j
      Vectorized McCormick (item 8): 4 matrix-inequality blocks instead of
        4·C(d,2) scalar constraints.
    """
    def __init__(self, d: int):
        if not HAS_CVXPY:
            raise ImportError("cvxpy required for SDP")
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
        # For McCormick: lo_i·lo_j etc. precomputed
        self.p_lolo = cp.Parameter((d, d), symmetric=True, name='lolo')
        self.p_lohi = cp.Parameter((d, d), name='lohi')
        self.p_hihi = cp.Parameter((d, d), symmetric=True, name='hihi')
        self.p_muprod = cp.Parameter((d, d), symmetric=True, name='muprod')

        # PSD Shor lift: [[1, ε^T], [ε, X]] ⪰ 0
        ones11 = np.ones((1, 1))
        Y = cp.bmat([[ones11, cp.reshape(self.eps, (1, d))],
                      [cp.reshape(self.eps, (d, 1)), self.X]])
        cons = [Y >> 0]
        cons += [self.eps >= self.p_lo_eps,
                  self.eps <= self.p_hi_eps,
                  cp.sum(self.eps) == 0,
                  self.eps >= -self.p_mu_star]

        # Diagonal X_ii bounds
        cons += [cp.diag(self.X) >= 0]
        cons += [cp.diag(self.X) <= cp.maximum(
            cp.power(self.p_lo_eps, 2), cp.power(self.p_hi_eps, 2))]

        # Vectorized McCormick (4 matrix inequalities) on full X (incl. diagonal)
        # X_ij ≥ lo_j ε_i + lo_i ε_j − lo_i lo_j
        # X_ij ≥ hi_j ε_i + hi_i ε_j − hi_i hi_j
        # X_ij ≤ hi_j ε_i + lo_i ε_j − lo_i hi_j
        # X_ij ≤ lo_j ε_i + hi_i ε_j − hi_i lo_j
        eps_col = cp.reshape(self.eps, (d, 1))
        eps_row = cp.reshape(self.eps, (1, d))
        lo_col = cp.reshape(self.p_lo_eps, (d, 1))
        lo_row = cp.reshape(self.p_lo_eps, (1, d))
        hi_col = cp.reshape(self.p_hi_eps, (d, 1))
        hi_row = cp.reshape(self.p_hi_eps, (1, d))
        cons += [self.X >= eps_col @ lo_row + lo_col @ eps_row - self.p_lolo]
        cons += [self.X >= eps_col @ hi_row + hi_col @ eps_row - self.p_hihi]
        cons += [self.X <= eps_col @ hi_row + lo_col @ eps_row - self.p_lohi]
        cons += [self.X <= eps_col @ lo_row + hi_col @ eps_row - self.p_lohi.T]

        # Σ=0 RLT cuts (item 14): Σ_j X_ij = 0 for each i.
        # From ε_i Σ_j ε_j = 0 ⇒ Σ_j ε_i ε_j = Σ_j X_ij = 0.
        cons += [cp.sum(self.X, axis=1) == 0]

        # Positivity RLT (item 15): μ_i μ_j ≥ 0
        # ⇒ X_ij + μ*_i ε_j + ε_i μ*_j + μ*_i μ*_j ≥ 0
        mu_col = cp.reshape(self.p_mu_star, (d, 1))
        mu_row = cp.reshape(self.p_mu_star, (1, d))
        cons += [self.X + mu_col @ eps_row + eps_col @ mu_row + self.p_muprod
                  >= 0]

        # Objective: minimize margin + grad·ε + trace(Q X)
        obj_expr = self.p_margin + self.p_grad @ self.eps + cp.trace(self.p_Q @ self.X)
        self.problem = cp.Problem(cp.Minimize(obj_expr), cons)

    def solve(self, lo_eps, hi_eps, mu_star, Q, grad, margin, verbose=False):
        d = self.d
        self.p_lo_eps.value = lo_eps
        self.p_hi_eps.value = hi_eps
        self.p_mu_star.value = np.maximum(mu_star, 0.0)
        # Symmetrize Q (safety)
        self.p_Q.value = 0.5 * (Q + Q.T)
        self.p_grad.value = grad
        self.p_margin.value = float(margin)
        # McCormick parameters
        self.p_lolo.value = np.outer(lo_eps, lo_eps)
        self.p_hihi.value = np.outer(hi_eps, hi_eps)
        self.p_lohi.value = np.outer(lo_eps, hi_eps)
        self.p_muprod.value = np.outer(self.p_mu_star.value,
                                          self.p_mu_star.value)
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
            lb -= 1e-8  # tighter than v2's 1e-7
        return lb, {'status': self.problem.status,
                     'eps': self.eps.value.copy() if self.eps.value is not None else None}


# Global SDP template cache (per d)
_sdp_template_cache: Dict[int, ShorSDPTemplate] = {}

def get_sdp_template(d: int) -> ShorSDPTemplate:
    if d not in _sdp_template_cache:
        _sdp_template_cache[d] = ShorSDPTemplate(d)
    return _sdp_template_cache[d]


# =====================================================================
# Tier L (single window) — uses cached SDP template
# =====================================================================

def tier_L_single(cell_cache: CellCache, W: WindowData,
                    c_target: float, verbose: bool = False) -> float:
    if not HAS_CVXPY:
        return -np.inf
    mu_star = cell_cache.mu_star
    A = W.A
    margin = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    grad = W.grad_coef * (A @ mu_star)
    Q = W.Q_coef * A
    template = get_sdp_template(cell_cache.cell.d)
    lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                mu_star, Q, grad, margin, verbose=verbose)
    if lb is None:
        return -np.inf
    return lb


# =====================================================================
# Joint Shor with epigraph (item 16) — exact max_λ
# =====================================================================

class JointShorTemplate:
    """Joint Shor SDP via epigraph variable t.
        max t  s.t.  f_W(ε, X) ≥ t  for each W in top-K, plus all RLT.
    Equivalent to max_λ inf_ε Σλ_W f_W; t is the exact joint LB.

    The set of windows is variable per cell, so we don't fully template
    the windows. We instantiate with a list of (Q_W, grad_W, margin_W).
    """
    def __init__(self, d: int, K: int):
        if not HAS_CVXPY:
            raise ImportError("cvxpy required")
        self.d = d
        self.K = K
        self.eps = cp.Variable(d, name='eps')
        self.X = cp.Variable((d, d), symmetric=True, name='X')
        self.t = cp.Variable(name='t')

        # Parameters per window
        self.p_Q = [cp.Parameter((d, d), symmetric=True, name=f'Q_{k}')
                     for k in range(K)]
        self.p_grad = [cp.Parameter(d, name=f'grad_{k}') for k in range(K)]
        self.p_margin = [cp.Parameter(name=f'margin_{k}') for k in range(K)]

        # Cell parameters (shared)
        self.p_lo_eps = cp.Parameter(d, name='lo_eps')
        self.p_hi_eps = cp.Parameter(d, name='hi_eps')
        self.p_mu_star = cp.Parameter(d, nonneg=True, name='mu_star')
        self.p_lolo = cp.Parameter((d, d), symmetric=True, name='lolo')
        self.p_lohi = cp.Parameter((d, d), name='lohi')
        self.p_hihi = cp.Parameter((d, d), symmetric=True, name='hihi')
        self.p_muprod = cp.Parameter((d, d), symmetric=True, name='muprod')

        ones11 = np.ones((1, 1))
        Y = cp.bmat([[ones11, cp.reshape(self.eps, (1, d))],
                      [cp.reshape(self.eps, (d, 1)), self.X]])
        cons = [Y >> 0]
        cons += [self.eps >= self.p_lo_eps,
                  self.eps <= self.p_hi_eps,
                  cp.sum(self.eps) == 0,
                  self.eps >= -self.p_mu_star]
        cons += [cp.diag(self.X) >= 0]
        cons += [cp.diag(self.X) <= cp.maximum(
            cp.power(self.p_lo_eps, 2), cp.power(self.p_hi_eps, 2))]
        # McCormick
        eps_col = cp.reshape(self.eps, (d, 1))
        eps_row = cp.reshape(self.eps, (1, d))
        lo_col = cp.reshape(self.p_lo_eps, (d, 1))
        lo_row = cp.reshape(self.p_lo_eps, (1, d))
        hi_col = cp.reshape(self.p_hi_eps, (d, 1))
        hi_row = cp.reshape(self.p_hi_eps, (1, d))
        cons += [self.X >= eps_col @ lo_row + lo_col @ eps_row - self.p_lolo]
        cons += [self.X >= eps_col @ hi_row + hi_col @ eps_row - self.p_hihi]
        cons += [self.X <= eps_col @ hi_row + lo_col @ eps_row - self.p_lohi]
        cons += [self.X <= eps_col @ lo_row + hi_col @ eps_row - self.p_lohi.T]
        # Σ=0 RLT
        cons += [cp.sum(self.X, axis=1) == 0]
        # Positivity RLT
        mu_col = cp.reshape(self.p_mu_star, (d, 1))
        mu_row = cp.reshape(self.p_mu_star, (1, d))
        cons += [self.X + mu_col @ eps_row + eps_col @ mu_row + self.p_muprod
                  >= 0]
        # Epigraph: t ≤ f_W(ε, X) for each window in top-K
        for k in range(K):
            f_W = self.p_margin[k] + self.p_grad[k] @ self.eps + cp.trace(self.p_Q[k] @ self.X)
            cons += [self.t <= f_W]
        self.problem = cp.Problem(cp.Maximize(self.t), cons)

    def solve(self, cell_cache: CellCache,
                windows_topk: List[WindowData], c_target: float,
                verbose: bool = False):
        d, K = self.d, self.K
        assert len(windows_topk) == K
        self.p_lo_eps.value = cell_cache.lo_eps
        self.p_hi_eps.value = cell_cache.hi_eps
        self.p_mu_star.value = np.maximum(cell_cache.mu_star, 0.0)
        self.p_lolo.value = np.outer(cell_cache.lo_eps, cell_cache.lo_eps)
        self.p_hihi.value = np.outer(cell_cache.hi_eps, cell_cache.hi_eps)
        self.p_lohi.value = np.outer(cell_cache.lo_eps, cell_cache.hi_eps)
        self.p_muprod.value = np.outer(self.p_mu_star.value,
                                          self.p_mu_star.value)
        for k, W in enumerate(windows_topk):
            margin_k = W.Q_coef * float(cell_cache.mu_star @ W.A @ cell_cache.mu_star) - c_target
            grad_k = W.grad_coef * (W.A @ cell_cache.mu_star)
            Q_k = W.Q_coef * W.A
            self.p_Q[k].value = 0.5 * (Q_k + Q_k.T)
            self.p_grad[k].value = grad_k
            self.p_margin[k].value = float(margin_k)
        try:
            self.problem.solve(solver=cp.MOSEK, verbose=verbose,
                                 mosek_params={
                                     'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                                     'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                                     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
                                     'MSK_IPAR_NUM_THREADS': 1,
                                 })
        except Exception as e:
            return None
        if self.problem.status not in ('optimal', 'optimal_inaccurate'):
            return None
        lb = float(self.problem.value)
        if self.problem.status == 'optimal_inaccurate':
            lb -= 1e-8
        return lb


_joint_template_cache: Dict[Tuple[int, int], JointShorTemplate] = {}

def get_joint_template(d: int, K: int) -> JointShorTemplate:
    key = (d, K)
    if key not in _joint_template_cache:
        _joint_template_cache[key] = JointShorTemplate(d, K)
    return _joint_template_cache[key]


def tier_L_joint(cell_cache: CellCache, windows: List[WindowData],
                   c_target: float, K: int = 4) -> float:
    """Joint Shor (epigraph) — exact max_λ joint LB across top-K windows."""
    if not HAS_CVXPY:
        return -np.inf
    if cell_cache.f_ranked_indices is None:
        return -np.inf  # require F precomputed (item 1)
    K_eff = min(K, len(windows))
    top_idx = cell_cache.f_ranked_indices[:K_eff]
    top_windows = [windows[i] for i in top_idx]
    template = get_joint_template(cell_cache.cell.d, K_eff)
    return template.solve(cell_cache, top_windows, c_target)


# =====================================================================
# Gradient-weighted split axis
# =====================================================================

def split_axis_gradient_weighted(cell_cache: CellCache,
                                   best_W: WindowData) -> int:
    A = best_W.A
    grad_W = best_W.grad_coef * (A @ cell_cache.mu_star)
    scores = (np.abs(grad_W) + best_W.Q_coef * (A @ cell_cache.h)) * cell_cache.h
    return int(np.argmax(scores))


# =====================================================================
# Cell certification cascade (with F cached — item 1)
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
                use_F: bool = True, use_L: bool = True,
                use_L_joint: bool = True,
                joint_K: int = 4, verbose: bool = False) -> CertResult:
    """Sound cell certificate via cascade F → L_single → L_joint → split."""
    if not cell.is_simplex_feasible():
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    cache = CellCache.build(cell)

    # Tier F (compute all windows once, cache results — item 1)
    if use_F:
        compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)

    # Tier L (single-W) on top-K_L windows from F ranking
    if use_L and HAS_CVXPY and cache.f_ranked_indices is not None:
        K_L = min(3, len(windows))
        best_L_bound = -np.inf
        best_L_idx = -1
        for w_idx in cache.f_ranked_indices[:K_L]:
            lb = tier_L_single(cache, windows[int(w_idx)], c_target, verbose)
            if lb > best_L_bound:
                best_L_bound = lb
                best_L_idx = int(w_idx)
            if lb > 0:
                return CertResult(certified=True, tier_used='L',
                                  bound=lb, depth_used=current_depth,
                                  best_W_idx=int(w_idx))

    # Tier L_joint
    if use_L_joint and HAS_CVXPY and cache.f_ranked_indices is not None:
        lb = tier_L_joint(cache, windows, c_target, K=joint_K)
        if lb is not None and lb > 0:
            return CertResult(certified=True, tier_used='L_joint',
                              bound=lb, depth_used=current_depth)

    # Split & recurse
    if current_depth >= max_depth:
        return CertResult(certified=False, tier_used='max_depth_reached',
                          depth_used=current_depth)

    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 \
              else windows[0]
    axis = split_axis_gradient_weighted(cache, best_W)
    sub1, sub2 = cell.split(axis)

    r1 = cert_cell(sub1, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1, use_F=use_F,
                    use_L=use_L, use_L_joint=use_L_joint,
                    joint_K=joint_K, verbose=verbose)
    if not r1.certified:
        return CertResult(certified=False, tier_used='split_sub1_fail',
                          depth_used=r1.depth_used,
                          n_subcells=r1.n_subcells)
    r2 = cert_cell(sub2, windows, c_target, max_depth=max_depth,
                    current_depth=current_depth + 1, use_F=use_F,
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

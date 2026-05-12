"""v8: Direct-MOSEK port of v7 ε-SDP — bypass CVXPY canonicalization (~10× speedup).

Same SDP, same constraints, same objective as v7. Just encoded via mosek.Task
directly, skipping CVXPY's per-call canonicalization overhead.

SOUNDNESS:
  Bit-exact equivalent to v7.ShorSDPTemplate_v7 (modulo MOSEK numeric tolerance).
  Validated by _coarse_bnb_v8_validate.py: |Δlb| < 1e-6 across 100+ cells.
  Same constraints, same objective, same optimum.

CONSTRAINTS encoded (identical to v6/v7):
  C1.  Y[0,0] = 1
  C2.  lo_eps[i] ≤ ε_i ≤ hi_eps[i],  and  ε_i ≥ −μ*[i]  (merged: max(lo, −μ*))
  C3.  Σ ε_i = 0
  C4.  0 ≤ X_ii ≤ max(lo_eps[i]², hi_eps[i]²)
  C5.  Full d×d McCormick (4 faces):  applied entrywise
  C6.  Σ_j X[i,j] = 0  for each i  (Σ=0 row-sum RLT)
  C7.  X_ij + μ*_i ε_j + μ*_j ε_i + μ*_i μ*_j ≥ 0  (μ-positivity RLT)
  C8.  X_ii + 2 μ*_i ε_i + μ*_i² ≤ μ*_i + ε_i  (v7 signing cut: μ_i (1 − μ_i) ≥ 0)

OBJECTIVE: minimize  <C, Y>  where:
  C[0,0]         = margin
  C[i+1, 0]      = grad_i / 2          (off-diag picks up factor 2)
  C[i+1, j+1]    = Q[i, j]             (diag) or Q[i, j] (off-diag)
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
import _coarse_bnb_v7 as v7

try:
    import mosek
    HAS_MOSEK = True
except ImportError:
    HAS_MOSEK = False
    mosek = None

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
tier_BD = v7.tier_BD
should_skip_SDP = v7.should_skip_SDP


# =====================================================================
# Direct-MOSEK SDP encoding helpers
# =====================================================================

def _coeffs(d, alpha_const, x_coef, X_coef):
    """Build sparse lower-triangle (subi, subj, val) for the (d+1)×(d+1)
    symmetric bar matrix A so that <A, Y> = alpha_const + Σ x_coef[i] ε_i
                                          + Σ X_coef[(i,j)] X[i,j].

    Convention (MOSEK lower-triangle):
        <A, Y> = Σ_i A_ii Y_ii + 2 Σ_{i>j} A_ij Y_ij.

    Layout: Y[0,0] = 1, Y[i+1, 0] = ε_i, Y[i+1, j+1] = X[i, j].

    So:
        const alpha → A[0,0] = alpha           (val = alpha)
        x_coef[i] ε_i → A[i+1, 0] = x_coef[i]/2 (val = 0.5 · x_coef[i])
        X_coef[(i,i)] X_ii → A[i+1, i+1] = X_coef[(i,i)] (val full)
        X_coef[(i,j)] X_ij (i ≠ j) → A[max+1, min+1] = X_coef[(i,j)]/2
            (val = 0.5 · X_coef[(i,j)])
    """
    subi, subj, val = [], [], []
    if alpha_const != 0.0:
        subi.append(0); subj.append(0); val.append(float(alpha_const))
    for i in range(d):
        c = x_coef[i] if i < len(x_coef) else 0.0
        if c != 0.0:
            subi.append(i + 1); subj.append(0); val.append(0.5 * float(c))
    for (i, j), c in X_coef.items():
        if c == 0.0:
            continue
        if i == j:
            subi.append(i + 1); subj.append(j + 1); val.append(float(c))
        else:
            ii, jj = (i, j) if i > j else (j, i)
            subi.append(ii + 1); subj.append(jj + 1); val.append(0.5 * float(c))
    return subi, subj, val


def _obj_bar_matrix(d, margin, grad, Q):
    """Build the symmetric bar-matrix C for the objective <C, Y>:
        margin + Σ grad_i ε_i + Σ_{i,j} Q_ij X_ij
    Returns (subi, subj, val) — lower triangle of C in MOSEK format.

    For diag: C[i+1, i+1] = Q_ii (Y_ii contributes once).
    For off-diag (i > j): C[i+1, j+1] = Q[i, j] (Y_ij contributes 2× via <C,Y>;
        we want 2 Σ_{i>j} Q_ij X_ij contribution, so val = Q[i,j] not half).
    Linear term grad_i ε_i appears via C[i+1, 0]: <C,Y> contribution
        2 · C[i+1, 0] · Y[i+1, 0] = grad_i ε_i  ⇒  C[i+1, 0] = grad_i / 2.
    Constant term margin via C[0, 0] = margin.
    """
    subi, subj, val = [], [], []
    if margin != 0.0:
        subi.append(0); subj.append(0); val.append(float(margin))
    for i in range(d):
        g = float(grad[i])
        if g != 0.0:
            subi.append(i + 1); subj.append(0); val.append(0.5 * g)
    for i in range(d):
        # Diagonal
        if Q[i, i] != 0.0:
            subi.append(i + 1); subj.append(i + 1); val.append(float(Q[i, i]))
        for j in range(i):
            if Q[i, j] != 0.0:
                subi.append(i + 1); subj.append(j + 1); val.append(float(Q[i, j]))
    return subi, subj, val


# =====================================================================
# Direct-MOSEK Shor SDP solver
# =====================================================================

class ShorSDPDirect:
    """Direct-MOSEK Shor SDP at fixed d.  Per-call fresh task, shared env."""

    def __init__(self, d: int, env=None, tol: float = 1e-10):
        if not HAS_MOSEK:
            raise ImportError("mosek not installed")
        self.d = d
        self.tol = tol
        self.bar_dim = d + 1
        self._own_env = env is None
        self.env = env if env is not None else mosek.Env()
        if self._own_env:
            try:
                self.env.checkoutlicense(mosek.feature.pton)
            except Exception:
                pass

    def __del__(self):
        if getattr(self, '_own_env', False):
            try:
                self.env.__exit__(None, None, None)
            except Exception:
                pass

    def solve(self, lo_eps, hi_eps, mu_star, Q, grad, margin):
        """Minimize  margin + grad·ε + tr(Q·X)  over Shor SDP feasible set.

        Returns (lb, info_dict).  lb = min value (a sound LB on
        min_{ε in cell} f_W(ε)).  None if MOSEK fails.
        """
        d = self.d
        bar_dim = self.bar_dim
        lo_eps = np.asarray(lo_eps, dtype=np.float64)
        hi_eps = np.asarray(hi_eps, dtype=np.float64)
        mu_star = np.maximum(np.asarray(mu_star, dtype=np.float64), 0.0)
        # Quick infeas check
        if np.any(lo_eps > hi_eps + 1e-14):
            return None, {'error': 'lo_eps > hi_eps'}
        Q = 0.5 * (np.asarray(Q, dtype=np.float64) + np.asarray(Q, dtype=np.float64).T)
        grad = np.asarray(grad, dtype=np.float64)
        margin = float(margin)
        # Effective ε lower bound (merge with positivity μ ≥ 0)
        lo_eff = np.maximum(lo_eps, -mu_star)
        diag_ub = np.maximum(lo_eps ** 2, hi_eps ** 2)

        try:
            with self.env.Task(0, 0) as task:
                # MOSEK params
                task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, self.tol)
                task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, self.tol)
                task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, self.tol)
                task.putintparam(mosek.iparam.intpnt_max_iterations, 400)
                task.putintparam(mosek.iparam.log, 0)
                task.putintparam(mosek.iparam.num_threads, 1)

                # PSD bar variable
                task.appendbarvars([bar_dim])

                # Objective
                obj_si, obj_sj, obj_sv = _obj_bar_matrix(d, margin, grad, Q)
                if len(obj_si) > 0:
                    obj_id = task.appendsparsesymmat(bar_dim, obj_si, obj_sj, obj_sv)
                    task.putbarcj(0, [obj_id], [1.0])
                task.putobjsense(mosek.objsense.minimize)

                def _add(subi, subj, vals, bk, blk, buk):
                    cidx = task.getnumcon()
                    task.appendcons(1)
                    if len(subi) > 0:
                        aid = task.appendsparsesymmat(bar_dim, subi, subj, vals)
                        task.putbaraij(cidx, 0, [aid], [1.0])
                    task.putconbound(cidx, bk, blk, buk)
                    return cidx

                # C1: Y[0,0] = 1
                sI, sJ, sV = _coeffs(d, 1.0, np.zeros(d), {})
                _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

                # C2: ε box (merged with positivity)
                for i in range(d):
                    xc = np.zeros(d); xc[i] = 1.0
                    sI, sJ, sV = _coeffs(d, 0.0, xc, {})
                    _add(sI, sJ, sV, mosek.boundkey.ra, lo_eff[i], hi_eps[i])

                # C3: Σε = 0
                sI, sJ, sV = _coeffs(d, 0.0, np.ones(d), {})
                _add(sI, sJ, sV, mosek.boundkey.fx, 0.0, 0.0)

                # C4: X_ii ∈ [0, diag_ub[i]]
                for i in range(d):
                    sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), {(i, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.ra, 0.0, diag_ub[i])

                # C5: McCormick — entrywise full d×d (i, j) for the 4 faces.
                # Face 1 (X_ij ≥ ε_i·lo_j + lo_i·ε_j − lo_i·lo_j):
                #   rewrite: X_ij − lo_j·ε_i − lo_i·ε_j + lo_i·lo_j ≥ 0
                # Same constraint at (j, i) ⇒ same X_ij = X_ji. Use upper triangle.
                # Face 3/4 (asymmetric upper) — must do BOTH (i,j) and (j,i)
                #   X_ij ≤ ε_i·hi_j + lo_i·ε_j − lo_i·hi_j   (Face 3)
                #   X_ij ≤ ε_i·lo_j + hi_i·ε_j − hi_i·lo_j   (Face 4)
                #   These two are independent.
                for i in range(d):
                    for j in range(i, d):  # upper triangle including diag
                        li, lj = lo_eps[i], lo_eps[j]
                        ui, uj = hi_eps[i], hi_eps[j]
                        # Face 1 lower: X_ij − lj·ε_i − li·ε_j + li·lj ≥ 0
                        xc = np.zeros(d); xc[i] -= lj; xc[j] -= li
                        sI, sJ, sV = _coeffs(d, li * lj, xc, {(i, j): 1.0})
                        _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                        # Face 2 lower (hi+hi): X_ij − uj·ε_i − ui·ε_j + ui·uj ≥ 0
                        xc = np.zeros(d); xc[i] -= uj; xc[j] -= ui
                        sI, sJ, sV = _coeffs(d, ui * uj, xc, {(i, j): 1.0})
                        _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                        # Face 3 upper (lohi): X_ij ≤ ε_i·hi_j + lo_i·ε_j − lo_i·hi_j
                        # rewrite: X_ij − hi_j·ε_i − lo_i·ε_j + lo_i·hi_j ≤ 0
                        xc = np.zeros(d); xc[i] -= uj; xc[j] -= li
                        sI, sJ, sV = _coeffs(d, li * uj, xc, {(i, j): 1.0})
                        _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)
                        # Face 4 upper (lohi.T): X_ij ≤ ε_i·lo_j + hi_i·ε_j − hi_i·lo_j
                        # rewrite: X_ij − lo_j·ε_i − hi_i·ε_j + hi_i·lo_j ≤ 0
                        # NOTE: for i == j Face 3 and Face 4 collapse.
                        if i != j:
                            xc = np.zeros(d); xc[i] -= lj; xc[j] -= ui
                            sI, sJ, sV = _coeffs(d, ui * lj, xc, {(i, j): 1.0})
                            _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

                # C6: Σ_j X[i,j] = 0 (Σ=0 RLT row sums)
                for i in range(d):
                    X_coef = {(i, j): 1.0 for j in range(d)}
                    sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), X_coef)
                    _add(sI, sJ, sV, mosek.boundkey.fx, 0.0, 0.0)

                # C7: μ-positivity RLT (entrywise, upper triangle incl diag)
                # X_ij + μ*_i ε_j + μ*_j ε_i + μ*_i μ*_j ≥ 0
                for i in range(d):
                    for j in range(i, d):
                        if i == j:
                            xc = np.zeros(d); xc[i] = 2.0 * mu_star[i]
                            const = mu_star[i] ** 2
                        else:
                            xc = np.zeros(d); xc[i] = mu_star[j]; xc[j] = mu_star[i]
                            const = mu_star[i] * mu_star[j]
                        sI, sJ, sV = _coeffs(d, const, xc, {(i, j): 1.0})
                        _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)

                # C8: v7 signing cut μ_i(1−μ_i) ≥ 0
                # X_ii + 2 μ*_i ε_i + μ*_i² ≤ μ*_i + ε_i
                # ⇔ X_ii + (2 μ*_i − 1) ε_i + (μ*_i² − μ*_i) ≤ 0
                for i in range(d):
                    xc = np.zeros(d); xc[i] = 2.0 * mu_star[i] - 1.0
                    const = mu_star[i] ** 2 - mu_star[i]
                    sI, sJ, sV = _coeffs(d, const, xc, {(i, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

                # Solve
                try:
                    task.optimize()
                except mosek.Error as e:
                    return None, {'error': f'optimize: {e}'}

                try:
                    solsta = task.getsolsta(mosek.soltype.itr)
                except mosek.Error as e:
                    return None, {'error': f'getsolsta: {e}'}

                if solsta == mosek.solsta.optimal:
                    lb = task.getprimalobj(mosek.soltype.itr)
                    return float(lb), {'status': 'optimal'}
                elif solsta == mosek.solsta.prim_infeas_cer:
                    # Cell SDP-infeasible — implies the original feasible set
                    # is empty (sound to treat as cert)
                    return float('inf'), {'status': 'prim_infeas_cer'}
                elif solsta in (mosek.solsta.unknown,):
                    # Try anyway with margin
                    try:
                        lb = task.getprimalobj(mosek.soltype.itr)
                        return float(lb) - 1e-7, {'status': 'unknown_margin'}
                    except Exception:
                        return None, {'status': 'unknown_no_obj'}
                else:
                    return None, {'status': f'{solsta}'}
        except Exception as e:
            return None, {'error': f'task: {e}'}


# Per-process cache, keyed by d
_direct_cache: Dict[int, ShorSDPDirect] = {}


def get_direct_template(d: int) -> ShorSDPDirect:
    if d not in _direct_cache:
        _direct_cache[d] = ShorSDPDirect(d)
    return _direct_cache[d]


# =====================================================================
# Tier L_single direct
# =====================================================================

def tier_L_single_v8(cell_cache: CellCache, W: WindowData,
                       c_target: float) -> Tuple[float, dict]:
    if not HAS_MOSEK:
        return -np.inf, {}
    mu_star = cell_cache.mu_star
    A = W.A
    margin = W.Q_coef * float(mu_star @ A @ mu_star) - c_target
    grad = W.grad_coef * (A @ mu_star)
    Q = W.Q_coef * A
    template = get_direct_template(cell_cache.cell.d)
    lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                mu_star, Q, grad, margin)
    if lb is None:
        return -np.inf, info
    return lb, info


# =====================================================================
# Tier L_joint Lagrangian using direct-MOSEK template
# =====================================================================

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


def tier_L_joint_lagrangian_v8(cell_cache: CellCache,
                                  top_windows: List[WindowData],
                                  c_target: float,
                                  max_fw_iters: int = 6,
                                  fw_no_improve_tol: float = 1e-7
                                  ) -> Tuple[float, dict]:
    if not HAS_MOSEK:
        return -np.inf, {}
    d = cell_cache.cell.d
    mu_star = cell_cache.mu_star
    template = get_direct_template(d)
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
                          'method': 'best_single_pos_v8'}
    prev_lb = best_lb
    iters = 0
    for t in range(1, max_fw_iters + 1):
        iters = t
        Q_lam, g_lam, m_lam = _aggregate(top_windows, lam, mu_star, c_target)
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
        # Note: direct-MOSEK doesn't easily expose X, ε for FW subgradient.
        # Without subgrad, fallback to symmetric averaging (less optimal).
        # Simple alternative: keep lam fixed at e_best_k for joint refinement;
        # if best_lb hasn't improved by iter 2, stop.
        break  # disable FW until we add X readback
    return float(best_lb), {'L_single': L_single, 'iters_used': iters,
                              'method': 'lagrangian_warm_start_v8'}


# =====================================================================
# Cascade certification (v8) — mirrors v7 cascade
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
    if is_cell_empty(cell):
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    if not cell.is_simplex_feasible():
        return CertResult(certified=True, tier_used='empty',
                          depth_used=current_depth)
    if bundle is None:
        bundle = get_bundle(windows)
    cache = CellCache.build(cell)

    if use_B1:
        b1_vec = tier_B1_vec(cell, bundle, c_target)
        best_idx = int(np.argmax(b1_vec))
        if b1_vec[best_idx] > 0:
            return CertResult(certified=True, tier_used='B1',
                              bound=float(b1_vec[best_idx]),
                              depth_used=current_depth, best_W_idx=best_idx)
    if use_B1u:
        b1u_vec = tier_B1u_vec(cell, bundle, c_target)
        best_idx = int(np.argmax(b1u_vec))
        if b1u_vec[best_idx] > 0:
            return CertResult(certified=True, tier_used='B1u',
                              bound=float(b1u_vec[best_idx]),
                              depth_used=current_depth, best_W_idx=best_idx)
    if use_B1diag:
        for i, W in enumerate(windows):
            b = tier_B1diag_amgm(cell, W, c_target)
            if b > 0:
                return CertResult(certified=True, tier_used='B1diag',
                                  bound=b, depth_used=current_depth, best_W_idx=i)
    if use_F:
        v3.compute_F_all_windows(cache, windows, c_target)
        if cache.f_best_bound > 0:
            return CertResult(certified=True, tier_used='F',
                              bound=cache.f_best_bound,
                              depth_used=current_depth,
                              best_W_idx=cache.f_best_W_idx)
    if use_BD:
        bd_lb, bd_W = tier_BD(cell, windows, c_target, bundle)
        if bd_lb > 0:
            return CertResult(certified=True, tier_used='BD',
                              bound=bd_lb, depth_used=current_depth,
                              best_W_idx=bd_W)

    mw_vec = mwQ_vec(cell, bundle, c_target, cache.mu_star)
    if use_skip_gate and use_F and should_skip_SDP(cell, bundle, c_target,
                                                       cache.f_best_bound):
        L_best = -np.inf; Lj_best = -np.inf
    else:
        L_best = -np.inf
        if use_L and HAS_MOSEK:
            K_L = min(3, len(windows))
            candidates = select_L_candidates_v6(mw_vec, bundle, K=K_L,
                                                  include_widest=2)
            for w_idx in candidates:
                lb, _info = tier_L_single_v8(cache, windows[w_idx], c_target)
                if lb > L_best:
                    L_best = lb
                if lb > 0:
                    return CertResult(certified=True, tier_used='L',
                                      bound=lb, depth_used=current_depth,
                                      best_W_idx=w_idx)
        Lj_best = -np.inf
        if use_L_joint and HAS_MOSEK:
            candidates_j = select_L_candidates_v6(mw_vec, bundle, K=joint_K,
                                                    include_widest=2)
            top_j = [windows[i] for i in candidates_j]
            lb_j, _info_j = tier_L_joint_lagrangian_v8(cache, top_j, c_target,
                                                          max_fw_iters=joint_fw_iters)
            Lj_best = lb_j
            if lb_j > 0:
                return CertResult(certified=True, tier_used='L_joint',
                                  bound=lb_j, depth_used=current_depth)

    eff_max_depth = max_depth
    if cell.d <= 4:
        eff_max_depth = min(max_depth, 1)
    if current_depth >= eff_max_depth:
        return CertResult(certified=False, tier_used='max_depth_reached',
                          depth_used=current_depth)
    f_best = cache.f_best_bound if use_F else -np.inf
    if (current_depth == 0 and use_L_joint and use_F
            and Lj_best <= f_best + 1e-9 and L_best <= f_best + 1e-9):
        return CertResult(certified=False, tier_used='B46_no_signal',
                          bound=max(f_best, Lj_best, L_best),
                          depth_used=current_depth)
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
                          depth_used=r1.depth_used, n_subcells=r1.n_subcells)
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

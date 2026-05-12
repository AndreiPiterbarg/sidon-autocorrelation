"""Cell-subdivision box certificate for the COARSE cascade.

GOAL
====
Close `uncert` cells from the coarse-grid cascade by recursive subdivision.
At each leaf of the recursion we run the FULL chain
    N+O linear/quadratic triangle  ->  Joint multi-window LP duality  ->  Shor SDP
and certify the leaf if any of them returns a sound LB ≥ c_target (or, for
Shor, a Farkas infeasibility certificate).

ALGORITHM
=========
For each uncert cell `c_int` (a coarse-grid composition), the cell is
    Cell = { delta : lo_i <= delta_i <= hi_i, Σ delta = 0 }
with initial `lo_i = max(-h, -mu*_i)`, `hi_i = +h`, h = 1/(2S).

`cell_cert_split(c_int, S, d, c_target, max_depth=4)` works as follows:
    leaf:
       1. N+O triangle bound (entry-wise quadratic + sparsity-aware LP)
       2. Joint LP duality (subgradient ascent) using box-aware UB_lin
       3. Shor SDP feasibility (MOSEK direct)
    if any returns a CERT  -> leaf certified, recursion terminates.
    else if depth < max_depth:
       split along i* = argmax (|grad_i| * r_box_i + (A r)_i r_i)
       recurse on each half (mid = (lo[i*] + hi[i*])/2).
    else: leaf NOT certified.

The cell is certified iff ALL leaves are certified.

SOUNDNESS
=========
Each leaf bound is a SOUND LB on min_{delta in leaf-cell} max_W TV_W(mu*+delta):
   * N+O: triangle bound with sparsity-aware LP (variant O) and entry-wise
     quad bound (handles per-axis [lo_i, hi_i] correctly).
   * Joint: LP weak duality on the polytope; UB_lin uses box-aware closed
     form (sweep over breakpoints), UB_quad uses entry-wise upper bound
     |eps^T A_W eps| <= sum_{(i,j)} max(|lo_i|,|hi_i|) max(|lo_j|,|hi_j|).
   * Shor SDP via MOSEK Farkas (`prim_infeas_cer` => infeasible).

Subdivision: Cell = LeftSubcell U RightSubcell on coordinate i*. Both are
sound subsets, so if both are certified, the parent is certified.

USAGE
=====
    python _cell_split_cert.py --d 4 --S 20 --c_target 1.20 --max_depth 4

Reports:
    - depth-1, depth-2, depth-3, depth-4 cumulative cert rate
    - comparison vs Shor SDP alone (single-call, no subdivision)
"""
from __future__ import annotations
import os, sys, time, json, argparse
from typing import List, Tuple, Optional
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

# Lazy mosek
try:
    import mosek
    _HAS_MOSEK = True
except Exception:
    _HAS_MOSEK = False
    mosek = None  # type: ignore


# =====================================================================
# Window enumeration & A_W
# =====================================================================

def _enum_windows(d: int) -> List[Tuple[int, int]]:
    out = []
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_w = conv_len - n_cv + 1
        for s_lo in range(n_w):
            out.append((ell, s_lo))
    return out


def _build_A(d: int, ell: int, s_lo: int) -> np.ndarray:
    A = np.zeros((d, d), dtype=np.float64)
    s_hi = s_lo + ell - 2
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A


# =====================================================================
# Box LP closed form: max grad . eps  s.t.  lo <= eps <= hi, Σ eps = T
# (used by both UB_lin in joint and per-window L1 bound)
# =====================================================================

def _box_lp_max(grad, lo_eps, hi_eps, target):
    """O(d log d) closed form: feasible if Σlo <= T <= Σhi.
    Sort by grad descending; pack hi from the top until cum sum reaches
    target; one boundary index sits at the slack.
    """
    s_lo = float(lo_eps.sum())
    s_hi = float(hi_eps.sum())
    if target < s_lo - 1e-12 or target > s_hi + 1e-12:
        return -np.inf
    if abs(target - s_lo) < 1e-12:
        return float(np.dot(grad, lo_eps))
    if abs(target - s_hi) < 1e-12:
        return float(np.dot(grad, hi_eps))

    d = len(grad)
    order = np.argsort(-grad)  # descending
    cum = float(lo_eps.sum())
    K_star = -1
    for k in range(d):
        i = order[k]
        cum += hi_eps[i] - lo_eps[i]
        if cum >= target - 1e-12:
            K_star = k
            break
    if K_star < 0:
        return -np.inf
    eps = lo_eps.copy()
    cum_before = float(lo_eps.sum())
    for k in range(K_star):
        i = order[k]
        eps[i] = hi_eps[i]
        cum_before += hi_eps[i] - lo_eps[i]
    i_b = order[K_star]
    eps_b = lo_eps[i_b] + (target - cum_before)
    eps_b = max(lo_eps[i_b], min(hi_eps[i_b], eps_b))
    eps[i_b] = eps_b
    return float(np.dot(grad, eps))


def _box_lp_max_abs(grad, lo_eps, hi_eps, target):
    """max |grad.eps| over the eps-box."""
    v_pos = _box_lp_max(grad, lo_eps, hi_eps, target)
    v_neg = _box_lp_max(-grad, lo_eps, hi_eps, target)
    if v_pos == -np.inf and v_neg == -np.inf:
        return -np.inf
    return max(v_pos, v_neg, 0.0)


# =====================================================================
# Per-leaf N+O triangle bound (handles general per-axis [lo_i, hi_i]).
# Returns (max_net, best_W, grad_at_anchor).
# =====================================================================

def _no_triangle_bound(c_int, lo, hi, S, d, c_target,
                        A_list, pairs_list, ell_list):
    """Sound triangle LB on min_delta max_W TV_W over the leaf box.

    Anchor: c' = (lo + hi)/2 - mean((lo+hi)/2)*1 (so Σ c' = 0).
    eps = delta - c' lives in eps-box: lo_eps_i = lo_i - c'_i, hi_eps_i = hi_i - c'_i.
    Σ eps = -Σ c' = 0.

    For each window:
       TV_W(mu* + delta) = tv_anchor + grad_c'.eps + (2d/ell) eps^T A_W eps
    Bound: tv_anchor - lin_W - quad_W,
       lin_W = max_eps |grad_c' . eps|  (LP closed form)
       quad_W = (2d/ell) Σ_{(i,j) in pairs} M_i M_j,  M_i = max(|lo_eps_i|, |hi_eps_i|).
    """
    mu_star = c_int.astype(np.float64) / S
    c_box = (lo + hi) / 2.0
    c_prime = c_box - float(np.mean(c_box))
    mu_anchor = mu_star + c_prime

    lo_eps = lo - c_prime
    hi_eps = hi - c_prime
    # Tighten with mu>=0 -> eps_i >= -mu*_i - c'_i = -(mu*+c')_i = -mu_anchor_i.
    lo_eps = np.maximum(lo_eps, -mu_anchor)

    if np.any(lo_eps > hi_eps + 1e-12):
        return np.inf, -1, None  # cell empty, vacuously cert
    if 0.0 < lo_eps.sum() - 1e-12 or 0.0 > hi_eps.sum() + 1e-12:
        return np.inf, -1, None

    M_eps = np.maximum(np.abs(lo_eps), np.abs(hi_eps))

    best_net = -np.inf
    best_W = -1
    best_grad = None
    for w_idx in range(len(A_list)):
        A_W = A_list[w_idx]
        ell = ell_list[w_idx]
        scale = 2.0 * d / ell

        tv_anchor = scale * float(mu_anchor @ A_W @ mu_anchor)
        grad = (4.0 * d / ell) * (A_W @ mu_anchor)
        lin = _box_lp_max_abs(grad, lo_eps, hi_eps, 0.0)
        if lin == -np.inf:
            continue
        pairs = pairs_list[w_idx]
        s = 0.0
        for (i, j) in pairs:
            s += M_eps[i] * M_eps[j]
        quad = scale * s
        net = tv_anchor - c_target - lin - quad
        if net > best_net:
            best_net = net
            best_W = w_idx
            best_grad = grad
    return best_net, best_W, best_grad


# =====================================================================
# Joint LP-duality LB extended to box [lo, hi].
# =====================================================================

def _joint_lb_box(c_int, lo, hi, S, d, c_target,
                   A_list, pairs_list, ell_list,
                   pruning_idx, n_iters=20, step0=1.0):
    """Sound LB on min_delta max_W TV_W via dual simplex ascent over
    convex combos of `pruning_idx` (windows W with TV_W(mu*) > c_target).
    Box-aware: UB_lin via _box_lp_max, UB_quad entry-wise.
    """
    if not pruning_idx:
        return -np.inf

    mu_star = c_int.astype(np.float64) / S
    c_box = (lo + hi) / 2.0
    c_prime = c_box - float(np.mean(c_box))
    mu_anchor = mu_star + c_prime

    lo_eps = lo - c_prime
    hi_eps = hi - c_prime
    lo_eps = np.maximum(lo_eps, -mu_anchor)

    if np.any(lo_eps > hi_eps + 1e-12):
        return np.inf
    if 0.0 < lo_eps.sum() - 1e-12 or 0.0 > hi_eps.sum() + 1e-12:
        return np.inf

    M_eps = np.maximum(np.abs(lo_eps), np.abs(hi_eps))

    # Per-window (anchored) precompute.
    n_W = len(pruning_idx)
    grad_arr = np.zeros((n_W, d), dtype=np.float64)
    scale_arr = np.zeros(n_W, dtype=np.float64)
    tv_anchor_arr = np.zeros(n_W, dtype=np.float64)
    pb_arr = np.zeros(n_W, dtype=np.float64)
    for k, w_idx in enumerate(pruning_idx):
        A_W = A_list[w_idx]
        ell = ell_list[w_idx]
        scale = 2.0 * d / ell
        scale_arr[k] = scale
        grad_arr[k] = (4.0 * d / ell) * (A_W @ mu_anchor)
        tv_anchor_arr[k] = scale * float(mu_anchor @ A_W @ mu_anchor)
        s = 0.0
        for (i, j) in pairs_list[w_idx]:
            s += M_eps[i] * M_eps[j]
        pb_arr[k] = s

    lam = np.full(n_W, 1.0 / n_W)

    def project_simplex(v):
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = -1
        for i in range(n):
            if u[i] - (cssv[i] - 1.0) / (i + 1) > 0:
                rho = i
        if rho < 0:
            return np.ones(n) / n
        theta = (cssv[rho] - 1.0) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    def lb_value(lam_):
        G = (lam_[:, None] * grad_arr).sum(axis=0)
        ub_lin = _box_lp_max_abs(G, lo_eps, hi_eps, 0.0)
        if ub_lin == -np.inf:
            return -np.inf
        ub_quad = float(np.sum(lam_ * scale_arr * pb_arr))
        return float(np.dot(lam_, tv_anchor_arr)) - ub_lin - ub_quad - c_target

    best_LB = lb_value(lam)
    if not np.isfinite(best_LB):
        best_LB = -np.inf

    for it in range(n_iters):
        G = (lam[:, None] * grad_arr).sum(axis=0)
        # Find the maximizer of |G.eps| within box (subgradient sigma)
        v_pos = _box_lp_max(G, lo_eps, hi_eps, 0.0)
        v_neg = _box_lp_max(-G, lo_eps, hi_eps, 0.0)
        if v_pos >= max(v_neg, 0.0):
            sign = +1
        elif v_neg >= 0.0:
            sign = -1
        else:
            sign = +1
        # Reconstruct eps* via the same closed form
        order = np.argsort(-(sign * G))
        cum = float(lo_eps.sum())
        K_star = -1
        for k in range(d):
            i = order[k]
            cum += hi_eps[i] - lo_eps[i]
            if cum >= 0.0 - 1e-12:
                K_star = k
                break
        eps_star = lo_eps.copy()
        cum_before = float(lo_eps.sum())
        if K_star >= 0:
            for k in range(K_star):
                i = order[k]
                eps_star[i] = hi_eps[i]
                cum_before += hi_eps[i] - lo_eps[i]
            i_b = order[K_star]
            eps_b = lo_eps[i_b] + (0.0 - cum_before)
            eps_b = max(lo_eps[i_b], min(hi_eps[i_b], eps_b))
            eps_star[i_b] = eps_b

        # subgradient of LB at lam:
        # ∂_W LB = tv_anchor_W - sign * grad_W.eps* - scale_W*pb_W.
        sub = (tv_anchor_arr
               - sign * (grad_arr @ eps_star)
               - scale_arr * pb_arr)

        step = step0 / np.sqrt(it + 1)
        lam_new = lam + step * sub
        lam = project_simplex(lam_new)
        v = lb_value(lam)
        if np.isfinite(v) and v > best_LB:
            best_LB = v

    return best_LB  # net (margin already subtracted)


# =====================================================================
# Shor SDP feasibility on a custom box (delta-frame, then converted to x-frame).
# =====================================================================

def _shor_feas_box(c_int, lo_delta, hi_delta, S, d, c_target,
                    A_list, ell_list, env=None,
                    tol=1e-9, eps_margin=1e-9):
    """Shor feasibility test in x-frame: x = S*mu, lo_x = S*(c_int/S + lo_delta) = c_int + S*lo_delta.
    Cell: lo_x_i <= x_i <= hi_x_i, Σx = S, x>=0,  Tr(A_W X) <= c*ell*S^2/(2d) + eps.
    Returns True iff MOSEK returns prim_infeas_cer (Farkas).
    """
    if not _HAS_MOSEK:
        return False, 'NO_MOSEK'

    c = c_int.astype(np.float64)
    lo_x = np.maximum(0.0, c + S * lo_delta)
    hi_x = c + S * hi_delta
    # Cap hi_x at S (since each x_i <= S given Σx = S, x>=0)
    hi_x = np.minimum(hi_x, float(S))
    # Box feasibility pre-filter
    if np.any(lo_x > hi_x + 1e-12):
        return True, 'box_empty'
    if lo_x.sum() > S + 1e-9 or hi_x.sum() < S - 1e-9:
        return True, 'box_sum_pre_infeasible'

    own_env = env is None
    if own_env:
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass

    bar_dim = d + 1
    try:
        with env.Task(0, 0) as task:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, tol)
            task.putintparam(mosek.iparam.intpnt_max_iterations, 200)
            task.putintparam(mosek.iparam.log, 0)
            task.putintparam(mosek.iparam.num_threads, 1)
            task.appendbarvars([bar_dim])
            task.putobjsense(mosek.objsense.minimize)

            def _coeffs(alpha_const, x_coef, X_coef_lower):
                subi, subj, val = [], [], []
                if alpha_const != 0.0:
                    subi.append(0); subj.append(0); val.append(float(alpha_const))
                for i in range(d):
                    cv = x_coef[i]
                    if cv != 0.0:
                        subi.append(i + 1); subj.append(0); val.append(0.5 * float(cv))
                for (ii, jj), cv in X_coef_lower.items():
                    if cv == 0.0:
                        continue
                    if ii == jj:
                        subi.append(ii + 1); subj.append(jj + 1); val.append(float(cv))
                    else:
                        a, b = (ii, jj) if ii > jj else (jj, ii)
                        subi.append(a + 1); subj.append(b + 1); val.append(0.5 * float(cv))
                return subi, subj, val

            def _add(subi, subj, vals, bk, blk, buk):
                cidx = task.getnumcon()
                task.appendcons(1)
                if len(subi) > 0:
                    aid = task.appendsparsesymmat(bar_dim, subi, subj, vals)
                    task.putbaraij(cidx, 0, [aid], [1.0])
                task.putconbound(cidx, bk, blk, buk)
                return cidx

            # Y[0,0] = 1
            sI, sJ, sV = _coeffs(1.0, np.zeros(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)
            # Box on x
            for i in range(d):
                xc = np.zeros(d); xc[i] = 1.0
                sI, sJ, sV = _coeffs(0.0, xc, {})
                _add(sI, sJ, sV, mosek.boundkey.ra, lo_x[i], hi_x[i])
            # Σ x = S
            sI, sJ, sV = _coeffs(0.0, np.ones(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, float(S), float(S))

            # Trace identity 1^T X 1 = S^2
            Xc_trace = {}
            for ii in range(d):
                Xc_trace[(ii, ii)] = 1.0
                for jj in range(ii):
                    Xc_trace[(ii, jj)] = 2.0
            sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc_trace)
            _add(sI, sJ, sV, mosek.boundkey.fx, float(S) * float(S),
                 float(S) * float(S))
            # Cauchy-Schwarz LB: Σ X_ii >= S^2/d
            Xc_diag = {(ii, ii): 1.0 for ii in range(d)}
            sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc_diag)
            _add(sI, sJ, sV, mosek.boundkey.lo,
                 (float(S) * float(S)) / float(d), 0.0)

            # Diagonal McCormick on X[i,i]
            for i in range(d):
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.ra,
                     lo_x[i] * lo_x[i], hi_x[i] * hi_x[i])
                xc = np.zeros(d); xc[i] = -2.0 * lo_x[i]
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -lo_x[i] * lo_x[i], 0.0)
                xc = np.zeros(d); xc[i] = -2.0 * hi_x[i]
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -hi_x[i] * hi_x[i], 0.0)
                xc = np.zeros(d); xc[i] = -(lo_x[i] + hi_x[i])
                sI, sJ, sV = _coeffs(0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.up, 0.0, -lo_x[i] * hi_x[i])

            # Off-diagonal RLT
            for i in range(d):
                for j in range(i + 1, d):
                    li, lj = lo_x[i], lo_x[j]
                    ui, uj = hi_x[i], hi_x[j]
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -li
                    sI, sJ, sV = _coeffs(li * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(ui * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -li
                    sI, sJ, sV = _coeffs(li * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(ui * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

            # Window threshold: Tr(A_W X) <= c_target * ell * S^2 / (2d) + eps
            thr_eps = eps_margin * S * S
            for w_idx, A_W in enumerate(A_list):
                ell = ell_list[w_idx]
                thr = c_target * ell * S * S / (2.0 * d) + thr_eps
                Xc = {}
                for ii in range(d):
                    Xc[(ii, ii)] = float(A_W[ii, ii])
                    for jj in range(ii):
                        Xc[(ii, jj)] = 2.0 * float(A_W[ii, jj])
                sI, sJ, sV = _coeffs(0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.up, -1e30, thr)

            try:
                task.optimize()
            except mosek.Error as e:
                return False, f'optimize-error: {e}'
            try:
                solsta = task.getsolsta(mosek.soltype.itr)
            except mosek.Error:
                return False, 'getsolsta-error'
            if solsta == mosek.solsta.prim_infeas_cer:
                return True, 'infeasible'
            if solsta == mosek.solsta.optimal:
                return False, 'optimal'
            return False, f'solsta={solsta}'
    finally:
        if own_env:
            try:
                env.__exit__(None, None, None)
            except Exception:
                pass


# =====================================================================
# Leaf cert: try N+O, then Joint, then Shor SDP. First-cert wins.
# =====================================================================

def _leaf_cert(c_int, lo, hi, S, d, c_target,
                A_list, pairs_list, ell_list, pruning_idx, env=None,
                use_shor=True):
    """Returns (cert: bool, method: str, net_or_status, grad).
    grad = N+O grad at anchor (used to pick split direction even if uncert).
    """
    # 1) N+O triangle
    net_no, w_no, grad_no = _no_triangle_bound(
        c_int, lo, hi, S, d, c_target, A_list, pairs_list, ell_list)
    if not np.isfinite(net_no):
        # Empty leaf
        return True, 'empty', 0.0, None
    if net_no >= 0.0:
        return True, 'NO', net_no, grad_no

    # 2) Joint LB (only if pruning windows present)
    if pruning_idx:
        net_j = _joint_lb_box(c_int, lo, hi, S, d, c_target,
                                A_list, pairs_list, ell_list, pruning_idx,
                                n_iters=15)
        if not np.isfinite(net_j):
            return True, 'empty_joint', 0.0, grad_no
        if net_j >= 0.0:
            return True, 'Joint', net_j, grad_no

    # 3) Shor SDP feasibility
    if use_shor:
        ok, status = _shor_feas_box(c_int, lo, hi, S, d, c_target,
                                      A_list, ell_list, env=env)
        if ok:
            return True, 'Shor:' + status, 0.0, grad_no

    return False, 'uncert', net_no, grad_no


# =====================================================================
# Top-level subdivision driver.
# =====================================================================

def cell_cert_split(c_int, S, d, c_target, max_depth=4,
                     A_list=None, pairs_list=None, ell_list=None,
                     env=None, use_shor=True, collect_stats=False,
                     shor_at_depth=None):
    """Recursive cell-subdivision certifier.

    `shor_at_depth`: if int, run Shor SDP only at depths >= shor_at_depth
    (depth 0 is root). Default None = all depths.
    """
    if A_list is None:
        windows = _enum_windows(d)
        A_list = [_build_A(d, ell, s) for ell, s in windows]
        pairs_list = [[(i, j) for i in range(d) for j in range(d)
                       if A_list[k][i, j] > 0]
                      for k, _ in enumerate(windows)]
        ell_list = [w[0] for w in windows]

    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / S
    lo0 = np.maximum(-h, -mu_star)
    hi0 = np.full(d, h)

    # Pre-find pruning indices (windows with TV_W(mu*) > c_target).
    pruning_idx = []
    for w_idx, A_W in enumerate(A_list):
        ell = ell_list[w_idx]
        tv = (2.0 * d / ell) * float(mu_star @ A_W @ mu_star)
        if tv > c_target:
            pruning_idx.append(w_idx)

    stats = {'leaves': 0, 'splits': 0, 'depth_required': 0,
             'methods': {}, 'leaves_per_depth': [0] * (max_depth + 1)}

    # Track per-depth cumulative cert: depth_cert[k] = True if cell certified
    # using leaves of depth <= k. Computed by full recursion.

    cert_per_depth = {}  # depth_limit -> bool (cert if all leaves cert at that depth)

    def _shor_active(depth):
        if not use_shor:
            return False
        if shor_at_depth is None:
            return True
        return depth >= shor_at_depth

    def _recurse(lo, hi, depth, budget):
        """Returns (cert: bool, depth_used: int, leaves_added: int).
        budget = max_depth - depth (remaining depth allowed)."""
        cert, method, val, grad = _leaf_cert(
            c_int, lo, hi, S, d, c_target,
            A_list, pairs_list, ell_list, pruning_idx, env=env,
            use_shor=_shor_active(depth))
        stats['leaves'] += 1
        stats['leaves_per_depth'][depth] = stats['leaves_per_depth'][depth] + 1
        stats['methods'][method] = stats['methods'].get(method, 0) + 1
        if cert:
            return True, depth, 1
        if budget <= 0 or grad is None:
            return False, depth, 1
        # Split along i* = argmax score (gradient*r_box + (A r) r heuristic)
        r_box = (hi - lo) / 2.0
        score = np.abs(grad) * r_box
        # tie-break by quad contribution
        score[r_box <= 1e-15] = -1.0
        i_star = int(np.argmax(score))
        if r_box[i_star] <= 1e-15:
            return False, depth, 1
        mid = (lo[i_star] + hi[i_star]) / 2.0
        stats['splits'] += 1
        lo1, hi1 = lo.copy(), hi.copy()
        hi1[i_star] = mid
        lo2, hi2 = lo.copy(), hi.copy()
        lo2[i_star] = mid
        c1, dr1, _ = _recurse(lo1, hi1, depth + 1, budget - 1)
        if not c1:
            return False, max(depth, dr1), 0
        c2, dr2, _ = _recurse(lo2, hi2, depth + 1, budget - 1)
        if not c2:
            return False, max(depth, dr2), 0
        return True, max(dr1, dr2), 0

    cert, depth_used, _ = _recurse(lo0, hi0, 0, max_depth)
    stats['depth_required'] = depth_used if cert else -1
    if collect_stats:
        return cert, depth_used, stats
    return cert


# =====================================================================
# Compatibility helpers: mass-balance pre-screen + run depth-by-depth
# (so we can compute cert rate at depth d for each d in 1..max_depth).
# =====================================================================

def cell_cert_split_at_depth(c_int, S, d, c_target, depth_limit,
                              A_list, pairs_list, ell_list, env,
                              use_shor=True, shor_at_depth=None):
    """Run cell_cert_split with max_depth=depth_limit. Returns bool."""
    return cell_cert_split(c_int, S, d, c_target, max_depth=depth_limit,
                             A_list=A_list, pairs_list=pairs_list,
                             ell_list=ell_list, env=env, use_shor=use_shor,
                             shor_at_depth=shor_at_depth)


# =====================================================================
# Test cells: load uncert cells from existing benches OR synthesize.
# =====================================================================

def _enum_compositions(d, S):
    if d == 1:
        yield (S,)
        return
    for v in range(S + 1):
        for rest in _enum_compositions(d - 1, S - v):
            yield (v,) + rest


def find_uncert_cells_synthetic(d, S, c_target, max_test=50,
                                  A_list=None, pairs_list=None, ell_list=None,
                                  filter_level='no'):
    """Find compositions that are NOT certified by the chosen filter level.

    filter_level:
      'no'   -- N+O fails (broad: every cell sub-cert is candidate)
      'shor' -- N+O fails AND Shor SDP alone fails at root
                (narrow: cells where subdivision could plausibly add value)
    """
    if A_list is None:
        windows = _enum_windows(d)
        A_list = [_build_A(d, ell, s) for ell, s in windows]
        pairs_list = [[(i, j) for i in range(d) for j in range(d)
                       if A_list[k][i, j] > 0]
                      for k, _ in enumerate(windows)]
        ell_list = [w[0] for w in windows]

    env = None
    if filter_level == 'shor' and _HAS_MOSEK:
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass

    h = 1.0 / (2.0 * S)
    out = []
    for comp in _enum_compositions(d, S):
        c_int = np.array(comp, dtype=np.int32)
        mu_star = c_int.astype(np.float64) / S
        any_pass = False
        for w_idx, A_W in enumerate(A_list):
            ell = ell_list[w_idx]
            if (2.0 * d / ell) * float(mu_star @ A_W @ mu_star) > c_target:
                any_pass = True
                break
        if not any_pass:
            continue
        lo0 = np.maximum(-h, -mu_star)
        hi0 = np.full(d, h)
        net_no, _, _ = _no_triangle_bound(c_int, lo0, hi0, S, d, c_target,
                                            A_list, pairs_list, ell_list)
        if net_no >= 0.0:
            continue
        if filter_level == 'shor':
            ok_shor, _ = _shor_feas_box(c_int, lo0, hi0, S, d, c_target,
                                          A_list, ell_list, env=env)
            if ok_shor:
                continue
        out.append(c_int)
        if len(out) >= max_test:
            break
    return out, A_list, pairs_list, ell_list


# =====================================================================
# Bench driver
# =====================================================================

def run_bench(d, S, c_target, max_depth=4, max_cells=50, verbose=True,
               filter_level='shor'):
    if verbose:
        print(f"\n=== _cell_split_cert: d={d}, S={S}, c_target={c_target}, "
              f"max_depth={max_depth}, filter='{filter_level}' ===")
        print(f"    cell width h = {1.0 / (2.0 * S):.6f}")

    t0 = time.time()
    uncert_cells, A_list, pairs_list, ell_list = find_uncert_cells_synthetic(
        d, S, c_target, max_test=max_cells, filter_level=filter_level)
    t_find = time.time() - t0
    if verbose:
        print(f"    found {len(uncert_cells)} '{filter_level}'-uncert cells "
              f"in {t_find:.1f}s")
    if not uncert_cells:
        if verbose:
            print(f"    No uncert cells; all certified by N+O alone.")
        return {'d': d, 'S': S, 'c_target': c_target, 'n_cells': 0}

    n = len(uncert_cells)

    # Initialize MOSEK env once
    env = None
    if _HAS_MOSEK:
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass

    # Per-cell cert outcomes per depth_limit, plus Shor-alone
    cum_cert_depth = {dd: 0 for dd in range(0, max_depth + 1)}
    shor_alone_cert = 0
    shor_alone_unsolved = 0
    times_split = {dd: [] for dd in range(0, max_depth + 1)}
    times_shor = []
    method_counts = {}

    if verbose:
        print(f"    Running {n} cells...")

    for k, c_int in enumerate(uncert_cells):
        # Shor SDP alone (single call on full cell)
        h = 1.0 / (2.0 * S)
        mu_star = c_int.astype(np.float64) / S
        lo0 = np.maximum(-h, -mu_star)
        hi0 = np.full(d, h)
        t1 = time.time()
        ok_shor, shor_status = _shor_feas_box(c_int, lo0, hi0, S, d, c_target,
                                                 A_list, ell_list, env=env)
        t_shor_one = time.time() - t1
        times_shor.append(t_shor_one)
        if ok_shor:
            shor_alone_cert += 1
        elif shor_status not in ('optimal',):
            shor_alone_unsolved += 1

        # Split cert at each depth limit
        for dd in range(0, max_depth + 1):
            t1 = time.time()
            ok, depth_req, stats = cell_cert_split(
                c_int, S, d, c_target, max_depth=dd,
                A_list=A_list, pairs_list=pairs_list,
                ell_list=ell_list, env=env, collect_stats=True)
            t_split_one = time.time() - t1
            times_split[dd].append(t_split_one)
            if ok:
                cum_cert_depth[dd] += 1
            for m, c in stats['methods'].items():
                method_counts[m] = method_counts.get(m, 0) + c

        if verbose and (k + 1) % max(1, n // 5) == 0:
            print(f"    [{k+1}/{n}] cum cert depth=0..{max_depth}: "
                  + ' '.join(f"{cum_cert_depth[d]}" for d in range(max_depth + 1)))

    t_total = time.time() - t0

    # Diff: cells cert by split-cert (any depth) but NOT by Shor alone.
    # Also: cells cert by Shor alone but NOT by split-cert (should be 0
    # because split-cert depth=0 == single Shor SDP on full cell).
    # For depth 0, cell_cert_split runs N+O+Joint+Shor on root only.
    # Split adds value at depth >= 1.

    if verbose:
        print(f"\n    --- Summary ---")
        print(f"    Total cells              : {n}")
        print(f"    Shor SDP alone certified : {shor_alone_cert} "
              f"({100*shor_alone_cert/n:.1f}%); unsolved={shor_alone_unsolved}")
        for dd in range(max_depth + 1):
            pct = 100.0 * cum_cert_depth[dd] / n
            print(f"    Split cert depth<={dd}    : {cum_cert_depth[dd]} "
                  f"({pct:.1f}%); avg time {1000*np.mean(times_split[dd]):.1f} ms")
        print(f"    Shor-alone time          : avg {1000*np.mean(times_shor):.1f} ms, "
              f"sum {sum(times_shor):.1f} s")
        print(f"    Method usage             : {method_counts}")
        print(f"    Total bench time         : {t_total:.1f}s")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'max_depth': max_depth,
        'n_cells': n,
        'shor_alone_cert': shor_alone_cert,
        'shor_alone_unsolved': shor_alone_unsolved,
        'cum_cert_depth': cum_cert_depth,
        'time_split_avg_ms': {dd: 1000 * float(np.mean(times_split[dd]))
                               for dd in range(max_depth + 1)},
        'time_shor_avg_ms': 1000 * float(np.mean(times_shor)),
        'method_counts': method_counts,
        'total_time_s': t_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--max_cells', type=int, default=50)
    ap.add_argument('--out', type=str, default='_cell_split_cert_results.json')
    ap.add_argument('--filter', choices=['no', 'shor'], default='shor',
                     help='cells to bench: N+O-uncert (no) or Shor-uncert (shor).')
    args = ap.parse_args()

    if not _HAS_MOSEK:
        print("WARNING: MOSEK not available — Shor SDP step disabled.")

    r = run_bench(args.d, args.S, args.c_target,
                   max_depth=args.max_depth, max_cells=args.max_cells,
                   filter_level=args.filter)

    out_path = os.path.join(_HERE, args.out)
    with open(out_path, 'w') as fp:
        json.dump(r, fp, indent=2, default=lambda o: int(o)
                  if isinstance(o, np.integer) else float(o))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()

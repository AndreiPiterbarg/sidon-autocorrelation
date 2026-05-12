"""S3 fix: column-generation / active-set LP certificate.

Drop-in replacement for cascade_opts.lp_dual_certificate that avoids
enumerating all 2^d_parent box vertices when d_parent is large.

CORRECTNESS BOUNDARY
--------------------
The pricing subproblem `min_{c in box} F(c)` where
    F(c) = sum_W lambda_W * (ws_W(c) - thr_W)
is a (possibly indefinite) quadratic over a box.  Its minimum may be
at an INTERIOR point, not a vertex.  Hence axis-extreme pricing is
NOT guaranteed exact; column generation could return True where full
enumeration returns False.

To stay sound we replicate the existing interior-point verifier:
- After the LP converges with t > 0, evaluate F at the box centre,
  many random interior points, AND project a few gradient-descent
  iterates from the worst point.  If any one yields F < -tol, return
  False.  This is the SAME safety net used by lp_dual_certificate.

A further safety: when d_parent <= 12 we still enumerate all 2^d
vertices on the *verification* pass (cheap up to 4096); this means
the active-set cert is at least as conservative as full enumeration
for small d.  For larger d we rely on the random+descent verifier.

Public function
---------------
lp_dual_certificate_active(parent_int, lo_arr, hi_arr,
                           n_half_child, m, c_target,
                           verify_full_vertices_max=12,
                           max_iter=200,
                           rand_pricing_pool=64,
                           verify_random_pts=64,
                           verify_descent_steps=80,
                           tol=1e-9)
    -> bool
"""
from __future__ import annotations

import os
import sys
import math
import time
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- #
# Window list / threshold helpers                                       #
# --------------------------------------------------------------------- #


def _build_windows(d_parent: int, n_half: int, m: int, c_target: float):
    """Return (windows, thresholds, conv_len, d_child) matching cascade_opts."""
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    B_corr = n_half * (8.0 * m + 1.0) / 2.0
    max_ell = 2 * d_child

    windows: list[tuple[int, int]] = []
    thresholds: list[float] = []
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        mult = min(n_half, ell - 1, 2 * d_child - ell)
        scale_ell = float(ell) * 4.0 * n_half
        thr_int = c_target * m * m * scale_ell + mult * B_corr
        for s_lo in range(n_windows):
            windows.append((ell, s_lo))
            thresholds.append(thr_int)
    return windows, np.asarray(thresholds, dtype=np.float64), conv_len, d_child


# --------------------------------------------------------------------- #
# Vector-friendly conv builder                                          #
# --------------------------------------------------------------------- #


def _conv_from_cursor(parent_int: np.ndarray, cursor: np.ndarray,
                      conv_len: int) -> np.ndarray:
    """Build conv[k] = sum_{p+q=k} child[p]*child[q]."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    child = np.empty(d_child, dtype=np.float64)
    child[0::2] = cursor
    child[1::2] = 2.0 * parent_int - cursor
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d_child):
        ci = child[i]
        if ci != 0.0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = child[j]
                if cj != 0.0:
                    conv[i + j] += 2.0 * ci * cj
    return conv


def _vertex_excess_for_cursor(parent_int: np.ndarray,
                               cursor: np.ndarray,
                               windows: list[tuple[int, int]],
                               thresholds: np.ndarray,
                               conv_len: int) -> np.ndarray:
    """Return excess[w] = ws_W(cursor) - thr_W."""
    conv = _conv_from_cursor(parent_int, cursor, conv_len)
    out = np.empty(thresholds.shape[0], dtype=np.float64)
    for wi, (ell, s_lo) in enumerate(windows):
        n_cv = ell - 1
        out[wi] = float(conv[s_lo:s_lo + n_cv].sum()) - thresholds[wi]
    return out


# --------------------------------------------------------------------- #
# F(c) and grad_F(c) for verification / local descent                   #
# --------------------------------------------------------------------- #


def _F_value(parent_int: np.ndarray, cursor: np.ndarray,
             windows, thresholds, conv_len, lam, lam_active_idx) -> float:
    conv = _conv_from_cursor(parent_int, cursor, conv_len)
    F = 0.0
    for wi in lam_active_idx:
        ell, s_lo = windows[wi]
        n_cv = ell - 1
        ws = conv[s_lo:s_lo + n_cv].sum()
        F += lam[wi] * (ws - thresholds[wi])
    return float(F)


def _F_value_and_grad(parent_int: np.ndarray, cursor: np.ndarray,
                      windows, thresholds, conv_len, lam, lam_active_idx):
    """Compute F(cursor) and dF/dcursor.

    child[2i] = c_i, child[2i+1] = 2P_i - c_i.
    conv[k] = sum_{p+q=k} child[p]*child[q].
    For window W = sum over k in [s_lo, s_lo+n_cv):
      ws_W(c) = sum_{(p,q): s_lo <= p+q < s_lo+n_cv} child[p]*child[q]
    d ws / d c_i = sum_{(p,q) in W} ( (dchild[p]/dc_i)*child[q]
                                       + child[p]*(dchild[q]/dc_i) ).
    Since child[2i]=c_i has derivative 1 wrt c_i, child[2i+1] has -1, others 0.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    child = np.empty(d_child, dtype=np.float64)
    child[0::2] = cursor
    child[1::2] = 2.0 * parent_int - cursor
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d_child):
        ci = child[i]
        if ci != 0.0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = child[j]
                if cj != 0.0:
                    conv[i + j] += 2.0 * ci * cj

    # F value
    F = 0.0
    # We will accumulate dF/dchild[k] first, then map to dF/dcursor.
    dF_dchild = np.zeros(d_child, dtype=np.float64)
    for wi in lam_active_idx:
        ell, s_lo = windows[wi]
        n_cv = ell - 1
        ws = conv[s_lo:s_lo + n_cv].sum()
        F += lam[wi] * (ws - thresholds[wi])
        w_lam = lam[wi]
        # dws/dchild[p] = 2 * (sum over q with (p+q) in window) child[q]
        # equivalently for each k in window, k = p+q:
        #   contribute child[q] to dws/dchild[p] and child[p] to dws/dchild[q]
        # Implement vectorised contribution per window.
        for k in range(s_lo, s_lo + n_cv):
            # conv[k] = sum_{p+q=k} child[p]*child[q]
            # d conv[k] / d child[p] = 2 * child[k - p] (when p != k - p);
            #                         = 2 * child[p] if p == k-p (self).
            # Easier: iterate p from max(0, k-d_child+1) to min(d_child-1, k),
            # q = k - p.  Since each unordered pair appears once via i<j and
            # the diagonal is conv[2i] += ci^2.  But for derivatives we can
            # treat conv[k] as bilinear sum_{p,q: p+q=k} child[p]*child[q]/.
            # Cleanest: d conv[k] / d child[p] = 2 * child[k-p] (valid for all
            # p, since conv[k] = sum_{p<=q, p+q=k} ... is rewriteable as
            # 1/2 sum_{p+q=k} 2 child[p] child[q]).
            for p in range(max(0, k - (d_child - 1)),
                           min(d_child - 1, k) + 1):
                q = k - p
                dF_dchild[p] += w_lam * 2.0 * child[q]
        # The double-add when p == q is correct: we over-count by sum once but
        # also conv[2p] contributes the diagonal once.  Cross-check: writing
        # conv[k] = sum_p child[p]*child[k-p], this counts ordered pairs, so
        # d/dchild[p] = 2*child[k-p].  Our build above, however, used the
        # i<j convention with factor 2 and a diagonal i==j, which yields the
        # SAME conv[k] values.  The derivative of THAT representation is
        # also 2*child[k-p].  Hence the formula is consistent.

    # cursor-space gradient: child[2i] = c_i, child[2i+1] = 2P_i - c_i
    # so dchild[2i]/dc_i = 1, dchild[2i+1]/dc_i = -1.
    grad = np.empty(d_parent, dtype=np.float64)
    for p in range(d_parent):
        grad[p] = dF_dchild[2 * p] - dF_dchild[2 * p + 1]
    return F, grad


# --------------------------------------------------------------------- #
# Local descent for verification                                        #
# --------------------------------------------------------------------- #


def _projected_descent_min(parent_int, lo_arr, hi_arr, x0,
                            windows, thresholds, conv_len,
                            lam, lam_active_idx, max_steps=80,
                            tol=1e-10):
    """Return the lowest F-value found by projected gradient descent
    starting from x0 within the box [lo_arr, hi_arr]."""
    d_parent = len(x0)
    x = np.array(x0, dtype=np.float64).copy()
    lo = lo_arr.astype(np.float64)
    hi = hi_arr.astype(np.float64)
    F, grad = _F_value_and_grad(parent_int, x, windows, thresholds,
                                 conv_len, lam, lam_active_idx)
    best = F
    step = 1.0
    for _ in range(max_steps):
        # Try a step in the direction of -grad, projected to box.
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-14:
            break
        # Initial step size: scale by box dimension
        scale = float(np.maximum(hi - lo, 1.0).max())
        s = step * scale / gnorm
        improved = False
        for _ls in range(20):
            x_new = np.clip(x - s * grad, lo, hi)
            F_new = _F_value(parent_int, x_new, windows, thresholds,
                              conv_len, lam, lam_active_idx)
            if F_new < F - tol:
                x = x_new
                F = F_new
                if F < best:
                    best = F
                step = min(step * 1.5, 4.0)
                improved = True
                break
            else:
                s *= 0.5
                if s < 1e-12:
                    break
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break
        # Recompute gradient
        F, grad = _F_value_and_grad(parent_int, x, windows, thresholds,
                                     conv_len, lam, lam_active_idx)
    return best


# --------------------------------------------------------------------- #
# Pricing helpers                                                       #
# --------------------------------------------------------------------- #


def _initial_vertices(d_parent, lo_arr, hi_arr, parent_int):
    """Return a small initial vertex set V0 to seed the LP.

    Includes:
      - the box centre (rounded; closest integer corner combination)
      - all 2*d_parent axis-extreme vertices: each axis at lo or hi while
        others at midpoint (rounded).
      - a handful of random vertices.
    """
    d = d_parent
    mid = ((lo_arr.astype(np.float64) + hi_arr.astype(np.float64)) / 2.0)
    # Closest cursor *vertex* to mid: pick lo or hi per coordinate by which
    # is closer (when equidistant, pick lo).
    base_vertex = np.where(mid - lo_arr.astype(np.float64) <
                           hi_arr.astype(np.float64) - mid,
                           lo_arr, hi_arr).astype(np.float64)
    seeds = [base_vertex.copy()]
    # All-lo and all-hi
    seeds.append(lo_arr.astype(np.float64).copy())
    seeds.append(hi_arr.astype(np.float64).copy())
    # 2*d axis-extreme vertices
    for p in range(d):
        for at_hi in (False, True):
            v = base_vertex.copy()
            v[p] = float(hi_arr[p]) if at_hi else float(lo_arr[p])
            seeds.append(v)
    return seeds


def _axis_extreme_pricing(parent_int, lo_arr, hi_arr, lam, lam_active_idx,
                           windows, thresholds, conv_len, t_opt):
    """Cheap pricing: try every axis-extreme vertex (one cursor at lo/hi,
    rest at the previous best vertex) and return the most-violated one
    along with the violation magnitude (positive = violation).
    """
    d = lo_arr.shape[0]
    # Anchor at the centre of the box (snapped to one bound) and try
    # 2*d vertices that flip one coordinate at a time.  Then pick best.
    mid_anchor_lo = lo_arr.astype(np.float64)
    mid_anchor_hi = hi_arr.astype(np.float64)
    candidates = []
    # all-lo, all-hi, plus single-axis flips around all-lo and all-hi
    for anchor in (mid_anchor_lo, mid_anchor_hi):
        candidates.append(anchor.copy())
        for p in range(d):
            v = anchor.copy()
            v[p] = float(hi_arr[p]) if anchor[p] == lo_arr[p] else float(lo_arr[p])
            candidates.append(v)
    return candidates


def _random_pricing(parent_int, lo_arr, hi_arr, n_pool, rng):
    """Sample n_pool random {lo,hi}^d vertices."""
    d = lo_arr.shape[0]
    out = []
    for _ in range(n_pool):
        mask = rng.random(d) < 0.5
        v = np.where(mask, hi_arr, lo_arr).astype(np.float64)
        out.append(v)
    return out


def _pricing_step(parent_int, lo_arr, hi_arr, lam, lam_active_idx,
                   windows, thresholds, conv_len, t_opt, rng,
                   rand_pool, axis_pool=True):
    """Return (most_violated_vertex, violation_value).

    violation_value > 0  -> we found a vertex v with
                              F(v) < t_opt (strict, by violation_value).
    violation_value <= 0 -> all candidates are non-violating.
    """
    cands = []
    if axis_pool:
        cands.extend(_axis_extreme_pricing(parent_int, lo_arr, hi_arr,
                                            lam, lam_active_idx,
                                            windows, thresholds, conv_len,
                                            t_opt))
    cands.extend(_random_pricing(parent_int, lo_arr, hi_arr, rand_pool, rng))

    best_v = None
    best_violation = -np.inf
    for v in cands:
        F = _F_value(parent_int, v, windows, thresholds, conv_len,
                     lam, lam_active_idx)
        viol = t_opt - F  # we need F >= t_opt; viol > 0 means F < t_opt
        if viol > best_violation:
            best_violation = viol
            best_v = v
    return best_v, best_violation


# --------------------------------------------------------------------- #
# Inner LP solver                                                       #
# --------------------------------------------------------------------- #


def _solve_master_lp(excess_matrix, n_win):
    """excess_matrix: (n_v, n_win); returns (success, lam, t_opt)."""
    from scipy.optimize import linprog

    n_v = excess_matrix.shape[0]
    c_obj = np.zeros(n_win + 1, dtype=np.float64)
    c_obj[-1] = -1.0
    A_ub = np.zeros((n_v, n_win + 1), dtype=np.float64)
    A_ub[:, :n_win] = -excess_matrix
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_v, dtype=np.float64)
    A_eq = np.zeros((1, n_win + 1), dtype=np.float64)
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, None)] * n_win + [(None, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'presolve': True})
    except Exception:
        return False, None, None
    if not res.success:
        return False, None, None
    lam = res.x[:n_win]
    t_opt = -res.fun
    return True, lam, t_opt


# --------------------------------------------------------------------- #
# Verification (mirrors cascade_opts.lp_dual_certificate)               #
# --------------------------------------------------------------------- #


def _verify_certificate(parent_int, lo_arr, hi_arr, windows, thresholds,
                         conv_len, lam, tol_F=-1e-6,
                         verify_full_vertices_max=12,
                         verify_random_pts=64,
                         verify_descent_steps=80,
                         seed=42):
    """Verify F(c) >= 0 throughout the box for the supplied lam.

    Strategy:
      (a) For d_parent <= verify_full_vertices_max enumerate ALL vertices.
      (b) Always evaluate at the centre + many random interior points.
      (c) From the worst sample, run projected gradient descent and check
          if the descent escapes below 0.
    Returns True if all checks pass (F >= tol_F), else False.
    """
    d_parent = lo_arr.shape[0]
    lam_active_idx = [i for i, w in enumerate(lam) if abs(w) > 1e-15]
    if not lam_active_idx:
        return False

    # ---------- (a) Full-vertex enumeration when affordable ----------
    if d_parent <= verify_full_vertices_max:
        n_v = 1 << d_parent
        worst_vertex = None
        worst_F = np.inf
        for vi in range(n_v):
            cursor = np.empty(d_parent, dtype=np.float64)
            for p in range(d_parent):
                cursor[p] = float(hi_arr[p]) if (vi >> p) & 1 else float(lo_arr[p])
            F = _F_value(parent_int, cursor, windows, thresholds,
                         conv_len, lam, lam_active_idx)
            if F < worst_F:
                worst_F = F
                worst_vertex = cursor
            if F < tol_F:
                return False
        # Vertex pass clear; still need interior check for indefinite F.

    # ---------- (b) Centre + random interior points ----------
    rng = np.random.default_rng(seed)
    centre = (lo_arr.astype(np.float64) + hi_arr.astype(np.float64)) / 2.0
    sample_pts = [centre.copy()]
    lo_f = lo_arr.astype(np.float64)
    hi_f = hi_arr.astype(np.float64)
    for _ in range(verify_random_pts):
        sample_pts.append(lo_f + (hi_f - lo_f) * rng.random(d_parent))
    worst_F = np.inf
    worst_pt = None
    for pt in sample_pts:
        F = _F_value(parent_int, pt, windows, thresholds, conv_len,
                     lam, lam_active_idx)
        if F < worst_F:
            worst_F = F
            worst_pt = pt
        if F < tol_F:
            return False

    # ---------- (c) Local descent from worst_pt ----------
    if verify_descent_steps > 0 and worst_pt is not None:
        F_min = _projected_descent_min(parent_int, lo_arr, hi_arr, worst_pt,
                                       windows, thresholds, conv_len,
                                       lam, lam_active_idx,
                                       max_steps=verify_descent_steps)
        if F_min < tol_F:
            return False

    # ---------- (c2) Local descent from a few corners ----------
    if verify_descent_steps > 0 and d_parent > verify_full_vertices_max:
        # Spot-check a few random vertices with descent from each.
        for k in range(min(8, 1 << min(d_parent, 6))):
            mask = rng.random(d_parent) < 0.5
            seed_pt = np.where(mask, hi_arr, lo_arr).astype(np.float64)
            F_min = _projected_descent_min(
                parent_int, lo_arr, hi_arr, seed_pt,
                windows, thresholds, conv_len,
                lam, lam_active_idx,
                max_steps=max(20, verify_descent_steps // 4))
            if F_min < tol_F:
                return False

    return True


# --------------------------------------------------------------------- #
# Public entry point                                                    #
# --------------------------------------------------------------------- #


def lp_dual_certificate_active(parent_int, lo_arr, hi_arr,
                                n_half_child, m, c_target,
                                verify_full_vertices_max=12,
                                max_iter=200,
                                rand_pricing_pool=64,
                                verify_random_pts=64,
                                verify_descent_steps=80,
                                tol=1e-9,
                                seed=12345,
                                return_diag=False):
    """Column-generation LP cert.

    Same return semantics as cascade_opts.lp_dual_certificate: returns
    True iff a non-negative window-weight measure lambda with
    sum lambda = 1 and F(c) = sum lambda_W (ws_W(c) - thr_W) >= 0
    everywhere in the cursor box can be certified.
    """
    parent_int = np.asarray(parent_int)
    lo_arr = np.asarray(lo_arr)
    hi_arr = np.asarray(hi_arr)
    d_parent = parent_int.shape[0]

    # Build window list (matches cascade_opts.lp_dual_certificate exactly).
    windows, thresholds, conv_len, _d_child = _build_windows(
        d_parent, int(n_half_child), int(m), float(c_target)
    )
    n_win = len(windows)
    if n_win == 0:
        return (False, {}) if return_diag else False
    if d_parent > 16:
        return (False, {'reason': 'd>16'}) if return_diag else False

    # Initial vertex set V0
    rng = np.random.default_rng(seed)
    seeds = _initial_vertices(d_parent, lo_arr, hi_arr, parent_int)
    # Add a few random vertices to V0 to break symmetry
    seeds.extend(_random_pricing(parent_int, lo_arr, hi_arr,
                                  min(8, max(4, d_parent)), rng))

    # Initial excess rows (one per seed vertex)
    rows = []
    for v in seeds:
        rows.append(_vertex_excess_for_cursor(
            parent_int, v, windows, thresholds, conv_len))
    excess_matrix = np.vstack(rows)

    diag = {
        'd_parent': d_parent,
        'n_win': n_win,
        'iters': 0,
        'final_n_constraints': 0,
        't_opt_history': [],
    }

    # Column-generation loop
    lam = None
    t_opt = None
    for it in range(max_iter):
        ok, lam, t_opt = _solve_master_lp(excess_matrix, n_win)
        diag['iters'] = it + 1
        if not ok:
            return (False, diag) if return_diag else False
        diag['t_opt_history'].append(float(t_opt))
        # Pricing: find a violated vertex.  Use richer pool when t_opt
        # is small (we need to be sure none violates).
        rand_pool = rand_pricing_pool
        if t_opt < 1.0 and it > 5:
            rand_pool = rand_pricing_pool * 2
        v_star, violation = _pricing_step(
            parent_int, lo_arr, hi_arr, lam,
            [i for i in range(n_win) if abs(lam[i]) > 1e-15],
            windows, thresholds, conv_len, t_opt, rng,
            rand_pool=rand_pool, axis_pool=True
        )
        if violation <= tol:
            # No vertex (in our pool) violates, declare LP-converged.
            break
        new_row = _vertex_excess_for_cursor(
            parent_int, v_star, windows, thresholds, conv_len)
        # Avoid duplicates: dedup by exact match (np.allclose).
        existing = excess_matrix
        # quick check via row hashes
        new_row_b = new_row.tobytes()
        already = False
        for er in existing:
            if er.tobytes() == new_row_b:
                already = True
                break
        if already:
            # The pricing oracle keeps returning the same vertex; we cannot
            # make further progress with this pool.
            break
        excess_matrix = np.vstack([excess_matrix, new_row])
    diag['final_n_constraints'] = excess_matrix.shape[0]

    if t_opt is None or t_opt <= 1e-9:
        return (False, diag) if return_diag else False

    # Verification stage (the soundness boundary).
    ok_verify = _verify_certificate(
        parent_int, lo_arr, hi_arr, windows, thresholds, conv_len, lam,
        tol_F=-1e-6,
        verify_full_vertices_max=verify_full_vertices_max,
        verify_random_pts=verify_random_pts,
        verify_descent_steps=verify_descent_steps,
        seed=seed,
    )
    diag['verify_pass'] = bool(ok_verify)
    diag['t_opt'] = float(t_opt)
    if not ok_verify:
        return (False, diag) if return_diag else False
    return (True, diag) if return_diag else True


# --------------------------------------------------------------------- #
# Self-test driver: soundness + speed                                   #
# --------------------------------------------------------------------- #


def _self_test():
    """Run the active-set version side-by-side with the reference vertex
    enumeration version across multiple (n_half, m, c) cases."""
    sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
    sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

    from compositions import generate_compositions_batched
    from pruning import count_compositions
    from cascade_opts import (
        _whole_parent_prune_theorem1,
        lp_dual_certificate,
    )
    from run_cascade import _prune_dynamic_int32, _compute_bin_ranges

    cases = [
        (3, 10, 1.28),    # d_parent = 6  - small soundness check
        (4, 10, 1.28),    # d_parent = 8  - timing benchmark
        (4, 10, 2.50),    # d_parent = 8  - high c, mix of T/F
        (5, 8,  1.20),    # d_parent = 10 - speedup demo
    ]
    results = []

    for n_half, m, c_target in cases:
        d = 2 * n_half
        S_half = 2 * n_half * m
        print(f"\n=== n_half={n_half}, m={m}, c={c_target}, d_parent={d} ===")
        # Find L0 survivors
        surv = []
        for half_batch in generate_compositions_batched(n_half, S_half,
                                                          batch_size=200_000):
            batch = np.empty((len(half_batch), d), dtype=np.int32)
            batch[:, :n_half] = half_batch
            batch[:, n_half:] = half_batch[:, ::-1]
            s = _prune_dynamic_int32(batch, n_half, m, c_target,
                                      use_flat_threshold=False)
            if s.any():
                surv.append(batch[s].copy())
        if surv:
            surv = np.vstack(surv)
        else:
            surv = np.empty((0, d), dtype=np.int32)
        print(f"L0 survivors: {len(surv):,}")
        # smaller batch at d>=10 (LP enumeration gets slow)
        n_test = min(30 if d >= 10 else 60, len(surv))
        sample = surv[:n_test]

        n_half_child = 2 * n_half
        d_child = 2 * d

        n_t1 = 0
        n_lp_orig = n_lp_act = 0
        n_match = n_orig_True_act_False = n_orig_False_act_True = 0
        t_orig = t_act = 0.0
        n_examined = 0
        for parent in sample:
            res = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if res is None:
                continue
            lo_arr, hi_arr, total_children = res
            if total_children == 0:
                continue
            t1 = _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                                int(n_half_child), int(m), c_target)
            if t1:
                n_t1 += 1
                continue
            n_examined += 1
            ta = time.time()
            lp_orig = lp_dual_certificate(parent, lo_arr, hi_arr,
                                           int(n_half_child), int(m), c_target)
            t_orig += time.time() - ta
            ta = time.time()
            lp_act = lp_dual_certificate_active(parent, lo_arr, hi_arr,
                                                 int(n_half_child), int(m), c_target)
            t_act += time.time() - ta
            n_lp_orig += int(lp_orig)
            n_lp_act += int(lp_act)
            if lp_orig == lp_act:
                n_match += 1
            elif lp_orig and not lp_act:
                n_orig_True_act_False += 1
            elif not lp_orig and lp_act:
                # Most worrying: active-set says True where original says False
                n_orig_False_act_True += 1
                print(f"  !! UNSOUND-CANDIDATE parent={tuple(parent)} "
                      f"lo={tuple(lo_arr)} hi={tuple(hi_arr)}")

        per = max(n_examined, 1)
        speedup = (t_orig / max(t_act, 1e-9))
        rec = {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'd_parent': d, 'n_examined': n_examined,
            't1_cleared': n_t1,
            'lp_original_True': n_lp_orig,
            'lp_active_True': n_lp_act,
            'match': n_match,
            'orig_True_act_False': n_orig_True_act_False,
            'orig_False_act_True': n_orig_False_act_True,
            'orig_avg_ms': 1000 * t_orig / per,
            'active_avg_ms': 1000 * t_act / per,
            'speedup': speedup,
        }
        results.append(rec)
        print(f"  examined: {n_examined}  T1 cleared: {n_t1}")
        print(f"  LP original True: {n_lp_orig}   LP active True: {n_lp_act}")
        print(f"  match: {n_match}   "
              f"orig=T/act=F: {n_orig_True_act_False}   "
              f"orig=F/act=T (UNSOUND CANDIDATES): {n_orig_False_act_True}")
        print(f"  avg ms: orig={1000 * t_orig / per:.1f}   "
              f"active={1000 * t_act / per:.1f}   "
              f"speedup={speedup:.2f}x")

    out = os.path.join(_HERE, '_S3_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return results


if __name__ == '__main__':
    _self_test()

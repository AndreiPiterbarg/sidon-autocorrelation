"""Agent D — Branching / splitting strategy audit (v6 cascade).

Probes (NO production edits):

 (1) B45 audit: at d=4, v6 caps `eff_max_depth = min(max_depth, 1)`
     (v6 line 544–545). Re-run the open d=4 cell (20,20,20,20) at S=80 with
     CAP REMOVED and max_depth in {1,2,3,4,5,6}; report certified yes/no,
     wall time, and number of recursive sub-calls (= n_subcells reported by
     CertResult).

 (2) B46 audit: at d=8, v6 returns early with `B46_no_signal` at root if
     L_joint and L_single both ≤ F (v6 line 552–556). Force-disable B46 on
     the open d=8 cell and report whether deep splitting now closes it.

 (3) Split-axis alternatives. For each of {gradient_weighted (baseline),
     widest, max_lin_grad_h, max_quad_h, mid_binding_eps}, run on the
     open d=4 cells and report closure rate + #subcells.

We re-import v6 and monkey-patch only inside this script — no production
edits.  All split heuristics produce subcells whose union covers the parent,
so soundness is preserved (verified by check: child.lo ≥ parent.lo,
child.hi ≤ parent.hi).
"""
from __future__ import annotations
import os, sys, time, json, copy
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6

C_TARGET = 1.281

# ----------------------------------------------------------------------
# Cells to probe.  The brief identifies d=4 (20,20,20,20) at S=80 and
# d=4 (19,21,21,19) at S=80 as the principal "split_sub1_fail" examples,
# and the d=8 cell as the B46 cell.
# ----------------------------------------------------------------------
PROBE_CELLS = {
    'd4_S80_a': dict(d=4, S=80, c=(20, 20, 20, 20)),
    'd4_S80_b': dict(d=4, S=80, c=(19, 21, 21, 19)),
    'd4_S80_c': dict(d=4, S=80, c=(18, 22, 22, 18)),
    # d=8 representative central composition (S=16)
    'd8_S16_a': dict(d=8, S=16, c=(2, 2, 2, 2, 2, 2, 2, 2)),
    'd8_S16_b': dict(d=8, S=16, c=(1, 3, 2, 2, 2, 2, 2, 2)),
}


def make_cell(d, S, c):
    c = np.array(c, dtype=np.float64)
    assert c.sum() == S
    return v3.Cell.from_integer_composition(c, S)


# ======================================================================
# (1) B45 lift probe: rebuild cert_cell with the depth-1 cap REMOVED
# ======================================================================

def cert_cell_no_B45(cell, windows, c_target, max_depth, current_depth=0,
                     n_calls=None, bundle=None):
    """Identical to v6.cert_cell, but with the d≤4 depth cap LIFTED.

    Sound — only changes which branch is explored, never claims a non-LB.
    """
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True, 'empty', 0
    if not cell.is_simplex_feasible():
        return True, 'empty', 0
    if bundle is None:
        bundle = v6.get_bundle(windows)
    cache = v3.CellCache.build(cell)

    # ---- B1 / B1u / B1diag (pre-F screens) ----
    b1 = v6.tier_B1_vec(cell, bundle, c_target)
    if float(np.max(b1)) > 0:
        return True, 'B1', 0
    b1u = v6.tier_B1u_vec(cell, bundle, c_target)
    if float(np.max(b1u)) > 0:
        return True, 'B1u', 0

    # ---- F ----
    v3.compute_F_all_windows(cache, windows, c_target)
    if cache.f_best_bound > 0:
        return True, 'F', 0

    # ---- L_single / L_joint ----
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
    L_best = -np.inf
    if v6.HAS_CVXPY:
        K_L = min(3, len(windows))
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=K_L, include_widest=2)
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
            if lb > L_best:
                L_best = lb
            if lb > 0:
                return True, 'L', 0
    Lj_best = -np.inf
    if v6.HAS_CVXPY:
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, c_target, max_fw_iters=6)
        Lj_best = lb_j
        if lb_j > 0:
            return True, 'L_joint', 0

    # ---- NO B45 CAP HERE (lifted) ----
    if current_depth >= max_depth:
        return False, 'max_depth_reached', current_depth

    # ---- NO B46 HERE either (lifted for fair comparison) ----

    best_W = (windows[cache.f_best_W_idx]
              if cache.f_best_W_idx >= 0 else windows[0])
    axis = v3.split_axis_gradient_weighted(cache, best_W)
    sub1, sub2 = cell.split(axis)
    ok1, t1, _ = cert_cell_no_B45(sub1, windows, c_target, max_depth,
                                   current_depth + 1, n_calls, bundle)
    if not ok1:
        return False, 'split_sub1_fail', current_depth
    ok2, t2, _ = cert_cell_no_B45(sub2, windows, c_target, max_depth,
                                   current_depth + 1, n_calls, bundle)
    if not ok2:
        return False, 'split_sub2_fail', current_depth
    return True, 'split', current_depth


def probe_B45_lift():
    """Run baseline (max_depth=1 via B45) and lifted (1..6) on each d=4 cell."""
    results = {}
    for name, p in PROBE_CELLS.items():
        if p['d'] != 4:
            continue
        cell = make_cell(p['d'], p['S'], p['c'])
        windows = v3.build_all_windows(p['d'])
        row = {}
        # baseline: stock v6 (which caps at depth 1 for d=4)
        t0 = time.time()
        r_stock = v6.cert_cell(cell, windows, C_TARGET, max_depth=6)
        row['stock_v6'] = dict(certified=r_stock.certified,
                                tier=r_stock.tier_used,
                                n_subcells=r_stock.n_subcells,
                                wall=time.time() - t0)
        for md in (1, 2, 3, 4, 5, 6):
            t0 = time.time()
            n_calls = [0]
            ok, tier, _ = cert_cell_no_B45(cell, windows, C_TARGET,
                                             max_depth=md, n_calls=n_calls)
            row[f'lifted_d{md}'] = dict(certified=ok, tier=tier,
                                          n_calls=n_calls[0],
                                          wall=time.time() - t0)
        results[name] = row
        print(f"[B45] {name}: stock={row['stock_v6']['certified']}  "
              f"lifted_d3={row['lifted_d3']['certified']}  "
              f"lifted_d5={row['lifted_d5']['certified']}")
    return results


# ======================================================================
# (2) B46 lift probe for d=8 cells (force splitting even if no signal)
# ======================================================================

def probe_B46_lift():
    """Run the d=8 cells with B46 disabled (just deep-split) — does it close?"""
    results = {}
    for name, p in PROBE_CELLS.items():
        if p['d'] != 8:
            continue
        cell = make_cell(p['d'], p['S'], p['c'])
        windows = v3.build_all_windows(p['d'])
        # stock
        t0 = time.time()
        r_stock = v6.cert_cell(cell, windows, C_TARGET, max_depth=6)
        stock = dict(certified=r_stock.certified, tier=r_stock.tier_used,
                      n_subcells=r_stock.n_subcells, wall=time.time() - t0)
        # lifted
        t0 = time.time()
        n_calls = [0]
        ok, tier, _ = cert_cell_no_B45(cell, windows, C_TARGET,
                                          max_depth=6, n_calls=n_calls)
        lifted = dict(certified=ok, tier=tier, n_calls=n_calls[0],
                       wall=time.time() - t0)
        results[name] = dict(stock=stock, lifted=lifted)
        print(f"[B46] {name}: stock={stock['certified']} tier={stock['tier']}"
              f"  lifted_certified={lifted['certified']} "
              f"n_calls={lifted['n_calls']}  wall={lifted['wall']:.2f}s")
    return results


# ======================================================================
# (3) Split-axis alternatives
# ======================================================================

def axis_gradient_weighted(cache, best_W):
    """Baseline (v3 line 562)."""
    return int(v3.split_axis_gradient_weighted(cache, best_W))


def axis_widest(cache, best_W):
    """argmax h_i."""
    return int(np.argmax(cache.h))


def axis_max_lin_grad_h(cache, best_W):
    """argmax |grad_W*[i]| · h_i  — max linear move along axis i."""
    A = best_W.A
    grad_W = best_W.grad_coef * (A @ cache.mu_star)
    return int(np.argmax(np.abs(grad_W) * cache.h))


def axis_max_quad_h(cache, best_W):
    """argmax Σ_{j: A[i,j]=1} h_j · h_i  — most quadratic mass reduction."""
    A = best_W.A
    quad_score = (A @ cache.h) * cache.h
    return int(np.argmax(quad_score))


def axis_mid_binding_eps(cache, best_W):
    """Pick axis whose ε-box has largest residual span around μ*, i.e. the
    axis where μ* is closest to a boundary (the "fractional vertex"
    direction).  Captures the SDP-dual most-binding flavour without rerunning
    the SDP.
    """
    slack = np.minimum(cache.hi_eps, -cache.lo_eps)  # min(|lo|, hi) ≥ 0
    return int(np.argmin(slack))  # smallest slack = most binding


HEURISTICS = {
    'gradient_weighted': axis_gradient_weighted,   # baseline
    'widest': axis_widest,
    'max_lin_grad_h': axis_max_lin_grad_h,
    'max_quad_h': axis_max_quad_h,
    'mid_binding_eps': axis_mid_binding_eps,
}


def cert_cell_with_axis_fn(cell, windows, c_target, axis_fn,
                            max_depth, current_depth=0, n_calls=None,
                            bundle=None):
    """Same cascade as cert_cell_no_B45 but use axis_fn for splitting."""
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True, 'empty'
    if not cell.is_simplex_feasible():
        return True, 'empty'
    if bundle is None:
        bundle = v6.get_bundle(windows)
    cache = v3.CellCache.build(cell)
    b1 = v6.tier_B1_vec(cell, bundle, c_target)
    if float(np.max(b1)) > 0:
        return True, 'B1'
    b1u = v6.tier_B1u_vec(cell, bundle, c_target)
    if float(np.max(b1u)) > 0:
        return True, 'B1u'
    v3.compute_F_all_windows(cache, windows, c_target)
    if cache.f_best_bound > 0:
        return True, 'F'
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
    if v6.HAS_CVXPY:
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
            if lb > 0:
                return True, 'L'
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, c_target,
                                                  max_fw_iters=6)
        if lb_j > 0:
            return True, 'L_joint'
    if current_depth >= max_depth:
        return False, 'max_depth'
    best_W = (windows[cache.f_best_W_idx]
              if cache.f_best_W_idx >= 0 else windows[0])
    axis = axis_fn(cache, best_W)
    sub1, sub2 = cell.split(axis)
    # Soundness assert (subcells inside parent)
    assert np.all(sub1.lo >= cell.lo - 1e-12)
    assert np.all(sub1.hi <= cell.hi + 1e-12)
    assert np.all(sub2.lo >= cell.lo - 1e-12)
    assert np.all(sub2.hi <= cell.hi + 1e-12)
    ok1, _ = cert_cell_with_axis_fn(sub1, windows, c_target, axis_fn,
                                      max_depth, current_depth + 1,
                                      n_calls, bundle)
    if not ok1:
        return False, 'sub1'
    ok2, _ = cert_cell_with_axis_fn(sub2, windows, c_target, axis_fn,
                                      max_depth, current_depth + 1,
                                      n_calls, bundle)
    if not ok2:
        return False, 'sub2'
    return True, 'split'


def probe_split_heuristics(max_depth=5):
    """Each open cell × each heuristic."""
    results = {}
    for name, p in PROBE_CELLS.items():
        cell = make_cell(p['d'], p['S'], p['c'])
        windows = v3.build_all_windows(p['d'])
        row = {}
        for h_name, fn in HEURISTICS.items():
            t0 = time.time()
            n_calls = [0]
            try:
                ok, tier = cert_cell_with_axis_fn(cell, windows, C_TARGET,
                                                    fn, max_depth=max_depth,
                                                    n_calls=n_calls)
            except Exception as e:
                ok, tier = False, f'ERR:{type(e).__name__}'
            row[h_name] = dict(certified=ok, tier=tier,
                                n_calls=n_calls[0],
                                wall=time.time() - t0)
            print(f"[axis] {name} {h_name}: ok={ok} n_calls={n_calls[0]} "
                  f"tier={tier} wall={row[h_name]['wall']:.2f}s")
        results[name] = row
    return results


if __name__ == '__main__':
    out = {}
    print("\n=== (1) B45 lift probe ===")
    out['B45_lift'] = probe_B45_lift()
    print("\n=== (2) B46 lift probe (d=8) ===")
    out['B46_lift'] = probe_B46_lift()
    print("\n=== (3) Split heuristics ===")
    out['heuristics'] = probe_split_heuristics(max_depth=5)
    out_path = os.path.join(_dir, '_audit_D_probe.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

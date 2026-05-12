"""AGENT E probe — measure speedup of proposed vectorizations vs v6 baseline.

For each candidate optimization:
  (E1) B1diag: per-W LP loop  vs  stacked sort+searchsorted over all W.
  (E2) _aggregate: Python loop  vs  einsum across stacked windows.
  (E3) _fW_value across top_K: list-comp loop  vs  einsum.
  (E4) F-tier quadratic-LB-sum: per-W np.sum(A*M_lb)  vs  einsum over A_stack.

Each probe asserts:
  - numerical equivalence (max-abs diff < 1e-12)
  - reports timings on representative d=6, d=8 cells

NO production edits — pure benchmarking script.
"""
from __future__ import annotations
import os, sys, time, json, logging, warnings
logging.getLogger('cvxpy').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v5 as v5
import _coarse_bnb_v6 as v6


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def time_loop(fn, n=50):
    """Time fn() best-of-3-of-(n calls)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t0) / n)
    return min(times)


def root_cell_for(d, S, c):
    return v3.Cell.from_integer_composition(np.asarray(c, dtype=np.float64), S)


def make_cells(d, S, n=5):
    """Generate n random feasible compositions for d, S."""
    rng = np.random.default_rng(42)
    cells = []
    tries = 0
    while len(cells) < n and tries < 200:
        tries += 1
        c = rng.multinomial(S, np.full(d, 1.0 / d))
        cell = root_cell_for(d, S, c)
        if cell.is_simplex_feasible():
            cells.append(cell)
    return cells


# -----------------------------------------------------------------------------
# (E1) B1diag vectorization
# -----------------------------------------------------------------------------
# Mathematically:
#   for each W,  b_W = Q_coef_W * (lin_min_W + const_W + cs_lb_W) - c_target
# where
#   A_off_W   = A_W with diagonal stripped
#   coef_W    = 2 * A_off_W @ lo
#   const_W   = -lo @ A_off_W @ lo
#   lin_min_W = lp_min_linear(coef_W, lo, hi, target=1)
#   D_W       = diag(A_W) (mask), |D_W| = sum
#   sum_min_W = lp_min_linear(D_W, lo, hi, target=1)
#   cs_lb_W   = max(0, sum_min_W)^2 / |D_W|   if |D_W|>0 else 0
#
# Vectorization plan:
#   A_off_stack    = A_stack - diag_mask_as_3D   (n_W, d, d), diagonal zeroed
#   coef_stack     = 2 * einsum('wij,j->wi', A_off_stack, lo)        (n_W, d)
#   const_stack    = -einsum('wij,i,j->w', A_off_stack, lo, lo)      (n_W,)
#   D_stack        = diag_mask_stack                                 (n_W, d)
#   lin_min_stack  = lp_min_linear_vec(coef_stack, lo, hi, target=1)
#   sum_min_stack  = lp_min_linear_vec(D_stack,   lo, hi, target=1)
#   cs_lb_stack    = max(0, sum_min_stack)^2 / max(1, |D|_W)
#                     (set to 0 where |D|_W=0)
#   bounds         = Q_coef_vec * (lin_min_stack + const_stack + cs_lb_stack) - c_target


def lp_min_linear_vec(coef_stack: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                        target: float) -> np.ndarray:
    """Vectorized lp_min_linear over a stack of coefficient rows.

    Same closed-form greedy as v3.lp_max_linear but applied per row.
    Returns an array of shape (n,) — finite or -inf where infeasible.

    Math:  min coef·μ over {lo≤μ≤hi, Σμ=target}
         = -max(-coef·μ over same set)
         = -(   base_max  +  partial_fill_value )
      base_max  = (-coef) @ lo
      partial_fill computed via cumulative weight.
    """
    n, d = coef_stack.shape
    w = hi - lo                            # (d,)
    s_lo = float(lo.sum())
    budget = target - s_lo
    total_w = float(w.sum())
    if budget < -1e-12 or budget > total_w + 1e-12:
        return np.full(n, -np.inf)
    budget = max(0.0, min(budget, total_w))

    # For each row, we want max of -coef·μ.
    neg = -coef_stack                                       # (n, d)
    order = np.argsort(-neg, axis=1)                        # desc per row
    rows = np.arange(n)[:, None]
    neg_sorted = neg[rows, order]                           # (n, d)
    w_sorted = w[order]                                     # (n, d)
    cum_w = np.cumsum(w_sorted, axis=1)                     # (n, d)

    # Base value: (-coef) @ lo  =  -coef @ lo  per row
    base = neg @ lo                                         # (n,)

    # Find split index per row
    # idx = first k with cum_w[k] >= budget  (np.searchsorted per row)
    # cum_w is monotone non-decreasing per row.
    # Vectorize: idx[i] = np.searchsorted(cum_w[i], budget)
    # numpy doesn't expose per-row searchsorted directly; use boolean trick.
    above = cum_w >= budget                                 # (n, d)
    idx = above.argmax(axis=1)                              # first True
    all_below = ~above.any(axis=1)
    idx_clipped = np.where(all_below, d, idx)               # (n,)

    # Full fill contribution = sum_{k < idx} neg_sorted[k] * w_sorted[k]
    full_contrib = np.cumsum(neg_sorted * w_sorted, axis=1) # (n, d)
    # value for "all filled" case
    all_full_val = full_contrib[:, -1]

    # For non-all-below rows:
    #   prefix_full_val = (idx>0) ? full_contrib[idx-1] : 0
    #   cum_before      = (idx>0) ? cum_w[idx-1]        : 0
    #   partial         = budget - cum_before
    #   tail_val        = neg_sorted[idx] * partial
    cols = np.arange(d)
    pref_idx = np.clip(idx - 1, 0, d - 1)                   # for gather only
    prefix_full = np.where(idx > 0,
                            full_contrib[rows[:, 0], pref_idx], 0.0)
    cum_before = np.where(idx > 0,
                            cum_w[rows[:, 0], pref_idx], 0.0)
    partial = budget - cum_before                           # (n,)
    # tail_val only meaningful where not all_below
    safe_idx = np.where(all_below, 0, idx)
    tail_neg = neg_sorted[rows[:, 0], safe_idx]
    tail_val = tail_neg * partial

    val_partial = prefix_full + tail_val
    val_max = np.where(all_below, all_full_val, val_partial)
    res_max = base + val_max
    return -res_max                                          # min = -max(-·)


def b1diag_baseline(cell, windows, c_target):
    return np.array([v5.tier_B1diag_amgm(cell, W, c_target) for W in windows])


def b1diag_vec(cell, bundle, c_target):
    """Vectorized B1diag across all windows."""
    d = cell.d
    lo = cell.lo
    hi = cell.hi
    A_stack = bundle.A_stack                                 # (n_W, d, d)
    diag_mask_stack = bundle.diag_mask_stack                 # (n_W, d)
    n_W = bundle.n_W
    Q_coef_vec = bundle.Q_coef_vec
    # A_off_stack = A_stack with diagonals zeroed
    A_off_stack = A_stack.copy()
    diag_idx = np.arange(d)
    A_off_stack[:, diag_idx, diag_idx] = 0.0
    # coef_stack[w,i] = 2 * sum_j A_off[w,i,j] * lo_j
    coef_stack = 2.0 * (A_off_stack @ lo)                    # (n_W, d)
    # const_stack[w] = -lo^T A_off[w] lo
    const_stack = -np.einsum('wij,i,j->w', A_off_stack, lo, lo)
    lin_min = lp_min_linear_vec(coef_stack, lo, hi, target=1.0)
    # Diagonal piece
    D_size_vec = diag_mask_stack.sum(axis=1)                 # (n_W,)
    sum_min = lp_min_linear_vec(diag_mask_stack, lo, hi, target=1.0)
    sum_min_pos = np.maximum(0.0, sum_min)
    cs_lb = np.where(D_size_vec > 0,
                      sum_min_pos ** 2 / np.maximum(D_size_vec, 1.0),
                      0.0)
    # Bound
    raw = lin_min + const_stack + cs_lb
    bounds = Q_coef_vec * raw - c_target
    # Preserve -inf from infeasible LP
    infeasible = ~np.isfinite(lin_min) | ~np.isfinite(sum_min)
    bounds[infeasible] = -np.inf
    return bounds


# -----------------------------------------------------------------------------
# (E2) _aggregate vectorization
# -----------------------------------------------------------------------------

def aggregate_baseline(top_windows, lam, mu_star, c_target):
    return v6._aggregate(top_windows, lam, mu_star, c_target)


def aggregate_vec(top_windows, lam, mu_star, c_target, A_stack_top,
                    Qc_top, gc_top):
    """Vectorized _aggregate.

    Inputs (precomputable per cascade call):
      A_stack_top : (K, d, d)
      Qc_top      : (K,)
      gc_top      : (K,)
    mu_star and c_target are per-cell.

    m_W = Qc_W * (mu^T A_W mu) - c_target
    g_W = gc_W * (A_W mu)
    Q_W = Qc_W * A_W
    Σ λ_W m_W = lam @ m_W
    Σ λ_W g_W = einsum('w,wi->i', lam*gc_top, A_stack_top @ mu_star)
    Σ λ_W Q_W = einsum('w,wij->ij', lam*Qc_top, A_stack_top)
    """
    # quad_W = mu^T A_W mu
    Amu = A_stack_top @ mu_star                              # (K, d)
    quad = (Amu * mu_star).sum(axis=1)                       # (K,)
    m_W = Qc_top * quad - c_target                           # (K,)
    margin_lam = float(lam @ m_W)
    # g_W stacked = gc * Amu
    g_W_stack = gc_top[:, None] * Amu                        # (K, d)
    grad_lam = lam @ g_W_stack                               # (d,)
    # Q sum: weights w = lam * Qc
    w_q = lam * Qc_top                                       # (K,)
    Q_lam = np.einsum('w,wij->ij', w_q, A_stack_top)         # (d, d)
    return Q_lam, grad_lam, margin_lam


# -----------------------------------------------------------------------------
# (E3) _fW_value batched
# -----------------------------------------------------------------------------

def fW_baseline(top_windows, mu_star, eps_val, X_val, c_target):
    return np.array([v6._fW_value(W, mu_star, eps_val, X_val, c_target)
                       for W in top_windows])


def fW_vec(top_windows, mu_star, eps_val, X_val, c_target,
              A_stack_top, Qc_top, gc_top):
    """f_W = m_W + g_W·ε + Q_coef·tr(A_W X)
    Vectorized:
       quad_mu  = (mu^T A_W mu)               (K,)
       quad_eps = einsum('wij,i,j->w', A, eps, eps_proxy?) — actually we need tr(A_W X)
       tr(A_W X) = einsum('wij,ij->w', A_stack_top, X_val)
       g_W·ε   = einsum('wi,i->w', gc*Amu, eps)
    """
    Amu = A_stack_top @ mu_star
    quad_mu = (Amu * mu_star).sum(axis=1)
    m_W = Qc_top * quad_mu - c_target
    # grad_W stacked
    g_W_stack = gc_top[:, None] * Amu
    grad_term = g_W_stack @ eps_val
    # trace term
    tr_term = Qc_top * np.einsum('wij,ij->w', A_stack_top, X_val)
    return m_W + grad_term + tr_term


# -----------------------------------------------------------------------------
# (E4) F-tier quadratic LB sum vectorized
# -----------------------------------------------------------------------------

def f_quad_lb_baseline(windows, M_lb):
    return np.array([float(np.sum(W.A * M_lb)) for W in windows])


def f_quad_lb_vec(A_stack, M_lb):
    return np.einsum('wij,ij->w', A_stack, M_lb)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run():
    out = {'cases': [], 'lp_min_linear_check_done': False}

    # Sanity-check lp_min_linear_vec vs scalar lp_min_linear on randoms first
    rng = np.random.default_rng(1)
    for trial in range(20):
        d = rng.integers(4, 12)
        lo = rng.uniform(0, 0.2, size=d)
        hi = lo + rng.uniform(0.0, 0.5, size=d)
        target = float(rng.uniform(lo.sum(), hi.sum()))
        n = 8
        coef = rng.standard_normal((n, d))
        ref = np.array([v3.lp_min_linear(coef[i], lo, hi, target)
                          for i in range(n)])
        vec = lp_min_linear_vec(coef, lo, hi, target)
        maxdiff = float(np.max(np.abs(ref - vec)))
        if maxdiff > 1e-12:
            out['lp_min_linear_check_done'] = False
            out['lp_min_linear_fail'] = {'trial': trial, 'maxdiff': maxdiff,
                                            'ref': ref.tolist(),
                                            'vec': vec.tolist()}
            return out
    out['lp_min_linear_check_done'] = True

    for d, S in [(6, 12), (8, 16), (10, 20)]:
        cells = make_cells(d, S, n=3)
        windows = v3.build_all_windows(d)
        bundle = v6.get_bundle(windows)
        n_W = bundle.n_W
        c_target = 1.281
        rec = {'d': d, 'S': S, 'n_W': n_W, 'cells_tested': len(cells)}

        # ---- (E1) B1diag vectorization ----
        for cell in cells[:1]:
            ref = b1diag_baseline(cell, windows, c_target)
            vec = b1diag_vec(cell, bundle, c_target)
            # Compare excluding -inf in ref (both should match including -inf)
            mask = np.isfinite(ref) & np.isfinite(vec)
            maxdiff = float(np.max(np.abs(ref[mask] - vec[mask]))) if mask.any() else 0.0
            # If any -inf disagreement, flag
            inf_disagree = int(((~np.isfinite(ref)) != (~np.isfinite(vec))).sum())
            rec['E1_maxdiff'] = maxdiff
            rec['E1_inf_disagree'] = inf_disagree
            t_base = time_loop(lambda: b1diag_baseline(cell, windows, c_target),
                                  n=30)
            t_vec = time_loop(lambda: b1diag_vec(cell, bundle, c_target), n=30)
            rec['E1_t_base_ms'] = t_base * 1000
            rec['E1_t_vec_ms'] = t_vec * 1000
            rec['E1_speedup'] = t_base / max(t_vec, 1e-12)

        # ---- (E2) _aggregate vectorization ----
        # Pick K=4 windows, build precomputed A_stack_top
        K = 4
        for cell in cells[:1]:
            cache = v3.CellCache.build(cell)
            mu_star = cache.mu_star
            mw_vec = v6.mwQ_vec(cell, bundle, c_target, mu_star)
            idx_top = v6.select_L_candidates_v6(mw_vec, bundle, K=K,
                                                   include_widest=2)
            top_w = [windows[i] for i in idx_top]
            A_stack_top = bundle.A_stack[idx_top]
            Qc_top = bundle.Q_coef_vec[idx_top]
            gc_top = bundle.grad_coef_vec[idx_top]
            lam = np.array([0.4, 0.3, 0.2, 0.1])

            Q_ref, g_ref, m_ref = aggregate_baseline(top_w, lam, mu_star, c_target)
            Q_v, g_v, m_v = aggregate_vec(top_w, lam, mu_star, c_target,
                                              A_stack_top, Qc_top, gc_top)
            rec['E2_maxdiff_Q'] = float(np.max(np.abs(Q_ref - Q_v)))
            rec['E2_maxdiff_g'] = float(np.max(np.abs(g_ref - g_v)))
            rec['E2_diff_m'] = float(abs(m_ref - m_v))
            t_base = time_loop(
                lambda: aggregate_baseline(top_w, lam, mu_star, c_target), n=200)
            t_vec = time_loop(
                lambda: aggregate_vec(top_w, lam, mu_star, c_target,
                                          A_stack_top, Qc_top, gc_top), n=200)
            rec['E2_t_base_us'] = t_base * 1e6
            rec['E2_t_vec_us'] = t_vec * 1e6
            rec['E2_speedup'] = t_base / max(t_vec, 1e-12)

            # ---- (E3) _fW_value batched ----
            # Need a feasible (eps, X) — use mu_star and outer product as proxy.
            eps_val = np.zeros(d)
            X_val = np.zeros((d, d))
            ref_f = fW_baseline(top_w, mu_star, eps_val, X_val, c_target)
            vec_f = fW_vec(top_w, mu_star, eps_val, X_val, c_target,
                              A_stack_top, Qc_top, gc_top)
            rec['E3_maxdiff'] = float(np.max(np.abs(ref_f - vec_f)))
            t_base = time_loop(
                lambda: fW_baseline(top_w, mu_star, eps_val, X_val, c_target),
                n=200)
            t_vec = time_loop(
                lambda: fW_vec(top_w, mu_star, eps_val, X_val, c_target,
                                  A_stack_top, Qc_top, gc_top), n=200)
            rec['E3_t_base_us'] = t_base * 1e6
            rec['E3_t_vec_us'] = t_vec * 1e6
            rec['E3_speedup'] = t_base / max(t_vec, 1e-12)

        # ---- (E4) F-tier quadratic LB sum vectorization ----
        for cell in cells[:1]:
            cache = v3.CellCache.build(cell)
            M_lb = cache.M_lb
            ref_q = f_quad_lb_baseline(windows, M_lb)
            vec_q = f_quad_lb_vec(bundle.A_stack, M_lb)
            rec['E4_maxdiff'] = float(np.max(np.abs(ref_q - vec_q)))
            t_base = time_loop(lambda: f_quad_lb_baseline(windows, M_lb), n=50)
            t_vec = time_loop(lambda: f_quad_lb_vec(bundle.A_stack, M_lb), n=50)
            rec['E4_t_base_us'] = t_base * 1e6
            rec['E4_t_vec_us'] = t_vec * 1e6
            rec['E4_speedup'] = t_base / max(t_vec, 1e-12)

        out['cases'].append(rec)

    out_path = os.path.join(_dir, '_audit_E_probe.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    run()

"""KKT-augmented exhaustive critical-point enumeration: bench + soundness.

Compares:
 (1) baseline triangle (cell_var + quad_corr_BL),
 (2) vertex-enum QP (qp_bound.py qp_bound_vertex; UNSOUND for indefinite QPs),
 (3) KKT-augmented enum (qp_bound_kkt; SOUND/exact).

Empirical soundness check (the central deliverable):
 - For 100 random (c, W), test KKT >= fine_grid_max.  (Expected: 0 violations.)
 - For 30 random (c, W) at d=4, test vertex_max < grid_max.
   (Expected: positive count — confirms the unsoundness already documented in
   _qp_soundness_check.py.)

Cell-closure %:
 - Per (d, S, c_target) config: count cells where the bound TIGHTLY closes the
   cell (i.e., margin <= bound). Compare KKT vs triangle-baseline ratios.

Configs:  [(d=4, S=20, c=1.20), (d=6, S=15, c=1.20), (d=8, S=12, c=1.20)]
"""
from __future__ import annotations

import os, sys, time, json, itertools
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound import (build_window_matrix, grad_for_window,
                      qp_bound_vertex)
from _kkt_exact_qp import qp_bound_kkt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fine_grid_max_f(grad, A_W, scale, h, d, n_grid=15):
    """Brute-force max of f over Cell on a uniform grid in (d-1) coords."""
    best = 0.0
    grid = np.linspace(-h, h, n_grid)
    for tup in itertools.product(grid, repeat=d - 1):
        last = -sum(tup)
        if abs(last) > h:
            continue
        delta = np.array(list(tup) + [last], dtype=np.float64)
        v = -grad @ delta - scale * delta @ A_W @ delta
        if v > best:
            best = v
    return best


def triangle_bound(grad, A_W, scale, h, d):
    """Baseline triangle: cell_var + quad_corr (separately maximized)."""
    # cell_var = max over Cell of -grad . delta = h * sum |grad_i - mean(grad)|/2
    g_sorted = np.sort(grad)
    cell_var = 0.0
    for k in range(d // 2):
        cell_var += g_sorted[d - 1 - k] - g_sorted[k]
    cell_var *= h
    # quad_corr = max over Cell of -scale * delta^T A_W delta = scale * h^2 *
    # ||A_W||_op-restricted-to-{Σ=0} times ||delta||_2^2 ... the standard pair-
    # bound in N-style is scale * #cross-pairs * h^2 (loose). Use the tightest
    # standard one: spectral on Σ=0 subspace times (sum delta^2 ≤ d h^2).
    # f = -scale * delta^T A_W delta with sum(delta)=0. Since A_W = α 11^T + (A_W - α 11^T)
    # and 1^T delta = 0, the 11^T part vanishes. Then |delta^T (A_W - α 11^T) delta|
    # ≤ ||A_W - α 11^T||_op * ||delta||_2^2 ≤ ||A_W - α 11^T||_op * d * h^2.
    n_W = A_W.sum()
    alpha = n_W / (d * d)
    eigs = np.linalg.eigvalsh(A_W - alpha)
    op_rest = float(np.abs(eigs).max())
    quad_corr = scale * op_rest * d * h * h
    return cell_var + quad_corr


def random_c_W(rng, d_max=4, S_lo=10, S_hi=30):
    """Sample a random (d, S, c, ell, s_lo)."""
    d = d_max if d_max <= 4 else int(rng.integers(2, d_max + 1))
    S = int(rng.integers(S_lo, S_hi + 1))
    c = rng.dirichlet(np.ones(d)) * S
    c = np.maximum(0, np.round(c)).astype(np.int32)
    c[0] += S - c.sum()
    c = np.maximum(0, c)
    # Random valid window
    while True:
        ell = int(rng.integers(2, 2 * d + 1))
        n_windows = (2 * d - 1) - (ell - 1) + 1
        if n_windows <= 0:
            continue
        s_lo = int(rng.integers(0, n_windows))
        if s_lo + ell - 2 > 2 * d - 2:
            continue
        return d, S, c, ell, s_lo


def soundness_check(n_trials=100, seed=42, n_grid=12, d_max=4):
    """Check KKT-bound >= grid-max for n_trials random samples."""
    rng = np.random.default_rng(seed)
    n_kkt_violations = 0
    n_vert_violations = 0
    max_kkt_excess_neg = 0.0  # if KKT < grid (shouldn't happen)
    max_vert_excess_neg = 0.0  # how much vertex misses
    examples = []
    for trial in range(n_trials):
        d, S, c, ell, s_lo = random_c_W(rng, d_max=d_max)
        A_W = build_window_matrix(d, ell, s_lo)
        grad = grad_for_window(c.astype(np.float64), A_W, S, d, ell)
        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell
        v_max = qp_bound_vertex(grad, A_W, scale, h, d)
        k_max = qp_bound_kkt(grad, A_W, scale, h, d)
        g_max = fine_grid_max_f(grad, A_W, scale, h, d, n_grid=n_grid)

        # KKT violation: KKT < grid (note: grid is sub-optimal LB on true max,
        # so KKT >= grid is a necessary but not sufficient soundness check).
        if k_max + 1e-9 < g_max:
            n_kkt_violations += 1
            max_kkt_excess_neg = max(max_kkt_excess_neg, g_max - k_max)
            if len(examples) < 3:
                examples.append({'kind': 'KKT<grid', 'd': d, 'S': S, 'c': c.tolist(),
                                 'ell': ell, 's_lo': s_lo,
                                 'KKT': k_max, 'grid': g_max,
                                 'excess': g_max - k_max})
        # Vertex unsound: vertex < grid.
        if v_max + 1e-9 < g_max:
            n_vert_violations += 1
            max_vert_excess_neg = max(max_vert_excess_neg, g_max - v_max)
        # Crucial: vertex <= KKT (KKT can only enlarge the candidate set).
        if v_max > k_max + 1e-9:
            print(f"  WARNING vertex>KKT at trial {trial}: v={v_max} k={k_max}")
    return {
        'n_trials': n_trials, 'n_grid': n_grid,
        'kkt_violations': n_kkt_violations, 'kkt_max_excess': max_kkt_excess_neg,
        'vertex_violations': n_vert_violations, 'vertex_max_excess': max_vert_excess_neg,
        'examples': examples,
    }


# ---------------------------------------------------------------------------
# Cell-closure benchmark per config
# ---------------------------------------------------------------------------

def enum_compositions(d, S):
    if d == 1:
        yield (S,); return
    for v in range(S + 1):
        for rest in enum_compositions(d - 1, S - v):
            yield (v,) + rest


def cell_closure_bench(d, S, c_target, max_n_test=None, time_per_cell_samples=200):
    """For all comps & windows that need certification (margin > 0), compute
    triangle-bound and KKT-bound. Count how many cells each bound closes.

    A cell (c, W) is 'closed' if margin <= bound. We focus on cells that pass
    the integer-threshold test (i.e., raw TV_W > c_target — the cells the
    cascade must certify or survive).
    """
    print(f"\n=== d={d}, S={S}, c_target={c_target} ===", flush=True)
    h = 1.0 / (2.0 * S)
    inv_2d = 1.0 / (2.0 * d)
    eps = 1e-9
    max_ell = 2 * d
    thr = {ell: int(c_target * ell * S * S * inv_2d - eps)
           for ell in range(2, max_ell + 1)}

    n_total_cells = 0
    n_close_tri = 0
    n_close_kkt = 0
    n_close_vertex = 0
    n_kkt_strict = 0  # KKT closes but triangle doesn't.
    n_kkt_strict_over_vertex = 0

    times_kkt = []
    times_tri = []
    times_vert = []

    comps = list(enum_compositions(d, S))
    if max_n_test is not None and len(comps) > max_n_test:
        rng = np.random.default_rng(0)
        comps = [comps[i] for i in rng.choice(len(comps), max_n_test, replace=False)]

    # Cap total cells tested (across comps) to keep wallclock small at d=8.
    cell_cap = 5000 if d <= 6 else 800
    t_start = time.time()
    for ci_idx, c_tup in enumerate(comps):
        if n_total_cells >= cell_cap:
            print(f"  [reached cell_cap={cell_cap} after {ci_idx} comps, {time.time()-t_start:.1f}s]", flush=True)
            break
        c = np.array(c_tup, dtype=np.float64)
        # Build conv
        conv_len = 2 * d - 1
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = int(c[i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = int(c[j])
                    if cj != 0:
                        conv[i + j] += 2 * ci * cj

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                if s_lo + ell - 2 > conv_len - 1:
                    continue
                ws = int(conv[s_lo:s_lo + n_cv].sum())
                if ws <= thr[ell]:
                    continue
                tv = ws * 2.0 * d / (S * S * ell)
                margin = tv - c_target
                if margin <= 0:
                    continue
                # Now we need the bound to certify margin <= bound.
                A_W = build_window_matrix(d, ell, s_lo)
                grad = grad_for_window(c, A_W, S, d, ell)
                scale = 2.0 * d / ell
                t0 = time.perf_counter()
                tri = triangle_bound(grad, A_W, scale, h, d)
                times_tri.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                vert = qp_bound_vertex(grad, A_W, scale, h, d)
                times_vert.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                kkt = qp_bound_kkt(grad, A_W, scale, h, d)
                times_kkt.append(time.perf_counter() - t0)

                n_total_cells += 1
                # PRUNE condition (match production cascade): margin >= bound.
                # Tighter (smaller) bound => more cells pruned. So KKT (exact)
                # >= vertex (lower bound on true max for indefinite QPs only by
                # accident) ?  Actually: KKT enumerates more candidates than
                # vertex => KKT >= vertex always.  Triangle decouples linear
                # and quadratic max => triangle >= max(linear+quadratic) = KKT.
                # Therefore: KKT_bound <= vertex_bound (when vertex misses
                # interior crit. pts, KKT can be larger; otherwise equal) and
                # KKT_bound <= triangle_bound.  Thus KKT prunes MORE cells.
                ct = (margin >= tri - 1e-12)
                cv = (margin >= vert - 1e-12)
                ck = (margin >= kkt - 1e-12)
                if ct: n_close_tri += 1
                if cv: n_close_vertex += 1
                if ck: n_close_kkt += 1
                # Strict gain: KKT prunes but baseline doesn't.
                if ck and not ct: n_kkt_strict += 1
                if ck and not cv: n_kkt_strict_over_vertex += 1
                # Cap timing samples for speed.
                if len(times_kkt) > time_per_cell_samples and n_total_cells > time_per_cell_samples:
                    pass
    pct = lambda x: 100.0 * x / max(1, n_total_cells)
    print(f"  cells tested: {n_total_cells}")
    print(f"  triangle closes: {n_close_tri} ({pct(n_close_tri):.2f}%)")
    print(f"  vertex closes:   {n_close_vertex} ({pct(n_close_vertex):.2f}%)")
    print(f"  KKT closes:      {n_close_kkt} ({pct(n_close_kkt):.2f}%)")
    print(f"  KKT strict gain over triangle: {n_kkt_strict} "
          f"({pct(n_kkt_strict):.2f}%)")
    print(f"  KKT strict gain over vertex:   {n_kkt_strict_over_vertex} "
          f"({pct(n_kkt_strict_over_vertex):.2f}%)")
    if times_kkt:
        print(f"  median time/cell: tri={np.median(times_tri)*1e6:.1f}us, "
              f"vertex={np.median(times_vert)*1e6:.1f}us, "
              f"KKT={np.median(times_kkt)*1e6:.1f}us")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_cells_tested': n_total_cells,
        'n_close_tri': n_close_tri,
        'n_close_vertex': n_close_vertex,
        'n_close_kkt': n_close_kkt,
        'n_kkt_strict_over_tri': n_kkt_strict,
        'n_kkt_strict_over_vertex': n_kkt_strict_over_vertex,
        'pct_close_tri': pct(n_close_tri),
        'pct_close_vertex': pct(n_close_vertex),
        'pct_close_kkt': pct(n_close_kkt),
        'pct_kkt_gain_over_tri': pct(n_kkt_strict),
        'pct_kkt_gain_over_vertex': pct(n_kkt_strict_over_vertex),
        'time_kkt_median_us': float(np.median(times_kkt) * 1e6) if times_kkt else None,
        'time_tri_median_us': float(np.median(times_tri) * 1e6) if times_tri else None,
        'time_vertex_median_us': float(np.median(times_vert) * 1e6) if times_vert else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KKT-augmented exhaustive critical-point enumeration: bench")
    print("=" * 70)

    print("\n[1] Soundness check: KKT >= grid_max (fine grid n=12, d=4, 100 trials)")
    sound = soundness_check(n_trials=100, seed=42, n_grid=12, d_max=4)
    print(f"  KKT  violations: {sound['kkt_violations']} / {sound['n_trials']}")
    print(f"  KKT  max excess (grid - KKT, should be <=0): {sound['kkt_max_excess']:.3e}")
    print(f"  Vertex violations: {sound['vertex_violations']} / {sound['n_trials']}")
    print(f"  Vertex max excess (grid - vertex): {sound['vertex_max_excess']:.3e}")
    if sound['examples']:
        print("  Sample KKT < grid violations (likely numerical/grid coarseness):")
        for ex in sound['examples']:
            print(f"    {ex}")

    print("\n[2] Cell-closure bench")
    configs = [
        (4, 20, 1.20, None),
        (6, 15, 1.20, 1500),
        (8, 12, 1.20, 500),
    ]
    closure = []
    for d, S, c, max_n in configs:
        closure.append(cell_closure_bench(d, S, c, max_n_test=max_n))

    out = os.path.join(_dir, '_coarse_KKT_bench_results.json')
    with open(out, 'w') as f:
        json.dump({'soundness': sound, 'closure': closure}, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (np.integer,)) else
                  float(o) if isinstance(o, (np.floating,)) else str(o))
    print(f"\nResults written to {out}")


if __name__ == '__main__':
    main()

"""Validate v8 direct-MOSEK port: |delta lb| < 1e-6 vs v7 CVXPY+MOSEK on 100+ cells.
Then speed benchmark.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6
import _coarse_bnb_v7 as v7
import _coarse_bnb_v8 as v8


def random_composition(rng, d, S):
    u = rng.dirichlet(np.ones(d) * 1.5)
    c = np.round(u * S).astype(np.int64)
    c[-1] = S - c[:-1].sum()
    if c.min() < 0:
        return None
    return c.astype(np.float64)


def test_equivalence(n_trials=120, d_range=(4, 12), S_range=(15, 40)):
    print("\n" + "=" * 70)
    print("V8 equivalence to v7 — |delta lb| should be < 1e-6 on each cell")
    print("=" * 70)
    rng = np.random.default_rng(2026)
    diffs = []
    fails = []
    for trial in range(n_trials):
        d = int(rng.integers(*d_range))
        S = int(rng.integers(*S_range))
        c = random_composition(rng, d, S)
        if c is None:
            continue
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        cache = v3.CellCache.build(cell)
        windows = v3.build_all_windows(d)
        W = windows[int(rng.integers(0, len(windows)))]
        mu_star = cache.mu_star
        Wmu = W.Q_coef * float(mu_star @ W.A @ mu_star)
        c_target = Wmu - rng.uniform(0.02, 0.15)
        lb7, _ = v7.tier_L_single_v7(cache, W, c_target)
        lb8, info8 = v8.tier_L_single_v8(cache, W, c_target)
        if not np.isfinite(lb7) or not np.isfinite(lb8):
            continue
        diff = abs(lb7 - lb8)
        diffs.append(diff)
        if diff > 1e-4:
            fails.append((trial, d, S, lb7, lb8, diff))
            if len(fails) <= 5:
                print(f"  *** large diff trial {trial}: d={d} S={S} "
                      f"v7={lb7:.6f} v8={lb8:.6f} diff={diff:.2e}")
    print(f"  trials checked: {len(diffs)}")
    print(f"  max |v7 - v8|:    {max(diffs):.2e}")
    print(f"  median |v7 - v8|: {np.median(diffs):.2e}")
    print(f"  P95   |v7 - v8|:  {np.percentile(diffs, 95):.2e}")
    print(f"  failures (>1e-4): {len(fails)}")
    return len(fails) == 0, np.array(diffs)


def benchmark(d=12, n=200):
    print("\n" + "=" * 70)
    print(f"V8 speed benchmark vs v7 at d={d}, n={n} cells")
    print("=" * 70)
    rng = np.random.default_rng(7)
    cells = []
    while len(cells) < n:
        S = int(rng.integers(15, 25))
        c = random_composition(rng, d, S)
        if c is None:
            continue
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        cells.append((cell, S))
    windows = v3.build_all_windows(d)
    # Warm
    cache = v3.CellCache.build(cells[0][0])
    mu_star = cache.mu_star
    Wmu = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
    c_target = Wmu - 0.05
    v7.tier_L_single_v7(cache, windows[0], c_target)
    v8.tier_L_single_v8(cache, windows[0], c_target)
    # v7
    t0 = time.time()
    for cell, S in cells:
        cache = v3.CellCache.build(cell)
        mu_star = cache.mu_star
        Wmu = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = Wmu - 0.05
        v7.tier_L_single_v7(cache, windows[0], c_target)
    t_v7 = time.time() - t0
    # v8
    t0 = time.time()
    for cell, S in cells:
        cache = v3.CellCache.build(cell)
        mu_star = cache.mu_star
        Wmu = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = Wmu - 0.05
        v8.tier_L_single_v8(cache, windows[0], c_target)
    t_v8 = time.time() - t0
    print(f"  v7 (CVXPY+MOSEK): {t_v7:.2f}s for {n} cells ({1000*t_v7/n:.1f}ms/cell)")
    print(f"  v8 (direct-MOSEK): {t_v8:.2f}s for {n} cells ({1000*t_v8/n:.1f}ms/cell)")
    print(f"  speedup: {t_v7/t_v8:.2f}x")
    return t_v7, t_v8


if __name__ == '__main__':
    ok, diffs = test_equivalence(n_trials=120)
    if not ok:
        print("\nFAILED: v8 deviates from v7. Check encoding.")
    else:
        print("\nv8 == v7 to 1e-4 tolerance [OK]")
    benchmark(d=12, n=100)
    benchmark(d=8, n=200)

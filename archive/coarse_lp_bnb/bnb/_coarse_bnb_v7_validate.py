"""Validate v7:
  V1. Imports + numerical equivalence: v7 L_single is at least as tight as v6
      (signing cut can only tighten). For most cells L_v7 >= L_v6 within MOSEK tolerance.
  V2. tier_BD soundness: for cells with zero coordinates, BD bound is sound
      (verified by MC sampling within the cell).
  V3. Skip-SDP gate soundness: when gate fires, the cell is genuinely uncertifiable
      by SDP (verified by running SDP anyway and checking lb <= 0).
  V4. Quick end-to-end speed comparison v6 vs v7 on d=12 sample.
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


def random_composition(rng, d, S):
    u = rng.dirichlet(np.ones(d) * 1.5)
    c = np.round(u * S).astype(np.int64)
    c[-1] = S - c[:-1].sum()
    if c.min() < 0:
        return None
    return c.astype(np.float64)


def test_V1_L_single_equiv():
    print("\n" + "=" * 70)
    print("V1: v7 L_single >= v6 L_single (signing cuts tighten)")
    print("=" * 70)
    rng = np.random.default_rng(101)
    n_trials = 12
    diffs = []
    for trial in range(n_trials):
        d = int(rng.integers(4, 8))
        S = int(rng.integers(20, 40))
        c = random_composition(rng, d, S)
        if c is None:
            continue
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        cache = v3.CellCache.build(cell)
        windows = v3.build_all_windows(d)
        for _ in range(3):
            W = windows[int(rng.integers(0, len(windows)))]
            mu_star = cache.mu_star
            Wmu = W.Q_coef * float(mu_star @ W.A @ mu_star)
            c_target = Wmu - rng.uniform(0.02, 0.1)
            lb6, _ = v6.tier_L_single_v6(cache, W, c_target)
            lb7, _ = v7.tier_L_single_v7(cache, W, c_target)
            diffs.append(lb7 - lb6)
    print(f"  trials: {len(diffs)}")
    print(f"  v7 - v6: min={min(diffs):.2e} mean={np.mean(diffs):.2e} "
          f"max={max(diffs):.2e}")
    # Sign should be >= 0 (tighter LB is better). Allow MOSEK tol.
    bad = [d for d in diffs if d < -1e-6]
    print(f"  v7 < v6 violations (LB regressed): {len(bad)}/{len(diffs)}")
    print("  V1 PASSED [OK]" if len(bad) == 0 else "  V1 NEEDS REVIEW")


def test_V2_BD_soundness():
    print("\n" + "=" * 70)
    print("V2: tier_BD soundness via MC on boundary cells")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n_trials = 8
    n_check = 0
    n_unsound = 0
    for trial in range(n_trials):
        d = int(rng.integers(6, 10))
        S = int(rng.integers(16, 30))
        # Force a boundary composition
        c = random_composition(rng, d, S)
        if c is None:
            continue
        # Force 1-3 zeros
        n_zeros = int(rng.integers(1, 4))
        for _ in range(n_zeros):
            non_zero = [i for i in range(d) if c[i] > 0]
            if not non_zero:
                break
            zero_idx = int(rng.choice(non_zero))
            donor_idx = int(rng.choice([i for i in non_zero if i != zero_idx])) if len(non_zero) > 1 else zero_idx
            if zero_idx == donor_idx:
                continue
            c[donor_idx] += c[zero_idx]
            c[zero_idx] = 0
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        windows = v3.build_all_windows(d)
        bundle = v6.get_bundle(windows)
        # Pick c_target a bit below max TV(μ*)
        cache = v3.CellCache.build(cell)
        mu_star = cache.mu_star
        max_TV_center = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = max_TV_center - 0.05
        bd_lb, bd_W = v7.tier_BD(cell, windows, c_target, bundle)
        if not np.isfinite(bd_lb):
            continue
        # MC samples to verify min over cell of max_W f_W >= bd_lb
        n = 2000
        from _coarse_bnb_v5_validate import sample_eps_in_cell, max_W_f_at_eps
        eps_samples = sample_eps_in_cell(cell, mu_star, n, rng)
        if len(eps_samples) == 0:
            continue
        f_max = max_W_f_at_eps(eps_samples, mu_star, windows, c_target)
        mc_min_of_max = float(f_max.min())
        n_check += 1
        # Soundness: bd_lb must NOT exceed the true min over the cell of max_W f_W
        if bd_lb > mc_min_of_max + 1e-5:
            n_unsound += 1
            print(f"  *** BD UNSOUND trial {trial}: BD={bd_lb:.6f} mc_min_max={mc_min_of_max:.6f}")
    print(f"  trials checked: {n_check}, unsound: {n_unsound}")
    print("  V2 PASSED [OK]" if n_unsound == 0 else "  V2 FAILED")


def test_V3_skip_gate():
    print("\n" + "=" * 70)
    print("V3: skip-SDP gate is conservative (only skips truly hopeless cells)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    n_check = 0
    n_skipped = 0
    n_should_not_skip = 0
    for trial in range(10):
        d = int(rng.integers(6, 10))
        S = int(rng.integers(16, 30))
        c = random_composition(rng, d, S)
        if c is None:
            continue
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        windows = v3.build_all_windows(d)
        bundle = v6.get_bundle(windows)
        cache = v3.CellCache.build(cell)
        mu_star = cache.mu_star
        max_TV = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = max_TV - 0.05
        v3.compute_F_all_windows(cache, windows, c_target)
        F_best = cache.f_best_bound
        if F_best > 0:
            continue
        n_check += 1
        skip = v7.should_skip_SDP(cell, bundle, c_target, F_best)
        if skip:
            n_skipped += 1
            # Run SDP anyway, verify it gives negative LB
            best_W_idx = int(np.argmax(
                [W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows]))
            lb, _ = v6.tier_L_single_v6(cache, windows[best_W_idx], c_target)
            if lb > 0:
                n_should_not_skip += 1
                print(f"  *** SKIP WRONG trial {trial}: F={F_best:.4f} but SDP={lb:.4f}>0")
    print(f"  trials: {n_check}, skip-gate fired: {n_skipped}, "
          f"wrong skips: {n_should_not_skip}")
    print("  V3 PASSED [OK]" if n_should_not_skip == 0 else "  V3 FAILED")


def test_V4_speed():
    print("\n" + "=" * 70)
    print("V4: end-to-end speed v6 vs v7 on small d=8 sample")
    print("=" * 70)
    d = 8; S = 16; c_target = 1.25
    rng = np.random.default_rng(99)
    samples = []
    while len(samples) < 30:
        u = rng.dirichlet(np.ones(d) * 1.0)
        c = np.round(u * S).astype(np.int64)
        c[-1] = S - c[:-1].sum()
        if c.min() < 0:
            continue
        samples.append(c.astype(np.float64))
    windows = v3.build_all_windows(d)
    # warm
    v6.certify_composition(samples[0], S, d, c_target, windows=windows, max_depth=2)
    v7.certify_composition(samples[0], S, d, c_target, windows=windows, max_depth=2)
    t0 = time.time()
    c6 = {}
    for c in samples:
        r = v6.certify_composition(c, S, d, c_target, windows=windows, max_depth=3)
        c6[r.tier_used] = c6.get(r.tier_used, 0) + 1
    t_v6 = time.time() - t0
    t0 = time.time()
    c7 = {}
    for c in samples:
        r = v7.certify_composition(c, S, d, c_target, windows=windows, max_depth=3)
        c7[r.tier_used] = c7.get(r.tier_used, 0) + 1
    t_v7 = time.time() - t0
    print(f"  v6 ({t_v6:.2f}s): {c6}")
    print(f"  v7 ({t_v7:.2f}s): {c7}")
    print(f"  v7/v6 ratio: {t_v6/t_v7:.2f}x")


if __name__ == '__main__':
    test_V1_L_single_equiv()
    test_V2_BD_soundness()
    test_V3_skip_gate()
    test_V4_speed()
    print("\nVALIDATION COMPLETE")

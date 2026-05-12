"""Validate _coarse_bnb_v5.py:
  V1. Monte-Carlo SOUNDNESS of tier_L_joint_lagrangian: bound ≤ min over cell of
      max_W f_W (the certification target).  Also compare to v4's UNSOUND L_joint.
  V2. Lagrangian dominance: v5 L_joint LB ≥ max_W L_single LB (warm start
      guarantees this).
  V3. End-to-end closure stats at d=8 S=16 on a small sample, c_target=1.275.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v4 as v4
import _coarse_bnb_v5 as v5


def random_cell(rng, d_range=(4, 8), S_range=(20, 80)):
    d = int(rng.integers(*d_range))
    S = int(rng.integers(*S_range))
    u = rng.dirichlet(np.ones(d) * rng.uniform(0.5, 3.0))
    c = np.round(u * S).astype(np.int64)
    c[-1] = S - c[:-1].sum()
    if c.min() < 0:
        return None
    return d, S, c.astype(np.float64)


def sample_eps_in_cell(cell, mu_star, n, rng):
    lo_eps = np.maximum(cell.lo - mu_star, -mu_star)
    hi_eps = cell.hi - mu_star
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        e = rng.uniform(lo_eps, hi_eps)
        e = e - e.mean()
        if np.all(e >= lo_eps - 1e-12) and np.all(e <= hi_eps + 1e-12):
            if np.all(mu_star + e >= -1e-12):
                out.append(e)
    return np.array(out)


def max_W_f_at_eps(eps_samples, mu_star, windows, c_target):
    """For each sample ε, return max_W f_W(ε)."""
    if len(eps_samples) == 0:
        return np.array([])
    mu = mu_star[np.newaxis, :] + eps_samples
    fvals = []
    for W in windows:
        tv = W.Q_coef * np.einsum('ni,ij,nj->n', mu, W.A, mu)
        fvals.append(tv - c_target)
    return np.max(np.stack(fvals, axis=0), axis=0)


# =========================================================
# V1. SOUNDNESS of v5 L_joint, and demonstrate v4 unsoundness
# =========================================================

def test_V1_soundness():
    print("\n" + "=" * 70)
    print("V1: Monte-Carlo soundness of v5 L_joint (and v4 unsoundness demo)")
    print("=" * 70)
    rng = np.random.default_rng(2026_05_11)
    n_trials = 12
    v5_violations = 0
    v4_unsound_cases = 0
    v5_lbs, v4_lbs, mc_maxmins = [], [], []
    for trial in range(n_trials):
        spec = random_cell(rng)
        if spec is None:
            continue
        d, S, c = spec
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        windows = v3.build_all_windows(d)
        mu_star = cell.center
        # Pick c_target near tightness
        Wmu = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = Wmu - rng.uniform(0.02, 0.10)
        cache = v3.CellCache.build(cell)
        v3.compute_F_all_windows(cache, windows, c_target)
        # v5 L_joint (sound)
        candidates_j = v5.select_L_candidates_v5(cache, windows, c_target,
                                                   K=4, include_widest=2)
        top_j = [windows[i] for i in candidates_j]
        lb_v5, info_v5 = v5.tier_L_joint_lagrangian(cache, top_j, c_target,
                                                       K=len(top_j),
                                                       max_fw_iters=4)
        # v4 L_joint (epigraph; possibly unsound)
        lb_v4 = v3.tier_L_joint(cache, windows, c_target, K=4)
        if lb_v4 is None:
            lb_v4 = -np.inf
        # MC: true min over cell of max_W f_W
        eps_samples = sample_eps_in_cell(cell, mu_star, 4000, rng)
        if len(eps_samples) == 0:
            continue
        f_max = max_W_f_at_eps(eps_samples, mu_star, windows, c_target)
        mc_min_of_max = float(f_max.min())  # this is what cert requires > 0
        v5_lbs.append(lb_v5)
        v4_lbs.append(float(lb_v4))
        mc_maxmins.append(mc_min_of_max)
        # SOUNDNESS check: lb ≤ mc_min_of_max
        if lb_v5 > mc_min_of_max + 1e-5:
            v5_violations += 1
            print(f"  *** V5 SOUNDNESS VIOLATION trial {trial}: "
                  f"v5_lb={lb_v5:.6f} mc_min_max={mc_min_of_max:.6f} "
                  f"d={d} S={S}")
        if lb_v4 > mc_min_of_max + 1e-5:
            v4_unsound_cases += 1
            print(f"  *** V4 unsound (as expected) trial {trial}: "
                  f"v4_lb={lb_v4:.6f} mc_min_max={mc_min_of_max:.6f} "
                  f"d={d} S={S}")
    print(f"\n  v5 violations:           {v5_violations} / {len(v5_lbs)}")
    print(f"  v4 unsound counterexamples: {v4_unsound_cases} / {len(v5_lbs)}")
    print(f"  v5 LB range: [{min(v5_lbs):.4f}, {max(v5_lbs):.4f}]")
    print(f"  v4 LB range: [{min(v4_lbs):.4f}, {max(v4_lbs):.4f}]")
    print(f"  MC min-of-max range: [{min(mc_maxmins):.4f}, {max(mc_maxmins):.4f}]")
    assert v5_violations == 0, f"v5 unsound on {v5_violations} cells"
    print("  V1 PASSED [OK]  (v5 sound; v4 demonstrated unsound)")


# =========================================================
# V2. Lagrangian dominance:  v5 L_joint LB ≥ best L_single
# =========================================================

def test_V2_dominance():
    print("\n" + "=" * 70)
    print("V2: v5 L_joint dominates best L_single (warm-start guarantees this)")
    print("=" * 70)
    rng = np.random.default_rng(7)
    n_trials = 10
    fail = 0
    diffs = []
    for trial in range(n_trials):
        spec = random_cell(rng, d_range=(4, 7), S_range=(30, 70))
        if spec is None:
            continue
        d, S, c = spec
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        windows = v3.build_all_windows(d)
        mu_star = cell.center
        Wmu = max(W.Q_coef * float(mu_star @ W.A @ mu_star) for W in windows)
        c_target = Wmu - rng.uniform(0.03, 0.10)
        cache = v3.CellCache.build(cell)
        v3.compute_F_all_windows(cache, windows, c_target)
        candidates_j = v5.select_L_candidates_v5(cache, windows, c_target,
                                                   K=4, include_widest=2)
        top_j = [windows[i] for i in candidates_j]
        lb_v5, info = v5.tier_L_joint_lagrangian(cache, top_j, c_target,
                                                    K=len(top_j),
                                                    max_fw_iters=5)
        best_single = info['best_single_lb']
        diff = lb_v5 - best_single
        diffs.append(diff)
        if diff < -1e-6:
            fail += 1
            print(f"  *** trial {trial}: v5 below best-single: "
                  f"v5={lb_v5:.6f} best_single={best_single:.6f}")
    print(f"  failures: {fail} / {len(diffs)}")
    if diffs:
        print(f"  v5 - best_single: min={min(diffs):.2e}  "
              f"mean={np.mean(diffs):.2e}  max={max(diffs):.2e}")
    assert fail == 0, "v5 L_joint failed to dominate L_single"
    print("  V2 PASSED [OK]")


# =========================================================
# V3. End-to-end closure stats at d=8 S=16, c=1.275 (the audited run)
# =========================================================

def test_V3_d8_S16_sample():
    print("\n" + "=" * 70)
    print("V3: d=8 S=16 c=1.275 closure on a small sample (compare v4 vs v5)")
    print("=" * 70)
    d = 8
    S = 16
    c_target = 1.275
    windows = v3.build_all_windows(d)
    rng = np.random.default_rng(123)
    # Sample 40 compositions; bias toward near-uniform (hard) cases
    samples = []
    while len(samples) < 40:
        u = rng.dirichlet(np.ones(d) * 2.0)  # bias toward uniform
        c = np.round(u * S).astype(np.int64)
        c[-1] = S - c[:-1].sum()
        if c.min() < 0:
            continue
        samples.append(c.astype(np.float64))
    v4_counts = {}
    v5_counts = {}
    t_v4 = t_v5 = 0.0
    for c in samples:
        t0 = time.time()
        r4 = v4.certify_composition(c, S, d, c_target, windows=windows,
                                       max_depth=3)
        t_v4 += time.time() - t0
        v4_counts[r4.tier_used] = v4_counts.get(r4.tier_used, 0) + 1
        t0 = time.time()
        r5 = v5.certify_composition(c, S, d, c_target, windows=windows,
                                       max_depth=3)
        t_v5 += time.time() - t0
        v5_counts[r5.tier_used] = v5_counts.get(r5.tier_used, 0) + 1
    print(f"  v4 ({t_v4:.1f}s): {v4_counts}")
    print(f"  v5 ({t_v5:.1f}s): {v5_counts}")
    n = len(samples)
    v4_open = sum(c for t, c in v4_counts.items() if not (t in {'B1', 'B1u', 'B1diag',
                                                                  'F', 'L', 'L_joint',
                                                                  'split', 'empty'}))
    v5_open = sum(c for t, c in v5_counts.items() if not (t in {'B1', 'B1u', 'B1diag',
                                                                  'F', 'L', 'L_joint',
                                                                  'split', 'empty'}))
    print(f"  v4 open: {v4_open}/{n}; v5 open: {v5_open}/{n}")
    # v5 may open more than v4 because v4 was incorrectly closing cells via
    # the unsound L_joint branch.  This is EXPECTED.


if __name__ == '__main__':
    test_V1_soundness()
    test_V2_dominance()
    test_V3_d8_S16_sample()
    print("\nVALIDATION COMPLETE")

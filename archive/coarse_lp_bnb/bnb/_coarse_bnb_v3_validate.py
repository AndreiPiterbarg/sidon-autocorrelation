"""Validate v3 against v2: soundness + speed + bound tightness."""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v2 as v2
import _coarse_bnb_v3 as v3


def test_v3_self_basic():
 """v3 self-test: F, L, L_joint produce numeric bounds on a small cell."""
 print("\n=== v3 SELF-TEST: small cell at d=4 ===", flush=True)
 d, S, c_target = 4, 40, 1.0
 windows = v3.build_all_windows(d)
 c = np.array([10, 10, 10, 10], dtype=np.float64)
 cell = v3.Cell.from_integer_composition(c, S)
 cache = v3.CellCache.build(cell)
 v3.compute_F_all_windows(cache, windows, c_target)
 print(f" F best bound: {cache.f_best_bound:.6f} W_idx={cache.f_best_W_idx}",
 flush=True)
 # L single on best W
 best_W = windows[cache.f_best_W_idx]
 L_bound = v3.tier_L_single(cache, best_W, c_target)
 print(f" L_single (best F window): {L_bound:.6f}", flush=True)
 # L_joint
 Lj_bound = v3.tier_L_joint(cache, windows, c_target, K=3)
 print(f" L_joint (top-3): {Lj_bound:.6f}", flush=True)


def test_v3_soundness_monte_carlo():
 """Same as v2's V1: ensure v3 bounds are LB on true min via MC."""
 print("\n=== v3 SOUNDNESS (Monte Carlo) ===", flush=True)
 rng = np.random.default_rng(2026_05_11)
 n_total = 0
 n_violations = 0
 worst = 0.0

 for trial in range(15):
 d = int(rng.integers(4, 7))
 S = int(rng.integers(20, 50))
 u = rng.dirichlet(np.ones(d) * rng.uniform(0.3, 3.0))
 c = np.round(u * S).astype(np.int64)
 c[-1] = S - c[:-1].sum()
 if c.min() < 0:
 continue
 cell = v3.Cell.from_integer_composition(c.astype(np.float64), S)
 cache = v3.CellCache.build(cell)
 windows = v3.build_all_windows(d)
 # Pick 3 windows
 W_indices = rng.choice(len(windows), size=min(3, len(windows)),
 replace=False)
 for w_idx in W_indices:
 W = windows[w_idx]
 c_target = float(W.Q_coef * cache.mu_star @ W.A @ cache.mu_star) - 0.05
 F_bound = v3.tier_F(cache, W, c_target)
 L_bound = v3.tier_L_single(cache, W, c_target)
 # MC samples
 eps_samples = []
 for _ in range(2000):
 e = rng.uniform(cache.lo_eps, cache.hi_eps)
 e = e - e.mean()
 if np.all(e >= cache.lo_eps - 1e-12) and \
 np.all(e <= cache.hi_eps + 1e-12) and \
 np.all(cache.mu_star + e >= -1e-12):
 eps_samples.append(e)
 if not eps_samples:
 continue
 eps_arr = np.array(eps_samples)
 mu_arr = cache.mu_star[None, :] + eps_arr
 tv = W.Q_coef * np.einsum('ni,ij,nj->n', mu_arr, W.A, mu_arr)
 f_vals = tv - c_target
 mc_min = float(f_vals.min())
 n_total += 1
 for label, b in (('F', F_bound), ('L', L_bound)):
 if b > mc_min + 1e-5:
 n_violations += 1
 print(f" *** {label} UNSOUND: bound={b:.4f} MC_min={mc_min:.4f}",
 flush=True)
 worst = max(worst, b - mc_min)
 print(f" {n_total} (cell, W) pairs, {n_violations} violations, worst {worst:.2e}",
 flush=True)
 assert n_violations == 0, "SOUNDNESS BUG"
 print(" v3 PASSED ", flush=True)


def test_v3_vs_v2_compositions():
 """A/B comparison: same compositions, compare cert + speed."""
 print("\n=== A/B v2 vs v3 on 20 d=4 compositions, c=1.28 ===", flush=True)
 d, S, c_target = 4, 160, 1.28
 test_cases = [
 (40, 40, 40, 40), (39, 41, 41, 39), (30, 50, 50, 30),
 (20, 60, 60, 20), (50, 30, 30, 50), (10, 70, 70, 10),
 (45, 35, 35, 45), (80, 0, 0, 80), (60, 20, 20, 60),
 (35, 45, 45, 35), (30, 30, 50, 50), (30, 50, 30, 50),
 (20, 40, 60, 40), (10, 50, 70, 30), (15, 35, 55, 55),
 (25, 35, 50, 50), (10, 30, 70, 50), (5, 25, 75, 55),
 (20, 30, 60, 50), (30, 35, 45, 50),
 ]
 windows_v2 = v2.build_all_windows(d)
 windows_v3 = v3.build_all_windows(d)
 # Warm SDP templates
 v3.get_sdp_template(d)
 v3.get_joint_template(d, 4)

 t0 = time.time()
 v2_results = []
 for c_tup in test_cases:
 c = np.array(c_tup, dtype=np.float64)
 r = v2.certify_composition(c, S, d, c_target,
 windows=windows_v2,
 max_depth=3, solver='MOSEK')
 v2_results.append(r.certified)
 t_v2 = time.time() - t0
 n_v2 = sum(v2_results)

 t0 = time.time()
 v3_results = []
 for c_tup in test_cases:
 c = np.array(c_tup, dtype=np.float64)
 r = v3.certify_composition(c, S, d, c_target,
 windows=windows_v3,
 max_depth=3)
 v3_results.append(r.certified)
 t_v3 = time.time() - t0
 n_v3 = sum(v3_results)

 print(f" v2: closed {n_v2}/{len(test_cases)} [{t_v2:.2f}s]", flush=True)
 print(f" v3: closed {n_v3}/{len(test_cases)} [{t_v3:.2f}s]", flush=True)
 print(f" speedup: {t_v2/t_v3:.2f}x", flush=True)
 # Soundness: v3 must close ⊇ v2 (v3 has tighter bounds)
 v2_only = [t for t, v2c, v3c in zip(test_cases, v2_results, v3_results)
 if v2c and not v3c]
 v3_only = [t for t, v2c, v3c in zip(test_cases, v2_results, v3_results)
 if v3c and not v2c]
 print(f" v2-only closed: {len(v2_only)} (should be 0 if v3 is monotonically tighter)",
 flush=True)
 print(f" v3-only closed: {len(v3_only)} (genuine bound improvements)",
 flush=True)
 if v3_only:
 print(" v3-only closed (bound improvements):")
 for t in v3_only[:5]:
 print(f" {t}")


if __name__ == '__main__':
 test_v3_self_basic()
 test_v3_soundness_monte_carlo()
 test_v3_vs_v2_compositions()
 print("\nVALIDATION DONE", flush=True)

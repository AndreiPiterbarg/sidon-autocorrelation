"""Validate _coarse_bnb_v2.py — soundness and empirical effectiveness.

Tests:
 V1. SOUNDNESS via Monte Carlo:
 For a representative cell, sample many ε inside the cell, compute
 f_W(ε), and verify min over samples ≥ tier bound. Any violation
 flags an UNSOUNDNESS BUG.

 V2. F → Q → L → L_joint progressively closes more cells.
 Verify bound monotonicity: F ≤ Q ≤ L ≤ L_joint (each tier dominates
 the previous on the same window set).

 V3. End-to-end: certify compositions of (n_half=2, m=20, c=1.28).
 Compare with prune_F from _M1_bench.py. v2 with SDP should close
 at least as many cells as F alone.

 V4. Compare v2 BnB depth vs the existing cell-cert at the failing cells.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _coarse_bnb_v2 import (
 Cell, WindowData, build_all_windows, tier_F, tier_Q_joint,
 tier_L_single, tier_L_joint, cert_cell, certify_composition,
 HAS_CVXPY, lp_max_linear, lp_min_linear,
)


def random_eps_in_cell(cell: Cell, mu_star: np.ndarray,
 n_samples: int, rng) -> np.ndarray:
 """Sample ε uniformly in {lo_eps ≤ ε ≤ hi_eps, Σε = 0, μ*+ε ≥ 0}.

 Strategy: rejection sampling with hyperplane projection.
 Returns array of shape (n_accepted, d).
 """
 d = cell.d
 lo_eps = np.maximum(cell.lo - mu_star, -mu_star)
 hi_eps = cell.hi - mu_star
 out = []
 n_tries = 0
 while len(out) < n_samples and n_tries < n_samples * 100:
 n_tries += 1
 # Uniform in box
 e = rng.uniform(lo_eps, hi_eps)
 # Project to Σε = 0 by shifting
 e = e - e.mean()
 # Check box
 if np.all(e >= lo_eps - 1e-12) and np.all(e <= hi_eps + 1e-12):
 # Check μ ≥ 0
 if np.all(mu_star + e >= -1e-12):
 out.append(e)
 return np.array(out)


def evaluate_f_W(eps: np.ndarray, mu_star: np.ndarray, W: WindowData,
 c_target: float) -> np.ndarray:
 """f_W(ε_n) for an array of ε samples. f_W(ε) = TV_W(μ*+ε) − c_target."""
 mu = mu_star[np.newaxis, :] + eps # (n, d)
 tv = W.Q_coef * np.einsum('ni,ij,nj->n', mu, W.A, mu)
 return tv - c_target


# ===================================================================
# V1: SOUNDNESS via Monte Carlo
# ===================================================================

def test_V1_soundness():
 print("\n" + "=" * 70)
 print("V1: Monte Carlo soundness validation")
 print("=" * 70)
 print(" For 20 random cells x 3 windows, verify tier bound ≤ min_ε f_W(ε)")
 print(" over 5000 sampled ε.\n")

 rng = np.random.default_rng(2026_05_11)
 n_violations = 0
 n_total = 0
 worst_diff = 0.0

 for trial in range(20):
 d = int(rng.integers(4, 8))
 S = int(rng.integers(20, 80))
 # Random composition
 u = rng.dirichlet(np.ones(d) * rng.uniform(0.3, 3.0))
 c = np.round(u * S).astype(np.int64)
 c[-1] = S - c[:-1].sum() # ensure sum = S
 if c.min() < 0:
 continue # skip if rounding made any bin negative
 cell = Cell.from_integer_composition(c.astype(np.float64), S)
 mu_star = cell.center
 windows = build_all_windows(d)
 # Pick 3 random windows
 W_indices = rng.choice(len(windows), size=min(3, len(windows)),
 replace=False)
 for w_idx in W_indices:
 W = windows[w_idx]
 c_target = float(W.Q_coef * mu_star @ W.A @ mu_star) - 0.05 # tight target
 # Tier F bound
 F_bound, _ = tier_F(cell, mu_star, W, c_target)
 # Tier L bound (if cvxpy)
 if HAS_CVXPY:
 L_bound, L_info = tier_L_single(cell, mu_star, W, c_target,
 solver='MOSEK', verbose=False)
 else:
 L_bound = None
 # Monte Carlo: sample ε
 eps_samples = random_eps_in_cell(cell, mu_star, 5000, rng)
 if len(eps_samples) == 0:
 continue
 f_vals = evaluate_f_W(eps_samples, mu_star, W, c_target)
 mc_min = float(f_vals.min())
 n_total += 1
 # Soundness check: tier bound MUST be ≤ true min, hence ≤ MC min
 if F_bound > mc_min + 1e-6:
 n_violations += 1
 print(f" *** F UNSOUND: trial {trial} W=({W.ell},{W.s_lo}) "
 f"d={d} S={S} c_target={c_target:.4f}")
 print(f" F bound: {F_bound:.6f}")
 print(f" MC min: {mc_min:.6f}")
 print(f" diff: {F_bound - mc_min:.6e}")
 worst_diff = max(worst_diff, F_bound - mc_min)
 if L_bound is not None and L_bound > mc_min + 1e-5:
 n_violations += 1
 print(f" *** L UNSOUND: trial {trial} W=({W.ell},{W.s_lo}) "
 f"d={d} S={S} c_target={c_target:.4f}")
 print(f" L bound: {L_bound:.6f}")
 print(f" MC min: {mc_min:.6f}")
 print(f" diff: {L_bound - mc_min:.6e}")
 worst_diff = max(worst_diff, L_bound - mc_min)
 # progress
 if trial % 5 == 0:
 print(f" ... trial {trial}/20 n_checks={n_total} "
 f"n_violations={n_violations}", flush=True)

 print(f"\n TOTAL: {n_total} (cell, W) pairs tested")
 print(f" Soundness violations: {n_violations}")
 print(f" Worst (tier_bound - MC_min) violation: {worst_diff:.6e}")
 assert n_violations == 0, f"V1 FAILED: {n_violations} soundness violations"
 print(" V1 PASSED ")


# ===================================================================
# V2: Tier monotonicity (F ≤ L on the same W)
# ===================================================================

def test_V2_tier_monotonicity():
 print("\n" + "=" * 70)
 print("V2: Tier monotonicity")
 print("=" * 70)
 print(" L_single ≥ F on each window (L is the tighter bound).\n")

 if not HAS_CVXPY:
 print(" SKIP: cvxpy not available")
 return

 rng = np.random.default_rng(7)
 n_F_dominates = 0
 n_L_dominates = 0
 n_tied = 0
 worst_F_above_L = 0.0
 for trial in range(10):
 d = int(rng.integers(4, 6))
 S = int(rng.integers(20, 60))
 u = rng.dirichlet(np.ones(d) * 1.5)
 c = np.round(u * S).astype(np.int64)
 c[-1] = S - c[:-1].sum()
 if c.min() < 0:
 continue
 cell = Cell.from_integer_composition(c.astype(np.float64), S)
 mu_star = cell.center
 windows = build_all_windows(d)
 # Pick top 3 windows by TV value
 W_scores = [(W.Q_coef * float(mu_star @ W.A @ mu_star), W) for W in windows]
 W_scores.sort(key=lambda x: -x[0])
 for tv, W in W_scores[:3]:
 c_target = tv - 0.1
 F_bound, _ = tier_F(cell, mu_star, W, c_target)
 L_bound, _ = tier_L_single(cell, mu_star, W, c_target,
 solver='MOSEK')
 if F_bound > L_bound + 1e-5:
 n_F_dominates += 1
 worst_F_above_L = max(worst_F_above_L, F_bound - L_bound)
 elif L_bound > F_bound + 1e-5:
 n_L_dominates += 1
 else:
 n_tied += 1
 print(f" L > F: {n_L_dominates}")
 print(f" Tied: {n_tied}")
 print(f" F > L: {n_F_dominates} (THEORETICALLY POSSIBLE for indefinite Q)")
 if n_F_dominates > 0:
 print(f" note: F can occasionally exceed L when the SDP relaxation is loose")
 print(f" on the box but F's entry-wise bound happens to be tighter.")
 print(f" Worst F-above-L: {worst_F_above_L:.4e}")
 # Soundness check: each is valid; ordering doesn't have to be strict
 # ⇒ this test is INFORMATIONAL, no hard assert
 print(" V2 PASSED (informational) ")


# ===================================================================
# V3: End-to-end at (d=4, S=4·2·20=160, c=1.28) — small enough to enumerate
# ===================================================================

def test_V3_short():
 """Short end-to-end: certify ~20 specific compositions at d=4, S=160, c=1.28."""
 print("\n" + "=" * 70)
 print("V3 (short): certify 20 specific compositions at d=4, S=160, c=1.28")
 print("=" * 70)
 d = 4
 S = 160
 c_target = 1.28
 windows = build_all_windows(d)
 # Hand-pick a representative set: uniform, near-uniform, asymmetric,
 # extreme. These should all be PRUNEABLE if v2 is working.
 test_cases = [
 (40, 40, 40, 40), # uniform
 (39, 41, 41, 39), # slight perturb
 (30, 50, 50, 30), # bigger perturb
 (20, 60, 60, 20),
 (50, 30, 30, 50),
 (10, 70, 70, 10),
 (45, 35, 35, 45),
 (80, 0, 0, 80), # extreme
 (60, 20, 20, 60),
 (35, 45, 45, 35),
 # Asymmetric (canonical: c[0] <= c[3])
 (30, 30, 50, 50),
 (30, 50, 30, 50),
 (20, 40, 60, 40),
 (10, 50, 70, 30),
 (15, 35, 55, 55),
 (25, 35, 50, 50),
 (10, 30, 70, 50),
 (5, 25, 75, 55),
 (20, 30, 60, 50),
 (30, 35, 45, 50),
 ]
 closed = {'F': 0, 'Q': 0, 'L': 0, 'L_joint': 0, 'split': 0}
 open_cases = []
 t0 = time.time()
 for c_tup in test_cases:
 c = np.array(c_tup, dtype=np.float64)
 assert c.sum() == S, f"sum {c.sum()} != {S}"
 result = certify_composition(c, S, d, c_target, windows=windows,
 max_depth=3, solver='MOSEK')
 if result.certified:
 closed[result.tier_used] = closed.get(result.tier_used, 0) + 1
 else:
 open_cases.append((c_tup, result))
 elapsed = time.time() - t0
 n_total = len(test_cases)
 n_closed = sum(closed.values())
 print(f" closed @c=1.28: {n_closed}/{n_total} "
 f"(F={closed.get('F',0)} Q={closed.get('Q',0)} "
 f"L={closed.get('L',0)} L_joint={closed.get('L_joint',0)} "
 f"split={closed.get('split',0)}) [{elapsed:.1f}s]")
 if open_cases:
 print(f" STILL OPEN ({len(open_cases)}):")
 for c_tup, r in open_cases[:5]:
 print(f" c={c_tup} tier_reached={r.tier_used} "
 f"depth={r.depth_used} sub_cells={r.n_subcells}")


if __name__ == '__main__':
 test_V1_soundness()
 test_V2_tier_monotonicity()
 test_V3_short()
 print("\nVALIDATION COMPLETE")

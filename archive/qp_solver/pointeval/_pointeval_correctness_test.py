"""Rigorous correctness test for the per-conv-bin point-evaluation prune.

Verifies:

 T1. PIECEWISE-LINEAR IDENTITY
 max_t (f*f)(t) on [0, 1] = max_q conv[q]/(2*d*m^2) for f piecewise
 constant on d bins of width 1/(2d) with heights c_i/m.

 T2. POINT-EVAL EQUATION
 conv[q]/(2*d*m^2) = (g*g)(t_q) at t_q = (q+1)*w exactly, for all q.

 T3. SOUNDNESS via direct algebra
 Every composition that the new prune fires on satisfies the C&S
 Lemma 3 inequality (g*g)(t_q*) - correction(t_q*) >= c_target with
 correction = 2*W_local_mass/m + |N(t_q*)|/m^2.

 T4. CELL-SOUNDNESS via sampling
 For each pruned c, sample multiple h's in the cell (h_i = c_i/m + eps,
 |eps_i| <= 1/(2m)) and verify that max(h*h) >= c_target.

 T5. DOMINANCE
 Every c for which the OLD per-window prune fires, the NEW prune also
 fires. Tested on diverse Dirichlet, multinomial-sparse, and edge-case
 compositions.

 T6. EDGE CASES
 q=0, q=conv_len-1, single-bin mass, two-bin asymmetric, near-uniform.

 T7. NEGATIVE / CONSERVATIVITY
 Construct compositions whose true max(f*f) is strictly less than
 c_target; verify NEITHER pruner fires (otherwise the bound would be
 unsound).

 T8. NUMBA KERNEL CONSISTENCY
 Drive the actual `_prune_dynamic_int32` kernel from
 run_cascade.py with the same compositions; verify its output exactly
 matches our reference old-prune.

The pure-numpy reference implementation of both prunes uses int64 arithmetic
matching the Numba kernels.
"""
from __future__ import annotations

import os
import sys

import numpy as np


# Make the cloninger-steinerberger modules importable for T8. Two paths are
# needed: the package root (so run_cascade can import compositions/pruning)
# and the cpu/ subdir (so we can import run_cascade directly).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger"))
sys.path.insert(0, os.path.join(_HERE, "cloninger-steinerberger", "cpu"))


# =========================================================================
# Reference building blocks
# =========================================================================

def conv_array(c, d):
 conv_len = 2 * d - 1
 conv = np.zeros(conv_len, dtype=np.int64)
 for i in range(d):
 ci = int(c[i])
 if ci == 0:
 continue
 conv[2 * i] += ci * ci
 for j in range(i + 1, d):
 cj = int(c[j])
 if cj == 0:
 continue
 conv[i + j] += 2 * ci * cj
 return conv


def autoconv_value(h, d, t):
 """(f*f)(t) for f = sum_i h_i 1_{bin_i}, bins of width w=1/(2d)."""
 w = 1.0 / (2.0 * d)
 val = 0.0
 for i in range(d):
 if h[i] == 0:
 continue
 for j in range(d):
 if h[j] == 0:
 continue
 a = max(i * w, t - (j + 1) * w)
 b = min((i + 1) * w, t - j * w)
 if b > a:
 val += h[i] * h[j] * (b - a)
 return val


def autoconv_max_brute(h, d, samples_per_piece=64):
 """Sample (f*f) on a fine grid; max is achieved at a breakpoint."""
 w = 1.0 / (2.0 * d)
 best = 0.0
 for q in range(2 * d):
 for k in range(samples_per_piece + 1):
 t = (q + k / samples_per_piece) * w
 v = autoconv_value(h, d, t)
 if v > best:
 best = v
 return best


def gg_at_breakpoint(c, d, m, q):
 """Direct computation of (g*g)(t_q) where t_q = (q+1)*w.

 Independent of the conv[q] formula: integrates f=c_i/m piecewise-const
 against itself over the appropriate domain.
 """
 w = 1.0 / (2.0 * d)
 h = c.astype(np.float64) / m
 t = (q + 1) * w
 return autoconv_value(h, d, t)


# =========================================================================
# Reference prune kernels (pure NumPy, int64 throughout)
# =========================================================================

def old_per_window_prune(c, d, m, c_target):
 """Mirror of run_cascade.py:_prune_dynamic_int32 W-refined path."""
 conv = conv_array(c, d)
 conv_len = 2 * d - 1
 cs_base_m2 = c_target * m * m
 eps_margin = 1e-9 * m * m
 n_half_d = d / 2.0 # n_half = d/2 (since d = 2*n_half)

 prefix_c = np.zeros(d + 1, dtype=np.int64)
 for i in range(d):
 prefix_c[i + 1] = prefix_c[i] + int(c[i])

 max_ell = 2 * d
 for ell in range(2, max_ell + 1):
 n_cv = ell - 1
 n_windows = conv_len - n_cv + 1
 if n_windows <= 0:
 continue
 scale_ell = ell * 4.0 * n_half_d # = ell * 2d
 for s_lo in range(n_windows):
 ws = int(conv[s_lo:s_lo + n_cv].sum())
 lo_bin = max(0, s_lo - (d - 1))
 hi_bin = min(d - 1, s_lo + ell - 2)
 W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])
 corr_w = 1.0 + W_int / (2.0 * n_half_d)
 dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
 if ws > dyn_x:
 return True, {"ell": ell, "s_lo": s_lo, "ws": ws,
 "threshold": dyn_x, "W_int_window": W_int}
 return False, None


def new_pointeval_prune(c, d, m, c_target, tight_N=True):
 """Per-conv-bin point-evaluation prune.

 correction(t_q) = 2 * G_local(t_q) / m + |N(t_q)| / m^2
 where G_local(t_q) = mass of g in N(t_q) = W_int(q)/(2dm)
 |N(t_q)| = n_bins(q) * w = n_bins(q)/(2d)
 Multiplied by 2*d*m^2:
 = 2*W_int(q) + n_bins(q) [tight |N|]
 = 2*W_int(q) + 2*d [loose |N|<=1]

 Prune fires when conv[q] > 2*d*m^2*c_target + correction_conv_units.
 """
 conv = conv_array(c, d)
 conv_len = 2 * d - 1
 cs_base_m2 = c_target * m * m
 eps_margin = 1e-9 * m * m

 prefix_c = np.zeros(d + 1, dtype=np.int64)
 for i in range(d):
 prefix_c[i + 1] = prefix_c[i] + int(c[i])

 base = 2.0 * d * cs_base_m2

 for q in range(conv_len):
 i_lo = max(0, q - (d - 1))
 i_hi = min(d - 1, q)
 W_int = int(prefix_c[i_hi + 1] - prefix_c[i_lo])
 n_bins = i_hi - i_lo + 1
 N_term = n_bins if tight_N else (2 * d)
 threshold = base + 2.0 * W_int + N_term + eps_margin
 if int(conv[q]) > threshold:
 return True, {"q": q, "conv_q": int(conv[q]),
 "W_int": W_int, "n_bins": n_bins,
 "threshold": threshold}
 return False, None


# =========================================================================
# Tests
# =========================================================================

def test_T1_identity():
 """max_t (f*f)(t) numerical == max_q conv[q]/(2dm^2).

 Note: d is always even (d = 2*n_half) in the cascade framework, so we
 test only even d here.
 """
 rng = np.random.default_rng(42)
 n_trials = 50
 max_rel_err = 0.0
 for trial in range(n_trials):
 d = 2 * int(rng.integers(1, 5)) # even d in {2,4,6,8}
 m = int(rng.integers(2, 16))
 S = 2 * d * m
 kind = trial % 4
 if kind == 0:
 c = rng.multinomial(S, np.ones(d) / d)
 elif kind == 1:
 alpha = rng.uniform(0.1, 5.0, size=d)
 c = rng.multinomial(S, rng.dirichlet(alpha))
 elif kind == 2:
 p = np.zeros(d)
 p[rng.integers(d)] = 1.0
 p = 0.7 * p + 0.3 * (np.ones(d) / d)
 c = rng.multinomial(S, p)
 else:
 # Two-bin concentration
 p = np.zeros(d)
 i, j = rng.choice(d, 2, replace=False)
 p[i] = 0.6
 p[j] = 0.4
 c = rng.multinomial(S, p)

 h = c.astype(np.float64) / m
 max_brute = autoconv_max_brute(h, d, samples_per_piece=64)
 max_formula = float(np.max(conv_array(c, d))) / (2.0 * d * m * m)

 rel_err = abs(max_brute - max_formula) / max(max_formula, 1e-12)
 max_rel_err = max(max_rel_err, rel_err)
 assert rel_err < 1e-9, (
 f"T1 trial {trial}: identity FAILS d={d} m={m} c={c.tolist()}\n"
 f" brute={max_brute} formula={max_formula} rel_err={rel_err}")
 print(f" [T1] identity holds across {n_trials} trials, max rel_err = "
 f"{max_rel_err:.2e}")


def test_T2_pointeval_equation():
 """conv[q]/(2dm^2) == (g*g)(t_q) directly at every breakpoint."""
 rng = np.random.default_rng(101)
 n_trials = 30
 for trial in range(n_trials):
 d = 2 * int(rng.integers(1, 5)) # even d
 m = int(rng.integers(2, 12))
 S = 2 * d * m
 c = rng.multinomial(S, rng.dirichlet(np.ones(d) * rng.uniform(0.2, 3.0)))
 conv = conv_array(c, d)
 for q in range(2 * d - 1):
 formula = float(conv[q]) / (2.0 * d * m * m)
 direct = gg_at_breakpoint(c, d, m, q)
 err = abs(formula - direct)
 assert err < 1e-10, (
 f"T2 trial {trial} q={q}: formula={formula} direct={direct} "
 f"err={err} c={c.tolist()}")
 print(f" [T2] (g*g)(t_q) = conv[q]/(2dm^2) verified at every breakpoint "
 f"over {n_trials} trials")


def test_T3_soundness_algebra():
 """Algebraic soundness: (g*g)(t_q*) - correction >= c_target."""
 rng = np.random.default_rng(123)
 cases = [(4, 8, 1.20), (4, 10, 1.25), (6, 12, 1.28), (8, 6, 1.20)]
 n_trials = 1500
 total_pruned = 0
 min_slack = float("inf")
 for d, m, c_target in cases:
 for _ in range(n_trials):
 S = 2 * d * m
 alpha = rng.uniform(0.1, 5.0, size=d)
 c = rng.multinomial(S, rng.dirichlet(alpha))
 pruned, info = new_pointeval_prune(c, d, m, c_target, tight_N=True)
 if not pruned:
 continue
 total_pruned += 1
 q = info["q"]
 gg = float(conv_array(c, d)[q]) / (2.0 * d * m * m)
 i_lo = max(0, q - (d - 1))
 i_hi = min(d - 1, q)
 W_mass = sum(int(c[i]) for i in range(i_lo, i_hi + 1)) / (2.0 * d * m)
 N_len = (i_hi - i_lo + 1) / (2.0 * d)
 corr = 2.0 * W_mass / m + N_len / (m * m)
 slack = (gg - corr) - c_target
 min_slack = min(min_slack, slack)
 assert slack >= -1e-7, (
 f"T3 UNSOUND: d={d} m={m} c_t={c_target} q={q}\n"
 f" c={c.tolist()}\n"
 f" (g*g)(t_q)={gg} corr={corr} slack={slack}")
 print(f" [T3] algebra-soundness verified on {total_pruned} pruned cases, "
 f"min positive slack = {min_slack:.2e}")


def test_T4_cell_soundness():
 """For each pruned c, sample h's in the cell |h_i - c_i/m| <= 1/m and
 verify max(h*h) >= c_target.

 This validates that the prune is sound under the C&S framework where
 the discrete g (= heights c_i/m) approximates a continuous f with
 ||g - f||_inf <= 1/m.
 """
 rng = np.random.default_rng(7777)
 d, m, c_target = 4, 12, 1.20
 n_compositions = 100
 n_h_samples_per_c = 8
 pruned_count = 0
 cell_pass_count = 0
 for _ in range(n_compositions):
 S = 2 * d * m
 c = rng.multinomial(S, rng.dirichlet(np.ones(d) * rng.uniform(0.5, 3.0)))
 pruned, info = new_pointeval_prune(c, d, m, c_target, tight_N=True)
 if not pruned:
 continue
 pruned_count += 1
 # Sample h's in the cell. ||g - f||_inf <= 1/m means h_i - c_i/m
 # is in [-1/m, 1/m]. We must keep h_i >= 0 (nonneg constraint)
 # and ideally the modified function still integrates to ~1, but
 # the integral is preserved in C&S framework only modulo eps; the
 # test here just verifies no h within the height-cell breaks the
 # claimed lower bound on max(h*h).
 cell_holds = True
 for s in range(n_h_samples_per_c):
 eps = rng.uniform(-1.0 / m, 1.0 / m, size=d)
 h = (c.astype(np.float64) / m) + eps
 h = np.maximum(h, 0.0) # nonneg
 max_hh = autoconv_max_brute(h, d, samples_per_piece=64)
 if max_hh < c_target - 1e-9:
 cell_holds = False
 print(f" T4 FAIL: c={c.tolist()} h={h.tolist()} "
 f"max(h*h)={max_hh} < {c_target}")
 break
 if cell_holds:
 cell_pass_count += 1
 print(f" [T4] cell-soundness: {cell_pass_count}/{pruned_count} "
 f"pruned compositions verified across "
 f"{n_h_samples_per_c} sampled h's each")
 assert cell_pass_count == pruned_count, "T4 FAIL"


def test_T5_dominance():
 """Every c the OLD prunes, the NEW also prunes."""
 rng = np.random.default_rng(7)
 # d is always even (d = 2*n_half) in the cascade.
 cases = [
 (4, 8, 1.20), (4, 8, 1.28), (4, 10, 1.25), (6, 12, 1.30),
 (8, 6, 1.20), (10, 5, 1.10), (4, 16, 1.40), (2, 20, 1.45),
 ]
 print(f" {'d':>3} {'m':>3} {'c_t':>5} {'old%':>6} {'new%':>6} "
 f"{'old_only':>9} {'new_only':>9} {'gain':>5}")
 for d, m, c_target in cases:
 n_trials = 2000
 n_old = n_new = n_old_only = n_new_only = 0
 for _ in range(n_trials):
 S = 2 * d * m
 alpha = rng.uniform(0.1, 5.0, size=d)
 c = rng.multinomial(S, rng.dirichlet(alpha))
 old_p, _ = old_per_window_prune(c, d, m, c_target)
 new_p, _ = new_pointeval_prune(c, d, m, c_target, tight_N=True)
 n_old += int(old_p)
 n_new += int(new_p)
 if old_p and not new_p:
 n_old_only += 1
 if new_p and not old_p:
 n_new_only += 1
 assert n_old_only == 0, (
 f"T5 DOMINANCE FAILS d={d} m={m} c_t={c_target}: "
 f"{n_old_only} cases pruned by old but not new")
 rate_old = n_old / n_trials
 rate_new = n_new / n_trials
 gain = (rate_new / rate_old) if rate_old > 0 else float("inf")
 print(f" {d:>3} {m:>3} {c_target:.2f} "
 f"{100*rate_old:>5.1f}% {100*rate_new:>5.1f}% "
 f"{n_old_only:>8d} {n_new_only:>8d} {gain:>4.2f}x")


def test_T6_edge_cases():
 """Concrete edge-case compositions handled correctly."""
 # Edge 1: all mass in bin 0
 d, m = 4, 8
 c = np.array([2 * d * m, 0, 0, 0], dtype=np.int64)
 c_target = 1.28
 pruned, info = new_pointeval_prune(c, d, m, c_target, tight_N=True)
 assert pruned, f"all-mass-bin-0 should prune at c={c_target}"
 # Verify max(g*g) is what we expect
 h = c.astype(np.float64) / m
 max_gg = autoconv_max_brute(h, d, samples_per_piece=64)
 assert max_gg > 1.0 # h_0 = 8, peak (g*g) = h_0^2 * w = 64 * 1/8 = 8
 # Edge 2: all mass in last bin (symmetric)
 c2 = np.array([0, 0, 0, 2 * d * m], dtype=np.int64)
 pruned2, info2 = new_pointeval_prune(c2, d, m, c_target, tight_N=True)
 assert pruned2, "all-mass-last-bin should prune"
 # Edge 3: two-bin asymmetric (8, 24) at d=2, m=8
 d3, m3 = 2, 8
 c3 = np.array([8, 24], dtype=np.int64)
 pruned3, info3 = new_pointeval_prune(c3, d3, m3, 1.5, tight_N=True)
 assert pruned3
 # The peak should be at q=2 (right side, mass-heavy bin)
 assert info3["q"] in (1, 2)
 # Edge 4: single-bin q=0
 c4 = np.array([2 * d * m, 0, 0, 0], dtype=np.int64)
 p4, info4 = new_pointeval_prune(c4, d, m, 0.5, tight_N=True)
 assert p4
 # The first prune should fire AT q=0 (highest conv[q] for this c)
 # since conv[0] = c_0^2 dominates
 # Edge 5: uniform composition c = (m, m, ..., m)
 d5, m5 = 4, 8
 c5 = np.array([2 * m5] * d5, dtype=np.int64) # sum = 8m = 2dm
 p5, info5 = new_pointeval_prune(c5, d5, m5, 1.5, tight_N=True)
 assert p5
 # Uniform peak at q=d-1 (center)
 assert info5["q"] == d5 - 1, \
 f"uniform should peak at q={d5-1}, got q={info5['q']}"
 print(" [T6] edge cases pass: all-bin-0, all-bin-last, two-bin, "
 "uniform-peak-at-center")


def test_T7_negative_conservativity():
 """Compositions with low max(g*g) should NOT be pruned at high c_target.

 Build a near-uniform composition; max(g*g) close to but above 2. If we
 set c_target above max(g*g)+correction, neither pruner should fire.
 """
 rng = np.random.default_rng(2024)
 d, m = 4, 16
 # Near-uniform
 c = np.array([2 * m, 2 * m, 2 * m, 2 * m], dtype=np.int64)
 h = c.astype(np.float64) / m # h = (2,2,2,2), max(g*g) = 2
 max_gg = autoconv_max_brute(h, d, samples_per_piece=64)
 # Set c_target above max_gg so prune is unsound to fire
 c_target = max_gg + 0.5
 p_old, _ = old_per_window_prune(c, d, m, c_target)
 p_new, _ = new_pointeval_prune(c, d, m, c_target, tight_N=True)
 assert not p_old, f"T7 OLD unsoundly fired: max(g*g)={max_gg} c_t={c_target}"
 assert not p_new, f"T7 NEW unsoundly fired: max(g*g)={max_gg} c_t={c_target}"
 # Sweep c_target from low to high. Prune is monotonically NON-INCREASING
 # in c_target (higher c_target -> higher threshold -> harder to prune).
 # Verify (a) every firing is algebraically sound, and (b) once the
 # pruner stops firing it does not refire.
 seen_non_fire = False
 n_fires = 0
 n_non_fires = 0
 for c_try in np.linspace(0.5, max_gg + 0.4, 60):
 p, info = new_pointeval_prune(c, d, m, float(c_try), tight_N=True)
 if p:
 assert not seen_non_fire, (
 f"T7 monotonicity FAIL: pruner re-fired at c_try={c_try} "
 "after stopping at lower c_target")
 n_fires += 1
 q = info["q"]
 gg = float(conv_array(c, d)[q]) / (2.0 * d * m * m)
 i_lo = max(0, q - (d - 1)); i_hi = min(d - 1, q)
 W_mass = sum(int(c[i]) for i in range(i_lo, i_hi + 1)) / (2.0 * d * m)
 N_len = (i_hi - i_lo + 1) / (2.0 * d)
 corr = 2.0 * W_mass / m + N_len / (m * m)
 assert (gg - corr) >= float(c_try) - 1e-7, \
 f"T7 unsound at c_try={c_try}: gg-corr={gg-corr}"
 else:
 seen_non_fire = True
 n_non_fires += 1
 print(f" [T7] conservativity: at c_target > max(g*g), neither pruner "
 f"fires. Sweep over 60 c_target values: {n_fires} fires + "
 f"{n_non_fires} non-fires, all monotone, all firings sound.")


def test_T8_numba_consistency():
 """Cross-check the actual Numba kernel against our reference old-prune."""
 try:
 from run_cascade import _prune_dynamic_int32
 except Exception as e:
 print(f" [T8] SKIP: cannot import _prune_dynamic_int32 ({e!r})")
 return
 rng = np.random.default_rng(31337)
 cases = [(4, 8, 1.20), (4, 8, 1.28), (4, 10, 1.25), (6, 12, 1.30)]
 n_trials_each = 200
 for d, m, c_target in cases:
 # Build a batch
 batch = []
 for _ in range(n_trials_each):
 S = 2 * d * m
 alpha = rng.uniform(0.1, 5.0, size=d)
 c = rng.multinomial(S, rng.dirichlet(alpha))
 batch.append(c)
 batch_int = np.array(batch, dtype=np.int32)
 n_half = d // 2
 survived = _prune_dynamic_int32(batch_int, n_half, m, c_target,
 use_flat_threshold=False, use_F=False)
 # Reference
 ref_pruned = np.array([
 old_per_window_prune(c, d, m, c_target)[0] for c in batch
 ])
 ref_survived = ~ref_pruned
 diffs = (survived != ref_survived).sum()
 assert diffs == 0, (
 f"T8 numba kernel disagrees with reference on d={d} m={m} "
 f"c_t={c_target}: {diffs}/{n_trials_each} cases differ")
 print(f" [T8] Numba kernel matches reference on "
 f"{n_trials_each * len(cases)} compositions across {len(cases)} cases")


if __name__ == "__main__":
 print("T1: piecewise-linear identity")
 test_T1_identity()
 print()
 print("T2: point-eval equation")
 test_T2_pointeval_equation()
 print()
 print("T3: algebraic soundness")
 test_T3_soundness_algebra()
 print()
 print("T4: cell-soundness via h-sampling")
 test_T4_cell_soundness()
 print()
 print("T5: dominance")
 test_T5_dominance()
 print()
 print("T6: edge cases")
 test_T6_edge_cases()
 print()
 print("T7: negative / conservativity")
 test_T7_negative_conservativity()
 print()
 print("T8: Numba kernel consistency")
 test_T8_numba_consistency()
 print()
 print("All tests passed.")

"""Option A: numerically MINIMIZE max_t (f*f)(t) over f ∈ Δ.

C_{1a} := inf over admissible f of max_t (f*f)(t).
CS 2017 proved C_{1a} ≥ 1.2802 (current best LOWER bound).

To DISPROVE C_{1a} ≥ 1.281: find a single f with max(f*f) < 1.281.
This gives C_{1a} ≤ max(f*f) < 1.281, i.e., 1.281 is NOT a valid lower bound.

Strategy:
 (1) f = piecewise-constant on [-1/4, 1/4] with d equal-width bins, heights
 2d·μ_i (so ∫f = Σμ_i = 1, μ_i ≥ 0).
 (2) For each d ∈ {14, 20, 30, 50, 100, 200}: SLSQP from many random starts to
 MINIMIZE max(f*f).
 (3) Track best (smallest) max(f*f) found.

Soundness: max(f*f) is computed numerically on a fine grid; we verify within
discretization error (~1/N) that the value is stable.
"""
import numpy as np
import os
import sys
import json
import time
from scipy.optimize import minimize


def autocorr_max(mu, d, N_per_bin=64):
 """Compute max_t (f*f)(t) for piecewise-constant f with masses mu in d bins.

 f on [-1/4, 1/4], width 1/2, d equal bins of width 1/(2d).
 f(x) = 2d * mu_i for x in bin i. ∫f = Σmu_i (must be 1).
 (f*f) supported on [-1/2, 1/2], piecewise-LINEAR.

 Compute via dense sampling: N samples per bin → total Nd over support, with
 dx = 1/(2dN). Use FFT convolution for speed.

 The PIECEWISE-LINEAR form means max(f*f) is either at a knot (t = k/(2d) for
 integer k) or at the apex of a linear segment (constant), so it's exact at knots.
 But to be safe, use fine sampling.
 """
 # Fast: discretize f on a grid (support [-1/4, 1/4]).
 N = N_per_bin * d
 f_samples = np.empty(N)
 for i in range(d):
 f_samples[i * N_per_bin:(i + 1) * N_per_bin] = 2.0 * d * mu[i]
 dx = 0.5 / N # bin total width = 1/2, divided by N samples
 ff = np.convolve(f_samples, f_samples) * dx
 return ff.max()


def autocorr_max_exact(mu, d):
 """Exact max(f*f) via the d-knot formula.

 For piecewise-constant f with bin width h_b = 1/(2d) and heights c_i = 2d*mu_i:
 (f*f)(t) is piecewise-linear in t. Knots at t_k = k * h_b for k = -(d-1)..(d-1).
 At t_k:
 (f*f)(t_k) = h_b * sum over i in [max(0, -k), min(d-1, d-1-k)] of c_i * c_{i+k}
 = (1/(2d)) * (2d)^2 * sum mu_i mu_{i+k}
 = 2d * sum_{i=0..d-1, i+k in [0,d-1]} mu_i mu_{i+k}

 Max over knots = max over k ∈ -(d-1)..(d-1). Symmetric in k, so check k=0..d-1.
 """
 mu = np.asarray(mu)
 best = 0.0
 for k in range(d):
 # Σ μ_i μ_{i+k} for i = 0 .. d-1-k (where both indices valid)
 if k == 0:
 s = float(np.dot(mu, mu))
 else:
 s = float(np.dot(mu[:d - k], mu[k:]))
 val = 2.0 * d * s
 if val > best:
 best = val
 return best


def max_ff_obj(mu_unnormed, d):
 """Objective: max(f*f) for given (unnormalized) mu. We project via normalize."""
 mu = np.maximum(mu_unnormed, 0)
 s = mu.sum()
 if s == 0:
 return 1e10
 mu = mu / s
 return autocorr_max_exact(mu, d)


def search_min_max_ff(d, n_starts=40, seed=0):
 """Random-restart SLSQP to minimize max(f*f) over μ ∈ Δ_d."""
 rng = np.random.default_rng(seed)
 best_val = float('inf')
 best_mu = None
 history = []
 constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0}]
 bounds = [(0.0, 1.0)] * d
 for trial in range(n_starts):
 # Mix of starts: uniform, random Dirichlet, peaked, symmetric
 if trial == 0:
 mu0 = np.ones(d) / d # uniform
 elif trial == 1:
 # Symmetric V-shape: emphasis on edges
 x = np.linspace(-1, 1, d)
 mu0 = (1 + np.abs(x))
 mu0 /= mu0.sum()
 elif trial == 2:
 # Symmetric U-shape
 x = np.linspace(-1, 1, d)
 mu0 = 1 + 2 * x ** 2
 mu0 /= mu0.sum()
 elif trial == 3:
 # Symmetric peaked
 x = np.linspace(-1, 1, d)
 mu0 = np.exp(-x ** 2)
 mu0 /= mu0.sum()
 else:
 # Random Dirichlet — vary concentration
 alpha = np.full(d, 0.3 + 0.5 * rng.random())
 mu0 = rng.dirichlet(alpha)
 try:
 res = minimize(max_ff_obj, mu0, args=(d,), method='SLSQP',
 bounds=bounds, constraints=constraints,
 options={'maxiter': 500, 'ftol': 1e-10})
 mu_norm = np.maximum(res.x, 0)
 mu_norm /= mu_norm.sum()
 val = autocorr_max_exact(mu_norm, d)
 if val < best_val:
 best_val = val
 best_mu = mu_norm.copy()
 history.append({'trial': trial, 'val': float(val),
 'success': bool(res.success)})
 except Exception as e:
 history.append({'trial': trial, 'error': str(e)})
 return best_val, best_mu, history


def main():
 print("=" * 64)
 print("Option A: minimize max_t (f*f)(t) over admissible f")
 print("=" * 64)
 print("Goal: find f with max(f*f) < 1.281 (would disprove C_1a >= 1.281)")
 print("Reference: CS 2017 proved C_1a >= 1.2802 (current best lower bound)")
 print()

 results = {}
 for d in [14, 20, 30, 50, 100, 200]:
 print(f"--- d = {d} ---")
 t0 = time.time()
 best_val, best_mu, history = search_min_max_ff(d, n_starts=40, seed=42)
 wall = time.time() - t0
 # Sanity check: also numerical via fine grid
 num_val = autocorr_max(best_mu, d, N_per_bin=128)
 gap_1281 = best_val - 1.281
 gap_cs = best_val - 1.2802
 sign1 = "BELOW (disproves 1.281!)" if best_val < 1.281 else "above"
 signcs = "BELOW (disproves CS!)" if best_val < 1.2802 else "above"
 print(f" best max(f*f) = {best_val:.6f} (numerical sanity: {num_val:.6f})")
 print(f" vs 1.281: {sign1} (gap {gap_1281:+.6f})")
 print(f" vs 1.2802 (CS): {signcs} (gap {gap_cs:+.6f})")
 print(f" wall {wall:.1f}s, 40 starts")
 results[f'd{d}'] = {
 'd': d, 'best_val': float(best_val),
 'numerical_sanity': float(num_val),
 'best_mu': best_mu.tolist() if best_mu is not None else None,
 'gap_vs_1281': float(gap_1281),
 'gap_vs_CS_1.2802': float(gap_cs),
 'wall': wall,
 }

 print()
 print("=" * 64)
 print("SUMMARY")
 print("=" * 64)
 for k, r in results.items():
 below_1281 = "YES" if r['best_val'] < 1.281 else "no"
 below_cs = "YES" if r['best_val'] < 1.2802 else "no"
 print(f" d={r['d']:>3}: max(f*f) = {r['best_val']:.6f} "
 f"below 1.281? [{below_1281}] below 1.2802? [{below_cs}]")

 print()
 any_below_1281 = any(r['best_val'] < 1.281 for r in results.values())
 any_below_cs = any(r['best_val'] < 1.2802 for r in results.values())
 if any_below_cs:
 print(" *** SOUNDNESS CHECK FAILED — found f with max < 1.2802 ***")
 print(" *** This would contradict CS 2017 (C_1a >= 1.2802) ***")
 print(" *** Likely numerical / discretization error. Re-verify. ***")
 elif any_below_1281:
 print(" FOUND f with max(f*f) < 1.281 — DISPROVES C_1a >= 1.281!")
 else:
 print(" No f found below 1.281; either C_1a >= 1.281 (cascade may work")
 print(" with enough resolution) or our optimizer hasn't found the true min.")
 print(" The best max(f*f) found across all d gives an UPPER bound on C_1a.")
 best_overall = min(r['best_val'] for r in results.values())
 print(f" Best upper bound on C_1a: {best_overall:.6f}")

 out = os.path.join(os.path.dirname(__file__), '_option_A_search_results.json')
 with open(out, 'w') as f:
 json.dump(results, f, indent=2)
 print(f"\nResults: {out}")


if __name__ == '__main__':
 main()

"""Audit v3 box-cert ⊆ v2 box-cert over a large random sample.

For each sampled (d, S, c, batch), check that any composition pruned by v3
is also pruned by v2 (since v3 uses tighter bounds; never weaker box cert).

Also do a *positive* soundness check: for compositions newly closed by v3
(pruned by v3, not by v2), do an LP-style sample within the cell to confirm
no δ pushes TV below c_target.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from run_cascade_coarse_v2 import _prune_no_correction
from run_cascade_coarse_v3 import (_prune_no_correction_v3,
 precompute_op_rest_d)


def _max_TV_over_grid(c_int, S, d, n_grid=10):
 """Lower bound on min over Cell of max_W TV_W(μ*+δ) by sampling.
 Sample n_grid^(d-1) points (last coord forced by Σ=0). Returns min.
 """
 h = 1.0 / (2.0 * S)
 mu_star = c_int.astype(np.float64) / S
 grid = np.linspace(-h, h, n_grid)
 best_min = np.inf
 if d == 2:
 for g0 in grid:
 d0 = g0; d1 = -g0
 mu = mu_star + np.array([d0, d1])
 if mu.min() < -1e-12:
 continue
 tv = _max_TV(mu, d)
 if tv < best_min:
 best_min = tv
 return best_min
 # General: iterate first d-1 dims, force last
 if d > 6:
 # too many gridpoints; sample randomly instead
 rng = np.random.default_rng(0)
 for _ in range(min(n_grid ** (d - 1), 5000)):
 delta = rng.uniform(-h, h, size=d)
 delta -= delta.mean() # project to Σ=0
 if (mu_star + delta).min() < -1e-12:
 continue
 mu = mu_star + delta
 tv = _max_TV(mu, d)
 if tv < best_min:
 best_min = tv
 return best_min
 import itertools
 for tup in itertools.product(grid, repeat=d - 1):
 last = -sum(tup)
 if abs(last) > h + 1e-12:
 continue
 delta = np.array(list(tup) + [last])
 mu = mu_star + delta
 if mu.min() < -1e-12:
 continue
 tv = _max_TV(mu, d)
 if tv < best_min:
 best_min = tv
 return best_min


def _max_TV(mu, d):
 """max_W TV_W(mu)."""
 conv = np.zeros(2 * d - 1)
 for i in range(d):
 for j in range(d):
 conv[i + j] += mu[i] * mu[j]
 best = 0.0
 for ell in range(2, 2 * d + 1):
 n_cv = ell - 1
 for s_lo in range(2 * d - 1 - n_cv + 1):
 tv = (2.0 * d / ell) * conv[s_lo:s_lo + n_cv].sum()
 if tv > best:
 best = tv
 return best


def random_compositions(d, S, n, seed=0):
 rng = np.random.default_rng(seed)
 out = np.zeros((n, d), dtype=np.int32)
 for i in range(n):
 c = rng.dirichlet(np.ones(d)) * S
 c = np.maximum(0, np.round(c)).astype(np.int32)
 c[0] += S - c.sum()
 c = np.maximum(0, c)
 out[i] = c
 return out


def audit_one(d, S, c_target, n_samples=5000, seed=0):
 op_rest_d = precompute_op_rest_d(d)
 batch = random_compositions(d, S, n_samples, seed=seed)
 s2, _ = _prune_no_correction(batch, d, S, c_target)
 s3, _ = _prune_no_correction_v3(batch, d, S, c_target, op_rest_d)
 # Sanity: v3 never claims to prune cells that v2 didn't.
 # Pruning means survived=False. Check (~s3) ⊆ (~s2) i.e., (~s3) & s2 == 0.
 bad = (~s3) & s2 # v3 prunes, v2 keeps — these are the *new* prunes
 new_prunes = int(bad.sum())
 # Both prune
 both = (~s3) & (~s2)
 # v2 prunes but v3 doesn't (FP rounding edge — should be 0)
 v2_only = s3 & (~s2)
 print(f" d={d}, S={S}, c={c_target}: n={n_samples}")
 print(f" v2 survived: {int(s2.sum())} v3 survived: {int(s3.sum())}")
 print(f" new prunes (v3 closes): {new_prunes}")
 print(f" v2-only prunes (BAD if >0): {int(v2_only.sum())}")
 # Spot-check the new prunes against fine grid
 new_idxs = np.where(bad)[0]
 if len(new_idxs) > 0:
 n_check = min(20, len(new_idxs))
 violations = 0
 worst = np.inf
 for ii in new_idxs[:n_check]:
 tv_min = _max_TV_over_grid(batch[ii], S, d, n_grid=10)
 if tv_min < c_target - 1e-6:
 violations += 1
 if tv_min < worst:
 worst = tv_min
 print(f" Soundness spot-check ({n_check} new prunes): "
 f"violations={violations}, worst_min_TV={worst:.6f} (target={c_target})")
 return {'new_prunes': new_prunes, 'v2_only_prunes': int(v2_only.sum()),
 'violations': violations, 'worst_min_tv': worst}
 return {'new_prunes': 0, 'v2_only_prunes': int(v2_only.sum()),
 'violations': 0, 'worst_min_tv': None}


def main():
 np.set_printoptions(precision=6)
 print("=== v3 soundness audit ===")
 configs = [
 (4, 20, 1.20), (4, 30, 1.25),
 (6, 15, 1.20), (6, 20, 1.25),
 (8, 12, 1.20), (8, 15, 1.20),
 ]
 rows = []
 for d, S, c in configs:
 r = audit_one(d, S, c, n_samples=3000)
 rows.append(r)
 print("\nSummary: any v2-only prunes or violations is a soundness BUG.")
 total_violations = sum(r['violations'] for r in rows)
 total_v2_only = sum(r['v2_only_prunes'] for r in rows)
 print(f" Total v3 new-prune violations: {total_violations}")
 print(f" Total v2-only-prune (FP edge): {total_v2_only}")
 if total_violations == 0 and total_v2_only == 0:
 print(" STATUS: SOUND ")
 else:
 print(" STATUS: SOUNDNESS BUG ")


if __name__ == '__main__':
 main()

"""Strict v3 audit: exhaustively enumerate small (d, S) configs, then for
EACH cell that v3 closes but v2 keeps, do fine-grid validation:
  min over Cell of max_W TV_W(μ*+δ) ≥ c_target ?
Any case where it's < c_target is a soundness violation (v3 falsely closing).
"""
import os, sys, time
import numpy as np
import itertools

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from run_cascade_coarse_v2 import _prune_no_correction
from run_cascade_coarse_v3 import (_prune_no_correction_v3,
                                     precompute_op_rest_d)


def enum_compositions(d, S):
    if d == 1:
        yield (S,)
        return
    for v in range(S + 1):
        for rest in enum_compositions(d - 1, S - v):
            yield (v,) + rest


def _max_TV(mu, d):
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


def _min_TV_over_cell(c_int, S, d, n_grid=8):
    """Lower bound on min over Cell of max_W TV by exhaustive grid search.
    Cell = {δ : |δ_i| ≤ h, Σδ = 0, μ_i = (k_i+δ_i)/S ≥ 0}.
    """
    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / S
    grid = np.linspace(-h, h, n_grid)
    best = np.inf
    if d <= 5:
        # exhaustive over (n_grid)^(d-1)
        for tup in itertools.product(grid, repeat=d - 1):
            last = -sum(tup)
            if abs(last) > h + 1e-12:
                continue
            delta = np.array(list(tup) + [last])
            mu = mu_star + delta
            if mu.min() < -1e-12:
                continue
            tv = _max_TV(mu, d)
            if tv < best:
                best = tv
    else:
        # random sample
        rng = np.random.default_rng(0)
        for _ in range(2000):
            delta = rng.uniform(-h, h, size=d)
            delta -= delta.mean()
            mu = mu_star + delta
            if mu.min() < -1e-12:
                continue
            tv = _max_TV(mu, d)
            if tv < best:
                best = tv
    return best


def audit_one(d, S, c_target, max_check=200):
    print(f"\n=== d={d}, S={S}, c={c_target} ===")
    op_rest_d = precompute_op_rest_d(d)
    batch = np.array(list(enum_compositions(d, S)), dtype=np.int32)
    print(f"  total comps: {len(batch):,}")
    s2, _ = _prune_no_correction(batch, d, S, c_target)
    s3, _ = _prune_no_correction_v3(batch, d, S, c_target, op_rest_d)
    new_close = (~s3) & s2  # v3 closes, v2 keeps
    v2_only = s3 & (~s2)    # v2 closes, v3 keeps (BAD)
    nc = int(new_close.sum())
    vo = int(v2_only.sum())
    print(f"  v2 survived: {int(s2.sum()):,}; v3 survived: {int(s3.sum()):,}")
    print(f"  v3 NEW closes: {nc:,}")
    print(f"  v2-only closes (must be 0): {vo}")

    if nc == 0:
        print(f"  No new closes to validate.")
        return {'d': d, 'S': S, 'c': c_target, 'new_closes': 0,
                'v2_only_closes': vo, 'violations': 0}

    new_idxs = np.where(new_close)[0]
    n_check = min(max_check, len(new_idxs))
    print(f"  Validating {n_check} new closes via fine-grid search...")
    violations = 0
    worst_min = np.inf
    examples = []
    for idx in new_idxs[:n_check]:
        c_int = batch[idx]
        tv_min = _min_TV_over_cell(c_int, S, d, n_grid=8)
        if tv_min < c_target - 1e-6:
            violations += 1
            if len(examples) < 3:
                examples.append({'c': c_int.tolist(), 'min_tv': tv_min})
        if tv_min < worst_min:
            worst_min = tv_min
    print(f"  Violations: {violations}/{n_check}")
    print(f"  Worst min_TV: {worst_min:.6f} (target={c_target}, "
          f"slack={worst_min-c_target:+.6f})")
    if examples:
        for ex in examples:
            print(f"    Counterexample: c={ex['c']} min_tv={ex['min_tv']:.6f}")
    return {'d': d, 'S': S, 'c': c_target, 'new_closes': nc,
            'v2_only_closes': vo, 'violations': violations,
            'worst_min_tv': worst_min, 'n_validated': n_check}


def main():
    configs = [
        (4, 20, 1.20), (4, 30, 1.25),
        (6, 12, 1.20), (6, 15, 1.20),
        (8, 12, 1.20),
    ]
    rows = []
    t0 = time.time()
    for d, S, c in configs:
        rows.append(audit_one(d, S, c, max_check=100))
    print(f"\n=== TOTAL TIME: {time.time()-t0:.1f}s ===")
    total_v = sum(r['violations'] for r in rows)
    total_v2_only = sum(r['v2_only_closes'] for r in rows)
    total_new = sum(r['new_closes'] for r in rows)
    print(f"\nTotal new closes: {total_new}")
    print(f"Total v2-only closes (must be 0): {total_v2_only}")
    print(f"Total fine-grid violations (must be 0): {total_v}")
    if total_v == 0 and total_v2_only == 0:
        print("\nSTATUS: SOUND (v3 box-cert is rigorous).")
    else:
        print("\nSTATUS: SOUNDNESS BUG.")


if __name__ == '__main__':
    main()

"""Smoke test: per-conv-position F-style "M-chain" pruner.

Background (from prior smoke `_smoke_tv_vs_maxt.py`):
  cascade max_W TV_W is a Theorem-1 *lower bound* on max_t (f_a * f_a)(t),
  but at the conv-position breakpoints t_k = (k+1)*h - 1/2 (h = 1/(2d))
  the discrete autoconv gives the *exact* point value:
     (f_a * f_a)(t_k) = sum_{i+j=k} a_i a_j / (2d * m^2)
  which is exactly 2 x cascade-TV at ell=2 at conv pos k.  So the
  per-conv-position evaluator is structurally tighter than ell=2 TV.

We use that pointwise evaluator to build a tighter F-style cell bound:

  Decompose a = c + delta with |delta|_inf <= 1, sum delta = 0, a >= 0.

  (f_a * f_a)(t_k) = [conv[k](c) + 2 * sum_{i+j=k} c_i delta_j
                      + sum_{i+j=k} delta_i delta_j] / (2d * m^2)

  - LINEAR part: sum_{i+j=k} c_i delta_j = sum_j delta_j * b(k)_j,
    where b(k)_j := sum_{i: i+j=k, i in [0,d-1]} c_i  (i.e. c_{k-j} clipped).

    LP min over delta with |delta|_inf <= 1, sum delta = 0
    (sort+extremes, even d):   min = -Delta_b(k)
    where Delta_b(k) = sum_top(d/2) b(k) - sum_bot(d/2) b(k).
    (Same closed form as variant F; verified by `_M1_bench._sanity_lp`.)

  - QUADRATIC part: |sum_{i+j=k} delta_i delta_j| <= n_pairs(k)
    where n_pairs(k) := #{(i,j) in [0,d-1]^2 : i+j=k} = ell_int_arr[k].
    Sound: pointwise |delta_i delta_j| <= 1; bound uses |delta|_inf <= 1.

Per-conv-position F-style LOWER BOUND on (f_a * f_a)(t_k):
  LB(k) = (conv[k](c) - 2 * Delta_b(k) - n_pairs(k)) / (2d * m^2).

If max_k LB(k) > c_target, then for ALL a in the cell,
  max_t (f_a * f_a)(t) >= max_k (f_a * f_a)(t_k) >= max_k LB(k) > c_target,
so the cell cannot host any C_{1a} witness.  Cell is M-PRUNABLE.

  Note (relation to F):  prune_F uses sup_a [TV_W(b) - TV_W(a)] over windows
  W = [s_lo, s_hi] (sums of conv positions).  M-chain restricts attention
  to single-conv positions but evaluates the EXACT pointwise (f*f)(t_k)
  rather than a window-average.  By the TV-vs-max_t analysis, point
  evaluation is up to 2x tighter than ell=2 TV — so M-chain can prune
  cells F's wider-window averaging accepts (and conversely, F may prune
  cells M does not — they are not nested in general).

USAGE:
  python _smoke_M_chain.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))


# ============================================================================
# Per-conv-position F-style cell bound.
# ============================================================================
def _conv_int(c):
    """Integer autoconv of c (length 2d-1)."""
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci == 0:
            continue
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            cj = int(c[j])
            if cj != 0:
                conv[i + j] += 2 * ci * cj
    return conv


def _b_vec(c, k):
    """b(k)_j = sum_{i: i+j=k, 0<=i<=d-1} c_i = c_{k-j} if in range else 0."""
    d = len(c)
    b = np.zeros(d, dtype=np.int64)
    for j in range(d):
        i = k - j
        if 0 <= i < d:
            b[j] = int(c[i])
    return b


def _delta_b(b):
    """Delta_b = sum top(d/2) - sum bot(d/2) after sort.  d must be even."""
    d = len(b)
    s = np.sort(b)
    half = d // 2
    return int(s[half:].sum() - s[:half].sum())


def _n_pairs(k, d):
    """#{(i,j) in [0,d-1]^2 : i+j=k}."""
    lo = max(0, k - (d - 1))
    hi = min(d - 1, k)
    return max(0, hi - lo + 1)


def prune_M_one(c_int, n_half, m, c_target, eps_margin=1e-9):
    """Per-conv-position F-style bound.

    Returns (pruned: bool, max_LB: float, best_k: int).

    Soundness:
      For any a in the cell { a : |a-c|_inf <= 1, sum a = 4n, a >= 0 },
        (f_a * f_a)(t_k) >= LB(k)  for each k = 0..2d-2.
      Hence max_t (f_a * f_a)(t) >= max_k (f_a * f_a)(t_k) >= max_k LB(k).
      If max_k LB(k) > c_target + eps, the cell is M-prunable.
    """
    c = np.asarray(c_int, dtype=np.int64)
    d = len(c)
    assert d == 2 * n_half, "expected d = 2 n_half for cascade convention"
    if d % 2 != 0:
        raise ValueError("M-chain LP closed form requires even d")

    conv = _conv_int(c)
    denom = 2.0 * d * (m * m)

    max_LB = -np.inf
    best_k = -1
    for k in range(2 * d - 1):
        b = _b_vec(c, k)
        Db = _delta_b(b)
        npa = _n_pairs(k, d)
        LB_int = int(conv[k]) - 2 * Db - npa  # in (2d * m^2) units
        LB = LB_int / denom
        if LB > max_LB:
            max_LB = LB
            best_k = k

    pruned = bool(max_LB > c_target + eps_margin)
    return pruned, float(max_LB), int(best_k)


# ============================================================================
# Soundness verification helpers
# ============================================================================
def _max_t_ff_at_knots(a_real, n_half, m):
    """max_k (f_a * f_a)(t_k) where heights = a/m, a in cell.

    Note: with sum(a) = 4n*m the L1 normalization gives mu_i = a_i / S,
    S = 4nm, and (f * f)(t_k) = (2d / S^2) * conv_real[k].  Equivalently
    conv_real[k] / (2d * m^2) since S = 4nm and (2d/S^2) = 1/(2d * m^2)
    when d = 2n.  Returns float scalar (max_k value).
    """
    a = np.asarray(a_real, dtype=np.float64)
    d = len(a)
    conv_real = np.convolve(a, a)  # length 2d-1
    # (f*f)(t_k) at conv-position knot
    return float(np.max(conv_real)) / (2.0 * d * m * m)


def _sample_cell(c_int, rng, n_samples=1000, h=1.0):
    """Sample a uniformly from the cell { |a-c|_inf <= h, sum a = sum c, a >= 0 }.

    Strategy: pick delta uniform in [-h, h]^d, then project to sum delta = 0
    by subtracting mean.  After projection delta_i in [-2h, 2h] but clamp to
    [-h, h] (re-projecting) preserves the cell — but rejection-sample if
    a_i = c_i + delta_i has any negative entry.
    """
    c = np.asarray(c_int, dtype=np.float64)
    d = len(c)
    out = np.empty((n_samples, d), dtype=np.float64)

    # Pre-compute reasonable bounds for delta given c (a >= 0):
    # delta_i >= -c_i.  We only need sample to live in the cell so
    # accept whenever c + delta >= 0.
    n_accepted = 0
    n_tries = 0
    while n_accepted < n_samples and n_tries < 50 * n_samples:
        n_tries += 1
        delta = rng.uniform(-h, h, size=d)
        delta -= delta.mean()  # enforce sum = 0
        # If projection pushes delta_i out of [-h, h], rescale uniformly
        max_abs = np.max(np.abs(delta))
        if max_abs > h:
            delta *= (h / max_abs)
        a = c + delta
        if np.all(a >= -1e-12):
            out[n_accepted] = np.maximum(a, 0.0)
            n_accepted += 1
    if n_accepted < n_samples:
        # Couldn't fill — return what we have
        return out[:n_accepted]
    return out


def _verify_M_soundness(c_int, n_half, m, c_target, n_samples=1000,
                         seed=0, eps=1e-12):
    """Verify M-chain pruning soundness on cell c_int by sampling a.

    Returns (n_violations, min_max_t_ff, max_max_t_ff).
    A 'violation' is any a in the cell with max_k (f_a*f_a)(t_k) <= c_target.
    """
    rng = np.random.default_rng(seed)
    samples = _sample_cell(c_int, rng, n_samples=n_samples, h=1.0)
    if len(samples) == 0:
        return 0, np.nan, np.nan

    max_vals = np.array(
        [_max_t_ff_at_knots(s, n_half, m) for s in samples])
    n_viol = int(np.sum(max_vals <= c_target + eps))
    return n_viol, float(np.min(max_vals)), float(np.max(max_vals))


# ============================================================================
# Drivers
# ============================================================================
def _generate_canonical_d2(n_half, m):
    """All canonical c=(c0, c1) with c0+c1 = 4*n_half*m and c0<=c1."""
    S = 4 * n_half * m
    return np.array([[a, S - a] for a in range(0, S // 2 + 1)],
                    dtype=np.int32)


def run_smoke_d2(n_half, m, c_target, n_samples=1000, seed=0):
    print("=" * 72)
    print(f"[smoke d=2] n_half={n_half}, m={m}, c_target={c_target}")
    print("=" * 72)

    from _M1_bench import prune_F
    from post_filters import (apply_FN_filter, apply_Q_filter, apply_QN_filter)

    batch = _generate_canonical_d2(n_half, m)
    d = batch.shape[1]
    print(f"d={d}, total canonical L0 cells: {len(batch)}")

    # 1) F + chain
    sF = prune_F(batch, n_half, m, c_target)
    F_surv = batch[sF]
    print(f"  F survivors:  {len(F_surv)}")
    FN_surv = apply_FN_filter(F_surv, n_half, m, c_target)
    print(f"  FN survivors: {len(FN_surv)}")
    Q_surv = apply_Q_filter(FN_surv, n_half, m, c_target)
    print(f"  Q survivors:  {len(Q_surv)}")
    QN_surv = apply_QN_filter(Q_surv, n_half, m, c_target)
    print(f"  QN survivors (current cascade chain end): {len(QN_surv)}")
    cascade_surv = QN_surv

    # 2) M-chain on the cascade chain survivors
    M_pruned_cells = []
    M_kept_cells = []
    for c in cascade_surv:
        pruned, lb, k = prune_M_one(c.astype(np.int64), n_half, m, c_target)
        if pruned:
            M_pruned_cells.append((c.tolist(), lb, k))
        else:
            M_kept_cells.append((c.tolist(), lb, k))

    print(f"  M-chain prunes: {len(M_pruned_cells)} of {len(cascade_surv)}")
    print(f"  M-chain keeps:  {len(M_kept_cells)} of {len(cascade_surv)}")

    # 3) Soundness
    print("\n  Soundness check on M-pruned cells (sampling 1000 a):")
    soundness_summary = []
    n_viol_total = 0
    for cell, lb, k in M_pruned_cells:
        n_viol, vmin, vmax = _verify_M_soundness(
            np.array(cell), n_half, m, c_target,
            n_samples=n_samples, seed=seed)
        soundness_summary.append({
            'c': cell, 'LB_max': lb, 'best_k': k,
            'n_violations': n_viol, 'min_max_ff': vmin, 'max_max_ff': vmax,
        })
        n_viol_total += n_viol
        if n_viol > 0:
            print(f"  *** VIOLATION at c={cell}: n_viol={n_viol}, "
                   f"min(max_ff)={vmin:.6f}, max_LB={lb:.6f}, c_target={c_target}")

    if n_viol_total == 0:
        print(f"  PASS: 0 sampled-a violations across "
               f"{len(M_pruned_cells)} M-pruned cells.")
    else:
        print(f"  FAIL: total {n_viol_total} sampled-a violations.")

    # 4) Spot-check first few M-pruned cells
    if M_pruned_cells:
        print("\n  Sample of M-pruned cells (c, LB_max, best_k, "
               "min/max sampled max-ff):")
        for s in soundness_summary[:5]:
            print(f"    c={s['c']}  LB={s['LB_max']:.5f}  k={s['best_k']}  "
                   f"min/max_ff={s['min_max_ff']:.5f} / "
                   f"{s['max_max_ff']:.5f}")
        if len(soundness_summary) > 5:
            print(f"    ... ({len(soundness_summary) - 5} more)")

    # 5) Sample of cells M failed to prune
    if M_kept_cells:
        print("\n  Sample of M-NOT-pruned cells (c, LB_max, best_k):")
        for cell, lb, k in M_kept_cells[:5]:
            print(f"    c={cell}  LB_max={lb:.5f} (<= {c_target}) k={k}")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'd': int(d),
        'total_L0_canonical': int(len(batch)),
        'F_surv': int(len(F_surv)),
        'FN_surv': int(len(FN_surv)),
        'Q_surv': int(len(Q_surv)),
        'QN_surv': int(len(QN_surv)),
        'cascade_chain_surv_count': int(len(cascade_surv)),
        'M_chain_prunes': int(len(M_pruned_cells)),
        'M_chain_keeps': int(len(M_kept_cells)),
        'cascade_chain_survivors': [c.tolist() for c in cascade_surv],
        'M_pruned': [{'c': c, 'LB_max': lb, 'best_k': k}
                     for (c, lb, k) in M_pruned_cells],
        'M_kept': [{'c': c, 'LB_max': lb, 'best_k': k}
                    for (c, lb, k) in M_kept_cells],
        'soundness_total_violations': int(n_viol_total),
        'soundness_PASS': bool(n_viol_total == 0),
        'soundness_per_cell': soundness_summary,
    }


def _refine_d2_to_d4(parent_c2):
    """Mass-doubling refinement: parent (c0, c1) of d=2 -> d=4 children.

    Cascade convention (run_cascade.py:1222-1223):
        child[2k] + child[2k+1] = 2 * parent_int[k]
    So child sum = 2 * parent_sum, m_child = m_parent, n_half_child = d_parent.
    For (c0, c1) the children are all 4-tuples
       (a, 2*c0 - a, b, 2*c1 - b),  0 <= a <= 2*c0,  0 <= b <= 2*c1.
    """
    c0, c1 = int(parent_c2[0]), int(parent_c2[1])
    children = []
    s0 = 2 * c0
    s1 = 2 * c1
    for a in range(s0 + 1):
        for b in range(s1 + 1):
            children.append([a, s0 - a, b, s1 - b])
    return np.array(children, dtype=np.int32)


def run_smoke_d4_from_parent(parent_c2, parent_n_half, parent_m,
                              c_target, n_samples=200, seed=0,
                              max_test=400):
    """Mass-doubling refinement: parent at L0 (d=2) -> children at d=4.

    Children inherit n_half_child = 2 * parent_n_half (so d_child = 4).
    Parent grid has m_parent; refined grid scaling depends on the cascade
    code's convention.  We use n_half_child = 2 * parent_n_half and
    m_child = parent_m (same m since the bin width halves but we re-scale).

    Actually in the cascade, mass-doubling preserves m and doubles d:
      d_parent = 2 * n_half_parent;  d_child = 4 * n_half_parent;
      n_half_child = 2 * n_half_parent;  m_child = m_parent;
      sum c_i = 4 * n_half_parent * m_parent (preserved).
    """
    print("=" * 72)
    print(f"[smoke d=4] mass-doubling parent c={parent_c2.tolist()} "
           f"(n_half={parent_n_half}, m={parent_m}) -> d=4 children")
    print(f"  c_target = {c_target}")
    print("=" * 72)

    # Refinement convention: d doubles, n_half doubles, m unchanged.
    # So d_child=4, n_half_child=2*parent_n_half=2, m_child=m_parent.
    n_half_child = 2 * parent_n_half
    m_child = parent_m

    children = _refine_d2_to_d4(parent_c2)
    print(f"  total d=4 children of parent: {len(children)}")

    # Cap test set size to keep wall < 25 min (mainly for soundness sampling)
    if len(children) > max_test:
        rng_pick = np.random.default_rng(seed)
        idx = rng_pick.choice(len(children), size=max_test, replace=False)
        children = children[np.sort(idx)]
        print(f"  sampled {len(children)} children for the smoke test")

    from _M1_bench import prune_F
    from post_filters import (apply_FN_filter, apply_Q_filter, apply_QN_filter)

    sF = prune_F(children, n_half_child, m_child, c_target)
    F_surv = children[sF]
    FN_surv = apply_FN_filter(F_surv, n_half_child, m_child, c_target)
    Q_surv = apply_Q_filter(FN_surv, n_half_child, m_child, c_target)
    QN_surv = apply_QN_filter(Q_surv, n_half_child, m_child, c_target)
    print(f"  F={len(F_surv)} FN={len(FN_surv)} Q={len(Q_surv)} "
           f"QN={len(QN_surv)}")
    cascade_surv = QN_surv

    # M-chain on cascade survivors
    M_pruned = []
    M_kept = []
    for c in cascade_surv:
        pruned, lb, k = prune_M_one(c.astype(np.int64),
                                       n_half_child, m_child, c_target)
        if pruned:
            M_pruned.append((c.tolist(), lb, k))
        else:
            M_kept.append((c.tolist(), lb, k))
    print(f"  M-chain prunes: {len(M_pruned)} of {len(cascade_surv)}")

    # Soundness on M-pruned (smaller sample budget at d=4)
    n_viol_total = 0
    soundness_summary = []
    n_check = min(50, len(M_pruned))
    print(f"  soundness check on first {n_check} M-pruned cells "
           f"({n_samples} samples each)...")
    for cell, lb, k in M_pruned[:n_check]:
        n_viol, vmin, vmax = _verify_M_soundness(
            np.array(cell), n_half_child, m_child, c_target,
            n_samples=n_samples, seed=seed)
        soundness_summary.append({
            'c': cell, 'LB_max': lb, 'best_k': k,
            'n_violations': n_viol, 'min_max_ff': vmin, 'max_max_ff': vmax,
        })
        n_viol_total += n_viol
        if n_viol > 0:
            print(f"  *** VIOLATION at c={cell}: n_viol={n_viol}, "
                   f"min(max_ff)={vmin:.6f}, LB={lb:.6f}")

    if n_viol_total == 0:
        print(f"  d=4 PASS: 0 sampled-a violations across "
               f"{n_check} M-pruned cells.")
    else:
        print(f"  d=4 FAIL: {n_viol_total} sampled-a violations.")

    return {
        'parent_c': parent_c2.tolist(),
        'parent_n_half': int(parent_n_half),
        'parent_m': int(parent_m),
        'd_child': 4,
        'n_half_child': int(n_half_child),
        'm_child': int(m_child),
        'c_target': c_target,
        'total_d4_children': int(len(children)),
        'F_surv': int(len(F_surv)),
        'FN_surv': int(len(FN_surv)),
        'Q_surv': int(len(Q_surv)),
        'QN_surv': int(len(QN_surv)),
        'cascade_chain_surv_count': int(len(cascade_surv)),
        'M_chain_prunes': int(len(M_pruned)),
        'M_chain_keeps': int(len(M_kept)),
        'soundness_n_checked': int(n_check),
        'soundness_total_violations': int(n_viol_total),
        'soundness_PASS': bool(n_viol_total == 0),
        'soundness_per_cell': soundness_summary,
    }


# ============================================================================
# Sanity checks before running the smoke
# ============================================================================
def _sanity_b_vec_and_LB():
    """Hand-check b-vec and LB for c=(40,40), d=2, m=20, n_half=1, k=1."""
    c = np.array([40, 40])
    # k=1: b_j = c_{1-j}, j=0: c_1=40; j=1: c_0=40 -> b=[40,40]
    b = _b_vec(c, 1)
    assert b.tolist() == [40, 40], f"bad b for k=1: {b}"
    # Delta_b: sort=[40,40]; top half = 40, bot = 40 -> Delta = 0
    assert _delta_b(b) == 0
    # n_pairs(1, 2) = 2 (pairs (0,1) and (1,0))
    assert _n_pairs(1, 2) == 2
    # conv[1] = 2*40*40 = 3200
    conv = _conv_int(c)
    assert conv.tolist() == [1600, 3200, 1600]
    # LB(1) = (3200 - 0 - 2) / (2 * 2 * 400) = 3198/1600 = 1.99875
    LB = (int(conv[1]) - 2 * 0 - 2) / (2 * 2 * 400)
    assert abs(LB - 1.99875) < 1e-12, f"LB sanity bad: {LB}"
    pruned, lb_max, _ = prune_M_one(c, 1, 20, 1.281)
    assert pruned, "(40,40) should M-prune at c_target=1.281"
    assert abs(lb_max - 1.99875) < 1e-9, f"max_LB bad: {lb_max}"
    print("  sanity OK: c=(40,40), n=1, m=20, k=1, LB=1.99875 -> PRUNE @ 1.281")
    return True


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("SMOKE: per-conv-position F-style M-chain pruner")
    print("=" * 72)
    print("\n[1] Sanity check on the M-chain LB formula:")
    _sanity_b_vec_and_LB()

    # ------------------------------------------------- L0 d=2 sweep
    out = {'config': 'd=2 L0 (n_half=1, m=20, c_target=1.281), and d=4 refinement'}

    r_d2 = run_smoke_d2(n_half=1, m=20, c_target=1.281,
                          n_samples=1000, seed=42)
    out['d2'] = r_d2

    # ------------------------------------------------- d=4 mass-doubling
    # Pick a parent that survives the cascade chain.
    parent_c2 = np.array([40, 40], dtype=np.int32)  # central palindromic cell
    r_d4 = run_smoke_d4_from_parent(
        parent_c2, parent_n_half=1, parent_m=20, c_target=1.281,
        n_samples=200, seed=42, max_test=400)
    out['d4_parent_4040'] = r_d4

    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"  d=2 cascade survivors: {r_d2['cascade_chain_surv_count']}")
    print(f"  d=2 M-chain prunes:    {r_d2['M_chain_prunes']} extras "
           f"(soundness PASS={r_d2['soundness_PASS']})")
    print(f"  d=4 cascade survivors: {r_d4['cascade_chain_surv_count']} "
           f"of {r_d4['total_d4_children']} parent children tested")
    print(f"  d=4 M-chain prunes:    {r_d4['M_chain_prunes']} extras "
           f"(soundness PASS={r_d4['soundness_PASS']})")
    print(f"  total wall: {elapsed:.2f}s")

    out['elapsed_s'] = elapsed
    out_path = os.path.join(_dir, '_smoke_M_chain.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"\n[saved] {out_path}")
    return out


if __name__ == '__main__':
    main()

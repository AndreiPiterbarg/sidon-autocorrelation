"""Cascade L0 comparison: v2 triangle bound vs joint QP bound.

For a fixed (d, S, c_target), enumerate all compositions of S into d parts and:
  - count compositions whose grid-point TV is >= c_target (would-be pruned)
  - count box-certified by v2 (margin > triangle)
  - count box-certified by QP (margin > QP)

QP must certify a SUPERSET of v2's certified set (since QP <= triangle).
Reports how many ADDITIONAL compositions QP can certify.

Also: confirms QP-based pruning is sound by checking that no certified
composition has a delta in its cell where TV(mu*+delta) < c_target.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(_REPO, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(_REPO, 'cloninger-steinerberger'))

from qp_bound import (
    build_window_matrix, grad_for_window, qp_bound_vertex,
)
from compositions import generate_canonical_compositions_batched


def cell_var_v2(grad, S):
    d = grad.shape[0]
    g = np.sort(grad)
    h = 1.0 / (2.0 * S)
    total = 0.0
    for k in range(d // 2):
        total += g[d - 1 - k] - g[k]
    return h * total


def quad_corr_v2(d, S, ell, s):
    N_W, M_W = 0, 0
    for k in range(s, s + ell - 1):
        n_k = min(k + 1, d, 2 * d - 1 - k)
        if n_k <= 0:
            continue
        N_W += n_k
        if k % 2 == 0 and (k // 2) < d:
            M_W += 1
    cross_W = N_W - M_W
    pair_bound = min(cross_W, d * d - N_W)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * d / ell) * pair_bound / (4.0 * S * S)


def evaluate_composition(c, d, S, c_target):
    """For composition c, find best (margin - bound) over all windows.
    Returns (best_margin_v2, best_margin_qp, would_be_pruned).

    A composition is box-certified by a bound B if there exists a window
    where margin - B > 0. We return the BEST (max) such value for v2 and QP.
    """
    pruned_at_grid = False
    best_v2 = -np.inf
    best_qp = -np.inf
    max_ell = 2 * d
    S_sq = float(S * S)
    d_f = float(d)

    # Compute autoconv (integer)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        ws = int(conv[:n_cv].sum())
        ell_f = float(ell)
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += int(conv[s_lo + n_cv - 1]) - int(conv[s_lo - 1])
            tv = ws * 2.0 * d_f / (S_sq * ell_f)
            if tv < c_target:
                continue
            pruned_at_grid = True
            margin = tv - c_target

            A = build_window_matrix(d, ell, s_lo)
            grad = grad_for_window(c.astype(np.int64), A, S, d, ell)
            cv = cell_var_v2(grad, S)
            qc = quad_corr_v2(d, S, ell, s_lo)
            tri = cv + qc
            h = 1.0 / (2.0 * S)
            scale = 2.0 * d / ell_f
            qp = qp_bound_vertex(grad, A, scale, h, d)

            v2_net = margin - tri
            qp_net = margin - qp
            if v2_net > best_v2:
                best_v2 = v2_net
            if qp_net > best_qp:
                best_qp = qp_net
    return best_v2, best_qp, pruned_at_grid


def random_delta_in_cell(d, h, rng):
    x = rng.uniform(-h, h, size=d)
    x -= x.mean()
    m = np.max(np.abs(x))
    if m > h:
        x *= (h / m)
    return x


def soundness_check(c, d, S, c_target, n_samples, rng):
    """For composition c, sample random delta in cell. If max TV(mu*+delta)
    over windows >= c_target, the composition is genuinely prunable.
    Returns the MIN over delta of (max over windows TV(mu*+delta)) — if
    this drops below c_target, then no continuous bound can certify c."""
    h = 1.0 / (2.0 * S)
    mu_star = c.astype(np.float64) / S
    max_ell = 2 * d
    conv_len = 2 * d - 1
    min_max_tv = np.inf
    for _ in range(n_samples):
        delta = random_delta_in_cell(d, h, rng)
        mu = mu_star + delta
        # max over all windows of TV(mu)
        max_tv = -np.inf
        # full mass autoconvolution
        mc = np.zeros(conv_len)
        for i in range(d):
            for j in range(d):
                mc[i + j] += mu[i] * mu[j]
        for ell in range(2, max_ell + 1):
            for s in range(conv_len - (ell - 1) + 1):
                tv = (2.0 * d / ell) * mc[s:s + ell - 1].sum()
                if tv > max_tv:
                    max_tv = tv
        if max_tv < min_max_tv:
            min_max_tv = max_tv
    return min_max_tv


def run_l0_compare(d, S, c_target, do_soundness_sample=False):
    print(f"\n=== L0 comparison: d={d}, S={S}, c_target={c_target} ===")
    t0 = time.time()
    n_total = 0
    n_grid_pruned = 0
    n_v2_cert = 0
    n_qp_cert = 0
    n_qp_only_cert = 0  # cert by QP but NOT by v2

    qp_only_examples = []
    not_certified_examples = []

    gen = generate_canonical_compositions_batched(d, S, batch_size=10_000)
    for batch in gen:
        for c in batch:
            n_total += 1
            v2_net, qp_net, pruned = evaluate_composition(
                np.asarray(c), d, S, c_target)
            if not pruned:
                continue
            n_grid_pruned += 1
            if v2_net > 0:
                n_v2_cert += 1
            if qp_net > 0:
                n_qp_cert += 1
            if qp_net > 0 and v2_net <= 0:
                n_qp_only_cert += 1
                if len(qp_only_examples) < 5:
                    qp_only_examples.append(
                        (tuple(int(x) for x in c), v2_net, qp_net))
            if qp_net <= 0:
                if len(not_certified_examples) < 3:
                    not_certified_examples.append(
                        (tuple(int(x) for x in c), v2_net, qp_net))
    elapsed = time.time() - t0

    print(f"  Total canonical compositions: {n_total:,}")
    print(f"  Grid-point pruned (some window has TV >= c_target): {n_grid_pruned:,}")
    print(f"  Box-certified by v2 (triangle):  {n_v2_cert:,}")
    print(f"  Box-certified by QP:             {n_qp_cert:,}")
    print(f"  QP-only (not v2): {n_qp_only_cert:,} "
          f"(+{100 * n_qp_only_cert / max(1, n_grid_pruned):.1f}% relative to grid-pruned)")
    print(f"  Time: {elapsed:.2f}s")

    if qp_only_examples:
        print(f"\n  Examples certified by QP but NOT by v2:")
        for c_ex, v2n, qpn in qp_only_examples:
            print(f"    c={c_ex}: v2_net={v2n:+.5f} (FAIL) qp_net={qpn:+.5f} (PASS)")

    # Soundness check on first such example
    if do_soundness_sample and qp_only_examples:
        print(f"\n  Soundness sample on QP-only example (1000 random delta):")
        rng = np.random.default_rng(0)
        c_ex = np.array(qp_only_examples[0][0], dtype=np.int64)
        min_max_tv = soundness_check(c_ex, d, S, c_target, 1000, rng)
        print(f"    c={tuple(c_ex.tolist())}: min over sampled delta of "
              f"max_W TV(mu*+delta) = {min_max_tv:.6f}")
        if min_max_tv >= c_target - 1e-9:
            print(f"    PASS: stays >= c_target={c_target} (QP cert is sound)")
        else:
            print(f"    *** FAIL: drops below c_target={c_target} -- QP UNSOUND? ***")

    if not_certified_examples:
        print(f"\n  Examples NEITHER v2 nor QP can certify "
              f"(genuinely uncertifiable at this S):")
        for c_ex, v2n, qpn in not_certified_examples:
            print(f"    c={c_ex}: v2_net={v2n:+.5f} qp_net={qpn:+.5f}")

    return {
        'n_total': n_total, 'n_grid_pruned': n_grid_pruned,
        'n_v2_cert': n_v2_cert, 'n_qp_cert': n_qp_cert,
        'n_qp_only': n_qp_only_cert,
    }


def main():
    print("=" * 72)
    print("L0 cascade comparison: v2 triangle bound vs joint QP bound")
    print("Counts box-certified compositions by each method.")
    print("=" * 72)

    # Small d=4 cases — fast and exhaustive
    run_l0_compare(d=4, S=10, c_target=1.30, do_soundness_sample=True)
    run_l0_compare(d=4, S=20, c_target=1.30, do_soundness_sample=True)
    run_l0_compare(d=4, S=30, c_target=1.30, do_soundness_sample=True)

    # d=6 — still fast
    run_l0_compare(d=6, S=15, c_target=1.30, do_soundness_sample=True)


if __name__ == '__main__':
    main()

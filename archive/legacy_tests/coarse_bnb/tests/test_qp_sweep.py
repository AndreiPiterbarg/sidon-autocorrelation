"""Sweep certification rates across (d, S, c_target) for v2 vs QP.

Goal: identify regimes where the QP bound recovers compositions that v2
cannot certify, especially near the cliff where the cascade can fully prune.

For each (d, S, c_target):
  - n_total                 = number of canonical compositions
  - n_grid_pruned           = some window has TV(c) >= c_target (necessary)
  - n_uncertified_by_v2     = grid-pruned but v2 fails to box-certify
  - n_uncertified_by_qp     = grid-pruned but QP fails to box-certify
  - n_recovered_by_qp       = certified by QP but not by v2
  - certifies_all (v2 / QP) = does each method box-certify ALL grid-pruned comps?

`certifies_all` is the cascade-level success metric: when True, the L0
proof at this (d, S, c_target) is COMPLETE (no survivors at L0).
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

from qp_bound import build_window_matrix, grad_for_window, qp_bound_vertex
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
    """For composition c, return (best v2_net, best qp_net, grid_pruned).

    A composition is box-certified by a method if there exists ANY window
    where margin > bound. We search across all windows and report the best.
    QP is computed only when v2 fails (lazy — saves time when v2 already
    certifies).
    """
    pruned_at_grid = False
    best_v2 = -np.inf
    best_qp = -np.inf
    max_ell = 2 * d
    S_sq = float(S * S)
    d_f = float(d)

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

    pruning_windows = []  # (ell, s, margin) for windows that grid-prune

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
            pruning_windows.append((ell, s_lo, margin))

    if not pruned_at_grid:
        return (-np.inf, -np.inf, False)

    # Sort by margin descending (try most promising windows first)
    pruning_windows.sort(key=lambda x: -x[2])

    for (ell, s_lo, margin) in pruning_windows:
        ell_f = float(ell)
        A = build_window_matrix(d, ell, s_lo)
        grad = grad_for_window(c.astype(np.int64), A, S, d, ell)
        cv = cell_var_v2(grad, S)
        qc = quad_corr_v2(d, S, ell, s_lo)
        v2_net = margin - (cv + qc)
        if v2_net > best_v2:
            best_v2 = v2_net

        h = 1.0 / (2.0 * S)
        scale = 2.0 * d / ell_f
        qp = qp_bound_vertex(grad, A, scale, h, d)
        qp_net = margin - qp
        if qp_net > best_qp:
            best_qp = qp_net
        # Early exit: if QP already certifies and we've checked top windows
        if qp_net > 0.05:
            break

    return (best_v2, best_qp, True)


def run_sweep(d, S, c_target_list, verbose=True):
    rows = []
    t0 = time.time()
    n_total = 0
    by_target = {ct: {'pruned': 0, 'v2_uncert': 0, 'qp_uncert': 0,
                      'recovered': 0, 'examples': []}
                 for ct in c_target_list}
    gen = generate_canonical_compositions_batched(d, S, batch_size=10_000)
    for batch in gen:
        for c in batch:
            n_total += 1
            for ct in c_target_list:
                v2_net, qp_net, pruned = evaluate_composition(
                    np.asarray(c), d, S, ct)
                if not pruned:
                    continue
                d_ct = by_target[ct]
                d_ct['pruned'] += 1
                if v2_net <= 0:
                    d_ct['v2_uncert'] += 1
                if qp_net <= 0:
                    d_ct['qp_uncert'] += 1
                if v2_net <= 0 and qp_net > 0:
                    d_ct['recovered'] += 1
                    if len(d_ct['examples']) < 3:
                        d_ct['examples'].append(
                            (tuple(int(x) for x in c), v2_net, qp_net))

    elapsed = time.time() - t0
    if verbose:
        print(f"\n=== d={d}, S={S}, n_total={n_total:,}, "
              f"elapsed={elapsed:.1f}s ===")
        print(f"{'c_target':>9} {'pruned':>8} {'v2_uncert':>10} "
              f"{'qp_uncert':>10} {'recovered':>10} "
              f"{'v2_full?':>9} {'qp_full?':>9}")
        for ct in c_target_list:
            d_ct = by_target[ct]
            v2_full = d_ct['v2_uncert'] == 0 and d_ct['pruned'] > 0
            qp_full = d_ct['qp_uncert'] == 0 and d_ct['pruned'] > 0
            no_pruning = d_ct['pruned'] < n_total
            v2_str = 'YES' if v2_full else 'no '
            qp_str = 'YES' if qp_full else 'no '
            if no_pruning:
                v2_str = qp_str = '(some grid not pruned)'
                v2_str = 'n/a'
                qp_str = 'n/a'
            print(f"{ct:>9.4f} {d_ct['pruned']:>8} {d_ct['v2_uncert']:>10} "
                  f"{d_ct['qp_uncert']:>10} {d_ct['recovered']:>10} "
                  f"{v2_str:>9} {qp_str:>9}")
        # Print recovery examples
        for ct in c_target_list:
            d_ct = by_target[ct]
            if d_ct['recovered'] > 0:
                print(f"  c_target={ct} examples (recovered by QP):")
                for c_ex, v2n, qpn in d_ct['examples']:
                    print(f"    c={c_ex}: v2_net={v2n:+.5f} qp_net={qpn:+.5f}")
    return n_total, by_target


def main():
    print("=" * 72)
    print("Certification sweep: v2 triangle vs joint QP, across (d, S, c_target)")
    print("Question: at what (S, c_target) does each method certify ALL "
          "grid-pruned compositions?")
    print("=" * 72)

    # d=4: cheap
    targets_4 = [1.25, 1.28, 1.2802, 1.30, 1.32, 1.35]
    for S in (10, 20, 30, 50):
        run_sweep(d=4, S=S, c_target_list=targets_4)

    # d=6: small
    targets_6 = [1.25, 1.28, 1.2802, 1.30, 1.32]
    for S in (10, 15, 20, 30):
        run_sweep(d=6, S=S, c_target_list=targets_6)

    # d=8: cap at S=20 to keep it under a minute
    targets_8 = [1.25, 1.28, 1.30, 1.32, 1.35]
    for S in (8, 12, 16):
        run_sweep(d=8, S=S, c_target_list=targets_8)


if __name__ == '__main__':
    main()

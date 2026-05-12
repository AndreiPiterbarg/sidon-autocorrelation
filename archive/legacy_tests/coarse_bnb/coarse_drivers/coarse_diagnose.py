#!/usr/bin/env python
"""Diagnose box cert failure: analyze margin/cell_var/quad_corr distribution.

For each pruned composition, report margin, cell_var, quad_corr separately.
Find what's bottlenecking box cert and whether there's a parameter regime
where it can pass for c >= 1.28.
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np
from numba import njit
from pruning import count_compositions, asymmetry_threshold
from compositions import generate_canonical_compositions_batched
from run_cascade import _build_pair_prefix


@njit(cache=True)
def _analyze_pruned(batch_int, d, S, c_target, prefix_nk, prefix_mk):
    """For each pruned composition, compute margin, cell_var, quad_corr
    for the BEST window (the one with highest net = margin - cv - qc).

    Returns arrays: (margin_arr, cell_var_arr, quad_corr_arr, net_arr)
    for pruned compositions only.
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    # Output arrays (overallocated)
    margin_arr = np.empty(B, dtype=np.float64)
    cv_arr = np.empty(B, dtype=np.float64)
    qc_arr = np.empty(B, dtype=np.float64)
    net_arr = np.empty(B, dtype=np.float64)
    n_pruned = 0

    grad_arr = np.empty(d, dtype=np.float64)

    for b in range(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        best_net = np.float64(-1e30)
        best_margin = np.float64(0)
        best_cv = np.float64(0)
        best_qc = np.float64(0)
        any_pruned = False

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                if ws > dyn_it:
                    any_pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g
                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var /= (2.0 * S_d)

                    hi_idx = s_lo + ell - 1
                    N_W = prefix_nk[hi_idx] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi_idx] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl = d_sq - N_W
                    pb = min(cross_W, compl)
                    if pb > 0:
                        qc = (2.0 * d_d / ell_f) * np.float64(pb) * inv_4S2
                    else:
                        qc = 0.0

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net
                        best_margin = margin
                        best_cv = cell_var
                        best_qc = qc

        if any_pruned:
            margin_arr[n_pruned] = best_margin
            cv_arr[n_pruned] = best_cv
            qc_arr[n_pruned] = best_qc
            net_arr[n_pruned] = best_net
            n_pruned += 1

    return (margin_arr[:n_pruned], cv_arr[:n_pruned],
            qc_arr[:n_pruned], net_arr[:n_pruned])


def analyze(d0, S, c_target):
    """Full diagnostic analysis."""
    n_total = count_compositions(d0, S)
    prefix_nk, prefix_mk = _build_pair_prefix(d0)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: d0={d0}, S={S}, c_target={c_target}")
    print(f"  compositions={n_total:,}")
    print(f"{'='*70}")

    all_margins = []
    all_cvs = []
    all_qcs = []
    all_nets = []
    n_survived = 0
    n_processed = 0

    threshold_a = asymmetry_threshold(c_target)
    gen = generate_canonical_compositions_batched(d0, S, batch_size=200_000)

    for batch in gen:
        n_processed += len(batch)
        # Asymmetry filter
        left_bins = d0 // 2
        left = batch[:, :left_bins].sum(axis=1).astype(np.float64)
        left_frac = left / float(S)
        asym_mask = (left_frac > 1 - threshold_a) & (left_frac < threshold_a)
        batch = batch[asym_mask]
        if len(batch) == 0:
            continue

        margins, cvs, qcs, nets = _analyze_pruned(
            batch, d0, S, c_target, prefix_nk, prefix_mk)
        n_survived += len(batch) - len(margins)
        if len(margins) > 0:
            all_margins.append(margins)
            all_cvs.append(cvs)
            all_qcs.append(qcs)
            all_nets.append(nets)

    if not all_margins:
        print("  No pruned compositions!")
        return

    margins = np.concatenate(all_margins)
    cvs = np.concatenate(all_cvs)
    qcs = np.concatenate(all_qcs)
    nets = np.concatenate(all_nets)

    print(f"  Pruned: {len(margins):,}, Survived: {n_survived:,}")
    print(f"\n  {'':>12} {'min':>10} {'p5':>10} {'median':>10} {'p95':>10} {'max':>10}")
    print(f"  {'-'*62}")
    for name, arr in [('margin', margins), ('cell_var', cvs),
                       ('quad_corr', qcs), ('net', nets)]:
        if len(arr) == 0:
            continue
        print(f"  {name:>12} {np.min(arr):>10.6f} {np.percentile(arr,5):>10.6f} "
              f"{np.median(arr):>10.6f} {np.percentile(arr,95):>10.6f} "
              f"{np.max(arr):>10.6f}")

    n_pass = int(np.sum(nets >= 0))
    n_fail = int(np.sum(nets < 0))
    print(f"\n  Box cert: {n_pass:,} pass, {n_fail:,} fail "
          f"({n_pass/(n_pass+n_fail)*100:.1f}% pass)")

    if n_fail > 0:
        # What would S need to be for worst case to pass?
        # net = margin - cv - qc
        # cv ~ cv_raw / S, qc ~ qc_raw / S^2
        # Need: margin > cv_raw/S + qc_raw/S^2
        # For the worst case:
        worst_idx = np.argmin(nets)
        m_w = margins[worst_idx]
        cv_w = cvs[worst_idx]
        qc_w = qcs[worst_idx]
        # cv scales as 1/S, qc as 1/S^2
        # At current S: cv_w = cv_raw/S => cv_raw = cv_w * S
        # margin stays constant (independent of S)
        cv_raw = cv_w * S
        qc_raw = qc_w * S * S
        # Need: m_w > cv_raw/S_new + qc_raw/S_new^2
        # Approximate: S_new > cv_raw / m_w (ignoring qc term)
        if m_w > 0:
            s_needed = cv_raw / m_w
            print(f"\n  Worst case: margin={m_w:.6f}, cv={cv_w:.6f}, qc={qc_w:.6f}")
            print(f"  cv_raw = {cv_raw:.2f}, S needed ~ {s_needed:.0f}")
            print(f"  L0 comps at that S: ~{count_compositions(d0, int(s_needed)):,}")
        else:
            print(f"\n  Worst case has zero/negative margin — cannot fix with S")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Key question: at d0=4, what S is needed for c=1.28?
    for c in [1.25, 1.28, 1.30]:
        for d0 in [4, 6, 8, 10, 12]:
            S_test = max(20, d0 * 3)
            n = count_compositions(d0, S_test)
            if n > 50_000_000:
                continue
            analyze(d0, S_test, c)


if __name__ == '__main__':
    main()

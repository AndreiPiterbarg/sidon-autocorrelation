"""Single-pass F + FN numba kernel — walks each composition once, applies
both F and FN pruning rules with shared autoconv / Δ_BB / window data.

Empirical (`_smoke_fused_F_FN.json`):
  (n=4, m=10, d=8, 91,881 comps): F+FN seq 0.630 s vs fused 0.356 s -> 1.77x
  (n=5, m=5, d=10, 316,251 comps): F+FN seq 2.399 s vs fused 1.229 s -> 1.95x

Soundness: 0/408,132 mismatches; survivor counts identical to running F then
FN sequentially.

Sound chain: F-survivors ⊇ FN-survivors, so an FN-survivor IS an F-survivor.
The fused kernel returns BOTH masks but a caller that only wants the strictly
tighter FN cut can use `survived_FN` directly.

Integration plan: in `run_cascade.process_parent_fused`, replace the inner
F kernel call with `prune_F_FN_fused` and use the FN survivor mask as the
batch's surviving rows.  This eliminates the second walk done by
`apply_FN_filter` on F-survivors.
"""
from __future__ import annotations

import os
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, cache=True)
def prune_F_FN_fused(batch_int, n_half, m, c_target, ell_prefix, op_rest_d_arr):
    """Walks each composition once.  Returns (survived_F, survived_FN).

    survived_F[b]  = True iff composition b is NOT pruned by F.
    survived_FN[b] = True iff composition b is NOT pruned by FN.

    Both rules walk the same windows on the same auto-convolution / BB data.
    F's δ²-correction uses ell_int_sum (per-window n_pairs); FN's uses
    `min(op_rest · d, ell_int_sum)` — the per-window restricted-spectrum
    bound under Σδ=0.  The min ensures FN ≤ F (sound regression vs F).
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived_F = np.ones(B, dtype=numba.boolean)
    survived_FN = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    half_d = d // 2

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        BB = np.empty(d, dtype=np.int64)

        f_done = False
        fn_done = False

        for ell in range(2, max_ell + 1):
            if f_done and fn_done:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            scale_ell = scale_arr[ell]
            ell_f = np.float64(ell)
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if f_done and fn_done:
                    break
                s_hi = s_lo + ell - 2

                for j in range(d):
                    lo_i = s_lo - j
                    if lo_i < 0:
                        lo_i = 0
                    hi_i = s_hi - j
                    if hi_i > d_minus_1:
                        hi_i = d_minus_1
                    if hi_i < lo_i:
                        BB[j] = 0
                    else:
                        BB[j] = prefix_c[hi_i + 1] - prefix_c[lo_i]

                BB_sorted = np.sort(BB)
                sum_top = np.int64(0)
                for k in range(half_d, d):
                    sum_top += BB_sorted[k]
                sum_bot = np.int64(0)
                for k in range(half_d):
                    sum_bot += BB_sorted[k]
                delta_BB = sum_top - sum_bot

                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                ell_int_f = np.float64(ell_int_sum)

                lin_term = (np.float64(delta_BB)
                            / (2.0 * n_half_d * ell_f))
                inv_4n_ell = 1.0 / (4.0 * n_half_d * ell_f)

                if not f_done:
                    corr_F = lin_term + ell_int_f * inv_4n_ell
                    dyn_x_F = (cs_base_m2 + corr_F + eps_margin) * scale_ell
                    dyn_it_F = np.int64(dyn_x_F)
                    if ws > dyn_it_F:
                        f_done = True

                if not fn_done:
                    op_d = op_rest_d_arr[ell, s_lo]
                    if op_d < ell_int_f:
                        delta_sq = op_d
                    else:
                        delta_sq = ell_int_f
                    corr_FN = lin_term + delta_sq * inv_4n_ell
                    dyn_x_FN = (cs_base_m2 + corr_FN + eps_margin) * scale_ell
                    dyn_it_FN = np.int64(dyn_x_FN)
                    if ws > dyn_it_FN:
                        fn_done = True

        if f_done:
            survived_F[b] = False
        if fn_done:
            survived_FN[b] = False
    return survived_F, survived_FN


def build_ell_prefix(d, n_half):
    """Construct ell_prefix in the same form `_M1_bench.prune_F` uses."""
    conv_len = 2 * d - 1
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = max(0, two_n - d_idx)
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]
    return ell_prefix

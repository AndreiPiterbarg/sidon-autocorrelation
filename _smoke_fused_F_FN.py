"""Smoke test: fused F+FN kernel.

Walk compositions ONCE; compute autoconv, BB, Δ_BB, ws, ell_int_sum once per
window; apply BOTH F's and FN's pruning rules in the same inner loop.

Since FN's δ² bound is min(op_rest*d, ell_int_sum), we have FN ≤ F per-window
(FN is always at least as tight as F).  So FN-survivors ⊆ F-survivors and the
fused mask = FN-survivors element-wise.

We still maintain a separate "would F prune?" tracker so we can return
(survived_F, survived_FN) and verify the equivalence sFused == (sF & sFN) holds.

DERIVATION:
  Per-window correction (m² units):
    F:  Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
    FN: Δ_BB/(2n·ell) + min(op_rest·d, ell_int_sum)/(4n·ell)
  ws = window-sum of conv (integer); prune-when ws > (cs_base_m2 + corr_w + eps) * scale_ell
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _N_bench import precompute_op_norm_restricted
from _FN_bench import prune_FN


# ============================================================
# Fused F+FN kernel: walks compositions ONCE; computes BB / Δ_BB
# / ws / ell_int_sum per window once; checks BOTH F's and FN's
# pruning rules.  Returns (survived_F, survived_FN).  Since FN ≤ F
# per-window, FN-survivors ⊆ F-survivors.
# ============================================================
@njit(parallel=True, cache=True)
def prune_F_FN_fused(batch_int, n_half, m, c_target, ell_prefix, op_rest_d_arr):
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
        # Autoconvolution (computed once)
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

        f_done = False   # F has already pruned this composition
        fn_done = False  # FN has already pruned this composition

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

                # Compute BB[j] once for this window
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

                # Δ_BB once
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

                # F: corr_F = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
                if not f_done:
                    corr_F = lin_term + ell_int_f * inv_4n_ell
                    dyn_x_F = (cs_base_m2 + corr_F + eps_margin) * scale_ell
                    dyn_it_F = np.int64(dyn_x_F)
                    if ws > dyn_it_F:
                        f_done = True

                # FN: corr_FN = Δ_BB/(2n·ell) + min(op_rest·d, ell_int_sum)/(4n·ell)
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


# ============================================================
# Test driver
# ============================================================
def build_ell_prefix(d, n_half):
    conv_len = 2 * d - 1
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = (k + 1) - two_n
        if d_idx < 0:
            d_idx = -d_idx
        v = two_n - d_idx
        if v < 0:
            v = 0
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]
    return ell_prefix


def run_smoke(n_half, m, c_target, batch_size=200_000):
    d = 2 * n_half
    S_half = 2 * n_half * m
    n_total = count_compositions(n_half, S_half)
    conv_len = 2 * d - 1
    max_ell = 2 * d

    print(f"\n=== n_half={n_half}, m={m}, d={d}, c={c_target} ===")
    print(f"     palindromic comps: {n_total:,}")

    # Precompute
    ell_prefix = build_ell_prefix(d, n_half)
    print(f"     precomputing op-norm restricted...", end=' ', flush=True)
    t_pre = time.time()
    op_rest, _ = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d
    print(f"[{time.time()-t_pre:.2f}s]")

    # JIT warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    print(f"     warming JIT...", end=' ', flush=True)
    t_jit = time.time()
    prune_F(warm, n_half, m, c_target)
    prune_FN(warm, n_half, m, c_target, ell_prefix, op_rest_d)
    prune_F_FN_fused(warm, n_half, m, c_target, ell_prefix, op_rest_d)
    print(f"[{time.time()-t_jit:.2f}s]")

    n_proc = 0
    surv_F_seq = 0
    surv_FN_seq = 0
    surv_F_fused = 0
    surv_FN_fused = 0
    n_mismatch_F = 0
    n_mismatch_FN = 0
    t_seq_F = 0.0
    t_seq_FN = 0.0
    t_fused = 0.0

    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)

        # Sequential: F then FN (FN walks all compositions, not just F-survivors,
        # to match _FN_bench.py's run() pattern; that's the fair comparison).
        ta = time.time()
        sF_seq = prune_F(batch, n_half, m, c_target)
        t_seq_F += time.time() - ta

        tb = time.time()
        sFN_seq = prune_FN(batch, n_half, m, c_target, ell_prefix, op_rest_d)
        t_seq_FN += time.time() - tb

        # Fused
        tc = time.time()
        sF_f, sFN_f = prune_F_FN_fused(batch, n_half, m, c_target,
                                         ell_prefix, op_rest_d)
        t_fused += time.time() - tc

        surv_F_seq += int(sF_seq.sum())
        surv_FN_seq += int(sFN_seq.sum())
        surv_F_fused += int(sF_f.sum())
        surv_FN_fused += int(sFN_f.sum())

        # Soundness checks
        n_mismatch_F += int(np.sum(sF_seq != sF_f))
        n_mismatch_FN += int(np.sum(sFN_seq != sFN_f))

    seq_total = t_seq_F + t_seq_FN
    speedup = seq_total / max(t_fused, 1e-9)

    print(f"     F survivors:  seq={surv_F_seq:,}   fused={surv_F_fused:,}   "
          f"mismatch={n_mismatch_F}")
    print(f"     FN survivors: seq={surv_FN_seq:,}   fused={surv_FN_fused:,}   "
          f"mismatch={n_mismatch_FN}")
    print(f"     time F (seq): {t_seq_F:.3f}s")
    print(f"     time FN (seq): {t_seq_FN:.3f}s")
    print(f"     time F+FN (seq total): {seq_total:.3f}s")
    print(f"     time fused: {t_fused:.3f}s")
    print(f"     SPEEDUP: {speedup:.2f}x")
    sound = (n_mismatch_F == 0 and n_mismatch_FN == 0)
    print(f"     SOUND: {sound}")

    return {
        'n_half': n_half, 'm': m, 'd': d, 'c_target': c_target,
        'n_processed': n_proc,
        'surv_F_seq': surv_F_seq, 'surv_F_fused': surv_F_fused,
        'surv_FN_seq': surv_FN_seq, 'surv_FN_fused': surv_FN_fused,
        'mismatch_F': n_mismatch_F, 'mismatch_FN': n_mismatch_FN,
        't_seq_F': t_seq_F, 't_seq_FN': t_seq_FN,
        't_seq_total': seq_total, 't_fused': t_fused,
        'speedup': speedup, 'sound': sound,
    }


def main():
    configs = [
        (4, 10, 1.28),  # d=8
        (5, 5, 1.28),   # d=10
    ]
    results = []
    for nh, m, c in configs:
        r = run_smoke(nh, m, c)
        results.append(r)

    out_path = os.path.join(_dir, '_smoke_fused_F_FN.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    geo_speedup = 1.0
    n_cfgs = 0
    for r in results:
        sound = "OK" if r['sound'] else "FAIL"
        print(f"  d={r['d']:2d} (n={r['n_half']},m={r['m']}): "
              f"seq={r['t_seq_total']:.2f}s, fused={r['t_fused']:.2f}s, "
              f"speedup={r['speedup']:.2f}x  [{sound}]")
        geo_speedup *= r['speedup']
        n_cfgs += 1
    geo_speedup = geo_speedup ** (1.0 / max(1, n_cfgs))
    print(f"\n  Geometric mean speedup: {geo_speedup:.2f}x")
    return geo_speedup


if __name__ == '__main__':
    main()

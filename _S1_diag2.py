"""Lighter diagnostic: just print the range sizes and the SDP results.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from cascade_opts import (_whole_parent_prune_theorem1,
                          lp_dual_certificate, sdp_certify_parent,
                          _theorem1_threshold_table)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges
from _S1_sdp_mosek import sdp_certify_parent_mosek, _build_window_quadratics


def diag_one(n_half, m, c, max_parents=10):
    d = 2 * n_half
    S_half = 2 * n_half * m

    surv = []
    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        s = _prune_dynamic_int32(batch, n_half, m, c, use_flat_threshold=False)
        if s.any():
            surv.append(batch[s].copy())
    if surv:
        surv = np.vstack(surv)
    else:
        surv = np.empty((0, d), dtype=np.int32)
    print(f"=== n_half={n_half} m={m} c={c}  (d={d})  L0 survivors: {len(surv)} ===")
    sys.stdout.flush()

    n_half_child = 2 * n_half
    sample = surv[:max_parents]

    n_sdp_reach = 0
    sdp_parents = []
    for idx, parent in enumerate(sample):
        d_child = 2 * d
        try:
            res = _compute_bin_ranges(parent, m, c, d_child, n_half_child)
        except Exception:
            continue
        if res is None:
            continue
        lo_arr, hi_arr, total_children = res
        if total_children == 0:
            continue
        if _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                        int(n_half_child), int(m), c):
            continue
        if d <= 10:
            try:
                if lp_dual_certificate(parent, lo_arr, hi_arr,
                                       int(n_half_child), int(m), c):
                    continue
            except Exception:
                pass
        n_sdp_reach += 1
        rsizes = [int(hi_arr[i]) - int(lo_arr[i]) + 1 for i in range(d)]
        n_enum = 1
        for r in rsizes:
            n_enum *= r
        # Run MOSEK
        t0 = time.time()
        ms_res, ms_status = sdp_certify_parent_mosek(parent, lo_arr, hi_arr,
                                                    int(n_half_child), int(m), c,
                                                    return_status=True)
        ms_t = (time.time() - t0) * 1000
        # max excess at midpoint
        win_quads = _build_window_quadratics(parent, lo_arr, hi_arr,
                                              int(n_half_child), int(m), c)
        max_ex_mid = -np.inf
        max_ex_box_min = -np.inf
        for wi, (Q, g, k, thr) in enumerate(win_quads):
            xc = (lo_arr.astype(float) + hi_arr.astype(float)) / 2.0
            ws_c = float(xc @ Q @ xc + g @ xc + k)
            ex = ws_c - thr
            if ex > max_ex_mid:
                max_ex_mid = ex
            # min over box (continuous, by sampling 4 corners on first 2 dims, etc.)
            # Just use random sample
            np.random.seed(7)
            best = -np.inf
            for _ in range(200):
                xs = np.random.uniform(lo_arr.astype(float),
                                       hi_arr.astype(float))
                ws = float(xs @ Q @ xs + g @ xs + k - thr)
                if ws > best:
                    best = ws
            if best > max_ex_box_min:
                max_ex_box_min = best
        sdp_parents.append({
            'idx': idx,
            'parent': parent.tolist(),
            'lo': lo_arr.tolist(),
            'hi': hi_arr.tolist(),
            'rsizes': rsizes,
            'n_enum': n_enum,
            'ms_res': ms_res,
            'ms_status': str(ms_status),
            'ms_time_ms': ms_t,
            'max_excess_at_midpoint': max_ex_mid,
            'max_random_excess': max_ex_box_min,
        })
        print(f"  [{idx}] parent={parent.tolist()} ranges={rsizes} "
              f"n_enum={n_enum:,} MOSEK={ms_res}({ms_status},{ms_t:.0f}ms) "
              f"midpoint_max_excess={max_ex_mid:.2f} random_max_excess={max_ex_box_min:.2f}")
        sys.stdout.flush()

    print(f"\nTotal: {n_sdp_reach} parents reached SDP stage out of {len(sample)}")
    return sdp_parents


if __name__ == '__main__':
    sdp_parents = diag_one(2, 30, 1.20, max_parents=20)

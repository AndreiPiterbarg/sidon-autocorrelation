"""Two more K-sweeps to corroborate the first.

SWEEP A: fill in the K=0→K=16 knee at K∈{4,8,12} on the SAME 50 boxes
         (seed=42). Tells us where the close-rate curve actually breaks.

SWEEP B: cross-validate K=16 and K=32 numbers on a DIFFERENT 50 boxes
         (seed=43). Confirms the 92%/100% close rates aren't sample-luck.

Together with the first sweep we have a robust K-vs-coverage curve.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.box import Box
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_int_ge_fast,
)


def run_sweep(boxes_lo, boxes_hi, indices, K_values, label, cache, windows):
    print(f"\n{'#'*72}\n# {label}: {len(indices)} boxes × K∈{K_values}\n{'#'*72}",
          flush=True)
    results = {K: [] for K in K_values}
    t0 = time.time()
    for idx, k in enumerate(indices):
        lo = boxes_lo[k].astype(np.float64); hi = boxes_hi[k].astype(np.float64)
        B = Box(lo=lo, hi=hi); lo_int, hi_int = B.to_ints()
        per_box = {}
        for K in K_values:
            ts = time.time()
            try:
                cert = bound_sdp_escalation_int_ge_fast(
                    lo_int, hi_int, windows, 10,
                    target_num=12, target_den=10,
                    cache=cache, n_window_psd_cones=K,
                    n_threads=1, time_limit_s=20.0,
                )
            except Exception:
                cert = False
            dt = time.time() - ts
            per_box[K] = (bool(cert), dt)
            results[K].append({'idx': int(k), 'cert': bool(cert), 'sec': dt})
        if (idx + 1) % 10 == 0 or idx == 0:
            line = f"  [{idx+1:3d}/{len(indices)}]"
            for K in K_values:
                c, t = per_box[K]
                line += f"  K={K:>3}:{'✓' if c else '✗'}({t:.1f}s)"
            print(line, flush=True)
    print(f"\n{'K':>5} {'cert':>10} {'%':>8} {'avg_s':>8} {'med_s':>8}", flush=True)
    print('-' * 50, flush=True)
    for K in K_values:
        rs = results[K]
        n_cert = sum(1 for r in rs if r['cert'])
        secs = [r['sec'] for r in rs]
        print(f"{K:>5} {n_cert:>5}/{len(rs):<4} {100*n_cert/len(rs):>7.1f}% "
              f"{np.mean(secs):>8.2f} {np.median(secs):>8.2f}", flush=True)
    print(f"  wall {time.time()-t0:.1f}s", flush=True)
    return results


def main():
    npz = np.load('stuck_d10_master_queue.npz')
    los, his = npz['lo'], npz['hi']
    N = len(los)

    windows = build_windows(10)
    cache = build_sdp_escalation_cache_fast(10, windows, target=1.2)
    print(f"|W|={len(windows)}", flush=True)

    # SWEEP A: same 50 boxes (seed=42), fill in K=4,8,12,24
    rng_a = np.random.default_rng(42)
    sample_a = rng_a.choice(N, 50, replace=False)
    sweep_a = run_sweep(los, his, sample_a, [4, 8, 12, 24],
                        "SWEEP A: knee resolution (seed=42)", cache, windows)

    # SWEEP B: DIFFERENT 50 boxes (seed=43), revalidate K=16, K=32
    rng_b = np.random.default_rng(43)
    sample_b = rng_b.choice(N, 50, replace=False)
    sweep_b = run_sweep(los, his, sample_b, [0, 16, 32],
                        "SWEEP B: cross-validation (seed=43)", cache, windows)

    # Combined K-vs-cert table for easy extrapolation
    print(f"\n{'#'*72}\n# COMBINED CURVE (d=10, |W|=190)\n{'#'*72}", flush=True)
    print(f"{'K':>5} {'cov%':>6} {'cert(A)':>10} {'cert(B)':>10}", flush=True)
    K_known = sorted(set([0, 4, 8, 12, 16, 24, 32]))
    for K in K_known:
        cov = 100 * K / 190
        if K in sweep_a:
            ca = sum(1 for r in sweep_a[K] if r['cert'])
            ca_s = f"{ca}/50 ({100*ca/50:.0f}%)"
        else:
            ca_s = "—"
        if K in sweep_b:
            cb = sum(1 for r in sweep_b[K] if r['cert'])
            cb_s = f"{cb}/50 ({100*cb/50:.0f}%)"
        else:
            cb_s = "—"
        print(f"{K:>5} {cov:>5.1f}% {ca_s:>10} {cb_s:>10}", flush=True)

    with open('K_sweep_d10_v2.json', 'w') as f:
        json.dump({
            'sweep_A_seed42': {str(K): sweep_a[K] for K in sweep_a},
            'sweep_B_seed43': {str(K): sweep_b[K] for K in sweep_b},
        }, f, indent=2)
    print("\nsaved K_sweep_d10_v2.json", flush=True)


if __name__ == "__main__":
    main()

"""Realistic d=22 K-sweep with proper time limits.

Lesson from sweep v1: SDP at d=22 K=32 takes ~70s factorisation alone.
30s time limit hit before any IPM iteration → uncertain verdict.
We give 300s per call this time and only sample 8 boxes (so total
~2h max).

If K=32 still gives mostly 'uncertain' even at 300s, the d=22 problem
is too big for per-box order-2 SDP at this K — we need either much
higher K (smaller per-window cost?) or completely different approach.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_int_ge_fast,
)


def main():
    src = 'runs_local/d22_t1p2805_split_K9/iter_006/children_after_lp.npz'
    z = np.load(src, allow_pickle=True)
    los_int = z['lo_int']; his_int = z['hi_int']
    lp_vals = z['lp_val']
    N = len(los_int)
    print(f"loaded {N} d=22 LP-residual boxes", flush=True)
    print(f"  lp_val range: {lp_vals.min():.4f} - {lp_vals.max():.4f}", flush=True)

    rng = np.random.default_rng(42)
    sample = rng.choice(N, 8, replace=False)
    print(f"sample = {[int(s) for s in sample]}", flush=True)

    target_num, target_den = 12805, 10000
    target_f = target_num / target_den
    d = 22

    windows = build_windows(d)
    nW = len(windows)
    print(f"|W|={nW}, target={target_f}", flush=True)

    print("building cache...", flush=True)
    cache = build_sdp_escalation_cache_fast(d, windows, target=target_f)

    K_values = [32, 80, 120]
    print(f"\nK values: {K_values}, time_limit=300s each", flush=True)
    print("=" * 80, flush=True)

    results = {K: [] for K in K_values}
    t_start = time.time()
    for idx, k in enumerate(sample):
        lo_int = [int(x) for x in los_int[k]]
        hi_int = [int(x) for x in his_int[k]]
        lp_v = float(lp_vals[k])
        print(f"\n[box {idx+1}/{len(sample)}, idx={int(k)}, lp_val={lp_v:.4f}]", flush=True)
        for K in K_values:
            print(f"  K={K} starting...", flush=True)
            t0 = time.time()
            try:
                cert = bound_sdp_escalation_int_ge_fast(
                    lo_int, hi_int, windows, d,
                    target_num=target_num, target_den=target_den,
                    cache=cache, n_window_psd_cones=K,
                    n_threads=1, time_limit_s=300.0,
                )
                dt = time.time() - t0
                tag = '✓ CERT' if cert else '✗ fail'
                print(f"    {tag} in {dt:.0f}s", flush=True)
                results[K].append({'idx': int(k), 'cert': bool(cert), 'sec': dt})
            except Exception as e:
                dt = time.time() - t0
                print(f"    EXC after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)
                results[K].append({'idx': int(k), 'cert': False, 'sec': dt})
        elapsed_total = time.time() - t_start
        print(f"  [running total wall: {elapsed_total/60:.1f} min]", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print(f"{'K':>5} {'cov%':>6} {'cert':>6} {'%':>6} {'avg_s':>8} {'med_s':>8} {'max_s':>8}",
          flush=True)
    print("=" * 80, flush=True)
    for K in K_values:
        rs = results[K]
        n_cert = sum(1 for r in rs if r['cert'])
        secs = [r['sec'] for r in rs]
        cov = 100 * K / nW
        print(f"{K:>5} {cov:>5.1f}% {n_cert:>2}/{len(rs):<2} "
              f"{100*n_cert/len(rs):>5.0f}% "
              f"{np.mean(secs):>8.0f} {np.median(secs):>8.0f} {np.max(secs):>8.0f}",
              flush=True)

    print(f"\ntotal wall: {(time.time()-t_start)/60:.1f} min", flush=True)
    with open('K_sweep_d22_v2.json', 'w') as f:
        json.dump({str(K): results[K] for K in K_values}, f, indent=2)
    print("saved K_sweep_d22_v2.json", flush=True)


if __name__ == "__main__":
    main()

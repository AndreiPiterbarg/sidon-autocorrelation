"""K-sweep on REAL d=22 stuck boxes from the d=22 12h orchestrator run.

Source: runs_local/d22_t1p2805_split_K9/iter_006/children_after_lp.npz
        (416 boxes that failed the float epigraph LP at iter 6;
        these are exactly the boxes the orchestrator's K=0/16/32 SDP
        sweep was applied to.)

Validates the d=10 extrapolation (8% window-coverage = 88% close,
13% coverage = 100% close) at d=22 / |W|=946.

Target: 1.2805 (the orchestrator's target).
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
    los_int = z['lo_int']    # (N, 22) Python ints at scale 2^60
    his_int = z['hi_int']
    depths = z['depth']
    lp_vals = z['lp_val']
    N = len(los_int)
    print(f"loaded {N} d=22 LP-residual boxes from {src}", flush=True)
    print(f"  depth: {depths.min()}–{depths.max()}", flush=True)
    print(f"  lp_val: {lp_vals.min():.4f}–{lp_vals.max():.4f}", flush=True)

    rng = np.random.default_rng(42)
    sample = rng.choice(N, 25, replace=False)

    target_num, target_den = 12805, 10000  # = 1.2805
    target_f = target_num / target_den
    d = 22

    windows = build_windows(d)
    nW = len(windows)
    print(f"|W_22| = {nW}", flush=True)

    print("building SDP cache (~2-5 s)...", flush=True)
    t_cache = time.time()
    cache = build_sdp_escalation_cache_fast(d, windows, target=target_f)
    print(f"  cache built in {time.time()-t_cache:.1f}s", flush=True)

    # K values: extrapolated from d=10 curve to bracket 88%, 94%, 100%
    K_values = [32, 80, 120, 160, 240]

    print(f"\nK-sweep on {len(sample)} d=22 boxes, K∈{K_values}", flush=True)
    print(f"target={target_f}", flush=True)
    print("=" * 72, flush=True)

    results = {K: [] for K in K_values}
    t0 = time.time()
    for idx, k in enumerate(sample):
        # lo/hi int are already at scale 2^60 — pass directly
        lo_int = [int(x) for x in los_int[k]]
        hi_int = [int(x) for x in his_int[k]]
        per_box = {}
        for K in K_values:
            ts = time.time()
            try:
                cert = bound_sdp_escalation_int_ge_fast(
                    lo_int, hi_int, windows, d,
                    target_num=target_num, target_den=target_den,
                    cache=cache, n_window_psd_cones=K,
                    n_threads=1, time_limit_s=120.0,
                )
            except Exception as e:
                cert = False
            dt = time.time() - ts
            per_box[K] = (bool(cert), dt)
            results[K].append({'idx': int(k), 'cert': bool(cert), 'sec': dt})
        line = f"[{idx+1:3d}/{len(sample)}] depth={int(depths[k]):3d} lp={lp_vals[k]:.4f}"
        for K in K_values:
            c, t = per_box[K]
            line += f"  K={K:>3}:{'✓' if c else '✗'}({t:.0f}s)"
        print(line, flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"{'K':>5} {'cov%':>6} {'cert':>10} {'%':>8} {'avg_s':>8} {'med_s':>8} {'max_s':>8}",
          flush=True)
    print("=" * 72, flush=True)
    for K in K_values:
        rs = results[K]
        n_cert = sum(1 for r in rs if r['cert'])
        secs = [r['sec'] for r in rs]
        cov = 100 * K / nW
        print(f"{K:>5} {cov:>5.1f}% {n_cert:>5}/{len(rs):<4} "
              f"{100*n_cert/len(rs):>7.1f}% "
              f"{np.mean(secs):>8.1f} {np.median(secs):>8.1f} {np.max(secs):>8.1f}",
              flush=True)

    # Cumulative cert (boxes closing at K_low or higher)
    cert_at = {K: set(i for i, r in enumerate(results[K]) if r['cert']) for K in K_values}
    print("\nIncremental close (K_higher \\ K_lower):", flush=True)
    for i, K in enumerate(K_values[1:]):
        Klow = K_values[i]
        diff = cert_at[K] - cert_at[Klow]
        print(f"  K={K} \\ K={Klow}: +{len(diff)} new certs "
              f"({100*len(diff)/len(sample):.1f}%)", flush=True)

    print(f"\nwall: {time.time()-t0:.1f}s", flush=True)

    with open('K_sweep_d22.json', 'w') as f:
        json.dump({str(K): results[K] for K in K_values}, f, indent=2)
    print("saved K_sweep_d22.json", flush=True)


if __name__ == "__main__":
    main()

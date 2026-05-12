"""K-sweep on d=10 stuck boxes: K=0, 16, 32, full at target=1.2.

For each box, runs each K as an INDEPENDENT attempt (not iterative on
each other's failures). This gives the true per-K close rate to
compare against the d=22 orchestrator's iterative numbers.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.box import Box
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_int_ge_fast,
)


def main():
    npz = np.load('stuck_d10_master_queue.npz')
    los, his = npz['lo'], npz['hi']
    N = len(los)
    rng = np.random.default_rng(42)
    sample = rng.choice(N, 50, replace=False)
    print(f"sampling {len(sample)} of {N} stuck d=10 boxes", flush=True)

    windows = build_windows(10)
    cache = build_sdp_escalation_cache_fast(10, windows, target=1.2)
    print(f"|W|={len(windows)} cache built", flush=True)

    K_values = [0, 16, 32, 999]  # 999 ≈ full PSD
    results = {K: [] for K in K_values}

    t0 = time.time()
    for idx, k in enumerate(sample):
        lo = los[k].astype(np.float64); hi = his[k].astype(np.float64)
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
            except Exception as e:
                cert = False
            dt = time.time() - ts
            per_box[K] = (bool(cert), dt)
            results[K].append({'idx': int(k), 'cert': bool(cert), 'sec': dt})
        if (idx + 1) % 5 == 0 or idx == 0:
            line = f"[{idx+1:3d}/{len(sample)}]"
            for K in K_values:
                c, t = per_box[K]
                flag = '✓' if c else '✗'
                line += f"  K={K:>3}:{flag}({t:.1f}s)"
            print(line, flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"{'K':>5} {'cert':>10} {'%':>8} {'avg_s':>8} {'med_s':>8} {'max_s':>8}",
          flush=True)
    print("=" * 78, flush=True)
    for K in K_values:
        rs = results[K]
        n_cert = sum(1 for r in rs if r['cert'])
        secs = [r['sec'] for r in rs]
        print(f"{K:>5} {n_cert:>5}/{len(rs):<4} {100*n_cert/len(rs):>7.1f}% "
              f"{np.mean(secs):>8.2f} {np.median(secs):>8.2f} {np.max(secs):>8.2f}",
              flush=True)
    print("=" * 78, flush=True)

    # Inclusion analysis: how many boxes does K=16 close that K=0 doesn't, etc.
    n = len(results[0])
    cert_at_K = {K: set(i for i, r in enumerate(results[K]) if r['cert']) for K in K_values}
    print("\nIncremental close (K_higher closes that K_lower didn't):", flush=True)
    print(f"  K=16 \\ K=0:    {len(cert_at_K[16] - cert_at_K[0])} boxes "
          f"({100*len(cert_at_K[16] - cert_at_K[0])/n:.1f}%)", flush=True)
    print(f"  K=32 \\ K=16:   {len(cert_at_K[32] - cert_at_K[16])} boxes "
          f"({100*len(cert_at_K[32] - cert_at_K[16])/n:.1f}%)", flush=True)
    print(f"  K=999 \\ K=32:  {len(cert_at_K[999] - cert_at_K[32])} boxes "
          f"({100*len(cert_at_K[999] - cert_at_K[32])/n:.1f}%)", flush=True)
    print(f"  K=999 \\ K=0:   {len(cert_at_K[999] - cert_at_K[0])} boxes "
          f"({100*len(cert_at_K[999] - cert_at_K[0])/n:.1f}%)", flush=True)

    print(f"\ntotal wall: {time.time() - t0:.1f}s", flush=True)
    with open('K_sweep_d10.json', 'w') as f:
        json.dump({str(K): results[K] for K in K_values}, f, indent=2)


if __name__ == "__main__":
    main()

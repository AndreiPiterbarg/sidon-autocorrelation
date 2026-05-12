"""Run order-2 SDP on a sample of stuck d=10 boxes to see if it closes.

If the SDP returns 'infeas' (cert) on most stuck boxes → low-K SDP is
the missing tier. If it returns 'feas' or 'uncertain' → even order-2
SDP is too loose and we need order 3 or a different attack.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_lb_float_fast,
)


def main():
    target = 1.2
    d = 10
    npz = np.load('stuck_d10_master_queue.npz')
    los = npz['lo']
    his = npz['hi']
    deps = npz['depths']
    N = len(los)

    # Sample 15 stuck boxes spanning depth range
    order = np.argsort(deps)
    pick = order[np.linspace(0, N - 1, 15).astype(int)]

    windows = build_windows(d)

    print("Building SDP cache (~1-2s)...", flush=True)
    cache = build_sdp_escalation_cache_fast(d, windows, target=target)

    print(f"\n{'idx':>5} {'depth':>5} {'width':>10} {'bdry':>4} "
          f"{'K=0':>10} {'K=16':>10} {'K=32':>10} {'K=full':>10}  cert?")
    print("=" * 90)

    for k in pick:
        lo = los[k].astype(np.float64)
        hi = his[k].astype(np.float64)
        depth = int(deps[k])
        wmax = float(np.max(hi - lo))
        nb = int((lo <= 1e-12).sum())

        results = {}
        for K in [0, 16, 32, 999]:
            try:
                t0 = time.time()
                info = bound_sdp_escalation_lb_float_fast(
                    lo, hi, windows, d, target=target, cache=cache,
                    n_window_psd_cones=K,
                    n_threads=1, time_limit_s=30.0,
                    return_diagnostic=True,
                )
                elapsed = time.time() - t0
                if isinstance(info, tuple):
                    info = info[1]
                lb = info.get('lb_value', float('-inf'))
                verdict = info.get('verdict', '?')
                results[K] = (lb, verdict, elapsed)
            except Exception as e:
                results[K] = (float('-inf'), f'EXC:{type(e).__name__}', 0.0)

        cert_any = any(v[1] == 'infeas' for v in results.values())
        marker = " YES" if cert_any else " no "
        print(f"{k:5d} {depth:5d} {wmax:10.2e} {nb:4d} "
              f"{results[0][0]:10.4f} {results[16][0]:10.4f} "
              f"{results[32][0]:10.4f} {results[999][0]:10.4f} "
              f" {marker}")
        verdicts = ' '.join(f'{K}={results[K][1]}' for K in [0,16,32,999])
        print(f"      {verdicts}")


if __name__ == "__main__":
    main()

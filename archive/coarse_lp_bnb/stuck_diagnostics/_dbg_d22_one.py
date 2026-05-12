"""Single-box d=22 SDP test with verbose tracing."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_int_ge_fast,
)

print("loading...", flush=True)
z = np.load('runs_local/d22_t1p2805_split_K9/iter_006/children_after_lp.npz', allow_pickle=True)
los_int = z['lo_int']; his_int = z['hi_int']
print(f"loaded {len(los_int)} boxes", flush=True)

print("building windows + cache...", flush=True)
windows = build_windows(22)
cache = build_sdp_escalation_cache_fast(22, windows, target=1.2805)
print(f"cache ok, |W|={len(windows)}", flush=True)

for K in [32, 80]:
    k = 0
    lo_int = [int(x) for x in los_int[k]]
    hi_int = [int(x) for x in his_int[k]]
    print(f"\n--- box {k}, K={K}, time_limit_s=60 ---", flush=True)
    t0 = time.time()
    try:
        cert = bound_sdp_escalation_int_ge_fast(
            lo_int, hi_int, windows, 22,
            target_num=12805, target_den=10000,
            cache=cache, n_window_psd_cones=K,
            n_threads=1, time_limit_s=60.0,
        )
        print(f"  DONE in {time.time()-t0:.1f}s, cert={cert}", flush=True)
    except Exception as e:
        print(f"  EXCEPTION after {time.time()-t0:.1f}s: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()

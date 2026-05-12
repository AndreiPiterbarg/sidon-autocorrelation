"""Box 1 (LP=1.257) at K=0 only — focused test."""
import sys, glob, time
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast as build_cache,
    bound_sdp_escalation_lb_float_fast,
)

d = 22
target = 1.281
windows = build_windows(d)

# Reload Box 1 from the same source: stuck_boxes_w1.npz idx 3.
data = np.load('stuck_boxes_w1.npz', allow_pickle=True)
lo_arr, hi_arr = data['lo'], data['hi']
# Use the same filtering — find LP=1.257 box.
candidates = []
for i in range(lo_arr.shape[0]):
    lo = lo_arr[i].astype(np.float64)
    hi = hi_arr[i].astype(np.float64)
    if lo.shape[0] != d:
        continue
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        continue
    lp = bound_epigraph_lp_float(lo, hi, windows, d)
    if target - 0.05 <= lp < target:
        candidates.append((lp, lo, hi, i))
candidates.sort(key=lambda x: x[0], reverse=True)
# Box 1 = index 1 in sorted list (Box 0 was the highest LP)
if len(candidates) < 2:
    print(f"only {len(candidates)} stuck candidates", flush=True)
    sys.exit(1)
lp, lo, hi, idx = candidates[1]
hw = float((hi - lo).max() / 2)
print(f"BOX 1: LP={lp:.6f} hw={hw:.4f} idx={idx}", flush=True)
print(f"  lo[:5]={lo[:5]}", flush=True)
print(f"  hi[:5]={hi[:5]}", flush=True)

print(f"\nBuilding cache (target={target})...", flush=True)
t0 = time.time()
cache = build_cache(d, windows, target=target)
print(f"  cache build: {time.time()-t0:.2f}s", flush=True)

print(f"\nRunning K=0 with 48 threads, time_limit=600s...", flush=True)
t0 = time.time()
res = bound_sdp_escalation_lb_float_fast(
    lo, hi, windows, d, cache=cache, target=target,
    n_window_psd_cones=0, time_limit_s=600.0, n_threads=48,
)
dt = time.time() - t0
print(f"\n=== RESULT: K=0  t={dt:.1f}s  verdict={res.get('verdict')}  "
      f"lambda*={res.get('lambda_star'):.4f}  status={res.get('solsta')} ===",
      flush=True)

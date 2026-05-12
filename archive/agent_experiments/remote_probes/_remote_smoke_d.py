"""Quick scaling smoke: d in {12, 14}. Box around peaked centroid."""
import sys, time, os
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)

target = float(os.environ.get('SD_TARGET', '1.281'))
ds = [int(x) for x in os.environ.get('SD_DS', '12,14').split(',')]
time_lim = float(os.environ.get('SD_TIME', '120'))

for d in ds:
    print(f"=== d={d} target={target} ===", flush=True)
    windows = build_windows(d)
    mu = np.full(d, 1.0/d); mu[0]*=0.5; mu[-1]*=1.5; mu/=mu.sum()
    if mu[0] > mu[-1]: mu = mu[::-1]
    radius = 5e-3
    lo = np.maximum(mu-radius, 0.0); hi = np.minimum(mu+radius, 1.0)
    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows, target=target)
    info = cache['info']
    print(f"  cache: {time.time()-t0:.2f}s n_y={info['n_y']} "
          f"n_bar={info['n_bar']} n_bar_entries={info['n_bar_entries']}",
          flush=True)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                          target=target, time_limit_s=time_lim)
    dt = time.time() - t0
    print(f"  solve: {dt:.2f}s verdict={res.get('verdict')} "
          f"lambda={res.get('lambda_star'):.4f} status={res.get('solsta')}",
          flush=True)

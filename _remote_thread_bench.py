"""Benchmark MOSEK thread scaling on the dual-Farkas SDP."""
import sys, time, os
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)

d = int(os.environ.get('TB_D', '14'))
target = float(os.environ.get('TB_T', '1.281'))
time_lim = float(os.environ.get('TB_TIME', '300'))
threads_list = [int(x) for x in os.environ.get('TB_THREADS', '1,4,8,16,32,48').split(',')]

print(f"d={d} target={target} thread bench")
windows = build_windows(d)
mu = np.full(d, 1.0/d); mu[0]*=0.5; mu[-1]*=1.5; mu/=mu.sum()
if mu[0] > mu[-1]: mu = mu[::-1]
radius = 5e-3
lo = np.maximum(mu-radius, 0.0); hi = np.minimum(mu+radius, 1.0)

for n_thr in threads_list:
    cache = build_sdp_escalation_cache(d, windows, target=target)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                          target=target,
                                          time_limit_s=time_lim,
                                          n_threads=n_thr)
    dt = time.time() - t0
    v = res.get('verdict')
    lam = res.get('lambda_star', float('nan'))
    print(f"  n_thr={n_thr:>3}  solve={dt:6.2f}s  verdict={v}  lambda={lam:.4f}",
          flush=True)

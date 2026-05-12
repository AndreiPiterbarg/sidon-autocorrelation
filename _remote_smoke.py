"""Remote smoke + perf test for the MOSEK SDP escalation."""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)


def smoke_d(d, hw):
    t0 = time.time()
    windows = build_windows(d)
    cache = build_sdp_escalation_cache(d, windows)
    t_cache = time.time() - t0
    if d == 4:
        mu = np.array([1/3, 1/6, 1/6, 1/3])
    elif d == 6:
        mu = np.array([0.18, 0.10, 0.08, 0.08, 0.10, 0.46])
        mu = mu / mu.sum()
    else:
        rng = np.random.default_rng(0)
        mu = rng.dirichlet(np.ones(d))
        if mu[0] > mu[-1]:
            mu = mu[::-1]
    lo = np.maximum(mu - hw, 0.0)
    hi = np.minimum(mu + hw, 1.0)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        lo = np.maximum(mu - 0.04, 0.0)
        hi = np.minimum(mu + 0.04, 1.0)
    # First solve (cold)
    t1 = time.time()
    res = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                         time_limit_s=60.0)
    t_solve_1 = time.time() - t1
    # Second solve (warm Fusion matrices, fresh model)
    t2 = time.time()
    res2 = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                          time_limit_s=60.0)
    t_solve_2 = time.time() - t2
    msg = res.get('error_msg', '')
    n_cliques = len(cache.get('mom_blocks', []))
    n_loc = len(cache.get('loc_blocks', []))
    if n_cliques > 0:
        max_cb = max(b.n_cb for b in cache['mom_blocks'])
    else:
        max_cb = 0
    if n_loc > 0:
        max_cb_loc = max(b.n_cb_loc for b in cache['loc_blocks'])
    else:
        max_cb_loc = 0
    print(f"d={d:>3} n_y={cache['n_y']:>6} bw={cache['bandwidth']:>3} "
          f"cliques={n_cliques:>2} max_B={max_cb:>4} max_B1={max_cb_loc:>3} "
          f"n_W={cache['n_W_kept']:>5} | "
          f"cache={t_cache:5.2f}s solve_cold={t_solve_1:5.2f}s "
          f"solve_warm={t_solve_2:5.2f}s | "
          f"status={res['status']:<20} "
          f"obj_dual={res['obj_val_dual']:.6f} "
          f"r_p={res.get('r_prim', 'na'):.2e} r_d={res.get('r_dual', 'na'):.2e} "
          f"{msg}")


if __name__ == '__main__':
    import os
    sys.stdout.reconfigure(line_buffering=True)
    targets = ((4, 5e-3), (6, 5e-3), (10, 0.02), (16, 0.025),
               (22, 0.025), (30, 0.025))
    only = os.environ.get('SMOKE_ONLY')
    if only:
        targets = [(int(only), 0.025)]
    for d, hw in targets:
        try:
            smoke_d(d, hw)
        except Exception as e:
            print(f"d={d} ERROR: {type(e).__name__}: {e}", flush=True)

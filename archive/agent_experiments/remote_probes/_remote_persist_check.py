"""Validate the persistent-model approach: build once, solve many.

Confirms:
1. First solve includes model build cost; subsequent solves are pure solve.
2. Multiple boxes give consistent results (status, residuals).
3. Persistent model doesn't accumulate state between calls.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, _solve_box_sdp_feasibility_at_t,
)


def main(d, hw, target, n_boxes, use_persistent=True):
    label = "persistent" if use_persistent else "rebuild"
    print(f"=== d={d} hw={hw} target={target} mode={label} ===", flush=True)
    windows = build_windows(d)
    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows)
    print(f"  cache build: {time.time()-t0:.2f}s, n_y={cache['n_y']}, "
          f"bw={cache['bandwidth']}, cliques={len(cache['mom_blocks'])}",
          flush=True)
    rng = np.random.default_rng(0)
    for trial in range(n_boxes):
        mu = rng.dirichlet(np.ones(d))
        if mu[0] > mu[-1]:
            mu = mu[::-1]
        lo = np.maximum(mu - hw, 0.0)
        hi = np.minimum(mu + hw, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        t1 = time.time()
        res = _solve_box_sdp_feasibility_at_t(
            lo, hi, windows, d, t_test=target,
            cache=cache, time_limit_s=60.0,
            use_persistent=use_persistent,
        )
        dt = time.time() - t1
        msg = res.get('error_msg', '')
        print(f"  trial {trial}: t={dt:6.2f}s status={res['status']:<35} "
              f"infeas={res.get('is_infeasible')} feas={res.get('is_feasible')} "
              f"r_p={res.get('r_prim','na'):.2e} {msg}",
              flush=True)


if __name__ == '__main__':
    import os
    d = int(os.environ.get('CHK_D', '16'))
    hw = float(os.environ.get('CHK_HW', '0.025'))
    target = float(os.environ.get('CHK_T', '1.281'))
    n = int(os.environ.get('CHK_N', '5'))
    mode = os.environ.get('CHK_MODE', 'persistent')
    main(d, hw, target, n, use_persistent=(mode == 'persistent'))

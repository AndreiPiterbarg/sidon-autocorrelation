"""Sweep bandwidth at d=22 for cert tightness vs solve speed.

For ONE LP-failing residual box at d=22 hw=0.025 around mu_star_d22 (or
random Dirichlet), measure the SDP value at progressively larger
bandwidths. The smallest bw at which SDP_safe >= target=1.281 is the
required bandwidth.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float, _safe_cushion,
)


def main(d, hw, target):
    print(f"=== d={d} hw={hw} target={target} bandwidth sweep ===", flush=True)
    windows = build_windows(d)

    # Find a stuck box: LP_val close to but below target.
    rng = np.random.default_rng(0)
    box_lo = box_hi = None
    lp_val = None
    for trial in range(50):
        mu = rng.dirichlet(np.ones(d))
        if mu[0] > mu[-1]:
            mu = mu[::-1]
        lo = np.maximum(mu - hw, 0.0)
        hi = np.minimum(mu + hw, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        v = bound_epigraph_lp_float(lo, hi, windows, d)
        if target - 0.05 <= v < target:
            box_lo = lo
            box_hi = hi
            lp_val = v
            print(f"  picked trial {trial}: LP={v:.6f}", flush=True)
            break
    if box_lo is None:
        print(f"  no stuck box found")
        return

    bandwidths = [3, 5, 7, 9, 11, 13, 15, 19, 21]
    for bw in bandwidths:
        if bw >= d:
            continue
        t0 = time.time()
        try:
            cache = build_sdp_escalation_cache(d, windows, bandwidth=bw)
            t_cache = time.time() - t0
            t1 = time.time()
            res = bound_sdp_escalation_lb_float(
                box_lo, box_hi, windows, d, cache=cache, time_limit_s=180.0)
            dt = time.time() - t1
            if res['is_feasible_status']:
                cushion = _safe_cushion(res['r_prim'], res['r_dual'],
                                         res['duality_gap'])
                lb_safe = float(res['obj_val_dual']) - cushion
                cert = lb_safe >= target
                n_cliques = len(cache['mom_blocks'])
                max_B = max((b.n_cb for b in cache['mom_blocks']), default=0)
                print(f"  bw={bw:>2} cliques={n_cliques:>2} max_B={max_B:>4} | "
                      f"cache={t_cache:5.2f}s solve={dt:6.2f}s | "
                      f"SDP={res['obj_val_dual']:.6f} safe={lb_safe:.6f} "
                      f"cert={cert} status={res['status']}",
                      flush=True)
            else:
                print(f"  bw={bw:>2} FAILED status={res['status']} "
                      f"t={dt:.1f}s", flush=True)
        except Exception as e:
            print(f"  bw={bw:>2} ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == '__main__':
    import os
    d = int(os.environ.get('SWEEP_D', '22'))
    main(d, 0.025, 1.281)

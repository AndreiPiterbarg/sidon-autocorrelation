"""Test the Lasserre SDP cert rate at the d=30 transition zone.

For random H_d boxes at d=30, hw=0.025:
- Compute the epigraph LP value (the cheap tier).
- Skip if LP value already certifies (LP_val >= target).
- Skip if LP value is far from target (LP_val < target - 0.05) — SDP unlikely to help.
- For boxes with target - 0.05 <= LP_val < target, run the SDP escalation.
- Report cert rate and timing.

This mimics the real BnB cascade: epigraph LP first, SDP only on residual
"close-but-not-cert" boxes. The publish gate is cert rate >= 70%.
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


def main(d, hw, target, n_max=20):
    print(f"=== d={d} hw={hw} target={target} ===", flush=True)
    windows = build_windows(d)
    print(f"  building cache...", flush=True)
    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows)
    info = cache.get('info', {})
    print(f"  cache build: {time.time()-t0:.2f}s, n_y={info.get('n_y','?')}, "
          f"target={cache.get('target','?')}, "
          f"bar_sizes={info.get('bar_sizes','?')[:3] if info.get('bar_sizes') else '?'}+...",
          flush=True)
    rng = np.random.default_rng(0)
    n_lp_cert = 0
    n_lp_far = 0
    n_sdp_attempts = 0
    n_sdp_certs = 0
    sdp_times = []
    sdp_lbs = []
    for trial in range(n_max * 3):
        if n_sdp_attempts >= n_max:
            break
        mu = rng.dirichlet(np.ones(d))
        if mu[0] > mu[-1]:
            mu = mu[::-1]
        lo = np.maximum(mu - hw, 0.0)
        hi = np.minimum(mu + hw, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        # Cheap LP cert filter
        lp_lb = bound_epigraph_lp_float(lo, hi, windows, d)
        if lp_lb >= target:
            n_lp_cert += 1
            continue
        if lp_lb < target - 0.05:
            n_lp_far += 1
            continue
        # SDP escalation
        n_sdp_attempts += 1
        t1 = time.time()
        res = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                             time_limit_s=60.0)
        dt = time.time() - t1
        sdp_times.append(dt)
        if res['is_feasible_status']:
            cushion = _safe_cushion(res['r_prim'], res['r_dual'],
                                     res['duality_gap'])
            lb_safe = float(res['obj_val_dual']) - cushion
            sdp_lbs.append(lb_safe)
            cert = lb_safe >= target
            if cert:
                n_sdp_certs += 1
            print(f"  trial {trial:>3}: LP={lp_lb:.4f} SDP_dual={res['obj_val_dual']:.4f} "
                  f"safe={lb_safe:.4f} cert={cert} t={dt:.1f}s status={res['status']}",
                  flush=True)
        else:
            print(f"  trial {trial:>3}: LP={lp_lb:.4f} SDP FAILED status={res['status']} "
                  f"t={dt:.1f}s {res.get('error_msg', '')}", flush=True)
    print(f"\n--- SUMMARY ---")
    print(f"  LP-already-cert: {n_lp_cert}")
    print(f"  LP-too-far     : {n_lp_far}")
    print(f"  SDP attempts   : {n_sdp_attempts}")
    print(f"  SDP certs      : {n_sdp_certs} ({100*n_sdp_certs/max(1,n_sdp_attempts):.0f}%)")
    if sdp_times:
        print(f"  SDP timing     : min={min(sdp_times):.1f}s med={np.median(sdp_times):.1f}s max={max(sdp_times):.1f}s")


if __name__ == '__main__':
    import os
    d = int(os.environ.get('CERT_D', '30'))
    hw = float(os.environ.get('CERT_HW', '0.025'))
    target = float(os.environ.get('CERT_T', '1.281'))
    n = int(os.environ.get('CERT_N', '10'))
    main(d, hw, target, n)

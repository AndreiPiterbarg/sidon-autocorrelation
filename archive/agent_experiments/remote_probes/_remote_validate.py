"""Validate the dual-Farkas SDP on KNOWN cert-able boxes.

For each d, build a tight box around the per-d optimum mu_star (loaded
from mu_star_d{d}.npz if present, else from a uniform centroid for
small d). Then test SDP cert at target = val(d) - margin. The SDP
should return INFEASIBLE (cert), proving the dual-Farkas code works.

Also reports build_cache + first solve + warm-update solve times so
we can see the per-call cost at each d.
"""
import sys
import time
import numpy as np
import os
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)


def get_mu_star(d):
    """Return mu_star for d if cached, else a uniform fallback."""
    npz_path = f'mu_star_d{d}.npz'
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        if 'mu' in data.files:
            mu = np.asarray(data['mu'], dtype=np.float64)
            if mu[0] > mu[-1]:
                mu = mu[::-1]
            return mu, data.get('f', None)
    # Heuristic: peaked-on-extremes centroid (good for our problem).
    mu = np.full(d, 1.0 / d)
    mu[0] *= 0.5; mu[-1] *= 1.5
    mu /= mu.sum()
    return mu, None


def main(d, target, radius, time_limit=120.0):
    print(f"\n=== d={d} target={target} radius={radius} ===", flush=True)
    windows = build_windows(d)
    mu_star, f_star = get_mu_star(d)
    print(f"  mu_star: {mu_star[:5]}... f_star={f_star}", flush=True)

    lo = np.maximum(mu_star - radius, 0.0)
    hi = np.minimum(mu_star + radius, 1.0)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        radius2 = radius * 0.5
        lo = np.maximum(mu_star - radius2, 0.0)
        hi = np.minimum(mu_star + radius2, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            print(f"  box still infeasible — skipping", flush=True)
            return

    lp = bound_epigraph_lp_float(lo, hi, windows, d)
    print(f"  LP value: {lp:.6f}  (target {target})  "
          f"LP-cert? {lp >= target}", flush=True)

    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows, target=target)
    t_cache = time.time() - t0
    info = cache['info']
    print(f"  cache build: {t_cache:.2f}s  n_y={info.get('n_y')}  "
          f"n_bar={info.get('n_bar')}  n_bar_entries={info.get('n_bar_entries')}",
          flush=True)

    # First solve (cold).
    t0 = time.time()
    res1 = bound_sdp_escalation_lb_float(lo, hi, windows, d, cache=cache,
                                          target=target,
                                          time_limit_s=time_limit)
    t_solve1 = time.time() - t0
    print(f"  cold solve : {t_solve1:.2f}s  verdict={res1.get('verdict')}  "
          f"lambda={res1.get('lambda_star'):.4f}  "
          f"status={res1.get('solsta')}", flush=True)

    # Warm solve at slightly different box (shrink hw by 10%).
    radius2 = radius * 0.9
    lo2 = np.maximum(mu_star - radius2, 0.0)
    hi2 = np.minimum(mu_star + radius2, 1.0)
    t0 = time.time()
    res2 = bound_sdp_escalation_lb_float(lo2, hi2, windows, d, cache=cache,
                                          target=target,
                                          time_limit_s=time_limit)
    t_solve2 = time.time() - t0
    print(f"  warm solve : {t_solve2:.2f}s  verdict={res2.get('verdict')}  "
          f"lambda={res2.get('lambda_star'):.4f}", flush=True)


if __name__ == '__main__':
    cfgs = [
        # (d, target, radius)
        (10, 1.20, 5e-3),   # val(10)≈1.24, easy to cert
        (16, 1.30, 5e-3),   # val(16)≈ near 1.31?
        (22, 1.281, 3e-3),  # the actual benchmark box
    ]
    only_d = os.environ.get('VALID_D')
    if only_d:
        cfgs = [c for c in cfgs if c[0] == int(only_d)]
    time_lim = float(os.environ.get('VALID_TIME', '120'))
    for d, target, radius in cfgs:
        try:
            main(d, target, radius, time_limit=time_lim)
        except Exception as e:
            print(f"d={d} ERROR: {type(e).__name__}: {e}", flush=True)

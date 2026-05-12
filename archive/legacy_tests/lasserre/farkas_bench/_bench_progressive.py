"""Progressive benchmark of slow vs fast-bignum Farkas bisection.

Prints a line per configuration as soon as it finishes.  Runs in
ascending (d, order) so the user can see progress and interrupt
at any time without losing data.
"""
from __future__ import annotations
import os, sys, time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from certified_lasserre.farkas_certify import farkas_certify_bisect

CONFIGS = [
    ("SLOW",       dict(use_fast_residual=False)),
    ("FAST_BIGNUM", dict(use_fast_residual=True, fast_D_L=10**9, fast_use_bignum=True)),
]

CASES = [
    # (d, order, t_hi_initial_bracket, max_bisect)
    (4, 2, 1.12, 8),
    (4, 3, 1.12, 8),
    (6, 2, 1.15, 8),
    (6, 3, 1.22, 8),
    (8, 2, 1.22, 6),
    (8, 3, 1.25, 6),
]


def _one(d, order, t_hi, max_b, name, kw):
    t0 = time.time()
    try:
        res = farkas_certify_bisect(
            d=d, order=order, t_lo=1.0, t_hi=t_hi, tol=1e-4, max_bisect=max_b,
            max_denom_S=10**9, max_denom_mu=10**10, eig_margin=1e-9,
            verbose=False, **kw,
        )
        wall = time.time() - t0
        print(f"[RESULT] d={d} k={order} {name:>12}: "
              f"t_cert={float(res.lb_rig):.6f}  wall={wall:.2f}s",
              flush=True)
    except Exception as e:
        wall = time.time() - t0
        print(f"[RESULT] d={d} k={order} {name:>12}: FAIL after {wall:.2f}s: "
              f"{type(e).__name__}: {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    print(f"[START] {len(CASES)} cases × {len(CONFIGS)} configs", flush=True)
    for (d, order, t_hi, max_b) in CASES:
        print(f"[CASE] d={d} k={order}", flush=True)
        for name, kw in CONFIGS:
            _one(d, order, t_hi, max_b, name, kw)
    print("[DONE]", flush=True)

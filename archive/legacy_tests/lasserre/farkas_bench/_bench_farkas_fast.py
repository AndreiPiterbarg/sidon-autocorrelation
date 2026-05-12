"""Benchmark: slow vs fast Farkas residual at a certifiable t_test.

For each (d, order), runs farkas_certify_at twice (slow and fast path),
reports solver time, rounding+residual time, total, and whether the
infeasibility verdict matches.  The residual arithmetic is structurally
different (limit_denominator vs fixed-denom rounding) so the safety
margin will differ numerically but the CERTIFIED/NOT flag should agree
on probes that are not right on the boundary.
"""
from __future__ import annotations
import os, sys, time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from certified_lasserre.farkas_certify import farkas_certify_at


CASES = [
    # (d, order, t_test_below_val_for_infeasibility)
    (4, 3, 1.095),   # val(4,3) ~ 1.102
    (6, 2, 1.105),   # val(6,2) ~ 1.125 (joint_bisect)
    (8, 2, 1.160),   # val(8,2) ~ 1.183 estimate — if this is too tight, lower
]


def bench_one(d, order, t_test):
    print(f"\n=== d={d} order={order} t_test={t_test} ===")
    hdr = f"{'path':>8} {'total':>8} {'solver':>8} {'round+res':>10} {'verdict':>14} {'margin':>12}"
    print(hdr)
    print('-' * len(hdr))
    runs = [('slow', False, None), ('fast_1e4', True, 10**4),
            ('fast_1e5', True, 10**5), ('fast_1e6', True, 10**6),
            ('fast_1e7', True, 10**7)]
    for name, fast, D_L in runs:
        t0 = time.time()
        try:
            kwargs = dict(
                d=d, order=order, t_test=t_test,
                max_denom_S=10**9, max_denom_mu=10**10,
                eig_margin=1e-9, nthreads=8, verbose=False,
                use_fast_residual=fast,
            )
            if D_L is not None:
                kwargs['fast_D_L'] = D_L
            res, cert = farkas_certify_at(**kwargs)
            dt = time.time() - t0
            round_res = res.round_time
            solver = res.solver_time
            status = res.status
            margin = res.safety_margin_float
            print(f"{name:>9} {dt:>8.2f} {solver:>8.2f} {round_res:>10.2f} "
                  f"{status:>14} {margin:>+12.2e}")
        except Exception as e:
            dt = time.time() - t0
            print(f"{name:>9} FAILED after {dt:.2f}s: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    for (d, order, t) in CASES:
        try:
            bench_one(d, order, t)
        except Exception as e:
            print(f"FAILED d={d} order={order}: {type(e).__name__}: {e}")

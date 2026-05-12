"""Profile farkas_certify_at to quantify the bottleneck before vectorizing.

Runs a single probe at (d, order) just ABOVE known val(d) so the solver
returns an infeasibility certificate, then times each phase.
"""
from __future__ import annotations
import os, sys, time, cProfile, pstats, io

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from certified_lasserre.farkas_certify import farkas_certify_at


def profile_one(d: int, order: int, t_test: float):
    """Run one farkas_certify_at probe and dump a cProfile summary."""
    print(f"\n=== PROFILE d={d} order={order} t_test={t_test} ===")
    prof = cProfile.Profile()
    prof.enable()
    t0 = time.time()
    res, cert = farkas_certify_at(d=d, order=order, t_test=t_test,
                                    max_denom_S=10**9, max_denom_mu=10**10,
                                    eig_margin=1e-9, nthreads=8,
                                    verbose=True)
    elapsed = time.time() - t0
    prof.disable()
    print(f"\n[profile] total = {elapsed:.2f}s  status={res.status}  "
          f"margin={res.safety_margin_float:+.3e}")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats('cumulative') \
        .print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    # d=4 order=3: val ~ 1.102, probe just above
    profile_one(d=4, order=3, t_test=1.095)
    # d=6 order=3: val ~ 1.170, probe just above; this is where the loop hurts
    profile_one(d=6, order=3, t_test=1.165)
    # d=8 order=3: val ~ 1.205, probe just above; main scalability test
    profile_one(d=8, order=3, t_test=1.200)

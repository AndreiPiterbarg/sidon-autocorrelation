"""S1 timing: synthetic stress test of MOSEK SDP at d_parent in {4,6,8,10,12}.

Builds a synthetic 'parent' (palindromic central) of given d_parent and
runs the MOSEK cert just for timing. The actual feasibility outcome is
not the focus here; we want time-per-call.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _S1_sdp_mosek import sdp_certify_parent_mosek
from cascade_opts import sdp_certify_parent


def _make_synthetic(d_parent, m_total):
    """Build a roughly palindromic parent_int summing to m_total."""
    base = m_total // d_parent
    rem = m_total - base * d_parent
    parent = np.full(d_parent, base, dtype=np.int32)
    # Distribute rem across the centre
    for i in range(rem):
        parent[d_parent // 2 - i // 2 + (i & 1)] += 1
    # Box: each cursor in [0, 2*parent[i]]
    lo = np.zeros(d_parent, dtype=np.int32)
    hi = (2 * parent).astype(np.int32)
    return parent, lo, hi


def main():
    n_half_child = 12  # arbitrary moderately large
    m = 10
    c_target = 1.28
    print(f"d_parent | MOSEK time (mean of 5)  | CVXPY time (single)")
    print("-" * 64)
    for d_parent in [4, 6, 8, 10, 12]:
        # parent shape: ~ palindromic with roughly balanced bins
        parent, lo, hi = _make_synthetic(d_parent, 4 * n_half_child * m // 2)

        # Warm-up + 5 timed MOSEK calls
        t_ms = []
        for _ in range(6):
            t0 = time.time()
            res, status = sdp_certify_parent_mosek(parent, lo, hi,
                                                  n_half_child, m, c_target,
                                                  return_status=True)
            t_ms.append(time.time() - t0)
        t_ms = np.array(t_ms[1:])  # drop warm-up

        # CVXPY (single call, often slower)
        t0 = time.time()
        try:
            cv_res = sdp_certify_parent(parent, lo, hi,
                                        n_half_child, m, c_target)
        except Exception as e:
            cv_res = f"err:{e}"
        t_cv = time.time() - t0

        print(f"  {d_parent:>2}   | {1000*t_ms.mean():>7.1f} ms (status={status})"
              f"  | {1000*t_cv:>7.1f} ms (cv_res={cv_res})")


if __name__ == '__main__':
    main()

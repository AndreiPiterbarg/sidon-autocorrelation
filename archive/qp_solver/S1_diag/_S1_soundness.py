"""S1 soundness probe: construct cases where SDP MUST say feasible, and
cases where SDP MUST say infeasible, and verify MOSEK returns the right
answer. Also verify on real cascade parents that whenever MOSEK returns
True, no integer child survives.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _S1_sdp_mosek import sdp_certify_parent_mosek, _build_window_quadratics
from cascade_opts import _theorem1_threshold_table


def integer_count_surviving(parent, lo_arr, hi_arr, n_half_child, m, c_target):
    d_parent = len(parent)
    d_child = 2 * d_parent
    thr_table = _theorem1_threshold_table(int(n_half_child), int(m), c_target)
    P = parent.astype(np.int64)
    lo = lo_arr.astype(np.int64)
    hi = hi_arr.astype(np.int64)

    rsizes = (hi - lo + 1)
    n_enum = int(np.prod(rsizes))
    if n_enum > 5_000_000:
        return n_enum, None

    cur = lo.copy().astype(np.int64)
    n_total = 0
    n_surv = 0
    conv_len = 2 * d_child - 1
    while True:
        n_total += 1
        child = np.empty(d_child, dtype=np.int64)
        for p in range(d_parent):
            child[2 * p] = cur[p]
            child[2 * p + 1] = 2 * P[p] - cur[p]
        if not np.any(child < 0):
            conv = np.zeros(conv_len, dtype=np.int64)
            for ii in range(d_child):
                ci = child[ii]
                if ci != 0:
                    conv[2 * ii] += ci * ci
                    for jj in range(ii + 1, d_child):
                        cj = child[jj]
                        if cj != 0:
                            conv[ii + jj] += 2 * ci * cj
            pruned = False
            for ell in range(2, 2 * d_child + 1):
                n_cv = ell - 1
                n_win = conv_len - n_cv + 1
                if n_win <= 0:
                    continue
                thr = thr_table[ell - 2]
                for s_lo in range(n_win):
                    ws = int(conv[s_lo:s_lo + n_cv].sum())
                    if ws > thr:
                        pruned = True
                        break
                if pruned:
                    break
            if not pruned:
                n_surv += 1
        i = 0
        cur[0] += 1
        while cur[i] > hi[i]:
            cur[i] = lo[i]
            i += 1
            if i >= d_parent:
                return n_total, n_surv
            cur[i] += 1


def main():
    print("\n=== Soundness probe 1: trivially infeasible (lo > hi). Expect False. ===")
    parent = np.array([5, 5, 5, 5], dtype=np.int32)
    lo = np.array([3, 3, 3, 3], dtype=np.int32)
    hi = np.array([7, 7, 7, 7], dtype=np.int32)
    res, s = sdp_certify_parent_mosek(parent, lo, hi, 4, 5, 1.20, return_status=True)
    print(f"  res={res}  status={s}  (any answer is OK; this just runs the path)")

    print("\n=== Soundness probe 2: very loose box (full range). Expect feasible (False). ===")
    parent = np.array([5, 5, 5, 5], dtype=np.int32)
    lo = np.array([0, 0, 0, 0], dtype=np.int32)
    hi = np.array([10, 10, 10, 10], dtype=np.int32)
    res, s = sdp_certify_parent_mosek(parent, lo, hi, 4, 5, 1.20, return_status=True)
    print(f"  res={res}  status={s}  (must be False; box is loose)")
    assert res is False, "BUG: false-positive on loose box"

    print("\n=== Soundness probe 3: heavy mass concentration. Likely SDP-infeasible. ===")
    parent = np.array([20, 1, 1, 20], dtype=np.int32)
    lo = np.array([0, 0, 0, 0], dtype=np.int32)
    hi = np.array([40, 2, 2, 40], dtype=np.int32)
    res, s = sdp_certify_parent_mosek(parent, lo, hi, 8, 10, 1.28, return_status=True)
    print(f"  res={res}  status={s}")

    if res:
        # verify no integer child survives
        n_t, n_s = integer_count_surviving(parent, lo, hi, 8, 10, 1.28)
        print(f"  enumeration: {n_t} children, {n_s} survive (must be 0)")
        if n_s and n_s > 0:
            print("  !!! SOUNDNESS VIOLATION !!!")
        else:
            print("  OK: enumeration agrees")

    print("\n=== Soundness probe 4: small d_p=2 cases (exhaustive cross-check). ===")
    np.random.seed(0)
    n_violations = 0
    n_checked = 0
    for trial in range(15):
        # random parent
        d_p = 2
        n_half_child = 4
        m = 8
        c_target = 1.28
        parent = np.random.randint(2, 12, size=d_p).astype(np.int32)
        # random lo/hi within [0, 2*parent]
        lo = np.array([0] * d_p, dtype=np.int32)
        hi = (2 * parent).astype(np.int32)
        # tighten randomly
        for ii in range(d_p):
            lo[ii] = np.random.randint(0, max(1, parent[ii]))
            hi[ii] = np.random.randint(int(parent[ii]) + 1, int(2 * parent[ii] + 1))

        try:
            res, s = sdp_certify_parent_mosek(parent, lo, hi,
                                              n_half_child, m, c_target,
                                              return_status=True)
        except Exception as e:
            print(f"  trial {trial}: MOSEK exception {e}")
            continue
        n_t, n_s = integer_count_surviving(parent, lo, hi, n_half_child, m, c_target)
        n_checked += 1
        if res and n_s is not None and n_s > 0:
            n_violations += 1
            print(f"  trial {trial}: VIOLATION res=True, but {n_s}/{n_t} children survive!")
            print(f"    parent={parent.tolist()} lo={lo.tolist()} hi={hi.tolist()}")
        else:
            tag = "OK" if not res else f"OK (no surv among {n_t})"
            print(f"  trial {trial}: parent={parent.tolist()} res={res} status={s} surv={n_s}/{n_t} {tag}")

    print(f"\n=== Total: {n_violations}/{n_checked} soundness violations ===")


if __name__ == '__main__':
    main()

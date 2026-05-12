"""Cross-checks that the MOSEK Task SDP cert agrees with CVXPY at small d.

Test harness that:
  (a) Calls existing `sdp_certify_parent` (CVXPY/SCS) and `_S1_sdp_mosek.py`
      on the same parent.
  (b) Runs full child enumeration and verifies that whenever MOSEK returns
      True (infeasible cert), the *actual* set of children all violate at
      least one window threshold.

This protects against false-positive certifications.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from cascade_opts import (_whole_parent_prune_theorem1,
                          lp_dual_certificate, sdp_certify_parent,
                          _theorem1_threshold_table)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges
from _S1_sdp_mosek import sdp_certify_parent_mosek


def enumerate_children(parent_int, lo_arr, hi_arr, n_half_child, m, c_target):
    """Brute-force: try all children, see if ANY survives ws_W <= thr_W for all W.

    Returns (n_total, n_surv).
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    # threshold table
    thr_table = _theorem1_threshold_table(int(n_half_child), int(m), c_target)

    # Enumerate (this is exponential; only sane for very small problems)
    children = []
    def rec(i, cur):
        if i == d_parent:
            children.append(tuple(cur))
            return
        for x in range(int(lo_arr[i]), int(hi_arr[i]) + 1):
            rec(i + 1, cur + [x])
    rec(0, [])

    n_surv = 0
    n_total = len(children)
    for cur in children:
        child = np.zeros(d_child, dtype=np.int64)
        for p in range(d_parent):
            child[2 * p] = cur[p]
            child[2 * p + 1] = 2 * int(parent_int[p]) - cur[p]
        # Check that child is non-negative
        if np.any(child < 0):
            continue
        # Compute autoconvolution
        conv_len = 2 * d_child - 1
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            ci = child[i]
            for j in range(d_child):
                cj = child[j]
                conv[i + j] += ci * cj
        # For each window check ws <= threshold (Theorem 1 form)
        is_pruned = False
        for ell in range(2, 2 * d_child + 1):
            n_cv = ell - 1
            n_win = conv_len - n_cv + 1
            if n_win <= 0:
                continue
            thr = thr_table[ell - 2]
            for s_lo in range(n_win):
                ws = int(conv[s_lo:s_lo + n_cv].sum())
                if ws > thr:
                    is_pruned = True
                    break
            if is_pruned:
                break
        if not is_pruned:
            n_surv += 1
    return n_total, n_surv


def assess_one(n_half, m, c, max_parents=20):
    d = 2 * n_half
    S_half = 2 * n_half * m

    surv = []
    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        s = _prune_dynamic_int32(batch, n_half, m, c, use_flat_threshold=False)
        if s.any():
            surv.append(batch[s].copy())
    if surv:
        surv = np.vstack(surv)
    else:
        surv = np.empty((0, d), dtype=np.int32)
    print(f"\n=== n_half={n_half} m={m} c={c}  (d={d})  L0 survivors: {len(surv)} ===")
    if len(surv) == 0:
        return

    n_half_child = 2 * n_half
    sample = surv[:max_parents]
    n_t1 = n_lp = n_cvxpy = n_mosek = 0
    n_lp_skipped = 0
    n_total = 0
    cvxpy_times = []
    mosek_times = []
    soundness_violations = 0

    parents_for_sdp = []  # (parent, lo, hi) where T1 and LP did not clear
    for parent in sample:
        d_child = 2 * d
        try:
            res = _compute_bin_ranges(parent, m, c, d_child, n_half_child)
        except Exception:
            continue
        if res is None:
            continue
        lo_arr, hi_arr, total_children = res
        if total_children == 0:
            continue
        n_total += 1
        if _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                        int(n_half_child), int(m), c):
            n_t1 += 1
            continue
        if d <= 10:
            try:
                lp = lp_dual_certificate(parent, lo_arr, hi_arr,
                                         int(n_half_child), int(m), c)
            except Exception:
                lp = False
            if lp:
                n_lp += 1
                continue
        else:
            n_lp_skipped += 1

        parents_for_sdp.append((parent, lo_arr, hi_arr))

    print(f"  After T1+LP: {len(parents_for_sdp)} parents need SDP (out of {n_total})")
    print(f"  T1 cleared: {n_t1}, LP cleared: {n_lp}, LP skipped: {n_lp_skipped}")

    for (parent, lo_arr, hi_arr) in parents_for_sdp:
        # CVXPY
        t0 = time.time()
        try:
            cv = sdp_certify_parent(parent, lo_arr, hi_arr,
                                    int(n_half_child), int(m), c)
        except Exception:
            cv = False
        t_cv = time.time() - t0
        cvxpy_times.append(t_cv)
        if cv:
            n_cvxpy += 1

        # MOSEK
        t0 = time.time()
        ms, status = sdp_certify_parent_mosek(parent, lo_arr, hi_arr,
                                              int(n_half_child), int(m), c,
                                              return_status=True)
        t_ms = time.time() - t0
        mosek_times.append(t_ms)
        if ms:
            n_mosek += 1

        # Optional soundness probe: if MOSEK says True and the child space
        # is small (<=64 enumerable), enumerate and verify.
        if ms and len(parent) <= 4:
            n_total_ch, n_surv_ch = enumerate_children(parent, lo_arr, hi_arr,
                                                      int(n_half_child),
                                                      int(m), c)
            if n_surv_ch > 0:
                soundness_violations += 1
                print(f"  !!! SOUNDNESS VIOLATION on parent={parent.tolist()}, "
                      f"lo={lo_arr.tolist()}, hi={hi_arr.tolist()}: "
                      f"{n_surv_ch}/{n_total_ch} children survive but MOSEK says infeasible (status={status})")

    n_for_sdp = len(parents_for_sdp)
    if n_for_sdp == 0:
        print("  (No parents reach SDP stage.)")
        return

    cv_times = np.array(cvxpy_times) if cvxpy_times else np.array([0.0])
    ms_times = np.array(mosek_times) if mosek_times else np.array([0.0])
    print(f"  CVXPY  cleared: {n_cvxpy:3}/{n_for_sdp}  "
          f"avg time {1000*cv_times.mean():.1f} ms  "
          f"max {1000*cv_times.max():.1f} ms")
    print(f"  MOSEK  cleared: {n_mosek:3}/{n_for_sdp}  "
          f"avg time {1000*ms_times.mean():.1f} ms  "
          f"max {1000*ms_times.max():.1f} ms")
    print(f"  Soundness probe (d_p<=4): {soundness_violations} violations")
    return {
        'cvxpy_cleared': n_cvxpy, 'mosek_cleared': n_mosek,
        'n_for_sdp': n_for_sdp,
        'cvxpy_avg_ms': float(1000*cv_times.mean()),
        'mosek_avg_ms': float(1000*ms_times.mean()),
        'soundness_violations': soundness_violations,
    }


if __name__ == '__main__':
    # Just a quick sanity check on smallest problem
    assess_one(2, 30, 1.20, max_parents=20)

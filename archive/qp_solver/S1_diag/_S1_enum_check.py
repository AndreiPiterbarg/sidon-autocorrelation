"""For one specific small parent (n=2, m=30, c=1.20), enumerate the
INTEGER children to check if any survive the windows.

If any integer child survives -> SDP relaxation is correct in saying
'optimal' (feasible) and SDP cannot prune. S1 is genuinely useless here.

If NO integer child survives -> the SDP relaxation is loose; we need
sigma-1 / 2nd-moment lift / bilinear cuts to tighten it.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from cascade_opts import (_whole_parent_prune_theorem1,
                          lp_dual_certificate, sdp_certify_parent,
                          _theorem1_threshold_table)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges


def integer_count_surviving(parent, lo_arr, hi_arr, n_half_child, m, c_target):
    """Counts integer children (cursors c[0..d-1] in their boxes) that survive
    EVERY Theorem-1 window threshold. Returns first surviving cursor (or None).

    For symmetric (palindromic) parents we exploit the structure to enumerate
    only half of the cursor.
    """
    d_parent = len(parent)
    d_child = 2 * d_parent
    thr_table = _theorem1_threshold_table(int(n_half_child), int(m), c_target)
    P = parent.astype(np.int64)
    lo = lo_arr.astype(np.int64)
    hi = hi_arr.astype(np.int64)

    # Total cursor enumeration
    rsizes = (hi - lo + 1)
    n_enum = int(np.prod(rsizes))
    if n_enum > 10_000_000:
        return None, n_enum, "too_many"

    # Enumerate via mixed-radix counter
    rs = rsizes.astype(np.int64)
    cur = lo.copy().astype(np.int64)
    n_total = 0
    n_surv = 0
    first_surv = None
    conv_len = 2 * d_child - 1
    while True:
        n_total += 1
        # Build child
        child = np.empty(d_child, dtype=np.int64)
        for p in range(d_parent):
            child[2 * p] = cur[p]
            child[2 * p + 1] = 2 * P[p] - cur[p]
        if not np.any(child < 0):
            # Compute conv
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
                if first_surv is None:
                    first_surv = cur.copy()
        # increment
        i = 0
        cur[0] += 1
        while cur[i] > hi[i]:
            cur[i] = lo[i]
            i += 1
            if i >= d_parent:
                return first_surv, n_total, n_surv
            cur[i] += 1


def main():
    n_half, m, c = 2, 30, 1.20
    d = 2 * n_half
    S_half = 2 * n_half * m
    n_half_child = 2 * n_half

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
    print(f"L0 survivors: {len(surv)}")

    # Take first 2 SDP-stage parents
    sdp_stage = []
    for parent in surv[:20]:
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
        if _whole_parent_prune_theorem1(parent, lo_arr, hi_arr,
                                        int(n_half_child), int(m), c):
            continue
        if d <= 10:
            try:
                if lp_dual_certificate(parent, lo_arr, hi_arr,
                                       int(n_half_child), int(m), c):
                    continue
            except Exception:
                pass
        sdp_stage.append((parent, lo_arr, hi_arr))

    print(f"SDP-stage parents: {len(sdp_stage)}")

    # For first 5 SDP-stage parents: smaller box (use central 3x3x3x3 = 81 children) check
    for idx, (parent, lo_arr, hi_arr) in enumerate(sdp_stage[:5]):
        # Restrict to smaller central window for tractability
        d_parent = len(parent)
        rsizes = hi_arr - lo_arr + 1
        # Try the full enumeration if <= 1M
        n_enum = int(np.prod(rsizes.astype(np.int64)))
        print(f"\n[{idx}] parent={parent.tolist()} ranges={rsizes.tolist()} n_enum={n_enum:,}")
        if n_enum > 5_000_000:
            # Try a 5x5 region around the midpoint
            mid = (lo_arr + hi_arr) // 2
            half_w = 2
            sub_lo = np.maximum(lo_arr, mid - half_w)
            sub_hi = np.minimum(hi_arr, mid + half_w)
            print(f"      subbox: lo={sub_lo.tolist()} hi={sub_hi.tolist()}")
            t0 = time.time()
            first_surv, n_tot, n_surv = integer_count_surviving(
                parent, sub_lo, sub_hi,
                int(n_half_child), int(m), c)
            print(f"      sub-enum: total={n_tot} surv={n_surv} first={first_surv if first_surv is None else first_surv.tolist()} ({time.time()-t0:.1f}s)")
        else:
            t0 = time.time()
            first_surv, n_tot, n_surv = integer_count_surviving(
                parent, lo_arr, hi_arr,
                int(n_half_child), int(m), c)
            print(f"      full enum: total={n_tot} surv={n_surv} first={first_surv if first_surv is None else first_surv.tolist()} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()

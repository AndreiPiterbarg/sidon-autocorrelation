"""Diagnostic: when MOSEK says 'OPTIMAL' (i.e., SDP feasible, no cert),
is there actually an integer child that survives, or is the SDP relaxation
just too loose?

If integer enumeration shows 0 surviving children but the SDP is feasible,
then no SDP-based cert can ever fire — the relaxation gap is fundamental.
In that case, S1 is NOT a useful direction.
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
from _S1_sdp_mosek import sdp_certify_parent_mosek, _build_window_quadratics


def diag_one(n_half, m, c, max_parents=10):
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
    print(f"=== n_half={n_half} m={m} c={c}  (d={d})  L0 survivors: {len(surv)} ===")

    n_half_child = 2 * n_half
    sample = surv[:max_parents]

    for idx, parent in enumerate(sample):
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
        # Now this parent reaches the SDP stage.
        # Enumerate ALL children (must be feasible for d_p <= 4 and ranges <= ~30)
        d_parent = d
        thr_table = _theorem1_threshold_table(int(n_half_child), int(m), c)
        # Bound on enumeration count
        rsizes = [int(hi_arr[i]) - int(lo_arr[i]) + 1 for i in range(d_parent)]
        n_enum = 1
        for r in rsizes:
            n_enum *= r
        if n_enum > 500_000:
            print(f"  parent={parent.tolist()} too many ({n_enum}) children to enumerate")
            continue

        # Enumerate
        n_surv = 0
        # Find any surviving child if it exists
        first_surv = None
        def rec(i, cur, conv_running):
            nonlocal n_surv, first_surv
            if i == d_parent:
                # check
                conv_len = 2 * d_child - 1
                # full child
                child = np.zeros(d_child, dtype=np.int64)
                for p in range(d_parent):
                    child[2 * p] = cur[p]
                    child[2 * p + 1] = 2 * int(parent[p]) - cur[p]
                if np.any(child < 0):
                    return
                conv = np.zeros(conv_len, dtype=np.int64)
                for ii in range(d_child):
                    ci = child[ii]
                    if ci == 0:
                        continue
                    for jj in range(d_child):
                        conv[ii + jj] += ci * child[jj]
                # Theorem 1 check
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
                        first_surv = list(cur)
                return
            for x in range(int(lo_arr[i]), int(hi_arr[i]) + 1):
                rec(i + 1, cur + [x], None)
        rec(0, [], None)

        # Run MOSEK
        t0 = time.time()
        ms_res, ms_status = sdp_certify_parent_mosek(parent, lo_arr, hi_arr,
                                                    int(n_half_child), int(m), c,
                                                    return_status=True)
        ms_t = (time.time() - t0) * 1000

        # Build win quads to check ws_W at first surviving child
        win_quads = _build_window_quadratics(parent, lo_arr, hi_arr,
                                              int(n_half_child), int(m), c)
        max_excess = -np.inf
        bind_window = -1
        for wi, (Q, g, k, thr) in enumerate(win_quads):
            # eval at "center" of box (for clue about how tight the SDP is)
            xc = (lo_arr.astype(float) + hi_arr.astype(float)) / 2.0
            ws_c = float(xc @ Q @ xc + g @ xc + k)
            ex = ws_c - thr
            if ex > max_excess:
                max_excess = ex
                bind_window = wi

        print(f"  [{idx}] parent={parent.tolist()} lo={lo_arr.tolist()} hi={hi_arr.tolist()} "
              f"n_enum={n_enum} surv={n_surv} MOSEK={ms_res}({ms_status}, {ms_t:.0f}ms)")
        if first_surv is not None:
            print(f"      first surviving cursor: {first_surv}")
        print(f"      max excess at center over windows: {max_excess:.3f} (window #{bind_window})")


if __name__ == '__main__':
    diag_one(2, 30, 1.20, max_parents=5)

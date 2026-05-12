"""Test: if we used the TIGHT (no box-cert correction) threshold, would
the SDP relaxation become infeasible?

This tells us whether the SDP gap is from the relaxation itself or just
from the loose threshold.
"""
import os, sys, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from cascade_opts import (_whole_parent_prune_theorem1, lp_dual_certificate,
                          sdp_certify_parent, _theorem1_threshold_table)
from run_cascade import _prune_dynamic_int32, _compute_bin_ranges
import mosek


def sdp_certify_tight(parent_int, lo_arr, hi_arr, n_half_child, m, c_target,
                      use_box_cert=True, verbose=False):
    """Same as sdp_certify_parent_mosek but optionally drops the box-cert
    correction in the threshold (use_box_cert=False).
    """
    d = len(parent_int)
    if d == 0:
        return False, "d=0"
    lo = np.array([float(lo_arr[i]) for i in range(d)])
    hi = np.array([float(hi_arr[i]) for i in range(d)])

    d_child = 2 * d
    n_half = int(n_half_child)
    conv_len = 2 * d_child - 1

    P = np.array([float(parent_int[i]) for i in range(d)])

    signs = np.empty(d_child)
    consts = np.empty(d_child)
    parent_of = np.empty(d_child, dtype=int)
    for i in range(d):
        signs[2 * i] = 1.0
        signs[2 * i + 1] = -1.0
        consts[2 * i] = 0.0
        consts[2 * i + 1] = 2.0 * P[i]
        parent_of[2 * i] = i
        parent_of[2 * i + 1] = i

    B_corr = float(n_half) * (8.0 * m + 1.0) / 2.0
    win_quads = []
    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_win = conv_len - n_cv + 1
        if n_win <= 0:
            continue
        mult = min(n_half, ell - 1, 2 * d_child - ell)
        scale = float(ell) * 4.0 * n_half
        thr_loose = c_target * m * m * scale + mult * B_corr
        thr_tight = c_target * m * m * scale  # no correction
        thr_use = thr_loose if use_box_cert else thr_tight

        for s_lo in range(n_win):
            Q = np.zeros((d, d))
            g = np.zeros(d)
            k = 0.0
            for r in range(s_lo, s_lo + n_cv):
                p_lo = max(0, r - d_child + 1)
                p_hi = min(d_child, r + 1)
                for p in range(p_lo, p_hi):
                    q = r - p
                    if q < 0 or q >= d_child:
                        continue
                    a = parent_of[p]
                    b = parent_of[q]
                    sp = signs[p]
                    sq = signs[q]
                    cp_ = consts[p]
                    cq = consts[q]
                    Q[a, b] += sp * sq
                    g[a] += sp * cq
                    g[b] += cp_ * sq
                    k += cp_ * cq
            Q_sym = (Q + Q.T) / 2.0
            win_quads.append((Q_sym, g, k, thr_use))

    bar_dim = d + 1
    with mosek.Env() as env:
        env.checkoutlicense(mosek.feature.pts)
        with env.Task(0, 0) as task:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, 1e-10)
            if not verbose:
                task.putintparam(mosek.iparam.log, 0)

            task.appendbarvars([bar_dim])
            task.putobjsense(mosek.objsense.minimize)

            def add_lin(subi, subj, val, bk, blk, buk):
                ci = task.getnumcon()
                task.appendcons(1)
                if len(subi):
                    aid = task.appendsparsesymmat(bar_dim, subi, subj, val)
                    task.putbaraij(ci, 0, [aid], [1.0])
                task.putconbound(ci, bk, blk, buk)

            # Y[0,0] = 1
            add_lin([0], [0], [1.0], mosek.boundkey.fx, 1.0, 1.0)
            # box on x
            for i in range(d):
                add_lin([i + 1], [0], [0.5], mosek.boundkey.ra, lo[i], hi[i])
            # diag X bounds
            for i in range(d):
                add_lin([i + 1], [i + 1], [1.0],
                        mosek.boundkey.ra, lo[i] ** 2, hi[i] ** 2)
            # RLT cuts
            for i in range(d):
                for j in range(i + 1, d):
                    li, lj_ = lo[i], lo[j]
                    ui, uj_ = hi[i], hi[j]
                    # X[i,j] - lj*x_i - li*x_j + li*lj >= 0
                    subi = [j + 1, i + 1, j + 1, 0]
                    subj = [i + 1, 0, 0, 0]
                    val = [0.5, -lj_ * 0.5, -li * 0.5, li * lj_]
                    add_lin(subi, subj, val, mosek.boundkey.lo, 0.0, 0.0)
                    # X[i,j] - uj*x_i - ui*x_j + ui*uj >= 0
                    val = [0.5, -uj_ * 0.5, -ui * 0.5, ui * uj_]
                    add_lin([j + 1, i + 1, j + 1, 0], [i + 1, 0, 0, 0],
                            val, mosek.boundkey.lo, 0.0, 0.0)
                    # -X[i,j] + uj*x_i + li*x_j - li*uj >= 0
                    val = [-0.5, uj_ * 0.5, li * 0.5, -li * uj_]
                    add_lin([j + 1, i + 1, j + 1, 0], [i + 1, 0, 0, 0],
                            val, mosek.boundkey.lo, 0.0, 0.0)
                    # -X[i,j] + lj*x_i + ui*x_j - ui*lj >= 0
                    val = [-0.5, lj_ * 0.5, ui * 0.5, -ui * lj_]
                    add_lin([j + 1, i + 1, j + 1, 0], [i + 1, 0, 0, 0],
                            val, mosek.boundkey.lo, 0.0, 0.0)
            # window constraints
            for (Q_sym, g_vec, k_val, thr_val) in win_quads:
                subi = []
                subj = []
                val = []
                # k_val on Y[0,0]
                if k_val != 0:
                    subi.append(0); subj.append(0); val.append(float(k_val))
                # g[i] on Y[i+1, 0]
                for i in range(d):
                    if g_vec[i] != 0:
                        subi.append(i + 1); subj.append(0); val.append(float(g_vec[i]) * 0.5)
                # X coefs
                for ii in range(d):
                    if Q_sym[ii, ii] != 0:
                        subi.append(ii + 1); subj.append(ii + 1); val.append(float(Q_sym[ii, ii]))
                    for jj in range(ii):
                        cf = 2.0 * float(Q_sym[ii, jj])
                        if cf != 0:
                            subi.append(ii + 1); subj.append(jj + 1); val.append(cf * 0.5)
                add_lin(subi, subj, val, mosek.boundkey.up, -1e30, float(thr_val))

            task.optimize()
            solsta = task.getsolsta(mosek.soltype.itr)
            return (solsta == mosek.solsta.prim_infeas_cer), str(solsta)


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

    sdp_stage = []
    for parent in surv[:30]:
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
    n_loose = n_tight = 0
    for (parent, lo_arr, hi_arr) in sdp_stage:
        ms_loose, st_loose = sdp_certify_tight(parent, lo_arr, hi_arr,
                                               int(n_half_child), int(m), c,
                                               use_box_cert=True)
        ms_tight, st_tight = sdp_certify_tight(parent, lo_arr, hi_arr,
                                               int(n_half_child), int(m), c,
                                               use_box_cert=False)
        n_loose += int(ms_loose)
        n_tight += int(ms_tight)
        print(f"  parent={parent.tolist()} loose={ms_loose}({st_loose}) tight={ms_tight}({st_tight})")

    print(f"\nLoose threshold (matches LP): {n_loose}/{len(sdp_stage)}")
    print(f"Tight threshold (drop box-cert): {n_tight}/{len(sdp_stage)}")


if __name__ == '__main__':
    main()

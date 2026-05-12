"""Direct MOSEK Task API replacement for `sdp_certify_parent`.

Function `sdp_certify_parent_mosek(parent_int, lo_arr, hi_arr, n_half_child,
m, c_target)` returns True iff the Shor/Lasserre relaxation proves the
QCQP infeasible (i.e., no cursor x in [lo, hi] satisfies ws_W(x) <= thr_W
for ALL windows W). When infeasible, every child of `parent_int` is pruned.

Why this exists: the CVXPY+SCS implementation in
`cloninger-steinerberger/cpu/cascade_opts.py` (lines 440-571) emits
"Solution may be inaccurate" on small parents and reports 0/17 clearance
on (n=2, m=30, c=1.20). Direct MOSEK Task API with high-accuracy IPM
tolerances is the next step.

Lifted variable
---------------
We use a single PSD bar matrix Y of size (d+1)x(d+1) where d := d_parent.
Identify (with lower-triangle indices i >= j):

    Y[0,0]    = 1         (anchored)
    Y[i+1,0]  = x_i       (i = 0..d-1)
    Y[i+1,j+1] = X[i,j]   (i, j = 0..d-1; X symmetric)

The Schur-complement view: [[1, x^T], [x, X]] >= 0  iff  X >= xx^T.
We do NOT add additional scalar variables; everything lives on Y.

Inner products
--------------
For symmetric A in the (d+1)x(d+1) space, MOSEK uses
    <A, Y> = sum_i A[i,i]*Y[i,i] + 2*sum_{i>j} A[i,j]*Y[i,j].
So to extract a coefficient `alpha` on Y[i,j] (i > j) in a constraint,
supply A[i,j] = alpha/2. To extract `alpha` on Y[i,i], supply A[i,i] = alpha.

Constraints
-----------
1. Y[0,0] == 1                                (anchor)
2. lo[i] <= Y[i+1,0] <= hi[i]                 (box on x)
3. lo[i]^2 <= Y[i+1,i+1] <= hi[i]^2           (diagonal X bounds)
4. RLT cuts on off-diagonal X[i,j]:
       (x_i - lo_i)(x_j - lo_j) >= 0  -->  X[i,j] - lo_j*x_i - lo_i*x_j + lo_i*lo_j >= 0
       (hi_i - x_i)(hi_j - x_j) >= 0  -->  X[i,j] - hi_j*x_i - hi_i*x_j + hi_i*hi_j >= 0
       (x_i - lo_i)(hi_j - x_j) >= 0  --> -X[i,j] + hi_j*x_i + lo_i*x_j - lo_i*hi_j >= 0
       (hi_i - x_i)(x_j - lo_j) >= 0  --> -X[i,j] + lo_j*x_i + hi_i*x_j - hi_i*lo_j >= 0
5. Window: <Q_W, X> + g_W^T x + k_W <= thr_W

Feasibility test
----------------
We solve `min 0` subject to all constraints. We declare the original QCQP
infeasible (and so prune all children) iff MOSEK reports
PRIMAL_INFEASIBLE_CER (or, conservatively, PRIMAL_INFEASIBLE) on the IPM
status. UNKNOWN/OPTIMAL is treated as not certified (return False).
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import numpy as np


def _build_window_quadratics(parent_int, lo_arr, hi_arr, n_half_child, m, c_target):
    """Return list of (Q_sym, g, k, thr) for every window.

    This mirrors the build in `sdp_certify_parent` (cascade_opts.py:484-521)
    so the underlying QP is identical.
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    n_half = int(n_half_child)
    conv_len = 2 * d_child - 1

    P = np.array([float(parent_int[i]) for i in range(d_parent)])

    signs = np.empty(d_child)
    consts = np.empty(d_child)
    parent_of = np.empty(d_child, dtype=int)
    for i in range(d_parent):
        signs[2 * i] = 1.0
        signs[2 * i + 1] = -1.0
        consts[2 * i] = 0.0
        consts[2 * i + 1] = 2.0 * P[i]
        parent_of[2 * i] = i
        parent_of[2 * i + 1] = i

    B_corr = float(n_half) * (8.0 * m + 1.0) / 2.0

    out = []
    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_win = conv_len - n_cv + 1
        if n_win <= 0:
            continue
        mult = min(n_half, ell - 1, 2 * d_child - ell)
        scale = float(ell) * 4.0 * n_half
        thr = c_target * m * m * scale + mult * B_corr

        for s_lo in range(n_win):
            Q = np.zeros((d_parent, d_parent))
            g = np.zeros(d_parent)
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
            out.append((Q_sym, g, k, thr))
    return out


def _coeffs_for_constraint(d, alpha_const, x_coeffs, X_coeffs):
    """Given a constraint of the form

        alpha_const * Y[0,0] + sum_i x_coeffs[i] * Y[i+1,0]
        + sum_{i>=j} X_coeffs[i,j] * Y[i+1,j+1] = <A, Y>

    return (subi, subj, val) lower-triangular sparse for A so that
        <A, Y> = sum_i A[i,i]*Y[i,i] + 2*sum_{i>j} A[i,j]*Y[i,j]
    matches the desired linear functional in Y.

    For i == j (diagonal of A): coefficient on Y[i,i] is A[i,i] itself.
    For i > j (strict lower triangle): coefficient on Y[i,j] is 2*A[i,j].
    """
    n = d + 1
    subi = []
    subj = []
    val = []

    # Y[0,0] coefficient is alpha_const
    if alpha_const != 0.0:
        subi.append(0)
        subj.append(0)
        val.append(alpha_const)

    # Y[i+1,0] coefficient is x_coeffs[i] (off-diagonal i+1 > 0, so factor 1/2)
    for i in range(d):
        c = x_coeffs[i]
        if c != 0.0:
            subi.append(i + 1)
            subj.append(0)
            val.append(c * 0.5)

    # X[i,j] = Y[i+1, j+1]; we want sum_{i>=j} X_coeffs[i,j] * Y[i+1,j+1]
    # X_coeffs is keyed by (i,j) with i >= j.
    # If i == j: A[i+1,i+1] = X_coeffs[i,i].  (factor 1.)
    # If i >  j: A[i+1,j+1] = X_coeffs[i,j] / 2.  (factor 1/2.)
    for (i, j), c in X_coeffs.items():
        if c == 0.0:
            continue
        if i == j:
            subi.append(i + 1)
            subj.append(j + 1)
            val.append(c)
        else:
            ii = max(i, j)
            jj = min(i, j)
            subi.append(ii + 1)
            subj.append(jj + 1)
            val.append(c * 0.5)

    return subi, subj, val


def sdp_certify_parent_mosek(parent_int, lo_arr, hi_arr,
                              n_half_child, m, c_target,
                              verbose=False, return_status=False):
    """SDP-based parent prune via MOSEK Task API. See module docstring."""
    try:
        import mosek
    except ImportError:
        if return_status:
            return False, "mosek-not-importable"
        return False

    d = len(parent_int)
    if d == 0:
        return (False, "d=0") if return_status else False

    lo = np.array([float(lo_arr[i]) for i in range(d)])
    hi = np.array([float(hi_arr[i]) for i in range(d)])

    win_quads = _build_window_quadratics(parent_int, lo_arr, hi_arr,
                                         n_half_child, m, c_target)
    if not win_quads:
        return (False, "no-windows") if return_status else False

    # Quick necessary check: for any window, is g^T*x + k <= thr feasible at
    # all under the BOX alone (with X arbitrary PSD)?  This is implicit in
    # the SDP and we don't shortcut it; we just feed everything to MOSEK.

    # Also: the box yields naturally x_i^2 <= max(lo_i^2, hi_i^2) and
    # x_i^2 >= 0. We use the tighter bound from McCormick / RLT.

    bar_dim = d + 1

    with mosek.Env() as env:
        # license check is implicit on first solve; checkout up front to
        # surface licensing errors clearly.
        try:
            env.checkoutlicense(mosek.feature.pts)
        except mosek.Error as e:
            if return_status:
                return False, f"license-error: {e}"
            return False

        with env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log,
                                lambda msg: sys.stdout.write(msg))

            # High-accuracy IPM tolerances.
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1.0e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1.0e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0e-10)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, 1.0e-10)
            # Slightly more iterations than default just in case.
            task.putintparam(mosek.iparam.intpnt_max_iterations, 200)
            # We don't need verbose log unless requested.
            if not verbose:
                task.putintparam(mosek.iparam.log, 0)

            # === Decision space ===
            # No scalar variables; one bar matrix Y of dim (d+1).
            task.appendbarvars([bar_dim])

            # Objective: minimize 0 (feasibility test)
            task.putobjsense(mosek.objsense.minimize)
            # leave c (linear obj) zero, bar c also empty -> objective is 0.

            # Helper to add a single constraint
            #   bound_lo <= <A, Y> <= bound_hi
            # given A specified in lower-triangle (subi, subj, val).
            def _add_lin_con(subi, subj, val, bk, blk, buk):
                """Append one linear constraint <A,Y> with bound triple."""
                con_idx = task.getnumcon()
                task.appendcons(1)
                if len(subi) > 0:
                    a_id = task.appendsparsesymmat(bar_dim, subi, subj, val)
                    task.putbaraij(con_idx, 0, [a_id], [1.0])
                task.putconbound(con_idx, bk, blk, buk)
                return con_idx

            # 1) Y[0,0] == 1
            sI, sJ, sV = _coeffs_for_constraint(d, 1.0, np.zeros(d), {})
            _add_lin_con(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

            # 2) Box on x: lo[i] <= x_i <= hi[i]
            for i in range(d):
                xc = np.zeros(d)
                xc[i] = 1.0
                sI, sJ, sV = _coeffs_for_constraint(d, 0.0, xc, {})
                _add_lin_con(sI, sJ, sV, mosek.boundkey.ra, lo[i], hi[i])

            # 3) Diagonal X: lo[i]^2 <= X[i,i] <= hi[i]^2
            for i in range(d):
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs_for_constraint(d, 0.0, np.zeros(d), Xc)
                _add_lin_con(sI, sJ, sV, mosek.boundkey.ra,
                             lo[i] * lo[i], hi[i] * hi[i])

            # 4) RLT cuts on off-diagonal pairs (i, j), i < j
            #    Note we treat X as symmetric and store only X[i,j] with i>=j;
            #    the constraints below all use the unordered pair, so we map
            #    to canonical (max,min). Since X[i,j] = X[j,i], the four cuts
            #    give the McCormick envelope in both halves automatically.
            for i in range(d):
                for j in range(i + 1, d):
                    li, lj_ = lo[i], lo[j]
                    ui, uj_ = hi[i], hi[j]
                    # (a) X[i,j] - lj*x_i - li*x_j + li*lj >= 0
                    xc = np.zeros(d); xc[i] = -lj_; xc[j] = -li
                    Xc = {(j, i): 1.0}  # lower-triangle: j>i
                    sI, sJ, sV = _coeffs_for_constraint(d, li * lj_, xc, Xc)
                    _add_lin_con(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    # (b) X[i,j] - uj*x_i - ui*x_j + ui*uj >= 0
                    xc = np.zeros(d); xc[i] = -uj_; xc[j] = -ui
                    Xc = {(j, i): 1.0}
                    sI, sJ, sV = _coeffs_for_constraint(d, ui * uj_, xc, Xc)
                    _add_lin_con(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    # (c) -X[i,j] + uj*x_i + li*x_j - li*uj >= 0
                    xc = np.zeros(d); xc[i] = uj_; xc[j] = li
                    Xc = {(j, i): -1.0}
                    sI, sJ, sV = _coeffs_for_constraint(d, -li * uj_, xc, Xc)
                    _add_lin_con(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    # (d) -X[i,j] + lj*x_i + ui*x_j - ui*lj >= 0
                    xc = np.zeros(d); xc[i] = lj_; xc[j] = ui
                    Xc = {(j, i): -1.0}
                    sI, sJ, sV = _coeffs_for_constraint(d, -ui * lj_, xc, Xc)
                    _add_lin_con(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)

            # 5) Window inequalities: <Q_W, X> + g_W^T x + k_W <= thr_W
            for (Q_sym, g_vec, k_val, thr_val) in win_quads:
                Xc = {}
                for ii in range(d):
                    for jj in range(ii + 1):  # lower triangle ii >= jj
                        v = Q_sym[ii, jj] + (Q_sym[jj, ii] if ii != jj else 0.0)
                        # tr(Q_sym X) = sum_ij Q_sym[i,j] X[j,i]
                        #             = sum_i Q_sym[i,i] X[i,i] + 2 sum_{i>j} Q_sym[i,j] X[i,j]
                        # we already represent X via lower triangle X[i,j] with i>=j,
                        # so combined coef on Y[i+1,j+1] for i>j is 2*Q_sym[i,j],
                        # for i==j is Q_sym[i,i].  However, _coeffs_for_constraint
                        # splits the off-diagonal (i>j) by 0.5 internally; so we
                        # must pass the full off-diagonal coefficient (= 2*Q_sym[i,j])
                        # divided by what _coeffs would do. To keep things simple,
                        # we pass:
                        #   X_coeffs[(i,i)] = Q_sym[i,i]      (matches diagonal)
                        #   X_coeffs[(i,j)] = 2*Q_sym[i,j]    (i > j)
                        pass
                # Build X_coeffs cleanly:
                Xc = {}
                for ii in range(d):
                    Xc[(ii, ii)] = float(Q_sym[ii, ii])
                    for jj in range(ii):
                        Xc[(ii, jj)] = 2.0 * float(Q_sym[ii, jj])
                sI, sJ, sV = _coeffs_for_constraint(d, float(k_val),
                                                   np.array(g_vec, dtype=float),
                                                   Xc)
                _add_lin_con(sI, sJ, sV, mosek.boundkey.up,
                             -1.0e30, float(thr_val))

            # === Solve ===
            try:
                task.optimize()
            except mosek.Error as e:
                if return_status:
                    return False, f"optimize-error: {e}"
                return False

            # Check IPM solution status.
            try:
                solsta = task.getsolsta(mosek.soltype.itr)
            except mosek.Error:
                if return_status:
                    return False, "getsolsta-error"
                return False

            # MOSEK status names map directly to enums.
            # PRIMAL_INFEASIBLE_CER => SDP relaxation is infeasible => prune.
            # Anything else (OPTIMAL, UNKNOWN, DUAL_INFEASIBLE_CER, etc.) =>
            # do not certify.
            if solsta == mosek.solsta.prim_infeas_cer:
                if return_status:
                    return True, "primal-infeasible-cer"
                return True
            if return_status:
                return False, f"solsta={solsta}"
            return False


# ---------------------------------------------------------------------------
# Optional: small CLI for direct testing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test: a parent-shape that should typically be infeasible
    # for relaxed SDP at small d.
    import time

    d_p = 4
    parent = np.array([5, 10, 10, 5], dtype=np.int32)
    lo = np.array([0, 0, 0, 0], dtype=np.int32)
    hi = np.array([10, 10, 10, 10], dtype=np.int32)
    n_half_child = 4
    m = 5
    c_target = 1.20

    t0 = time.time()
    res, status = sdp_certify_parent_mosek(parent, lo, hi, n_half_child,
                                           m, c_target, return_status=True)
    elapsed = time.time() - t0
    print(f"Result={res}  status={status}  time={elapsed*1000:.1f} ms")

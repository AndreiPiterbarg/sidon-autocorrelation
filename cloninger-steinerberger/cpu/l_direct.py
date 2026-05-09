"""Direct-MOSEK Task API path for variant L (Shor SDP) — bypasses CVXPY.

The default `_L_bench.prune_L_one(solver='MOSEK')` goes through
CVXPY -> ECOS / MOSEK with cone canonicalization.  Profiling
(`_smoke_profile_chain.json`) shows that on per-composition SDPs
**~84 % of wall time is CVXPY canonicalization** and only ~1 % is MOSEK
proper at d=6.  This module skips CVXPY entirely by encoding the same
Shor SDP directly via `mosek.Task`, with optional `mosek.Env` reuse
across calls.

Empirical (`_smoke_mosek_direct.json` at d=6, n_half=3, m=10, c=1.28,
172 SDPs):
    CVXPY+MOSEK : 322.6 ms median, 58.96 s total
    direct, fresh env : 38.1 ms median, 6.45 s total  (8.5x)
    direct, shared env: 17.0 ms median, 2.84 s total  (20.7x)

Soundness: only `solsta == prim_infeas_cer` (Farkas certificate) is
counted as a prune — same criterion as `_L_bench._shor_feasibility`.
Prune sets match exactly between CVXPY and direct-MOSEK paths (verified
on the 172-SDP smoke).

The SDP encoding mirrors `_L_bench._shor_feasibility` (lines 129–222):
  * PSD bar variable Y in S^{d+1}_+, with Y[0,0]=1, Y[i+1,0]=x_i,
    Y[i+1,j+1]=X[i,j].
  * Box: lo[i] <= x_i <= hi[i], with lo = max(0, c-1), hi = c+1.
  * Sum: 1·x = 4nm.
  * Diagonal McCormick on X[i,i]: lo^2 <= X[i,i] <= hi^2 plus 3 linear cuts.
  * Off-diagonal RLT (4 cuts per i<j pair).
  * Window: Tr(A_W X) <= 4n·ell·(c_target·m^2 + eps_margin·m^2).
"""
from __future__ import annotations

import os
import numpy as np

try:
    import mosek
    _MOSEK_AVAILABLE = True
except Exception:
    _MOSEK_AVAILABLE = False
    mosek = None  # type: ignore


def _coeffs(d, alpha_const, x_coef, X_coef_lower):
    """Build sparse lower-triangle (subi, subj, val) for the (d+1)x(d+1)
    bar matrix `A` so that <A, Y> matches:
        alpha_const + sum_i x_coef[i] x_i
                    + sum_{(i,j), i>=j} X_coef[i,j] X[i,j].
    Layout: Y[0,0]=1, Y[i+1,0]=x_i, Y[i+1,j+1]=X[i,j].
    Recall <A,Y> = sum_i A[i,i] Y[i,i] + 2 sum_{i>j} A[i,j] Y[i,j].
    """
    subi, subj, val = [], [], []
    if alpha_const != 0.0:
        subi.append(0); subj.append(0); val.append(float(alpha_const))
    for i in range(d):
        c = x_coef[i]
        if c != 0.0:
            subi.append(i + 1); subj.append(0); val.append(0.5 * float(c))
    for (i, j), c in X_coef_lower.items():
        if c == 0.0:
            continue
        if i == j:
            subi.append(i + 1); subj.append(j + 1); val.append(float(c))
        else:
            ii, jj = (i, j) if i > j else (j, i)
            subi.append(ii + 1); subj.append(jj + 1); val.append(0.5 * float(c))
    return subi, subj, val


def _make_cell(c_int, m):
    """Cell box: lo = max(0, c-1), hi = c+1.  Identical to _L_bench._make_cell."""
    c = np.asarray(c_int, dtype=np.float64)
    lo = np.maximum(0.0, c - 1.0)
    hi = c + 1.0
    return lo, hi


def prune_L_direct(c_int, A_mats, windows, n_half, m, c_target,
                    env=None, tol=1e-9, eps_margin=1e-9,
                    lo_override=None, hi_override=None,
                    add_trace_cuts=True):
    """Direct MOSEK Task implementation of variant L Shor SDP.

    Args:
        c_int : composition vector (length d, integer).
        A_mats : list of d×d window indicator matrices A_W.
        windows : list of (ell, s_lo) tuples (parallel to A_mats).
        n_half, m, c_target : usual cascade parameters.
        env : optional pre-allocated `mosek.Env`.  If None, a fresh Env is
            created and torn down per call (slower).  Pass a shared env
            across calls for the 2-3× extra speedup over per-call env.
        tol : MOSEK interior-point tolerance.
        eps_margin : sound conservative margin on each window threshold;
            the SDP is RELAXED by `eps_margin·m²` so any infeasibility
            certificate is rigorous by margin.
        lo_override, hi_override : optional explicit box (length-d arrays).
            If both provided, the parent-cell `_make_cell(c_int, m)` box is
            replaced with this explicit one.  Used for SUB-CELL splits in
            `_smoke_split_cell_SDP_optimal.py`.  Soundness: a smaller box
            is a tighter relaxation, so SDP-infeasible on a sub-cell
            implies infeasibility on that sub-cell only (NOT the parent).

    Returns:
        (pruned: bool, status_str).  `pruned` iff
        `solsta == mosek.solsta.prim_infeas_cer` (Farkas certificate).
    """
    if not _MOSEK_AVAILABLE:
        raise RuntimeError(
            "MOSEK is not importable; install `mosek` to use prune_L_direct.")

    d = len(c_int)
    bar_dim = d + 1
    if lo_override is not None and hi_override is not None:
        lo = np.asarray(lo_override, dtype=np.float64)
        hi = np.asarray(hi_override, dtype=np.float64)
    else:
        lo, hi = _make_cell(c_int, m)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    own_env = env is None
    if own_env:
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass

    try:
        with env.Task(0, 0) as task:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, tol)
            task.putintparam(mosek.iparam.intpnt_max_iterations, 200)
            task.putintparam(mosek.iparam.log, 0)
            task.putintparam(mosek.iparam.num_threads, 1)

            task.appendbarvars([bar_dim])
            task.putobjsense(mosek.objsense.minimize)

            def _add(subi, subj, vals, bk, blk, buk):
                cidx = task.getnumcon()
                task.appendcons(1)
                if len(subi) > 0:
                    aid = task.appendsparsesymmat(bar_dim, subi, subj, vals)
                    task.putbaraij(cidx, 0, [aid], [1.0])
                task.putconbound(cidx, bk, blk, buk)
                return cidx

            sI, sJ, sV = _coeffs(d, 1.0, np.zeros(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

            for i in range(d):
                xc = np.zeros(d); xc[i] = 1.0
                sI, sJ, sV = _coeffs(d, 0.0, xc, {})
                _add(sI, sJ, sV, mosek.boundkey.ra, lo[i], hi[i])

            sI, sJ, sV = _coeffs(d, 0.0, np.ones(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, nm, nm)

            if add_trace_cuts:
                # Trace identity (option C cut #1): for any rank-1 X = xx^T with
                # 1·x = nm, we have 1^T X 1 = (1·x)^2 = nm^2.  In the relaxation
                # X ⪰ x x^T, only `1^T X 1 ≥ nm^2` is implied.  Adding
                # `1^T X 1 = nm^2` excludes non-rank-1 PSD lifts that wouldn't
                # correspond to any real x; sound (rank-1 cases unaffected).
                #
                # 1^T X 1 = Σ_i X_ii + 2 Σ_{i>j} X_ij.  In the lower-triangle
                # encoding used by `_coeffs`, off-diagonal coef = 2 maps to
                # MOSEK val = 1.0 (which contributes 2*Y_ij to <A,Y>).
                Xc_trace = {}
                for ii in range(d):
                    Xc_trace[(ii, ii)] = 1.0
                    for jj in range(ii):
                        Xc_trace[(ii, jj)] = 2.0
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc_trace)
                _add(sI, sJ, sV, mosek.boundkey.fx, nm * nm, nm * nm)

                # Cauchy-Schwarz lower bound (option C cut #2):
                #     Σ_i x_i^2 ≥ (Σ x_i)^2 / d = nm^2 / d.
                # In the SDP, X_ii ≥ x_i^2 (PSD-implied), so Σ X_ii ≥ Σ x_i^2 ≥
                # nm^2 / d.  Adds a tighter LB on Σ X_ii than the per-coord
                # `lo[i]^2` cumulative bound when the cell is symmetric.
                Xc_diag = {(ii, ii): 1.0 for ii in range(d)}
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc_diag)
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     (nm * nm) / float(d), 0.0)

            for i in range(d):
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.ra,
                     lo[i] * lo[i], hi[i] * hi[i])
                xc = np.zeros(d); xc[i] = -2.0 * lo[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -lo[i] * lo[i], 0.0)
                xc = np.zeros(d); xc[i] = -2.0 * hi[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo,
                     -hi[i] * hi[i], 0.0)
                xc = np.zeros(d); xc[i] = -(lo[i] + hi[i])
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.up, 0.0, -lo[i] * hi[i])

            for i in range(d):
                for j in range(i + 1, d):
                    li, lj = lo[i], lo[j]
                    ui, uj = hi[i], hi[j]
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -li
                    sI, sJ, sV = _coeffs(d, li * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(d, ui * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -li
                    sI, sJ, sV = _coeffs(d, li * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(d, ui * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

            for A_mat, (ell, _) in zip(A_mats, windows):
                thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
                Xc = {}
                for ii in range(d):
                    Xc[(ii, ii)] = float(A_mat[ii, ii])
                    for jj in range(ii):
                        Xc[(ii, jj)] = 2.0 * float(A_mat[ii, jj])
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.up, -1e30, thr)

            try:
                task.optimize()
            except mosek.Error as e:
                return False, f"optimize-error: {e}"

            try:
                solsta = task.getsolsta(mosek.soltype.itr)
            except mosek.Error:
                return False, "getsolsta-error"

            if solsta == mosek.solsta.prim_infeas_cer:
                return True, "infeasible"
            if solsta == mosek.solsta.optimal:
                return False, "optimal"
            return False, f"solsta={solsta}"
    finally:
        if own_env:
            try:
                env.__exit__(None, None, None)
            except Exception:
                pass

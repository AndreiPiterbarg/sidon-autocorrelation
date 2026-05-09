"""FAST split-cell SDP for L-survivors at d=10 (n=5, m=5, c=1.28).

Optimizations vs `_smoke_split_cell_SDP.py`:
============================================
(1) Direct MOSEK Task API (sibling `prune_L_direct_box` w/ explicit lo, hi).
    Bypasses CVXPY canonicalization (84% of L SDP wall in baseline).
(2) Per-worker `mosek.Env` reuse (created once in worker init).
(3) Sub-cell pre-screens (microseconds each, all sound):
       (a) box-sum: lo.sum() <= 4nm <= hi.sum()
       (b) F-bound on sub-cell: tightened |δ|_∞ ≤ 1/m, σ-restricted polytope
(4) Smart sigma ordering for early termination: σ patterns whose box mean-sum
    is far from 4nm are likely INFEASIBLE => try first to maximize early-prune
    count.  Wait — for non-prunable survivors we want a feasible sub-cell ASAP,
    so order σ by mean closeness to 4nm; for prunable parents, order doesn't
    matter (must do all 1024).  Compromise: do the *closest-to-4nm* first.
(5) Multiprocessing.Manager().Event() flag for early termination across workers.
(6) Sub-cells of ONE survivor distributed across ALL workers (not one survivor
    per worker).  Each worker reuses Env across sub-cells.

SOUNDNESS:
==========
- The sub-cell SDP itself is identical in structure to the baseline's
  `_shor_feasibility`, just with explicit lo/hi instead of `_make_cell(c, m)`.
- Pre-screen (a) is exact: box has sum-LP infeasible iff sum(lo) > 4nm or
  sum(hi) < 4nm (continuous relaxation of box+sum constraint).
- Pre-screen (b) F-bound is sound: a tighter bound on the same sup_{δ in
  sub-cell polytope} (TV_W(b) - TV_W(a)) than parent F.  If F-bound says
  "no W has m^2 TV_W(b) - corr_F^σ(W) > c_target m^2", that does NOT prune
  the sub-cell — the F-bound is an UPPER bound on TV_W(b)-TV_W(a), so its
  failure to prune does not let us prune.  But: if for every δ in the
  sub-cell polytope, EXISTS W with m^2 TV_W(b) - δ-bound(W) > c_target m^2,
  the sub-cell IS pruned.  Equivalently: pre-screen (b) prunes if
       inf_{σ-poly δ} max_W [m^2 TV_W(b) - 2 δ^T B^W + δ^T A^W δ] > c_target m^2
  Bounding δ^T A^W δ by h^2 #pairs makes the LHS LP-tractable; we use the
  closed-form on the σ-restricted polytope.

USAGE: python _smoke_split_cell_SDP_fast.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from _Q_bench import _build_windows
from _L_bench import _build_A_matrices

try:
    import mosek
    _MOSEK_AVAILABLE = True
except Exception:
    _MOSEK_AVAILABLE = False
    mosek = None  # type: ignore


# =====================================================================
# Direct MOSEK Task API for sub-cell SDP feasibility (mirror of
# l_direct.prune_L_direct but with explicit (lo, hi) box).
# =====================================================================
def _coeffs(d, alpha_const, x_coef, X_coef_lower):
    """Build sparse lower-triangle (subi, subj, val) for the (d+1)x(d+1)
    bar matrix `A` so that <A, Y> matches:
        alpha_const + sum_i x_coef[i] x_i
                    + sum_{(i,j), i>=j} X_coef[i,j] X[i,j].
    Identical to l_direct._coeffs (fixed in-repo).
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


def prune_L_direct_box(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                        env=None, tol=1e-9, eps_margin=1e-9):
    """Direct MOSEK Task variant L Shor SDP with explicit (lo, hi) box.

    Sibling of `cloninger-steinerberger/cpu/l_direct.prune_L_direct` that
    accepts an arbitrary axis-aligned box rather than the parent box
    `lo=max(0,c-1), hi=c+1`.  Used for sub-cell SDPs.

    Returns: (pruned: bool, status_str).  `pruned` iff
        `solsta == mosek.solsta.prim_infeas_cer` (Farkas certificate).
    """
    if not _MOSEK_AVAILABLE:
        raise RuntimeError("MOSEK is not importable; install `mosek`.")

    d = len(c_int)
    bar_dim = d + 1
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
            # Aggressive presolve: speeds up infeasibility detection.
            task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)

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

            # Y[0,0] = 1
            sI, sJ, sV = _coeffs(d, 1.0, np.zeros(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

            # Box: lo <= x <= hi
            for i in range(d):
                xc = np.zeros(d); xc[i] = 1.0
                sI, sJ, sV = _coeffs(d, 0.0, xc, {})
                _add(sI, sJ, sV, mosek.boundkey.ra, lo[i], hi[i])

            # sum x = 4nm
            sI, sJ, sV = _coeffs(d, 0.0, np.ones(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, nm, nm)

            # Diagonal McCormick on X[i,i]
            for i in range(d):
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.ra,
                     lo[i] * lo[i], hi[i] * hi[i])
                xc = np.zeros(d); xc[i] = -2.0 * lo[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo, -lo[i] * lo[i], 0.0)
                xc = np.zeros(d); xc[i] = -2.0 * hi[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo, -hi[i] * hi[i], 0.0)
                xc = np.zeros(d); xc[i] = -(lo[i] + hi[i])
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.up, 0.0, -lo[i] * hi[i])

            # Off-diagonal RLT
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

            # Window constraints
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


# =====================================================================
# Pre-screen (b): F-bound on the σ-restricted sub-cell polytope.
# =====================================================================
def _f_bound_subcell(c_int, sigma, windows, n_half, m, c_target,
                      eps_margin=1e-9):
    """F-style bound restricted to the σ sub-cell.

    Sub-cell polytope (in δ-coords, m^2 units):
        δ_i in [0, 1/m]   if σ_i = +1
        δ_i in [-1/m, 0]  if σ_i = -1
        Σ δ_i = 0
        a = b - δ >= 0  (≡ δ_i ≤ b_i, satisfied since |δ| ≤ 1/m and b_i ≥ 0)

    For each window W, the per-window correction is at most
        corr_F^σ(W) := max_{δ in σ-poly} (1/(4n·ell_W)) [2 sum_j BB^W_j δ_j
                                                         - sum_{(i,j)∈W} δ_i δ_j]

    Bound (drop quadratic — gives upper bound):
        corr_F^σ(W) ≤ corr_lin^σ(W) + corr_quad(W)
    where
        corr_quad(W) = h² ell_int_sum^W / (4n·ell_W),  h = 1/m
                      [same as F's parent quadratic term]
        corr_lin^σ(W) = (1/(2n·ell_W)) · [LP optimum]
        LP: max 2 δ^T BB^W   s.t.  δ in σ-poly.
        Closed form for LP:
            σ-poly is a translated unit cube intersected with sum=0.
            Decompose δ = δ' where δ'_i = σ_i (h - σ_i δ_i)/2 — actually,
            simpler: substitute u_i = δ_i if σ_i=+1 (u_i in [0,h]), or
            u_i = -δ_i if σ_i=-1 (u_i in [0,h]).  Then Σ σ_i u_i = 0 and
            objective = 2 Σ σ_i BB^W_j u_i.

    Returns:
        (pruned: bool, status_str).  `pruned` iff there exists W with
        m^2 TV_W(b) - corr_lin^σ(W) - corr_quad(W) > c_target m^2 + eps.

    Note: the quadratic relaxation here is LOOSER than what L's SDP enforces.
    F-bound only catches the same things parent-F catches under σ-restriction;
    in practice, since we're already at the L-survivor stage, this rarely
    prunes — but it's microseconds, so worth running.
    """
    d = len(c_int)
    n_d = float(n_half)
    h = 1.0 / float(m)
    cs_m2 = c_target * m * m
    eps_thr = eps_margin * m * m

    # Compute autoconv windows (ws_W) and BB^W_j arrays.  Same as
    # _composition_window_data, inline to avoid import overhead.
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c_int[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj
    prefix_c = np.zeros(d + 1, dtype=np.int64)
    for i in range(d):
        prefix_c[i + 1] = prefix_c[i] + int(c_int[i])

    sigma = np.asarray(sigma, dtype=np.int8)

    for (ell, s_lo) in windows:
        n_cv = ell - 1
        s_hi = s_lo + n_cv - 1
        ws = int(np.sum(conv[s_lo:s_lo + n_cv]))

        # BB^W_j  (integer)
        BB = np.empty(d, dtype=np.int64)
        for j in range(d):
            lo_i = max(0, s_lo - j)
            hi_i = min(d - 1, s_hi - j)
            if hi_i < lo_i:
                BB[j] = 0
            else:
                BB[j] = int(prefix_c[hi_i + 1] - prefix_c[lo_i])

        # ell_int_sum (number of (i,j) pairs in W with i,j in [0,d))
        ell_int_sum = 0
        for k in range(s_lo, s_lo + n_cv):
            d_idx = abs((k + 1) - d)
            v = max(0, d - d_idx)
            ell_int_sum += v

        # m^2 TV_W(b) = ws / (4n·ell_W)
        m2_TV = float(ws) / (4.0 * n_d * float(ell))

        # Solve LP: max 2 sum_j BB_j δ_j
        # s.t. δ in σ-poly: σ_i = +1 -> δ_i in [0, h]; σ_i=-1 -> δ_i in [-h, 0];
        #      Σ δ_i = 0.
        #
        # LP closed form: substitute u_i := σ_i δ_i (so u_i in [0, h] for ALL i).
        # Then Σ σ_i u_i = 0  =>  Σ_{σ=+1} u_i = Σ_{σ=-1} u_i = some value s in
        # [0, h·d/2] (since each u_i in [0,h]).  Objective:
        #     2 Σ σ_j BB^W_j δ_j = 2 Σ_j BB^W_j σ_j δ_j
        #                         = 2 Σ_j (σ_j·BB_j) (σ_j·u_j)·σ_j
        # Wait — δ_j = σ_j u_j (since σ_j = ±1, σ_j·σ_j = 1, so u_j = σ_j δ_j
        # gives δ_j = σ_j u_j).
        # Objective: 2 Σ_j BB_j δ_j = 2 Σ_j BB_j σ_j u_j = 2 Σ_j (BB_j σ_j) u_j.
        #
        # LP: max 2 Σ_j w_j u_j  with w_j = BB_j σ_j;  u_j in [0,h];
        #     Σ_{σ=+1} u_j = Σ_{σ=-1} u_j  (linked).
        #
        # Let s := Σ_{σ=+1} u_j = Σ_{σ=-1} u_j.  Then for fixed s,
        #     positives: max Σ_{σ=+1} w_j u_j s.t. u in [0,h], Σ u = s
        #     => sort w_j (σ=+1) descending; saturate top until total mass = s.
        #     max value = h·sum_top_h - (h - r) · w_{cutoff}, with s = h·(k-1)+r.
        #     piecewise-linear concave in s.
        #     same logic for negatives (sort w_j (σ=-1) descending).
        # Total = 2(P(s) + N(s)).
        # P(s) and N(s) are concave piecewise-linear on s in [0, h·(d/2)].
        # Their sum is concave; max over s in [0, h·d/2] is attained at a
        # break-point of either P or N.
        #
        # CHEAP UPPER BOUND (used here, sound): drop the linkage Σ_+ = Σ_-.
        # Then independent: each u_j chosen as h·1[w_j > 0].
        # max LP_relax = 2 h · Σ_j max(0, w_j) = 2 h · Σ_j max(0, BB_j σ_j)
        # This is an UPPER BOUND on true LP, so corr_lin <= LP_relax/(2n·ell).
        # The pruning test  m2_TV - corr - cs_m2 > eps  uses an upper bound on
        # corr (which is what we want — it gives a LOWER bound on excess, so
        # prune is sound).  Wait NO: we need an upper bound on corr, but
        # LP_relax is an upper bound on TRUE LP, hence upper bound on corr.
        # Then m2_TV - LP_relax_corr is a LOWER bound on m2_TV - corr.  But
        # we want to prune when m2_TV - corr > cs_m2; the pruning test
        #     m2_TV - LP_relax_corr > cs_m2
        # is therefore SOUND but possibly weaker (we miss prunes where
        # true corr < LP_relax_corr but the test fails).  Since the goal of
        # the pre-screen is to skip an SDP, weaker is OK; we just lose
        # speedup, never soundness.

        # w_j = BB_j * σ_j
        w = BB.astype(np.float64) * sigma.astype(np.float64)
        # LP_relax = 2 h Σ_j max(0, w_j)
        lp_relax = 2.0 * h * float(np.sum(np.maximum(0.0, w)))
        corr_lin = lp_relax / (2.0 * n_d * float(ell))
        corr_quad = (h * h) * float(ell_int_sum) / (4.0 * n_d * float(ell))
        corr = corr_lin + corr_quad

        if m2_TV - corr > cs_m2 + eps_thr:
            return True, f'F_subcell_W=({ell},{s_lo})'

    return False, 'F_subcell_no_prune'


# =====================================================================
# Sub-cell builder + smart sigma ordering
# =====================================================================
def _build_subcell_box(c_int, sigma):
    """Return (lo, hi) for sub-cell σ given parent integer composition c."""
    d = len(c_int)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        ci = float(c_int[i])
        if sigma[i] > 0:
            lo[i] = max(0.0, ci - 1.0)
            hi[i] = ci
        else:
            lo[i] = ci
            hi[i] = ci + 1.0
    return lo, hi


def _sigma_priority(c_int, sigma, four_nm):
    """Priority for σ ordering.  Lower = more likely FEASIBLE => try first
    when looking for early termination.

    For σ at parent c: the box has center sum =
        Σ (c_i - 0.5)·1[σ_i=+1] + (c_i + 0.5)·1[σ_i=-1]
        = Σ c_i + 0.5·(#(-1) - #(+1))
        = 4nm + 0.5·(d - 2·#(+1))
    For 4nm to be inside [sum(lo), sum(hi)] = [4nm - sum_{σ=+1}, 4nm + sum_{σ=-1}]
    where the ±0.5 etc, the box mean is FURTHER from 4nm when σ has very
    unbalanced ± counts.  Best: σ has equal #(+1) = #(-1) = d/2 (balanced).

    Priority = absolute deviation of mean-sum from 4nm = 0.5·|d - 2·#(+1)|.
    Ties broken by sum of |c_i - 0.5·σ_i| ... (ad hoc).
    """
    d = len(sigma)
    n_pos = int(np.sum(np.asarray(sigma) > 0))
    return 0.5 * abs(d - 2 * n_pos)


# =====================================================================
# Worker for processing sub-cells with cached env, A_mats, etc.
# =====================================================================
_WORKER_STATE = {}


def _worker_init(stop_event):
    """Initializer: import path setup; build mosek.Env once; cache windows.

    We pin BLAS threads to 1 (avoids MOSEK<->BLAS thread contention when
    running 12 workers each with num_threads=1).
    """
    global _WORKER_STATE
    import sys as _sys, os as _os
    # Pin BLAS to single-thread BEFORE numpy imports.  Already imported
    # in main but worker has fresh interpreter; safe to set here.
    _os.environ['OMP_NUM_THREADS'] = '1'
    _os.environ['OPENBLAS_NUM_THREADS'] = '1'
    _os.environ['MKL_NUM_THREADS'] = '1'
    _os.environ['NUMEXPR_NUM_THREADS'] = '1'

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _here)
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger'))
    _sys.path.insert(0, _os.path.join(_here, 'cloninger-steinerberger', 'cpu'))

    import mosek as _mosek
    env = _mosek.Env()
    try:
        env.checkoutlicense(_mosek.feature.pton)
    except Exception:
        pass
    _WORKER_STATE['env'] = env
    _WORKER_STATE['stop_event'] = stop_event


def _worker_one_subcell(args):
    """Worker: test one sub-cell.

    Args (tuple): (sigma_idx, c_int_list, sigma_list, n_half, m, c_target)
    Returns: (sigma_idx, sub_pruned: bool, status: str, t_solve: float,
              skipped_via_prescreen: str)
    """
    global _WORKER_STATE
    sigma_idx, c_int_list, sigma_list, n_half, m, c_target = args

    # Check stop event (set by main process when feasible sub-cell found)
    stop_event = _WORKER_STATE.get('stop_event')
    if stop_event is not None and stop_event.is_set():
        return sigma_idx, True, 'aborted', 0.0, 'aborted'

    import numpy as _np
    from _Q_bench import _build_windows
    from _L_bench import _build_A_matrices

    c_int = _np.asarray(c_int_list, dtype=_np.int32)
    sigma = _np.asarray(sigma_list, dtype=_np.int8)
    d = len(c_int)
    nm = float(4 * n_half * m)

    # Cache windows + A_mats per worker
    cache_key = ('wA', d)
    if cache_key not in _WORKER_STATE:
        windows, _ = _build_windows(d)
        A_mats = _build_A_matrices(d, windows)
        _WORKER_STATE[cache_key] = (windows, A_mats)
    windows, A_mats = _WORKER_STATE[cache_key]

    lo, hi = _build_subcell_box(c_int, sigma)

    # Pre-screen #1: box-sum (microseconds)
    s_lo = float(_np.sum(lo))
    s_hi = float(_np.sum(hi))
    if nm < s_lo - 1e-9 or nm > s_hi + 1e-9:
        return sigma_idx, True, 'box_sum_pre_inf', 0.0, 'box_sum'

    # Pre-screen #2: F-bound on sub-cell (microseconds)
    pruned_F, _ = _f_bound_subcell(c_int, sigma, windows, n_half, m, c_target)
    if pruned_F:
        return sigma_idx, True, 'F_subcell_pre_inf', 0.0, 'F_subcell'

    # SDP (with shared env)
    env = _WORKER_STATE['env']
    t0 = time.time()
    try:
        pruned, status = prune_L_direct_box(
            c_int, lo, hi, A_mats, windows, n_half, m, c_target,
            env=env, tol=1e-9, eps_margin=1e-9)
    except Exception as e:
        return sigma_idx, False, f'EXC:{type(e).__name__}', time.time() - t0, 'sdp'
    t_solve = time.time() - t0
    return sigma_idx, bool(pruned), str(status), t_solve, 'sdp'


def _worker_chunk_subcells(args):
    """Worker: test a CHUNK of sub-cells sequentially.

    Reduces dispatch overhead vs one task per sub-cell.  Each worker reuses
    the same env, windows, A_mats across the chunk.

    Args (tuple): (chunk_id, c_int_list, sigma_list_batch, n_half, m, c_target)
        sigma_list_batch : list of (sigma_idx, sigma_list)
    Returns: list of per-sub-cell results, in order:
        (sigma_idx, sub_pruned, status, t_solve, prescreen)
    """
    global _WORKER_STATE
    chunk_id, c_int_list, sigma_list_batch, n_half, m, c_target = args

    import numpy as _np
    from _Q_bench import _build_windows
    from _L_bench import _build_A_matrices

    stop_event = _WORKER_STATE.get('stop_event')

    c_int = _np.asarray(c_int_list, dtype=_np.int32)
    d = len(c_int)
    nm = float(4 * n_half * m)

    cache_key = ('wA', d)
    if cache_key not in _WORKER_STATE:
        windows, _ = _build_windows(d)
        A_mats = _build_A_matrices(d, windows)
        _WORKER_STATE[cache_key] = (windows, A_mats)
    windows, A_mats = _WORKER_STATE[cache_key]
    env = _WORKER_STATE['env']

    out = []
    for sigma_idx, sigma_list in sigma_list_batch:
        if stop_event is not None and stop_event.is_set():
            out.append((sigma_idx, True, 'aborted', 0.0, 'aborted'))
            continue
        sigma = _np.asarray(sigma_list, dtype=_np.int8)
        lo, hi = _build_subcell_box(c_int, sigma)
        s_lo = float(_np.sum(lo))
        s_hi = float(_np.sum(hi))
        if nm < s_lo - 1e-9 or nm > s_hi + 1e-9:
            out.append((sigma_idx, True, 'box_sum_pre_inf', 0.0, 'box_sum'))
            continue
        pruned_F, _ = _f_bound_subcell(c_int, sigma, windows, n_half, m,
                                          c_target)
        if pruned_F:
            out.append((sigma_idx, True, 'F_subcell_pre_inf', 0.0, 'F_subcell'))
            continue
        t0 = time.time()
        try:
            pruned, status = prune_L_direct_box(
                c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                env=env, tol=1e-9, eps_margin=1e-9)
        except Exception as e:
            t_solve = time.time() - t0
            out.append((sigma_idx, False, f'EXC:{type(e).__name__}', t_solve, 'sdp'))
            # If it's a feasible-finding event, set stop on early-term.  But
            # we don't know early_terminate at worker level; the main loop
            # handles cancellation via its own logic.
            continue
        t_solve = time.time() - t0
        out.append((sigma_idx, bool(pruned), str(status), t_solve, 'sdp'))
        # If this sub-cell is FEASIBLE (not pruned), the parent is NOT
        # split-prunable.  Set stop_event so other workers exit chunk-loops
        # early.  Sound: feasibility is decided by SDP solver returning
        # `optimal` (we don't claim pruned without `prim_infeas_cer`).
        if not pruned and stop_event is not None:
            stop_event.set()
    return out


# =====================================================================
# Main split-prune driver: distribute sub-cells of ONE survivor across all
# workers using a smart-ordered task queue + early termination via Event.
# =====================================================================
def split_prune_one_fast(c_int, n_half, m, c_target, n_workers=12,
                          early_terminate=True, verbose=True,
                          worker_pool=None, manager=None):
    """Test split-pruning of one composition with FAST optimizations.

    Args:
        c_int : composition vector (length d, sum 4nm).
        n_half, m, c_target : usual cascade params.
        n_workers : process pool size.
        early_terminate : abort sub-cell SDPs as soon as a feasible sub-cell
            is found.
        worker_pool : pre-existing ProcessPoolExecutor (reuse across survivors
            to avoid re-init cost).
        manager : pre-existing multiprocessing.Manager.

    Returns:
        (split_pruned: bool, n_inf: int, n_total_done: int, t_total: float,
         feasible_sigma: optional list, statuses: dict, presceen_stats: dict)
    """
    d = len(c_int)
    sigmas = list(product([1, -1], repeat=d))
    n_total = len(sigmas)
    c_int_list = c_int.tolist()
    nm = float(4 * n_half * m)

    # Smart sigma ordering: closer to balanced (#+1 = d/2) tried first.
    # This means: when the parent IS split-prunable, all sigmas need running
    # anyway; ordering doesn't matter.  When NOT split-prunable, the feasible
    # σ is most often one with balanced ±, so try those first.
    sigma_priorities = [(idx, _sigma_priority(c_int, list(s), nm))
                         for idx, s in enumerate(sigmas)]
    # Sort ascending: most balanced first.
    sigma_priorities.sort(key=lambda x: x[1])
    ordered_sigma_indices = [p[0] for p in sigma_priorities]

    own_pool = worker_pool is None
    own_manager = manager is None
    if own_manager:
        manager = mp.Manager()
    stop_event = manager.Event()

    if own_pool:
        worker_pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init, initargs=(stop_event,))

    # Chunk sub-cells to reduce dispatch overhead.  We want enough chunks
    # to keep all workers busy but not so many that overhead dominates.
    # 4-8 chunks per worker is a good balance.
    n_chunks = max(n_workers * 4, 32)
    chunk_size = max(1, (n_total + n_chunks - 1) // n_chunks)
    chunk_args = []
    for ci in range(0, n_total, chunk_size):
        batch = [(idx, list(sigmas[idx]))
                  for idx in ordered_sigma_indices[ci:ci + chunk_size]]
        chunk_args.append((ci // chunk_size, c_int_list, batch,
                            n_half, m, c_target))

    t0 = time.time()
    n_inf = 0
    n_pre_inf_box = 0
    n_pre_inf_F = 0
    n_sdp_done = 0
    feasible_sigma = None
    statuses = {}
    completed = 0
    sdp_solve_times = []

    futs = {worker_pool.submit(_worker_chunk_subcells, args): args[0]
             for args in chunk_args}
    try:
        for fut in as_completed(futs):
            chunk_results = fut.result()
            for sigma_idx, sub_pruned, status, t_solve, prescreen in chunk_results:
                completed += 1
                statuses[status] = statuses.get(status, 0) + 1
                if prescreen == 'box_sum':
                    n_pre_inf_box += 1
                elif prescreen == 'F_subcell':
                    n_pre_inf_F += 1
                elif prescreen == 'sdp':
                    n_sdp_done += 1
                    sdp_solve_times.append(t_solve)
                if sub_pruned:
                    n_inf += 1
                else:
                    if feasible_sigma is None:
                        feasible_sigma = list(sigmas[sigma_idx])
                    if early_terminate:
                        stop_event.set()
            if feasible_sigma is not None and early_terminate:
                # Stop dispatching new chunks; pending workers will see
                # stop_event and exit early.
                for f in futs:
                    if not f.done():
                        f.cancel()
                break
            if verbose and completed >= 256 and (completed // 256) > (
                    (completed - len(chunk_results)) // 256):
                avg_sdp = (sum(sdp_solve_times) / max(1, len(sdp_solve_times))
                            * 1000) if sdp_solve_times else 0.0
                print(f"      ... {completed}/{n_total} sub-cells   "
                      f"(inf={n_inf}, pre={n_pre_inf_box}+{n_pre_inf_F}, "
                      f"sdp={n_sdp_done}, sdp_avg={avg_sdp:.1f}ms, "
                      f"{time.time()-t0:.1f}s)")
    finally:
        if own_pool:
            try:
                worker_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    t_total = time.time() - t0
    split_pruned = (feasible_sigma is None)
    presceen_stats = {
        'n_pre_inf_box_sum': n_pre_inf_box,
        'n_pre_inf_F_subcell': n_pre_inf_F,
        'n_sdp_done': n_sdp_done,
        'sdp_avg_ms': float(sum(sdp_solve_times) / max(1, len(sdp_solve_times))
                              * 1000) if sdp_solve_times else 0.0,
    }
    return (split_pruned, n_inf, completed, t_total, feasible_sigma,
            statuses, presceen_stats)


# =====================================================================
# Driver
# =====================================================================
def main():
    n_half = 5
    m = 5
    c_target = 1.28
    d = 2 * n_half
    n_workers = int(os.environ.get('N_WORKERS', max(1, min(12, (os.cpu_count() or 4) - 2))))
    print(f"Solver: MOSEK direct, n_workers: {n_workers}")

    # Step 1: load cached L-survivors
    cache = os.path.join(_HERE, '_smoke_split_cell_l_survivors.json')
    if not os.path.exists(cache):
        print(f"ERROR: cache file {cache} not found.  Run "
              f"_smoke_split_cell_SDP.py first to generate L-survivors.")
        return
    with open(cache) as fp:
        cache_data = json.load(fp)
    if not (cache_data.get('n_half') == n_half and cache_data.get('m') == m
             and cache_data.get('c_target') == c_target):
        print(f"ERROR: cache mismatch: {cache_data}")
        return
    l_survivors = [np.asarray(c, dtype=np.int32)
                    for c in cache_data['l_survivors']]
    print(f"\n[1] Loaded {len(l_survivors)} cached L-survivors")

    if len(l_survivors) == 0:
        print("No L-survivors; nothing to split-prune.")
        return

    # Step 2: split-prune each survivor.  Reuse worker pool + Manager across
    # survivors to amortize init cost.
    print(f"\n[2] Splitting cells: 2^{d} = {2**d} sub-cells per survivor "
          f"({n_workers} parallel workers, shared per-worker env)")
    manager = mp.Manager()
    # Per-survivor we'll create a new stop_event; init pool once.

    # The stop_event in worker_init is bound when init runs; for cross-survivor
    # reuse we must reset between calls.  Workaround: spawn a fresh pool per
    # survivor (init cost ~0.3s, dominated by SDP cost anyway).  This avoids
    # event-leak between survivors.

    results = []
    n_split_pruned = 0
    t_global = time.time()

    for i, c_int in enumerate(l_survivors):
        print(f"\n  Survivor {i+1}/{len(l_survivors)}: c={c_int.tolist()}")
        sp_pruned, n_inf, n_done, t_one, feasible_sigma, statuses, prescreen = (
            split_prune_one_fast(c_int, n_half, m, c_target,
                                  n_workers=n_workers,
                                  early_terminate=True,
                                  verbose=True,
                                  manager=manager))
        if sp_pruned:
            n_split_pruned += 1
            print(f"    SPLIT-PRUNED: {n_inf}/{n_done} sub-cells INFEASIBLE  "
                  f"(no feasible sub-cell)  ({t_one:.1f}s)")
        else:
            print(f"    NOT split-pruned: feasible σ={feasible_sigma}  "
                  f"({n_inf}/{n_done} sub-cells INFEASIBLE before stop)  "
                  f"({t_one:.1f}s)")
        print(f"    statuses: {statuses}")
        print(f"    prescreen: {prescreen}")
        results.append({
            'c_int': c_int.tolist(),
            'split_pruned': bool(sp_pruned),
            'n_sub_infeasible': int(n_inf),
            'n_sub_tested': int(n_done),
            'time_s': float(t_one),
            'feasible_sigma': feasible_sigma,
            'statuses': statuses,
            'prescreen_stats': prescreen,
        })
        elapsed = time.time() - t_global
        print(f"    [running total: {n_split_pruned}/{i+1} split-pruned, "
              f"{elapsed:.1f}s elapsed]")

        if elapsed > 25 * 60:
            print(f"\n  WALL-TIME LIMIT (25 min) -- stop after survivor {i+1}")
            break

    print(f"\n\n=========================================================")
    print(f"FINAL: split-prunes {n_split_pruned} of {len(results)} L-survivors "
          f"(out of {len(l_survivors)} total)")
    print(f"Total time: {time.time() - t_global:.1f}s")
    print(f"=========================================================\n")

    out_path = os.path.join(_HERE, '_smoke_split_cell_SDP_fast.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
            'solver': 'MOSEK_direct',
            'n_workers': n_workers,
            'n_l_survivors_total': len(l_survivors),
            'n_l_survivors_tested': len(results),
            'n_split_pruned': n_split_pruned,
            'total_time_s': float(time.time() - t_global),
            'results': results,
        }, fp, indent=2)
    print(f"Wrote {out_path}")
    return n_split_pruned, len(results)


if __name__ == '__main__':
    mp.freeze_support()  # required on Windows
    main()

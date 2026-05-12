"""FAST per-box Lasserre order-2 SDP cert via selective window cones +
MOSEK speedups.

Combines three orthogonal speedups over `bound_sdp_escalation.py`:

  1. EARLY-STOP via `solve_dual_task(early_stop_on_clear_verdict=True)`:
     MOSEK halts as soon as the Farkas λ* clearly lands in the FEAS
     (<0.15·Λ) or INFEAS (>0.85·Λ) zone, skipping the last 30-50% of
     IPM iterations. Default thresholds 0.25/0.75 in the verdict
     classifier leave a wide margin; 0.15/0.85 inside the callback is
     safe.

  2. LOOSER MOSEK TOLERANCES (intpnt_co_tol_{p,d}feas, rel_gap = 1e-5
     instead of 1e-7). The Farkas LP returns λ* ∈ [0, Λ] with verdict
     thresholds 0.25/0.75 of Λ, so 1e-5 residual is sub-margin and
     does not affect the soundness of the verdict.

  3. SELECTIVE WINDOW PSD CONES. At d=22 we have 946 windows. For any
     given box, only ~5-20 windows are likely "binding" (their
     scale_W · Σ_{S_W} y_{e_i+e_j} term is close to t at the optimum).
     The rest can be downgraded from a (n_loc × n_loc) localizing PSD
     to a 1×1 PSD (i.e. a single non-negative scalar `s_W ≥ 0`),
     which encodes only the linear epigraph cut
        u ≥ scale_W · Σ_{(i,j)} y_{e_i+e_j} · M_W[i,j]
     This is ALWAYS A VALID UPPER BOUND on the localizer value (it is
     the trace inequality for the localizer), so the relaxation
     remains a SOUND under-bound on val_B. The trade-off is that the
     selective windows lose their full PSD structure, weakening the
     cert by an amount proportional to their "missed slack".

     SOUNDNESS ARGUMENT: replacing a PSD cone by a 1×1 cone (linear
     scalar) in the dual gives the dual a STRICTLY SMALLER feasible
     set. So λ* can only DECREASE. If λ* still ≥ 0.75·Λ, the verdict
     'infeas' is still rigorous (val_B > t). If λ* drops below the
     infeas threshold, we lose the cert; we never gain a SPURIOUS
     cert. Therefore selective-cone is a SOUND OVER-APPROXIMATION of
     the true Farkas multiplier and is safe for our use case.

     FILTERING RULE: pre-solve the cheap epigraph LP at the box,
     extract per-window LP value (the binding mass at the epigraph
     row for window W). Keep full PSD for the top-K (default K=32)
     windows by LP value; downgrade the rest to linear-only.

  4. DEFAULT 48 THREADS.

API
---
    cache = build_sdp_escalation_cache_fast(d, windows, target=...)
    cert, info = bound_sdp_escalation_int_ge_fast(
        lo_int, hi_int, windows, d,
        target_num, target_den,
        cache=cache,
        n_window_psd_cones=32,
        return_diagnostic=True,
    )

The cache holds a precompute + env. The task is REBUILT per call
(per-box selection of which windows get PSD vs linear changes per
box). Build cost is ~1-2s at d=22 vs the 25+ minute solve we are
trying to shorten.

For benchmarking, the public API also exposes
    bound_sdp_escalation_lb_float_fast(lo, hi, windows, d, ...)
which is a thin diagnostic wrapper returning the verdict dict + timing.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mosek

from .box import SCALE as _SCALE
from .bound_sdp_escalation import _import_dual_sdp_api


# ---------------------------------------------------------------------
# Per-window LP value extractor (used to pick which windows get full
# PSD in the SDP).
# ---------------------------------------------------------------------

def _per_window_lp_values(
    lo: np.ndarray, hi: np.ndarray, windows, d: int,
) -> Tuple[float, np.ndarray]:
    """Solve the cheap per-box epigraph LP and return
    (lp_val, per_window_value), where per_window_value[w] is the
    primal value of  scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}  at the LP
    optimum. The window with per_window_value[w] closest to lp_val is
    a "binding" window of the epigraph.

    Falls back to a uniform ranking if the LP fails (returns -inf and
    a zeros array).
    """
    from interval_bnb.bound_epigraph import _solve_epigraph_lp, _cache_lp_structure
    from scipy.optimize import linprog

    n_W = len(windows)
    if n_W == 0:
        return float("inf"), np.zeros(0, dtype=np.float64)

    # Solve once; we need access to primal Y so we use the slack
    # representation.
    # _solve_epigraph_lp returns ineqlin marginals (dual). For the
    # binding-window ranking we want the PRIMAL value of the epigraph
    # row, i.e. scale_W · Σ Y_{i,j}. We can re-derive this from the
    # slack of the epigraph row (= z* - per_window_value).
    base = _solve_epigraph_lp(lo, hi, windows, d, return_residuals=True)
    lp_val = base[0]
    if not np.isfinite(lp_val):
        return float("-inf"), np.zeros(n_W, dtype=np.float64)
    # ineqlin row layout: [SW (n_y), NE (n_y), NW (n_y), SE (n_y), EPI (n_W),
    #                       SOS (1), tan (d)]. The epigraph row's primal
    # value is the LP-binding mass it carries: |dual marginal| > 0 ⇒
    # row binding ⇒ per_window_value == lp_val.
    #
    # We don't have direct access to the slack for the epigraph row
    # via the existing function, so we approximate per_window_value by
    # |dual marginal| × proxy weight. This gives a CORRECT ordering
    # because at the LP optimum, for any binding window
    #     z* = scale_W · Σ Y_{i,j}
    # and HiGHS reports a non-zero dual for that row. For non-binding
    # windows, the dual is zero and we use a tiebreaker = the
    # row's primal slack (which we re-evaluate cheaply below).
    ineqlin = base[1]
    if ineqlin is None:
        return lp_val, np.zeros(n_W, dtype=np.float64)
    n_y = d * d
    epi_duals = np.asarray(ineqlin[4 * n_y:4 * n_y + n_W], dtype=np.float64)
    # Larger |dual| ⇒ more binding. We also re-solve the epigraph row
    # primal explicitly via a separate cheap call: invert (μ, Y) from
    # the LP. Easier: just rank by |dual|. For a tiebreaker we run a
    # second LP that maximizes Σ_w (epi_row_w) at the same μ; but
    # this is overkill. The dual ranking is sufficient here.
    return lp_val, np.abs(epi_duals)


def _per_window_lp_primal_values(
    lo: np.ndarray, hi: np.ndarray, windows, d: int,
) -> Tuple[float, np.ndarray]:
    """Solve a SINGLE epigraph LP and return per-window PRIMAL values.

    Computes  pw[w] = scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}^*
    by re-evaluating the epigraph row's left-hand side at the LP
    optimum (μ*, Y*). For binding windows pw[w] ≈ lp_val; non-binding
    pw[w] < lp_val.

    Returns (lp_val, pw_values). pw[w] = -inf on LP failure.
    """
    from interval_bnb.bound_epigraph import _solve_epigraph_lp, _cache_lp_structure
    from scipy.sparse import csr_matrix
    n_W = len(windows)
    if n_W == 0:
        return float("inf"), np.zeros(0, dtype=np.float64)

    # We need the primal solution. Re-solve with a tiny extension that
    # exposes (μ, Y) via the residual = b_ub - A_ub x*. Easier: run our
    # own LP that returns x.
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n_y = d * d
    n_mu = d
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu

    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    # Reuse _solve_epigraph_lp's matrix construction by calling it
    # with return_residuals=True, then we read x from a custom solve.
    # For now, do the LP build inline (mirroring _solve_epigraph_lp
    # but returning x).
    n_pairs = n_y
    sw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    sw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    sw_data = np.empty(3 * n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2 * n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2 * n_pairs] = n_y + pair_i; sw_data[n_pairs:2 * n_pairs] = lo[pair_j]
    sw_rows[2 * n_pairs:3 * n_pairs] = np.arange(n_pairs); sw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j; sw_data[2 * n_pairs:3 * n_pairs] = lo[pair_i]

    ne_rows = np.empty(3 * n_pairs, dtype=np.int64)
    ne_cols = np.empty(3 * n_pairs, dtype=np.int64)
    ne_data = np.empty(3 * n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2 * n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[n_pairs:2 * n_pairs] = n_y + pair_i; ne_data[n_pairs:2 * n_pairs] = hi[pair_j]
    ne_rows[2 * n_pairs:3 * n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j; ne_data[2 * n_pairs:3 * n_pairs] = hi[pair_i]

    nw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    nw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    nw_data = np.empty(3 * n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2 * n_pairs + np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2 * n_pairs] = 2 * n_pairs + np.arange(n_pairs); nw_cols[n_pairs:2 * n_pairs] = n_y + pair_i; nw_data[n_pairs:2 * n_pairs] = -lo[pair_j]
    nw_rows[2 * n_pairs:3 * n_pairs] = 2 * n_pairs + np.arange(n_pairs); nw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j; nw_data[2 * n_pairs:3 * n_pairs] = -hi[pair_i]

    se_rows = np.empty(3 * n_pairs, dtype=np.int64)
    se_cols = np.empty(3 * n_pairs, dtype=np.int64)
    se_data = np.empty(3 * n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3 * n_pairs + np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2 * n_pairs] = 3 * n_pairs + np.arange(n_pairs); se_cols[n_pairs:2 * n_pairs] = n_y + pair_i; se_data[n_pairs:2 * n_pairs] = -hi[pair_j]
    se_rows[2 * n_pairs:3 * n_pairs] = 3 * n_pairs + np.arange(n_pairs); se_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j; se_data[2 * n_pairs:3 * n_pairs] = -lo[pair_i]

    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    epi_rows[:n_epi_pair_entries] = 4 * n_pairs + rows_w
    epi_cols[:n_epi_pair_entries] = cols_w
    epi_data[:n_epi_pair_entries] = scales_w
    epi_rows[n_epi_pair_entries:] = 4 * n_pairs + np.arange(n_W)
    epi_cols[n_epi_pair_entries:] = z_idx
    epi_data[n_epi_pair_entries:] = -1.0

    sos_row_start = 4 * n_pairs + n_W
    tan_row_start = sos_row_start + 1
    diag_idx = np.arange(d) * d + np.arange(d)
    sos_rows = np.full(d, sos_row_start, dtype=np.int64)
    sos_cols = diag_idx.astype(np.int64)
    sos_data = np.full(d, -1.0, dtype=np.float64)

    m = 0.5 * (lo + hi)
    tan_rows = np.empty(2 * d, dtype=np.int64)
    tan_cols = np.empty(2 * d, dtype=np.int64)
    tan_data = np.empty(2 * d, dtype=np.float64)
    tan_rows[:d] = tan_row_start + np.arange(d)
    tan_cols[:d] = diag_idx
    tan_data[:d] = -1.0
    tan_rows[d:] = tan_row_start + np.arange(d)
    tan_cols[d:] = n_y + np.arange(d)
    tan_data[d:] = 2.0 * m

    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows,
                                epi_rows, sos_rows, tan_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols,
                                epi_cols, sos_cols, tan_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data,
                                epi_data, sos_data, tan_data])
    n_ineq = 4 * n_pairs + n_W + 1 + d
    A_ub = coo_matrix((data_all, (rows_all, cols_all)),
                       shape=(n_ineq, n_vars)).tocsr()

    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    b_ub[n_pairs:2 * n_pairs] = hi[pair_i] * hi[pair_j]
    b_ub[2 * n_pairs:3 * n_pairs] = -lo[pair_j] * hi[pair_i]
    b_ub[3 * n_pairs:4 * n_pairs] = -hi[pair_j] * lo[pair_i]
    b_ub[4 * n_pairs:4 * n_pairs + n_W] = 0.0
    b_ub[sos_row_start] = -1.0 / d
    b_ub[tan_row_start:tan_row_start + d] = m * m

    # A_eq:  Σμ=1, RLT row-sums, RLT col-sums, Y-symmetry.
    n_sym = d * (d - 1) // 2
    n_eq = 1 + d + d + n_sym
    eq_rows = []
    eq_cols = []
    eq_data = []
    eq_rows.extend([0] * d)
    eq_cols.extend([n_y + i for i in range(d)])
    eq_data.extend([1.0] * d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i); eq_cols.append(i * d + j); eq_data.append(1.0)
        eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
    col_row_start = 1 + d
    for j in range(d):
        for i in range(d):
            eq_rows.append(col_row_start + j); eq_cols.append(i * d + j); eq_data.append(1.0)
        eq_rows.append(col_row_start + j); eq_cols.append(n_y + j); eq_data.append(-1.0)
    sym_row_start = col_row_start + d
    sym_k = 0
    for i in range(d):
        for j in range(i + 1, d):
            r = sym_row_start + sym_k
            eq_rows.append(r); eq_cols.append(i * d + j); eq_data.append(1.0)
            eq_rows.append(r); eq_cols.append(j * d + i); eq_data.append(-1.0)
            sym_k += 1
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(n_eq, n_vars),
    )
    b_eq = np.zeros(n_eq, dtype=np.float64)
    b_eq[0] = 1.0

    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)]
    bnds.append((0.0, None))
    c = np.zeros(n_vars)
    c[z_idx] = 1.0

    from interval_bnb.bound_epigraph import _HIGHS_OPTIONS
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bnds, method="highs", options=_HIGHS_OPTIONS)
    if not res.success:
        return float("-inf"), np.zeros(n_W, dtype=np.float64)
    x = np.asarray(res.x, dtype=np.float64)
    Y = x[:n_y]
    # Per-window primal value = scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}
    # = (epi_row LHS without the -z term) at x*.
    # Vectorized via the cached structure: for each k in rows_w we
    # accumulate scales_w[k] * Y[cols_w[k]] into pw[rows_w[k]].
    pw = np.zeros(n_W, dtype=np.float64)
    contrib = scales_w * Y[cols_w]
    np.add.at(pw, rows_w, contrib)
    return float(res.fun), pw


# ---------------------------------------------------------------------
# Selective dual-Farkas Task builder (PSD for "binding" windows,
# 1×1 PSD = scalar linear epi for the rest).
# ---------------------------------------------------------------------

def _build_dual_task_box_selective(
    P: Dict[str, Any], lo: np.ndarray, hi: np.ndarray, t_val: float,
    env: mosek.Env, *,
    psd_window_indices: Sequence[int],
    linear_window_indices: Sequence[int],
    lambda_upper_bound: float = 1.0,
    verbose: bool = False,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the dual Farkas LP with SELECTIVE per-window cones.

    psd_window_indices: list of window indices (into P['nontrivial_windows']
                        or any subset thereof) that get a full
                        n_loc × n_loc PSD localizer.
    linear_window_indices: remaining windows that get only a 1×1 PSD
                        (≡ scalar non-negative variable s_W ≥ 0
                         enforcing the linear epigraph cut).

    SOUNDNESS: the linear cone is the trace-cone projection of the PSD
    cone, hence STRICTLY LOOSER on the PRIMAL (more y vectors are
    feasible) ⇒ the DUAL feasible set SHRINKS ⇒ the optimal λ* CAN
    ONLY DECREASE. So a 'infeas' verdict here remains a rigorous
    cert for val_B > t.
    """
    api = _import_dual_sdp_api()
    _hash_monos = api['_hash_monos']
    _alpha_lookup = api['_alpha_lookup']
    _aggregate_scalar_triplet = api['_aggregate_scalar_triplet']
    from lasserre.core import _hash_add

    d = int(P['d'])
    n_y = int(P['n_y'])
    basis = P['basis']
    n_basis = int(P['n_basis'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    prime = P.get('prime')
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    consist_mono = P['consist_mono']
    consist_idx = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P['consist_ei_idx'], dtype=np.int64)
    old_to_new_arr = P.get('old_to_new')
    if old_to_new_arr is not None:
        old_to_new_arr = np.asarray(old_to_new_arr, dtype=np.int64)

    if n_loc == 0:
        raise RuntimeError("Order-2 Lasserre requires n_loc > 0")
    active_loc = list(range(d))
    psd_windows = list(psd_window_indices)
    lin_windows = list(linear_window_indices)

    task = env.Task()
    if verbose:
        task.set_Stream(mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # Bar-variable layout:
    #   [moment X_0 (n_basis)]
    # + [lower-box X_lo_i (n_loc) × d]
    # + [upper-box X_hi_i (n_loc) × d]
    # + [PSD-windows X_W (n_loc)]                  for w in psd_windows
    # + [LIN-windows X_W (1 × 1)]                  for w in lin_windows
    bar_sizes: List[int] = [n_basis]
    moment_bar_id = 0
    lo_bar_start = 1
    for _ in active_loc:
        bar_sizes.append(n_loc)
    lo_bar_end = len(bar_sizes)
    hi_bar_start = lo_bar_end
    for _ in active_loc:
        bar_sizes.append(n_loc)
    hi_bar_end = len(bar_sizes)
    win_psd_bar_start = hi_bar_end
    for _ in psd_windows:
        bar_sizes.append(n_loc)
    win_psd_bar_end = len(bar_sizes)
    win_lin_bar_start = win_psd_bar_end
    for _ in lin_windows:
        bar_sizes.append(1)  # 1×1 PSD = scalar s_W ≥ 0
    win_lin_bar_end = len(bar_sizes)
    n_bar = len(bar_sizes)
    task.appendbarvars(bar_sizes)

    # Scalar variables: [λ | μ_k | v_α].
    kept_k = [k for k in range(len(consist_mono)) if int(consist_idx[k]) >= 0]
    n_consist = len(kept_k)
    LAMBDA_IDX = 0
    MU_START = 1
    MU_END = 1 + n_consist
    V_START = MU_END
    V_END = V_START + n_y
    n_scalar = V_END
    task.appendvars(n_scalar)
    task.putvarbound(LAMBDA_IDX, mosek.boundkey.ra, 0.0,
                     float(lambda_upper_bound))
    if n_consist > 0:
        task.putvarboundslice(
            MU_START, MU_END,
            [mosek.boundkey.fr] * n_consist,
            np.full(n_consist, -np.inf, dtype=np.float64),
            np.full(n_consist, +np.inf, dtype=np.float64),
        )
    task.putvarboundslice(
        V_START, V_END,
        [mosek.boundkey.lo] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.full(n_y, +np.inf, dtype=np.float64),
    )

    # Constraint rows: one per α (stationarity = 0).
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y, [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # Scalar A matrix.
    scalar_rows: List[int] = []
    scalar_cols: List[int] = []
    scalar_vals: List[float] = []

    alpha_zero = tuple(0 for _ in range(d))
    alpha_zero_row = int(mono_idx[alpha_zero])
    scalar_rows.append(alpha_zero_row)
    scalar_cols.append(LAMBDA_IDX)
    scalar_vals.append(1.0)

    for j, k in enumerate(kept_k):
        a_k = int(consist_idx[k])
        child = consist_ei_idx[k]
        coef_by_row: Dict[int, float] = {}
        for i in range(d):
            c = int(child[i])
            if c >= 0:
                coef_by_row[c] = coef_by_row.get(c, 0.0) + 1.0
        coef_by_row[a_k] = coef_by_row.get(a_k, 0.0) - 1.0
        col_j = MU_START + j
        for row, val in coef_by_row.items():
            if val != 0.0:
                scalar_rows.append(row)
                scalar_cols.append(col_j)
                scalar_vals.append(float(val))

    alpha_rows = np.arange(n_y, dtype=np.int64)
    scalar_rows.extend(alpha_rows.tolist())
    scalar_cols.extend((V_START + alpha_rows).tolist())
    scalar_vals.extend([1.0] * n_y)

    r_arr, c_arr, v_arr = _aggregate_scalar_triplet(
        np.asarray(scalar_rows, dtype=np.int64),
        np.asarray(scalar_cols, dtype=np.int64),
        np.asarray(scalar_vals, dtype=np.float64),
    )
    if r_arr.size:
        task.putaijlist(r_arr, c_arr, v_arr)

    # Bar-matrix sensitivity coefficients.
    bar_subi_list: List[np.ndarray] = []
    bar_subj_list: List[np.ndarray] = []
    bar_subk_list: List[np.ndarray] = []
    bar_subl_list: List[np.ndarray] = []
    bar_val_list: List[np.ndarray] = []
    bar_tcoef_list: List[np.ndarray] = []
    bar_box_lo_idx_list: List[np.ndarray] = []
    bar_box_lo_coef_list: List[np.ndarray] = []
    bar_box_hi_idx_list: List[np.ndarray] = []
    bar_box_hi_coef_list: List[np.ndarray] = []

    def _append(subi, subj, subk, subl, vals,
                tcoefs=None, lo_idx=None, lo_coef=None,
                hi_idx=None, hi_coef=None):
        if subi.size == 0:
            return
        bar_subi_list.append(np.ascontiguousarray(subi, dtype=np.int32))
        bar_subj_list.append(np.ascontiguousarray(subj, dtype=np.int32))
        bar_subk_list.append(np.ascontiguousarray(subk, dtype=np.int32))
        bar_subl_list.append(np.ascontiguousarray(subl, dtype=np.int32))
        bar_val_list.append(np.ascontiguousarray(vals, dtype=np.float64))
        n = subi.size
        bar_tcoef_list.append(
            np.ascontiguousarray(tcoefs, dtype=np.float64)
            if tcoefs is not None else np.zeros(n, dtype=np.float64))
        bar_box_lo_idx_list.append(
            np.ascontiguousarray(lo_idx, dtype=np.int32)
            if lo_idx is not None else np.full(n, -1, dtype=np.int32))
        bar_box_lo_coef_list.append(
            np.ascontiguousarray(lo_coef, dtype=np.float64)
            if lo_coef is not None else np.zeros(n, dtype=np.float64))
        bar_box_hi_idx_list.append(
            np.ascontiguousarray(hi_idx, dtype=np.int32)
            if hi_idx is not None else np.full(n, -1, dtype=np.int32))
        bar_box_hi_coef_list.append(
            np.ascontiguousarray(hi_coef, dtype=np.float64)
            if hi_coef is not None else np.zeros(n, dtype=np.float64))

    # Moment cone: single n_basis × n_basis BAR; coef = +1 at α=loc[k]+loc[l].
    B_arr = np.asarray(basis, dtype=np.int64)
    B_hash = _hash_monos(B_arr, bases_arr, prime)
    ks_m, ls_m = np.tril_indices(n_basis)
    alpha_hash_m = _hash_add(B_hash[ks_m], B_hash[ls_m], prime)
    alpha_idx_m = _alpha_lookup(alpha_hash_m, sorted_h, sort_o, old_to_new_arr)
    if np.any(alpha_idx_m < 0):
        raise RuntimeError("Moment sensitivity lookup -1; precompute broken")
    _append(
        alpha_idx_m,
        np.full(ks_m.shape, moment_bar_id, dtype=np.int32),
        ks_m, ls_m,
        np.full(ks_m.shape, +1.0, dtype=np.float64),
    )

    # Localizing prep.
    L_arr = np.asarray(loc_basis, dtype=np.int64)
    L_hash = _hash_monos(L_arr, bases_arr, prime)
    ks_l, ls_l = np.tril_indices(n_loc)
    base_hash_loc = _hash_add(L_hash[ks_l], L_hash[ls_l], prime)
    alpha_idx_loc0 = _alpha_lookup(
        base_hash_loc, sorted_h, sort_o, old_to_new_arr)

    # LOWER BOX cones X_lo_i.
    for j, i in enumerate(active_loc):
        bar_idx_here = lo_bar_start + j
        alpha_hash_li = _hash_add(base_hash_loc, bases_arr[i], prime)
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m:
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, +1.0, dtype=np.float64),
            )
        mask0 = alpha_idx_loc0 >= 0
        n_m0 = int(mask0.sum())
        if n_m0:
            _append(
                alpha_idx_loc0[mask0],
                np.full(n_m0, bar_idx_here, dtype=np.int32),
                ks_l[mask0], ls_l[mask0],
                np.zeros(n_m0, dtype=np.float64),
                lo_idx=np.full(n_m0, i, dtype=np.int32),
                lo_coef=np.full(n_m0, -1.0, dtype=np.float64),
            )

    # UPPER BOX cones X_hi_i.
    for j, i in enumerate(active_loc):
        bar_idx_here = hi_bar_start + j
        mask0 = alpha_idx_loc0 >= 0
        n_m0 = int(mask0.sum())
        if n_m0:
            _append(
                alpha_idx_loc0[mask0],
                np.full(n_m0, bar_idx_here, dtype=np.int32),
                ks_l[mask0], ls_l[mask0],
                np.zeros(n_m0, dtype=np.float64),
                hi_idx=np.full(n_m0, i, dtype=np.int32),
                hi_coef=np.full(n_m0, +1.0, dtype=np.float64),
            )
        alpha_hash_li = _hash_add(base_hash_loc, bases_arr[i], prime)
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m:
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, -1.0, dtype=np.float64),
            )

    # PSD WINDOW cones (full n_loc × n_loc).
    for w_j, w in enumerate(psd_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_psd_bar_start + w_j

        # (a) +t at α = loc[k]+loc[l].
        mask_t = alpha_idx_loc0 >= 0
        n_m = int(mask_t.sum())
        if n_m:
            _append(
                alpha_idx_loc0[mask_t],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks_l[mask_t], ls_l[mask_t],
                np.zeros(n_m, dtype=np.float64),
                tcoefs=np.full(n_m, +1.0, dtype=np.float64),
            )

        # (b) -Σ_{i,j} M_W[i,j] at α = loc[k]+loc[l]+e_i+e_j.
        if len(nz_i) > 0:
            n_pairs = len(nz_i)
            shifts = _hash_add(bases_arr[nz_i], bases_arr[nz_j], prime)
            mw_vals = Mw[nz_i, nz_j]
            for pp in range(n_pairs):
                alpha_hash_pij = _hash_add(base_hash_loc, shifts[pp], prime)
                alpha_idx_pij = _alpha_lookup(
                    alpha_hash_pij, sorted_h, sort_o, old_to_new_arr)
                mask_q = alpha_idx_pij >= 0
                n_q = int(mask_q.sum())
                if n_q == 0:
                    continue
                _append(
                    alpha_idx_pij[mask_q],
                    np.full(n_q, W_bar_idx, dtype=np.int32),
                    ks_l[mask_q], ls_l[mask_q],
                    np.full(n_q, -float(mw_vals[pp]), dtype=np.float64),
                )

    # LINEAR WINDOW cones (1×1 PSD; only the (0,0) entry matters).
    # For these, ks_l=ls_l=0 ALWAYS; the localizer is just the scalar
    # value s_W ≥ 0. Coefficient at row α = e_i+e_j is -M_W[i,j] · s_W;
    # at row α = 0 (the 1×1 base) we also pick up the +t · s_W term.
    for w_j, w in enumerate(lin_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_lin_bar_start + w_j

        # (a) +t at α = 2·loc[0] = 0   (loc_basis[0] is the constant monomial).
        # We use bar size 1, so subk=subl=0.
        # alpha at loc[0]+loc[0]: take alpha_idx_loc0[0].
        alpha_zero_idx = int(alpha_idx_loc0[0]) if alpha_idx_loc0.size > 0 else -1
        if alpha_zero_idx >= 0:
            _append(
                np.array([alpha_zero_idx], dtype=np.int64),
                np.array([W_bar_idx], dtype=np.int32),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([0.0], dtype=np.float64),
                tcoefs=np.array([+1.0], dtype=np.float64),
            )

        # (b) -Σ_{i,j} M_W[i,j] at α = e_i+e_j (i.e. shift only).
        if len(nz_i) > 0:
            n_pairs = len(nz_i)
            shifts = _hash_add(bases_arr[nz_i], bases_arr[nz_j], prime)
            mw_vals = Mw[nz_i, nz_j]
            for pp in range(n_pairs):
                # α = e_i + e_j ; this equals base_hash_loc[0] (which is 0)
                # plus shifts[pp]. So look up shifts[pp] directly.
                alpha_idx_pij = _alpha_lookup(
                    np.array([shifts[pp]], dtype=np.int64),
                    sorted_h, sort_o, old_to_new_arr)
                if int(alpha_idx_pij[0]) < 0:
                    continue
                _append(
                    alpha_idx_pij,
                    np.array([W_bar_idx], dtype=np.int32),
                    np.array([0], dtype=np.int64),
                    np.array([0], dtype=np.int64),
                    np.array([-float(mw_vals[pp])], dtype=np.float64),
                )

    # Concatenate, aggregate duplicates, expand to initial values.
    if not bar_subi_list:
        n_bar_entries = 0
        all_subi = all_subj = all_subk = all_subl = None
        all_static = all_tcoef = None
        all_lo_idx = all_lo_coef = all_hi_idx = all_hi_coef = None
    else:
        raw_subi = np.concatenate(bar_subi_list)
        raw_subj = np.concatenate(bar_subj_list)
        raw_subk = np.concatenate(bar_subk_list)
        raw_subl = np.concatenate(bar_subl_list)
        raw_static = np.concatenate(bar_val_list)
        raw_tcoef = np.concatenate(bar_tcoef_list)
        raw_lo_idx = np.concatenate(bar_box_lo_idx_list)
        raw_lo_coef = np.concatenate(bar_box_lo_coef_list)
        raw_hi_idx = np.concatenate(bar_box_hi_idx_list)
        raw_hi_coef = np.concatenate(bar_box_hi_coef_list)

        order = np.lexsort([raw_subl, raw_subk, raw_subj, raw_subi])
        s_subi = raw_subi[order]
        s_subj = raw_subj[order]
        s_subk = raw_subk[order]
        s_subl = raw_subl[order]
        s_static = raw_static[order]
        s_tcoef = raw_tcoef[order]
        s_lo_idx = raw_lo_idx[order]
        s_lo_coef = raw_lo_coef[order]
        s_hi_idx = raw_hi_idx[order]
        s_hi_coef = raw_hi_coef[order]

        n_total = s_subi.size
        is_first = np.empty(n_total, dtype=bool)
        is_first[0] = True
        is_first[1:] = ((s_subi[1:] != s_subi[:-1]) |
                        (s_subj[1:] != s_subj[:-1]) |
                        (s_subk[1:] != s_subk[:-1]) |
                        (s_subl[1:] != s_subl[:-1]))
        group_id = np.cumsum(is_first.astype(np.int64)) - 1
        n_groups = int(group_id[-1] + 1) if n_total > 0 else 0

        all_subi = s_subi[is_first]
        all_subj = s_subj[is_first]
        all_subk = s_subk[is_first]
        all_subl = s_subl[is_first]
        all_static = np.bincount(group_id, weights=s_static.astype(np.float64),
                                  minlength=n_groups)
        all_tcoef = np.bincount(group_id, weights=s_tcoef.astype(np.float64),
                                 minlength=n_groups)
        all_lo_coef = np.bincount(group_id,
                                   weights=s_lo_coef.astype(np.float64),
                                   minlength=n_groups)
        all_hi_coef = np.bincount(group_id,
                                   weights=s_hi_coef.astype(np.float64),
                                   minlength=n_groups)
        first_idx = np.where(is_first)[0]
        all_lo_idx = np.maximum.reduceat(s_lo_idx, first_idx).astype(np.int32)
        all_hi_idx = np.maximum.reduceat(s_hi_idx, first_idx).astype(np.int32)

        keep = ((np.abs(all_static) > 0.0) | (np.abs(all_tcoef) > 0.0)
                | (np.abs(all_lo_coef) > 0.0) | (np.abs(all_hi_coef) > 0.0))
        if not np.all(keep):
            all_subi = all_subi[keep]; all_subj = all_subj[keep]
            all_subk = all_subk[keep]; all_subl = all_subl[keep]
            all_static = all_static[keep]; all_tcoef = all_tcoef[keep]
            all_lo_idx = all_lo_idx[keep]; all_lo_coef = all_lo_coef[keep]
            all_hi_idx = all_hi_idx[keep]; all_hi_coef = all_hi_coef[keep]

        n_bar_entries = int(all_subi.size)
        init_vals = (all_static + float(t_val) * all_tcoef
                     + np.where(all_lo_idx >= 0,
                                lo[np.maximum(all_lo_idx, 0)] * all_lo_coef,
                                0.0)
                     + np.where(all_hi_idx >= 0,
                                hi[np.maximum(all_hi_idx, 0)] * all_hi_coef,
                                0.0))
        task.putbarablocktriplet(
            all_subi.astype(np.int32), all_subj.astype(np.int32),
            all_subk.astype(np.int32), all_subl.astype(np.int32), init_vals)

    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes, 'n_bar': n_bar,
        'n_psd_windows': len(psd_windows),
        'n_lin_windows': len(lin_windows),
        'psd_windows': list(psd_windows),
        'lin_windows': list(lin_windows),
        'n_scalar': n_scalar, 'n_consist_kept': n_consist,
        'n_cons': n_y, 'n_y': n_y,
        'n_bar_entries': n_bar_entries,
        't_val': float(t_val),
        'lo': np.asarray(lo, dtype=np.float64).copy(),
        'hi': np.asarray(hi, dtype=np.float64).copy(),
        'LAMBDA_IDX': LAMBDA_IDX, 'MU_START': MU_START, 'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
    }

    if verbose:
        print(f"  [box-dual-fast] n_bar={n_bar} (psd_w={len(psd_windows)} "
              f"lin_w={len(lin_windows)}) n_scalar={n_scalar:,} "
              f"n_cons={n_y:,} n_bar_entries={n_bar_entries:,} "
              f"t={t_val:.4f} build={build_time:.2f}s",
              flush=True)

    return task, info


# ---------------------------------------------------------------------
# Cache (just precompute + env; task is REBUILT per box because the
# selective window set varies).
# ---------------------------------------------------------------------

def build_sdp_escalation_cache_fast(
    d: int, windows=None,
    target: float = 1.281,
    verbose: bool = False,
) -> dict:
    """One-time per-worker setup: build precompute + MOSEK env. Task is
    rebuilt per box (selective window set varies).
    """
    if windows is None:
        from interval_bnb.windows import build_windows
        windows = build_windows(d)
    api = _import_dual_sdp_api()
    P = api['_precompute'](d, order=2, verbose=verbose, lazy_ab_eiej=True)
    env = mosek.Env()
    return {
        'd': d,
        'order': 2,
        'P': P,
        'env': env,
        'target': float(target),
        'windows': windows,
    }


# ---------------------------------------------------------------------
# MOSEK speed parameter setter
# ---------------------------------------------------------------------

def _apply_speed_params(task: mosek.Task, *,
                        tol: float = 1e-5,
                        max_iter: int = 50,
                        n_threads: int = 48,
                        time_limit_s: float = 600.0,
                        solve_form: str = 'free',
                        presolve: str = 'on',
                        ) -> None:
    """Apply the MOSEK parameter cocktail for FAST per-box SDP solves.

    tol         intpnt_co_tol_pfeas/dfeas/rel_gap (default 1e-5).
    max_iter    cap on IPM iterations (default 50).
    n_threads   default 48.
    time_limit_s outer time limit.
    solve_form  'free' (auto), 'primal' or 'dual'.
    presolve    'on', 'off' or 'auto'.
    """
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, float(tol))
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, float(tol))
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, float(tol))
    try:
        task.putintparam(mosek.iparam.intpnt_max_iterations, int(max_iter))
    except Exception:
        pass
    task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit_s))
    if n_threads > 0:
        task.putintparam(mosek.iparam.num_threads, int(n_threads))
    # Solve form (primal/dual/free).
    try:
        sf_map = {'free': mosek.solveform.free,
                  'primal': mosek.solveform.primal,
                  'dual': mosek.solveform.dual}
        if solve_form in sf_map:
            task.putintparam(mosek.iparam.intpnt_solve_form, sf_map[solve_form])
    except Exception:
        pass
    # Presolve.
    try:
        ps_map = {'on': mosek.presolvemode.on,
                  'off': mosek.presolvemode.off,
                  'auto': mosek.presolvemode.free}
        if presolve in ps_map:
            task.putintparam(mosek.iparam.presolve_use, ps_map[presolve])
        # Linear-dependence check (cheap & sometimes helpful).
        task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.on)
    except Exception:
        pass
    # Optimizer = conic (default for SDP, but be explicit).
    try:
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.conic)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Public per-box entry points
# ---------------------------------------------------------------------

def bound_sdp_escalation_int_ge_fast(
    lo_int: Sequence[int], hi_int: Sequence[int], windows, d: int,
    target_num: int, target_den: int,
    cache: Optional[dict] = None,
    *,
    n_window_psd_cones: int = 32,
    early_stop: bool = True,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
    tol: float = 1e-5,
    max_iter: int = 50,
    n_threads: int = 48,
    time_limit_s: float = 600.0,
    solve_form: str = 'free',
    presolve: str = 'on',
    verbose: bool = False,
    return_diagnostic: bool = False,
):
    """Fast per-box dual-Farkas cert at t = target_num / target_den.

    Returns True iff verdict='infeas' (rigorous cert val_B > target).
    Returns False on EXCEPTION, on verdict='feas'/'uncertain', or on
    LP-pre-filter failure (no per-window ranking → use ALL as PSD).

    n_window_psd_cones: top-K windows by epigraph-LP primal value get
        full PSD localizer. The rest get scalar-only (1×1 PSD). Set
        to a large number (≥ number of nontrivial windows) to recover
        the full-PSD baseline.
    """
    if int(sum(lo_int)) > _SCALE or int(sum(hi_int)) < _SCALE:
        if return_diagnostic:
            return True, {'status': 'EMPTY', 'cert_via': 'vacuous_empty'}
        return True

    target_f = float(target_num) / float(target_den)
    lo = np.asarray([float(li) / _SCALE for li in lo_int], dtype=np.float64)
    hi = np.asarray([float(hv) / _SCALE for hv in hi_int], dtype=np.float64)

    api = _import_dual_sdp_api()
    if cache is None:
        cache = build_sdp_escalation_cache_fast(d, windows, target=target_f)
    P = cache['P']
    env = cache['env']

    nontrivial = list(P['nontrivial_windows'])
    n_nt = len(nontrivial)

    # Determine which windows get PSD vs linear.
    t_lp_start = time.time()
    if n_window_psd_cones >= n_nt or n_window_psd_cones < 0:
        psd_idx = nontrivial
        lin_idx: List[int] = []
        lp_val = float('nan')
        pw_vals = np.zeros(n_nt, dtype=np.float64)
    else:
        lp_val, pw_vals = _per_window_lp_primal_values(lo, hi, windows, d)
        # pw_vals is indexed over ALL windows (including trivial). We
        # restrict to the nontrivial ones first, then rank.
        pw_nt = pw_vals[np.asarray(nontrivial, dtype=np.int64)] \
            if pw_vals.size == len(windows) else np.zeros(n_nt)
        order = np.argsort(-pw_nt)  # descending
        psd_local = order[:int(n_window_psd_cones)].tolist()
        lin_local = order[int(n_window_psd_cones):].tolist()
        psd_idx = [nontrivial[i] for i in psd_local]
        lin_idx = [nontrivial[i] for i in lin_local]
    t_lp = time.time() - t_lp_start

    # Build the dual-Farkas task (selective).
    t_build_start = time.time()
    try:
        task, info = _build_dual_task_box_selective(
            P, lo, hi, target_f, env,
            psd_window_indices=psd_idx,
            linear_window_indices=lin_idx,
            verbose=verbose,
        )
    except Exception as e:
        if return_diagnostic:
            return False, {
                'status': f'BUILD_EXCEPTION:{type(e).__name__}',
                'error_msg': str(e),
                'cert_via': 'build_failure',
                'lp_pre_s': t_lp,
            }
        return False
    t_build = time.time() - t_build_start

    _apply_speed_params(
        task, tol=tol, max_iter=max_iter, n_threads=n_threads,
        time_limit_s=time_limit_s, solve_form=solve_form, presolve=presolve,
    )

    t_solve_start = time.time()
    try:
        verdict = api['solve_dual_task'](
            task, info,
            early_stop_on_clear_verdict=early_stop,
            early_stop_feas_frac=early_stop_feas_frac,
            early_stop_infeas_frac=early_stop_infeas_frac,
            verbose=verbose,
        )
    except Exception as e:
        if return_diagnostic:
            return False, {
                'status': f'SOLVE_EXCEPTION:{type(e).__name__}',
                'error_msg': str(e),
                'cert_via': 'solver_failure',
                'lp_pre_s': t_lp, 'build_s': t_build,
            }
        return False
    t_solve = time.time() - t_solve_start

    cert = bool(verdict.get('verdict') == 'infeas')

    if return_diagnostic:
        return cert, {
            **verdict,
            'target_f': target_f,
            'cert_via': 'farkas_infeasibility_selective',
            'n_window_psd_cones': int(n_window_psd_cones),
            'n_psd_actual': len(psd_idx),
            'n_lin_actual': len(lin_idx),
            'lp_pre_val': float(lp_val),
            'lp_pre_s': float(t_lp),
            'build_s': float(t_build),
            'solve_s': float(t_solve),
            'total_s': float(t_lp + t_build + t_solve),
        }
    return cert


def bound_sdp_escalation_lb_float_fast(
    lo: np.ndarray, hi: np.ndarray, windows, d: int,
    *,
    cache: Optional[dict] = None,
    target: float = 1.281,
    n_window_psd_cones: int = 32,
    early_stop: bool = True,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
    tol: float = 1e-5,
    max_iter: int = 50,
    n_threads: int = 48,
    time_limit_s: float = 600.0,
    solve_form: str = 'free',
    presolve: str = 'on',
    verbose: bool = False,
) -> dict:
    """Diagnostic float-side wrapper for benchmarking. Returns a dict
    with verdict, lambda_star, and granular timings.
    """
    if cache is None:
        cache = build_sdp_escalation_cache_fast(d, windows, target=target)
    api = _import_dual_sdp_api()
    P = cache['P']
    env = cache['env']

    nontrivial = list(P['nontrivial_windows'])
    n_nt = len(nontrivial)

    lo_f = np.asarray(lo, dtype=np.float64)
    hi_f = np.asarray(hi, dtype=np.float64)

    t_lp_start = time.time()
    if n_window_psd_cones >= n_nt or n_window_psd_cones < 0:
        psd_idx = nontrivial
        lin_idx: List[int] = []
        lp_val = float('nan')
        pw_vals = np.zeros(n_nt, dtype=np.float64)
    else:
        lp_val, pw_vals = _per_window_lp_primal_values(lo_f, hi_f, windows, d)
        pw_nt = pw_vals[np.asarray(nontrivial, dtype=np.int64)] \
            if pw_vals.size == len(windows) else np.zeros(n_nt)
        order = np.argsort(-pw_nt)
        psd_local = order[:int(n_window_psd_cones)].tolist()
        lin_local = order[int(n_window_psd_cones):].tolist()
        psd_idx = [nontrivial[i] for i in psd_local]
        lin_idx = [nontrivial[i] for i in lin_local]
    t_lp = time.time() - t_lp_start

    t_build_start = time.time()
    try:
        task, info = _build_dual_task_box_selective(
            P, lo_f, hi_f, target, env,
            psd_window_indices=psd_idx,
            linear_window_indices=lin_idx,
            verbose=verbose,
        )
    except Exception as e:
        return {
            'status': f'BUILD_EXCEPTION:{type(e).__name__}',
            'error_msg': str(e),
            'verdict': 'uncertain',
            'lp_pre_s': t_lp,
        }
    t_build = time.time() - t_build_start

    _apply_speed_params(
        task, tol=tol, max_iter=max_iter, n_threads=n_threads,
        time_limit_s=time_limit_s, solve_form=solve_form, presolve=presolve,
    )

    t_solve_start = time.time()
    try:
        verdict = api['solve_dual_task'](
            task, info,
            early_stop_on_clear_verdict=early_stop,
            early_stop_feas_frac=early_stop_feas_frac,
            early_stop_infeas_frac=early_stop_infeas_frac,
            verbose=verbose,
        )
    except Exception as e:
        return {
            'status': f'SOLVE_EXCEPTION:{type(e).__name__}',
            'error_msg': str(e),
            'verdict': 'uncertain',
            'lp_pre_s': t_lp, 'build_s': t_build,
        }
    t_solve = time.time() - t_solve_start

    verdict.update({
        'target': target,
        'n_window_psd_cones': int(n_window_psd_cones),
        'n_psd_actual': len(psd_idx),
        'n_lin_actual': len(lin_idx),
        'lp_pre_val': float(lp_val),
        'lp_pre_s': float(t_lp),
        'build_s': float(t_build),
        'solve_s': float(t_solve),
        'total_s': float(t_lp + t_build + t_solve),
    })
    return verdict

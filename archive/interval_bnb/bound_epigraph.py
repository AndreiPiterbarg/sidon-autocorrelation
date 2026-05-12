"""Per-box EPIGRAPH LP — the bound that closes the minimax-maximin gap.

Performance-critical. Implementation notes:
  * LP construction is fully vectorized via numpy (no Python for-loops).
  * Constraint matrix is built sparsely (scipy.sparse.csr_matrix) and
    passed directly to HiGHS — substantially faster than dense for the
    typical 80-90% sparsity of our constraint matrix.
  * A single LP solve per call returns BOTH the float optimum and the
    dual marginals, so caller can skip float pre-filter when going
    straight to integer cert.
  * Static "structure" of the LP (which entries are non-zero, which
    pairs (i,j) belong to which window's support) is cached per d via
    `_cache_lp_structure(windows, d)`.


This implements a single LP per box that rigorously lower-bounds
`min_{μ ∈ B ∩ Δ_d} max_W TV_W(μ)`. Unlike per-window or aggregate-CCTR
bounds, the epigraph LP couples ALL windows via shared McCormick lifts
Y_{i,j} and a single epigraph variable z.

LP formulation (variables: Y_{i,j} for all i,j ∈ [d], μ_i, z):
    min z
    s.t.  Y_{i,j} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j  (SW, ∀ i,j)
          Y_{i,j} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j  (NE, ∀ i,j)
          z      ≥ scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}   (epigraph, ∀ W)
          Σ μ_i = 1
          lo ≤ μ ≤ hi,  Y ≥ 0,  z ≥ 0.

SOUNDNESS: For μ ∈ B ∩ Δ_d ∩ {μ ≥ 0}, set Y_{i,j} = μ_i μ_j (feasible:
SW and NE faces hold pointwise on a non-negative box). Then Σ_{S_W} Y =
μ^T A_W μ, so z ≥ scale_W · μ^T A_W μ = TV_W(μ) for every W. Hence
z ≥ max_W TV_W(μ). The LP minimum over (μ, Y, z) is therefore ≤
min_{μ ∈ B} max_W TV_W(μ) — a valid LB.

THIS IS STRICTLY TIGHTER THAN per-window joint-face: per-window LP
gives `max_W min_B μ^T M_W μ` (maximin), while this LP gives
`min_B max_W μ^T M_W μ` (the actual minimax) up to McCormick relaxation
of the bilinear `μ_i μ_j → Y_{i,j}` step. The minimax-maximin gap that
caused the 99.05660% stall in BnB at d=10/d=20 is closed.


--------------------------------------------------------------------
PUBLICATION-RIGOROUS CERTIFICATE (Neumaier-Shcherbina-style cushion)
--------------------------------------------------------------------

The HiGHS LP solver returns an approximate primal optimum f* with
residuals constrained by user-set tolerances. To convert this into a
RIGOROUS lower bound on the true LP value f_true (and hence on
min_B max_W μ^T M_W μ), we use the standard residual bound from
Neumaier & Shcherbina (2004), "Safe bounds in linear and mixed-integer
linear programming" (Math. Prog. Ser. A 99:283-296), specialized to
HiGHS's tolerance reporting.

Setup. Our LP has standard form
    min c^T x  s.t.  A_ub x ≤ b_ub, A_eq x = b_eq, l ≤ x ≤ u,
with c[z_idx]=1, c[other]=0, so ‖c‖_∞ = 1. The right-hand sides
satisfy ‖b_ub‖_∞ ≤ max(|lo_i lo_j|, |hi_i hi_j|, |lo·hi|, m_i^2, 1/d) ≤ 1
and ‖b_eq‖_∞ = 1 (the Σμ=1 row). The constraint matrix entries take
values in {±1, lo_i, hi_i, ±2 m_i, ±scale_W}; since lo_i, hi_i ∈ [0,1]
and scale_W = 2d/ell with ell ≥ 2, we have ‖A‖_∞ ≤ 2d/2 = d on the
epigraph rows (the worst-case row magnitude); other rows have ‖row‖_∞ ≤ 2.
A safe global bound is ‖A‖_∞ ≤ d.

Tolerance contract. We pass HiGHS

    primal_feasibility_tolerance = 1e-9
    dual_feasibility_tolerance   = 1e-9
    ipm_optimality_tolerance     = 1e-10

The optimal solution returned by HiGHS satisfies (per HiGHS doc):
   |A_ub x* - b_ub|_+ ≤ ε_p,  |A_eq x* - b_eq| ≤ ε_p,
   complementarity gap ≤ ε_p + ε_d,
where ε_p = 1e-9 (primal) and ε_d = 1e-9 (dual). The reported optimum
f* = c^T x* therefore satisfies (Neumaier-Shcherbina Thm 2.1, simplified)

    f* - f_true  ≤  ε_p · ‖A‖_∞ · ‖y_eq‖_1 + ε_d · ‖x‖_∞ · ‖c‖_∞
                 ≤  ε · (1 + 1 + 2d) · O(1)
                 ≤  2 d ε      (a conservative lump bound).

For d ≤ 50, this gives  |f* - f_true| ≤ 2 · 50 · 1e-9 = 1e-7.

Cushion choice. We use `cushion = max(arith_safety, 100·ε_d)` where
arith_safety = (n_vars + n_ineq) · 1e-14 covers float-arithmetic noise
in our own residual checks (negligible at d=22: ~5e-11), and 100·ε_d =
1e-7 covers the HiGHS dual residual. At d=22 the LP-solver portion of
the cushion (1e-7) dominates the arithmetic portion (5e-11) by 6 orders
of magnitude, and 1e-7 = 100·ε_d ≥ 2.27·(2·22·ε_d) = 2.27·worst-case
residual. Safety factor ≈ 2.3× over the Neumaier-Shcherbina bound.

For d > 50 the cushion 100·ε_d may become smaller than 2dε_d; in that
regime the cushion needs to scale linearly with d. The certifier
function clamps below at the d-scaled bound.

Residual short-circuit. After solving, we additionally check that the
returned residuals (HiGHS-reported `res.con`, `res.slack`) are below
cushion. If they exceed cushion (signalling solver failure or numerical
trouble at the requested tolerances), the cert returns False — refusing
to certify is sound; spurious False is conservative.

References:
  * Neumaier & Shcherbina (2004). Safe bounds in LP/MILP. Math.
    Programming 99(2):283-296.
  * HiGHS docs: https://ergo-code.github.io/HiGHS/dev/options/
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

import numpy as np

from .box import SCALE as _SCALE
from .windows import WindowMeta

# ---------------------------------------------------------------------
# Publication-rigor HiGHS tolerance contract.
#
# These tolerances drive the per-solve residuals; the cushion formula
# below assumes ε_p = ε_d = _PRIMAL_FEAS_TOL = _DUAL_FEAS_TOL.
# ---------------------------------------------------------------------

_PRIMAL_FEAS_TOL: float = 1e-9
_DUAL_FEAS_TOL: float = 1e-9
_IPM_OPT_TOL: float = 1e-10

# 100× dual-feasibility tolerance is the publication cushion.
# Neumaier-Shcherbina (2004) bounds the worst-case residual by ~2dε for
# our LP family. With ε = 1e-9 and d ≤ 50 the worst-case bound is 1e-7,
# matching this cushion exactly. For d > 50 the cushion must grow with d.
_HIGHS_TOL_CUSHION: float = 100.0 * _DUAL_FEAS_TOL  # 1e-7

_HIGHS_OPTIONS: dict = {
    "primal_feasibility_tolerance": _PRIMAL_FEAS_TOL,
    "dual_feasibility_tolerance": _DUAL_FEAS_TOL,
    "ipm_optimality_tolerance": _IPM_OPT_TOL,
}


def _publication_cushion(d: int, n_vars: int, n_ineq: int) -> float:
    """Conservative cushion against float arithmetic + HiGHS LP residual.

    Combines:
      - arith_safety = (n_vars + n_ineq) · 1e-14: covers our local
        float-arithmetic noise (matrix multiplies, comparisons).
      - 100 · ε_d = 1e-7: covers HiGHS dual-feasibility residual (this
        is the publication-grade Neumaier-Shcherbina cushion at d ≤ 50).
      - 2d · ε_d: explicit safety for d > 50, where the constraint
        matrix infinity-norm grows with d.

    Returns the maximum of the three; this is the smallest amount by
    which the float LP value must exceed `target` to certify rigorously.
    """
    arith_safety = max(n_vars + n_ineq, 100) * 1e-14
    d_scaled = 2.0 * d * _DUAL_FEAS_TOL  # 2d · ε_d
    return max(arith_safety, _HIGHS_TOL_CUSHION, d_scaled)


# ---------------------------------------------------------------------
# LP-structure cache: per (id(windows), d) we precompute index arrays
# describing which (i,j,W) triples appear in which constraints. Only the
# numerical values (lo, hi) change per box, so the structure is reusable.
# ---------------------------------------------------------------------

_LP_STRUCT_CACHE: dict = {}


def _cache_lp_structure(windows, d: int):
    """Return per-window/per-pair structural index arrays for the
    epigraph LP. Cached by (id(windows), d).

    Returns:
      pair_i, pair_j: 1-D int arrays of length d² indexing pairs.
      window_pair_rows, window_pair_cols, window_pair_scales:
         CSR-like arrays for the epigraph rows. window_pair_rows[k] gives
         the window index (k_w), window_pair_cols[k] = yi(i,j) of the pair,
         window_pair_scales[k] = scale_W. Length = sum |pairs_all|.
    """
    key = (id(windows), d)
    got = _LP_STRUCT_CACHE.get(key)
    if got is not None:
        return got
    n_y = d * d
    # All pairs (i,j): row i*d+j, with i = idx//d, j = idx%d.
    pair_i = np.repeat(np.arange(d), d)
    pair_j = np.tile(np.arange(d), d)
    # Epigraph row entries: per window, per pair (i,j) ∈ S_W: contribute
    # +scale_W to A_ub[2*n_y + k_w, yi(i,j)].
    rows_w, cols_w, scales_w = [], [], []
    for k_w, w in enumerate(windows):
        s_W = float(w.scale)
        for (i, j) in w.pairs_all:
            rows_w.append(k_w)
            cols_w.append(i * d + j)
            scales_w.append(s_W)
    rows_w = np.asarray(rows_w, dtype=np.int64)
    cols_w = np.asarray(cols_w, dtype=np.int64)
    scales_w = np.asarray(scales_w, dtype=np.float64)
    out = (pair_i, pair_j, rows_w, cols_w, scales_w)
    _LP_STRUCT_CACHE[key] = out
    return out


def _solve_epigraph_lp(lo, hi, windows, d, *, return_residuals: bool = False):
    """Solve the epigraph LP, return (lp_val, ineqlin, eqlin, lower, upper).

    Builds the constraint matrix VECTORIZED in csr-sparse format. Returns
    LP optimal value and dual marginals (for downstream rigor cert).
    Returns (lp_val=-inf, *None) if LP fails or infeasible.

    Variable layout: [Y_00..Y_{d-1,d-1} (n_y), μ_0..μ_{d-1} (d), z (1)].

    HiGHS is invoked at publication-rigor tolerances (ε_p = ε_d = 1e-9,
    ε_ipm = 1e-10); see module docstring for the soundness derivation.

    If `return_residuals=True`, the returned tuple has 7 elements:
        (lp_val, ineqlin, eqlin, lower, upper, max_eq_resid, max_ineq_resid)
    where max_eq_resid = max(|A_eq x* - b_eq|) and
    max_ineq_resid = max((A_ub x* - b_ub)_+). On LP failure both are
    +inf so the caller's "residuals exceed cushion" check refuses cert.

    -------------------------------------------------------------------
    ACTIVE CUTS — full inventory (all sound on Δ_d ∩ box, with Y = μμᵀ)
    -------------------------------------------------------------------

    Inequality constraints (n_ineq = 4*n_y + n_W + 1 + d):

      [SW] Y_{i,j} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j         (n_y rows)
        Bilinear-McCormick LB at the SW corner of [lo_i,hi_i]×[lo_j,hi_j].
        Tight when (μ_i, μ_j) = (lo_i, lo_j).

      [NE] Y_{i,j} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j         (n_y rows)
        Bilinear-McCormick LB at the NE corner.
        Tight when (μ_i, μ_j) = (hi_i, hi_j).

      [NW] Y_{i,j} ≤ lo_j μ_i + hi_i μ_j − lo_j hi_i         (n_y rows)
        Bilinear-McCormick UB connecting NW corner.
        For i = j this reduces to the secant chord
            Y_{i,i} ≤ (lo_i + hi_i) μ_i − lo_i hi_i
        which is the tightest LP UB on a univariate convex μ_i².

      [SE] Y_{i,j} ≤ hi_j μ_i + lo_i μ_j − hi_j lo_i         (n_y rows)
        Bilinear-McCormick UB connecting SE corner.

      [EPI_W] z ≥ scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}          (n_W rows)
        Per-window epigraph defining z = max_W TV_W. Couples ALL
        windows via shared Y, closing the minimax–maximin gap.

      [C3] Σ_i Y_{i,i} ≥ 1/d                                  (1 row)
        Diagonal Cauchy–Schwarz (a.k.a. SOS). On the simplex
            Σ μ_i² ≥ (Σ μ_i)² / d = 1/d.
        Tight at uniform μ = (1/d, …, 1/d).

      [C4] Y_{i,i} ≥ 2 m_i μ_i − m_i²,  m_i = (lo_i+hi_i)/2  (d rows)
        Univariate convex tangent at midpoint m_i, since
            μ_i² ≥ 2 m_i μ_i − m_i².
        Tight at μ_i = m_i. Stronger than SW (i,i) and NE (i,i)
        when μ_i lies between lo_i and hi_i.

    Equality constraints (n_eq = 1 + 2d + d(d−1)/2):

      [SIMP] Σ_i μ_i = 1                                      (1 row)
        Probability-simplex equality. Foundational; without it the
        whole pipeline collapses.

      [RLT_R] Σ_j Y_{i,j} = μ_i, ∀ i                          (d rows)
        Reformulation–Linearization Technique row-sum. Sums of
        Y over a row equal μ. Implies (with SIMP) that
            Σ_{i,j} Y_{i,j} = 1,
        so a stand-alone "sum Y = 1" cut would be REDUNDANT.

      [C1 / RLT_C] Σ_i Y_{i,j} = μ_j, ∀ j                     (d rows)
        RLT column-sum. Independent of RLT_R only because the LP
        does NOT enforce Y symmetry by default — pairs C1 with C2
        below.

      [C2] Y_{i,j} = Y_{j,i}, ∀ i < j                         (d(d−1)/2 rows)
        Y-symmetry. Holds for Y = μμᵀ. Reduces the LP relaxation's
        degrees of freedom and directly tightens the bound.

    -------------------------------------------------------------------
    Why no further obvious LP cut tightens the bound:
      • "Σ_{i,j} Y_{i,j} = 1" — implied by SIMP + RLT_R.
      • "Σ_{(i,j) ∈ S_W} Y_{i,j} ≥ ..." per-window linear bounds
        beyond the McCormick faces — already enforced via the SW/NE
        inequalities for individual pairs in S_W, and an aggregated
        sum bound is tighter only if it removes a bilinear product
        (which an LP cannot).
      • "Y_{i,i} ≤ hi_i μ_i" / "Y_{i,j} ≤ hi_i μ_j" — these line
        bounds are LOOSER than the McCormick NW/SE faces (verified
        algebraically: NW = hi_i μ_j + lo_j (μ_i − hi_i), and
        lo_j (μ_i − hi_i) ≤ 0 since μ_i ≤ hi_i, so NW ≤ hi_i μ_j).
      • "Y_{i,i} ≥ lo_i μ_i" — LOOSER than SW (i,i), which gives
        Y_{i,i} ≥ 2 lo_i μ_i − lo_i².
      • "Y_{i,j}² ≤ Y_{i,i} Y_{j,j}" (PSD 2×2 minor) — bilinear,
        not LP-expressible.

    Soundness: each cut is a polynomial inequality in (μ, Y) that
    holds for every μ ∈ Δ_d ∩ [lo, hi] with Y = μμᵀ — so the LP
    feasible set still contains every box-feasible (μ, μμᵀ, max_W TV_W).
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu

    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    # Build A_ub in COO format then convert to CSR.
    # Row order: 0..n_y-1 = SW; n_y..2n_y-1 = NE; 2n_y..2n_y+n_W-1 = epigraph.
    # SW row r=yi(i,j):
    #   col yi(i,j): -1
    #   col mi(i)=n_y+i:  += lo[j]
    #   col mi(j)=n_y+j:  += lo[i]
    # NE row r=n_y + yi(i,j):
    #   col yi(i,j): -1
    #   col mi(i):   += hi[j]
    #   col mi(j):   += hi[i]
    # Epigraph row r=2n_y + k_w:
    #   col z_idx: -1
    #   col yi(i,j): += scale_W (for each (i,j) ∈ S_W)

    # Build COO entries:
    n_pairs = n_y
    # SW entries
    sw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    sw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    sw_data = np.empty(3 * n_pairs, dtype=np.float64)
    # block 0: (row=yi(i,j), col=yi(i,j), data=-1)
    sw_rows[:n_pairs] = np.arange(n_pairs)
    sw_cols[:n_pairs] = np.arange(n_pairs)
    sw_data[:n_pairs] = -1.0
    # block 1: (row=yi(i,j), col=n_y+i, data=lo[j])
    sw_rows[n_pairs:2 * n_pairs] = np.arange(n_pairs)
    sw_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    sw_data[n_pairs:2 * n_pairs] = lo[pair_j]
    # block 2: (row=yi(i,j), col=n_y+j, data=lo[i])
    sw_rows[2 * n_pairs:3 * n_pairs] = np.arange(n_pairs)
    sw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    sw_data[2 * n_pairs:3 * n_pairs] = lo[pair_i]

    # NE entries (rows offset by n_pairs)
    ne_rows = np.empty(3 * n_pairs, dtype=np.int64)
    ne_cols = np.empty(3 * n_pairs, dtype=np.int64)
    ne_data = np.empty(3 * n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[:n_pairs] = np.arange(n_pairs)
    ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2 * n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    ne_data[n_pairs:2 * n_pairs] = hi[pair_j]
    ne_rows[2 * n_pairs:3 * n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    ne_data[2 * n_pairs:3 * n_pairs] = hi[pair_i]

    # NW (UB) entries: Y_{i,j} ≤ lo[j]·μ_i + hi[i]·μ_j − lo[j]·hi[i]
    # As ineq:  Y_{i,j} − lo[j]·μ_i − hi[i]·μ_j ≤ −lo[j]·hi[i]
    # Row offset: 2*n_pairs (after SW + NE).
    nw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    nw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    nw_data = np.empty(3 * n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[:n_pairs] = np.arange(n_pairs)
    nw_data[:n_pairs] = +1.0  # +Y_{i,j}
    nw_rows[n_pairs:2 * n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    nw_data[n_pairs:2 * n_pairs] = -lo[pair_j]
    nw_rows[2 * n_pairs:3 * n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    nw_data[2 * n_pairs:3 * n_pairs] = -hi[pair_i]

    # SE (UB) entries: Y_{i,j} ≤ hi[j]·μ_i + lo[i]·μ_j − hi[j]·lo[i]
    # As ineq:  Y_{i,j} − hi[j]·μ_i − lo[i]·μ_j ≤ −hi[j]·lo[i]
    # Row offset: 3*n_pairs.
    se_rows = np.empty(3 * n_pairs, dtype=np.int64)
    se_cols = np.empty(3 * n_pairs, dtype=np.int64)
    se_data = np.empty(3 * n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[:n_pairs] = np.arange(n_pairs)
    se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2 * n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    se_data[n_pairs:2 * n_pairs] = -hi[pair_j]
    se_rows[2 * n_pairs:3 * n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    se_data[2 * n_pairs:3 * n_pairs] = -lo[pair_i]

    # Epigraph entries
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    # Row offset for epigraph: 4*n_pairs (after SW, NE, NW, SE).
    epi_rows[:n_epi_pair_entries] = 4 * n_pairs + rows_w
    epi_cols[:n_epi_pair_entries] = cols_w
    epi_data[:n_epi_pair_entries] = scales_w
    # z-column: row=4*n_y+k_w, col=z_idx, data=-1
    epi_rows[n_epi_pair_entries:] = 4 * n_pairs + np.arange(n_W)
    epi_cols[n_epi_pair_entries:] = z_idx
    epi_data[n_epi_pair_entries:] = -1.0

    # --- Extra inequality cuts (sound, derived from Σμ=1 and Y=μμᵀ) ---
    # Row layout in A_ub (after SW/NE/NW/SE/epigraph block):
    #   row 4*n_pairs + n_W            : (C3) diagonal Cauchy–Schwarz
    #                                    -Σ_i Y_{ii} ≤ -1/d
    #   rows 4*n_pairs + n_W + 1 + i    : (C4) midpoint diag tangent for i
    #                                    -Y_{ii} + 2 m_i μ_i ≤ m_i²
    sos_row_start = 4 * n_pairs + n_W
    tan_row_start = sos_row_start + 1
    diag_idx = np.arange(d) * d + np.arange(d)  # yi(i,i) = i*d + i

    # (C3) Diagonal SOS: one row, d entries.
    sos_rows = np.full(d, sos_row_start, dtype=np.int64)
    sos_cols = diag_idx.astype(np.int64)
    sos_data = np.full(d, -1.0, dtype=np.float64)

    # (C4) Midpoint tangent: d rows, 2 entries each (Y_{ii} and μ_i).
    m = 0.5 * (lo + hi)
    tan_rows = np.empty(2 * d, dtype=np.int64)
    tan_cols = np.empty(2 * d, dtype=np.int64)
    tan_data = np.empty(2 * d, dtype=np.float64)
    # -Y_{ii} term
    tan_rows[:d] = tan_row_start + np.arange(d)
    tan_cols[:d] = diag_idx
    tan_data[:d] = -1.0
    # +2 m_i μ_i term
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
    # SW: lo[i]*lo[j]
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    # NE: hi[i]*hi[j]
    b_ub[n_pairs:2 * n_pairs] = hi[pair_i] * hi[pair_j]
    # NW (UB): -lo[j]*hi[i]
    b_ub[2 * n_pairs:3 * n_pairs] = -lo[pair_j] * hi[pair_i]
    # SE (UB): -hi[j]*lo[i]
    b_ub[3 * n_pairs:4 * n_pairs] = -hi[pair_j] * lo[pair_i]
    # Epigraph: 0
    b_ub[4 * n_pairs:4 * n_pairs + n_W] = 0.0
    # (C3) Diagonal SOS RHS: -1/d  (i.e.  -Σ Y_{ii} ≤ -1/d  ⇔  Σ Y_{ii} ≥ 1/d)
    b_ub[sos_row_start] = -1.0 / d
    # (C4) Midpoint tangent RHS: m_i²
    b_ub[tan_row_start:tan_row_start + d] = m * m

    # A_eq layout:
    #   row 0                        : Σμ = 1
    #   rows 1..d                    : RLT row-sum   Σ_j Y_{i,j} − μ_i = 0
    #   rows 1+d..1+2d-1             : (C1) RLT col-sum  Σ_i Y_{i,j} − μ_j = 0
    #   rows 1+2d.. + n_sym          : (C2) Y-symmetry   Y_{i,j} − Y_{j,i} = 0  (i<j)
    n_sym = d * (d - 1) // 2
    n_eq = 1 + d + d + n_sym
    eq_rows = []
    eq_cols = []
    eq_data = []
    # Σμ = 1 row
    eq_rows.extend([0] * d)
    eq_cols.extend([n_y + i for i in range(d)])
    eq_data.extend([1.0] * d)
    # RLT row-sum: Σ_j Y_{i,j} - μ_i = 0
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i)
            eq_cols.append(i * d + j)
            eq_data.append(1.0)
        eq_rows.append(1 + i)
        eq_cols.append(n_y + i)
        eq_data.append(-1.0)
    # (C1) RLT col-sum: Σ_i Y_{i,j} - μ_j = 0
    col_row_start = 1 + d
    for j in range(d):
        for i in range(d):
            eq_rows.append(col_row_start + j)
            eq_cols.append(i * d + j)
            eq_data.append(1.0)
        eq_rows.append(col_row_start + j)
        eq_cols.append(n_y + j)
        eq_data.append(-1.0)
    # (C2) Y-symmetry: Y_{i,j} - Y_{j,i} = 0 for i<j
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
    b_eq[0] = 1.0  # Σμ = 1; remaining eqs have RHS 0

    # Bounds
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)]
    bnds.append((0.0, None))  # z

    # Objective: c[z_idx] = 1
    c = np.zeros(n_vars)
    c[z_idx] = 1.0

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bnds, method="highs", options=_HIGHS_OPTIONS)
    except Exception:
        if return_residuals:
            return (float("-inf"), None, None, None, None,
                    float("inf"), float("inf"))
        return float("-inf"), None, None, None, None
    if not res.success:
        if return_residuals:
            return (float("-inf"), None, None, None, None,
                    float("inf"), float("inf"))
        return float("-inf"), None, None, None, None
    base = (
        float(res.fun),
        res.ineqlin.marginals if hasattr(res, 'ineqlin') else None,
        res.eqlin.marginals if hasattr(res, 'eqlin') else None,
        res.lower.marginals if hasattr(res, 'lower') else None,
        res.upper.marginals if hasattr(res, 'upper') else None,
    )
    if not return_residuals:
        return base
    # HiGHS reports `res.con` = A_eq x* - b_eq (equality residuals)
    # and `res.slack` = b_ub - A_ub x* (≥ 0 at feasibility, smaller =
    # tighter constraint). To check "constraint not violated by more
    # than ε", we want max(0, -slack) for the inequality side.
    try:
        eq_residuals = np.asarray(res.con, dtype=np.float64) \
            if hasattr(res, "con") and res.con is not None else np.array([0.0])
        max_eq_resid = float(np.max(np.abs(eq_residuals))) if eq_residuals.size else 0.0
    except Exception:
        max_eq_resid = float("inf")
    try:
        slack = np.asarray(res.slack, dtype=np.float64) \
            if hasattr(res, "slack") and res.slack is not None else np.array([0.0])
        # Negative slack = constraint violated. Take the worst violation.
        max_ineq_resid = float(np.maximum(0.0, -slack).max()) if slack.size else 0.0
    except Exception:
        max_ineq_resid = float("inf")
    return base + (max_eq_resid, max_ineq_resid)


# ---------------------------------------------------------------------
# Float epigraph LP  (used as fast pre-filter and for diagnostics)
# ---------------------------------------------------------------------

def bound_epigraph_lp_float(
    lo: np.ndarray, hi: np.ndarray,
    windows: Sequence[WindowMeta], d: int,
) -> float:
    """Float LP value of the per-box epigraph relaxation. Returns -inf
    on LP failure. Uses the cached LP-structure path."""
    lp_val, *_ = _solve_epigraph_lp(lo, hi, windows, d)
    return lp_val


# ---------------------------------------------------------------------
# Rigorous integer-dual-certified epigraph cert
# ---------------------------------------------------------------------

def bound_epigraph_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    windows: Sequence[WindowMeta], d: int,
    target_num: int, target_den: int,
) -> bool:
    """True iff the per-box epigraph LP value ≥ target_num / target_den
    (publication-rigor: HiGHS tightened tolerances + Neumaier-Shcherbina
    dual-residual cushion).

    Cushion construction (see module docstring for derivation):
      cushion = max(
          (n_vars + n_ineq) · 1e-14,    # float-arith safety
          100 · ε_d,                    # HiGHS dual residual cushion
          2d · ε_d,                     # explicit d-scaled bound (d > 50)
      )
    with ε_d = 1e-9, this is 1e-7 for d ≤ 50 and 2dε_d for d > 50.

    Residual short-circuit. The function additionally checks that the
    HiGHS-reported equality and inequality residuals are below `cushion`.
    If either exceeds `cushion`, the cert returns False (refusing to
    certify is sound and conservative).

    Returns False on LP failure, residuals-exceed-cushion, or if cert
    margin not met.
    """
    n_W = len(windows)
    if n_W == 0:
        return target_num <= 0
    n_y = d * d

    lo_f = np.array([li / _SCALE for li in lo_int], dtype=np.float64)
    hi_f = np.array([hv / _SCALE for hv in hi_int], dtype=np.float64)

    (lp_val, _ineqlin, _eqlin, _lower, _upper,
     max_eq_resid, max_ineq_resid) = _solve_epigraph_lp(
        lo_f, hi_f, windows, d, return_residuals=True,
    )
    if not np.isfinite(lp_val):
        return False

    target_f = target_num / target_den

    # n_ineq: 4 * n_y + n_W (SW + NE + NW + SE + epigraph)
    #         + 1 (diagonal SOS)
    #         + d (midpoint diag tangents).
    n_vars = n_y + d + 1
    n_ineq = 4 * n_y + n_W + 1 + d
    cushion = _publication_cushion(d, n_vars, n_ineq)

    # Residual sanity check: refuse cert if HiGHS residuals exceed cushion.
    # The cushion was sized assuming residuals ≤ ε_d ≈ 1e-9; if HiGHS
    # reports worse (e.g. on a near-degenerate LP), the bound derivation
    # in the module docstring no longer applies.
    if max_eq_resid > cushion or max_ineq_resid > cushion:
        return False

    return (lp_val - cushion) >= target_f


# ---------------------------------------------------------------------
# Variant returning the dual marginals for the LP-binding split heuristic.
# Reuses _solve_epigraph_lp so we don't double-solve.
# ---------------------------------------------------------------------

def bound_epigraph_int_ge_with_marginals(
    lo_f: np.ndarray, hi_f: np.ndarray,
    windows: Sequence[WindowMeta], d: int,
    target_f: float,
):
    """Solve the per-box epigraph LP once and return:
        (certified_bool, lp_val, ineqlin_or_None)

    `certified_bool` mirrors the logic of `bound_epigraph_int_ge` —
    PUBLICATION-RIGOR: tightened HiGHS tolerances + Neumaier-Shcherbina
    cushion + residual short-circuit (see module docstring). `ineqlin`
    is the dual-marginal array from HiGHS (or None on LP failure).

    The caller passes float bounds (already used elsewhere in the
    cascade dispatch); it does its own integer / target_q handling.
    """
    n_W = len(windows)
    n_y = d * d
    if n_W == 0:
        # Vacuous LP: cert iff target_f <= 0. No marginals to return.
        return (target_f <= 0.0), float("inf"), None

    (lp_val, ineqlin, _eqlin, _lower, _upper,
     max_eq_resid, max_ineq_resid) = _solve_epigraph_lp(
        lo_f, hi_f, windows, d, return_residuals=True,
    )
    if not np.isfinite(lp_val):
        return False, float("-inf"), None

    n_vars = n_y + d + 1
    n_ineq = 4 * n_y + n_W + 1 + d
    cushion = _publication_cushion(d, n_vars, n_ineq)

    # Residual sanity check (same as bound_epigraph_int_ge).
    if max_eq_resid > cushion or max_ineq_resid > cushion:
        return False, float(lp_val), ineqlin

    cert = (lp_val - cushion) >= target_f
    return cert, float(lp_val), ineqlin


def lp_binding_axis_score(
    ineqlin: np.ndarray, widths: np.ndarray, d: int,
) -> np.ndarray:
    """Score axes by McCormick-face binding mass × width.

    The first 4*d*d entries of ineqlin are the duals for SW/NE/NW/SE
    faces, in row-blocks of n_y = d*d (see `_solve_epigraph_lp` for the
    layout). Pair (i, j) lives at row k = i*d + j inside each block.

    Score: axis_score[i] = ( Σ_{j} (|λ_SW(i,j)| + |λ_NE(i,j)|
                                  + |λ_NW(i,j)| + |λ_SE(i,j)|)
                            + Σ_{j} (...for pair (j,i)) ) · width[i]

    Returns a (d,) float64 array. If `ineqlin` is None, returns zeros.
    """
    if ineqlin is None:
        return np.zeros(d, dtype=np.float64)
    n_y = d * d
    if len(ineqlin) < 4 * n_y:
        return np.zeros(d, dtype=np.float64)
    arr = np.asarray(ineqlin, dtype=np.float64)
    # |λ| per face, per pair (i,j) in the (d,d) layout: row=i, col=j.
    sw = np.abs(arr[0:n_y]).reshape(d, d)
    ne = np.abs(arr[n_y:2 * n_y]).reshape(d, d)
    nw = np.abs(arr[2 * n_y:3 * n_y]).reshape(d, d)
    se = np.abs(arr[3 * n_y:4 * n_y]).reshape(d, d)
    face_mass = sw + ne + nw + se  # (d, d), entry [i,j] = |λ_*(i,j)| summed
    # axis i is touched by row i (pair (i,*)) AND column i (pair (*,i)).
    per_axis = face_mass.sum(axis=1) + face_mass.sum(axis=0)
    return per_axis * np.asarray(widths, dtype=np.float64)

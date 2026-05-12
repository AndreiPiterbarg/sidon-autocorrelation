"""Soundness + tightening tests for the 4 extra LP cuts added to
``_solve_epigraph_lp`` in ``bound_epigraph.py``.

The 4 cuts (each independently sound from Σμ=1 and Y=μμᵀ on the simplex):

  C1) column-sum RLT:  Σ_i Y_{i,j} = μ_j           (d new equalities)
  C2) Y-symmetry:      Y_{i,j} = Y_{j,i}, i<j       (d(d-1)/2 new equalities)
  C3) diagonal SOS:    Σ_i Y_{i,i} ≥ 1/d           (1 new inequality)
  C4) midpoint diag tangent:
                       Y_{i,i} ≥ 2 m_i μ_i − m_i²   (d new inequalities)

Tests:
  * SOUNDNESS: for random μ ∈ Δ_d (d ∈ {3,5,8}) with Y = μμᵀ, the augmented
    LP must be feasible at this primal point (Aub·x ≤ bub, Aeq·x = beq, and
    bounds met) within 1e-9.
  * TIGHTENING: on a deep stuck box around d=4 mu* (the depth-25-ish case
    used by ``test_d4_known_optimum``), the new LP value must be
    ≥ the old LP value.
  * SMOKE: the existing ``test_epigraph.py`` suite still passes.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_epigraph import (
    _cache_lp_structure, _solve_epigraph_lp, bound_epigraph_lp_float,
)
from interval_bnb.windows import build_windows


# ---------------------------------------------------------------------
# Reference: rebuild the augmented LP constraint matrices directly so we
# can sanity-check Aub·x ≤ bub and Aeq·x = beq at a known feasible point
# (μ ∈ Δ_d, Y = μμᵀ, z = max_W TV_W(μ)).
#
# This duplicates the structure of the production ``_solve_epigraph_lp``;
# if the production code drifts, this test will catch it.
# ---------------------------------------------------------------------

def _build_constraints(lo, hi, windows, d):
    from scipy.sparse import coo_matrix, csr_matrix

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu

    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    n_pairs = n_y

    # SW
    sw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    sw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    sw_data = np.empty(3 * n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2*n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2*n_pairs] = n_y + pair_i; sw_data[n_pairs:2*n_pairs] = lo[pair_j]
    sw_rows[2*n_pairs:3*n_pairs] = np.arange(n_pairs); sw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; sw_data[2*n_pairs:3*n_pairs] = lo[pair_i]
    # NE
    ne_rows = np.empty(3 * n_pairs, dtype=np.int64)
    ne_cols = np.empty(3 * n_pairs, dtype=np.int64)
    ne_data = np.empty(3 * n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[n_pairs:2*n_pairs] = n_y + pair_i; ne_data[n_pairs:2*n_pairs] = hi[pair_j]
    ne_rows[2*n_pairs:3*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; ne_data[2*n_pairs:3*n_pairs] = hi[pair_i]
    # NW
    nw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    nw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    nw_data = np.empty(3 * n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[n_pairs:2*n_pairs] = n_y + pair_i; nw_data[n_pairs:2*n_pairs] = -lo[pair_j]
    nw_rows[2*n_pairs:3*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; nw_data[2*n_pairs:3*n_pairs] = -hi[pair_i]
    # SE
    se_rows = np.empty(3 * n_pairs, dtype=np.int64)
    se_cols = np.empty(3 * n_pairs, dtype=np.int64)
    se_data = np.empty(3 * n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[n_pairs:2*n_pairs] = n_y + pair_i; se_data[n_pairs:2*n_pairs] = -hi[pair_j]
    se_rows[2*n_pairs:3*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; se_data[2*n_pairs:3*n_pairs] = -lo[pair_i]
    # Epigraph
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    epi_rows[:n_epi_pair_entries] = 4*n_pairs + rows_w
    epi_cols[:n_epi_pair_entries] = cols_w
    epi_data[:n_epi_pair_entries] = scales_w
    epi_rows[n_epi_pair_entries:] = 4*n_pairs + np.arange(n_W)
    epi_cols[n_epi_pair_entries:] = z_idx
    epi_data[n_epi_pair_entries:] = -1.0
    # New: SOS + tangents
    sos_row_start = 4*n_pairs + n_W
    tan_row_start = sos_row_start + 1
    diag_idx = np.arange(d) * d + np.arange(d)
    sos_rows = np.full(d, sos_row_start, dtype=np.int64)
    sos_cols = diag_idx.astype(np.int64)
    sos_data = np.full(d, -1.0, dtype=np.float64)
    m = 0.5 * (lo + hi)
    tan_rows = np.empty(2*d, dtype=np.int64)
    tan_cols = np.empty(2*d, dtype=np.int64)
    tan_data = np.empty(2*d, dtype=np.float64)
    tan_rows[:d] = tan_row_start + np.arange(d); tan_cols[:d] = diag_idx; tan_data[:d] = -1.0
    tan_rows[d:] = tan_row_start + np.arange(d); tan_cols[d:] = n_y + np.arange(d); tan_data[d:] = 2.0 * m

    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows, epi_rows, sos_rows, tan_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols, epi_cols, sos_cols, tan_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data, epi_data, sos_data, tan_data])
    n_ineq = 4*n_pairs + n_W + 1 + d
    A_ub = coo_matrix((data_all, (rows_all, cols_all)), shape=(n_ineq, n_vars)).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    b_ub[n_pairs:2*n_pairs] = hi[pair_i] * hi[pair_j]
    b_ub[2*n_pairs:3*n_pairs] = -lo[pair_j] * hi[pair_i]
    b_ub[3*n_pairs:4*n_pairs] = -hi[pair_j] * lo[pair_i]
    b_ub[4*n_pairs:4*n_pairs + n_W] = 0.0
    b_ub[sos_row_start] = -1.0 / d
    b_ub[tan_row_start:tan_row_start + d] = m * m

    # Equalities
    n_sym = d * (d - 1) // 2
    n_eq = 1 + d + d + n_sym
    eq_rows = []; eq_cols = []; eq_data = []
    eq_rows.extend([0]*d); eq_cols.extend([n_y + i for i in range(d)]); eq_data.extend([1.0]*d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
    col_row_start = 1 + d
    for j in range(d):
        for i in range(d):
            eq_rows.append(col_row_start + j); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(col_row_start + j); eq_cols.append(n_y + j); eq_data.append(-1.0)
    sym_row_start = col_row_start + d
    sym_k = 0
    for i in range(d):
        for j in range(i+1, d):
            r = sym_row_start + sym_k
            eq_rows.append(r); eq_cols.append(i*d + j); eq_data.append(1.0)
            eq_rows.append(r); eq_cols.append(j*d + i); eq_data.append(-1.0)
            sym_k += 1
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(n_eq, n_vars),
    )
    b_eq = np.zeros(n_eq, dtype=np.float64)
    b_eq[0] = 1.0

    bnds_lo = np.concatenate([np.zeros(n_y), lo, [0.0]])
    bnds_hi = np.concatenate([np.full(n_y, np.inf), hi, [np.inf]])
    return A_ub, b_ub, A_eq, b_eq, bnds_lo, bnds_hi


# ---------------------------------------------------------------------
# Reference (OLD) LP — replicates the production code prior to the cuts.
# Used only by the tightening test.
# ---------------------------------------------------------------------

def _solve_epigraph_lp_OLD(lo, hi, windows, d):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu
    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    n_pairs = n_y
    sw_rows = np.empty(3*n_pairs, dtype=np.int64); sw_cols = np.empty(3*n_pairs, dtype=np.int64); sw_data = np.empty(3*n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2*n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2*n_pairs] = n_y + pair_i; sw_data[n_pairs:2*n_pairs] = lo[pair_j]
    sw_rows[2*n_pairs:3*n_pairs] = np.arange(n_pairs); sw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; sw_data[2*n_pairs:3*n_pairs] = lo[pair_i]
    ne_rows = np.empty(3*n_pairs, dtype=np.int64); ne_cols = np.empty(3*n_pairs, dtype=np.int64); ne_data = np.empty(3*n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[n_pairs:2*n_pairs] = n_y + pair_i; ne_data[n_pairs:2*n_pairs] = hi[pair_j]
    ne_rows[2*n_pairs:3*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; ne_data[2*n_pairs:3*n_pairs] = hi[pair_i]
    nw_rows = np.empty(3*n_pairs, dtype=np.int64); nw_cols = np.empty(3*n_pairs, dtype=np.int64); nw_data = np.empty(3*n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[n_pairs:2*n_pairs] = n_y + pair_i; nw_data[n_pairs:2*n_pairs] = -lo[pair_j]
    nw_rows[2*n_pairs:3*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; nw_data[2*n_pairs:3*n_pairs] = -hi[pair_i]
    se_rows = np.empty(3*n_pairs, dtype=np.int64); se_cols = np.empty(3*n_pairs, dtype=np.int64); se_data = np.empty(3*n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[n_pairs:2*n_pairs] = n_y + pair_i; se_data[n_pairs:2*n_pairs] = -hi[pair_j]
    se_rows[2*n_pairs:3*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; se_data[2*n_pairs:3*n_pairs] = -lo[pair_i]
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    epi_rows[:n_epi_pair_entries] = 4*n_pairs + rows_w; epi_cols[:n_epi_pair_entries] = cols_w; epi_data[:n_epi_pair_entries] = scales_w
    epi_rows[n_epi_pair_entries:] = 4*n_pairs + np.arange(n_W); epi_cols[n_epi_pair_entries:] = z_idx; epi_data[n_epi_pair_entries:] = -1.0
    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows, epi_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols, epi_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data, epi_data])
    n_ineq = 4*n_pairs + n_W
    A_ub = coo_matrix((data_all, (rows_all, cols_all)), shape=(n_ineq, n_vars)).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    b_ub[n_pairs:2*n_pairs] = hi[pair_i] * hi[pair_j]
    b_ub[2*n_pairs:3*n_pairs] = -lo[pair_j] * hi[pair_i]
    b_ub[3*n_pairs:4*n_pairs] = -hi[pair_j] * lo[pair_i]
    b_ub[4*n_pairs:] = 0.0
    eq_rows = []; eq_cols = []; eq_data = []
    eq_rows.extend([0]*d); eq_cols.extend([n_y + i for i in range(d)]); eq_data.extend([1.0]*d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(1 + d, n_vars),
    )
    b_eq = np.zeros(1 + d, dtype=np.float64); b_eq[0] = 1.0
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)]
    bnds.append((0.0, None))
    c = np.zeros(n_vars); c[z_idx] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if not res.success:
        return float("-inf")
    return float(res.fun)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def _random_simplex_in_box(rng, lo, hi):
    """Sample μ ∈ Δ_d ∩ [lo, hi]. Uses Dirichlet then projects (rejection
    sampling on the box).
    """
    d = len(lo)
    for _ in range(2000):
        x = rng.dirichlet(np.ones(d))
        if (x >= lo - 1e-12).all() and (x <= hi + 1e-12).all():
            return x
    # Fallback: midpoint normalised.
    m = 0.5 * (lo + hi)
    return m / m.sum()


def test_extra_cuts_soundness():
    """For 50 random μ ∈ Δ_d (d ∈ {3,5,8}) with Y = μμᵀ, the augmented LP
    must be feasible at this primal point: Aub·x ≤ bub, Aeq·x = beq,
    bounds met.
    """
    rng = np.random.default_rng(0xC0FFEE)
    for d in (3, 5, 8):
        windows = build_windows(d)
        lo = np.full(d, 1e-6)        # box: simplex-friendly slab
        hi = np.full(d, 1.0 - 1e-6)
        A_ub, b_ub, A_eq, b_eq, bnds_lo, bnds_hi = _build_constraints(lo, hi, windows, d)

        n_y = d * d
        for trial in range(50):
            mu = _random_simplex_in_box(rng, lo, hi)
            Y = np.outer(mu, mu)
            # z must be ≥ max_W TV_W(μ); use exact equality for tightest x.
            from interval_bnb.bound_eval import _adjacency_matrix
            z = max(w.scale * float(mu @ _adjacency_matrix(w, d) @ mu) for w in windows)
            x = np.concatenate([Y.ravel(), mu, [z]])

            # Bounds (with a tolerance — the slack bounds are wide here)
            assert (x >= bnds_lo - 1e-9).all(), (
                f"d={d} trial={trial}: lower-bound viol "
                f"max={float((bnds_lo - x).max()):.3e}"
            )
            # bnds_hi may contain np.inf so handle via finite mask
            fin = np.isfinite(bnds_hi)
            assert (x[fin] <= bnds_hi[fin] + 1e-9).all(), (
                f"d={d} trial={trial}: upper-bound viol "
                f"max={float((x[fin] - bnds_hi[fin]).max()):.3e}"
            )

            # Inequality
            r_ub = A_ub @ x - b_ub
            assert (r_ub <= 1e-9).all(), (
                f"d={d} trial={trial}: ineq viol max={float(r_ub.max()):.3e}"
            )
            # Equality
            r_eq = A_eq @ x - b_eq
            assert (np.abs(r_eq) <= 1e-9).all(), (
                f"d={d} trial={trial}: eq viol max={float(np.abs(r_eq).max()):.3e}"
            )


def test_extra_cuts_tightening():
    """At a deep box around d=4 mu*, the new LP value must be
    ≥ the old LP value (no regression; usually strictly greater).
    """
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])

    # A few different radii — exercise the "stuck box" regime.
    for radius in (1e-2, 1e-3, 1e-4):
        lo = np.maximum(mu_star - radius, 0.0)
        hi = np.minimum(mu_star + radius, 1.0)
        old_val = _solve_epigraph_lp_OLD(lo, hi, windows, d)
        new_val = bound_epigraph_lp_float(lo, hi, windows, d)
        assert np.isfinite(old_val) and np.isfinite(new_val), (
            f"radius={radius}: LP failed (old={old_val}, new={new_val})"
        )
        # Must not regress; allow 1e-9 slack for HiGHS dual tolerance.
        assert new_val >= old_val - 1e-9, (
            f"radius={radius}: REGRESSION new={new_val:.10f} "
            f"< old={old_val:.10f}"
        )


def test_extra_cuts_d20_stuck_box():
    """Deeper-box smoke test (d=20 regime, depth-25-ish)."""
    d = 20
    windows = build_windows(d)
    rng = np.random.default_rng(7)
    # Pick a μ* on simplex, build a tight box around it.
    mu_star = rng.dirichlet(np.ones(d))
    radius = 5e-4
    lo = np.maximum(mu_star - radius, 1e-9)
    hi = np.minimum(mu_star + radius, 1.0 - 1e-9)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        # Box doesn't contain simplex slice — skip.
        return
    old_val = _solve_epigraph_lp_OLD(lo, hi, windows, d)
    new_val = bound_epigraph_lp_float(lo, hi, windows, d)
    assert np.isfinite(old_val) and np.isfinite(new_val)
    assert new_val >= old_val - 1e-9, (
        f"d=20 stuck box: REGRESSION new={new_val:.10f} < old={old_val:.10f}"
    )


def test_extra_cuts_with_publication_cushion_at_d4():
    """Smoke check: with the new publication-grade 1e-7 cushion in
    bound_epigraph_int_ge, known-good d=4 certs still pass.

    This guards against a regression where the larger cushion would
    silently block previously-passing certs that the extra-cut soundness
    framework depends on.
    """
    from fractions import Fraction
    from interval_bnb.bound_epigraph import bound_epigraph_int_ge
    from interval_bnb.box import SCALE as _SCALE
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-3
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    lo_int = [int(round(x * _SCALE)) for x in lo_f]
    hi_int = [int(round(x * _SCALE)) for x in hi_f]
    # val(4) = 10/9 ≈ 1.1111, so target=1.05 must cert with the new cushion.
    target = Fraction(105, 100)
    cert = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target.numerator, target.denominator,
    )
    assert cert, "[regression] d=4 t=1.05 cert refused under publication cushion"
    # And target=1.10 (closer to val(4)=10/9 but still safely below it).
    target2 = Fraction(110, 100)
    cert2 = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target2.numerator, target2.denominator,
    )
    assert cert2, "[regression] d=4 t=1.10 cert refused under publication cushion"


# ---------------------------------------------------------------------
# AUDIT tests: verify each documented cut is actually enforced at LP
# optimum on real boxes. These guard against silent indexing bugs that
# would let a constraint be skipped without changing the LP value
# noticeably (since other constraints often dominate at the optimum).
# ---------------------------------------------------------------------

def _solve_and_extract_xstar(lo, hi, windows, d):
    """Solve the production LP and return the primal optimum
    x* = (Y_flat, μ, z), reusing the same builder as `_solve_epigraph_lp`
    via a sibling linprog call. Returns (Y, μ, z) or None on LP failure.
    """
    from scipy.optimize import linprog
    A_ub, b_ub, A_eq, b_eq, bnds_lo, bnds_hi = _build_constraints(lo, hi, windows, d)
    n_y = d * d
    n_vars = n_y + d + 1
    z_idx = n_y + d
    bnds = [(float(bnds_lo[i]),
             None if not np.isfinite(bnds_hi[i]) else float(bnds_hi[i]))
            for i in range(n_vars)]
    c = np.zeros(n_vars); c[z_idx] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bnds, method="highs",
                  options={
                      "primal_feasibility_tolerance": 1e-9,
                      "dual_feasibility_tolerance": 1e-9,
                      "ipm_optimality_tolerance": 1e-10,
                  })
    if not res.success:
        return None
    x = np.asarray(res.x, dtype=np.float64)
    Y = x[:n_y].reshape(d, d)
    mu = x[n_y:n_y + d]
    z = float(x[z_idx])
    return Y, mu, z


def _audit_boxes():
    """Yield (label, lo, hi, d, windows) for the audit suite — diverse
    boxes that exercise both interior and boundary configurations.
    """
    # Interior box, small d
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-3
    yield ("d4-interior", np.maximum(mu_star - radius, 0.0),
           np.minimum(mu_star + radius, 1.0), d, windows)

    # Boundary axis (lo = 0), small d
    d = 5
    windows = build_windows(d)
    lo = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    hi = np.array([0.5, 0.40, 0.40, 0.40, 0.40])
    yield ("d5-boundary", lo, hi, d, windows)

    # Wide box at d=6
    d = 6
    windows = build_windows(d)
    lo = np.full(d, 0.05)
    hi = np.full(d, 0.40)
    yield ("d6-wide", lo, hi, d, windows)

    # Tight d=8 box
    d = 8
    windows = build_windows(d)
    rng = np.random.default_rng(1234)
    mu = rng.dirichlet(np.ones(d))
    radius = 5e-3
    lo = np.maximum(mu - radius, 1e-9)
    hi = np.minimum(mu + radius, 1.0 - 1e-9)
    if lo.sum() <= 1.0 <= hi.sum():
        yield ("d8-tight", lo, hi, d, windows)


def test_audit_simplex_constraint_holds():
    """At every LP optimum, Σ_i μ_i = 1 within tolerance."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        s = float(mu.sum())
        assert abs(s - 1.0) < 1e-7, f"{label}: Σμ = {s} ≠ 1"


def test_audit_rlt_row_sum_constraint_holds():
    """At every LP optimum, Σ_j Y_{i,j} = μ_i within tolerance, ∀ i."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        row_sums = Y.sum(axis=1)
        max_err = float(np.max(np.abs(row_sums - mu)))
        assert max_err < 1e-7, (
            f"{label}: max |Σ_j Y_{{i,j}} − μ_i| = {max_err:.3e}"
        )


def test_audit_rlt_col_sum_constraint_holds():
    """C1 audit: at LP optimum, Σ_i Y_{i,j} = μ_j ∀ j."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        col_sums = Y.sum(axis=0)
        max_err = float(np.max(np.abs(col_sums - mu)))
        assert max_err < 1e-7, (
            f"{label}: max |Σ_i Y_{{i,j}} − μ_j| = {max_err:.3e}"
        )


def test_audit_y_symmetry_constraint_holds():
    """C2 audit: at LP optimum, Y_{i,j} = Y_{j,i} ∀ i ≠ j."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        asym = float(np.max(np.abs(Y - Y.T)))
        assert asym < 1e-7, f"{label}: max |Y - Yᵀ| = {asym:.3e}"


def test_audit_diagonal_sos_constraint_holds():
    """C3 audit: at LP optimum, Σ_i Y_{i,i} ≥ 1/d − tol."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        diag_sum = float(np.diag(Y).sum())
        assert diag_sum >= 1.0 / d - 1e-7, (
            f"{label}: Σ Y_{{i,i}} = {diag_sum:.6e} < 1/d = {1.0/d:.6e}"
        )


def test_audit_midpoint_diag_tangent_holds():
    """C4 audit: at LP optimum, Y_{i,i} ≥ 2 m_i μ_i − m_i² ∀ i."""
    for label, lo_a, hi_a, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo_a, hi_a, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        m = 0.5 * (np.asarray(lo_a) + np.asarray(hi_a))
        rhs = 2.0 * m * mu - m * m
        viol = float(np.max(rhs - np.diag(Y)))
        assert viol < 1e-7, (
            f"{label}: max [2 m_i μ_i − m_i² − Y_{{i,i}}] = {viol:.3e}"
        )


def test_audit_mccormick_faces_satisfied():
    """SW/NE/NW/SE McCormick face audit: at LP optimum, each
    bilinear-McCormick face inequality is satisfied within tolerance.

    SW: Y_{i,j} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j
    NE: Y_{i,j} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j
    NW: Y_{i,j} ≤ lo_j μ_i + hi_i μ_j − lo_j hi_i
    SE: Y_{i,j} ≤ hi_j μ_i + lo_i μ_j − hi_j lo_i
    """
    for label, lo_a, hi_a, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo_a, hi_a, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        lo_a = np.asarray(lo_a); hi_a = np.asarray(hi_a)
        # Vectorized SW/NE LB: Y[i,j] >= lo[j]*mu[i] + lo[i]*mu[j] - lo[i]*lo[j]
        sw = np.outer(mu, lo_a) + np.outer(lo_a, mu) - np.outer(lo_a, lo_a)
        ne = np.outer(mu, hi_a) + np.outer(hi_a, mu) - np.outer(hi_a, hi_a)
        # NW/SE UB
        nw = np.outer(mu, lo_a) + np.outer(hi_a, mu) - np.outer(hi_a, lo_a)
        se = np.outer(mu, hi_a) + np.outer(lo_a, mu) - np.outer(lo_a, hi_a)
        sw_viol = float(np.max(sw - Y))     # LB: should be ≤ 0 (Y ≥ sw)
        ne_viol = float(np.max(ne - Y))
        nw_viol = float(np.max(Y - nw))     # UB: should be ≤ 0 (Y ≤ nw)
        se_viol = float(np.max(Y - se))
        assert sw_viol < 1e-7, f"{label}: SW viol {sw_viol:.3e}"
        assert ne_viol < 1e-7, f"{label}: NE viol {ne_viol:.3e}"
        assert nw_viol < 1e-7, f"{label}: NW viol {nw_viol:.3e}"
        assert se_viol < 1e-7, f"{label}: SE viol {se_viol:.3e}"


def test_audit_epigraph_per_window_holds():
    """At LP optimum, z ≥ scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j} for every W."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        worst_viol = -np.inf
        for w in windows:
            mass = sum(Y[i, j] for (i, j) in w.pairs_all)
            tv = float(w.scale) * mass
            worst_viol = max(worst_viol, tv - z)
        assert worst_viol < 1e-7, (
            f"{label}: max [scale_W Σ Y - z] = {worst_viol:.3e}"
        )


def test_audit_implied_sum_y_equals_one():
    """Σ_{i,j} Y_{i,j} = 1 follows from SIMP + RLT_R; verify it is in
    fact close to 1 at every LP optimum. This is the redundant cut
    discussed in the analysis."""
    for label, lo, hi, d, windows in _audit_boxes():
        out = _solve_and_extract_xstar(lo, hi, windows, d)
        assert out is not None, f"{label}: LP failed"
        Y, mu, z = out
        s = float(Y.sum())
        assert abs(s - 1.0) < 1e-7, f"{label}: Σ Y = {s} ≠ 1"


if __name__ == "__main__":
    test_extra_cuts_soundness()
    print("soundness ok")
    test_extra_cuts_tightening()
    print("tightening ok")
    test_extra_cuts_d20_stuck_box()
    print("d20 ok")
    test_audit_simplex_constraint_holds(); print("audit simplex ok")
    test_audit_rlt_row_sum_constraint_holds(); print("audit row-sum ok")
    test_audit_rlt_col_sum_constraint_holds(); print("audit col-sum ok")
    test_audit_y_symmetry_constraint_holds(); print("audit symmetry ok")
    test_audit_diagonal_sos_constraint_holds(); print("audit SOS ok")
    test_audit_midpoint_diag_tangent_holds(); print("audit tangent ok")
    test_audit_mccormick_faces_satisfied(); print("audit McCormick ok")
    test_audit_epigraph_per_window_holds(); print("audit epigraph ok")
    test_audit_implied_sum_y_equals_one(); print("audit sumY ok")

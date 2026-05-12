"""Microbench: scipy.linprog vs highspy direct vs highspy warm-start.

Compares 3 implementations of the per-box epigraph LP:
  A. scipy.optimize.linprog(method='highs')  (CURRENT)
  B. highspy.Highs() with passModel per call (NO WARM START)
  C. highspy.Highs() reused; per-box changeCoeff/changeBounds (WARM START)

Runs on a fixed sample of real boxes from children_after_lp.npz, reports
per-call wall time and verifies LP values agree across implementations.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from interval_bnb.box import SCALE as _SCALE
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import bound_epigraph_lp_float, _cache_lp_structure


# =====================================================================
#  Variant B & C builder: precompute static structure (CSR pattern)
# =====================================================================

class FastEpigraphLP:
    """Build the epigraph LP once; mutate per-box.

    Structure (sparsity pattern) is invariant; only:
      * SW/NE/NW/SE inequality data (12 * n_y entries) and rhs depend on lo/hi
      * midpoint-tangent data (d entries) and rhs depend on m=(lo+hi)/2
      * μ_i column bounds [lo_i, hi_i] depend on lo, hi
    """
    def __init__(self, d: int, windows, options: dict):
        import highspy
        from highspy import Highs, HighsLp, HighsOptions, ObjSense
        self.d = d
        self.windows = windows
        self.n_W = len(windows)
        self.n_y = d * d
        self.n_mu = d
        self.n_vars = self.n_y + d + 1
        self.z_idx = self.n_y + d

        pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
        self.pair_i = pair_i
        self.pair_j = pair_j
        self.rows_w = rows_w
        self.cols_w = cols_w
        self.scales_w = scales_w

        n_pairs = self.n_y
        self.n_pairs = n_pairs
        n_W = self.n_W

        n_ineq = 4 * n_pairs + n_W + 1 + d
        n_sym = d * (d - 1) // 2
        n_eq = 1 + 2 * d + n_sym

        self.n_ineq = n_ineq
        self.n_eq = n_eq

        # Build the FULL constraint matrix once. Coefficients that depend
        # on lo/hi get filled with zeros initially (will be mutated per box).
        # This avoids building structure per call.
        self._build_static_matrices()

        # Init Highs
        h = Highs()
        h.setOptionValue('output_flag', False)
        for k, v in options.items():
            try:
                h.setOptionValue(k, v)
            except Exception:
                pass

        # Pass full LP. Use COMBINED A: stack A_eq on top of A_ub, with
        # row bounds matching equality (b=b) or inequality (-inf, b).
        # Highs supports row bound (lo, hi) where lo = -inf for ineq.
        self._pass_initial_model(h, options)
        self.h = h

        # Cache mutable index arrays (where to overwrite per-box).
        # In CSR format A is stored as values[ptr[r]:ptr[r+1]] = data.
        # changeCoeff is row,col-indexed → we need (row, col, value) triples.
        self._cache_mutation_indices()

    def _build_static_matrices(self):
        """Build A as ONE CSR with rows = [eq rows ; ineq rows] (n_eq + n_ineq)."""
        from scipy.sparse import csr_matrix, coo_matrix
        d = self.d
        n_y = self.n_y
        n_pairs = self.n_pairs
        n_W = self.n_W
        n_eq = self.n_eq

        # ---- Static A_eq part (does NOT depend on lo/hi) ----
        eq_rows = []
        eq_cols = []
        eq_data = []
        # Σμ = 1
        for i in range(d):
            eq_rows.append(0); eq_cols.append(n_y + i); eq_data.append(1.0)
        # RLT row: Σ_j Y_{i,j} − μ_i = 0  for each i
        for i in range(d):
            for j in range(d):
                eq_rows.append(1 + i); eq_cols.append(i * d + j); eq_data.append(1.0)
            eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
        # RLT col: Σ_i Y_{i,j} − μ_j = 0  for each j
        col_row_start = 1 + d
        for j in range(d):
            for i in range(d):
                eq_rows.append(col_row_start + j); eq_cols.append(i * d + j); eq_data.append(1.0)
            eq_rows.append(col_row_start + j); eq_cols.append(n_y + j); eq_data.append(-1.0)
        # Y-symmetry  Y_{i,j} − Y_{j,i} = 0,  i<j
        sym_row_start = col_row_start + d
        sym_k = 0
        for i in range(d):
            for j in range(i + 1, d):
                r = sym_row_start + sym_k
                eq_rows.append(r); eq_cols.append(i * d + j); eq_data.append(1.0)
                eq_rows.append(r); eq_cols.append(j * d + i); eq_data.append(-1.0)
                sym_k += 1
        self.b_eq = np.zeros(n_eq, dtype=np.float64)
        self.b_eq[0] = 1.0

        eq_rows = np.array(eq_rows, dtype=np.int64)
        eq_cols = np.array(eq_cols, dtype=np.int64)
        eq_data = np.array(eq_data, dtype=np.float64)

        # ---- Ineq A_ub: SW/NE/NW/SE (data PLACEHOLDER) + epi + sos + tan ----
        # SW row r=p: col yi=p data=-1 (constant); col n_y+pair_i[p]: lo[pair_j[p]];
        #             col n_y+pair_j[p]: lo[pair_i[p]] (BOTH placeholders 0 here)
        # We pre-allocate the structure with all three cells per row.
        ub_rows_sw = np.repeat(np.arange(n_pairs), 3)  # rows 0..n_pairs-1, 3x each
        ub_cols_sw = np.empty(3 * n_pairs, dtype=np.int64)
        ub_cols_sw[0::3] = np.arange(n_pairs)        # col yi
        ub_cols_sw[1::3] = n_y + self.pair_i         # col n_y+i
        ub_cols_sw[2::3] = n_y + self.pair_j         # col n_y+j
        ub_data_sw = np.zeros(3 * n_pairs, dtype=np.float64)
        ub_data_sw[0::3] = -1.0  # constant

        # Same shape for NE
        ub_rows_ne = np.repeat(n_pairs + np.arange(n_pairs), 3)
        ub_cols_ne = ub_cols_sw.copy()
        ub_data_ne = np.zeros(3 * n_pairs, dtype=np.float64)
        ub_data_ne[0::3] = -1.0

        # NW
        ub_rows_nw = np.repeat(2 * n_pairs + np.arange(n_pairs), 3)
        ub_cols_nw = ub_cols_sw.copy()
        ub_data_nw = np.zeros(3 * n_pairs, dtype=np.float64)
        ub_data_nw[0::3] = +1.0

        # SE
        ub_rows_se = np.repeat(3 * n_pairs + np.arange(n_pairs), 3)
        ub_cols_se = ub_cols_sw.copy()
        ub_data_se = np.zeros(3 * n_pairs, dtype=np.float64)
        ub_data_se[0::3] = +1.0

        # Epigraph rows (CONSTANT)
        ep_rows = []
        ep_cols = []
        ep_data = []
        for k_w in range(n_W):
            # iterate over the entries for this window
            mask = self.rows_w == k_w
            for c, s in zip(self.cols_w[mask], self.scales_w[mask]):
                ep_rows.append(4 * n_pairs + k_w)
                ep_cols.append(int(c))
                ep_data.append(float(s))
            ep_rows.append(4 * n_pairs + k_w)
            ep_cols.append(self.z_idx)
            ep_data.append(-1.0)
        ep_rows = np.array(ep_rows, dtype=np.int64)
        ep_cols = np.array(ep_cols, dtype=np.int64)
        ep_data = np.array(ep_data, dtype=np.float64)

        # SOS row (constant: -Y_ii ≤ -1/d)
        diag_idx = np.arange(d) * d + np.arange(d)
        sos_row_idx = 4 * n_pairs + n_W
        sos_rows = np.full(d, sos_row_idx, dtype=np.int64)
        sos_cols = diag_idx.astype(np.int64)
        sos_data = np.full(d, -1.0, dtype=np.float64)

        # Tan rows (PLACEHOLDER: data depends on m_i)
        tan_row_start = sos_row_idx + 1
        tan_rows = np.empty(2 * d, dtype=np.int64)
        tan_cols = np.empty(2 * d, dtype=np.int64)
        tan_data = np.zeros(2 * d, dtype=np.float64)
        tan_rows[:d] = tan_row_start + np.arange(d)
        tan_cols[:d] = diag_idx
        tan_data[:d] = -1.0  # constant
        tan_rows[d:] = tan_row_start + np.arange(d)
        tan_cols[d:] = n_y + np.arange(d)
        # tan_data[d:] = 2.0 * m  -- per-box

        # Stack ineq parts (relative row indices) — combine SW/NE/NW/SE/epi/sos/tan
        ub_rows = np.concatenate([ub_rows_sw, ub_rows_ne, ub_rows_nw, ub_rows_se,
                                  ep_rows, sos_rows, tan_rows])
        ub_cols = np.concatenate([ub_cols_sw, ub_cols_ne, ub_cols_nw, ub_cols_se,
                                  ep_cols, sos_cols, tan_cols])
        ub_data = np.concatenate([ub_data_sw, ub_data_ne, ub_data_nw, ub_data_se,
                                  ep_data, sos_data, tan_data])

        # Combined A: equality rows first (rows 0..n_eq-1), then inequality (rows n_eq..)
        rows_all = np.concatenate([eq_rows, n_eq + ub_rows])
        cols_all = np.concatenate([eq_cols, ub_cols])
        data_all = np.concatenate([eq_data, ub_data])

        n_total_rows = n_eq + self.n_ineq
        A = coo_matrix((data_all, (rows_all, cols_all)),
                       shape=(n_total_rows, self.n_vars)).tocsr()
        self.A = A
        # Save the offsets so we can locate (row, col) entries in CSR for mutation
        self.eq_row_offset = 0
        self.ub_row_offset = n_eq

        # Save SW/NE/NW/SE flat-index arrays for fast per-box mutation:
        # for block X (X in [SW, NE, NW, SE]), row r=p in [0..n_pairs), the 3 cells
        # are at A_ub rows X_offset+p, columns yi=p, n_y+pair_i[p], n_y+pair_j[p].
        # We need a way to write into A.data quickly OR call changeCoeff per cell.
        self.tan_row_start = tan_row_start

    def _pass_initial_model(self, h, options):
        """Pass the initial LP (with placeholder coefficients) to highs."""
        from highspy import HighsLp, ObjSense, kHighsInf, MatrixFormat
        d = self.d
        n_y = self.n_y
        n_W = self.n_W
        n_eq = self.n_eq
        n_ineq = self.n_ineq
        n_total = n_eq + n_ineq

        c = np.zeros(self.n_vars, dtype=np.float64)
        c[self.z_idx] = 1.0

        # Column bounds
        col_lower = np.zeros(self.n_vars, dtype=np.float64)
        col_upper = np.full(self.n_vars, kHighsInf, dtype=np.float64)
        # μ_i columns: placeholder [0, kHighsInf]; will be set per box

        # Row bounds:
        # eq rows: [b_eq[r], b_eq[r]]
        # ineq rows: [-inf, b_ub[r]]; b_ub for SW/NE/NW/SE/tan = placeholder 0
        row_lower = np.full(n_total, -kHighsInf, dtype=np.float64)
        row_upper = np.full(n_total, kHighsInf, dtype=np.float64)
        # equality rows
        row_lower[:n_eq] = self.b_eq
        row_upper[:n_eq] = self.b_eq
        # ineq: epi row rhs = 0, sos = -1/d, others placeholder
        sos_row = self.ub_row_offset + 4 * self.n_pairs + n_W
        # epigraph rows (already at rhs=0):
        row_upper[self.ub_row_offset + 4 * self.n_pairs : self.ub_row_offset + 4 * self.n_pairs + n_W] = 0.0
        row_upper[sos_row] = -1.0 / d

        lp = HighsLp()
        lp.num_col_ = self.n_vars
        lp.num_row_ = n_total
        lp.sense_ = ObjSense.kMinimize
        lp.col_cost_ = c
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        # Pass A in CSC format (HighsLp wants column-wise by default)
        A_csc = self.A.tocsc()
        lp.a_matrix_.format_ = MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = self.n_vars
        lp.a_matrix_.num_row_ = n_total
        lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = A_csc.data.astype(np.float64)
        h.passModel(lp)

    def _cache_mutation_indices_helper_unused(self):
        pass

    def _cache_mutation_indices(self):
        """Save numpy-friendly arrays of (row, col) for cells that vary per box.

        For each block, at iteration p ∈ [0..n_pairs):
          row = block_row_offset + p; cols = (yi=p, n_y+pair_i[p], n_y+pair_j[p]).
          The first cell (yi=p) is constant (-1 or +1); only cells 2 and 3 vary.
        """
        d = self.d
        n_y = self.n_y
        n_pairs = self.n_pairs
        # SW/NE/NW/SE non-constant cells: 2 per row, n_pairs rows = 2*n_pairs each
        # Indexing: for SW block, row offset = ub_row_offset + 0
        ub = self.ub_row_offset
        self._sw_row = ub + np.arange(n_pairs)
        self._ne_row = ub + n_pairs + np.arange(n_pairs)
        self._nw_row = ub + 2 * n_pairs + np.arange(n_pairs)
        self._se_row = ub + 3 * n_pairs + np.arange(n_pairs)
        self._col_i = (n_y + self.pair_i).astype(np.int64)
        self._col_j = (n_y + self.pair_j).astype(np.int64)
        # Tan rows: row idx = tan_row_start + i, col = n_y + i, data = 2*m_i
        self._tan_rows = (self.ub_row_offset + self.tan_row_start
                          - 0 + np.arange(d))  # already absolute (tan_row_start uses ub-relative)
        # Actually tan_row_start was computed as (4*n_pairs + n_W + 1) which is
        # already an UB-relative index. So absolute = ub + tan_row_start.
        # _build_static_matrices used n_eq + ub_rows for combined matrix;
        # tan_rows had ub-relative idx; so combined absolute = n_eq + tan_rel.
        # Let's recompute correctly:
        self._tan_rows = (self.ub_row_offset + self.tan_row_start
                          + np.arange(d))
        self._tan_cols = (n_y + np.arange(d)).astype(np.int64)
        self._tan_yii_rows = self._tan_rows.copy()
        self._tan_yii_cols = (np.arange(d) * d + np.arange(d)).astype(np.int64)

        # SOS / tan / SW / NE / NW / SE row indices for batch row-bound updates:
        # rows that need rhs updated per box:
        #   SW: rhs = lo[i]*lo[j], n_pairs rows
        #   NE: rhs = hi[i]*hi[j], n_pairs rows
        #   NW: rhs = -lo[j]*hi[i], n_pairs rows
        #   SE: rhs = -hi[j]*lo[i], n_pairs rows
        #   tan: rhs = m_i², d rows
        self._all_var_rows = np.concatenate([
            self._sw_row, self._ne_row, self._nw_row, self._se_row, self._tan_rows
        ]).astype(np.int32)

        # Pre-compute diagonal mask for SW/NE/NW/SE: when pair_i==pair_j the
        # COO→CSR collapses two contributions into one cell, so we must write
        # the SUM (not overwrite each separately).
        self._diag_mask = (self.pair_i == self.pair_j)
        # Off-diagonal pair indices (n_off = n_pairs - d at d=22 → 462)
        self._off_idx = np.where(~self._diag_mask)[0]
        self._diag_idx_p = np.where(self._diag_mask)[0]
        # Their (μ_i, μ_j) cols are n_y+pair_i[p] and n_y+pair_j[p]
        self._off_col_i = (self.n_y + self.pair_i[self._off_idx]).astype(np.int64)
        self._off_col_j = (self.n_y + self.pair_j[self._off_idx]).astype(np.int64)
        # For diagonal: only one column (n_y + i)
        self._diag_col = (self.n_y + self.pair_i[self._diag_idx_p]).astype(np.int64)
        # Pre-compute SW/NE/NW/SE row indices for off-diag and diag
        self._sw_row_off = self._sw_row[self._off_idx].astype(np.int64)
        self._sw_row_diag = self._sw_row[self._diag_idx_p].astype(np.int64)
        self._ne_row_off = self._ne_row[self._off_idx].astype(np.int64)
        self._ne_row_diag = self._ne_row[self._diag_idx_p].astype(np.int64)
        self._nw_row_off = self._nw_row[self._off_idx].astype(np.int64)
        self._nw_row_diag = self._nw_row[self._diag_idx_p].astype(np.int64)
        self._se_row_off = self._se_row[self._off_idx].astype(np.int64)
        self._se_row_diag = self._se_row[self._diag_idx_p].astype(np.int64)

    def solve_box_passmodel(self, lo: np.ndarray, hi: np.ndarray):
        """Variant D: rebuild full LP via passModel each call (no warm-start)."""
        from highspy import HighsLp, ObjSense, kHighsInf, MatrixFormat, HighsModelStatus
        d = self.d
        n_y = self.n_y
        n_W = self.n_W
        n_pairs = self.n_pairs
        pair_i = self.pair_i
        pair_j = self.pair_j

        lo_i = lo[pair_i]; lo_j = lo[pair_j]
        hi_i = hi[pair_i]; hi_j = hi[pair_j]
        m = 0.5 * (lo + hi)

        # Mutate the placeholder data array in-place.
        # The CSC `value_` array has a fixed layout; we cached the master
        # CSC indices. We just need to overwrite the SW/NE/NW/SE/tan cells.
        # Easiest: rebuild data array from scratch using the SAME structure
        # by leveraging coo_matrix → tocsc with deterministic ordering.

        # SW data: row p, cols (p, n_y+pair_i, n_y+pair_j) values (-1, lo_j, lo_i)
        # but for diag p, two cells coalesce to one with value 2*lo_i (shape changes!)
        # → easier: compute COMBINED data per (row, col) via np.add.at.
        # But we can simplify: rebuild a fresh LP each call using the
        # original (vectorized) construction in bound_epigraph.py logic.

        from scipy.sparse import coo_matrix
        n_ineq = self.n_ineq
        # Build A_ub for this box (vectorized)
        # SW
        sw_rows_a = np.repeat(np.arange(n_pairs), 3)
        sw_cols_a = np.empty(3 * n_pairs, dtype=np.int64)
        sw_data_a = np.empty(3 * n_pairs, dtype=np.float64)
        sw_cols_a[0::3] = np.arange(n_pairs); sw_data_a[0::3] = -1.0
        sw_cols_a[1::3] = n_y + pair_i; sw_data_a[1::3] = lo_j
        sw_cols_a[2::3] = n_y + pair_j; sw_data_a[2::3] = lo_i
        # NE
        ne_rows_a = n_pairs + sw_rows_a
        ne_cols_a = sw_cols_a.copy()
        ne_data_a = np.empty(3 * n_pairs, dtype=np.float64)
        ne_data_a[0::3] = -1.0
        ne_data_a[1::3] = hi_j
        ne_data_a[2::3] = hi_i
        # NW (UB)
        nw_rows_a = 2 * n_pairs + sw_rows_a
        nw_cols_a = sw_cols_a.copy()
        nw_data_a = np.empty(3 * n_pairs, dtype=np.float64)
        nw_data_a[0::3] = +1.0
        nw_data_a[1::3] = -lo_j
        nw_data_a[2::3] = -hi_i
        # SE (UB)
        se_rows_a = 3 * n_pairs + sw_rows_a
        se_cols_a = sw_cols_a.copy()
        se_data_a = np.empty(3 * n_pairs, dtype=np.float64)
        se_data_a[0::3] = +1.0
        se_data_a[1::3] = -hi_j
        se_data_a[2::3] = -lo_i
        # epigraph (rebuilt from cached _cache_lp_structure)
        epi_rows_a = 4 * n_pairs + self.rows_w
        epi_cols_a = self.cols_w
        epi_data_a = self.scales_w
        # z col addition for epi
        epi_z_rows = 4 * n_pairs + np.arange(n_W)
        epi_z_cols = np.full(n_W, self.z_idx, dtype=np.int64)
        epi_z_data = np.full(n_W, -1.0, dtype=np.float64)
        # SOS row
        sos_row = 4 * n_pairs + n_W
        diag_idx_y = np.arange(d) * d + np.arange(d)
        sos_rows_a = np.full(d, sos_row, dtype=np.int64)
        sos_cols_a = diag_idx_y
        sos_data_a = np.full(d, -1.0, dtype=np.float64)
        # tan rows
        tan_row_start = sos_row + 1
        tan_rows_a = np.empty(2 * d, dtype=np.int64)
        tan_cols_a = np.empty(2 * d, dtype=np.int64)
        tan_data_a = np.empty(2 * d, dtype=np.float64)
        tan_rows_a[:d] = tan_row_start + np.arange(d)
        tan_cols_a[:d] = diag_idx_y
        tan_data_a[:d] = -1.0
        tan_rows_a[d:] = tan_row_start + np.arange(d)
        tan_cols_a[d:] = n_y + np.arange(d)
        tan_data_a[d:] = 2.0 * m

        ub_rows_all = np.concatenate([sw_rows_a, ne_rows_a, nw_rows_a, se_rows_a,
                                       epi_rows_a, epi_z_rows, sos_rows_a, tan_rows_a])
        ub_cols_all = np.concatenate([sw_cols_a, ne_cols_a, nw_cols_a, se_cols_a,
                                       epi_cols_a, epi_z_cols, sos_cols_a, tan_cols_a])
        ub_data_all = np.concatenate([sw_data_a, ne_data_a, nw_data_a, se_data_a,
                                       epi_data_a, epi_z_data, sos_data_a, tan_data_a])

        # Combined: stack equality rows on top
        n_eq = self.n_eq
        eq_csr = self.A[:n_eq]  # static eq part (stored in self.A)
        eq_coo = eq_csr.tocoo()
        rows_all = np.concatenate([eq_coo.row, n_eq + ub_rows_all])
        cols_all = np.concatenate([eq_coo.col, ub_cols_all])
        data_all = np.concatenate([eq_coo.data, ub_data_all])

        n_total = n_eq + n_ineq
        A = coo_matrix((data_all, (rows_all, cols_all)),
                       shape=(n_total, self.n_vars)).tocsc()

        # b_ub
        b_ub = np.empty(n_ineq, dtype=np.float64)
        b_ub[:n_pairs] = lo_i * lo_j
        b_ub[n_pairs:2*n_pairs] = hi_i * hi_j
        b_ub[2*n_pairs:3*n_pairs] = -lo_j * hi_i
        b_ub[3*n_pairs:4*n_pairs] = -hi_j * lo_i
        b_ub[4*n_pairs:4*n_pairs+n_W] = 0.0
        b_ub[sos_row] = -1.0 / d
        b_ub[tan_row_start:tan_row_start+d] = m * m

        # Build row bounds: eq rows = b_eq; ineq rows = (-inf, b_ub)
        row_lower = np.full(n_total, -kHighsInf, dtype=np.float64)
        row_upper = np.full(n_total, kHighsInf, dtype=np.float64)
        row_lower[:n_eq] = self.b_eq
        row_upper[:n_eq] = self.b_eq
        row_upper[n_eq:] = b_ub

        col_lower = np.zeros(self.n_vars, dtype=np.float64)
        col_upper = np.full(self.n_vars, kHighsInf, dtype=np.float64)
        col_lower[n_y:n_y+d] = lo
        col_upper[n_y:n_y+d] = hi

        c = np.zeros(self.n_vars, dtype=np.float64)
        c[self.z_idx] = 1.0

        lp = HighsLp()
        lp.num_col_ = self.n_vars
        lp.num_row_ = n_total
        lp.sense_ = ObjSense.kMinimize
        lp.col_cost_ = c
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.format_ = MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = self.n_vars
        lp.a_matrix_.num_row_ = n_total
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data.astype(np.float64)
        self.h.passModel(lp)
        self.h.run()
        info = self.h.getInfo()
        status = self.h.getModelStatus()
        if status != HighsModelStatus.kOptimal:
            return float('-inf'), None
        return float(info.objective_function_value), None

    def solve_box(self, lo: np.ndarray, hi: np.ndarray):
        """Solve for one box. Returns (lp_val, ineqlin) or (-inf, None)."""
        d = self.d
        n_pairs = self.n_pairs
        h = self.h
        pair_i = self.pair_i
        pair_j = self.pair_j

        lo_i = lo[pair_i]
        lo_j = lo[pair_j]
        hi_i = hi[pair_i]
        hi_j = hi[pair_j]
        m = 0.5 * (lo + hi)

        # ----- mutate coefficients (changeCoeff is per-cell — Python loop) -----
        # Off-diagonal (i!=j): two cells per row, separate columns.
        n_off = self._off_idx.size
        off_lo_j = lo[self.pair_j[self._off_idx]]
        off_lo_i = lo[self.pair_i[self._off_idx]]
        off_hi_j = hi[self.pair_j[self._off_idx]]
        off_hi_i = hi[self.pair_i[self._off_idx]]
        for k in range(n_off):
            ri = int(self._off_col_i[k])
            rj = int(self._off_col_j[k])
            h.changeCoeff(int(self._sw_row_off[k]), ri, float(off_lo_j[k]))
            h.changeCoeff(int(self._sw_row_off[k]), rj, float(off_lo_i[k]))
            h.changeCoeff(int(self._ne_row_off[k]), ri, float(off_hi_j[k]))
            h.changeCoeff(int(self._ne_row_off[k]), rj, float(off_hi_i[k]))
            h.changeCoeff(int(self._nw_row_off[k]), ri, float(-off_lo_j[k]))
            h.changeCoeff(int(self._nw_row_off[k]), rj, float(-off_hi_i[k]))
            h.changeCoeff(int(self._se_row_off[k]), ri, float(-off_hi_j[k]))
            h.changeCoeff(int(self._se_row_off[k]), rj, float(-off_lo_i[k]))
        # Diagonal (i==j): single cell per row, with SUMMED contribution.
        # SW (i==i): two contributions of lo[i] each → 2*lo[i]
        # NE: 2*hi[i]
        # NW: -lo[i] + -hi[i] = -(lo+hi)[i]
        # SE: -hi[i] + -lo[i] = -(lo+hi)[i]
        n_diag = self._diag_idx_p.size  # = d
        for k in range(n_diag):
            i = int(self.pair_i[self._diag_idx_p[k]])
            col = int(self._diag_col[k])
            h.changeCoeff(int(self._sw_row_diag[k]), col, float(2.0 * lo[i]))
            h.changeCoeff(int(self._ne_row_diag[k]), col, float(2.0 * hi[i]))
            h.changeCoeff(int(self._nw_row_diag[k]), col, float(-(lo[i] + hi[i])))
            h.changeCoeff(int(self._se_row_diag[k]), col, float(-(lo[i] + hi[i])))
        # Tan diag: A[tan_rows[i], tan_cols[i]] = 2*m_i
        for i in range(d):
            h.changeCoeff(int(self._tan_rows[i]), int(self._tan_cols[i]), float(2.0 * m[i]))

        # ----- update row bounds (b_ub) — batched ----
        # rhs (upper bound, since rows are -inf <= a x <= rhs):
        sw_rhs = lo_i * lo_j
        ne_rhs = hi_i * hi_j
        nw_rhs = -lo_j * hi_i
        se_rhs = -hi_j * lo_i
        tan_rhs = m * m
        rhs_all = np.concatenate([sw_rhs, ne_rhs, nw_rhs, se_rhs, tan_rhs])
        from highspy import kHighsInf
        lhs_all = np.full(rhs_all.shape, -kHighsInf, dtype=np.float64)
        h.changeRowsBounds(int(self._all_var_rows.size),
                           self._all_var_rows.astype(np.int32),
                           lhs_all, rhs_all.astype(np.float64))

        # ----- update column bounds for μ_i ----
        mu_cols = (self.n_y + np.arange(d)).astype(np.int32)
        h.changeColsBounds(d, mu_cols, lo.astype(np.float64), hi.astype(np.float64))

        # ----- solve -----
        h.run()
        info = h.getInfo()
        sol = h.getSolution()
        status = h.getModelStatus()
        from highspy import HighsModelStatus
        if status != HighsModelStatus.kOptimal:
            self._last_status = status
            return float('-inf'), None
        return float(info.objective_function_value), np.asarray(sol.row_dual)


# =====================================================================
# Variant B: highspy with passModel each call (no warm start)
# =====================================================================

def variant_B_solve(d: int, windows, options: dict, lo: np.ndarray, hi: np.ndarray):
    """Build LP fresh and pass via highspy each call."""
    import highspy
    from highspy import Highs, HighsLp, HighsModelStatus, kHighsInf, ObjSense
    from scipy.sparse import coo_matrix
    # Reuse the structural builder from FastEpigraphLP for one-off
    # (this DOES include the build cost per call — we want that)
    obj = FastEpigraphLP(d, windows, options)  # builds + passModel once
    return obj.solve_box(lo, hi)


# =====================================================================
# MICROBENCH
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='runs/d22_pod_iter5_higherK/iter_001/children_after_lp.npz')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--d', type=int, default=22)
    args = ap.parse_args()

    z = np.load(args.input, allow_pickle=True)
    n_total = int(z['hash'].size)
    rng = np.random.default_rng(2026)
    sample = rng.choice(n_total, args.n, replace=False)
    print(f'sampled {args.n} boxes (of {n_total}) from {args.input}', flush=True)

    boxes_lo = []
    boxes_hi = []
    for k in sample:
        lo_int = z['lo_int'][k]
        hi_int = z['hi_int'][k]
        boxes_lo.append(np.asarray([float(x) / _SCALE for x in lo_int], dtype=np.float64))
        boxes_hi.append(np.asarray([float(x) / _SCALE for x in hi_int], dtype=np.float64))

    d = args.d
    windows = build_windows(d)

    # Warm cache
    _ = _cache_lp_structure(windows, d)

    # ----- Variant A: scipy.linprog (current) -----
    print('\n=== Variant A: scipy.linprog (current) ===', flush=True)
    lp_vals_A = []
    t0 = time.time()
    for lo, hi in zip(boxes_lo, boxes_hi):
        lp = bound_epigraph_lp_float(lo, hi, windows, d)
        lp_vals_A.append(float(lp))
    t_A = time.time() - t0
    print(f'  total {t_A:.2f}s  per-box {t_A/args.n*1000:.1f}ms', flush=True)
    sample_str = ', '.join(f'{v:.6f}' for v in lp_vals_A[:5])
    print(f'  LP vals (first 5): {sample_str}', flush=True)

    # ----- Variant C: highspy WARM (build once, mutate per call) -----
    print('\n=== Variant C: highspy persistent + warm-start ===', flush=True)
    options = {
        'primal_feasibility_tolerance': 1e-9,
        'dual_feasibility_tolerance': 1e-9,
        'ipm_optimality_tolerance': 1e-10,
        'output_flag': False,
    }
    t_build = time.time()
    fast = FastEpigraphLP(d, windows, options)
    t_build = time.time() - t_build
    print(f'  build cost (one-time): {t_build:.2f}s', flush=True)
    lp_vals_C = []
    t0 = time.time()
    for k, (lo, hi) in enumerate(zip(boxes_lo, boxes_hi)):
        lp_v, _ = fast.solve_box(lo, hi)
        lp_vals_C.append(lp_v)
        if k == 0 and lp_v == float('-inf'):
            print(f'  DBG: first box returned -inf, status={fast._last_status}', flush=True)
    t_C = time.time() - t0
    print(f'  total {t_C:.2f}s  per-box {t_C/args.n*1000:.1f}ms', flush=True)
    sample_str_c = ', '.join(f'{v:.6f}' for v in lp_vals_C[:5])
    print(f'  LP vals (first 5): {sample_str_c}', flush=True)

    # ----- compare values -----
    print('\n=== Value agreement check ===', flush=True)
    diffs = [abs(a - c) for a, c in zip(lp_vals_A, lp_vals_C)]
    max_diff = max(diffs)
    print(f'  max |A - C|: {max_diff:.2e}  '
          f'(mean {sum(diffs)/len(diffs):.2e})', flush=True)
    if max_diff > 1e-6:
        print('  **MISMATCH** — variant C does NOT agree with current LP!', flush=True)
        for k in range(args.n):
            if diffs[k] > 1e-6:
                print(f'    box {k}: A={lp_vals_A[k]:.9f}  C={lp_vals_C[k]:.9f}', flush=True)
                if k > 5: break

    # ----- Variant D: highspy passModel each call (no warm-start, vectorized build) -----
    print('\n=== Variant D: highspy passModel each call (no warm-start) ===', flush=True)
    fast_d = FastEpigraphLP(d, windows, options)  # fresh instance to avoid basis bleed
    lp_vals_D = []
    t0 = time.time()
    for lo, hi in zip(boxes_lo, boxes_hi):
        lp_v, _ = fast_d.solve_box_passmodel(lo, hi)
        lp_vals_D.append(lp_v)
    t_D = time.time() - t0
    print(f'  total {t_D:.2f}s  per-box {t_D/args.n*1000:.1f}ms', flush=True)
    sample_str_d = ', '.join(f'{v:.6f}' for v in lp_vals_D[:5])
    print(f'  LP vals (first 5): {sample_str_d}', flush=True)
    diffs_d = [abs(a - dval) for a, dval in zip(lp_vals_A, lp_vals_D)]
    print(f'  max |A - D|: {max(diffs_d):.2e}', flush=True)

    print('\n=== Speedup summary ===', flush=True)
    print(f'  Variant A (scipy.linprog):                {t_A/args.n*1000:6.1f} ms/box', flush=True)
    print(f'  Variant C (highspy warm-start):           {t_C/args.n*1000:6.1f} ms/box  ({t_A/t_C:.2f}x)', flush=True)
    print(f'  Variant D (highspy passModel each):       {t_D/args.n*1000:6.1f} ms/box  ({t_A/t_D:.2f}x)', flush=True)


if __name__ == '__main__':
    main()

"""Pre-elimination of consistency + simplex equalities.

================================================================================
MATHEMATICAL JUSTIFICATION
================================================================================

The Lasserre SDP (as built by ``lasserre_scalable._precompute``) contains the
following linear equalities on the moment vector y:

    y_0 = 1                                        (simplex normalization)
    y_α - Σ_i y_{α + e_i} = 0   for every α       (consistency, from Σ_i μ_i = 1)

These are EXACT linear relations that hold identically on the feasible set.
They can be substituted out at modeling time — every pivoted variable y_j is
expressed as an affine combination of the remaining (free) moment variables,

    y_j  =  c_j  +  Σ_{l ∈ free} t_{j,l} · ỹ_l                    (⋆)

This defines a linear transform

    y  =  T · ỹ  +  c

where T ∈ R^{n_y × ñ_y} is sparse, c ∈ R^{n_y} is the constant part (pivoted
entries carry the simplex/consistency constants), and ỹ ∈ R^{ñ_y} is the
reduced-moment vector over the free columns.  The same SDP is solved in the
reduced basis:

    - every coefficient matrix  C y  on the original y becomes  (CT)·ỹ + C·c;
    - every pick array  y[pick]  becomes  T[pick, :]·ỹ + c[pick];
    - nonnegativity y_α ≥ 0 becomes T[α, :]·ỹ + c_α ≥ 0.

**This is NOT a relaxation.**  The feasible set in ỹ-space, mapped back
via (⋆) to y-space, coincides with the original feasible set cut by the
equalities listed above.  Every SDP cone value, every scalar window bound,
and the bisection verdict are preserved bit-exactly.

Gauss–Jordan with column pivoting is used because it produces the FINAL form
directly — each pivot row has a 1 on the pivot column and zeros on all other
pivoted columns.  Reading each pivoted row off that final matrix gives the
substitution in terms of free variables only (no chaining needed).

--------------------------------------------------------------------------------
Fill control
--------------------------------------------------------------------------------

At each pivot step the algorithm estimates fill as

    (#_other_rows_containing_col) × (row_nnz − 1)

and rejects the pivot when this exceeds ``max_fill_ratio × row_nnz``.
A rejected row stays as a residual linear equality (still sound; just not
substituted out).  The fallback guarantees the algorithm never explodes
memory — worst case, some rows stay as residual equalities which MOSEK will
simply see in the reduced SDP alongside the other constraints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp


__all__ = [
    'PreElimTransform',
    'assemble_consistency_equalities',
    'gauss_jordan_eliminate',
    'build_preelim_transform',
]


PIVOT_TOL = 1e-10
DEFAULT_MAX_FILL_RATIO = 10.0


# ---------------------------------------------------------------------------
# Transform object
# ---------------------------------------------------------------------------

class PreElimTransform:
    """Encapsulates the substitution y = T · ỹ + c.

    Attributes
    ----------
    T          : scipy.sparse.csr_matrix of shape (n_y, n_y_red).
    offset     : np.ndarray of shape (n_y,).
    pivot_cols : np.int64 array of pivoted original-y column indices.
    free_cols  : np.int64 array of kept (free) original-y column indices.
                  These are the coordinates of ỹ (in the same order).
    new_idx    : dict mapping original-y free column -> reduced ỹ index.
    pivot_to_row : dict mapping pivoted col -> the row in the original
                    equality system that was used to eliminate it.
    residual_A : leftover equalities that could not be pivoted (sparse,
                  shape (n_resid, n_y)).  The reduced SDP must still enforce
                  ``residual_A · y = residual_b`` — equivalently
                  ``(residual_A · T) · ỹ = residual_b − residual_A · c``.
    residual_b : residual RHS.
    n_y        : original moment-vector length.
    n_y_red    : reduced (ỹ) length = len(free_cols).
    """

    def __init__(self,
                 T: sp.csr_matrix,
                 offset: np.ndarray,
                 pivot_cols,
                 free_cols,
                 pivot_to_row: Dict[int, int],
                 residual_A: sp.csr_matrix,
                 residual_b: np.ndarray):
        self.T = T.tocsr()
        self.offset = np.asarray(offset, dtype=np.float64).ravel()
        self.pivot_cols = np.asarray(pivot_cols, dtype=np.int64)
        self.free_cols = np.asarray(free_cols, dtype=np.int64)
        self.new_idx: Dict[int, int] = {
            int(j): k for k, j in enumerate(self.free_cols)
        }
        self.pivot_to_row: Dict[int, int] = dict(pivot_to_row)
        self.residual_A = residual_A.tocsr() if residual_A is not None \
            else sp.csr_matrix((0, T.shape[0]))
        self.residual_b = np.asarray(residual_b, dtype=np.float64).ravel()
        self.n_y = int(T.shape[0])
        self.n_y_red = int(T.shape[1])

    # ----- Primary operations -----

    def reconstruct(self, y_red: np.ndarray) -> np.ndarray:
        """y = T · ỹ + c, the inverse of ``project``."""
        y_red = np.asarray(y_red, dtype=np.float64).ravel()
        return np.asarray(self.T @ y_red).ravel() + self.offset

    def project(self, y_full: np.ndarray) -> np.ndarray:
        """Return ỹ = y[free_cols] (forward projection from a full y)."""
        y_full = np.asarray(y_full, dtype=np.float64).ravel()
        return y_full[self.free_cols].copy()

    def verify_full_y(self, y_full: np.ndarray, tol: float = 1e-8) -> bool:
        """Check that y_full satisfies the pivot equalities: i.e. that
        projecting-then-reconstructing reproduces it."""
        y_red = self.project(y_full)
        y_back = self.reconstruct(y_red)
        return bool(np.max(np.abs(y_full - y_back)) < tol)

    # ----- Coefficient substitutions -----

    def substitute_pick(self, pick) -> Tuple[sp.csr_matrix, np.ndarray]:
        """For an original-y pick array p of length m, return (C, c) with
        ``C @ ỹ + c == y[p]``.  Shapes: C is (m, n_y_red), c is (m,).

        Missing entries (pick value of -1, sentinel) become zero rows in C
        and zero entries in c.
        """
        pick_arr = np.asarray(pick, dtype=np.int64)
        mask_valid = pick_arr >= 0
        safe_pick = np.where(mask_valid, pick_arr, 0)
        C_full = self.T[safe_pick, :]
        # Zero out rows where the original pick was -1 (missing moment).
        # Convert to CSR, build row mask.
        if not np.all(mask_valid):
            # Build a diagonal scaling with 0/1.
            mask_f = mask_valid.astype(np.float64)
            D = sp.diags(mask_f, 0, format='csr')
            C_full = D @ C_full
        c_full = np.where(mask_valid, self.offset[safe_pick], 0.0)
        return C_full.tocsr(), c_full

    def substitute_matrix(self, B: sp.csr_matrix
                           ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """For a sparse matrix B of shape (m, n_y), return (B_new, c_new) with
        ``B_new @ ỹ + c_new == B @ y``.  Uses sparse matrix composition."""
        if B.shape[1] != self.n_y:
            raise ValueError(
                f"substitute_matrix: B has {B.shape[1]} cols, "
                f"expected {self.n_y}")
        B_new = (B @ self.T).tocsr()
        c_new = np.asarray(B @ self.offset, dtype=np.float64).ravel()
        return B_new, c_new


# ---------------------------------------------------------------------------
# Assemble the (consistency + simplex) equality system
# ---------------------------------------------------------------------------

def assemble_consistency_equalities(
    P: Dict[str, Any], *, include_simplex: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, List[str]]:
    """Build the sparse equality matrix (A, b) for A @ y = b.

    Rows emitted, in order:
      1. (if include_simplex) y_0 = 1   [row label 'simplex']
      2. y_α - Σ_i y_{α + e_i} = 0  for each α in P['consist_mono'] with at
         least one child α + e_i mapped to an index ≥ 0 in the moment set.

    Duplicate column contributions (from children that collapse to the same
    moment index — possible under Z/2 pre-elim, not by default here) are
    summed.

    Parameters
    ----------
    P : precompute dict (see lasserre_scalable._precompute).
    include_simplex : include the y_0 = 1 row.  Default True.

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (n_eq, n_y).
    b : np.ndarray of shape (n_eq,).
    tags : list of length n_eq, one string per row, for debugging.
    """
    d = P['d']
    n_y = P['n_y']
    idx = P['idx']
    consist_idx = P['consist_idx']
    consist_ei_idx = P['consist_ei_idx']
    zero = tuple(0 for _ in range(d))

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    b_list: List[float] = []
    tags: List[str] = []
    r = 0

    if include_simplex:
        rows.append(r)
        cols.append(int(idx[zero]))
        vals.append(1.0)
        b_list.append(1.0)
        tags.append('simplex')
        r += 1

    for k in range(len(P['consist_mono'])):
        ai = int(consist_idx[k])
        if ai < 0:
            continue
        child_idx = consist_ei_idx[k]
        coef_by_col: Dict[int, float] = {}
        has_child = False
        for ci in range(d):
            c = int(child_idx[ci])
            if c >= 0:
                coef_by_col[c] = coef_by_col.get(c, 0.0) + 1.0
                has_child = True
        if not has_child:
            continue
        coef_by_col[ai] = coef_by_col.get(ai, 0.0) - 1.0
        for c, coef in coef_by_col.items():
            if abs(coef) > 0.0:
                rows.append(r)
                cols.append(int(c))
                vals.append(float(coef))
        b_list.append(0.0)
        tags.append(f'consist[{k}]')
        r += 1

    A = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(r, n_y),
        dtype=np.float64,
    )
    A.sum_duplicates()
    A.eliminate_zeros()
    b = np.asarray(b_list, dtype=np.float64)
    return A, b, tags


# ---------------------------------------------------------------------------
# Gauss–Jordan elimination
# ---------------------------------------------------------------------------

def gauss_jordan_eliminate(
    A: sp.csr_matrix,
    b: np.ndarray,
    *,
    tol: float = PIVOT_TOL,
    max_fill_ratio: float = DEFAULT_MAX_FILL_RATIO,
    forbidden_cols: Optional[Set[int]] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[int, float]], np.ndarray,
           List[int], List[int],
           sp.csr_matrix, np.ndarray]:
    """Column-pivot Gauss–Jordan on (A, b).

    Pivot column selection heuristic, in order of preference:
      1. Column is not already pivoted and not in ``forbidden_cols``.
      2. |A[k, j]| ≥ tol.
      3. Smallest fill estimate
             (col_count - 1) × (row_nnz - 1)
         — i.e., columns that appear in fewer other rows are cheaper to
         eliminate from.
      4. Ties broken by smaller column count, then larger |value| (for
         numerical stability).

    Fill cap: if the fill estimate of the best available pivot exceeds
    ``max_fill_ratio × row_nnz``, the row is skipped (left as a residual
    equality) rather than committing an expensive pivot.

    Parameters
    ----------
    A        : sparse equality matrix, shape (n_eq, n_y).
    b        : RHS vector, shape (n_eq,).
    tol      : pivot-magnitude floor.
    max_fill_ratio : per-row fill multiplier.
    forbidden_cols : columns never allowed to be pivoted (e.g. objective
                      variables).

    Returns
    -------
    rows       : list of length n_eq, each entry is dict {col: val} holding
                   the row in final Gauss-Jordan form.
    b          : updated RHS.
    pivot_rows : rows used as pivots, in pivoting order.
    pivot_cols : columns pivoted on, aligned with pivot_rows.
    resid_A    : residual (n_resid × n_y) matrix — rows that could not pivot.
    resid_b    : residual RHS vector.
    """
    n_eq, n_y = A.shape
    A_csr = A.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()

    # Row-dicts for O(row_nnz) row operations.
    row_dicts: List[Dict[int, float]] = [dict() for _ in range(n_eq)]
    for i in range(n_eq):
        start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
        for p in range(start, end):
            row_dicts[i][int(A_csr.indices[p])] = float(A_csr.data[p])

    b_vec = np.asarray(b, dtype=np.float64).copy()

    # Inverted index: col -> set of rows currently containing it.
    col_rows: Dict[int, Set[int]] = {}
    for i in range(n_eq):
        for c in row_dicts[i]:
            col_rows.setdefault(c, set()).add(i)

    pivoted_cols: Set[int] = set()
    pivot_rows_list: List[int] = []
    pivot_cols_list: List[int] = []

    if forbidden_cols is None:
        forbidden_cols = set()

    # Visit rows sparsest-first to minimise fill.
    row_order = sorted(range(n_eq), key=lambda i: len(row_dicts[i]))

    for k in row_order:
        row_k = row_dicts[k]
        if not row_k:
            continue

        row_nnz = len(row_k)
        best_j: Optional[int] = None
        best_score = None

        for c, v in row_k.items():
            if abs(v) < tol:
                continue
            if c in pivoted_cols:
                continue
            if c in forbidden_cols:
                continue
            col_count = len(col_rows.get(c, ()))
            fill_est = max(0, col_count - 1) * max(0, row_nnz - 1)
            # tuple: (fill_est, col_count, -|v|) — lex-min.
            score = (fill_est, col_count, -abs(v))
            if best_score is None or score < best_score:
                best_score = score
                best_j = c

        if best_j is None:
            continue

        if best_score[0] > max_fill_ratio * row_nnz:
            # Fill cap: leave this row as residual.
            continue

        # Normalize pivot row so pivot coefficient = 1.
        pivot_val = row_k[best_j]
        inv_pivot = 1.0 / pivot_val
        for c in list(row_k.keys()):
            row_k[c] *= inv_pivot
        row_k[best_j] = 1.0  # ensure exactly 1 post-normalize
        b_vec[k] *= inv_pivot

        # Eliminate best_j from every OTHER row.
        affected = list(col_rows[best_j] - {k})
        for i in affected:
            row_i = row_dicts[i]
            coef_i = row_i.get(best_j, 0.0)
            if abs(coef_i) < tol:
                continue
            for c, v_k in row_k.items():
                new_val = row_i.get(c, 0.0) - coef_i * v_k
                if abs(new_val) > tol:
                    if c not in row_i:
                        col_rows.setdefault(c, set()).add(i)
                    row_i[c] = new_val
                else:
                    if c in row_i:
                        del row_i[c]
                        col_rows[c].discard(i)
            if best_j in row_i:
                del row_i[best_j]
            col_rows[best_j].discard(i)
            b_vec[i] -= coef_i * b_vec[k]

        pivoted_cols.add(best_j)
        pivot_rows_list.append(k)
        pivot_cols_list.append(best_j)

    pivot_row_set = set(pivot_rows_list)
    resid_rows_idx: List[int] = []
    for i in range(n_eq):
        if i in pivot_row_set:
            continue
        if row_dicts[i]:
            resid_rows_idx.append(i)

    rr, rc, rv = [], [], []
    rb: List[float] = []
    for new_k, i in enumerate(resid_rows_idx):
        for c, v in row_dicts[i].items():
            rr.append(new_k)
            rc.append(c)
            rv.append(v)
        rb.append(float(b_vec[i]))
    resid_A = sp.csr_matrix(
        (rv, (rr, rc)),
        shape=(len(resid_rows_idx), n_y),
        dtype=np.float64,
    )
    resid_b = np.asarray(rb, dtype=np.float64)

    if verbose:
        print(f"  gauss_jordan: {n_eq:,} eqs × {n_y:,} cols -> "
              f"{len(pivot_cols_list):,} pivots, "
              f"{resid_A.shape[0]:,} residual rows", flush=True)

    return (row_dicts, b_vec, pivot_rows_list, pivot_cols_list,
            resid_A, resid_b)


# ---------------------------------------------------------------------------
# Top-level transform builder
# ---------------------------------------------------------------------------

def build_preelim_transform(
    P: Dict[str, Any], *,
    tol: float = PIVOT_TOL,
    max_fill_ratio: float = DEFAULT_MAX_FILL_RATIO,
    forbidden_cols: Optional[Set[int]] = None,
    protect_degrees: Optional[Set[int]] = None,
    verbose: bool = True,
) -> PreElimTransform:
    """End-to-end: assemble consistency+simplex equalities, run Gauss-Jordan,
    materialise T and offset, and wrap in a ``PreElimTransform``.

    ``protect_degrees``: set of monomial TOTAL DEGREES that must not be
    pivoted (stay as free ỹ coordinates).  For the Lasserre SDP the "backbone"
    moment variables are degrees 1 and 2 — they appear in the moment matrix
    diagonal and in window Q_W as single-entry coefficients.  Pivoting them
    replaces a single-variable pick with a ~d-term linear combination,
    degrading Schur-complement conditioning near rank-deficient primals.
    Default: {1, 2}.  Pass ``set()`` to disable.
    """
    if protect_degrees is None:
        protect_degrees = {1, 2}

    mono_list = P['mono_list']
    protected_from_deg: Set[int] = set()
    if protect_degrees:
        for j, alpha in enumerate(mono_list):
            if int(sum(alpha)) in protect_degrees:
                protected_from_deg.add(j)
    # Merge with any user-supplied forbidden_cols.
    all_forbidden: Set[int] = set(forbidden_cols or set())
    all_forbidden |= protected_from_deg

    A, b, _tags = assemble_consistency_equalities(P)
    n_y = P['n_y']
    n_eq = A.shape[0]

    rows, b_out, pivot_rows, pivot_cols, resid_A, resid_b = \
        gauss_jordan_eliminate(
            A, b,
            tol=tol,
            max_fill_ratio=max_fill_ratio,
            forbidden_cols=all_forbidden,
            verbose=verbose,
        )

    if verbose and protect_degrees:
        print(f"  protect_degrees={sorted(protect_degrees)} — "
              f"{len(protected_from_deg)} moment indices protected from pivoting",
              flush=True)

    free_cols = sorted(set(range(n_y)) - set(pivot_cols))
    new_idx = {j: k for k, j in enumerate(free_cols)}
    n_y_red = len(free_cols)

    T_rows: List[int] = []
    T_cols: List[int] = []
    T_vals: List[float] = []
    offset = np.zeros(n_y, dtype=np.float64)

    # Identity piece: each free col maps to its ỹ coordinate.
    for j in free_cols:
        T_rows.append(j)
        T_cols.append(new_idx[j])
        T_vals.append(1.0)

    # Pivoted piece: y_pivot = b_k - Σ_{c ∈ free} (coef) ỹ_{new_idx[c]}.
    # The normalised pivot row has A[k, pivot_col] = 1, so
    #   y_pivot = b_k - Σ_{c ≠ pivot_col} A[k, c] · y_c
    #           = b_k - Σ_{c ∈ free} A[k, c] · ỹ_{new_idx[c]}
    # (all pivoted-column entries have been zeroed by Gauss–Jordan).
    pivot_to_row: Dict[int, int] = {}
    for k, j in zip(pivot_rows, pivot_cols):
        pivot_to_row[int(j)] = int(k)
        row_k = rows[k]
        offset[j] = float(b_out[k])
        for c, v in row_k.items():
            if c == j:
                continue
            if c in new_idx:
                T_rows.append(j)
                T_cols.append(new_idx[c])
                T_vals.append(-v)
            else:
                raise RuntimeError(
                    f"Gauss-Jordan invariant violated: pivot row {k} still "
                    f"references pivoted column {c} (own pivot={j})"
                )

    T = sp.csr_matrix(
        (T_vals, (T_rows, T_cols)),
        shape=(n_y, n_y_red),
        dtype=np.float64,
    )
    T.sum_duplicates()
    T.eliminate_zeros()

    if verbose:
        print(f"  preelim: n_y {n_y:,} -> {n_y_red:,} "
              f"({100.0 * n_y_red / max(1, n_y):.1f}%);  "
              f"n_eq {n_eq:,} -> {resid_A.shape[0]:,} residual "
              f"(eliminated {len(pivot_cols):,} equalities)",
              flush=True)

    return PreElimTransform(
        T=T,
        offset=offset,
        pivot_cols=pivot_cols,
        free_cols=free_cols,
        pivot_to_row=pivot_to_row,
        residual_A=resid_A,
        residual_b=resid_b,
    )

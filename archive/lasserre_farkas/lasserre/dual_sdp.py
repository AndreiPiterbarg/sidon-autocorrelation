"""Dual Lasserre SDP in MOSEK Task API — Farkas infeasibility certifier.

================================================================================
MATHEMATICAL BACKGROUND
================================================================================

**Primal Lasserre relaxation at fixed t (feasibility form):**

    find  y
    s.t.  y_0 = 1                                (λ: free scalar dual)
          Σ_i y_{α+e_i} − y_α = 0 ∀α∈consist      (μ_k: free scalar dual)
          y_α ≥ 0                                (v_α ≥ 0 scalar dual)
          M_k(y) ⪰ 0                              (X_0 ⪰ 0 bar dual)
          M_{k−1}(μ_i y) ⪰ 0                      (X_i ⪰ 0 bar dual)
          M_{k−1}((1−μ_i)y) ⪰ 0                   (X'_i ⪰ 0 bar dual, optional)
          t·M_{k−1}(y) − M_{k−1}(q_W y) ⪰ 0       (X_W ⪰ 0 bar dual)

**Farkas infeasibility LP** (primal-feas iff this LP has max = 0):

    max λ
    s.t.  [α=0]·λ + Σ_k μ_k · C_{k,α} + ⟨X_0, E_0[α]⟩
          + Σ_i ⟨X_i, E_i[α]⟩ + Σ_i ⟨X'_i, E'_i[α]⟩
          + t·Σ_W ⟨X_W, E_W^t[α]⟩ − Σ_W ⟨X_W, E_W^Q[α]⟩
          + v_α = 0                              ∀α

          0 ≤ λ ≤ 1,  μ free,  v_α ≥ 0,  X_•, X_W ⪰ 0.

Sign convention derives from conic Farkas: primal {Ax=b, x∈K} infeasible
iff ∃u with A^T u ∈ −K* and b^T u > 0.  With slack v_β ≥ 0 on the y-block
inequality A^T u ≤ 0, and renaming bar-multipliers to PSD form (X := −Ω),
the inner-product coefficients on E_•[α] all become ``+``.

Sensitivity matrices (all symmetric — specified via lower triangle
k ≥ l via ``putbarablocktriplet``):

    E_0[α]_{β,γ}   = 𝟙[basis[β] + basis[γ] = α]
    E_i[α]_{β,γ}   = 𝟙[loc[β] + loc[γ] + e_i = α]
    E'_i[α]_{β,γ}  = 𝟙[loc[β] + loc[γ] = α] − 𝟙[loc[β] + loc[γ] + e_i = α]
    E_W^t[α]_{β,γ} = 𝟙[loc[β] + loc[γ] = α]
    E_W^Q[α]_{β,γ} = Σ_{i,j} M_W[i,j] · 𝟙[loc[β] + loc[γ] + e_i + e_j = α]

C_{k,α} = (#{i : consist_mono[k] + e_i = α}) − 𝟙[consist_mono[k] = α].

**Verdict mapping** (same as the moment-primal bisection — feas/infeas
semantics are IDENTICAL, so the driver plugs in without flipping):

    optimum λ ≈ 1  ⟹  Farkas certificate exists  ⟹  primal INFEASIBLE at t
                  ⟹  val_L(d) > t  ⟹  bisection advances lo.
    optimum λ ≈ 0  ⟹  no Farkas                  ⟹  primal FEASIBLE at t
                  ⟹  val_L(d) ≤ t  ⟹  bisection pulls hi.

================================================================================
WHY Task API (not Fusion)
================================================================================

At d=16 L=3 scale the dual has ~29K constraint rows and tens of millions
of bar-matrix coefficient entries.  Task API's ``putbarablocktriplet``
bulk-submits the entire coefficient tensor in a single Python → C call
with numpy int32/float64 arrays; Fusion would build the equivalent as an
Expr DAG entry-by-entry across the Python/Java boundary, which at d=16
is minutes of pure build overhead.

IPM factorization cost is identical under both APIs (same SDP structure);
the speedup is entirely in build time.

================================================================================
FEATURE SUPPORT (as of v2)
================================================================================

  * Moment cone X_0 (one BAR of size n_basis, OR block-diagonalised into
    X_0^sym (n_sym) + X_0^anti (n_anti) when ``z2_blockdiag_map`` is passed).
  * Localizing cones X_i (one BAR per active i; defaults to all i = 0..d−1).
  * Upper-localizing cones X'_i  ((1 − μ_i) ≥ 0), selectable via
    ``include_upper_loc``.
  * Window cones X_W (one BAR per active W; defaults to all nontrivial W).
  * Z/2 pre-elimination: pass a precompute dict that has been canonicalized
    via ``lasserre.z2_elim.canonicalize_z2`` — the builder auto-detects via
    the ``old_to_new`` / ``z2_pre_elim`` flag and remaps hash-table lookups
    through ``old_to_new`` so every sensitivity lookup lands on canonical
    ỹ indices.  Duplicate (subi, subj, subk, subl) bar triplets that arise
    from multiple original monomials collapsing into one canonical row
    are aggregated via scipy-style sort+sum before submission.
  * σ-rep dropping for localizing + window cones: pass ``active_loc`` and
    ``active_windows`` restricted to one representative per orbit
    (``lasserre.z2_blockdiag.localizing_sigma_reps`` / ``window_sigma_reps``).

The consistency / simplex equalities are left in the dual as free μ_k, λ
multipliers (no xf.T composition on stationarity rows — equivalent
verdicts, slightly more scalar variables but no stationarity-row cost
difference).

================================================================================
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import mosek

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'tests'))

# Hash utilities (pure-numpy monomial lookup).
from lasserre_fusion import _hash_monos, _hash_lookup  # noqa: E402


__all__ = [
    'build_dual_task',
    'solve_dual_task',
    'update_task_t',
    'DualBuildInfo',
]


# ---------------------------------------------------------------------------
# Helper: canonical alpha lookup that remaps through old_to_new when the
# precompute has been canonicalized via canonicalize_z2.
# ---------------------------------------------------------------------------

def _alpha_lookup(alpha_hash: np.ndarray,
                   sorted_h: np.ndarray, sort_o: np.ndarray,
                   old_to_new: Optional[np.ndarray]) -> np.ndarray:
    """Vectorised monomial-hash → (canonical) row index.  Returns -1 for
    monomials not present in the support.

    If ``old_to_new`` is provided (canonicalize_z2 applied), the result is
    remapped from original-y indices to canonical-ỹ indices.  The ``-1``
    sentinel is preserved.
    """
    raw = _hash_lookup(alpha_hash, sorted_h, sort_o)
    if old_to_new is None:
        return raw
    out = np.where(raw < 0, -1, old_to_new[np.maximum(raw, 0)])
    return out


# ---------------------------------------------------------------------------
# Helper: scipy-style COO → deduplicated (sum-duplicates) triplet,
# operating on the key (subi, subj, subk, subl).
# ---------------------------------------------------------------------------

def _aggregate_bar_triplet(subi: np.ndarray, subj: np.ndarray,
                            subk: np.ndarray, subl: np.ndarray,
                            val:  np.ndarray,
                            t_coef: Optional[np.ndarray] = None,
                            ) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray,
                                       Optional[np.ndarray]]:
    """Aggregate duplicate (subi, subj, subk, subl) entries by summing
    their ``val`` (and optionally ``t_coef``) contributions.  Zeros after
    aggregation are dropped.

    MOSEK's ``putbarablocktriplet`` rejects duplicate index quadruples
    with ``err_sym_mat_duplicate``; after ``canonicalize_z2`` or σ-rep
    merging, duplicates are common (multiple raw monomials collapsing to
    the same canonical row at the same (k, l) of the same bar variable).

    When ``t_coef`` is supplied (one coefficient-of-t per contribution),
    it is aggregated in parallel so the caller can separate the
    t-dependent part from the static part after aggregation — used by the
    task-reuse code path in ``solve_mosek_dual`` to update only the
    entries whose value changes with t.

    Zero-drop policy: an entry is dropped iff BOTH aggregated val and
    aggregated t_coef (if supplied) round to zero.  An entry with
    ``val==0, t_coef!=0`` is kept (it's a pure-t entry).
    """
    if subi.size == 0:
        empty = np.array([], dtype=np.int32)
        return (empty, empty, empty, empty,
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64) if t_coef is not None else None)

    # Pack (subi, subj, subk, subl) into a single int64 key for sorting.
    # Pick per-field widths based on actual ranges, verifying the
    # total fits in 62 bits to leave headroom for int64 arithmetic.
    subi64 = subi.astype(np.int64)
    subj64 = subj.astype(np.int64)
    subk64 = subk.astype(np.int64)
    subl64 = subl.astype(np.int64)

    def _bits(n: int) -> int:
        # Minimum bits needed to represent values in [0, n].
        return max(1, int(np.ceil(np.log2(n + 2))))

    w_i = _bits(int(subi64.max()))
    w_j = _bits(int(subj64.max()))
    w_k = _bits(int(subk64.max()))
    w_l = _bits(int(subl64.max()))
    if w_i + w_j + w_k + w_l > 62:
        raise RuntimeError(
            f"_aggregate_bar_triplet: packed key needs {w_i+w_j+w_k+w_l} "
            f"bits, exceeds 62.  (Per-field widths: i={w_i} j={w_j} "
            f"k={w_k} l={w_l}.)  Cannot safely pack in int64.")

    shift_l = 0
    shift_k = w_l
    shift_j = w_l + w_k
    shift_i = w_l + w_k + w_j
    key = ((subi64 << shift_i)
           | (subj64 << shift_j)
           | (subk64 << shift_k)
           | (subl64 << shift_l))

    order = np.argsort(key, kind='stable')
    key_s = key[order]
    val_s = val[order]
    subi_s = subi[order]
    subj_s = subj[order]
    subk_s = subk[order]
    subl_s = subl[order]

    # Group summation via np.add.reduceat at run boundaries.
    new_run = np.concatenate(([True], key_s[1:] != key_s[:-1]))
    starts = np.nonzero(new_run)[0]
    val_sum = np.add.reduceat(val_s, starts)

    if t_coef is not None:
        t_coef_s = t_coef[order]
        tc_sum = np.add.reduceat(t_coef_s, starts)
        # Keep an entry iff either its static or its t-coefficient is non-zero.
        keep = (np.abs(val_sum) > 0.0) | (np.abs(tc_sum) > 0.0)
        return (subi_s[starts][keep], subj_s[starts][keep],
                subk_s[starts][keep], subl_s[starts][keep],
                val_sum[keep], tc_sum[keep])
    else:
        keep = np.abs(val_sum) > 0.0
        return (subi_s[starts][keep], subj_s[starts][keep],
                subk_s[starts][keep], subl_s[starts][keep],
                val_sum[keep], None)


# ---------------------------------------------------------------------------
# Helper: scalar (A-matrix) triplet aggregation, on (row, col) key.
# ---------------------------------------------------------------------------

def _aggregate_scalar_triplet(rows: np.ndarray, cols: np.ndarray,
                               vals: np.ndarray,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate (row, col) duplicates by summing values.  ``putaijlist``
    accepts duplicates (they'd sum), but scipy sparse construction is
    cleaner and cheaper when we pre-aggregate."""
    if rows.size == 0:
        return rows, cols, vals
    M = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(int(rows.max()) + 1, int(cols.max()) + 1),
        dtype=np.float64,
    )
    M.sum_duplicates()
    M.eliminate_zeros()
    coo = M.tocoo()
    return (coo.row.astype(np.int32, copy=False),
            coo.col.astype(np.int32, copy=False),
            coo.data.astype(np.float64, copy=False))


# =========================================================================
# Top-level builder
# =========================================================================

def build_dual_task(
    P: Dict[str, Any], *,
    t_val: float,
    env: Optional[mosek.Env] = None,
    include_upper_loc: bool = False,
    z2_blockdiag_map: Optional[Dict[str, Any]] = None,
    active_loc: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
    lambda_upper_bound: float = 1.0,
    verbose: bool = True,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the Farkas-infeasibility dual SDP as a ``mosek.Task``.

    Parameters
    ----------
    P                  : precompute dict from ``lasserre_scalable._precompute``,
                         optionally post-processed via
                         ``lasserre.z2_elim.canonicalize_z2``.
                         Required keys: d, n_y, basis/n_basis, loc_basis/n_loc,
                         idx, mono_list, M_mats, windows, nontrivial_windows,
                         consist_mono, consist_idx, consist_ei_idx, bases,
                         sorted_h, sort_o.  Optional: old_to_new (when
                         canonicalized).
    t_val              : value of the bisection parameter t baked into the
                         window sensitivity matrices.  Re-call to probe a
                         different t.
    env                : optional pre-created MOSEK environment.
    include_upper_loc  : add X'_i bar variables for (1 − μ_i) ≥ 0 localizing
                         cones (matches ``add_upper_loc=True`` in the primal).
    z2_blockdiag_map   : dict with keys ('T_sym','T_anti','n_sym','n_anti')
                         from ``lasserre.z2_blockdiag.build_blockdiag_picks``.
                         When supplied, the moment BAR variable is replaced
                         by two BAR blocks (sym + anti).  REQUIRES the
                         precompute to be canonicalized (Z/2 pre-elim).
    active_loc         : localizing indices i to include as BAR cones.
                         Defaults to range(d); pass
                         ``fixed + [p for p,_ in pairs]`` from
                         ``localizing_sigma_reps(d)`` for σ-rep dropping.
    active_windows     : window indices (positions into ``P['windows']``) to
                         include.  Defaults to ``P['nontrivial_windows']``.
                         Pass restricted list via ``window_sigma_reps`` for
                         σ-rep dropping.
    lambda_upper_bound : cap on λ; the max-λ objective converges to this
                         value iff the primal is infeasible, else to 0.

    Returns
    -------
    task : mosek.Task with the Farkas LP fully encoded (objective: max λ).
    info : dict with bar-var sizes, variable offsets, cone counts, etc.
    """
    d = int(P['d'])
    n_y = int(P['n_y'])
    basis = P['basis']
    n_basis = int(P['n_basis'])
    loc_basis = P['loc_basis']
    n_loc = int(P['n_loc'])
    mono_idx = P['idx']
    M_mats = P['M_mats']
    bases_arr = np.asarray(P['bases'], dtype=np.int64)
    sorted_h = np.asarray(P['sorted_h'])
    sort_o = np.asarray(P['sort_o'])
    consist_mono = P['consist_mono']
    consist_idx = np.asarray(P['consist_idx'], dtype=np.int64)
    consist_ei_idx = np.asarray(P['consist_ei_idx'], dtype=np.int64)
    old_to_new_arr = P.get('old_to_new')
    if old_to_new_arr is not None:
        old_to_new_arr = np.asarray(old_to_new_arr, dtype=np.int64)

    z2_canon = old_to_new_arr is not None

    if z2_blockdiag_map is not None and not z2_canon:
        raise ValueError(
            "z2_blockdiag_map requires the precompute to be canonicalized "
            "via lasserre.z2_elim.canonicalize_z2 (so that y is σ-invariant "
            "in the reduced basis).  Call canonicalize_z2(P) first.")

    # At order=1 there are no localising cones (loc_basis is empty) and no
    # window PSD cones — only the top moment matrix.  Force both lists empty
    # so ``appendbarvars`` never receives a zero-dim block.
    if n_loc == 0:
        active_loc = []
        active_windows = []
    else:
        if active_loc is None:
            active_loc = list(range(d))
        if active_windows is None:
            active_windows = list(P['nontrivial_windows'])

    if env is None:
        env = mosek.Env()
    task = env.Task()

    if verbose:
        task.set_Stream(
            mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # -----------------------------------------------------------------
    # 1. Bar-variable layout.
    #
    #   [moment]   either (X_0) of size n_basis,
    #              or (X_0^sym, X_0^anti) of sizes (n_sym, n_anti) when
    #              z2_blockdiag_map is provided.
    #   [loc_i]    one BAR of size n_loc per i in active_loc.
    #   [uloc_i]   (if include_upper_loc) one BAR of size n_loc per i in
    #              active_loc.
    #   [win_W]    one BAR of size n_loc per W in active_windows.
    # -----------------------------------------------------------------
    bar_sizes: List[int] = []
    moment_bar_ids: List[int] = []

    if z2_blockdiag_map is None:
        moment_bar_ids.append(0)
        bar_sizes.append(n_basis)
    else:
        n_sym = int(z2_blockdiag_map['n_sym'])
        n_anti = int(z2_blockdiag_map['n_anti'])
        moment_bar_ids.append(0)  # X_0^sym
        bar_sizes.append(n_sym)
        if n_anti > 0:
            moment_bar_ids.append(1)  # X_0^anti
            bar_sizes.append(n_anti)

    loc_bar_start = len(bar_sizes)
    for _ in active_loc:
        bar_sizes.append(n_loc)
    loc_bar_end = len(bar_sizes)

    uloc_bar_start = loc_bar_end
    if include_upper_loc:
        for _ in active_loc:
            bar_sizes.append(n_loc)
    uloc_bar_end = len(bar_sizes)

    win_bar_start = uloc_bar_end
    for _ in active_windows:
        bar_sizes.append(n_loc)
    win_bar_end = len(bar_sizes)

    n_bar = len(bar_sizes)
    task.appendbarvars(bar_sizes)

    # -----------------------------------------------------------------
    # 2. Scalar variables: [λ | μ_k (kept consist eqs) | v_α].
    # -----------------------------------------------------------------
    kept_k: List[int] = [
        k for k in range(len(consist_mono)) if int(consist_idx[k]) >= 0
    ]
    n_consist = len(kept_k)

    LAMBDA_IDX = 0
    MU_START = 1
    MU_END = 1 + n_consist
    V_START = MU_END
    V_END = V_START + n_y
    n_scalar = V_END

    task.appendvars(n_scalar)

    # Variable bounds.
    task.putvarbound(
        LAMBDA_IDX, mosek.boundkey.ra, 0.0, float(lambda_upper_bound))
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

    # -----------------------------------------------------------------
    # 3. Constraint rows: one per α (stationarity), all = 0.
    # -----------------------------------------------------------------
    task.appendcons(n_y)
    task.putconboundslice(
        0, n_y,
        [mosek.boundkey.fx] * n_y,
        np.zeros(n_y, dtype=np.float64),
        np.zeros(n_y, dtype=np.float64),
    )

    # -----------------------------------------------------------------
    # 4. Scalar coefficients (A matrix).
    # -----------------------------------------------------------------
    scalar_rows: List[int] = []
    scalar_cols: List[int] = []
    scalar_vals: List[float] = []

    # λ: +1 at α = 0.
    alpha_zero = tuple(0 for _ in range(d))
    if alpha_zero not in mono_idx:
        raise RuntimeError(
            "Zero monomial (0,...,0) is missing from P['idx'] — required by "
            "Lasserre normalisation y_0 = 1.")
    alpha_zero_row = int(mono_idx[alpha_zero])
    scalar_rows.append(alpha_zero_row)
    scalar_cols.append(LAMBDA_IDX)
    scalar_vals.append(1.0)

    # μ_k: C_{k, α} = (#{i : consist_mono[k] + e_i = α}) − 𝟙[consist_mono[k] = α].
    # After canonicalize_z2, different children may map to the same row;
    # scalar aggregation handles this.
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

    # v_α: +1 on diagonal.
    alpha_rows = np.arange(n_y, dtype=np.int64)
    scalar_rows.extend(alpha_rows.tolist())
    scalar_cols.extend((V_START + alpha_rows).tolist())
    scalar_vals.extend([1.0] * n_y)

    # Aggregate + submit.
    r_arr, c_arr, v_arr = _aggregate_scalar_triplet(
        np.asarray(scalar_rows, dtype=np.int64),
        np.asarray(scalar_cols, dtype=np.int64),
        np.asarray(scalar_vals, dtype=np.float64),
    )
    if r_arr.size:
        task.putaijlist(r_arr, c_arr, v_arr)

    # -----------------------------------------------------------------
    # 5. Bar-matrix sensitivity coefficients.
    # -----------------------------------------------------------------
    bar_subi_list: List[np.ndarray] = []
    bar_subj_list: List[np.ndarray] = []
    bar_subk_list: List[np.ndarray] = []
    bar_subl_list: List[np.ndarray] = []
    bar_val_list:  List[np.ndarray] = []
    bar_tcoef_list: List[np.ndarray] = []  # parallel coefficient-of-t array

    def _append(subi, subj, subk, subl, vals, tcoefs=None):
        if subi.size == 0:
            return
        bar_subi_list.append(np.ascontiguousarray(subi, dtype=np.int32))
        bar_subj_list.append(np.ascontiguousarray(subj, dtype=np.int32))
        bar_subk_list.append(np.ascontiguousarray(subk, dtype=np.int32))
        bar_subl_list.append(np.ascontiguousarray(subl, dtype=np.int32))
        bar_val_list.append(np.ascontiguousarray(vals, dtype=np.float64))
        if tcoefs is None:
            bar_tcoef_list.append(np.zeros(subi.size, dtype=np.float64))
        else:
            bar_tcoef_list.append(
                np.ascontiguousarray(tcoefs, dtype=np.float64))

    # ---- Moment cone: either single BAR X_0 or (sym + anti) blocks ----
    if z2_blockdiag_map is None:
        # Single n_basis × n_basis BAR.
        B_arr = np.asarray(basis, dtype=np.int64)
        B_hash = _hash_monos(B_arr, bases_arr)
        ks_m, ls_m = np.tril_indices(n_basis)
        alpha_hash_m = B_hash[ks_m] + B_hash[ls_m]
        alpha_idx_m = _alpha_lookup(
            alpha_hash_m, sorted_h, sort_o, old_to_new_arr)
        if np.any(alpha_idx_m < 0):
            raise RuntimeError(
                "Moment sensitivity lookup produced -1; precompute P is "
                "inconsistent with its hash table.")
        _append(
            alpha_idx_m,
            np.full(ks_m.shape, moment_bar_ids[0], dtype=np.int32),
            ks_m, ls_m,
            np.full(ks_m.shape, +1.0, dtype=np.float64),
        )
    else:
        # Block-diagonalised moment cone.  T_sym/T_anti have columns in
        # canonical-ỹ space (since build_blockdiag_picks was called on
        # the canonicalized P['idx']).  Extract lower triangle via flat
        # row = u*n+v, u ≥ v.
        T_sym = z2_blockdiag_map['T_sym'].tocoo()
        n_sym_here = int(z2_blockdiag_map['n_sym'])
        if T_sym.nnz:
            u = T_sym.row // n_sym_here
            v = T_sym.row % n_sym_here
            alpha_col = T_sym.col
            val = T_sym.data
            # Keep u ≥ v.  The matrix is symmetric (M_sym[u,v] = M_sym[v,u]
            # by construction), so submitting just the lower triangle is
            # correct under MOSEK's symmetric bar-matrix convention.
            mask = u >= v
            _append(
                alpha_col[mask].astype(np.int32),
                np.full(int(mask.sum()), moment_bar_ids[0], dtype=np.int32),
                u[mask].astype(np.int32), v[mask].astype(np.int32),
                val[mask].astype(np.float64),
            )

        if len(moment_bar_ids) > 1:
            T_anti = z2_blockdiag_map['T_anti'].tocoo()
            n_anti_here = int(z2_blockdiag_map['n_anti'])
            if T_anti.nnz:
                u = T_anti.row // n_anti_here
                v = T_anti.row % n_anti_here
                alpha_col = T_anti.col
                val = T_anti.data
                mask = u >= v
                _append(
                    alpha_col[mask].astype(np.int32),
                    np.full(int(mask.sum()), moment_bar_ids[1],
                            dtype=np.int32),
                    u[mask].astype(np.int32), v[mask].astype(np.int32),
                    val[mask].astype(np.float64),
                )

    # ---- Localizing cones X_i: coef = +1, α = loc[k] + loc[l] + e_i ----
    # At order=1 (n_loc==0) the Lasserre relaxation has no localizing or
    # window PSD cones — skip the entire block.
    if n_loc > 0:
        L_arr = np.asarray(loc_basis, dtype=np.int64)
        L_hash = _hash_monos(L_arr, bases_arr)
        ks_l, ls_l = np.tril_indices(n_loc)
        base_hash_loc = L_hash[ks_l] + L_hash[ls_l]  # α = loc[k] + loc[l]
        alpha_idx_loc0 = _alpha_lookup(
            base_hash_loc, sorted_h, sort_o, old_to_new_arr)
    else:
        L_arr = np.zeros((0, d), dtype=np.int64)
        L_hash = np.zeros(0, dtype=np.int64)
        ks_l = np.zeros(0, dtype=np.int64)
        ls_l = np.zeros(0, dtype=np.int64)
        base_hash_loc = np.zeros(0, dtype=np.int64)
        alpha_idx_loc0 = np.zeros(0, dtype=np.int64)

    for j, i in enumerate(active_loc):
        alpha_hash_li = base_hash_loc + bases_arr[i]
        alpha_idx_li = _alpha_lookup(
            alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
        mask = alpha_idx_li >= 0
        n_m = int(mask.sum())
        if n_m == 0:
            continue
        bar_idx_here = loc_bar_start + j
        _append(
            alpha_idx_li[mask],
            np.full(n_m, bar_idx_here, dtype=np.int32),
            ks_l[mask], ls_l[mask],
            np.full(n_m, +1.0, dtype=np.float64),
        )

    # ---- Upper-localizing cones X'_i (optional) ----
    #   E'_i[α] = E_t[α] − E_i[α], coefficients ±1.
    if include_upper_loc:
        mask_t0 = alpha_idx_loc0 >= 0
        for j, i in enumerate(active_loc):
            bar_idx_here = uloc_bar_start + j
            # +1·𝟙[loc+loc = α]
            if np.any(mask_t0):
                n_m = int(mask_t0.sum())
                _append(
                    alpha_idx_loc0[mask_t0],
                    np.full(n_m, bar_idx_here, dtype=np.int32),
                    ks_l[mask_t0], ls_l[mask_t0],
                    np.full(n_m, +1.0, dtype=np.float64),
                )
            # −1·𝟙[loc+loc+e_i = α]
            alpha_hash_li = base_hash_loc + bases_arr[i]
            alpha_idx_li = _alpha_lookup(
                alpha_hash_li, sorted_h, sort_o, old_to_new_arr)
            mask = alpha_idx_li >= 0
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            _append(
                alpha_idx_li[mask],
                np.full(n_m, bar_idx_here, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, -1.0, dtype=np.float64),
            )

    # ---- Window cones X_W: +t·E_W^t − E_W^Q ----
    for w_j, w in enumerate(active_windows):
        Mw = np.asarray(M_mats[w], dtype=np.float64)
        nz_i, nz_j = np.nonzero(Mw)
        W_bar_idx = win_bar_start + w_j

        # (a) t-part: coef = +t at (α = loc[k] + loc[l]).
        # Recorded with val=0 and t_coef=+1 so the task-reuse path can
        # rewrite this entry when t changes without disturbing static terms.
        mask_t = alpha_idx_loc0 >= 0
        n_m = int(mask_t.sum())
        if n_m:
            _append(
                alpha_idx_loc0[mask_t],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks_l[mask_t], ls_l[mask_t],
                np.zeros(n_m, dtype=np.float64),
                tcoefs=np.ones(n_m, dtype=np.float64),
            )

        # (b) Q-part: −M_W[ii, jj] at (α = loc[k] + loc[l] + e_ii + e_jj).
        # M_W is symmetric; iterate ii ≥ jj and double the coefficient
        # for the strict off-diagonals so we don't submit duplicates.
        for ii, jj in zip(nz_i.tolist(), nz_j.tolist()):
            if ii < jj:
                continue
            raw = float(Mw[ii, jj])
            if raw == 0.0:
                continue
            coef = -raw if ii == jj else -2.0 * raw
            alpha_hash_q = (base_hash_loc
                            + bases_arr[ii] + bases_arr[jj])
            alpha_idx_q = _alpha_lookup(
                alpha_hash_q, sorted_h, sort_o, old_to_new_arr)
            mask = alpha_idx_q >= 0
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            _append(
                alpha_idx_q[mask],
                np.full(n_m, W_bar_idx, dtype=np.int32),
                ks_l[mask], ls_l[mask],
                np.full(n_m, coef, dtype=np.float64),
            )

    # ---- Concatenate + aggregate + bulk submit ----
    # We carry a parallel ``t_coef`` array through aggregation so the
    # task-reuse path can identify entries whose value depends on t
    # and rewrite just those on a t-change.
    dyn_subi = dyn_subj = dyn_subk = dyn_subl = None
    dyn_static = dyn_tcoef = None
    if bar_subi_list:
        all_subi = np.concatenate(bar_subi_list)
        all_subj = np.concatenate(bar_subj_list)
        all_subk = np.concatenate(bar_subk_list)
        all_subl = np.concatenate(bar_subl_list)
        all_val = np.concatenate(bar_val_list)
        all_tcoef = np.concatenate(bar_tcoef_list)

        all_subi, all_subj, all_subk, all_subl, all_val, all_tcoef = \
            _aggregate_bar_triplet(
                all_subi, all_subj, all_subk, all_subl,
                all_val, all_tcoef)
        n_bar_entries = int(all_subi.size)
        if n_bar_entries:
            # Initial submission: static part + t_val * t_coef.
            init_vals = all_val + float(t_val) * all_tcoef
            task.putbarablocktriplet(
                all_subi, all_subj, all_subk, all_subl, init_vals)

            # Cache dynamic (t-dependent) entries for ``update_task_t``.
            is_dyn = np.abs(all_tcoef) > 0.0
            if np.any(is_dyn):
                dyn_subi = all_subi[is_dyn].copy()
                dyn_subj = all_subj[is_dyn].copy()
                dyn_subk = all_subk[is_dyn].copy()
                dyn_subl = all_subl[is_dyn].copy()
                dyn_static = all_val[is_dyn].copy()
                dyn_tcoef = all_tcoef[is_dyn].copy()
    else:
        n_bar_entries = 0

    # -----------------------------------------------------------------
    # 6. Objective: maximize λ.
    # -----------------------------------------------------------------
    task.putobjsense(mosek.objsense.maximize)
    task.putcj(LAMBDA_IDX, 1.0)

    build_time = time.time() - t0

    info = {
        'build_time_s': build_time,
        'bar_sizes': bar_sizes,
        'n_bar': n_bar,
        'moment_bar_ids': moment_bar_ids,
        'loc_bar_start': loc_bar_start,
        'loc_bar_end': loc_bar_end,
        'uloc_bar_start': uloc_bar_start,
        'uloc_bar_end': uloc_bar_end,
        'win_bar_start': win_bar_start,
        'win_bar_end': win_bar_end,
        'active_loc': list(active_loc),
        'active_windows': list(active_windows),
        'n_scalar': n_scalar,
        'n_consist_kept': n_consist,
        'n_cons': n_y,
        'n_y': n_y,
        'n_bar_entries': n_bar_entries,
        't_val': float(t_val),
        'LAMBDA_IDX': LAMBDA_IDX,
        'MU_START': MU_START,
        'V_START': V_START,
        'lambda_upper_bound': float(lambda_upper_bound),
        'z2_canonicalized': z2_canon,
        'z2_blockdiag': z2_blockdiag_map is not None,
        'include_upper_loc': bool(include_upper_loc),
        # --- task-reuse payload (for update_task_t) ---
        # We cache the FULL aggregated triplet so update_task_t can re-submit
        # every bar entry (not a subset) — MOSEK's putbarablocktriplet on
        # subset-updates appears to be unreliable after a previous optimize()
        # at d ≥ 6 L=3, so we always resubmit the complete triplet.
        '_all_subi': all_subi if bar_subi_list else None,
        '_all_subj': all_subj if bar_subi_list else None,
        '_all_subk': all_subk if bar_subi_list else None,
        '_all_subl': all_subl if bar_subi_list else None,
        '_all_static': all_val if bar_subi_list else None,
        '_all_tcoef': all_tcoef if bar_subi_list else None,
        '_dyn_count': 0 if dyn_subi is None else int(dyn_subi.size),
        'n_bar_entries_total': n_bar_entries,
        'n_dynamic_entries': 0 if dyn_subi is None else int(dyn_subi.size),
    }

    if verbose:
        print(f"  [dual-task] n_bar={n_bar}  n_scalar={n_scalar:,}  "
              f"n_cons={n_y:,}  n_bar_entries={n_bar_entries:,}  "
              f"n_dyn={info['n_dynamic_entries']:,}  "
              f"t={t_val:.6f}  build={build_time:.2f}s  "
              f"(z2_canon={z2_canon} blockdiag={info['z2_blockdiag']} "
              f"upper_loc={include_upper_loc} "
              f"n_loc_active={len(active_loc)} "
              f"n_win_active={len(active_windows)})",
              flush=True)

    return task, info


# =====================================================================
# Task-reuse update: only re-submit the t-dependent bar entries
# =====================================================================

def update_task_t(task: mosek.Task, info: Dict[str, Any],
                   t_val: float) -> None:
    """Update the t-dependent bar entries of an existing task built by
    ``build_dual_task`` so that the Farkas LP now represents feasibility
    at the new ``t_val``.  The caller can then re-run ``task.optimize()``.

    Only the entries whose value depends on t (cached in ``info`` as
    ``_dyn_*``) are re-submitted.  Everything else — symbolic factor
    ordering, presolve structure, constraint matrix sparsity pattern —
    is preserved so MOSEK can reuse it across successive optimize() calls.

    If there are no t-dependent entries (e.g. a problem without window
    cones), this is a no-op.

    Updates ``info['t_val']`` in place.
    """
    all_subi = info.get('_all_subi')
    if all_subi is None or all_subi.size == 0:
        info['t_val'] = float(t_val)
        return
    all_subj = info['_all_subj']
    all_subk = info['_all_subk']
    all_subl = info['_all_subl']
    all_static = info['_all_static']
    all_tcoef = info['_all_tcoef']

    # Re-submit the ENTIRE aggregated bar triplet with values evaluated
    # at the new t.  Subset updates (sending only the t-dependent subset)
    # turned out to give inconsistent results after successive optimizes at
    # d ≥ 6 L=3.  Resubmitting the full triplet is unambiguously correct
    # and still avoids the Python-side assembly cost (aggregation /
    # lookups) from a cold rebuild — it's the hot bulk-C-call only.
    new_vals = all_static + float(t_val) * all_tcoef
    task.putbarablocktriplet(
        all_subi, all_subj, all_subk, all_subl, new_vals)
    info['t_val'] = float(t_val)

    # Drop any cached prior solution + force cold IPM start on the next
    # optimize().  Observed at d=6 L=3 full-stack: without this, MOSEK
    # reuses the previous iterate as a warm-start, traps the IPM in the
    # prior verdict's cone, and reports the stale verdict even after the
    # coefficients change.
    try:
        task.putintparam(mosek.iparam.intpnt_hotstart,
                           mosek.intpnthotstart.none)
        task.putintparam(mosek.iparam.intpnt_starting_point,
                           mosek.startpointtype.free)
    except Exception:
        pass
    for sol_kind in (mosek.soltype.itr, mosek.soltype.bas, mosek.soltype.itg):
        try:
            task.deletesolution(sol_kind)
        except Exception:
            pass


# =====================================================================
# Solve + verdict
# =====================================================================

def solve_dual_task(
    task: mosek.Task,
    info: Dict[str, Any], *,
    feas_threshold: float = 0.25,
    infeas_threshold: float = 0.75,
    early_stop_on_clear_verdict: bool = False,
    early_stop_gap_tol: float = 1e-2,
    early_stop_feas_frac: float = 0.15,
    early_stop_infeas_frac: float = 0.85,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run ``task.optimize()`` and classify the primal (moment) feasibility
    at the t_val baked into the task.

    The Farkas LP is ``max λ`` with ``0 ≤ λ ≤ lambda_upper_bound``.  The
    task is always feasible (the zero dual certificate works), so MOSEK
    returns OPTIMAL.  The value of λ* determines the verdict:

        λ* ≥ infeas_threshold · Λ   ⟹  Farkas cert found → primal INFEASIBLE
        λ* ≤ feas_threshold   · Λ   ⟹  no cert           → primal FEASIBLE
        otherwise                    ⟹  uncertain

    where Λ = lambda_upper_bound.  Default thresholds 0.25 / 0.75 are
    deliberately asymmetric — boundary cases land at intermediate λ* under
    finite solver tolerance.
    """
    lam_ub = float(info['lambda_upper_bound'])

    # B1 sacrifice: info-callback-based early termination once λ* is clearly
    # in the FEAS or INFEAS zone.  Returning non-zero from the info callback
    # tells MOSEK to halt the current optimization.  The returned solution
    # is still the last interior iterate — good enough for the Farkas
    # verdict (feas_threshold=0.25, infeas_threshold=0.75 leaves a large
    # margin, and we only stop inside 0.15/0.85).
    _early_stop_state = {'triggered': False, 'iter': -1,
                          'primal_obj': float('nan'),
                          'gap': float('nan')}
    if early_stop_on_clear_verdict:
        fthr = float(early_stop_feas_frac) * lam_ub
        ithr = float(early_stop_infeas_frac) * lam_ub
        gtol = float(early_stop_gap_tol)

        def _info_cb(caller, dinfo, iinfo, liinfo):
            try:
                if caller != mosek.callbackcode.intpnt:
                    return 0
                primal = dinfo[mosek.dinfitem.intpnt_primal_obj]
                dual = dinfo[mosek.dinfitem.intpnt_dual_obj]
                it = iinfo[mosek.iinfitem.intpnt_iter]
                denom = max(1.0, abs(primal), abs(dual))
                gap = abs(primal - dual) / denom
                clear_feas = (abs(primal) < fthr) and (gap < gtol)
                clear_infeas = (primal > ithr) and (gap < gtol)
                if (clear_feas or clear_infeas) and it >= 3:
                    _early_stop_state['triggered'] = True
                    _early_stop_state['iter'] = int(it)
                    _early_stop_state['primal_obj'] = float(primal)
                    _early_stop_state['gap'] = float(gap)
                    return 1  # non-zero => MOSEK halts optimization
            except Exception:
                return 0
            return 0

        try:
            task.set_InfoCallback(_info_cb)
        except Exception as exc:
            if verbose:
                print(f"  [early-stop] set_InfoCallback failed: {exc}; "
                      f"running without callback.", flush=True)

    ts = time.time()
    try:
        task.optimize()
    except Exception as exc:
        return {
            'verdict': 'uncertain',
            'status': f'error:{type(exc).__name__}:{exc}',
            'lambda_star': float('nan'),
            'wall_s': time.time() - ts,
        }
    wall = time.time() - ts

    solsta = task.getsolsta(mosek.soltype.itr)
    prosta = task.getprosta(mosek.soltype.itr)
    try:
        lam_star = float(task.getxxslice(
            mosek.soltype.itr, info['LAMBDA_IDX'],
            info['LAMBDA_IDX'] + 1)[0])
    except Exception:
        lam_star = float('nan')

    if solsta == mosek.solsta.optimal:
        if lam_star >= infeas_threshold * lam_ub:
            verdict = 'infeas'
        elif lam_star <= feas_threshold * lam_ub:
            verdict = 'feas'
        else:
            verdict = 'uncertain'
    elif solsta == mosek.solsta.dual_infeas_cer:
        verdict = 'infeas'
    elif solsta == mosek.solsta.prim_infeas_cer:
        verdict = 'uncertain'
    elif (early_stop_on_clear_verdict
          and _early_stop_state['triggered']
          and not (lam_star != lam_star)):  # lam_star is finite
        # B1: callback halted optimization.  Classify from current λ*.
        if lam_star >= infeas_threshold * lam_ub:
            verdict = 'infeas'
        elif lam_star <= feas_threshold * lam_ub:
            verdict = 'feas'
        else:
            verdict = 'uncertain'
    else:
        verdict = 'uncertain'

    status = (f"solsta={str(solsta).split('.')[-1]} "
              f"prosta={str(prosta).split('.')[-1]} "
              f"lam*={lam_star:.6e}")

    if verbose:
        print(f"  [dual-task] {status}  verdict={verdict}  "
              f"solve={wall:.2f}s", flush=True)

    out = {
        'verdict': verdict,
        'status': status,
        'lambda_star': lam_star,
        'solsta': str(solsta),
        'prosta': str(prosta),
        'wall_s': wall,
    }
    if early_stop_on_clear_verdict and _early_stop_state['triggered']:
        out['early_stop'] = {
            'triggered': True,
            'iter': _early_stop_state['iter'],
            'primal_obj_at_halt': _early_stop_state['primal_obj'],
            'gap_at_halt': _early_stop_state['gap'],
        }
        if verbose:
            print(f"  [early-stop] halted at IPM iter "
                  f"{_early_stop_state['iter']}  "
                  f"primal={_early_stop_state['primal_obj']:.3e}  "
                  f"gap={_early_stop_state['gap']:.3e}", flush=True)
    return out


# =====================================================================
# Metadata container
# =====================================================================

class DualBuildInfo(dict):
    """Thin typed wrapper around the info dict returned by build_dual_task."""
    bar_sizes: List[int]
    n_bar: int
    moment_bar_ids: List[int]
    loc_bar_start: int
    loc_bar_end: int
    uloc_bar_start: int
    uloc_bar_end: int
    win_bar_start: int
    win_bar_end: int
    active_loc: List[int]
    active_windows: List[int]
    n_scalar: int
    n_consist_kept: int
    n_cons: int
    n_y: int
    n_bar_entries: int
    t_val: float
    LAMBDA_IDX: int
    MU_START: int
    V_START: int
    lambda_upper_bound: float
    build_time_s: float
    z2_canonicalized: bool
    z2_blockdiag: bool
    include_upper_loc: bool


# =====================================================================
# Rank diagnostic of Farkas certificates (Technique B)
# =====================================================================

def _unpack_lower_tri(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack MOSEK lower-triangle packed vector into a symmetric (n, n) matrix.

    MOSEK's task.getbarsj returns the lower triangle in column-major order:
    [X[0,0], X[1,0], ..., X[n-1,0], X[1,1], X[2,1], ..., X[n-1,1], ..., X[n-1,n-1]].
    """
    X = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for j in range(n):
        for i in range(j, n):
            X[i, j] = packed[idx]
            if i != j:
                X[j, i] = packed[idx]
            idx += 1
    return X


def compute_cert_rank(
    task,
    info: Dict[str, Any],
    *,
    tols: Tuple[float, ...] = (1e-4, 1e-6, 1e-8, 1e-10),
    top_k: int = 20,
    sample_windows: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Extract bar-variable solutions and measure rank / spectrum decay.

    Computes SVD of:
      * each moment bar (X_0, plus X_0^anti if Z/2 block-diag),
      * the largest localising cone (in Frobenius norm),
      * a sample of the largest window cones (up to ``sample_windows``).

    Returns dict with per-cone: singular values (top_k), rank at each tol,
    Frobenius norm, ratio spectrum[k]/spectrum[0] for k=0..top_k-1.
    """
    bar_sizes = info['bar_sizes']
    moment_bar_ids = info['moment_bar_ids']
    loc_bar_start = int(info['loc_bar_start'])
    loc_bar_end = int(info['loc_bar_end'])
    win_bar_start = int(info['win_bar_start'])
    win_bar_end = int(info['win_bar_end'])
    active_windows = list(info.get('active_windows', []))

    def _analyze(bar_idx: int, label: str) -> Dict[str, Any]:
        n = int(bar_sizes[bar_idx])
        packed = task.getbarsj(mosek.soltype.itr, bar_idx)
        packed = np.asarray(packed, dtype=np.float64)
        X = _unpack_lower_tri(packed, n)
        fro = float(np.linalg.norm(X, 'fro'))
        if fro == 0.0 or n == 0:
            return {
                'label': label, 'n': n, 'fro_norm': fro,
                'rank_by_tol': {f'{t:.0e}': 0 for t in tols},
                'top_singular': [], 'spec_ratio': [],
            }
        sig = np.linalg.svd(X, compute_uv=False)
        sig = np.asarray(sig, dtype=np.float64)
        sig_max = float(sig[0]) if sig.size > 0 else 0.0
        rank_by_tol = {}
        for tol in tols:
            rank_by_tol[f'{tol:.0e}'] = int(np.sum(sig > tol * sig_max))
        k_eff = min(top_k, sig.size)
        top = [float(sig[i]) for i in range(k_eff)]
        ratio = [float(sig[i] / sig_max) if sig_max > 0 else 0.0
                 for i in range(k_eff)]
        return {
            'label': label, 'n': n, 'fro_norm': fro,
            'rank_by_tol': rank_by_tol,
            'top_singular': top, 'spec_ratio': ratio,
            'sigma_max': sig_max,
        }

    out: Dict[str, Any] = {'cones': []}
    for i, bar_idx in enumerate(moment_bar_ids):
        tag = 'X_0' if i == 0 else f'X_0_anti_{i}'
        out['cones'].append(_analyze(bar_idx, tag))

    # Largest loc cone by Frobenius norm.
    loc_info: List[Dict[str, Any]] = []
    for j in range(loc_bar_start, loc_bar_end):
        loc_info.append(_analyze(j, f'loc_{j - loc_bar_start}'))
    if loc_info:
        loc_info.sort(key=lambda r: r.get('fro_norm', 0.0), reverse=True)
        out['cones'].append(loc_info[0])
        out['n_loc_cones'] = len(loc_info)

    # Sample largest window cones.
    win_info: List[Dict[str, Any]] = []
    for j in range(win_bar_start, win_bar_end):
        win_idx_in_active = j - win_bar_start
        w = (active_windows[win_idx_in_active]
             if win_idx_in_active < len(active_windows) else -1)
        win_info.append(_analyze(j, f'win_{w}'))
    if win_info:
        win_info.sort(key=lambda r: r.get('fro_norm', 0.0), reverse=True)
        for r in win_info[:sample_windows]:
            out['cones'].append(r)
        out['n_win_cones'] = len(win_info)

    if verbose:
        print("  [rank-diag]")
        for r in out['cones']:
            tol_str = "  ".join(
                f"tol={k}: r={v}" for k, v in r['rank_by_tol'].items())
            print(f"    {r['label']:<20} n={r['n']:<5} "
                  f"||X||_F={r['fro_norm']:.3e}  {tol_str}")
            if r.get('top_singular'):
                top5 = r['top_singular'][:5]
                top5_str = "  ".join(f"{s:.3e}" for s in top5)
                print(f"      top5 σ: {top5_str}")
        print(flush=True)

    return out

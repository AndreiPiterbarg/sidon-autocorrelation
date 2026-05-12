"""Z/2 pre-elimination: substitute y_α = y_{σ(α)} BEFORE MOSEK sees the problem.

================================================================================
MATHEMATICAL JUSTIFICATION (identical soundness to lasserre.z2_symmetry)
================================================================================

Let σ be the time-reversal involution on multi-indices, σ((α_0, …, α_{d−1})) =
(α_{d−1}, …, α_0).  Under the σ-equality constraints

    y_α = y_{σ(α)}   for every α with |α| ≤ 2k                     (⋆)

the moment vector y lives on the fixed-point subspace of the σ action.  Each
σ-orbit contributes exactly ONE scalar degree of freedom to that subspace:

    • a σ-fixed α (α = σ(α))        — one coordinate y_α,
    • an orbit pair {α, σ(α)} (α ≠ σ(α)) — one coordinate y_{canon} where
                                          canon ∈ {α, σ(α)} is the chosen
                                          representative.

Pre-elimination replaces the moment vector y ∈ R^{n_y} together with the (⋆)
equalities by a SMALLER vector ỹ ∈ R^{ñ_y} indexed by orbit representatives,
with NO σ-equalities.  Every reference y_α in the original problem is
rewritten y_α := ỹ_{canon(α)}, where canon sends every index to its orbit
representative.

**This is a variable substitution, not a relaxation.**  The feasible set in
the reduced space is IDENTICAL to the fixed-point subspace cut by the (⋆)
equalities.  Every SDP value — scalar-window bounds, moment PSD, localizing
PSD, window PSD — is preserved bit-exactly.

--------------------------------------------------------------------------------
Concrete rewriting rules
--------------------------------------------------------------------------------

1.  For any expression of the form  y[A]  where A is an index array into the
    original y vector, rewrite to  ỹ[old_to_new[A]] .  The new index array
    may contain REPEATS (different original entries that shared a σ-orbit);
    this is harmless for `Fusion.Expr.pick` and for MOSEK's linear-expression
    machinery.

2.  For any sparse coefficient matrix C with columns indexed by y (so that a
    constraint reads  C · y ⋈ b  for some relation ⋈), rewrite the column
    indices column_new = old_to_new[column_old] and AGGREGATE duplicate
    (row_new, column_new) entries by summing coefficients.  scipy.sparse
    matrix construction does this automatically via the COO → CSR / duplicate-
    summation path.

3.  Consistency equations  y_α = Σ_i y_{α + e_i}  :
    •  Two original equations (for α and σ(α)) are identical after substitution
       because σ takes α+e_i to σ(α)+e_{σ(i)} (index reversal also acts on
       unit vectors).  Deduplicate by emitting the equation for one orbit
       representative only.
    •  For a σ-fixed α, the d children pair up into d/2 σ-orbits (since d is
       even); each orbit contributes a single canonical index with coefficient
       2 in the reduced equation.  For a non-fixed α the d children lie in d
       distinct σ-orbits of the ambient index set (the σ-partners of α's
       children are children of σ(α), not of α), so the reduced equation has
       d distinct canonical coefficients of 1 (up to independent collisions,
       which we aggregate).

4.  Moment-matrix PSD (and all localizing / window PSD) cones are built from
    pick index arrays (Rule 1) and coefficient matrices (Rule 2).  After
    rewriting them, the PSD matrices are identical in value to the original
    constraints evaluated on a σ-invariant y.

--------------------------------------------------------------------------------
Soundness
--------------------------------------------------------------------------------

For any (t, y) feasible in the original SDP under the (⋆) equalities, the
canonicalized (t, ỹ) with ỹ_r = y_r is feasible in the reduced SDP: all
rewriting rules match the original values exactly.  Conversely, for any
(t, ỹ) feasible in the reduced SDP, the expansion y_α = ỹ_{canon(α)} satisfies
(⋆) (by construction) and every original constraint (each is built from
rewrites that evaluate to the same reduced-space value).  So the two feasible
sets project onto each other — lb is identical.

================================================================================
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


__all__ = ['canonicalize_z2', 'build_orbit_map']


# --------------------------------------------------------------------
# Orbit map
# --------------------------------------------------------------------

def build_orbit_map(mono_list: List[Tuple[int, ...]]
                     ) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """Compute the canonical σ-orbit representative for every multi-index.

    Parameters
    ----------
    mono_list : list of multi-index tuples of length d.

    Returns
    -------
    mono_list_canon : list of chosen orbit representatives (one per orbit).
                       Ordering is by first appearance in ``mono_list``.
    old_to_new      : int64 array of length len(mono_list).
                       old_to_new[i] = index into mono_list_canon for the
                       orbit containing mono_list[i].  Every original index
                       is mapped; no -1.

    Invariants
    ----------
    * old_to_new[i] == old_to_new[j]  whenever mono_list[j] = σ(mono_list[i]).
    * For a σ-fixed α: old_to_new maps one old index to one new index.
    * For an orbit pair {α, σα}: both old indices map to the same new index,
      associated with the smaller-original-index representative.
    """
    n_old = len(mono_list)
    idx_map = {tuple(m): i for i, m in enumerate(mono_list)}
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    mono_list_canon: List[Tuple[int, ...]] = []

    for i in range(n_old):
        if old_to_new[i] >= 0:
            continue
        alpha = tuple(mono_list[i])
        sigma_alpha = tuple(reversed(alpha))
        if sigma_alpha == alpha:
            # σ-fixed.
            new_idx = len(mono_list_canon)
            mono_list_canon.append(alpha)
            old_to_new[i] = new_idx
            continue
        j = idx_map.get(sigma_alpha)
        if j is None:
            # σ partner not present — keep this one as its own rep.
            new_idx = len(mono_list_canon)
            mono_list_canon.append(alpha)
            old_to_new[i] = new_idx
            continue
        # Orbit pair (i, j).  Pick the smaller-indexed tuple as representative.
        canon_old = min(i, j)
        new_idx = len(mono_list_canon)
        mono_list_canon.append(tuple(mono_list[canon_old]))
        old_to_new[i] = new_idx
        old_to_new[j] = new_idx

    if (old_to_new < 0).any():
        raise RuntimeError('build_orbit_map: some monomials unmapped')
    return mono_list_canon, old_to_new


def _remap_signed(arr: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """Remap index array preserving the -1 sentinel (missing children)."""
    a = np.asarray(arr, dtype=np.int64)
    out = np.where(a < 0, -1, old_to_new[np.maximum(a, 0)])
    return out


# --------------------------------------------------------------------
# Canonicalize a precompute dict in-place-of-copy
# --------------------------------------------------------------------

def canonicalize_z2(P: Dict[str, Any], *, verbose: bool = True
                     ) -> Dict[str, Any]:
    """Return a new precompute dict with Z/2 equalities substituted out.

    Every pick array (moment_pick, loc_picks, t_pick, ab_eiej_idx, idx_ij,
    consist_*) is remapped from original-y indexing to canonical-ỹ indexing.
    The moment vector dimension ``n_y`` shrinks accordingly.  The MOSEK
    build phase will see a problem WITHOUT Z/2 equality constraints; the
    σ-invariance of ỹ is automatic because two σ-partner coordinates are
    the SAME scalar variable.

    Downstream callers of this dict should NOT add Z/2 equalities again.

    Parameters
    ----------
    P        : output of ``lasserre_scalable._precompute``.
    verbose  : print a one-line summary of the size reduction.

    Returns
    -------
    P_new    : shallow-copied dict with the above fields canonicalized.
                Original arrays in P are not mutated.
    """
    mono_list = P['mono_list']
    n_old = len(mono_list)

    mono_list_canon, old_to_new = build_orbit_map(mono_list)
    n_new = len(mono_list_canon)

    P_new = copy.copy(P)
    P_new['mono_list'] = mono_list_canon
    # IMPORTANT: `idx` must accept ALL original tuples (both σ-partners of
    # every orbit pair) and map them to the same canonical index.  This
    # is required because downstream code (e.g. build_blockdiag_picks)
    # looks up arbitrary sums β_a + β_b in `idx`, and either partner
    # may be the natural representative of that sum.
    idx_canon = {tuple(m): int(old_to_new[i])
                 for i, m in enumerate(mono_list)}
    P_new['idx'] = idx_canon
    P_new['n_y'] = n_new
    P_new['old_to_new'] = old_to_new.copy()
    P_new['z2_pre_elim'] = True

    # ------- Pick arrays -------
    if P.get('moment_pick') is not None:
        P_new['moment_pick'] = _remap_signed(
            np.asarray(P['moment_pick']), old_to_new).tolist()

    if P.get('t_pick') is not None:
        P_new['t_pick'] = _remap_signed(
            np.asarray(P['t_pick']), old_to_new).tolist()

    if P.get('loc_picks'):
        P_new['loc_picks'] = [
            _remap_signed(np.asarray(lp), old_to_new).tolist()
            for lp in P['loc_picks']
        ]

    if P.get('ab_eiej_idx') is not None:
        P_new['ab_eiej_idx'] = _remap_signed(
            np.asarray(P['ab_eiej_idx']), old_to_new)

    if P.get('idx_ij') is not None:
        P_new['idx_ij'] = _remap_signed(
            np.asarray(P['idx_ij']), old_to_new)

    # ------- Consistency indices (used by the build loop) -------
    if P.get('consist_idx') is not None:
        P_new['consist_idx'] = _remap_signed(
            np.asarray(P['consist_idx']), old_to_new)
    if P.get('consist_ei_idx') is not None:
        P_new['consist_ei_idx'] = _remap_signed(
            np.asarray(P['consist_ei_idx']), old_to_new)

    # ------- Scalar windows (F_scipy and f_r/f_c/f_v) -------
    # Remap column indices and aggregate duplicates (two different original
    # moments that became the same canonical index must have their
    # coefficients SUMMED in the window-row dot product).
    if P.get('F_scipy') is not None:
        F_old = P['F_scipy']
        F_coo = F_old.tocoo()
        new_cols = old_to_new[F_coo.col.astype(np.int64)]
        # scipy dedups (row, col) by summing values on CSR/CSC conversion.
        F_new = sp.csr_matrix(
            (F_coo.data, (F_coo.row, new_cols)),
            shape=(F_coo.shape[0], n_new))
        F_new.sum_duplicates()
        F_new.eliminate_zeros()
        P_new['F_scipy'] = F_new

        # Also regenerate the parallel f_r/f_c/f_v triples.
        coo = F_new.tocoo()
        P_new['f_r'] = coo.row.astype(np.int64).tolist()
        P_new['f_c'] = coo.col.astype(np.int64).tolist()
        P_new['f_v'] = coo.data.astype(np.float64).tolist()

    if verbose:
        print(f"  Z/2 pre-elim: n_y {n_old:,} -> {n_new:,} "
              f"({100.0 * n_new / n_old:.1f}% of original)",
              flush=True)

    return P_new

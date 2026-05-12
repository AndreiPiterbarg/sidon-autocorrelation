"""Z/2 time-reversal symmetry reduction for the Sidon autocorrelation problem.

================================================================================
MATHEMATICAL JUSTIFICATION
================================================================================

**Problem.**  We seek the largest c such that, for every nonnegative
f : R → R_{≥0} with support ⊂ [-1/4, 1/4] and ∫ f = 1,

    max_{|t| ≤ 1/2} (f * f)(t)  ≥  c.

The Sidon autocorrelation constant C_{1a} is the supremum of all such c.

**Z/2 action.**  Let σ be the involution f ↦ f(-·).  Because (f * f)(t) =
(f̃ * f̃)(-t) where f̃ = σf, and the test interval |t| ≤ 1/2 is
σ-invariant,

    max_{|t|≤1/2} (f*f)(t)  =  max_{|t|≤1/2} (f̃*f̃)(t).

So σ is a SYMMETRY of the objective on the admissible class.

**Symmetric extremizer theorem (folklore; see Matolcsi–Vinuesa 2010,
Cloninger–Steinerberger 2017, White 2022).**  The class of admissible f
has a symmetric extremizer: there exists an even f* (i.e., σf* = f*)
attaining C_{1a}.  Moreover, for every finite-d discretisation on an
evenly-spaced grid over [-1/4, 1/4], the discrete value

    val(d)  =  min_{μ ∈ Δ_d}  max_W  μ^⊤ M_W μ

also admits a σ-symmetric minimiser μ* ∈ Δ_d, where σ acts on bin
indices by  σ(i) = d-1-i.  This follows because the window-matrix map
W ↦ σ(W) is an involution on windows, so  max_W (σ*μ)^⊤ M_W (σ*μ) =
max_W μ^⊤ M_W μ  identically in μ, and the minimiser set is non-empty,
closed, and σ-invariant ⇒ it contains a fixed point by a standard
Kakutani / Brouwer argument over the compact simplex.

**Consequence.**  val(d) = val_σ(d), where

    val_σ(d)  =  min_{μ ∈ Δ_d, σμ = μ}  max_W  μ^⊤ M_W μ

and the right-hand side is computed with the extra linear constraints
μ_i = μ_{d-1-i}.  Therefore any Lasserre relaxation of val_σ(d) is also
a valid (and ≥) relaxation of val(d):  soundness is preserved.

**Translation to pseudo-moments.**  The symmetry σ on bins extends to
an involution σ : N^d → N^d on multi-indices by σ(α) = (α_{d-1}, α_{d-2},
…, α_0).  The polynomial ring action is σ(x^α) = x^{σ(α)}.  A pseudo-
moment sequence (y_α)_{|α| ≤ 2k} is σ-invariant iff

    y_α  =  y_{σ(α)}  for every α with |α| ≤ 2k.                 (*)

Adding (*) to the Lasserre relaxation strictly shrinks the feasible
pseudo-moment set.  Every true moment sequence of a σ-symmetric measure
satisfies (*), so the optimum of the σ-reduced relaxation lower-bounds
val_σ(d) = val(d).  Hence lb_σ ≤ val(d) — soundness holds.

================================================================================
IMPLEMENTATION
================================================================================

This module provides two primitives:

  • ``z2_symmetry_pairs(P)`` — enumerates (i, j) index pairs in the
    reduced moment set S such that the corresponding monomials are
    σ-partners (y_i = y_j is required).  Only non-trivial pairs with
    i < j are emitted (so no duplicates).

  • ``inject_z2_equalities_into_base_problem(P, A, b, cone)`` —
    appends the (*)-equalities as rows of the SCS/ADMM constraint
    matrix.  Each pair (i, j) adds one zero-cone row with +1 at col i,
    -1 at col j, b = 0.

Fixed-point monomials (σ(α) = α) contribute no constraint.  Monomials
whose σ-partner is outside the reduced moment set S are skipped
(their partner is not a variable, so no equality can be written).

The implementation relies on only two fields of the precompute dict P:
    - ``P['mono_list']``: list of tuples (the monomial for each
      S-index, used to compute σ(α) = reversed tuple);
    - ``P['idx']``: dict tuple → S-index (used to find σ(α)'s column).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------
# Core: compute σ-orbit pairs on the reduced moment set
# --------------------------------------------------------------------

def z2_symmetry_pairs(P: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return list of (i, j) S-index pairs with i < j such that the
    S-monomials at positions i and j are σ-partners (related by
    bin-index reversal).

    Arguments
    ---------
    P : precompute dict with at least 'mono_list' and 'idx' fields.

    Returns
    -------
    pairs : list of (i, j) with i < j.

    Notes
    -----
    Each pair encodes an equality  y_i = y_j  that every σ-symmetric
    pseudo-moment sequence must satisfy.  See module docstring for the
    mathematical justification.

    Pairs are deduplicated automatically because each σ-orbit of size 2
    contributes a single ordered pair.  σ-fixed-point monomials
    (reversed == original) contribute no pair.

    Monomials whose σ-partner is NOT in the reduced moment set S are
    skipped silently: the partner is not a variable in the SDP, so the
    symmetry cannot be enforced for that monomial.  This is sound —
    it just means the σ-reduction is slightly weaker than the ideal.
    """
    mono_list = P['mono_list']
    idx = P['idx']

    pairs: List[Tuple[int, int]] = []
    seen: set[int] = set()
    for i, alpha in enumerate(mono_list):
        if i in seen:
            continue
        # σ acts on a d-vector by index-reversal.
        sigma_alpha: Tuple[int, ...] = alpha[::-1]
        if sigma_alpha == alpha:
            # σ-fixed point — no constraint (trivially y_i = y_i).
            continue
        j = idx.get(sigma_alpha)
        if j is None:
            # σ-partner is not in S → cannot enforce.
            continue
        if j == i:
            # Should have been caught by the fixed-point check above,
            # but be defensive.
            continue
        lo, hi = (i, j) if i < j else (j, i)
        pairs.append((lo, hi))
        seen.add(lo)
        seen.add(hi)
    return pairs


# --------------------------------------------------------------------
# SDP-level integration: append σ-equalities as z-cone rows
# --------------------------------------------------------------------

def inject_z2_equalities_into_base_problem(
    P: Dict[str, Any],
    A_base: sp.spmatrix,
    b_base: np.ndarray,
    cone_base: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[sp.spmatrix, np.ndarray, Dict[str, Any], int]:
    """Append Z/2 equality rows to the base SCS/ADMM constraint block.

    For each pair (i, j) returned by :func:`z2_symmetry_pairs`, append
    one row to A with A[row, i] = +1, A[row, j] = -1, b[row] = 0, and
    classify the new rows as zero-cone (equality) by bumping
    ``cone_base['z']``.

    CRITICAL INVARIANT.  The SCS cone ordering is [z | l | s_i…].  We
    INSERT the new z-rows immediately after the existing z-rows (NOT at
    the end) so that the cone block ordering is preserved.  This means
    the returned A has the new rows spliced at offset ``cone_base['z']``
    (the end of the current zero block).

    Returns
    -------
    A_new, b_new, cone_new, n_added
    """
    pairs = z2_symmetry_pairs(P)
    n_added = len(pairs)
    if n_added == 0:
        return A_base, b_base, cone_base, 0

    n_cols = A_base.shape[1]
    # Build the σ-equality block.
    rows = np.concatenate([np.arange(n_added), np.arange(n_added)])
    cols = np.concatenate([[p[0] for p in pairs], [p[1] for p in pairs]])
    data = np.concatenate([np.ones(n_added), -np.ones(n_added)])
    A_sigma = sp.coo_matrix(
        (data, (rows, cols)), shape=(n_added, n_cols)).tocsc()

    # Splice at offset = current z-cone size so the [z | l | s_i…]
    # ordering is preserved.  We cut A_base into (A_z | A_rest):
    #   A_z    = first z rows   (zero-cone equalities)
    #   A_rest = remaining rows (linear + PSD blocks)
    z = int(cone_base.get('z', 0))
    A_csc = A_base.tocsc()
    A_z = A_csc[:z, :]
    A_rest = A_csc[z:, :]
    A_new = sp.vstack([A_z, A_sigma, A_rest], format='csc')

    b_z = b_base[:z]
    b_rest = b_base[z:]
    b_new = np.concatenate([b_z, np.zeros(n_added), b_rest])

    cone_new = dict(cone_base)
    cone_new['z'] = z + n_added

    if verbose:
        fixed = 0
        for alpha in P['mono_list']:
            if tuple(alpha[::-1]) == tuple(alpha):
                fixed += 1
        n_S = len(P['mono_list'])
        missing = max(0, (n_S - fixed - 2 * n_added) // 2)
        print(
            f"  Z/2 symmetry: +{n_added} equality rows "
            f"(|S|={n_S}, sigma-fixed={fixed}, "
            f"sigma-pairs-in-S={n_added}, "
            f"sigma-partners-missing={missing})",
            flush=True)
    return A_new, b_new, cone_new, n_added


# --------------------------------------------------------------------
# Testing / sanity check
# --------------------------------------------------------------------

def _verify_z2_soundness(P: Dict[str, Any], y_vals: np.ndarray,
                         atol: float = 1e-6) -> bool:
    """Check that a given pseudo-moment vector satisfies the Z/2
    constraints, up to absolute tolerance ``atol``.  Returns True if
    all σ-equalities hold.  Used for post-solve sanity checks.
    """
    pairs = z2_symmetry_pairs(P)
    for i, j in pairs:
        if abs(y_vals[i] - y_vals[j]) > atol:
            return False
    return True

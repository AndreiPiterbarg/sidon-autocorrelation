"""Z/2 block-diagonalization of σ-invariant PSD cones.

Given a σ-invariant moment matrix M(y) (where σ(i) = d-1-i acts on bins,
extended to multi-indices by reversal) and a σ-invariant moment vector y
(i.e. y_α = y_{σ(α)}), M is σ-equivariant:
    σ · M · σ^T = M
and thus block-diagonalizes under the symmetric/antisymmetric orthogonal
basis change:

    e_α                           (α a σ-fixed basis monomial)
    (e_α + e_{σα}) / √2           (α representative of a σ-orbit pair)
    (e_α - e_{σα}) / √2           (orthogonal complement of the above pair)

Under this change of basis, M splits into
    M_sym  of size (F + P) × (F + P)
    M_anti of size  P × P
where F = #σ-fixed basis monomials, P = #orbit pairs, n = F + 2P.

Both blocks are affine in y. If y is σ-invariant (which we enforce via the
Z/2 equalities in lasserre.z2_symmetry), the resulting PSD requirements
    M_sym  ⪰ 0,  M_anti ⪰ 0
are equivalent to M(y) ⪰ 0. This is a LOSSLESS reformulation — same
feasible set, smaller PSD cones, ~4× cheaper Cholesky since n³ → 2·(n/2)³.

Concrete entries (let β_α denote the basis monomial at position α):
    M_sym[f,f']  = y_{β_f + β_f'}                 (both fixed)
    M_sym[f,p]   = √2 · y_{β_f + β_p}             (fixed × orbit-rep)
    M_sym[p,q]   = y_{β_p + β_q} + y_{β_p + σ(β_q)}
    M_anti[p,q]  = y_{β_p + β_q} − y_{β_p + σ(β_q)}

This module provides:
  • orbit_decomposition(basis, d)  — partition basis into fixed / pairs
  • build_sym_anti_maps(basis_from, basis_to, d, mono_idx)
                                   — sparse maps from y to vec(M_sym),
                                     vec(M_anti) for a general bilinear
                                     form M[a,b] = y_{basis_from[a]
                                     + basis_to[b]}.  basis_from = basis_to
                                     for a square σ-invariant cone; they
                                     can differ for localizing cones
                                     (here both are the same side).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


__all__ = [
    'orbit_decomposition',
    'build_blockdiag_picks',
    'sym_anti_sizes',
    'localizing_sigma_reps',
    'window_sigma_reps',
]


# --------------------------------------------------------------------
# Orbit decomposition of a monomial basis under σ (bin-index reversal)
# --------------------------------------------------------------------

def _sigma(alpha: Tuple[int, ...]) -> Tuple[int, ...]:
    """σ acts on a multi-index by reversing the coordinate tuple."""
    return tuple(reversed(alpha))


def orbit_decomposition(basis: Sequence[Tuple[int, ...]]
                         ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Partition basis positions into σ-fixed points and orbit pairs.

    Parameters
    ----------
    basis : sequence of multi-index tuples.  Must be closed under σ:
            for every β in basis, σ(β) is also in basis.

    Returns
    -------
    fixed : list of basis positions f with σ(basis[f]) = basis[f].
    pairs : list of (p, q) with p < q and σ(basis[p]) = basis[q].
            The "representative" of the orbit is always the smaller index.

    Raises
    ------
    ValueError if the basis is not σ-closed.
    """
    idx = {tuple(b): i for i, b in enumerate(basis)}
    fixed: List[int] = []
    pairs: List[Tuple[int, int]] = []
    seen: set[int] = set()
    for i, b in enumerate(basis):
        if i in seen:
            continue
        sb = _sigma(tuple(b))
        if sb == tuple(b):
            fixed.append(i)
            seen.add(i)
            continue
        j = idx.get(sb)
        if j is None:
            raise ValueError(
                f"Basis not closed under σ: σ({b}) = {sb} missing")
        lo, hi = (i, j) if i < j else (j, i)
        pairs.append((lo, hi))
        seen.add(lo)
        seen.add(hi)
    return fixed, pairs


def sym_anti_sizes(basis: Sequence[Tuple[int, ...]]) -> Tuple[int, int]:
    """Return (n_sym, n_anti) = (F+P, P) for the given basis."""
    fixed, pairs = orbit_decomposition(basis)
    return len(fixed) + len(pairs), len(pairs)


# --------------------------------------------------------------------
# Block-diag picks for a σ-invariant bilinear form M[a,b] = y_{β_a+β_b}
# --------------------------------------------------------------------

def build_blockdiag_picks(basis: Sequence[Tuple[int, ...]],
                           mono_idx: Dict[Tuple[int, ...], int],
                           n_y: int,
                           ) -> Dict[str, Any]:
    """Build sparse maps from y ∈ R^{n_y} to vec(M_sym), vec(M_anti).

    For basis {β_i}_i with σ-closure (n = F + 2P), the moment matrix
    M[a, b] = y_{β_a + β_b} splits under σ-orthogonal basis change:

        M_sym[u, v]   = sum over (α_u, α_v) representative pairs of
                        (1/norm_u · norm_v) · (y_{β_{α_u}+β_{α_v}}
                         + y_{β_{α_u}+σ(β_{α_v})}
                         + y_{σ(β_{α_u})+β_{α_v}}
                         + y_{σ(β_{α_u})+σ(β_{α_v})})
        M_anti[p, q]  = y_{β_p+β_q} − y_{β_p+σ(β_q)}
                        (with a 1/2 factor absorbed — see below).

    We compute the matrices T_sym ∈ R^{n_sym² × n_y} and
    T_anti ∈ R^{n_anti² × n_y} such that

        vec(M_sym)  = T_sym  @ y
        vec(M_anti) = T_anti @ y

    Returns dict with keys: T_sym, T_anti (scipy.csr), n_sym, n_anti,
    fixed, pairs.

    Parameters
    ----------
    basis : list of multi-index tuples — the PSD cone basis.
    mono_idx : dict (tuple → int) mapping every multi-index of degree
               ≤ 2*(basis max degree) to a position in y.  Must cover
               every β + β' and every β + σ(β') encountered.
    n_y : dimension of y.
    """
    fixed, pairs = orbit_decomposition(basis)
    F = len(fixed)
    P = len(pairs)
    n_sym = F + P
    n_anti = P

    # Basis labels for convenience:
    basis_tup = [tuple(b) for b in basis]

    # Representative (first index) for each symmetric block row.
    #   sym_repr[0..F-1]     = fixed basis position
    #   sym_repr[F..F+P-1]   = orbit-pair representative (smaller index)
    sym_repr: List[int] = list(fixed) + [p for (p, _) in pairs]
    # For each symmetric row u, a list of basis-position contributions
    # (k_u, coef_u) such that the row of U_sym is sum_k coef_u · e_{k_u}.
    # Convention (orthonormal columns of U_sym):
    #   fixed  f :       1.0 · e_f                         (norm 1)
    #   pair   p<q : (1/√2)·e_p + (1/√2)·e_q              (norm 1)
    sqrt_half = 1.0 / np.sqrt(2.0)
    sym_cols: List[List[Tuple[int, float]]] = []
    for f in fixed:
        sym_cols.append([(f, 1.0)])
    for (p, q) in pairs:
        sym_cols.append([(p, sqrt_half), (q, sqrt_half)])

    # Antisymmetric columns of U_anti: (1/√2)·e_p − (1/√2)·e_q for each pair.
    anti_cols: List[List[Tuple[int, float]]] = []
    for (p, q) in pairs:
        anti_cols.append([(p, sqrt_half), (q, -sqrt_half)])

    # Helper: look up y-index for a given sum of two basis monomials.
    def _y_idx(beta_a: Tuple[int, ...], beta_b: Tuple[int, ...]) -> int:
        s = tuple(x + y for x, y in zip(beta_a, beta_b))
        j = mono_idx.get(s)
        if j is None:
            raise KeyError(f"mono_idx missing sum {s}")
        return int(j)

    # Build T_sym and T_anti as COO.
    sym_rows: List[int] = []
    sym_cols_idx: List[int] = []
    sym_vals: List[float] = []

    # M_sym[u, v] = sum_{(k1, c1) in sym_cols[u]} sum_{(k2, c2) in sym_cols[v]}
    #                   c1 · c2 · y_{β_{k1} + β_{k2}}
    for u in range(n_sym):
        for v in range(n_sym):
            vec_row = u * n_sym + v
            acc: Dict[int, float] = {}
            for (k1, c1) in sym_cols[u]:
                b1 = basis_tup[k1]
                for (k2, c2) in sym_cols[v]:
                    b2 = basis_tup[k2]
                    yj = _y_idx(b1, b2)
                    acc[yj] = acc.get(yj, 0.0) + c1 * c2
            for yj, cval in acc.items():
                if cval == 0.0:
                    continue
                sym_rows.append(vec_row)
                sym_cols_idx.append(yj)
                sym_vals.append(cval)

    anti_rows: List[int] = []
    anti_cols_idx: List[int] = []
    anti_vals: List[float] = []
    for u in range(n_anti):
        for v in range(n_anti):
            vec_row = u * n_anti + v
            acc: Dict[int, float] = {}
            for (k1, c1) in anti_cols[u]:
                b1 = basis_tup[k1]
                for (k2, c2) in anti_cols[v]:
                    b2 = basis_tup[k2]
                    yj = _y_idx(b1, b2)
                    acc[yj] = acc.get(yj, 0.0) + c1 * c2
            for yj, cval in acc.items():
                if cval == 0.0:
                    continue
                anti_rows.append(vec_row)
                anti_cols_idx.append(yj)
                anti_vals.append(cval)

    T_sym = sp.csr_matrix(
        (sym_vals, (sym_rows, sym_cols_idx)),
        shape=(n_sym * n_sym, n_y))
    T_anti = sp.csr_matrix(
        (anti_vals, (anti_rows, anti_cols_idx)),
        shape=(n_anti * n_anti, n_y))

    return {
        'T_sym': T_sym,
        'T_anti': T_anti,
        'n_sym': n_sym,
        'n_anti': n_anti,
        'fixed': fixed,
        'pairs': pairs,
    }


# --------------------------------------------------------------------
# σ-pair reduction for localizing and window PSD cones
# --------------------------------------------------------------------
#
# Localizing cone M_2(μ_i · y) has entries M[a, b] = y_{loc[a] + loc[b] + e_i}.
# Under σ, this maps to M_2(μ_{d-1-i} · y)[a, b] = y_{loc[a] + loc[b] + e_{d-1-i}}.
# Using σ-invariance of y (y_α = y_{σ(α)}):
#     y_{α + e_{d-1-i}} = y_{σ(α) + e_i},
# so M_2(μ_{d-1-i} · y) = Π · M_2(μ_i · y) · Π^T, where Π is the permutation
# that σ-reverses the localizing basis.  Permutation similarity preserves
# PSD-ness, therefore the two cones have equivalent PSD constraints.
# Keeping only canonical representatives is LOSSLESS under the σ-equalities.
#
# Window cones t·M_2(y) − Q_W(y) ⪰ 0 where W = (ell, s) and
#     Q_W(y) has basis-entries driven by the indicator mask
#     (sums ≥ s) & (sums ≤ s + ell − 2) on bin-index sums.
# Under σ (i.e. swapping bin indices i ↔ d−1−i), sums[i,j] → 2(d−1) − sums[i,j],
# which shifts the mask to (2d−ell−s ≤ sums ≤ 2d−2−s).  So σ(W) = (ell, 2d−ell−s).
# The same PSD-similarity argument applies: Q_{σ(W)} = Π Q_W Π^T under
# σ-invariant y, so M_W ⪰ 0 ⟺ M_{σ(W)} ⪰ 0.  Keeping one of each
# {W, σ(W)} pair is LOSSLESS.


def localizing_sigma_reps(d: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return canonical σ-representatives and pairs for localizing indices.

    σ acts on {0, 1, ..., d-1} by i ↦ d-1-i.  Returns
        fixed : list of i with σ(i) = i (only possible if d is odd).
        pairs : list of (i, j) with i < j and σ(i) = j.  Exactly one
                representative per orbit, chosen as the smaller index.
    """
    fixed: List[int] = []
    pairs: List[Tuple[int, int]] = []
    seen: set[int] = set()
    for i in range(d):
        if i in seen:
            continue
        si = d - 1 - i
        if si == i:
            fixed.append(i)
            seen.add(i)
            continue
        lo, hi = (i, si) if i < si else (si, i)
        pairs.append((lo, hi))
        seen.add(lo)
        seen.add(hi)
    return fixed, pairs


def window_sigma_reps(d: int, windows: Sequence[Tuple[int, int]]
                       ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Partition window indices into σ-fixed ones and pairs.

    σ acts on (ell, s) by (ell, s) ↦ (ell, 2d − ell − s).

    Arguments
    ---------
    d       : bin count.
    windows : the list returned by build_window_matrices(d) — so that
              indices into `windows` are the canonical positions used
              elsewhere in the solver.

    Returns
    -------
    fixed : list of window POSITIONS (indices into `windows`) that satisfy
            σ((ell, s)) = (ell, s).
    pairs : list of (i, j) with i < j and σ(windows[i]) = windows[j].
            i is the canonical representative of the orbit.
    """
    idx = {tuple(w): i for i, w in enumerate(windows)}
    fixed: List[int] = []
    pairs: List[Tuple[int, int]] = []
    seen: set[int] = set()
    for i, (ell, s) in enumerate(windows):
        if i in seen:
            continue
        s_sigma = 2 * d - ell - s
        sw = (ell, s_sigma)
        if sw == (ell, s):
            fixed.append(i)
            seen.add(i)
            continue
        j = idx.get(sw)
        if j is None:
            # σ-partner outside the enumerated window list — keep i as
            # representative but flag by placing it alone (treat as fixed).
            fixed.append(i)
            seen.add(i)
            continue
        lo, hi = (i, j) if i < j else (j, i)
        pairs.append((lo, hi))
        seen.add(lo)
        seen.add(hi)
    return fixed, pairs


# --------------------------------------------------------------------
# Self-test — verifies T_sym, T_anti reproduce M via explicit U matrices
# --------------------------------------------------------------------

def _self_test(d: int = 4, order: int = 2) -> None:
    """Sanity check against a reference explicit change-of-basis."""
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
    from lasserre_fusion import enum_monomials  # type: ignore
    from lasserre_scalable import _precompute as _pc  # type: ignore

    P = _pc(d, order, verbose=False)
    basis = P['basis']
    mono_list = P['mono_list']
    idx = P['idx']
    n_y = P['n_y']
    n = len(basis)

    bd = build_blockdiag_picks(basis, idx, n_y)
    n_sym = bd['n_sym']
    n_anti = bd['n_anti']

    # Reference: build the full U and check U^T M U is block diag.
    fixed, pairs = orbit_decomposition(basis)
    sqrt_half = 1.0 / np.sqrt(2.0)
    U_sym = np.zeros((n, n_sym))
    for col, f in enumerate(fixed):
        U_sym[f, col] = 1.0
    for col, (p, q) in enumerate(pairs):
        U_sym[p, len(fixed) + col] = sqrt_half
        U_sym[q, len(fixed) + col] = sqrt_half
    U_anti = np.zeros((n, n_anti))
    for col, (p, q) in enumerate(pairs):
        U_anti[p, col] = sqrt_half
        U_anti[q, col] = -sqrt_half

    # Random σ-invariant y.
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n_y)
    for i, mono in enumerate(mono_list):
        j = idx.get(tuple(reversed(tuple(mono))))
        if j is not None and j < i:
            y[i] = y[j]  # enforce σ-equality

    M = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            beta = tuple(bi + bj for bi, bj in zip(basis[a], basis[b]))
            jidx = idx[beta]
            M[a, b] = y[jidx]

    M_sym_ref = U_sym.T @ M @ U_sym
    M_anti_ref = U_anti.T @ M @ U_anti

    M_sym_from_T = (bd['T_sym'] @ y).reshape(n_sym, n_sym)
    M_anti_from_T = (bd['T_anti'] @ y).reshape(n_anti, n_anti)

    err_sym = np.max(np.abs(M_sym_from_T - M_sym_ref))
    err_anti = np.max(np.abs(M_anti_from_T - M_anti_ref))
    print(f"d={d} order={order}: n={n} F={len(fixed)} P={len(pairs)} "
          f"n_sym={n_sym} n_anti={n_anti} "
          f"err_sym={err_sym:.2e} err_anti={err_anti:.2e}")
    assert err_sym < 1e-10, f"sym block mismatch: {err_sym}"
    assert err_anti < 1e-10, f"anti block mismatch: {err_anti}"


if __name__ == '__main__':
    for d in (4, 6, 8):
        for k in (2, 3):
            _self_test(d, k)
    print("OK — block-diag decomposition verified.")

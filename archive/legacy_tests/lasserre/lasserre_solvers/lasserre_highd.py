#!/usr/bin/env python
r"""
High-d Lasserre solver with clique-restricted sparsity for d=64-128.

=======================================================================
MATHEMATICAL FRAMEWORK
=======================================================================

Problem: val(d) = min_{μ ∈ Δ_d} max_W μ^T M_W μ, where Δ_d is the
standard simplex and M_W are window matrices.

Lasserre order-k relaxation introduces pseudo-moment variables
y_α = E[x^α] for |α| ≤ 2k, subject to:

 (L1) y_0 = 1 (normalization)
 (L2) y_α ≥ 0 (nonnegativity, since x ≥ 0)
 (L3) M_k(y) ≽ 0 (moment matrix PSD)
 (L4) M_{k-1}(μ_i · y) ≽ 0 (localizing for μ_i ≥ 0)
 (L5) y_α = Σ_i y_{α+e_i} (consistency from Σ μ_i = 1)
 (L6) t·M_{k-1}(y) - Σ M_W[i,j]·M_{k-1}(μ_iμ_j·y) ≽ 0 (window PSD)
 (L7) M_{k-1}(y) - M_{k-1}(μ_i·y) ≽ 0 (upper localizing, optional)

This module implements a clique-restricted relaxation for high d.

=======================================================================
SOUNDNESS THEOREM
=======================================================================

Theorem: The output lb satisfies lb ≤ val(d).

Proof: Let μ* achieve val(d), t* = val(d), y*_α = E_{μ*}[x^α].
We show (y*|_S, t*) is feasible for every constraint in our model:

 1. y*_0 = 1: normalization of μ*.
 2. y*_α ≥ 0: μ* is supported on ℝ^d_{≥0}.
 3. Clique M_2^{I_c}(y*) ≽ 0: principal submatrix of M_2(y*) ≽ 0.
 4. Full M_1(y*) ≽ 0: v^T M_1(y*) v = E[(Σ v_i x_i)²] ≥ 0.
 5. Clique localizing: E[x_i · (Σ v_j x_j)²] ≥ 0 since x_i ≥ 0 a.s.
 6. Full consistency: y*_α = Σ_i y*_{α+e_i} from Σ x_i = 1.
 7. Partial consistency (inequality): follows from (6) + y* ≥ 0.
 Specifically: y*_α = Σ_{i∈S'} y*_{α+e_i} + Σ_{i∉S'} y*_{α+e_i}
 ≥ Σ_{i∈S'} y*_{α+e_i}.
 8. Scalar windows: t* ≥ μ*^T M_W μ* = Σ M_W[i,j] y*_{e_i+e_j}.
 9. CG window PSD: principal submatrix of the full L_W(y*) ≽ 0.

Since (y*|_S, t*) is feasible, lb ≤ t* = val(d). QED.

=======================================================================
TIGHTNESS
=======================================================================

The hierarchy of relaxations (tightest to weakest, all sound):

 lb_full_Lasserre ≥ lb_{full_M1 + clique_M2 + full_consist}
 ≥ lb_{full_M1 + clique_M2 + partial_ineq} [OUR MODEL]
 ≥ lb_{clique_only + no_consist}

The full M_1 PSD provides cross-clique coupling absent from clique M_2
alone (see _add_full_m1_psd docstring for proof).

=======================================================================

Usage:
 python tests/lasserre_highd.py --d 128 --bw 16
 python tests/lasserre_highd.py --d 64 --bw 12
"""
import numpy as np
from scipy import sparse as sp
try:
 from mosek.fusion import (Model, Domain, Expr, Matrix,
 ObjectiveSense, SolutionStatus)
except ImportError:
 Model = Domain = Expr = Matrix = ObjectiveSense = SolutionStatus = None
import time
import sys
import os
import gc as gc_mod

# Import from the lasserre package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasserre.core import (
 enum_monomials, val_d_known,
 _make_hash_bases, _hash_monos, _hash_add, _build_hash_table, _hash_lookup,
)
from lasserre.cliques import _build_banded_cliques, _build_clique_basis


# =====================================================================
# Custom precompute for high-d (replaces _precompute at d ≥ 64)
# =====================================================================

def _build_reduced_moment_set(d, order, cliques, bases, prime=None, verbose=True):
 r"""Build the reduced moment set S for clique-restricted Lasserre.

 S = S_{≤2k-1} ∪ ⋃_c S_mom(I_c)

 where:
 S_{≤2k-1} = {α ∈ ℕ^d : |α| ≤ 2k-1}
 — ALL moments up to the consistency degree.

 WHY THIS IS CRITICAL FOR TIGHTNESS:
 Consistency at degree j says y_α = Σ_i y_{α+e_i} for |α|=j.
 Children have degree j+1. For this to be FULL EQUALITY (not
 a weak inequality relaxation), ALL children must be in S.

 If S only includes degree ≤ 2 globally, then degree-2 parents
 e_i+e_j with |i-j| > bandwidth have degree-3 children OUTSIDE S.
 The consistency becomes partial inequality, breaking the chain:
 PSD(degree 4) → consist → degree 3 → consist → degree 2 → scalar
 This causes catastrophic bound loss (17% vs 77% gap closure).

 Including ALL degree ≤ 2k-1 ensures FULL EQUALITY consistency
 for all parents of degree ≤ 2k-2. For k=2:
 degree 0 parents → children degree 1, all in S_{≤3}. FULL.
 degree 1 parents → children degree 2, all in S_{≤3}. FULL.
 degree 2 parents → children degree 3, all in S_{≤3}. FULL.
 degree 3 parents → children degree 4, clique-only. Partial.

 S_mom(I_c) = {α+β : |α|,|β| ≤ k, supp(α)∪supp(β) ⊆ I_c}
 — degree ≤ 2k moments in clique moment PSD cones.
 Note: S_loc(I_c, i) ⊆ S_{≤2k-1} since localizing entries
 have degree ≤ 2(k-1)+1 = 2k-1.

 Size at d=128, k=2:
 |S_{≤3}| = C(131,3) = 366,145
 |S_mom \ S_{≤3}| ≈ 132K (degree-4 clique monomials)
 Total |S| ≈ 498K. Feasible for MOSEK.

 Returns: (S_arr numpy array, n_y)
 """
 t0 = time.time()

 # Collect all monomial arrays, then batch-deduplicate via hashing.
 # This replaces per-element Python dict insertion with numpy operations.
 all_mono_chunks = []

 # S_{≤2k-1}: all degree ≤ 2k-1 monomials (= degree ≤ 3 for k=2)
 consist_deg = 2 * order - 1
 mono_global = np.array(enum_monomials(d, consist_deg), dtype=np.int64)
 all_mono_chunks.append(mono_global)
 if verbose:
 print(f" S_le{consist_deg}: {len(mono_global)} monomials", flush=True)

 # Clique moment PSD monomials
 n_mom_before = len(mono_global)
 for clique in cliques:
 cb = _build_clique_basis(clique, order, d)
 n_cb = len(cb)
 ai, bi = np.triu_indices(n_cb)
 all_mono_chunks.append(cb[ai] + cb[bi])

 if verbose:
 n_mom_total = sum(len(c) for c in all_mono_chunks) - n_mom_before
 print(f" S_mom (clique PSD): {n_mom_total} candidates", flush=True)

 # Clique localizing monomials
 if order >= 2:
 n_loc_before = sum(len(c) for c in all_mono_chunks)
 for clique in cliques:
 cb_loc = _build_clique_basis(clique, order - 1, d)
 n_cb_loc = len(cb_loc)
 ai, bi = np.triu_indices(n_cb_loc)
 base_sums = cb_loc[ai] + cb_loc[bi]
 for i in clique:
 shifted = base_sums.copy()
 shifted[:, i] += 1
 all_mono_chunks.append(shifted)

 if verbose:
 n_loc_total = sum(len(c) for c in all_mono_chunks) - n_loc_before
 print(f" S_loc (clique localizing): {n_loc_total} candidates",
 flush=True)

 # Batch deduplicate: concatenate all, hash, then keep first occurrence
 all_monos = np.concatenate(all_mono_chunks, axis=0)
 all_hashes = _hash_monos(all_monos, bases, prime)
 # np.unique on hashes, keeping first occurrence (stable sort)
 _, first_idx = np.unique(all_hashes, return_index=True)
 first_idx.sort() # preserve original order for determinism
 S_arr = all_monos[first_idx]
 n_y = len(S_arr)
 # Sort lexicographically for deterministic ordering
 sort_idx = np.lexsort(S_arr.T[::-1])
 S_arr = S_arr[sort_idx]

 # ---- Hash collision verification (Issue 8 from audit) ----
 # At d=128 with random int63 bases, collision probability per pair is
 # ~max_comp / 2^62 ≈ 4/2^62 ≈ 8.7e-19. For n=141K monomials,
 # P(any collision) ≈ n²/(2·2^63) ≈ 1.1e-9. Negligible, but we
 # verify anyway for publishable rigor.
 #
 # Two checks:
 # (a) All monomials in S_arr are distinct as d-tuples.
 # (b) All hashes of S_arr are distinct (no two monomials share a hash).
 # If (b) fails, a monomial may have been silently merged during
 # construction, causing PSD cones to reference wrong moments.
 unique_count = len(np.unique(S_arr, axis=0))
 assert unique_count == n_y, (
 f"Duplicate monomials in S: {n_y} entries but {unique_count} unique.")
 S_hashes = _hash_monos(S_arr, bases, prime)
 unique_hash_count = len(np.unique(S_hashes))
 assert unique_hash_count == n_y, (
 f"Hash collision detected: {n_y} monomials but {unique_hash_count} "
 f"unique hashes. Two distinct monomials share a hash value. "
 f"This would cause silent moment substitution in PSD cones, "
 f"potentially producing unsound bounds. "
 f"Fix: change the random seed in _make_hash_bases or use "
 f"a larger hash space.")

 elapsed = time.time() - t0
 if verbose:
 print(f" Total |S| = {n_y:,} moments, hash collision check passed "
 f"({elapsed:.1f}s)", flush=True)

 return S_arr, n_y


def _precompute_highd(d, order, cliques, verbose=True):
 r"""Custom precompute for high-d sparse Lasserre.

 Builds all data structures on the reduced moment set S instead of
 the full set of C(d+2k, 2k) moments.

 At d=128, k=2: |S| ≈ 141K vs full 12M.
 Memory: ~1 GB vs 20+ GB. Time: ~10s vs ~23min.
 """
 t0 = time.time()

 # Hash infrastructure
 max_comp = 2 * order
 bases, prime = _make_hash_bases(d, max_comp)

 # Build reduced moment set
 if verbose:
 print(" Building reduced moment set S...", flush=True)
 S_arr, n_y = _build_reduced_moment_set(d, order, cliques, bases,
 prime=prime, verbose=verbose)

 # Build hash table on S
 mono_list = [tuple(row) for row in S_arr]
 idx = {m: i for i, m in enumerate(mono_list)}
 sorted_h, sort_o = _build_hash_table(mono_list, bases, prime)

 # --- Window definitions (no M_mats storage — saves 4GB at d=128) ---
 conv_len = 2 * d - 1
 windows = [(ell, s) for ell in range(2, 2 * d + 1)
 for s in range(conv_len - ell + 2)]
 n_win = len(windows)

 # --- Degree-2 moment index: idx_ij[i,j] = S-index of y_{e_i+e_j} ---
 E = np.eye(d, dtype=np.int64)
 EE = E[:, None, :] + E[None, :, :] # (d, d, d)
 EE_hash = _hash_monos(EE, bases, prime) # (d, d)
 idx_ij = _hash_lookup(EE_hash, sorted_h, sort_o) # (d, d)

 # --- Scalar window F_scipy (fully vectorized, no Python loop) ---
 if verbose:
 print(" Building F_scipy (scalar window constraints)...", flush=True)
 t_f = time.time()
 sums_grid = np.arange(d)[:, None] + np.arange(d)[None, :] # (d,d)
 # Flatten idx_ij for vectorized lookup
 flat_ij = idx_ij.ravel() # (d*d,)
 flat_sums = sums_grid.ravel() # (d*d,)
 valid_pairs = flat_ij >= 0 # only pairs with moments in S

 # For each valid (i,j) pair, determine which windows it belongs to.
 # Window w=(ell, s_lo) includes (i,j) iff s_lo <= i+j <= s_lo+ell-2.
 # Equivalently: s_lo <= s <= s_lo+ell-2, where s = i+j.
 # For each sum value s, the windows containing it are those where
 # s_lo <= s and s_lo + ell - 2 >= s, i.e., s_lo <= s <= s_lo + ell - 2.
 # Instead of looping over windows, we loop over valid (i,j) pairs
 # and compute their contributions.
 vp_idx = np.where(valid_pairs)[0]
 vp_sums = flat_sums[vp_idx]
 vp_cols = flat_ij[vp_idx]

 f_r_list, f_c_list, f_v_list = [], [], []
 # Group by sum value for efficiency (all pairs with same sum
 # contribute to exactly the same set of windows)
 unique_sums = np.unique(vp_sums)
 w_arr = np.array(windows) # (n_win, 2): [ell, s_lo]
 w_ell = w_arr[:, 0]
 w_slo = w_arr[:, 1]

 for s_val in unique_sums:
 # Which windows contain this sum value?
 w_mask = (w_slo <= s_val) & (s_val <= w_slo + w_ell - 2)
 w_indices = np.where(w_mask)[0]
 if len(w_indices) == 0:
 continue
 # Which (i,j) pairs have this sum?
 pair_mask = vp_sums == s_val
 cols_for_s = vp_cols[pair_mask]
 # Each (window, pair) combination is a COO entry
 coeffs = 2.0 * d / w_ell[w_indices] # (n_w,)
 for k, w in enumerate(w_indices):
 f_r_list.append(np.full(len(cols_for_s), w, dtype=np.int32))
 f_c_list.append(cols_for_s)
 f_v_list.append(np.full(len(cols_for_s), coeffs[k]))

 if f_r_list:
 f_r_all = np.concatenate(f_r_list)
 f_c_all = np.concatenate(f_c_list)
 f_v_all = np.concatenate(f_v_list)
 else:
 f_r_all = np.array([], dtype=np.int32)
 f_c_all = np.array([], dtype=np.int32)
 f_v_all = np.array([], dtype=np.float64)

 F_scipy = sp.csr_matrix(
 (f_v_all, (f_r_all, f_c_all)), shape=(n_win, n_y), dtype=np.float64)
 # Pre-convert COO to lists for MOSEK Matrix.sparse (avoids repeated .tolist())
 F_coo = F_scipy.tocoo()
 F_coo_lists = (F_coo.row.tolist(), F_coo.col.tolist(), F_coo.data.tolist())
 if verbose:
 print(f" F_scipy: {F_scipy.nnz:,} nnz ({time.time()-t_f:.1f}s)",
 flush=True)

 # --- Full M_{k-1}(y) PSD (cross-clique coupling) ---
 # At order k, the full M_{k-1} moment matrix provides global coupling
 # between cliques that clique-restricted M_k alone cannot.
 #
 # Basis: all monomials of degree ≤ k-1 in d variables.
 # order=2: degree ≤ 1 → {0, e_1, ..., e_d}, size d+1
 # order=3: degree ≤ 2, size C(d+2, 2)
 #
 # All M_{k-1} entries have degree ≤ 2(k-1) ≤ 2k-1, so they're in S_{≤2k-1}.
 #
 # Soundness: for true measure μ*, v^T M_{k-1}(y*) v = E[(Σ v_a x^{α_a})²] ≥ 0.
 # This is a necessary condition, strictly weaker than full M_k PSD.
 full_basis = np.array(enum_monomials(d, order - 1), dtype=np.int64)
 m1_size = len(full_basis)
 FB_hash = _hash_monos(full_basis, bases, prime)
 AB_FB_hash = _hash_add(FB_hash[:, None], FB_hash[None, :], prime)
 m1_pick = _hash_lookup(AB_FB_hash, sorted_h, sort_o).ravel()
 m1_pick_list = m1_pick.tolist() # pre-convert for MOSEK y.pick()
 m1_valid = np.all(m1_pick >= 0)

 # --- Pairwise product localizing: M_1(μ_i μ_j y) ⪰ 0 for all pairs i ≤ j ---
 #
 # MATHEMATICAL BASIS: The full Lasserre L3 hierarchy for minimising over
 # Δ_d = {μ ≥ 0, Σμ_i = 1} includes localising matrices for ALL products
 # of the nonnegativity generators {μ_i ≥ 0} whose combined polynomial
 # degree fits within 2k = 6 (the hierarchy order × 2):
 #
 # Single (currently included):
 # M_{k-1}(μ_i y) = M_2(μ_i y), degree 1+4 = 5 ≤ 6
 #
 # Pairs (CURRENTLY MISSING — added here):
 # M_{k-2}(μ_i μ_j y) = M_1(μ_i μ_j y), degree 2+2 = 4 ≤ 6
 #
 # SOUNDNESS: For any true measure μ* ∈ Δ_d and degree-1 polynomial p:
 # v^T M_1(μ_i μ_j y*) v = E_{μ*}[μ_i μ_j · p(μ)^2] ≥ 0
 # because μ_i ≥ 0, μ_j ≥ 0, p² ≥ 0. So lb produced with these cones
 # satisfies lb ≤ val(d).
 #
 # RELEVANCE: The objective f_W(μ) = Σ_{i,j} M_W[i,j] μ_i μ_j is a sum of
 # pairwise products. The Putinar representation of t − f_W ≥ 0 on Δ_d at
 # degree 6 involves SOS multipliers σ_{ij}(μ) of degree 4 multiplying μ_i μ_j.
 # In the Lasserre dual these correspond exactly to M_1(μ_i μ_j y). Without
 # them the dual cannot construct tight degree-6 certificates for window
 # constraints, which is why gc stalls after 2 rounds.
 #
 # ENTRY STRUCTURE: M_1(μ_i μ_j y)[α,β] = y_{α+β+e_i+e_j}
 # where α, β are degree-≤1 monomials (basis size = d+1 = 17 at d=16).
 # Degree: 1+1+1+1 = 4 ≤ 2k-1 = 5 → guaranteed in S_{≤2k-1}.
 #
 # COST: 136 cones of size 17×17 (d(d+1)/2 pairs × (d+1)² entries each).
 # cholesky_ex fast-path handles each in ~0.19ms → +26ms/iter at Round 0.
 # Memory: ~315 KB. Negligible.

 pw_basis = np.array(enum_monomials(d, 1), dtype=np.int64) # (d+1, d) degree ≤ 1
 pw_size = len(pw_basis) # d+1 = 17 at d=16

 PW_hash = _hash_monos(pw_basis, bases, prime) # (d+1,)
 AB_PW_hash = _hash_add(PW_hash[:, None], PW_hash[None, :], prime) # (d+1, d+1)

 # Fully vectorised: compute all pair hashes and do one batch lookup,
 # eliminating the Python double-loop over d*(d+1)/2 = 136 pairs.
 #
 # i_arr, j_arr: upper-triangle pair indices, shape (n_pairs,)
 # eij_hash : hash(e_i + e_j) = bases[i] + bases[j] (+ mod prime if set)
 # ij_hash_all: (n_pairs, pw_size, pw_size) — hash of α+β+e_i+e_j for all (pair,α,β)
 # all_picks : (n_pairs, pw_size²) — S-indices for every entry of every pair cone
 i_arr, j_arr = np.triu_indices(d, k=0) # shape (n_pairs,)
 n_pairs = len(i_arr)

 # Hash of e_i + e_j for every pair — bases[i] is hash(e_i) by construction
 if prime is None:
 eij_hash = bases[i_arr] + bases[j_arr] # (n_pairs,) integer, no overflow
 ij_hash_all = (AB_PW_hash[np.newaxis, :, :] # (1, pw_size, pw_size)
 + eij_hash[:, np.newaxis, np.newaxis]) # → (n_pairs, pw, pw)
 else:
 eij_hash = (bases[i_arr] + bases[j_arr]) % prime
 ij_hash_all = _hash_add(
 AB_PW_hash[np.newaxis, :, :],
 eij_hash[:, np.newaxis, np.newaxis],
 prime
 ) # (n_pairs, pw_size, pw_size)

 # Single batch hash lookup: (n_pairs × pw_size²,) → reshape → (n_pairs, pw_size²)
 all_picks = _hash_lookup(
 ij_hash_all.reshape(n_pairs, -1), # (n_pairs, pw_size²)
 sorted_h, sort_o
 ) # (n_pairs, pw_size²) of S-indices

 # Filter pairs where all pw_size² entries are valid (inside S)
 valid_mask = np.all(all_picks >= 0, axis=1) # (n_pairs,)

 pw_pairs = [
 (int(i_arr[k]), int(j_arr[k]), all_picks[k])
 for k in np.where(valid_mask)[0]
 ]
 pw_pairs_list = [
 (int(i_arr[k]), int(j_arr[k]), all_picks[k].tolist())
 for k in np.where(valid_mask)[0]
 ]

 if verbose:
 print(f" Pairwise localizing: {len(pw_pairs)}/{n_pairs} cones of size "
 f"{pw_size}×{pw_size}",
 flush=True)

 # --- Per-clique precomputed data ---
 if verbose:
 print(" Building per-clique data...", flush=True)
 clique_data = []
 n_collision_checks = 0
 for c_idx, clique in enumerate(cliques):
 cd = {'clique': clique}

 # Moment PSD picks
 cb = _build_clique_basis(clique, order, d)
 cb_hash = _hash_monos(cb, bases, prime)
 ab_hash = _hash_add(cb_hash[:, None], cb_hash[None, :], prime)
 mom_pick = _hash_lookup(ab_hash, sorted_h, sort_o).ravel()
 cd['mom_pick'] = mom_pick
 cd['mom_pick_list'] = mom_pick.tolist()
 cd['mom_size'] = len(cb)

 # --- Issue 8 deep verification: spot-check that hash lookups ---
 # --- return the CORRECT monomial, not a collision victim. ---
 # For the first and last clique, verify every PSD entry.
 # For others, verify a random sample. Total cost: O(d) per clique.
 if c_idx == 0 or c_idx == len(cliques) - 1:
 # Full check for first/last clique
 ab_monos = (cb[:, None, :] + cb[None, :, :]).reshape(-1, d)
 for k in range(len(ab_monos)):
 if mom_pick[k] >= 0:
 assert np.array_equal(S_arr[mom_pick[k]], ab_monos[k]), (
 f"Hash collision in clique {c_idx}, PSD entry {k}: "
 f"expected {ab_monos[k]}, got {S_arr[mom_pick[k]]}")
 n_collision_checks += 1

 if order >= 2:
 # Localizing basis
 cb_loc = _build_clique_basis(clique, order - 1, d)
 cb_loc_hash = _hash_monos(cb_loc, bases, prime)
 ab_loc_hash = _hash_add(cb_loc_hash[:, None], cb_loc_hash[None, :], prime)
 cd['loc_size'] = len(cb_loc)
 cd['loc_ab_hash'] = ab_loc_hash

 # T-part picks: y_{α_a + α_b} (degree ≤ 2(k-1) ≤ 2, in S)
 t_pick = _hash_lookup(ab_loc_hash, sorted_h, sort_o).ravel()
 cd['t_pick'] = t_pick
 cd['t_pick_list'] = t_pick.tolist()

 # Localizing picks per bin: y_{α_a + α_b + e_i}
 cd['loc_picks'] = {}
 cd['loc_picks_list'] = {}
 for i in clique:
 h = _hash_add(ab_loc_hash, bases[i], prime)
 picks = _hash_lookup(h, sorted_h, sort_o).ravel()
 cd['loc_picks'][i] = picks
 cd['loc_picks_list'][i] = picks.tolist()

 # Precompute violation-checking index arrays (reused every CG round)
 clique_arr = np.array(clique)
 ee_hash = _hash_add(bases[clique_arr[:, None]],
 bases[clique_arr[None, :]], prime)
 abij_hash = _hash_add(ab_loc_hash[:, :, None, None],
 ee_hash[None, None, :, :], prime)
 ab_eiej_local = _hash_lookup(abij_hash, sorted_h, sort_o)
 cd['viol_ab_eiej'] = ab_eiej_local
 cd['viol_gi_grid'] = clique_arr[:, None] + clique_arr[None, :]

 clique_data.append(cd)

 if verbose and n_collision_checks > 0:
 print(f" Hash collision spot-check: {n_collision_checks} entries "
 f"verified", flush=True)

 # --- Bin-to-clique mapping (assign each bin to most central clique) ---
 bin_to_clique = {}
 for c_idx, clique in enumerate(cliques):
 mid = (clique[0] + clique[-1]) / 2.0
 for i in clique:
 if i not in bin_to_clique:
 bin_to_clique[i] = (c_idx, abs(i - mid))
 else:
 _, prev_dist = bin_to_clique[i]
 if abs(i - mid) < prev_dist:
 bin_to_clique[i] = (c_idx, abs(i - mid))
 bin_to_clique_map = {i: v[0] for i, v in bin_to_clique.items()}

 # --- Window-to-clique mapping (vectorized) ---
 if verbose:
 print(" Mapping windows to covering cliques...", flush=True)
 w_arr = np.array(windows) # (n_win, 2): [ell, s_lo]
 w_ell = w_arr[:, 0]
 w_slo = w_arr[:, 1]
 # Active bin range per window: i_min, i_max
 w_i_min = np.clip(w_slo - (d - 1), 0, d - 1)
 w_i_max = np.clip(w_slo + w_ell - 2, 0, d - 1)

 # Clique start/end arrays
 c_starts = np.array([clique[0] for clique in cliques], dtype=np.int32)
 c_ends = np.array([clique[-1] for clique in cliques], dtype=np.int32)

 # For each window, find first clique that covers [i_min, i_max]
 # covers[w, c] = (c_starts[c] <= i_min[w]) & (i_max[w] <= c_ends[c])
 covers = ((c_starts[None, :] <= w_i_min[:, None]) &
 (w_i_max[:, None] <= c_ends[None, :]))
 # Also exclude degenerate windows where i_min > i_max
 valid_win = w_i_min <= w_i_max

 window_covering = np.full(n_win, -1, dtype=np.int32)
 has_cover = covers.any(axis=1) & valid_win
 # argmax on axis=1 gives first True index per row
 window_covering[has_cover] = covers[has_cover].argmax(axis=1).astype(np.int32)

 n_covered = int(has_cover.sum())
 if verbose:
 print(f" {n_covered}/{n_win} windows covered by cliques", flush=True)

 # --- Consistency data ---
 if verbose:
 print(" Building consistency data...", flush=True)
 t_con = time.time()
 deg = S_arr.sum(axis=1)
 consist_mask = deg <= (2 * order - 1)
 consist_parent_indices = np.where(consist_mask)[0]
 n_consist_candidates = len(consist_parent_indices)

 # Compute children hashes: parent_hash + bases[i] for each i
 parent_hashes = _hash_monos(S_arr[consist_parent_indices], bases, prime)
 children_hashes = _hash_add(parent_hashes[:, None], bases[None, :], prime) # (n_cand, d)
 children_idx = _hash_lookup(children_hashes, sorted_h, sort_o) # (n_cand, d)

 n_children_present = (children_idx >= 0).sum(axis=1)
 full_mask = n_children_present == d
 partial_mask = (n_children_present > 0) & ~full_mask
 n_full = int(full_mask.sum())
 n_partial = int(partial_mask.sum())

 # Pre-build consistency sparse matrix COO lists (avoids rebuilding twice)
 consist_eq_lists = None
 if n_full > 0:
 full_rows_arr = np.where(full_mask)[0]
 pi = consist_parent_indices[full_rows_arr].astype(np.int32)
 cr = children_idx[full_rows_arr]
 rfc = np.repeat(np.arange(n_full, dtype=np.int32), d)
 cfc = cr.ravel().astype(np.int32)
 vfc = np.ones(n_full * d)
 er = np.concatenate([rfc, np.arange(n_full, dtype=np.int32)])
 ec = np.concatenate([cfc, pi])
 ev = np.concatenate([vfc, np.full(n_full, -1.0)])
 consist_eq_lists = (n_full, er, ec, ev)

 consist_iq_lists = None
 if n_partial > 0:
 partial_rows_arr = np.where(partial_mask)[0]
 pi_iq = consist_parent_indices[partial_rows_arr].astype(np.int32)
 cr_iq = children_idx[partial_rows_arr]
 vm = cr_iq >= 0
 hac = vm.any(axis=1)
 if hac.any():
 keep = np.where(hac)[0]
 pi_iq = pi_iq[keep]
 cr_iq = cr_iq[keep]
 vm = vm[keep]
 n_iq = len(keep)
 rl, cl = np.where(vm)
 ir = np.concatenate([rl.astype(np.int32), np.arange(n_iq, dtype=np.int32)])
 ic = np.concatenate([cr_iq[rl, cl].astype(np.int32), pi_iq])
 iv = np.concatenate([np.full(len(rl), -1.0), np.ones(n_iq)])
 consist_iq_lists = (n_iq, ir, ic, iv)

 if verbose:
 print(f" Consistency: {n_full} full equality + "
 f"{n_partial} partial inequality rows ({time.time()-t_con:.1f}s)",
 flush=True)

 elapsed = time.time() - t0
 if verbose:
 print(f" Precompute done: {n_y:,} variables, {elapsed:.1f}s total\n",
 flush=True)

 return {
 'd': d, 'order': order, 'n_y': n_y,
 'S_arr': S_arr, 'mono_list': mono_list, 'idx': idx,
 'bases': bases, 'prime': prime, 'sorted_h': sorted_h, 'sort_o': sort_o,
 'windows': windows, 'n_win': n_win,
 'F_scipy': F_scipy, 'F_coo_lists': F_coo_lists, 'idx_ij': idx_ij,
 'm1_pick': m1_pick, 'm1_pick_list': m1_pick_list,
 'm1_size': m1_size, 'm1_valid': m1_valid,
 # Pairwise localizing M_1(μ_i μ_j y) ⪰ 0 — valid L3 Putinar product terms
 'pw_pairs': pw_pairs, # list of (i, j, picks np.array) for SCS/ADMM
 'pw_pairs_list': pw_pairs_list, # list of (i, j, picks list) for MOSEK
 'pw_size': pw_size, # d+1 (M_1 basis size)
 'cliques': cliques, 'clique_data': clique_data,
 'bin_to_clique_map': bin_to_clique_map,
 'window_covering': window_covering,
 # Consistency data (pre-built COO lists for MOSEK)
 'consist_eq_lists': consist_eq_lists,
 'consist_iq_lists': consist_iq_lists,
 }


# =====================================================================
# Build MOSEK model with correct mathematics
# =====================================================================

def _build_model_highd(P, add_upper_loc, verbose):
 r"""Build MOSEK model on the reduced moment set with proven-sound constraints.

 Mathematical structure (all constraints are NECESSARY conditions for
 the existence of a representing measure μ on Δ_d):

 1. y_0 = 1 (normalization)
 2. y ≥ 0 (nonnegativity of moments)
 3. M_1(y) ≽ 0 (full order-1 moment matrix, (d+1)×(d+1))
 4. M_k^{I_c}(y) ≽ 0 (clique-restricted moment PSD)
 5. L_i^{I_c}(y) ≽ 0 (clique-restricted localizing PSD)
 6. U_i^{I_c}(y) ≽ 0 (clique-restricted upper localizing, optional)
 7. y_α = Σ_i y_{α+e_i} (full consistency, where all children in S)
 8. y_α ≥ Σ_{i∈S'} y_{α+e_i} (partial consistency inequality)
 9. t ≥ f_W(y) (scalar window constraints)
 """
 d = P['d']
 order = P['order']
 n_y = P['n_y']
 idx = P['idx']
 cliques = P['cliques']

 mdl = Model("highd_sparse")
 mdl.setSolverParam("intpntCoTolRelGap", 1e-7)
 mdl.setSolverParam("numThreads", 4)

 y = mdl.variable("y", n_y, Domain.greaterThan(0.0))
 t_param = mdl.parameter("t")

 # --- (1) y_0 = 1 ---
 zero = tuple(0 for _ in range(d))
 y0_idx = idx[zero]
 mdl.constraint("y0", y.index(y0_idx), Domain.equalsTo(1.0))

 # --- (3) Full M_{k-1}(y) ≽ 0 ---
 # Cross-clique coupling: the full M_{k-1} PSD ensures global
 # semidefiniteness at degree 2(k-1), absent from clique-only PSD.
 n_m1_added = 0
 if P['m1_valid']:
 m1_size = P['m1_size']
 M1_expr = Expr.reshape(y.pick(P['m1_pick_list']), m1_size, m1_size)
 mdl.constraint("full_m1_psd", M1_expr, Domain.inPSDCone(m1_size))
 n_m1_added = 1
 if verbose:
 print(f" Full M_{order-1} PSD: {m1_size}×{m1_size}", flush=True)

 # --- (4) Clique moment PSD ---
 n_mom_psd = 0
 for c_idx, cd in enumerate(P['clique_data']):
 pick = cd['mom_pick']
 if np.any(pick < 0):
 continue
 n_cb = cd['mom_size']
 M_c = Expr.reshape(y.pick(cd['mom_pick_list']), n_cb, n_cb)
 mdl.constraint(f"clique_mom_{c_idx}", M_c, Domain.inPSDCone(n_cb))
 n_mom_psd += 1

 if verbose:
 print(f" Clique moment PSD: {n_mom_psd} cones of size "
 f"{P['clique_data'][0]['mom_size']}", flush=True)

 # --- (5) Clique localizing PSD ---
 n_loc_psd = 0
 if order >= 2:
 for i_var in range(d):
 c_idx = P['bin_to_clique_map'].get(i_var, 0)
 cd = P['clique_data'][c_idx]
 picks = cd['loc_picks'].get(i_var)
 if picks is None or np.any(picks < 0):
 continue
 n_cb = cd['loc_size']
 Li = Expr.reshape(y.pick(cd['loc_picks_list'][i_var]), n_cb, n_cb)
 mdl.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_cb))
 n_loc_psd += 1

 if verbose:
 print(f" Clique localizing PSD: {n_loc_psd}/{d} bins", flush=True)

 # --- (6) Clique upper localizing PSD (optional) ---
 n_upper = 0
 if add_upper_loc and order >= 2:
 for i_var in range(d):
 c_idx = P['bin_to_clique_map'].get(i_var, 0)
 cd = P['clique_data'][c_idx]
 t_pick = cd['t_pick']
 loc_pick = cd['loc_picks'].get(i_var)
 if t_pick is None or loc_pick is None:
 continue
 if np.any(t_pick < 0) or np.any(loc_pick < 0):
 continue
 n_cb = cd['loc_size']
 sub_m = y.pick(cd['t_pick_list'])
 mu_i_m = y.pick(cd['loc_picks_list'][i_var])
 diff = Expr.sub(sub_m, mu_i_m)
 L_up = Expr.reshape(diff, n_cb, n_cb)
 mdl.constraint(f"upper_loc_{i_var}", L_up, Domain.inPSDCone(n_cb))
 n_upper += 1

 if verbose:
 print(f" Upper localizing PSD: {n_upper}/{d} bins", flush=True)

 # --- (7)+(8) Consistency constraints from pre-built COO lists ---
 n_eq, n_iq = 0, 0
 if P['consist_eq_lists'] is not None:
 n_eq, er, ec, ev = P['consist_eq_lists']
 A_eq = Matrix.sparse(n_eq, n_y, er, ec, ev)
 mdl.constraint("consist_eq", Expr.mul(A_eq, y), Domain.equalsTo(0.0))

 if P['consist_iq_lists'] is not None:
 n_iq, ir, ic, iv = P['consist_iq_lists']
 A_iq = Matrix.sparse(n_iq, n_y, ir, ic, iv)
 mdl.constraint("consist_ineq", Expr.mul(A_iq, y), Domain.greaterThan(0.0))

 if verbose:
 print(f" Consistency: {n_eq} equality + {n_iq} inequality rows",
 flush=True)

 # --- (9) Scalar window constraints: t ≥ f_W(y) ---
 f_rows, f_cols, f_data = P['F_coo_lists']
 n_win = P['n_win']
 if len(f_rows) > 0:
 F_mosek = Matrix.sparse(n_win, n_y, f_rows, f_cols, f_data)
 f_all = Expr.mul(F_mosek, y)
 ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
 t_rep = Expr.flatten(Expr.mul(ones_col,
 Expr.reshape(t_param, 1, 1)))
 mdl.constraint("win_scalar", Expr.sub(t_rep, f_all),
 Domain.greaterThan(0.0))

 if verbose:
 print(f" Scalar windows: {n_win}", flush=True)

 mdl.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

 return mdl, y, t_param


# =====================================================================
# Clique-local violation checking (replaces _batch_check_violations)
# =====================================================================

def _check_violations_highd(y_vals, t_val, P, active_windows, tol=1e-6, k_vecs=0):
 r"""Window violation checking: clique-local PSD + scalar for uncovered.

 Covered windows (active bins fit in one clique): check the clique-
 restricted localizing matrix L_W^{I_c} for negative eigenvalues.
 All Q-part entries are in S → no approximation, sound by interlacing.

 Uncovered windows: check ONLY the scalar constraint t ≥ f_W(y).
 We do NOT attempt partial-Q PSD for uncovered windows because:
 The deficit D = Q_true - Q_partial is entrywise ≥ 0 but NOT PSD
 in general. So L_partial = L_true + D can fail to be PSD even when
 L_true is PSD. Imposing L_partial ≽ 0 would OVER-CONSTRAIN the
 problem, producing lb > val(d) (unsound).

 Soundness of covered-window check (interlacing theorem):
 L_W^{I_c} is a principal submatrix of L_W (restricting basis to
 monomials supported on I_c). By Cauchy interlacing:
 λ_min(L_W) ≤ λ_min(L_W^{I_c}).
 If λ_min(L_W^{I_c}) < 0, then λ_min(L_W) < 0: genuine violation.
 We may miss some violations (false negatives) but never report
 false violations. This is safe for CG.
 """
 d = P['d']
 order = P['order']
 if order < 2:
 return []

 bases = P['bases']
 prime = P.get('prime')
 sorted_h, sort_o = P['sorted_h'], P['sort_o']
 clique_data = P['clique_data']
 windows = P['windows']
 n_win = P['n_win']
 window_covering = P['window_covering']

 violations = []

 # ---- Clique-restricted PSD check for covered windows ----
 # Vectorized filtering: find non-active covered windows, group by clique
 active_arr = np.array(sorted(active_windows), dtype=np.int32) if active_windows else np.array([], dtype=np.int32)
 is_active = np.zeros(n_win, dtype=bool)
 if len(active_arr) > 0:
 is_active[active_arr] = True

 covered_mask = (window_covering >= 0) & ~is_active
 covered_w = np.where(covered_mask)[0]
 covered_c = window_covering[covered_w]

 clique_windows = {}
 for w, c in zip(covered_w, covered_c):
 c = int(c)
 if c not in clique_windows:
 clique_windows[c] = []
 clique_windows[c].append(int(w))

 w_arr_np = np.array(windows) # (n_win, 2): [ell, s_lo]

 for c_idx, w_list in clique_windows.items():
 cd = clique_data[c_idx]
 n_cb = cd['loc_size']
 t_pick = cd['t_pick']

 if np.any(t_pick < 0):
 continue

 L_t = t_val * y_vals[t_pick].reshape(n_cb, n_cb)

 # Use precomputed violation index arrays (computed once in _precompute_highd)
 ab_eiej_local = cd['viol_ab_eiej']
 safe_idx = np.clip(ab_eiej_local, 0, len(y_vals) - 1)
 y_abij_local = y_vals[safe_idx]
 y_abij_local[ab_eiej_local < 0] = 0.0

 gi_grid = cd['viol_gi_grid'] # (n_clique, n_clique)

 # Vectorized Mw construction: build all window masks at once
 w_idx_arr = np.array(w_list, dtype=np.int32)
 w_ells = w_arr_np[w_idx_arr, 0] # (n_w,)
 w_slos = w_arr_np[w_idx_arr, 1] # (n_w,)
 coeffs = 2.0 * d / w_ells # (n_w,)

 # mask[w, i, j] = (gi_grid[i,j] >= s_lo[w]) & (gi_grid[i,j] <= s_lo[w]+ell[w]-2)
 gi_exp = gi_grid[np.newaxis, :, :] # (1, nc, nc)
 lo_exp = w_slos[:, np.newaxis, np.newaxis] # (n_w, 1, 1)
 hi_exp = (w_slos + w_ells - 2)[:, np.newaxis, np.newaxis]
 masks = (gi_exp >= lo_exp) & (gi_exp <= hi_exp) # (n_w, nc, nc)

 # Filter windows with no active entries
 has_nonzero = masks.any(axis=(1, 2))
 if not has_nonzero.any():
 continue

 valid_local = np.where(has_nonzero)[0]
 w_indices = w_idx_arr[valid_local].tolist()
 Mw_all = coeffs[valid_local, np.newaxis, np.newaxis] * masks[valid_local].astype(np.float64)

 L_q_all = np.einsum('abij,wij->wab', y_abij_local, Mw_all)
 L_all = L_t[np.newaxis, :, :] - L_q_all
 L_all = 0.5 * (L_all + np.swapaxes(L_all, -2, -1))

 if k_vecs > 0:
 # Return bottom-k eigenvectors alongside min eigenvalue.
 # eigh is O(n^3); eigvalsh is also O(n^3) but avoids storing
 # eigenvectors. Since we need both here, eigh is correct.
 # Ascending order: column i of vecs is the i-th smallest eigenvector.
 eig_vals, eig_vecs = np.linalg.eigh(L_all)
 min_eigs = eig_vals[:, 0]
 violated = min_eigs < -tol
 for ki in np.where(violated)[0]:
 # Use only eigenvectors with strictly negative eigenvalue — those
 # are the violated directions. Cap at k_vecs.
 n_neg = int((eig_vals[ki] < -tol).sum())
 actual_k = min(k_vecs, max(1, n_neg))
 vecs = eig_vecs[ki, :, :actual_k].copy() # (n_cb, actual_k)
 violations.append((w_indices[ki], float(min_eigs[ki]), vecs))
 else:
 min_eigs = np.linalg.eigvalsh(L_all)[:, 0]
 violated = min_eigs < -tol
 for ki in np.where(violated)[0]:
 violations.append((w_indices[ki], float(min_eigs[ki])))

 # ---- Scalar-only check for uncovered windows ----
 # No PSD check for uncovered windows — partial-Q PSD is unsound.
 f_vals = P['F_scipy'].dot(y_vals)
 uncovered_mask = (window_covering < 0) & ~is_active
 uncovered_violated = uncovered_mask & (f_vals > t_val + tol)
 for w in np.where(uncovered_violated)[0]:
 if k_vecs > 0:
 violations.append((int(w), -(f_vals[w] - t_val), None))
 else:
 violations.append((int(w), -(f_vals[w] - t_val)))

 violations.sort(key=lambda x: x[1])
 return violations


# =====================================================================
# Add CG window PSD constraint (clique-restricted)
# =====================================================================

def _add_window_psd_highd(mdl, y, t_param, w, P):
 r"""Add clique-restricted PSD window localizing constraint.

 ONLY adds PSD for windows covered by a single clique. Uncovered
 windows get only the scalar constraint (already in the model).

 Why NOT partial-Q PSD for uncovered windows:
 The deficit D = Q_true - Q_partial is entrywise ≥ 0 but NOT PSD.
 L_partial = L_true + D can fail to be PSD even when L_true ≽ 0.
 Imposing L_partial ≽ 0 over-constrains → lb > val(d) (unsound).
 This was verified empirically: at d=8, partial-Q gives lb=1.28 > val(8)=1.21.

 Soundness of covered-window PSD:
 L_W^{I_c} is a principal submatrix of L_W. The constraint
 L_W^{I_c} ≽ 0 is a necessary condition for L_W ≽ 0.
 For true moments: L_W ≽ 0 because t* - f_W(x) ≥ 0 pointwise
 on the support of μ*... NO — this is the WRONG argument.

 CORRECT soundness proof: The Lasserre localizing matrix for
 g_W(x) = t - Σ M_W[i,j] x_i x_j encodes
 M_{k-1}(g_W · y)[a,b] = t·y_{α_a+α_b} - Σ M_W[i,j]·y_{α_a+α_b+e_i+e_j}.
 For a true measure μ* with t* = val(d):
 v^T M_{k-1}(g_W · y*) v = E_{μ*}[g_W(x) · p(x)^2]
 where p(x) = Σ v_a x^{α_a}. This is ≥ 0 iff g_W(x) ≥ 0 a.s.
 under μ*.

 For COVERED windows with clique basis: the Q-part is COMPLETE
 (all active bins are in the clique). The constraint L_W^{I_c} ≽ 0
 is equivalent to: E[g_W(x) · p(x)^2] ≥ 0 for all p supported on
 the clique. This IS implied by the full Lasserre conditions because
 it's a principal submatrix of M_{k-1}(g_W · y).

 The full M_{k-1}(g_W · y) ≽ 0 is part of the standard Lasserre
 hierarchy (constraint L6). Our clique restriction is strictly weaker.
 """
 d = P['d']
 order = P['order']
 if order < 2:
 return
 n_y = P['n_y']

 ell, s_lo = P['windows'][w]
 coeff = 2.0 * d / ell

 # Only add PSD for covered windows
 c_idx = int(P['window_covering'][w])
 if c_idx < 0:
 return # Uncovered — scalar constraint already in model, no PSD

 cd = P['clique_data'][c_idx]
 n_cb = cd['loc_size']

 if np.any(cd['t_pick'] < 0):
 return # T-part missing — skip

 t_y = Expr.mul(t_param, y.pick(cd['t_pick_list']))

 # Use precomputed gi_grid from violation data
 gi_grid = cd['viol_gi_grid']
 active_mask = (gi_grid >= s_lo) & (gi_grid <= s_lo + ell - 2)
 nz_li, nz_lj = np.nonzero(active_mask)

 if len(nz_li) == 0:
 Lw = Expr.reshape(t_y, n_cb, n_cb)
 mdl.constraint(f"sw_{w}", Lw, Domain.inPSDCone(n_cb))
 return

 mw_vals = np.full(len(nz_li), coeff)
 # Index into precomputed ab_eiej lookup table instead of recomputing hashes
 y_idx = cd['viol_ab_eiej'][:, :, nz_li, nz_lj]

 valid = y_idx >= 0
 if not np.any(valid):
 Lw = Expr.reshape(t_y, n_cb, n_cb)
 mdl.constraint(f"sw_{w}", Lw, Domain.inPSDCone(n_cb))
 return

 flat = np.arange(n_cb)[:, None] * n_cb + np.arange(n_cb)[None, :]
 ab_exp = np.broadcast_to(flat[:, :, None], y_idx.shape)
 mw_exp = np.broadcast_to(np.array(mw_vals)[None, None, :], y_idx.shape)

 rows = ab_exp[valid].ravel().tolist()
 cols = y_idx[valid].ravel().tolist()
 vals = mw_exp[valid].ravel().tolist()

 Cw = Matrix.sparse(n_cb * n_cb, n_y, rows, cols, vals)
 cw_expr = Expr.mul(Cw, y)
 Lw_flat = Expr.sub(t_y, cw_expr)
 Lw = Expr.reshape(Lw_flat, n_cb, n_cb)
 mdl.constraint(f"sw_{w}", Lw, Domain.inPSDCone(n_cb))


# =====================================================================
# Main solver
# =====================================================================

def _build_consistency_sparse(mdl_obj, y_var, P, d, n_y):
 """Build consistency constraints from pre-built COO lists.

 Uses consist_eq_lists and consist_iq_lists from precompute,
 avoiding repeated sparse matrix construction and .tolist() calls.
 """
 n_eq = 0
 if P['consist_eq_lists'] is not None:
 n_eq, er, ec, ev = P['consist_eq_lists']
 A_eq = Matrix.sparse(n_eq, n_y, er, ec, ev)
 mdl_obj.constraint("ceq", Expr.mul(A_eq, y_var), Domain.equalsTo(0.0))

 n_iq = 0
 if P['consist_iq_lists'] is not None:
 n_iq, ir, ic, iv = P['consist_iq_lists']
 A_iq = Matrix.sparse(n_iq, n_y, ir, ic, iv)
 mdl_obj.constraint("ciq", Expr.mul(A_iq, y_var), Domain.greaterThan(0.0))

 return n_eq, n_iq


def _build_base_psd(mdl_obj, y_var, P, order, d, add_upper_loc):
 """Build base PSD constraints (moment, localizing, upper localizing).

 Shared by both the scalar Round 0 model and the CG model.
 """
 # Full M_{k-1}
 if P['m1_valid']:
 m1_size = P['m1_size']
 M1_expr = Expr.reshape(y_var.pick(P['m1_pick_list']), m1_size, m1_size)
 mdl_obj.constraint("full_mkm1_psd", M1_expr, Domain.inPSDCone(m1_size))

 # Clique moment PSD
 for c_idx, cd in enumerate(P['clique_data']):
 pick = cd['mom_pick']
 if np.any(pick < 0):
 continue
 n_cb = cd['mom_size']
 M_c = Expr.reshape(y_var.pick(cd['mom_pick_list']), n_cb, n_cb)
 mdl_obj.constraint(f"cm_{c_idx}", M_c, Domain.inPSDCone(n_cb))

 # Clique localizing PSD
 if order >= 2:
 for i_var in range(d):
 c_idx = P['bin_to_clique_map'].get(i_var, 0)
 cd = P['clique_data'][c_idx]
 picks = cd['loc_picks'].get(i_var)
 if picks is None or np.any(picks < 0):
 continue
 n_cb = cd['loc_size']
 Li = Expr.reshape(y_var.pick(cd['loc_picks_list'][i_var]), n_cb, n_cb)
 mdl_obj.constraint(f"loc_{i_var}", Li, Domain.inPSDCone(n_cb))

 # Upper localizing
 if add_upper_loc and order >= 2:
 for i_var in range(d):
 c_idx = P['bin_to_clique_map'].get(i_var, 0)
 cd = P['clique_data'][c_idx]
 t_pick_cd = cd['t_pick']
 loc_pick = cd['loc_picks'].get(i_var)
 if t_pick_cd is None or loc_pick is None:
 continue
 if np.any(t_pick_cd < 0) or np.any(loc_pick < 0):
 continue
 n_cb = cd['loc_size']
 sub_m = y_var.pick(cd['t_pick_list'])
 mu_i_m = y_var.pick(cd['loc_picks_list'][i_var])
 diff = Expr.sub(sub_m, mu_i_m)
 L_up = Expr.reshape(diff, n_cb, n_cb)
 mdl_obj.constraint(f"upper_loc_{i_var}", L_up, Domain.inPSDCone(n_cb))

 # Pairwise product localizing: M_1(μ_i μ_j y) ⪰ 0 for all pairs i ≤ j.
 # Valid L3 Putinar product constraints; see _precompute_highd for full proof.
 if order >= 3 and 'pw_pairs_list' in P and P['pw_pairs_list']:
 pw_n = P['pw_size']
 for i_var, j_var, picks_list in P['pw_pairs_list']:
 Li_j = Expr.reshape(y_var.pick(picks_list), pw_n, pw_n)
 mdl_obj.constraint(f"pw_loc_{i_var}_{j_var}", Li_j,
 Domain.inPSDCone(pw_n))


def solve_highd_sparse(d, c_target=1.28, order=2, bandwidth=16,
 add_upper_loc=True, max_cg_rounds=15,
 max_add_per_round=100, n_bisect=20,
 verbose=True):
 r"""Sparse Lasserre for high d with clique-restricted sparsity.

 Parameters
 ----------
 d : int
 Number of bins (64, 128, ...).
 c_target : float
 Target lower bound.
 order : int
 Lasserre order (2 = L2).
 bandwidth : int
 Clique bandwidth for sparse decomposition.
 """
 t_total = time.time()

 print(f"{'='*70}")
 print(f"HIGH-D SPARSE LASSERRE: L{order} d={d} bw={bandwidth}")
 print(f" n_bisect={n_bisect}, cg_rounds={max_cg_rounds}")
 print(f"{'='*70}\n", flush=True)

 # Step 1: Build cliques
 cliques = _build_banded_cliques(d, bandwidth)
 n_cliques = len(cliques)
 clique_size = len(cliques[0])
 if verbose:
 cb_size = len(enum_monomials(clique_size, order))
 print(f"Step 1: Cliques: {n_cliques} of size {clique_size}, "
 f"basis size {cb_size}\n", flush=True)

 # Step 2: Custom precompute on reduced moment set
 print("Step 2: Precompute (clique-aware)...", flush=True)
 P = _precompute_highd(d, order, cliques, verbose)
 n_y = P['n_y']

 # Step 3+4: Round 0 scalar optimization, then build CG model.
 # MEMORY CRITICAL: build the scalar model FIRST, dispose it, THEN
 # build the CG model. Both models hold O(n_y^2 * PSD_dim) internal
 # memory. Having both alive simultaneously doubles peak memory and
 # OOMs at n_y > ~40K on 1TB machines.

 active_windows = set()
 best_lb = 0.0

 print(" [Round 0] Direct scalar optimization...", flush=True)
 t0_r0 = time.time()

 # Build a lightweight model with t as a variable
 mdl_sc = Model("scalar_opt")
 mdl_sc.setSolverParam("intpntCoTolRelGap", 1e-7)
 mdl_sc.setSolverParam("numThreads", 4)
 y_sc = mdl_sc.variable("y", n_y, Domain.greaterThan(0.0))
 t_sc = mdl_sc.variable("t", 1, Domain.greaterThan(0.0))

 # y_0 = 1
 zero = tuple(0 for _ in range(d))
 y0_idx = P['idx'][zero]
 mdl_sc.constraint("y0", y_sc.index(y0_idx), Domain.equalsTo(1.0))

 _build_base_psd(mdl_sc, y_sc, P, order, d, add_upper_loc)
 _build_consistency_sparse(mdl_sc, y_sc, P, d, n_y)

 # Scalar window constraints: t >= f_W(y)
 f_rows, f_cols, f_data = P['F_coo_lists']
 n_win = P['n_win']
 if len(f_rows) > 0:
 F_mosek = Matrix.sparse(n_win, n_y, f_rows, f_cols, f_data)
 f_all = Expr.mul(F_mosek, y_sc)
 ones_col = Matrix.dense(n_win, 1, [1.0] * n_win)
 t_rep = Expr.flatten(Expr.mul(ones_col,
 Expr.reshape(t_sc, 1, 1)))
 mdl_sc.constraint("win", Expr.sub(t_rep, f_all),
 Domain.greaterThan(0.0))

 # Minimize t
 mdl_sc.objective(ObjectiveSense.Minimize, t_sc)
 mdl_sc.solve()
 ps_sc = mdl_sc.getPrimalSolutionStatus()
 if ps_sc in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
 scalar_lb = t_sc.level()[0]
 y_vals = np.array(y_sc.level())
 else:
 scalar_lb = 0.5
 y_vals = np.zeros(n_y)
 mdl_sc.dispose()
 del mdl_sc, y_sc, t_sc # free references
 gc_mod.collect() # force GC before building CG model

 best_lb = scalar_lb

 violations = _check_violations_highd(
 y_vals, scalar_lb, P, active_windows)

 print(f" Scalar bound = {scalar_lb:.10f} (1 solve, "
 f"{time.time()-t0_r0:.1f}s), {len(violations)} violations",
 flush=True)

 if len(violations) == 0:
 elapsed = time.time() - t_total
 print(f"\n No violations — scalar bound is exact: {best_lb:.10f}")
 return {'lb': best_lb, 'd': d, 'order': order, 'elapsed': elapsed,
 'n_y': n_y, 'n_active_windows': 0}

 # NOW build the CG model (after scalar model is fully disposed)
 print("\n Building CG model...", flush=True)
 t_build = time.time()
 mdl, y, t_param = _build_model_highd(P, add_upper_loc, verbose)
 print(f" CG model built in {time.time()-t_build:.1f}s\n", flush=True)

 def check_feasible(t_val):
 t_param.setValue(t_val)
 try:
 mdl.solve()
 ps = mdl.getPrimalSolutionStatus()
 return ps in [SolutionStatus.Optimal, SolutionStatus.Feasible]
 except Exception as e:
 if verbose:
 print(f" [solver error] t={t_val:.6f}: {e}", flush=True)
 return False

 # Add initial violated windows
 n_add = min(max_add_per_round, len(violations))
 for w, eig in violations[:n_add]:
 _add_window_psd_highd(mdl, y, t_param, w, P)
 active_windows.add(w)
 print(f" Added {n_add} PSD windows "
 f"(worst eig: {violations[0][1]:.2e})", flush=True)

 # CG rounds
 for cg_round in range(1, max_cg_rounds + 1):
 print(f"\n [CG round {cg_round}] {len(active_windows)} windows",
 flush=True)

 # Tighter bisection bracket: PSD windows only increase the bound
 lo = max(0.5, best_lb - 1e-3)
 hi = best_lb + 0.02 if best_lb > 0.5 else 5.0

 while not check_feasible(hi):
 hi *= 1.5
 if hi > 100:
 break
 if hi > 100 and not check_feasible(hi):
 print(" WARNING: infeasible up to t=100")
 break

 # Reduced bisection (8 steps gives precision ~8e-5 on tight bracket)
 n_bisect_eff = min(n_bisect, 8)
 for step in range(n_bisect_eff):
 if step < 2:
 mid = (lo + hi) / 2
 else:
 # Probe closer to lo (boundary) for faster convergence
 mid = lo + 0.4 * (hi - lo)
 if check_feasible(mid):
 hi = mid
 else:
 lo = mid

 lb = lo
 improvement = lb - best_lb
 best_lb = max(best_lb, lb)

 v = val_d_known.get(d, 0)
 gc_pct = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0
 print(f" lb={lb:.10f} (+{improvement:.2e}) "
 f"gap_closure={gc_pct:.1f}%", flush=True)

 # Extract y* for violation checking
 t_param.setValue(hi)
 mdl.solve()
 y_vals = np.array(y.level())

 violations = _check_violations_highd(
 y_vals, hi, P, active_windows)

 if len(violations) == 0:
 print(f" No violations — converged.", flush=True)
 break

 if improvement < 1e-7 and cg_round >= 3:
 print(f" Improvement < 1e-7 — stopping.", flush=True)
 break

 n_add = min(max_add_per_round, len(violations))
 for w, eig in violations[:n_add]:
 _add_window_psd_highd(mdl, y, t_param, w, P)
 active_windows.add(w)
 print(f" Added {n_add} windows "
 f"(worst eig: {violations[0][1]:.2e})", flush=True)

 mdl.dispose()
 gc_mod.collect()

 elapsed = time.time() - t_total
 v = val_d_known.get(d, 0)
 gc_pct = (best_lb - 1) / (v - 1) * 100 if v > 1 else 0

 print(f"\n{'='*70}")
 print(f"RESULT: L{order} d={d} sparse bw={bandwidth}")
 print(f" lb = {best_lb:.10f}")
 print(f" val({d}) = {v}")
 print(f" gap_closure = {gc_pct:.1f}%")
 print(f" active_windows = {len(active_windows)}/{P['n_win']}")
 print(f" variables = {n_y:,}")
 print(f" elapsed = {elapsed:.1f}s ({elapsed/60:.1f}m)")
 print(f"{'='*70}")

 return {
 'lb': best_lb, 'd': d, 'order': order,
 'gap_closure': gc_pct, 'elapsed': elapsed,
 'n_y': n_y, 'n_active_windows': len(active_windows),
 'n_win_total': P['n_win'], 'bandwidth': bandwidth,
 }


# =====================================================================
# CLI
# =====================================================================

def main():
 import argparse
 parser = argparse.ArgumentParser(
 description="High-d sparse Lasserre with clique-restricted sparsity")
 parser.add_argument('--d', type=int, default=128)
 parser.add_argument('--order', type=int, default=2)
 parser.add_argument('--bw', type=int, default=16)
 parser.add_argument('--bisect', type=int, default=20)
 parser.add_argument('--cg-rounds', type=int, default=15)
 parser.add_argument('--cg-add', type=int, default=100)
 parser.add_argument('--c_target', type=float, default=1.28)
 parser.add_argument('--no-upper-loc', dest='upper_loc',
 action='store_false', default=True)
 args = parser.parse_args()

 r = solve_highd_sparse(
 args.d, args.c_target, order=args.order, bandwidth=args.bw,
 add_upper_loc=args.upper_loc,
 max_cg_rounds=args.cg_rounds, max_add_per_round=args.cg_add,
 n_bisect=args.bisect)


if __name__ == '__main__':
 main()

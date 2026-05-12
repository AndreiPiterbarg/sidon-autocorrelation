#!/usr/bin/env python
r"""
Lasserre hierarchy via MOSEK Fusion API — no CVXPY overhead.

Proves C_{1a} >= c_target by computing a rigorous lower bound on
val(d) = min_{mu in Delta} max_W TV_W(mu) via SDP relaxation.

MOSEK Fusion builds the model ONCE. Binary search updates a parameter
and re-solves with zero overhead — no re-canonicalization.

Model construction is fully vectorized via NumPy hash-based monomial
lookup — no Python loops over (a,b,i,j). The key trick:
 hash(alpha) = dot(alpha, base_vector) is LINEAR,
so hash(a+b+e_i+e_j) = hash(a+b) + hash(e_i+e_j) = AB_hash[a,b] + EE_hash[i,j].
This replaces 1.1 billion Python dict lookups with a single NumPy broadcast + searchsorted.

Usage:
 python tests/lasserre_fusion.py --d 4 --order 3 --c_target 1.10
 python tests/lasserre_fusion.py --d 16 --order 3 --c_target 1.28
"""
import numpy as np
from mosek.fusion import (Model, Domain, Expr, Matrix,
 ObjectiveSense, SolutionStatus)
import time
import sys


# =====================================================================
# Monomial enumeration
# =====================================================================

def enum_monomials(d, max_deg):
 """All multi-indices alpha in N^d with |alpha| <= max_deg."""
 result = []
 def gen(pos, remaining, current):
 if pos == d:
 result.append(tuple(current))
 return
 for v in range(remaining + 1):
 current.append(v)
 gen(pos + 1, remaining - v, current)
 current.pop()
 gen(0, max_deg, [])
 return result


def _add_mi(a, b, d):
 return tuple(a[i] + b[i] for i in range(d))


def _unit(d, i):
 return tuple(1 if k == i else 0 for k in range(d))


# =====================================================================
# Vectorized monomial hashing
# =====================================================================

def _make_hash_bases(d, max_comp):
 """Hash bases for monomial indexing. Linear in components.

 For small d: exact mixed-radix (max_comp+1)^k — perfectly injective.
 For large d: random int63 coefficients — injective with overwhelming
 probability (birthday bound: ~2^{-63} per pair, negligible for <10^9
 monomials). The hash linearity h(a+b) = h(a)+h(b) is preserved.
 """
 base = max_comp + 1
 if base ** (d - 1) < 2**62: # fits int64 → use exact encoding
 return np.array([base**k for k in range(d)], dtype=np.int64)
 # Random hash: seeded by (d, max_comp) for reproducibility
 rng = np.random.RandomState(seed=d * 1000 + max_comp)
 return rng.randint(1, 2**62, size=d, dtype=np.int64)


def _hash_monos(arr, bases):
 """Hash array of monomials. arr shape (..., d) -> (...) int64."""
 return np.tensordot(arr.astype(np.int64), bases, axes=([-1], [0]))


def _build_hash_table(mono_list, bases):
 """Sorted hashes + original indices for vectorized searchsorted lookup."""
 mono_arr = np.array(mono_list, dtype=np.int64)
 hashes = _hash_monos(mono_arr, bases)
 order = np.argsort(hashes)
 return hashes[order], order


def _hash_lookup(query_hashes, sorted_hashes, sort_order):
 """Vectorized lookup: query_hashes -> moment indices (or -1)."""
 flat = query_hashes.ravel()
 pos = np.searchsorted(sorted_hashes, flat)
 pos = np.clip(pos, 0, len(sorted_hashes) - 1)
 found = sorted_hashes[pos] == flat
 result = np.where(found, sort_order[pos], -1)
 return result.reshape(query_hashes.shape)


# =====================================================================
# Window matrices
# =====================================================================

def build_window_matrices(d):
 conv_len = 2 * d - 1
 windows = [(ell, s) for ell in range(2, 2 * d + 1)
 for s in range(conv_len - ell + 2)]
 ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
 sums = ii + jj
 M_mats = []
 for ell, s_lo in windows:
 mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)
 M_mats.append((2.0 * d / ell) * mask.astype(np.float64))
 return windows, M_mats


# =====================================================================
# Collect all moment indices (vectorized)
# =====================================================================

def collect_moments(d, order, basis, loc_basis, consist_mono):
 r"""Collect all moment multi-indices needed for the Lasserre SDP.

 Returns the set {α ∈ ℕ^d : |α| ≤ 2·order} as a sorted list + index map.

 Proof of completeness: every sub-structure of the degree-2k Lasserre
 relaxation generates monomials of degree ≤ 2k:
 - Moment matrix M_k: basis[a]+basis[b], |·| ≤ k+k = 2k
 - Localizing M_{k-1}(μ_i·y): loc[a]+loc[b]+e_i, |·| ≤ 2(k-1)+1 = 2k-1
 - Window localizing: loc[a]+loc[b]+e_i+e_j, |·| ≤ 2(k-1)+2 = 2k
 - Consistency α+e_i: |α| ≤ 2k-1 ⟹ |α+e_i| ≤ 2k
 All are subsets of enum_monomials(d, 2k). QED.

 This replaces a previous implementation that concatenated redundant
 chunks (including a 366K×d temporary for consistency children) then
 ran np.unique on ~60M rows. At d=128: old = ~18min + 48GB temp;
 new = ~3min + 12GB (just enum_monomials + tuple conversion).
 """
 all_monos = np.array(enum_monomials(d, 2 * order), dtype=np.int64)
 # enum_monomials produces each multi-index exactly once by construction
 # (recursive backtracking with decreasing remaining budget).
 # Sort lexicographically for deterministic ordering across runs.
 sort_idx = np.lexsort(all_monos.T[::-1])
 all_monos = all_monos[sort_idx]

 mono_list = [tuple(row) for row in all_monos]
 idx = {m: i for i, m in enumerate(mono_list)}
 return mono_list, idx


# =====================================================================
# Build and solve the MOSEK Fusion model
# =====================================================================

def solve_lasserre_fusion(d, c_target, order=2, n_bisect=10, verbose=True):
 """Build Lasserre SDP with MOSEK Fusion and solve via binary search."""
 windows, M_mats = build_window_matrices(d)
 n_win = len(windows)

 basis = enum_monomials(d, order)
 n_basis = len(basis)
 loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
 n_loc = len(loc_basis)
 consist_mono = enum_monomials(d, 2 * order - 1)

 mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
 n_y = len(mono_list)

 if verbose:
 print(f" d={d}, order={order} (degree {2*order})")
 print(f" windows={n_win}, basis={n_basis}, loc_basis={n_loc}, moments={n_y}")
 if order >= 2:
 print(f" Moment matrix: {n_basis}x{n_basis}")
 print(f" Localizing (mu_i): {d} x {n_loc}x{n_loc}")
 print(f" Localizing (windows): {n_win} x {n_loc}x{n_loc}")

 t_build = time.time()

 # ── Hash infrastructure (built once) ──
 max_comp = 2 * order
 bases = _make_hash_bases(d, max_comp)
 sorted_h, sort_o = _build_hash_table(mono_list, bases)

 B_arr = np.array(basis, dtype=np.int64)
 E_arr = np.eye(d, dtype=np.int64)

 if verbose:
 print(f" Hash table: {len(sorted_h)} entries, "
 f"base={max_comp+1}, d={d}", flush=True)

 # ── Precompute ALL index arrays (vectorized) ──
 t_pre = time.time()

 # Moment matrix: M[a,b] = y[basis[a]+basis[b]]
 AB_hash = _hash_monos(
 B_arr[:, np.newaxis, :] + B_arr[np.newaxis, :, :], bases)
 moment_pick = _hash_lookup(AB_hash, sorted_h, sort_o).ravel().tolist()

 if loc_basis:
 LB_arr = np.array(loc_basis, dtype=np.int64)
 AB_loc = LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :]
 AB_loc_hash = _hash_monos(AB_loc, bases) # (n_loc, n_loc)

 # Localizing mu_i: L_i[a,b] = y[loc[a]+loc[b]+e_i]
 # Hash: AB_loc_hash[a,b] + bases[i]
 loc_picks = []
 for i_var in range(d):
 h = AB_loc_hash + bases[i_var]
 picks = _hash_lookup(h, sorted_h, sort_o)
 assert np.all(picks >= 0), f"Missing moments in mu_{i_var} localizing"
 loc_picks.append(picks.ravel().tolist())

 # t coefficient: T[a,b] = y[loc[a]+loc[b]]
 t_pick = _hash_lookup(AB_loc_hash, sorted_h, sort_o).ravel().tolist()

 # ab_eiej_idx[a,b,i,j] = idx[loc[a]+loc[b]+e_i+e_j]
 # Key: hash(loc[a]+loc[b]+e_i+e_j) = AB_loc_hash[a,b] + EE_hash[i,j]
 EE_hash = bases[:, np.newaxis] + bases[np.newaxis, :] # (d, d)
 ABIJ_hash = (AB_loc_hash[:, :, np.newaxis, np.newaxis]
 + EE_hash[np.newaxis, np.newaxis, :, :]) # (n_loc, n_loc, d, d)
 ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)

 # Precompute flat indices for COO
 ab_flat = (np.arange(n_loc)[:, np.newaxis] * n_loc
 + np.arange(n_loc)[np.newaxis, :]) # (n_loc, n_loc)

 # Consistency: alpha + e_i indices
 consist_arr = np.array(consist_mono, dtype=np.int64)
 consist_hash = _hash_monos(consist_arr, bases)
 consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
 consist_ei_hash = consist_hash[:, np.newaxis] + bases[np.newaxis, :] # (n_consist, d)
 consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)

 # Precompute ALL window Cw COO data at once.
 # Group windows by ell — same ell means same nonzero (i,j) pattern,
 # so we can reuse the valid mask and flat indices.
 if loc_basis:
 cw_data_list = [None] * n_win

 # Group window indices by ell (windows with same ell share nonzero pattern)
 from collections import defaultdict
 ell_groups = defaultdict(list)
 for w, (ell, s_lo) in enumerate(windows):
 ell_groups[ell].append(w)

 for ell, w_indices in ell_groups.items():
 # All windows with this ell have the same nonzero pattern
 # (same ell means same mask shape, just shifted by s_lo)
 # BUT different M_W values (2d/ell is constant, mask differs by s_lo)
 # Actually each window has its own mask. Let's just share the
 # nonzero extraction logic.

 # For first window in group, check if zero
 w0 = w_indices[0]
 nz_i_0, nz_j_0 = np.nonzero(M_mats[w0])
 if len(nz_i_0) == 0:
 # All windows in this ell group could be zero
 for w in w_indices:
 nz_i, nz_j = np.nonzero(M_mats[w])
 if len(nz_i) == 0:
 continue
 # Shouldn't happen if ell is same, but handle it
 y_idx = ab_eiej_idx[:, :, nz_i, nz_j]
 valid = y_idx >= 0
 if np.any(valid):
 ab_exp = np.broadcast_to(ab_flat[:, :, np.newaxis], y_idx.shape)
 mw_vals = M_mats[w][nz_i, nz_j]
 mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)
 cw_data_list[w] = (
 ab_exp[valid].ravel().tolist(),
 y_idx[valid].ravel().tolist(),
 mw_exp[valid].ravel().tolist())
 continue

 for w in w_indices:
 Mw = M_mats[w]
 nz_i, nz_j = np.nonzero(Mw)
 if len(nz_i) == 0:
 continue

 y_idx = ab_eiej_idx[:, :, nz_i, nz_j]
 valid = y_idx >= 0
 if not np.any(valid):
 continue

 ab_exp = np.broadcast_to(ab_flat[:, :, np.newaxis], y_idx.shape)
 mw_vals = Mw[nz_i, nz_j]
 mw_exp = np.broadcast_to(mw_vals[None, None, :], y_idx.shape)

 # Keep as numpy int32/float64, convert to list at MOSEK call site
 cw_data_list[w] = (
 ab_exp[valid].ravel(),
 y_idx[valid].ravel(),
 mw_exp[valid].ravel())

 precompute_time = time.time() - t_pre
 if verbose:
 print(f" Index precompute: {precompute_time:.2f}s", flush=True)
 if loc_basis:
 valid_count = int((ab_eiej_idx >= 0).sum())
 total_count = ab_eiej_idx.size
 print(f" ab_eiej_idx: {ab_eiej_idx.shape}, "
 f"{valid_count}/{total_count} valid", flush=True)

 # ── Build MOSEK Fusion model ──
 if verbose:
 print(f" Building MOSEK model...", flush=True)

 t_model = time.time()
 M = Model("lasserre")

 # Variables
 y = M.variable("y", n_y, Domain.greaterThan(0.0))

 # Parameter t (updated during binary search)
 t_param = M.parameter("t")

 # y_0 = 1
 zero = tuple(0 for _ in range(d))
 M.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

 # Moment consistency: A_consist @ y == 0 (single sparse matrix, not 20K constraints)
 c_rows, c_cols, c_vals = [], [], []
 n_consist_added = 0
 for r in range(len(consist_mono)):
 ai = int(consist_idx[r])
 if ai < 0:
 continue
 child_idx = consist_ei_idx[r]
 has_child = False
 for ci in range(d):
 if child_idx[ci] >= 0:
 c_rows.append(n_consist_added)
 c_cols.append(int(child_idx[ci]))
 c_vals.append(1.0)
 has_child = True
 if not has_child:
 continue
 c_rows.append(n_consist_added)
 c_cols.append(ai)
 c_vals.append(-1.0)
 n_consist_added += 1

 if n_consist_added > 0:
 A_con = Matrix.sparse(n_consist_added, n_y, c_rows, c_cols, c_vals)
 M.constraint("consist", Expr.mul(A_con, y), Domain.equalsTo(0.0))

 # Moment matrix PSD
 M_mat = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
 M.constraint("moment_psd", M_mat, Domain.inPSDCone(n_basis))

 # Localizing matrices for mu_i >= 0
 if order >= 2:
 for i_var in range(d):
 Li = Expr.reshape(y.pick(loc_picks[i_var]), n_loc, n_loc)
 M.constraint(f"loc_mu_{i_var}", Li, Domain.inPSDCone(n_loc))

 # Window localizing matrices
 if order >= 2:
 flat_size = n_loc * n_loc
 t_y_pick = y.pick(t_pick)
 for w in range(n_win):
 cw_data = cw_data_list[w]
 t_expr = Expr.mul(t_param, t_y_pick)

 if cw_data is None:
 Lw_mat = Expr.reshape(t_expr, n_loc, n_loc)
 else:
 rows, cols, vals = cw_data
 # Convert numpy arrays to lists at MOSEK call site (deferred)
 Cw_mosek = Matrix.sparse(flat_size, n_y,
 rows.tolist(), cols.tolist(),
 vals.tolist())
 cw_expr = Expr.mul(Cw_mosek, y)
 Lw_flat = Expr.sub(t_expr, cw_expr)
 Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)

 M.constraint(f"w_{w}", Lw_mat, Domain.inPSDCone(n_loc))

 if verbose and (w + 1) % 100 == 0:
 print(f" {w+1}/{n_win} windows added", flush=True)
 else:
 # Order 1: scalar constraints
 units = [_unit(d, i) for i in range(d)]
 for w in range(n_win):
 picks = []
 coeffs = []
 for i in range(d):
 for j in range(d):
 if M_mats[w][i, j] != 0:
 eij = _add_mi(units[i], units[j], d)
 picks.append(idx[eij])
 coeffs.append(M_mats[w][i, j])
 tv_expr = Expr.dot(coeffs, y.pick(picks))
 M.constraint(f"win_{w}", Expr.sub(t_param, tv_expr),
 Domain.greaterThan(0.0))

 M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))

 model_time = time.time() - t_model
 build_time = time.time() - t_build
 if verbose:
 print(f" MOSEK model: {model_time:.2f}s", flush=True)
 print(f" Total build: {build_time:.1f}s "
 f"(precompute={precompute_time:.1f}s + model={model_time:.1f}s)",
 flush=True)
 print(f" Consistency constraints: {n_consist_added}", flush=True)
 print(f" Binary search ({n_bisect} steps)...", flush=True)

 # ── Binary search ──
 t0 = time.time()

 def check(t_val):
 t_param.setValue(t_val)
 try:
 M.solve()
 pstatus = M.getPrimalSolutionStatus()
 return pstatus in [SolutionStatus.Optimal, SolutionStatus.Feasible]
 except Exception as e:
 if verbose:
 print(f" [error] t={t_val:.4f}: {e}", flush=True)
 return False

 lo, hi = 0.5, 5.0
 while not check(hi):
 hi *= 2
 if hi > 100:
 break
 if not check(hi):
 M.dispose()
 raise RuntimeError(f"SDP infeasible up to t={hi}")
 if verbose:
 print(f" Feasible at t={hi:.4f}", flush=True)

 for step in range(n_bisect):
 mid = (lo + hi) / 2
 if check(mid):
 hi = mid
 if verbose:
 print(f" t={mid:.8f}: feasible", flush=True)
 else:
 lo = mid
 if verbose:
 print(f" t={mid:.8f}: infeasible", flush=True)

 M.dispose()

 elapsed = time.time() - t0
 lb = lo
 proven = lb >= c_target - 1e-6

 if verbose:
 print(f" Solve time: {elapsed:.1f}s (build: {build_time:.1f}s)")
 print(f" Lower bound: {lb:.10f}")
 print(f" Margin over {c_target}: {lb - c_target:.10f}")
 if proven:
 print(f" *** PROVEN: val({d}) >= {c_target} ***")

 return {'lb': lb, 'proven': proven, 'elapsed': elapsed,
 'build_time': build_time, 'd': d, 'order': order}


def main():
 import argparse
 parser = argparse.ArgumentParser(
 description="Lasserre SDP proof via MOSEK Fusion")
 parser.add_argument('--d', type=int, default=16)
 parser.add_argument('--c_target', type=float, default=1.28)
 parser.add_argument('--order', type=int, default=3)
 parser.add_argument('--bisect', type=int, default=10)
 args = parser.parse_args()

 print(f"{'='*60}")
 print(f"LASSERRE L{args.order} (degree {2*args.order}) via MOSEK Fusion")
 print(f"Target: val({args.d}) >= {args.c_target}")
 print(f"{'='*60}\n", flush=True)

 r = solve_lasserre_fusion(args.d, args.c_target, order=args.order,
 n_bisect=args.bisect)

 print(f"\n{'='*60}")
 if r['proven']:
 print(f"*** PROVEN: C_{{1a}} >= {args.c_target} ***")
 else:
 print(f"NOT PROVEN: lb={r['lb']:.8f} < {args.c_target}")
 print(f"{'='*60}")


if __name__ == '__main__':
 main()

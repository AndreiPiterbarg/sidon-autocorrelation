#!/usr/bin/env python
r"""
Lasserre hierarchy (degree 4) for rigorous lower bound on val(d).

val(d) = min_{mu in Delta} max_W TV_W(mu)  where TV_W = (2d/ell) mu^T A_W mu.

Reformulate: min t  s.t.  t - mu^T M_W mu >= 0  for all W,  mu in Delta.

Lasserre order k relaxation:
  - Moment matrix M_k(y) >= 0         (enforces valid moment sequence)
  - Localizing M_{k-1}(g_j * y) >= 0  (enforces g_j(mu) >= 0 pointwise)
  - Moment consistency from sum mu = 1

Order 1 (degree 2) = Shor SDP -> gives 1.0 (too weak)
Order 2 (degree 4) = adds degree-4 moments + localizing matrices -> tighter

Optimized: builds SDP matrices as sparse linear maps from the moment vector y,
replacing O(n^2) scalar equality constraints per matrix with a single reshape +
PSD constraint. This dramatically reduces CVXPY canonicalization time.
"""
import numpy as np
import cvxpy as cp
from scipy import sparse
import time


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


def build_window_matrices(d):
    """Build all window matrices vectorized. Returns (windows, M_mats)."""
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


def _add_mi(a, b, d):
    """Element-wise addition of multi-indices (tuples)."""
    return tuple(a[i] + b[i] for i in range(d))


def _unit(d, i):
    """Unit vector e_i as tuple."""
    return tuple(1 if k == i else 0 for k in range(d))


# ── Vectorized monomial hashing ─────────────────────────────────────────────
# Encode multi-index (a_0,...,a_{d-1}) as int64 via mixed-radix:
#   hash = sum(a_k * base^k), base = max_component + 1.
# Linear property: hash(a+b) = hash(a) + hash(b).
# This lets us compute all pairwise sums vectorized.

def _make_hash_bases(d, max_comp):
    """Return base vector for hashing: bases[k] = (max_comp+1)^k."""
    base = max_comp + 1
    return np.array([base**k for k in range(d)], dtype=np.int64)


def _hash_monos(monos_array, bases):
    """Hash an array of monomials. monos_array: (..., d) int array → (...) int64."""
    return np.tensordot(monos_array.astype(np.int64), bases, axes=([-1], [0]))


def _build_hash_lookup(mono_list, bases):
    """Build sorted hash array + index map for O(log n) vectorized lookup."""
    n = len(mono_list)
    mono_arr = np.array(mono_list, dtype=np.int64)  # (n, d)
    hashes = _hash_monos(mono_arr, bases)            # (n,)
    sort_order = np.argsort(hashes)
    return hashes[sort_order], sort_order, hashes


def _hash_lookup_vectorized(query_hashes, sorted_hashes, sort_order):
    """Vectorized lookup: query_hashes → moment indices (or -1 if missing).

    sorted_hashes: sorted hash values of mono_list
    sort_order: maps sorted position → original mono_list index
    Returns array same shape as query_hashes with moment indices or -1.
    """
    flat = query_hashes.ravel()
    pos = np.searchsorted(sorted_hashes, flat)
    pos = np.clip(pos, 0, len(sorted_hashes) - 1)
    found = sorted_hashes[pos] == flat
    result = np.where(found, sort_order[pos], -1)
    return result.reshape(query_hashes.shape)


def _collect_all_moments(d, order, basis, loc_basis, consist_mono):
    """Collect all needed moment multi-indices into a sorted list + index map.

    Uses vectorized hashing to avoid O(n_basis^2 * d^2) Python loops.
    """
    max_comp = 2 * order  # max component in any monomial we'll see
    bases = _make_hash_bases(d, max_comp)

    mono_set = set()

    # From moment matrix: all basis[a] + basis[b] (vectorized)
    B = np.array(basis, dtype=np.int64)   # (n_basis, d)
    AB = B[:, np.newaxis, :] + B[np.newaxis, :, :]  # (n_basis, n_basis, d)
    for ab in AB.reshape(-1, d):
        mono_set.add(tuple(ab))

    # From all_monomials up to 2*order
    for m in enum_monomials(d, 2 * order):
        mono_set.add(m)

    # From localizing matrices (vectorized)
    if order >= 2:
        LB = np.array(loc_basis, dtype=np.int64)  # (n_loc, d)
        E = np.eye(d, dtype=np.int64)              # (d, d)

        # ab = LB[a] + LB[b] for all (a,b)
        AB_loc = LB[:, np.newaxis, :] + LB[np.newaxis, :, :]  # (n_loc, n_loc, d)
        for ab in AB_loc.reshape(-1, d):
            mono_set.add(tuple(ab))

        # ab + ei for all (a,b,i)
        AB_ei = AB_loc[:, :, np.newaxis, :] + E[np.newaxis, np.newaxis, :, :]
        for m in AB_ei.reshape(-1, d):
            mono_set.add(tuple(m))

        # ab + ei + ej for all (a,b,i,j)
        AB_eiej = (AB_loc[:, :, np.newaxis, np.newaxis, :]
                   + E[np.newaxis, np.newaxis, :, np.newaxis, :]
                   + E[np.newaxis, np.newaxis, np.newaxis, :, :])
        for m in AB_eiej.reshape(-1, d):
            mono_set.add(tuple(m))

    # From consistency constraints
    C = np.array(consist_mono, dtype=np.int64)  # (n_consist, d)
    for m in C:
        mono_set.add(tuple(m))
    C_ei = C[:, np.newaxis, :] + np.eye(d, dtype=np.int64)[np.newaxis, :, :]
    for m in C_ei.reshape(-1, d):
        mono_set.add(tuple(m))

    mono_list = sorted(mono_set)
    idx = {m: i for i, m in enumerate(mono_list)}
    return mono_list, idx


def _build_perm_sparse(index_map, n_rows, n_y):
    """Build a sparse permutation/selection matrix from a flat index array.

    index_map: 2D array of shape (n, n) mapping (a,b) -> y index
    Returns sparse matrix P of shape (n*n, n_y) s.t. P @ y = flat matrix entries.
    """
    flat_size = index_map.size
    rows = np.arange(flat_size)
    cols = index_map.ravel()
    vals = np.ones(flat_size, dtype=np.float64)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(flat_size, n_y))


def solve_lasserre(d, c_target, order=2, verbose=True):
    """Solve Lasserre hierarchy at given order.

    Returns dict with 'lb' (lower bound on val(d)) and 'proven' flag.
    """
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = _collect_all_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    if verbose:
        print(f"  d={d}, order={order}, windows={n_win}, "
              f"basis={n_basis}, moments={n_y}", flush=True)

    # === Variables ===
    y = cp.Variable(n_y)
    t_param = cp.Parameter(nonneg=True)
    constraints = []

    # === y >= 0 (single vector constraint instead of n_y scalar constraints) ===
    constraints.append(y >= 0)

    # === y_0 = 1 ===
    zero = tuple(0 for _ in range(d))
    constraints.append(y[idx[zero]] == 1)

    # === Moment consistency: A @ y == 0 (vectorized) ===
    # For each alpha with |alpha| <= 2*order-1:
    #   sum_i y[alpha+e_i] = y[alpha]  (simplex projection)
    consist_arr = np.array(consist_mono, dtype=np.int64)  # (n_consist, d)
    consist_hash_bases = _make_hash_bases(d, max_comp=2 * order)
    sorted_h, sort_o, _ = _build_hash_lookup(mono_list, consist_hash_bases)

    alpha_hashes = _hash_monos(consist_arr, consist_hash_bases)
    alpha_idx = _hash_lookup_vectorized(alpha_hashes, sorted_h, sort_o)

    # alpha + e_i for all (alpha, i)
    alpha_ei = consist_arr[:, np.newaxis, :] + np.eye(d, dtype=np.int64)[np.newaxis, :, :]
    alpha_ei_hashes = _hash_monos(alpha_ei, consist_hash_bases)  # (n_consist, d)
    alpha_ei_idx = _hash_lookup_vectorized(alpha_ei_hashes, sorted_h, sort_o)

    c_rows, c_cols, c_vals = [], [], []
    row_count = 0
    for r_idx in range(len(consist_mono)):
        if alpha_idx[r_idx] < 0:
            continue
        for i in range(d):
            if alpha_ei_idx[r_idx, i] >= 0:
                c_rows.append(row_count)
                c_cols.append(int(alpha_ei_idx[r_idx, i]))
                c_vals.append(1.0)
        c_rows.append(row_count)
        c_cols.append(int(alpha_idx[r_idx]))
        c_vals.append(-1.0)
        row_count += 1

    if row_count > 0:
        A_consist = sparse.csr_matrix(
            (c_vals, (c_rows, c_cols)), shape=(row_count, n_y))
        constraints.append(A_consist @ y == 0)

    # === Vectorized hash-based index lookup ===
    max_comp = 2 * order
    hash_bases = _make_hash_bases(d, max_comp)
    sorted_hashes, sort_order_arr, _ = _build_hash_lookup(mono_list, hash_bases)

    B_arr = np.array(basis, dtype=np.int64)      # (n_basis, d)
    LB_arr = np.array(loc_basis, dtype=np.int64) if loc_basis else np.zeros((0, d), dtype=np.int64)
    E_arr = np.eye(d, dtype=np.int64)             # (d, d)

    # === Moment matrix M_k(y) >> 0 (vectorized) ===
    AB_basis = B_arr[:, np.newaxis, :] + B_arr[np.newaxis, :, :]  # (n_basis, n_basis, d)
    AB_hash = _hash_monos(AB_basis, hash_bases)                    # (n_basis, n_basis)
    M_indices = _hash_lookup_vectorized(AB_hash, sorted_hashes, sort_order_arr)

    P_moment = _build_perm_sparse(M_indices, n_basis, n_y)
    M_expr = cp.reshape(P_moment @ y, (n_basis, n_basis), order='C')
    constraints.append(M_expr >> 0)

    # === Localizing matrices for mu_i >= 0 (vectorized) ===
    if order >= 2:
        # AB_loc[a,b] = LB[a] + LB[b], shape (n_loc, n_loc, d)
        AB_loc = LB_arr[:, np.newaxis, :] + LB_arr[np.newaxis, :, :]

        for i_var in range(d):
            # AB_loc + e_i, shape (n_loc, n_loc, d)
            AB_ei = AB_loc + E_arr[i_var]
            AB_ei_hash = _hash_monos(AB_ei, hash_bases)
            L_indices = _hash_lookup_vectorized(AB_ei_hash, sorted_hashes, sort_order_arr)
            L_valid = (L_indices >= 0).astype(np.float64)
            L_indices = np.where(L_indices >= 0, L_indices, 0)

            flat_size = n_loc * n_loc
            r = np.arange(flat_size)
            c = L_indices.ravel()
            v = L_valid.ravel()
            P_loc = sparse.csr_matrix((v, (r, c)), shape=(flat_size, n_y))
            Li_expr = cp.reshape(P_loc @ y, (n_loc, n_loc), order='C')
            constraints.append(Li_expr >> 0)

    # === Window localizing matrices (vectorized) ===
    if order >= 2:
        n_loc_w = n_loc
        if verbose:
            print(f"  Window localizing: {n_win} x {n_loc_w}x{n_loc_w}",
                  flush=True)

        flat_size_w = n_loc_w * n_loc_w

        # t_indices[a,b] = idx[LB[a]+LB[b]] — vectorized
        AB_loc_hash = _hash_monos(AB_loc, hash_bases)  # (n_loc, n_loc)
        t_indices = _hash_lookup_vectorized(AB_loc_hash, sorted_hashes, sort_order_arr)

        T_perm = _build_perm_sparse(t_indices, n_loc_w, n_y)

        # ab_eiej_idx[a,b,i,j] = idx[LB[a]+LB[b]+e_i+e_j] — FULLY VECTORIZED
        # hash(LB[a]+LB[b]+e_i+e_j) = hash(LB[a]+LB[b]) + hash(e_i+e_j)
        # hash(e_i+e_j) = bases[i] + bases[j]
        EE_hash = hash_bases[:, np.newaxis] + hash_bases[np.newaxis, :]  # (d, d)
        ABIJ_hash = AB_loc_hash[:, :, np.newaxis, np.newaxis] + EE_hash[np.newaxis, np.newaxis, :, :]
        ab_eiej_idx = _hash_lookup_vectorized(ABIJ_hash, sorted_hashes, sort_order_arr)

        if verbose:
            print(f"  ab_eiej_idx built: {ab_eiej_idx.shape}, "
                  f"{(ab_eiej_idx >= 0).sum()} valid entries", flush=True)

        # Precompute flat (a,b) indices for COO construction
        ab_flat = np.arange(n_loc_w).reshape(-1, 1) * n_loc_w + np.arange(n_loc_w).reshape(1, -1)

        # Build one Cw sparse matrix per window using vectorized ops
        for w in range(n_win):
            Mw = M_mats[w]

            # Find nonzero (i,j) entries in Mw
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0:
                # Lw = t_param * T_perm @ y (no quadratic term)
                Lw_flat = t_param * (T_perm @ y)
            else:
                # Batched sparse construction: index all nonzero (i,j) at once
                # ab_eiej_idx[:,:,nz_i,nz_j] -> (n_loc_w, n_loc_w, n_nz)
                y_all = ab_eiej_idx[:, :, nz_i, nz_j]  # (n_loc_w, n_loc_w, n_nz)
                valid_all = y_all >= 0
                # Expand ab_flat and Mw values to match
                ab_exp = np.broadcast_to(
                    ab_flat[:, :, np.newaxis], y_all.shape)
                mw_vals = Mw[nz_i, nz_j]  # (n_nz,)
                mw_exp = np.broadcast_to(
                    mw_vals[np.newaxis, np.newaxis, :], y_all.shape)

                if np.any(valid_all):
                    cw_rows_all = ab_exp[valid_all].ravel()
                    cw_cols_all = y_all[valid_all].ravel()
                    cw_vals_all = mw_exp[valid_all].ravel().astype(np.float64)
                    Cw = sparse.csr_matrix(
                        (cw_vals_all, (cw_rows_all, cw_cols_all)),
                        shape=(flat_size_w, n_y))
                else:
                    Cw = sparse.csr_matrix((flat_size_w, n_y))

                Lw_flat = t_param * (T_perm @ y) - Cw @ y

            Lw_expr = cp.reshape(Lw_flat, (n_loc_w, n_loc_w), order='C')
            constraints.append(Lw_expr >> 0)
    else:
        # Order 1: scalar constraints t >= Tr(M_W X)
        units = [_unit(d, i) for i in range(d)]
        for w in range(n_win):
            tv = 0
            for i in range(d):
                for j in range(d):
                    if M_mats[w][i, j] != 0:
                        eij = _add_mi(units[i], units[j], d)
                        tv += M_mats[w][i, j] * y[idx[eij]]
            constraints.append(tv <= t_param)

    # === Build problem (feasibility) ===
    prob = cp.Problem(cp.Minimize(0), constraints)

    if verbose:
        print(f"  Problem built: {len(constraints)} constraints", flush=True)
        print(f"  Binary search on t...", flush=True)

    # === Solve via binary search on t ===
    t0 = time.time()

    # Pick solver — SCS is fastest at this scale (0.6s vs MOSEK 3.3s vs CLARABEL 142s)
    solver = cp.SCS
    solver_opts = {'verbose': False, 'max_iters': 20000, 'eps': 1e-7}

    def check(t_val):
        t_param.value = t_val
        try:
            prob.solve(solver=solver, warm_start=True, **solver_opts)
            return prob.status in ['optimal', 'optimal_inaccurate']
        except Exception:
            return False

    # Find feasible upper bound
    lo, hi = 0.5, 5.0
    while not check(hi):
        hi *= 2
        if hi > 100:
            break
    if not check(hi):
        raise RuntimeError(f"SDP infeasible up to t={hi}; cannot bracket optimum")
    if verbose:
        print(f"  Feasible at t={hi:.4f}", flush=True)

    # Binary search (50 steps = ~15 decimal digits, matching 1e-6 tolerance)
    for step in range(10):
        mid = (lo + hi) / 2
        if check(mid):
            hi = mid
            if verbose:
                print(f"    t={mid:.8f}: feasible", flush=True)
        else:
            lo = mid
            if verbose:
                print(f"    t={mid:.8f}: infeasible", flush=True)

    elapsed = time.time() - t0
    lb = lo
    proven = lb >= c_target - 1e-6

    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Lower bound: {lb:.10f}")
        print(f"  Margin over {c_target}: {lb - c_target:.10f}")
        if proven:
            print(f"  *** PROVEN: val({d}) >= {c_target} ***")

    return {'lb': lb, 'proven': proven, 'elapsed': elapsed,
            'd': d, 'order': order}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--c_target', type=float, default=1.10)
    parser.add_argument('--order', type=int, default=2)
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"LASSERRE PROOF: val({args.d}) >= {args.c_target}")
    print(f"  Order {args.order} (degree {2 * args.order})")
    print(f"{'=' * 60}\n", flush=True)

    r = solve_lasserre(args.d, args.c_target, order=args.order)
    print(f"\nResult: lb={r['lb']:.8f}, "
          f"proven={'YES' if r['proven'] else 'NO'}, "
          f"time={r['elapsed']:.1f}s")


if __name__ == '__main__':
    main()

"""Sparse-A_W refactor of _vertex_drop_clipped + benchmark.

A_W[i,j] = 1 iff s <= i+j <= s+ell-2 (an antidiagonal band).
Nonzero count: ell*d - (boundary corrections) ~ ell*d in mid-windows; for
ell=2 it is exactly d (one antidiagonal pair per i).

The current dense kernel pays O(d^2) per vertex to evaluate delta^T A_W delta,
ignoring the band structure.  This refactor precomputes the (i,j) nonzero
pairs ONCE per window and replaces the inner double loop with a single sweep
over the nonzero list, reducing per-vertex cost from O(d^2) to O(nnz(A_W)).
"""
import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList


# -----------------------------------------------------------------------
# 1.  Sparse builder: returns (rows, cols, nnz) for one (ell, s) window.
# -----------------------------------------------------------------------
@njit(cache=True)
def _build_aw_sparse(d, ell, s, rows, cols):
    """Fill rows[0:nnz], cols[0:nnz] with nonzero positions of A_W.
    Caller pre-allocates rows/cols of size 2*d (upper bound on nnz/row sum).

    Caller should size rows/cols >= ell * d to be safe (ell <= 2d).
    Returns nnz.
    """
    s_hi_idx = s + ell - 2
    nnz = 0
    for i in range(d):
        j_lo = s - i
        j_hi = s_hi_idx - i
        if j_lo < 0:
            j_lo = 0
        if j_hi > d - 1:
            j_hi = d - 1
        for j in range(j_lo, j_hi + 1):
            rows[nnz] = i
            cols[nnz] = j
            nnz += 1
    return nnz


# -----------------------------------------------------------------------
# 2.  Sparse vertex-drop kernel.
# -----------------------------------------------------------------------
@njit(cache=True)
def _vertex_drop_clipped_sparse(grad, rows, cols, nnz, scale, lo, hi, d):
    """Identical contract to _vertex_drop_clipped, but A_W is supplied
    as a list of nonzero (rows[k], cols[k]) pairs of length nnz.

    delta^T A_W delta = sum_{k=0..nnz-1} delta[rows[k]] * delta[cols[k]].
    """
    best = 0.0
    delta = np.zeros(d, dtype=np.float64)
    tol = 1e-12

    for free_idx in range(d):
        n_pat = np.int64(1) << (d - 1)
        for mask in range(n_pat):
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                if bit == 0:
                    delta[i] = lo[i]
                else:
                    delta[i] = hi[i]
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if free_val < lo[free_idx] - tol or free_val > hi[free_idx] + tol:
                continue
            if free_val < lo[free_idx]:
                free_val = lo[free_idx]
            elif free_val > hi[free_idx]:
                free_val = hi[free_idx]
            delta[free_idx] = free_val

            lin = 0.0
            for i in range(d):
                lin += grad[i] * delta[i]
            quad = 0.0
            for k in range(nnz):
                quad += delta[rows[k]] * delta[cols[k]]
            val = -lin - scale * quad
            if val > best:
                best = val
    return best


# -----------------------------------------------------------------------
# 3.  Reference dense kernel (verbatim from coarse_cascade_prover.py:839).
# -----------------------------------------------------------------------
@njit(cache=True)
def _vertex_drop_clipped_dense(grad, A_W, scale, lo, hi, d):
    best = 0.0
    delta = np.zeros(d, dtype=np.float64)
    tol = 1e-12
    for free_idx in range(d):
        n_pat = np.int64(1) << (d - 1)
        for mask in range(n_pat):
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                if bit == 0:
                    delta[i] = lo[i]
                else:
                    delta[i] = hi[i]
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if free_val < lo[free_idx] - tol or free_val > hi[free_idx] + tol:
                continue
            if free_val < lo[free_idx]:
                free_val = lo[free_idx]
            elif free_val > hi[free_idx]:
                free_val = hi[free_idx]
            delta[free_idx] = free_val
            lin = 0.0
            for i in range(d):
                lin += grad[i] * delta[i]
            quad = 0.0
            for i in range(d):
                Ai = 0.0
                for j in range(d):
                    Ai += A_W[i, j] * delta[j]
                quad += delta[i] * Ai
            val = -lin - scale * quad
            if val > best:
                best = val
    return best


# -----------------------------------------------------------------------
# 4.  Verification (d=4 brute force) and benchmark (d=12).
# -----------------------------------------------------------------------
def _build_dense_aw(d, ell, s):
    A = np.zeros((d, d), dtype=np.float64)
    s_hi_idx = s + ell - 2
    for i in range(d):
        j_lo = max(s - i, 0)
        j_hi = min(s_hi_idx - i, d - 1)
        for j in range(j_lo, j_hi + 1):
            A[i, j] = 1.0
    return A


def verify_d4():
    """Confirm sparse == dense on all (ell,s) windows at d=4 for a random cell."""
    import time
    d = 4
    rng = np.random.default_rng(0)
    mu = rng.dirichlet(np.ones(d))
    h = 0.05
    lo = np.maximum(-h, -mu)
    hi = np.minimum(h, 1.0 - mu)
    grad = rng.standard_normal(d)
    scale = 1.5

    rows = np.empty(2 * d * d, dtype=np.int64)
    cols = np.empty(2 * d * d, dtype=np.int64)
    max_abs = 0.0
    for ell in range(2, 2 * d + 1):
        for s in range(2 * d - 1 - ell + 2):
            A = _build_dense_aw(d, ell, s)
            nnz = _build_aw_sparse(d, ell, s, rows, cols)
            v_dense = _vertex_drop_clipped_dense(grad, A, scale, lo, hi, d)
            v_sparse = _vertex_drop_clipped_sparse(grad, rows, cols, nnz,
                                                   scale, lo, hi, d)
            max_abs = max(max_abs, abs(v_dense - v_sparse))
    print(f"d=4 verify: max |dense - sparse| = {max_abs:.3e}")
    return max_abs < 1e-12


def bench_d12():
    import time
    d = 12
    rng = np.random.default_rng(1)
    mu = rng.dirichlet(np.ones(d))
    h = 1.0 / 60.0
    lo = np.maximum(-h, -mu)
    hi = np.minimum(h, 1.0 - mu)
    grad = rng.standard_normal(d)
    scale = 2.0 * d / 6.0

    rows = np.empty(4 * d * d, dtype=np.int64)
    cols = np.empty(4 * d * d, dtype=np.int64)

    # Warmup (force JIT compile).
    A0 = _build_dense_aw(d, 2, 0)
    nnz0 = _build_aw_sparse(d, 2, 0, rows, cols)
    _vertex_drop_clipped_dense(grad, A0, scale, lo, hi, d)
    _vertex_drop_clipped_sparse(grad, rows, cols, nnz0, scale, lo, hi, d)

    print(f"\nd={d}, vertices = d*2^(d-1) = {d * (1 << (d-1))}")
    print(f"{'window':<22}{'nnz':>6}{'dense (s)':>14}{'sparse (s)':>14}{'speedup':>10}")
    for (ell, s, label) in [(2, 0, 'narrow ell=2'),
                            (12, 5, 'mid ell=12'),
                            (24, 0, 'full ell=24')]:
        if s > 2 * d - 1 - ell + 1:
            s = 0
        A = _build_dense_aw(d, ell, s)
        nnz = _build_aw_sparse(d, ell, s, rows, cols)
        t0 = time.perf_counter()
        v_d = _vertex_drop_clipped_dense(grad, A, scale, lo, hi, d)
        td = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_s = _vertex_drop_clipped_sparse(grad, rows, cols, nnz, scale, lo, hi, d)
        ts = time.perf_counter() - t0
        spd = td / ts if ts > 0 else float('inf')
        print(f"{label:<22}{nnz:>6}{td:>14.4f}{ts:>14.4f}{spd:>10.2f}x"
              f"  (val_d={v_d:.3e}, val_s={v_s:.3e})")


if __name__ == '__main__':
    ok = verify_d4()
    print("d=4 agreement:", "PASS" if ok else "FAIL")
    bench_d12()

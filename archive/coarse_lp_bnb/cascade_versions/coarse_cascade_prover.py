"""Coarse cascade prover for C_{1a} >= c_target (NO correction term).

Mathematical basis:
  For any nonneg f on [-1/4,1/4] with integral 1, partitioned into d bins:
    max(f*f) >= max_W TV_W(mu)
  where mu_i = integral of f over bin i (NO step-function approximation).

  By refinement monotonicity (empirically verified):
    if parent at d bins has max TV >= c, all children at 2d bins also do.

  So the cascade can prune parents without correction.

Algorithm:
  1. L0: enumerate all compositions of S into d_start parts, prune by TV >= c.
  2. L1..LK: for each survivor, split each bin into 2 children, prune.
  3. Box certification: for each pruned cell, QP-verify the Voronoi box.
  4. If 0 survivors at level K and all boxes certified: C_{1a} >= c. QED.

Grid: absolute mass quantum delta = 1/S.  Integer masses c_i sum to S.
  TV_W(ell,s) = (2d/ell) * sum_{k=s}^{s+ell-2} conv[k] / S^2
  Prune if ws_int > floor(c * ell * S^2 / (2d))

Usage:
  python coarse_cascade_prover.py
  python coarse_cascade_prover.py --c_target 1.30 --S 50
  python coarse_cascade_prover.py --c_target 1.28 --S 30 --d_start 2
"""
import argparse
import time
import os
import sys

import numpy as np
import numba
from numba import njit, prange

# §1: import canonical composition enumerator from sibling module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CS_DIR = os.path.join(_THIS_DIR, "cloninger-steinerberger")
if _CS_DIR not in sys.path:
    sys.path.insert(0, _CS_DIR)
from compositions import generate_canonical_compositions_batched  # noqa: E402


# =====================================================================
# Threshold computation
# =====================================================================

def compute_thresholds(c_target, S, d):
    """Precompute per-ell integer thresholds for dimension d.

    Prune if ws_int > thr[ell].
    Equivalent to TV >= c_target (approximately; sound direction).
    """
    max_ell = 2 * d
    thr = np.empty(max_ell + 1, dtype=np.int64)
    S2 = np.float64(S) * np.float64(S)
    two_d = np.float64(2 * d)
    for ell in range(2, max_ell + 1):
        # TV = 2d/(ell*S^2) * ws.  Prune if TV >= c_target.
        # ws >= c_target * ell * S^2 / (2d).
        # So prune if ws > floor(c_target * ell * S^2 / (2d) - eps).
        thr[ell] = np.int64(c_target * np.float64(ell) * S2 / two_d - 1e-9)
    return thr


def compute_xcap(c_target, S, d):
    """Max integer mass per bin before self-convolution alone prunes it.

    Single-bin self-conv: TV >= d * (c/S)^2 * 2 (from ell=2 self-window).
    Actually: conv[2i] = c^2, TV(ell=2, s=2i) = 2d/2 * c^2/S^2 = d*c^2/S^2.
    Prune if d*c^2/S^2 >= c_target => c >= S*sqrt(c_target/d).
    """
    return int(np.floor(S * np.sqrt(c_target / d)))


def count_lattice_offenders(c_target, S, d_max, tol=1e-9):
    """Diagnostic: for which (d, ell) is c_target * ell * S^2 / (2d) integer?

    These are the windows where the smallest pruned ws gives TV = c_target
    exactly (margin = 0). Box-cert can still PASS on such a cell if SOME
    OTHER window certifies it; this only flags potentially-marginal cases.

    Returns list of (d, ell, v) triples on the lattice.
    """
    offenders = []
    for d in range(2, d_max + 1):
        for ell in range(2, 2 * d + 1):
            v = c_target * ell * S * S / (2.0 * d)
            if abs(v - round(v)) < tol:
                offenders.append((d, ell, float(v)))
    return offenders


def s_shift_safe(c_target, S, d_max, max_search=200, tol=1e-9, strict=False):
    """Return smallest S' >= S minimizing the number of (d, ell) lattice hits.

    WARNING: for rational c_target with small denominator (e.g. 1.20=6/5,
    1.25=5/4, 1.28=32/25), MANY (d, ell) are on the lattice for EVERY S
    (e.g., d=3, ell=5, c=6/5 gives v = S^2 always integer). Strict
    avoidance of all lattice hits is impossible.

    With strict=True (default False): raise if no fully-safe S exists.
    With strict=False: return the S in [S, S+max_search] with the FEWEST
    lattice hits (if S itself has any hits), else S unchanged.
    """
    base_offenders = count_lattice_offenders(c_target, S, d_max, tol)
    if not base_offenders:
        return S
    best_S = S
    best_count = len(base_offenders)
    for S_try in range(S, S + max_search + 1):
        n = len(count_lattice_offenders(c_target, S_try, d_max, tol))
        if n == 0:
            return S_try
        if n < best_count:
            best_count = n
            best_S = S_try
    if strict:
        raise RuntimeError(
            f"No fully off-lattice S in [{S}, {S+max_search}] for "
            f"c_target={c_target}, d_max={d_max}. Best: S={best_S} "
            f"with {best_count} offenders.")
    return best_S


# =====================================================================
# Speedup #2: Spectral quad_drop precompute (Toeplitz/Hankel structure)
# =====================================================================
# A_W is the symmetric 0/1 matrix with (A_W)_ij = 1[s <= i+j <= s+ell-2].
# For sum-zero perturbations delta in V = {1}^perp:
#     delta^T A_W delta >= lambda_min^V(A_W) * ||delta||_2^2
# where lambda_min^V is the smallest eigenvalue of A_W restricted to V.
# Setting rho_W = max(0, -lambda_min^V(A_W)), this gives the SOUND bound
#     min_{delta in cell} delta^T A_W delta  >=  -rho_W * d * h^2
# which replaces the loose row-sum bound (ell-1) used in qd_c. Empirically
# rho_W is 3-10x smaller than ell-1 for ell >= 4, so the quadratic drop
# tightens by the same factor — capture rate jumps especially at small S.

_QDROP_TABLE_CACHE = {}
_EIGEN_TABLE_CACHE = {}


def _build_AW(d, ell, s):
    """Build A_W = symmetric 0/1 indicator (A_W)_ij = 1[s <= i+j <= s+ell-2]."""
    A = np.zeros((d, d), dtype=np.float64)
    s_hi_off = ell - 2
    for i in range(d):
        jl = s - i
        jh = s + s_hi_off - i
        if jl < 0:
            jl = 0
        if jh > d - 1:
            jh = d - 1
        for j in range(jl, jh + 1):
            A[i, j] = 1.0
    return A


def compute_qdrop_table(d):
    """Returns table Q[ell, s] = max(0, -lambda_min^V(A_W(ell,s,d))).

    Used as a tighter replacement for max_row(A_W) = ell-1 in the quad_drop
    bound of _phase1_lipschitz_lb_spec. Computed once per d via numpy.linalg.eigvalsh
    on the projected matrix P A_W P (P = I - 11^T/d, the centering projector).
    """
    if d in _QDROP_TABLE_CACHE:
        return _QDROP_TABLE_CACHE[d]
    conv_len = 2 * d - 1
    Q = np.zeros((2 * d + 1, conv_len), dtype=np.float64)
    P = np.eye(d) - np.ones((d, d)) / np.float64(d)
    for ell in range(2, 2 * d + 1):
        for s in range(conv_len - ell + 2):
            A = _build_AW(d, ell, s)
            tilde_A = P @ A @ P
            eigs = np.linalg.eigvalsh(tilde_A)
            lam_min = float(eigs.min())
            Q[ell, s] = max(0.0, -lam_min)
    _QDROP_TABLE_CACHE[d] = Q
    return Q


def compute_active_hessian(mu_star, d, val_d, tol=1e-3):
    """Given val(d) optimum mu_star, compute the KKT-weighted active Hessian
    H = sum_{W in A^*} alpha_W^* * 2*A_W where A^* = {W : TV_W(mu_star) close to val_d}
    and alpha_W^* are KKT multipliers (sum to 1, nonneg).

    The KKT stationarity at mu_star (interior simplex) gives
        sum_W alpha_W * grad(TV_W)(mu_star) = lambda * 1  (mod boundary)
    where grad(TV_W) = 2*A_W*mu_star. We solve via least-squares LP
    over alpha in simplex of A*, lambda free.

    Returns (H, alpha_star, active_indices, residual_norm).
    """
    from scipy.optimize import linprog, minimize
    conv_len = 2 * d - 1
    # Step 1: compute TV_W(mu_star) for all windows; find active set
    tv_per_window = []
    for ell in range(2, 2 * d + 1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            A = _build_AW(d, ell, s)
            tv_W = scale * float(mu_star @ A @ mu_star)
            tv_per_window.append((tv_W, ell, s, A, scale))
    tv_per_window.sort(key=lambda t: -t[0])
    val_max = tv_per_window[0][0]
    active = [(tv_W, ell, s, A, scale)
              for (tv_W, ell, s, A, scale) in tv_per_window
              if tv_W >= val_max - tol]
    n_active = len(active)
    if n_active == 0:
        return np.zeros((d, d)), np.zeros(0), [], 1.0

    # Step 2: build gradient stack G[k] = 2*scale_k * A_k @ mu_star, k=0..n_active-1
    grads = np.array([2.0 * a[4] * (a[3] @ mu_star) for a in active])  # shape (n_active, d)

    # Step 3: solve KKT LP
    # min ||sum_k alpha_k * grad_k - lambda * 1 - beta||^2
    # s.t. sum alpha_k = 1, alpha_k >= 0, beta_i >= 0, beta_i = 0 if mu_star_i > tol_active
    # Use scipy minimize on alpha (simplex projection inside).
    boundary = mu_star < tol  # bins where mu_star is ~0; beta_i can be nonzero
    n_bnd = int(boundary.sum())

    def obj(params):
        alpha = params[:n_active]
        lam = params[n_active]
        beta_b = params[n_active + 1:n_active + 1 + n_bnd]
        # Project alpha onto simplex
        alpha = np.maximum(alpha, 0)
        s = alpha.sum()
        if s < 1e-12:
            alpha = np.ones(n_active) / n_active
        else:
            alpha = alpha / s
        beta_full = np.zeros(d)
        beta_full[boundary] = np.maximum(beta_b, 0)
        residual = (alpha @ grads) - lam * np.ones(d) - beta_full
        return float(residual @ residual)

    x0 = np.zeros(n_active + 1 + n_bnd)
    x0[:n_active] = 1.0 / n_active
    x0[n_active] = float(np.mean(grads.mean(axis=0)))
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12})
    alpha_raw = res.x[:n_active]
    alpha_raw = np.maximum(alpha_raw, 0)
    if alpha_raw.sum() < 1e-12:
        alpha_star = np.ones(n_active) / n_active
    else:
        alpha_star = alpha_raw / alpha_raw.sum()
    residual_norm = float(np.sqrt(res.fun))

    # Step 4: build H = sum alpha_k * 2 * scale_k * A_k
    H = np.zeros((d, d), dtype=np.float64)
    active_indices = []
    for k, (tv_W, ell, s, A, scale) in enumerate(active):
        H += alpha_star[k] * 2.0 * scale * A
        active_indices.append((ell, s))
    return H, alpha_star, active_indices, residual_norm


@njit(cache=True)
def _cell_in_tube(mu_c, mu_star, H, R_sq_eff, d):
    """Test if (mu_c - mu_star)^T H (mu_c - mu_star) <= R_sq_eff.

    R_sq_eff = R_tube^2 + slack for cell radius and Lipschitz, where R_tube^2
    is the Lojasiewicz tube radius (= 2*(val_d - c) if no boundary).
    Cells satisfying this need fine treatment; cells outside are certified
    by the continuous Lojasiewicz LB (max_W TV_W(mu) >= val_d + (mu-mu*)^T H (mu-mu*)/2 > c).
    """
    z_H_z = 0.0
    # Compute z = mu_c - mu_star, then z^T H z = sum_ij H[i,j] z_i z_j
    # Done in O(d^2) per cell.
    for i in range(d):
        zi = mu_c[i] - mu_star[i]
        Hzi = 0.0
        for j in range(d):
            zj = mu_c[j] - mu_star[j]
            Hzi += H[i, j] * zj
        z_H_z += zi * Hzi
    return z_H_z <= R_sq_eff


@njit(parallel=True, cache=True)
def _tube_filter_batch(batch_int, mu_star, H, R_sq_eff, d, S, keep_mask):
    """For a batch of integer compositions, mark which ones are IN the tube.
    keep_mask[b] = 1 iff batch_int[b]/S is in the tube ellipsoid."""
    B = batch_int.shape[0]
    inv_S = 1.0 / float(S)
    for b in prange(B):
        z_H_z = 0.0
        for i in range(d):
            zi = float(batch_int[b, i]) * inv_S - mu_star[i]
            Hzi = 0.0
            for j in range(d):
                zj = float(batch_int[b, j]) * inv_S - mu_star[j]
                Hzi += H[i, j] * zj
            z_H_z += zi * Hzi
        keep_mask[b] = 1 if z_H_z <= R_sq_eff else 0


def find_mu_star_local(d, x_cap_frac=None, n_restarts=200, verbose=False):
    """Find val(d) minimizer mu* via Nelder-Mead multistart (NO SDP).

    Returns (val_d, mu_star). Forwards to lasserre/track1_val_d_finf.py.
    """
    sys.path.insert(0, os.path.join(_THIS_DIR, "lasserre"))
    from track1_val_d_finf import compute_val_d_with_mu  # noqa: E402
    val_d, mu_star = compute_val_d_with_mu(d, n_restarts=n_restarts, verbose=verbose)
    return val_d, mu_star


def compute_window_eigen_table(d):
    """Returns (V_table, lam_table, valid_mask) for trust-region cell-min QP.

      V_table[ell, s, :, :]   — orthonormal eigenvectors of P A_W(ell,s) P (d x d)
      lam_table[ell, s, :]    — eigenvalues sorted ascending (length d)
      valid_mask[ell, s]      — 1 if window is in range, else 0

    The kernel direction (eigenvalue ~ 0, eigenvector ~ 1/sqrt(d)) is always
    present because P annihilates the all-ones direction. Trust-region QP
    in V uses ALL d eigenvectors but the kernel contributes c_kernel = 0
    (since centered gradient has mean 0).
    """
    if d in _EIGEN_TABLE_CACHE:
        return _EIGEN_TABLE_CACHE[d]
    conv_len = 2 * d - 1
    V_table = np.zeros((2 * d + 1, conv_len, d, d), dtype=np.float64)
    lam_table = np.zeros((2 * d + 1, conv_len, d), dtype=np.float64)
    valid_mask = np.zeros((2 * d + 1, conv_len), dtype=np.int64)
    P = np.eye(d) - np.ones((d, d)) / np.float64(d)
    for ell in range(2, 2 * d + 1):
        for s in range(conv_len - ell + 2):
            A = _build_AW(d, ell, s)
            tilde_A = P @ A @ P
            lam, V = np.linalg.eigh(tilde_A)
            V_table[ell, s] = V
            lam_table[ell, s] = lam
            valid_mask[ell, s] = 1
    _EIGEN_TABLE_CACHE[d] = (V_table, lam_table, valid_mask)
    return V_table, lam_table, valid_mask


# =====================================================================
# L0: Branch-and-bound for initial compositions
# =====================================================================

@njit(cache=True)
def _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf, count_only):
    """BnB subtree with bins[0]=c0 fixed.

    Returns (n_survivors, n_tested).
    """
    conv_len = 2 * d - 1
    d_m1 = d - 1
    max_ell = 2 * d

    conv = np.zeros(conv_len, dtype=np.int64)
    bins = np.zeros(d, dtype=np.int32)
    rem_arr = np.zeros(d, dtype=np.int32)

    bins[0] = np.int32(c0)
    conv[0] = np.int64(c0) * np.int64(c0)
    rem_arr[0] = np.int32(S)
    rem_arr[1] = np.int32(S - c0)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    buf_cap = np.int64(0)
    if not count_only:
        buf_cap = np.int64(out_buf.shape[0])

    if d == 1:
        if c0 == S:
            n_tested = 1
        return n_surv, n_tested

    if d == 2:
        forced = S - c0
        if 0 <= forced <= x_cap:
            n_tested = 1
            bins[1] = np.int32(forced)
            conv[0] = np.int64(c0) * np.int64(c0)
            conv[1] = np.int64(2) * np.int64(c0) * np.int64(forced)
            conv[2] = np.int64(forced) * np.int64(forced)

            pruned = False
            for ell in range(2, max_ell + 1):
                if pruned:
                    break
                n_cv = ell - 1
                nw = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += conv[k]
                for s_lo in range(nw):
                    if s_lo > 0:
                        ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                    if ws > thr[ell]:
                        pruned = True
                        break

            if not pruned:
                if not count_only and n_surv < buf_cap:
                    out_buf[n_surv, 0] = np.int32(c0)
                    out_buf[n_surv, 1] = np.int32(forced)
                n_surv += 1
        return n_surv, n_tested

    # General case: d >= 3
    pos = 1
    bins[1] = np.int32(0)

    while True:
        c_val = bins[pos]
        rem = rem_arr[pos]

        if pos == d_m1:
            # Last bin: forced
            forced = rem
            if 0 <= forced <= x_cap:
                n_tested += 1
                bins[pos] = np.int32(forced)

                # Add conv contribution
                f64 = np.int64(forced)
                conv[2 * pos] += f64 * f64
                for j in range(pos):
                    conv[pos + j] += np.int64(2) * f64 * np.int64(bins[j])

                # Full window scan
                pruned_leaf = False
                for ell in range(2, max_ell + 1):
                    if pruned_leaf:
                        break
                    n_cv = ell - 1
                    nw = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(nw):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            pruned_leaf = True
                            break

                if not pruned_leaf:
                    if not count_only and n_surv < buf_cap:
                        for i in range(d):
                            out_buf[n_surv, i] = bins[i]
                    n_surv += 1

                # Undo conv
                conv[2 * pos] -= f64 * f64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * f64 * np.int64(bins[j])

            # Backtrack
            pos -= 1
            if pos < 1:
                break
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Non-last bin
        max_v = min(rem, x_cap)
        min_v = rem - (d_m1 - pos) * x_cap
        if min_v < 0:
            min_v = 0
        if c_val < min_v:
            bins[pos] = np.int32(min_v)
            c_val = min_v

        if c_val > max_v:
            if pos <= 1:
                break
            pos -= 1
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Add conv
        c64 = np.int64(c_val)
        if c_val > 0:
            conv[2 * pos] += c64 * c64
            for j in range(pos):
                conv[pos + j] += np.int64(2) * c64 * np.int64(bins[j])

        # Partial prune: windows within [0, 2*pos]
        max_cv_pos = 2 * pos
        pruned_partial = False
        for ell in range(2, max_ell + 1):
            if pruned_partial:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                pruned_partial = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    pruned_partial = True
                    break

        if pruned_partial:
            if c_val > 0:
                conv[2 * pos] -= c64 * c64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c64 * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Descend
        rem_arr[pos + 1] = np.int32(rem - c_val)
        pos += 1
        bins[pos] = np.int32(0)

    return n_surv, n_tested


@njit(parallel=True, cache=True)
def _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested):
    """Pass 1: count survivors per c0."""
    dummy = np.empty((0, d), dtype=np.int32)
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        ns, nt = _l0_bnb_inner(c0, d, S, x_cap, thr, dummy, True)
        counts[idx] = ns
        tested[idx] = nt


@njit(parallel=True, cache=True)
def _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf):
    """Pass 2: fill output buffer."""
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        cnt = counts[idx]
        if cnt == 0:
            continue
        off = offsets[idx]
        _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf[off:off + cnt], False)


def run_l0(d, S, c_target):
    """Run L0: enumerate all compositions of S into d parts, prune."""
    thr = compute_thresholds(c_target, S, d)
    x_cap = compute_xcap(c_target, S, d)

    min_c0 = max(0, S - (d - 1) * x_cap)
    max_c0 = min(S, x_cap)
    # Canonical: c0 <= S//2 (symmetry)
    max_c0 = min(max_c0, S // 2)
    n_c0 = max_c0 - min_c0 + 1

    if n_c0 <= 0:
        return np.empty((0, d), dtype=np.int32), 0, 0

    counts = np.zeros(n_c0, dtype=np.int64)
    tested = np.zeros(n_c0, dtype=np.int64)

    _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested)

    offsets = np.zeros(n_c0 + 1, dtype=np.int64)
    for i in range(n_c0):
        offsets[i + 1] = offsets[i] + counts[i]
    total_surv = int(offsets[n_c0])
    total_tested = int(np.sum(tested))

    if total_surv == 0:
        return np.empty((0, d), dtype=np.int32), 0, total_tested

    out_buf = np.empty((total_surv, d), dtype=np.int32)
    _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf)

    return out_buf, total_surv, total_tested


# =====================================================================
# L1+: Fused generate-and-prune for cascade children
# =====================================================================
#
# §3 of COARSE_CASCADE_PROVER_FIXES.md: sound subtree-prune LB.
#
# At descent point pos, child[0..2*pos+1] is FIXED (cursor[0..pos] set);
# child[2*pos+2..2*d_parent-1] is UNFIXED (cursor[pos+1..d_parent-1] free
# in [lo[i'], hi[i']]). For each conv[k] we compute min_contrib[k] := a
# valid lower bound on the additional contribution of the UNFIXED bins to
# conv[k], so partial_conv[k] + min_contrib[k] <= conv[k] for every
# completion. Per window (ell, s), ws_lb := sum_{k in [s, s+ell-2]} (...)
# is a sound LB on ws; if ws_lb > thr[ell] for any (ell, s), every
# completion has TV >= c_target on that window, so the subtree is pruned.
#
# Per-pair contribution (parent position i' unfixed, child positions
# k1=2i', k2=2i'+1, c = cursor[i'] in [L, H], P = parent[i']):
#   self  k1*k1: c^2 in [L^2, H^2]            -> min  L^2
#   self  k2*k2: (P-c)^2                       -> min  (P-H)^2
#   mutual k1+k2: 2*c*(P-c) concave           -> min  min(2L(P-L), 2H(P-H))
#   cross with fixed bin j (cj fixed >=0):
#     k1+j: 2*c*cj            -> min 2*L*cj
#     k2+j: 2*(P-c)*cj        -> min 2*(P-H)*cj
#   cross with other unfixed (i'', P2 = parent[i''], L2, H2):
#     k1+k1b: 2*c*c2          -> min 2*L*L2
#     k1+k2b: 2*c*(P2-c2)     -> min 2*L*(P2-H2)
#     k2+k1b: 2*(P-c)*c2      -> min 2*(P-H)*L2
#     k2+k2b: 2*(P-c)*(P2-c2) -> min 2*(P-H)*(P2-H2)


@njit(cache=True)
def _subtree_prune_min_contrib(child, parent, lo_arr, hi_arr, pos, d_parent,
                                conv, conv_len, thr, max_ell):
    """Returns True if some window (ell, s) has ws_lb > thr[ell] for ALL
    completions of cursor[pos+1..d_parent-1]. Sound subtree pruning.
    """
    # min_contrib[k] = LB on additional contribution to conv[k] from unfixed bins.
    min_contrib = np.zeros(conv_len, dtype=np.int64)
    fixed_end = 2 * pos + 2  # child[0..fixed_end-1] is set

    for ip in range(pos + 1, d_parent):
        P = np.int64(parent[ip])
        L = np.int64(lo_arr[ip])
        H = np.int64(hi_arr[ip])
        k1 = 2 * ip
        k2 = 2 * ip + 1
        ml = L            # min child[k1] over c in [L, H]
        mh = P - H        # min child[k2]

        # Self-terms
        if ml > 0:
            min_contrib[2 * k1] += ml * ml
        if mh > 0:
            min_contrib[2 * k2] += mh * mh

        # Mutual k1+k2 (concave -> endpoint min)
        m_lo = np.int64(2) * L * (P - L)
        m_hi = np.int64(2) * H * (P - H)
        if m_lo < m_hi:
            min_contrib[k1 + k2] += m_lo
        else:
            min_contrib[k1 + k2] += m_hi

        # Cross with FIXED bins
        for j in range(fixed_end):
            cj = np.int64(child[j])
            if cj > 0:
                if ml > 0:
                    min_contrib[k1 + j] += np.int64(2) * cj * ml
                if mh > 0:
                    min_contrib[k2 + j] += np.int64(2) * cj * mh

        # Cross with OTHER UNFIXED positions
        for ip2 in range(ip + 1, d_parent):
            P2 = np.int64(parent[ip2])
            L2 = np.int64(lo_arr[ip2])
            H2 = np.int64(hi_arr[ip2])
            k1b = 2 * ip2
            k2b = 2 * ip2 + 1
            ml2 = L2
            mh2 = P2 - H2
            if ml > 0 and ml2 > 0:
                min_contrib[k1 + k1b] += np.int64(2) * ml * ml2
            if ml > 0 and mh2 > 0:
                min_contrib[k1 + k2b] += np.int64(2) * ml * mh2
            if mh > 0 and ml2 > 0:
                min_contrib[k2 + k1b] += np.int64(2) * mh * ml2
            if mh > 0 and mh2 > 0:
                min_contrib[k2 + k2b] += np.int64(2) * mh * mh2

    # Window scan via prefix sums
    pc_prefix = np.empty(conv_len + 1, dtype=np.int64)
    mc_prefix = np.empty(conv_len + 1, dtype=np.int64)
    pc_prefix[0] = 0
    mc_prefix[0] = 0
    for k in range(conv_len):
        pc_prefix[k + 1] = pc_prefix[k] + conv[k]
        mc_prefix[k + 1] = mc_prefix[k] + min_contrib[k]

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_win = conv_len - n_cv + 1
        if n_win <= 0:
            continue
        for s in range(n_win):
            ws_lb = (pc_prefix[s + n_cv] - pc_prefix[s]) \
                  + (mc_prefix[s + n_cv] - mc_prefix[s])
            if ws_lb > thr[ell]:
                return True
    return False


@njit(cache=True)
def _cascade_child_bnb(parent, d_parent, S, x_cap, thr, out_buf):
    """Process one parent: BnB over all children, prune by TV >= c_target.

    Child bins: child[2i] = cursor[i], child[2i+1] = parent[i] - cursor[i].
    Cursors are independent (each ranges over its parent bin's valid splits).
    Subtree pruning via partial autoconvolution.

    Returns (n_survivors, n_tested).
    """
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    max_ell = 2 * d_child

    # Cursor ranges
    lo = np.empty(d_parent, dtype=np.int32)
    hi = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        lo[i] = np.int32(max(0, parent[i] - x_cap))
        hi[i] = np.int32(min(parent[i], x_cap))

    # Check product is nonzero
    product = np.int64(1)
    for i in range(d_parent):
        product *= np.int64(hi[i] - lo[i] + 1)
    if product == 0:
        return 0, np.int64(0)

    # DFS state
    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.zeros(d_child, dtype=np.int32)
    conv = np.zeros(conv_len, dtype=np.int64)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    max_surv = np.int64(out_buf.shape[0])

    # Quick-check state
    qc_ell = np.int32(0)
    qc_s = np.int32(0)

    pos = 0
    cursor[0] = lo[0]

    while True:
        c_val = cursor[pos]

        if c_val > hi[pos]:
            # Backtrack
            if pos == 0:
                break
            # Undo conv for position pos-1... wait, we undo the CURRENT pos
            # Actually we need to undo the pos we're leaving
            pos -= 1
            k1 = 2 * pos
            k2 = k1 + 1
            old1 = np.int64(child[k1])
            old2 = np.int64(child[k2])
            conv[2 * k1] -= old1 * old1
            conv[2 * k2] -= old2 * old2
            conv[k1 + k2] -= np.int64(2) * old1 * old2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * old1 * cj
                    conv[k2 + j] -= np.int64(2) * old2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        # Set child bins for this cursor position
        k1 = 2 * pos
        k2 = k1 + 1
        new1 = np.int64(c_val)
        new2 = np.int64(parent[pos] - c_val)
        child[k1] = np.int32(new1)
        child[k2] = np.int32(new2)

        # Add conv contribution
        conv[2 * k1] += new1 * new1
        conv[2 * k2] += new2 * new2
        conv[k1 + k2] += np.int64(2) * new1 * new2
        for j in range(k1):
            cj = np.int64(child[j])
            if cj != 0:
                conv[k1 + j] += np.int64(2) * new1 * cj
                conv[k2 + j] += np.int64(2) * new2 * cj

        # --- Partial prune: check windows fully within assigned range ---
        max_cv_pos = 2 * k2  # = 4*pos + 2
        partial_pruned = False
        for ell in range(2, max_ell + 1):
            if partial_pruned:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                partial_pruned = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    partial_pruned = True
                    break

        if partial_pruned:
            # Undo and advance cursor
            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        # --- Subtree prune (§3): sound LB on conv from completions of unfixed
        # cursors. If some window has ws_lb > thr[ell], EVERY descendant is
        # pruned. Skipped at leaf (already fully fixed) and when there's only
        # one position remaining (partial-prune already covers it).
        if pos < d_parent - 1:
            subtree_pruned = _subtree_prune_min_contrib(
                child, parent, lo, hi, pos, d_parent,
                conv, conv_len, thr, max_ell)
            if subtree_pruned:
                # Undo and advance cursor (same as partial_pruned branch)
                conv[2 * k1] -= new1 * new1
                conv[2 * k2] -= new2 * new2
                conv[k1 + k2] -= np.int64(2) * new1 * new2
                for j in range(k1):
                    cj = np.int64(child[j])
                    if cj != 0:
                        conv[k1 + j] -= np.int64(2) * new1 * cj
                        conv[k2 + j] -= np.int64(2) * new2 * cj
                child[k1] = 0
                child[k2] = 0
                cursor[pos] += 1
                continue

        if pos == d_parent - 1:
            # --- Leaf: all cursors assigned ---
            n_tested += 1

            # Quick check: retry previous killing window
            quick_killed = False
            if qc_ell > 0:
                n_cv_qc = qc_ell - 1
                ws_qc = np.int64(0)
                for k in range(qc_s, qc_s + n_cv_qc):
                    ws_qc += conv[k]
                if ws_qc > thr[qc_ell]:
                    quick_killed = True

            if not quick_killed:
                # Full window scan
                full_pruned = False
                for ell in range(2, max_ell + 1):
                    if full_pruned:
                        break
                    n_cv = ell - 1
                    n_win = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(n_win):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            full_pruned = True
                            qc_ell = np.int32(ell)
                            qc_s = np.int32(s_lo)
                            break

                if not full_pruned:
                    if n_surv < max_surv:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                    n_surv += 1

            # Undo and advance cursor
            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
        else:
            # Descend to next cursor
            pos += 1
            cursor[pos] = lo[pos]

    return n_surv, n_tested


@njit(cache=True)
def _count_one_parent(parent, d_parent, S, x_cap, thr):
    """Count survivors for one parent (no output buffer needed)."""
    dummy = np.empty((0, 2 * d_parent), dtype=np.int32)
    ns, nt = _cascade_child_bnb(parent, d_parent, S, x_cap, thr, dummy)
    return ns, nt


@njit(parallel=True, cache=True)
def _cascade_level_count_parallel(parents, d_parent, S, x_cap, thr,
                                   counts, tested):
    """Pass 1 (parallel): count survivors per parent."""
    n_parents = parents.shape[0]
    for p_idx in prange(n_parents):
        dummy = np.empty((0, 2 * d_parent), dtype=np.int32)
        ns, nt = _cascade_child_bnb(parents[p_idx], d_parent, S, x_cap,
                                     thr, dummy)
        counts[p_idx] = ns
        tested[p_idx] = nt


@njit(parallel=True, cache=True)
def _cascade_level_fill_parallel(parents, d_parent, S, x_cap, thr,
                                  offsets, counts, out_buf):
    """Pass 2 (parallel): fill the output buffer."""
    n_parents = parents.shape[0]
    for p_idx in prange(n_parents):
        cnt = counts[p_idx]
        if cnt == 0:
            continue
        off = offsets[p_idx]
        _cascade_child_bnb(parents[p_idx], d_parent, S, x_cap, thr,
                           out_buf[off:off + cnt])


def run_cascade_level(survivors_prev, d_parent, S, c_target, verbose=True):
    """Run one cascade level: generate and prune children of all survivors.

    Returns (survivors_array, total_survivors, total_tested).
    """
    d_child = 2 * d_parent
    n_parents = survivors_prev.shape[0]
    thr = compute_thresholds(c_target, S, d_child)
    x_cap = compute_xcap(c_target, S, d_child)

    if verbose:
        try:
            n_threads = numba.get_num_threads()
        except Exception:
            n_threads = 1
        print(f"    x_cap={x_cap}, d_child={d_child}, "
              f"n_parents={n_parents}, threads={n_threads}")

    # Pass 1: count survivors per parent (PARALLEL via prange)
    counts = np.zeros(n_parents, dtype=np.int64)
    tested = np.zeros(n_parents, dtype=np.int64)

    t0 = time.time()
    _cascade_level_count_parallel(survivors_prev, d_parent, S, x_cap, thr,
                                  counts, tested)
    if verbose:
        elapsed = time.time() - t0
        rate = n_parents / max(elapsed, 1e-6)
        print(f"      counted {n_parents:,} parents in {elapsed:.2f}s "
              f"({rate:.0f} parents/s, "
              f"{int(tested.sum()):,} tested, "
              f"{int(counts.sum()):,} survived)")

    total_surv = int(counts.sum())
    total_tested = int(tested.sum())

    if total_surv == 0:
        return np.empty((0, d_child), dtype=np.int32), 0, total_tested

    # Pass 2: fill output (PARALLEL via prange)
    offsets = np.zeros(n_parents + 1, dtype=np.int64)
    for i in range(n_parents):
        offsets[i + 1] = offsets[i] + counts[i]

    out_buf = np.empty((total_surv, d_child), dtype=np.int32)
    t1 = time.time()
    _cascade_level_fill_parallel(survivors_prev, d_parent, S, x_cap, thr,
                                 offsets[:-1], counts, out_buf)
    if verbose:
        print(f"      filled {total_surv:,} survivors in {time.time()-t1:.2f}s")

    return out_buf, total_surv, total_tested


# =====================================================================
# Canonicalization and deduplication
# =====================================================================

@njit(parallel=True, cache=True)
def _canonicalize_inplace(arr):
    """Replace each row with min(row, rev(row)) lexicographically."""
    B = arr.shape[0]
    d = arr.shape[1]
    half = d // 2
    for b in prange(B):
        swap = False
        for i in range(half):
            j = d - 1 - i
            if arr[b, j] < arr[b, i]:
                swap = True
                break
            elif arr[b, j] > arr[b, i]:
                break
        if swap:
            for i in range(half):
                j = d - 1 - i
                tmp = arr[b, i]
                arr[b, i] = arr[b, j]
                arr[b, j] = tmp


def dedup(arr):
    """Deduplicate rows via lexsort."""
    if len(arr) == 0:
        return arr
    d = arr.shape[1]
    keys = tuple(arr[:, d - 1 - i] for i in range(d))
    sort_idx = np.lexsort(keys)
    sorted_arr = arr[sort_idx]
    mask = np.ones(len(sorted_arr), dtype=bool)
    for i in range(1, len(sorted_arr)):
        if np.array_equal(sorted_arr[i], sorted_arr[i - 1]):
            mask[i] = False
    return sorted_arr[mask]


# =====================================================================
# Box certification (EXACT vertex enumeration, with McCormick LP fallback)
# =====================================================================
#
# §2 of COARSE_CASCADE_PROVER_FIXES.md: replace water-filling (one feasible
# point per window — heuristic) with EXACT vertex enumeration.
#
# Cell:   { mu : max(0, mu*_i - h) <= mu_i <= min(1, mu*_i + h),  sum mu = 1 }
# Equivalently delta := mu - mu*:  { lo_i <= delta_i <= hi_i,  sum delta = 0 },
#   lo_i = max(-h, -mu*_i),  hi_i = min(h, 1 - mu*_i),  h = 1/(2S).
#
# TV_W(mu) = (2d/ell) * mu^T A_W mu, A_W[i,j] = 1{ s <= i+j <= s+ell-2 }.
# TV_W(mu*+delta) = TV_W(mu*) + grad . delta + scale * delta^T A_W delta,
#   with grad = 2 * scale * A_W mu*, scale = 2d/ell.
# So:
#   min_{delta in cell} TV_W = TV_W(mu*) - max_{delta in cell} [-grad.delta - scale.delta^T A_W delta]
# The max-of-quadratic-on-polytope is attained at a vertex (vertex theorem).
# Vertices of the simplex-constrained box: choose a free index f, pin the other
# d-1 to {lo_i, hi_i}, solve free by sum=0; feasible iff free in [lo_f, hi_f].
# Total: d * 2^(d-1) candidates. Practical for d <= 16.

VERTEX_ENUM_MAX_D = 16


@njit(cache=True)
def _phase1_lipschitz_lb(mu_center, d, delta_q, c_target):
    """Tier 1 (cheap, sound): rigorous Lipschitz LB on min_{mu in cell} max_W TV_W(mu).

    For each window W=(ell, s), bound the cell-min via Lipschitz drop:
        min_{delta in cell} TV_W(mu* + delta) >= TV_W(mu*) - L1*U1 - quad_drop
    where:
        L1 = (max grad - min grad) / 2     (centered grad inf-norm; uses sum delta = 0)
        U1 = ||delta||_1 bound under sum delta = 0 in clipped cell
        quad_drop = min(scale*U1^2, scale*U1*h*max_row(A_W))   (two sound bounds)
    Both terms are sound upper bounds on |grad . delta + scale * delta^T A_W delta|.

    Returns (certified, best_lb): certified iff max_W LB_W >= c_target.
    Cost: O(d^3) (window scan, no vertex loop). ~ 1e3 ops at d=12 vs ~1e9 for vertex enum.
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    # Cell box (clipped to nonneg simplex)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem = 1.0 - mu_center[i]
        if rem < hi_i:
            hi_i = rem
        lo[i] = lo_i
        hi[i] = hi_i

    # ||delta||_1 bound: under sum delta = 0, sum delta+ = sum delta- so
    # ||delta||_1 = 2*sum delta+ <= 2*min(sum hi, -sum lo). Also
    # ||delta||_1 <= 2h*floor(d/2) (split d/2 entries at +h, d/2 at -h).
    sum_pos = 0.0
    sum_neg = 0.0
    for i in range(d):
        sum_pos += hi[i]
        sum_neg += -lo[i]
    U1 = sum_pos if sum_pos < sum_neg else sum_neg
    U1 *= 2.0  # ||delta||_1 = 2*min(sum delta+, sum |delta-|)
    cap_1 = 2.0 * h * np.float64(d // 2)
    if cap_1 < U1:
        U1 = cap_1

    # ||delta||_2 bound: max sum delta_i^2 with |delta_i| <= h is d*h^2
    # (all at +/-h). With sum=0 constraint, same bound (forces d/2 each sign).
    # Tighter for clipped cells: sum hi_i^2 + sum (-lo_i)^2 over the d
    # entries that achieve their bounds. Conservative: use d*h^2.
    U2 = h * np.sqrt(np.float64(d))

    # Pre-compute grad_arr for L2 norm; we'll fill it per (ell, s)
    grad_arr = np.empty(d, dtype=np.float64)

    best_lb = 0.0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2
        max_row = ell - 1
        if max_row > d:
            max_row = d
        max_row_f = np.float64(max_row)
        # Quadratic drop bound (three sound bounds, take min)
        # qd_a: |delta^T A_W delta| <= ||delta||_1^2 (since |A_W| <= 1 entrywise)
        # qd_b: |delta^T A_W delta| <= ||delta||_1 * h * max_row_sum(A_W)
        # qd_c: |delta^T A_W delta| <= ||delta||_2^2 * sigma_max(A_W) <= d*h^2 * (ell-1)
        #       (sigma_max(A_W) <= max_row_sum since A_W is nonneg)
        qd_a = scale * U1 * U1
        qd_b = scale * U1 * h * max_row_f
        qd_c = scale * U2 * U2 * max_row_f
        quad_drop = qd_a
        if qd_b < quad_drop: quad_drop = qd_b
        if qd_c < quad_drop: quad_drop = qd_c

        for s in range(conv_len - ell + 2):
            # Compute TV_W(mu*) and grad component-wise; track max/min/sum/sumsq
            tv_c = 0.0
            gmax = -1e300
            gmin = 1e300
            gsum = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                grad_i = 2.0 * scale * acc
                grad_arr[i] = grad_i
                tv_c += mu_center[i] * acc
                gsum += grad_i
                if grad_i > gmax:
                    gmax = grad_i
                if grad_i < gmin:
                    gmin = grad_i
            tv_c *= scale

            # Linear drop: SORTED EXACT LP min for unclipped symmetric box [-h,h]^d
            # with sum delta = 0. The minimum of grad.delta is achieved by
            # delta_(i) = +h for the floor(d/2) smallest sorted grad values and
            # delta_(i) = -h for the floor(d/2) largest sorted grad values
            # (with one coord at the median set to balance for odd d).
            #   exact_LP_min = h * (sum_low_half - sum_high_half)
            # where sum_low_half = sum of floor(d/2) smallest grad,
            #       sum_high_half = sum of floor(d/2) largest grad.
            # Centering grad by mean does not change grad.delta when sum delta=0.
            # For the CLIPPED (asymmetric) cell, the unclipped value is a SOUND
            # upper bound on the magnitude of the drop (clipping shrinks the
            # feasible set, so the actual LP min is >= unclipped LP min).
            mean_g = gsum / np.float64(d)
            # Centered, sorted grad
            for i in range(d):
                grad_arr[i] = (grad_arr[i] - mean_g)
            # Numba-supported np.sort
            sg = np.sort(grad_arr)
            half = d // 2
            sum_low_half = 0.0
            for k in range(half):
                sum_low_half += sg[k]
            sum_high_half = 0.0
            for k in range(d - half, d):
                sum_high_half += sg[k]
            # |LP_min| = h * (sum_high_half - sum_low_half)
            lin_drop = h * (sum_high_half - sum_low_half)
            # Also keep L1 / L2 bounds and take min (they handle clipped boxes
            # via U1, U2 which include clipping; sorted-LP uses unclipped h)
            L1_norm = 0.5 * (gmax - gmin)
            lin_drop_1 = L1_norm * U1
            l2sq = 0.0
            for i in range(d):
                l2sq += grad_arr[i] * grad_arr[i]
            L2_norm = np.sqrt(l2sq)
            lin_drop_2 = L2_norm * U2
            if lin_drop_1 < lin_drop:
                lin_drop = lin_drop_1
            if lin_drop_2 < lin_drop:
                lin_drop = lin_drop_2

            lb_W = tv_c - lin_drop - quad_drop
            if lb_W > best_lb:
                best_lb = lb_W
            if best_lb >= c_target:
                return True, best_lb

    return best_lb >= c_target, best_lb


@njit(cache=True)
def _phase1_lipschitz_lb_spec(mu_center, d, delta_q, c_target, qdrop_table):
    """Speedup #2: Phase 1 with SPECTRAL quad_drop using qdrop_table[ell, s].

    Identical to _phase1_lipschitz_lb except the qd_c bound uses
    rho_W := max(0, -lambda_min^V(A_W)) (precomputed) instead of the loose
    row-sum proxy max_row = ell - 1. Provably:
        delta^T A_W delta >= lambda_min^V(A_W) * ||delta||_2^2  (sum=0 delta)
        => -min(delta^T A_W delta) <= rho_W * ||delta||_2^2
    so quad_drop = scale * U2^2 * rho_W is a sound and TIGHT bound on the
    quadratic-side magnitude. Since rho_W <= ell-1 always (Perron-Frobenius
    on the row sum bounds the spectral radius), this is monotonically
    tighter than the existing qd_c. PSD-on-V windows give rho_W = 0 (zero
    quadratic drop — strictly tightest possible).
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem = 1.0 - mu_center[i]
        if rem < hi_i:
            hi_i = rem
        lo[i] = lo_i
        hi[i] = hi_i

    sum_pos = 0.0
    sum_neg = 0.0
    for i in range(d):
        sum_pos += hi[i]
        sum_neg += -lo[i]
    U1 = sum_pos if sum_pos < sum_neg else sum_neg
    U1 *= 2.0
    cap_1 = 2.0 * h * np.float64(d // 2)
    if cap_1 < U1:
        U1 = cap_1

    U2 = h * np.sqrt(np.float64(d))
    U2_sq = U2 * U2

    grad_arr = np.empty(d, dtype=np.float64)
    best_lb = 0.0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2
        max_row = ell - 1
        if max_row > d:
            max_row = d
        max_row_f = np.float64(max_row)

        for s in range(conv_len - ell + 2):
            tv_c = 0.0
            gmax = -1e300
            gmin = 1e300
            gsum = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                grad_i = 2.0 * scale * acc
                grad_arr[i] = grad_i
                tv_c += mu_center[i] * acc
                gsum += grad_i
                if grad_i > gmax:
                    gmax = grad_i
                if grad_i < gmin:
                    gmin = grad_i
            tv_c *= scale

            # Linear drop: SORTED EXACT LP min (see _phase1_lipschitz_lb).
            mean_g = gsum / np.float64(d)
            for i in range(d):
                grad_arr[i] = grad_arr[i] - mean_g
            sg = np.sort(grad_arr)
            half = d // 2
            sum_low_half = 0.0
            for k in range(half):
                sum_low_half += sg[k]
            sum_high_half = 0.0
            for k in range(d - half, d):
                sum_high_half += sg[k]
            lin_drop = h * (sum_high_half - sum_low_half)
            L1_norm = 0.5 * (gmax - gmin)
            lin_drop_1 = L1_norm * U1
            l2sq = 0.0
            for i in range(d):
                l2sq += grad_arr[i] * grad_arr[i]
            L2_norm = np.sqrt(l2sq)
            lin_drop_2 = L2_norm * U2
            if lin_drop_1 < lin_drop:
                lin_drop = lin_drop_1
            if lin_drop_2 < lin_drop:
                lin_drop = lin_drop_2

            # Quadratic drop with SPECTRAL bound from qdrop_table
            rho_W = qdrop_table[ell, s]
            qd_a = scale * U1 * U1
            qd_b = scale * U1 * h * max_row_f
            qd_c = scale * U2_sq * rho_W   # spectral: <= scale * U2^2 * (ell-1)
            quad_drop = qd_a
            if qd_b < quad_drop:
                quad_drop = qd_b
            if qd_c < quad_drop:
                quad_drop = qd_c

            lb_W = tv_c - lin_drop - quad_drop
            if lb_W > best_lb:
                best_lb = lb_W
            if best_lb >= c_target:
                return True, best_lb

    return best_lb >= c_target, best_lb


@njit(cache=True)
def _trust_region_qp_lb(c_vec, lam_vec, R_squared, d):
    """Closed-form sound LB on min_{x: ||x||_2^2 <= R_squared} c.x + sum_k lam_k x_k^2.

    Returns L*(nu) = -sum c_k^2 / (4 (lam_k+nu)) - nu * R_squared
    for some nu > max(0, -lam_min) found by Newton iteration. By weak
    duality, ANY feasible nu gives a sound LB on the QP minimum, so even
    if Newton fails to converge exactly the returned value is sound.

    Special cases:
      - All lam_k > 0 and unconstrained min ||x*||^2 <= R^2: nu = 0,
        return unconstrained min directly.
      - All c_k = 0: QP min = 0 (x = 0 feasible).
    """
    # Compute |c|^2 (early return if zero)
    csum_sq = 0.0
    for k in range(d):
        csum_sq += c_vec[k] * c_vec[k]
    if csum_sq < 1e-300:
        return 0.0

    lam_min = lam_vec[0]
    for k in range(1, d):
        if lam_vec[k] < lam_min:
            lam_min = lam_vec[k]

    # If PSD: try unconstrained min
    if lam_min > 1e-12:
        x_norm_sq = 0.0
        unconstrained_val = 0.0
        for k in range(d):
            denom = 2.0 * lam_vec[k]
            if denom > 1e-15:
                xk = -c_vec[k] / denom
                x_norm_sq += xk * xk
                unconstrained_val -= c_vec[k] * c_vec[k] / (4.0 * lam_vec[k])
        if x_norm_sq <= R_squared:
            return unconstrained_val

    # Constrained or indefinite: Newton on f(nu) = sum c_k^2 / (4 (lam_k+nu)^2) = R^2
    # ν must satisfy lam_k + nu > 0 for all k => nu > -lam_min (strict).
    nu_lb = -lam_min
    if nu_lb < 0.0:
        nu_lb = 0.0
    # Initial guess: just above the strict lower bound
    nu = nu_lb + 1e-8
    # Bracket: f is monotonically decreasing in nu, so find nu_hi with f(nu_hi) < R^2
    # Start big and reduce
    nu_hi = nu_lb + 1.0
    while True:
        f_hi = 0.0
        for k in range(d):
            denom = lam_vec[k] + nu_hi
            if denom > 1e-15:
                f_hi += c_vec[k] * c_vec[k] / (4.0 * denom * denom)
        if f_hi <= R_squared:
            break
        nu_hi *= 2.0
        if nu_hi > 1e18:
            break

    # Bisection (robust) — Newton has issues near singular point
    nu_lo_iter = nu_lb + 1e-12
    nu_hi_iter = nu_hi
    for it in range(60):
        nu_mid = 0.5 * (nu_lo_iter + nu_hi_iter)
        f_mid = 0.0
        for k in range(d):
            denom = lam_vec[k] + nu_mid
            if denom > 1e-15:
                f_mid += c_vec[k] * c_vec[k] / (4.0 * denom * denom)
        if f_mid > R_squared:
            nu_lo_iter = nu_mid
        else:
            nu_hi_iter = nu_mid
        if (nu_hi_iter - nu_lo_iter) < 1e-14 * (1.0 + abs(nu_mid)):
            break
    nu = 0.5 * (nu_lo_iter + nu_hi_iter)

    # Evaluate L*(nu) — by weak duality this is a sound LB on QP min
    qp_min = -nu * R_squared
    for k in range(d):
        denom = lam_vec[k] + nu
        if denom > 1e-15:
            qp_min -= c_vec[k] * c_vec[k] / (4.0 * denom)
    return qp_min


@njit(cache=True)
def _trust_region_qp_lb_badtr(c_vec, lam_vec, beta_modes, R_squared, d):
    """BADTR — sound LB on `min c.y + sum lam_k y_k^2` with TWO classes of constraints:
        (a) ||y||_2^2 <= R_squared  (L_2 ball)
        (b) y_k^2 <= beta_modes[k] for each k  (per-eigenmode bounds)

    Both classes have diagonal Hessians in the eigenbasis, so the Lagrangian
    dual decouples: with multipliers nu_2 >= 0 (L_2) and mu_k >= 0 (per-mode),
        L*(nu_2, mu) = sum_k [-c_k^2 / (4*(lam_k + nu_2 + mu_k)) - mu_k * beta_modes[k]] - nu_2 * R^2.

    KKT for mu_k:
        if |c_k|/(2*sqrt(beta_k)) > lam_k + nu_2:
            mu_k* = |c_k|/(2*sqrt(beta_k)) - lam_k - nu_2  (active)
            contribution to L*: -|c_k|*sqrt(beta_k) + beta_k*(lam_k + nu_2)
        else:
            mu_k* = 0  (inactive)
            contribution: -c_k^2 / (4*(lam_k + nu_2))

    Outer bisection on nu_2 finds the root of
        sum_{active(nu_2)} beta_k + sum_{inactive(nu_2)} c_k^2/(4*(lam_k + nu_2)^2) = R^2.

    The added per-mode constraints can ONLY tighten the LB compared to L_2 alone
    (each mu_k >= 0 in the Lagrangian dual corresponds to an EXTRA sound bound),
    so this LB is provably >= the existing _trust_region_qp_lb output.

    Inputs:
      c_vec    : [d] projected gradient in eigenbasis (centered, kernel coord = 0)
      lam_vec  : [d] eigenvalues of P*scale*A_W*P (kernel coord = 0)
      beta_modes: [d] per-mode bound = max over (lo<=delta<=hi) of (V^T delta)_k^2
      R_squared : sum_i max(lo_i^2, hi_i^2) (sum of beta_modes upper bound)

    Returns: sound LB on the QP min over the (L_2 ball intersect per-mode bounds).
    """
    csum_sq = 0.0
    for k in range(d):
        csum_sq += c_vec[k] * c_vec[k]
    if csum_sq < 1e-300:
        return 0.0  # zero gradient => x=0 optimal

    # Lower bracket: nu_2 must keep all (lam_k + nu_2 + mu_k) > 0.
    # For inactive k: need lam_k + nu_2 > 0 ; nu_lb = max(0, -min lam_k).
    lam_min = lam_vec[0]
    for k in range(1, d):
        if lam_vec[k] < lam_min:
            lam_min = lam_vec[k]
    nu_lb = -lam_min if lam_min < 0 else 0.0
    nu_lb = nu_lb + 1e-12  # strictly above singular

    # Upper bracket: find nu_hi such that f(nu_hi) <= R_squared.
    # f(nu_2) = sum_{active} beta_k + sum_{inactive} c_k^2/(4*(lam_k+nu_2)^2)
    # f is monotone non-increasing in nu_2 (active set shrinks, inactive terms shrink).
    nu_hi = nu_lb + 1.0
    for _bracket in range(80):
        f_hi = 0.0
        for k in range(d):
            beta_k = beta_modes[k]
            if beta_k < 1e-300:
                # mode is dead (kernel direction); use only inactive form
                d_val = lam_vec[k] + nu_hi
                if d_val > 1e-15:
                    f_hi += c_vec[k] * c_vec[k] / (4.0 * d_val * d_val)
                continue
            sqrt_b = np.sqrt(beta_k)
            thr = abs(c_vec[k]) / (2.0 * sqrt_b)
            if thr > lam_vec[k] + nu_hi:
                # active: contributes beta_k to f
                f_hi += beta_k
            else:
                d_val = lam_vec[k] + nu_hi
                if d_val > 1e-15:
                    f_hi += c_vec[k] * c_vec[k] / (4.0 * d_val * d_val)
        if f_hi <= R_squared:
            break
        nu_hi *= 2.0
        if nu_hi > 1e18:
            break

    # Bisect
    nu_lo_iter = nu_lb
    nu_hi_iter = nu_hi
    for _it in range(80):
        nu_mid = 0.5 * (nu_lo_iter + nu_hi_iter)
        f_mid = 0.0
        for k in range(d):
            beta_k = beta_modes[k]
            if beta_k < 1e-300:
                d_val = lam_vec[k] + nu_mid
                if d_val > 1e-15:
                    f_mid += c_vec[k] * c_vec[k] / (4.0 * d_val * d_val)
                continue
            sqrt_b = np.sqrt(beta_k)
            thr = abs(c_vec[k]) / (2.0 * sqrt_b)
            if thr > lam_vec[k] + nu_mid:
                f_mid += beta_k
            else:
                d_val = lam_vec[k] + nu_mid
                if d_val > 1e-15:
                    f_mid += c_vec[k] * c_vec[k] / (4.0 * d_val * d_val)
        if f_mid > R_squared:
            nu_lo_iter = nu_mid
        else:
            nu_hi_iter = nu_mid
        if (nu_hi_iter - nu_lo_iter) < 1e-14 * (1.0 + abs(nu_mid)):
            break
    nu_2 = 0.5 * (nu_lo_iter + nu_hi_iter)

    # Evaluate L*(nu_2, mu*(nu_2)) — sound LB by weak duality
    qp_min = -nu_2 * R_squared
    for k in range(d):
        beta_k = beta_modes[k]
        if beta_k < 1e-300:
            d_val = lam_vec[k] + nu_2
            if d_val > 1e-15:
                qp_min -= c_vec[k] * c_vec[k] / (4.0 * d_val)
            continue
        sqrt_b = np.sqrt(beta_k)
        thr = abs(c_vec[k]) / (2.0 * sqrt_b)
        if thr > lam_vec[k] + nu_2:
            # active: contribution -|c_k|*sqrt(beta_k) + beta_k*(lam_k + nu_2)
            qp_min += -abs(c_vec[k]) * sqrt_b + beta_k * (lam_vec[k] + nu_2)
        else:
            d_val = lam_vec[k] + nu_2
            if d_val > 1e-15:
                qp_min -= c_vec[k] * c_vec[k] / (4.0 * d_val)
    return qp_min


@njit(cache=True)
def _region_certify_shor(mu_center, lo, hi, d, c_target,
                          V_table, lam_table, valid_mask):
    """Speedup #3: closed-form Shor LB on REGION P = {mu_c + delta : lo <= delta <= hi, sum delta = 0}.

    This is the region-aggregation primitive: a single trust-region QP
    certificate covers the entire polytope P, which can include MANY grid
    cells. Applied to a single cell (lo = -h, hi = +h, clipped) it reduces
    to _box_certify_cell_trust_region. Applied to larger regions it becomes
    a bulk-certification: ONE QP solve for K cells gives O(K)-fold savings.

    Soundness: the QP is solved via the SAME closed-form Lagrangian/Newton
    over the L2 ball ||delta||_2^2 <= R^2 with R^2 = sum_i max(lo_i^2, hi_i^2)
    (sound upper bound on ||delta||^2 over the box). Returns a sound LB
    even when A_W is indefinite (kappa-shift implicit in the Lagrangian
    duality on the eigendecomposition of P A_W P).

    Args:
      mu_center : center of the region (sum to 1 expected, but not required)
      lo, hi    : per-coordinate delta box bounds (lo <= 0 <= hi)
      V_table, lam_table, valid_mask : eigendecomposition tables for A_W
    """
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    # R^2 = sum max(lo_i^2, hi_i^2) — sound upper bound on ||delta||_2^2
    R_sq = 0.0
    for i in range(d):
        a = lo[i] * lo[i]
        b = hi[i] * hi[i]
        if a > b:
            R_sq += a
        else:
            R_sq += b

    grad = np.empty(d, dtype=np.float64)
    g_centered = np.empty(d, dtype=np.float64)
    c_vec = np.empty(d, dtype=np.float64)
    lam_scaled = np.empty(d, dtype=np.float64)
    best_lb = 0.0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2

        for s in range(conv_len - ell + 2):
            if valid_mask[ell, s] == 0:
                continue
            tv_c = 0.0
            gsum = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                grad[i] = 2.0 * scale * acc
                tv_c += mu_center[i] * acc
                gsum += grad[i]
            tv_c *= scale

            mean_g = gsum / np.float64(d)
            for i in range(d):
                g_centered[i] = grad[i] - mean_g

            for k in range(d):
                acc = 0.0
                for i in range(d):
                    acc += V_table[ell, s, i, k] * g_centered[i]
                c_vec[k] = acc

            for k in range(d):
                lam_scaled[k] = scale * lam_table[ell, s, k]

            qp_min = _trust_region_qp_lb(c_vec, lam_scaled, R_sq, d)

            lb_W = tv_c + qp_min
            if lb_W > best_lb:
                best_lb = lb_W
            if best_lb >= c_target:
                return True, best_lb

    return best_lb >= c_target, best_lb


@njit(cache=True)
def _box_certify_cell_trust_region(mu_center, d, delta_q, c_target,
                                    V_table, lam_table, valid_mask):
    """Tier 2: closed-form trust-region QP LB on cell-min, per window.

    For each window W=(ell, s):
      LB_W := TV_W(mu_c) + min_{delta in V, ||delta||_2^2 <= R^2}
                                [g_W . delta + delta^T A_W delta]
    where R^2 is a sound upper bound on ||delta||_2^2 over the cell box
    (taken as sum_i max(lo_i^2, hi_i^2) — exact for symmetric box, tighter
    when clipping is severe).

    The QP is solved via eigendecomposition of P A_W P (precomputed) plus
    a Lagrangian Newton/bisection. Returns (cert, best_lb).

    Sound: L2 ball relaxation of the L_inf box. Tighter than Phase 1 (which
    is sound but uses split linear+quadratic bounds). Used as Tier 2
    between Phase 1 and Tier 3 vertex enum.
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    # Cell box (clipped to nonneg simplex)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem = 1.0 - mu_center[i]
        if rem < hi_i:
            hi_i = rem
        lo[i] = lo_i
        hi[i] = hi_i

    # R^2 = sum_i max(lo_i^2, hi_i^2) — sound upper bound on ||delta||_2^2
    # over the (clipped) cell box.
    R_sq = 0.0
    for i in range(d):
        a = lo[i] * lo[i]
        b = hi[i] * hi[i]
        if a > b:
            R_sq += a
        else:
            R_sq += b

    grad = np.empty(d, dtype=np.float64)
    g_centered = np.empty(d, dtype=np.float64)
    c_vec = np.empty(d, dtype=np.float64)
    best_lb = 0.0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2

        for s in range(conv_len - ell + 2):
            if valid_mask[ell, s] == 0:
                continue
            # Compute g_W and tv_c (analogous to _phase1)
            tv_c = 0.0
            gsum = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                grad[i] = 2.0 * scale * acc
                tv_c += mu_center[i] * acc
                gsum += grad[i]
            tv_c *= scale

            # Center the gradient: g_centered = grad - mean(grad)
            # (kernel direction has zero contribution since its eigenvector is 1/sqrt(d))
            mean_g = gsum / np.float64(d)
            for i in range(d):
                g_centered[i] = grad[i] - mean_g

            # The QP we solve over delta in V is:
            #   min g_centered . delta + delta^T (scale * A_W) delta
            # In eigenbasis of P A_W P (which is V_table[ell,s], lam_table[ell,s] * scale):
            # Project: c_k = V[:, k] . g_centered  (length d)
            # The kernel direction (eig=0, eigenvec=1/sqrt(d)) has c_kernel = sum(g_centered)/sqrt(d) = 0.
            for k in range(d):
                acc = 0.0
                for i in range(d):
                    acc += V_table[ell, s, i, k] * g_centered[i]
                c_vec[k] = acc

            # Solve trust-region QP with eigenvalues = scale * lam_table[ell, s, :]
            # We compute lam_scaled on-the-fly inside _trust_region_qp_lb? Easier: scale here.
            # Use a temporary buffer.
            # Actually call with lam_scaled[k] = scale * lam_table[ell, s, k]
            lam_scaled = np.empty(d, dtype=np.float64)
            for k in range(d):
                lam_scaled[k] = scale * lam_table[ell, s, k]

            qp_min = _trust_region_qp_lb(c_vec, lam_scaled, R_sq, d)

            lb_W = tv_c + qp_min
            if lb_W > best_lb:
                best_lb = lb_W
            if best_lb >= c_target:
                return True, best_lb

    return best_lb >= c_target, best_lb


@njit(cache=True)
def _box_certify_cell_badtr(mu_center, d, delta_q, c_target,
                              V_table, lam_table, valid_mask):
    """BADTR cell cert — Tier 2 with per-eigenmode bounds in addition to L_2 ball.

    Strictly tighter than _box_certify_cell_trust_region (current Tier 2):
    every additional dual variable mu_k >= 0 in BADTR can only increase
    the LB (or leave it unchanged when the per-mode constraint is inactive).

    Per window W=(ell, s):
      1. Compute g_W, tv_c (same as trust_region).
      2. Project g onto eigenbasis: c_k = V[:,k] . (g - mean(g)).
      3. Compute beta_modes[k] = max over (lo<=delta<=hi) of (V[:,k] . delta)^2
         in CLOSED FORM (linear function over a box; max at the corner where
         delta_i = hi_i if V_{i,k}>0 else lo_i, and similarly for min).
      4. Call _trust_region_qp_lb_badtr to get a sound LB on
         min_{||y||^2<=R^2, y_k^2<=beta_modes[k]} c.y + sum scale*lam_k y_k^2.
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem = 1.0 - mu_center[i]
        if rem < hi_i:
            hi_i = rem
        lo[i] = lo_i
        hi[i] = hi_i

    R_sq = 0.0
    for i in range(d):
        a = lo[i] * lo[i]
        b = hi[i] * hi[i]
        if a > b:
            R_sq += a
        else:
            R_sq += b

    grad = np.empty(d, dtype=np.float64)
    g_centered = np.empty(d, dtype=np.float64)
    c_vec = np.empty(d, dtype=np.float64)
    lam_scaled = np.empty(d, dtype=np.float64)
    beta_modes = np.empty(d, dtype=np.float64)
    best_lb = 0.0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2

        for s in range(conv_len - ell + 2):
            if valid_mask[ell, s] == 0:
                continue
            tv_c = 0.0
            gsum = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                grad[i] = 2.0 * scale * acc
                tv_c += mu_center[i] * acc
                gsum += grad[i]
            tv_c *= scale

            mean_g = gsum / np.float64(d)
            for i in range(d):
                g_centered[i] = grad[i] - mean_g

            # Project gradient + compute per-mode beta bounds
            # beta_k = max over box of (V[:,k] . delta)^2 = max(max_pos^2, max_neg^2)
            # where:
            #   max_pos_k = sum_i (V[i,k]>0 ? V[i,k]*hi[i] : V[i,k]*lo[i])
            #   max_neg_k = sum_i (V[i,k]>0 ? V[i,k]*lo[i] : V[i,k]*hi[i])
            # Kernel direction (k=0 conventionally, eigvec ~ 1/sqrt(d)) has
            # max_pos = (1/sqrt(d)) sum hi, max_neg = (1/sqrt(d)) sum lo.
            # For sum delta = 0 this is automatically 0, so beta_kernel is set
            # tiny (1e-300) — flagging the mode as "dead" in BADTR helper.
            for k in range(d):
                accc = 0.0
                vmax = 0.0
                vmin = 0.0
                for i in range(d):
                    vik = V_table[ell, s, i, k]
                    accc += vik * g_centered[i]
                    if vik > 0.0:
                        vmax += vik * hi[i]
                        vmin += vik * lo[i]
                    else:
                        vmax += vik * lo[i]
                        vmin += vik * hi[i]
                c_vec[k] = accc
                a_sq = vmax * vmax
                b_sq = vmin * vmin
                beta_k = a_sq if a_sq > b_sq else b_sq
                # Detect kernel direction: lam[k] near 0 AND |c_k| ~ 0.
                # The kernel is the all-ones direction; on the centered subspace
                # V it has eigenvalue 0 in lam_table. We mark it dead via tiny beta.
                if abs(lam_table[ell, s, k]) < 1e-13 and abs(c_vec[k]) < 1e-12:
                    beta_modes[k] = 1e-300
                else:
                    beta_modes[k] = beta_k

            for k in range(d):
                lam_scaled[k] = scale * lam_table[ell, s, k]

            qp_min = _trust_region_qp_lb_badtr(c_vec, lam_scaled, beta_modes, R_sq, d)

            lb_W = tv_c + qp_min
            if lb_W > best_lb:
                best_lb = lb_W
            if best_lb >= c_target:
                return True, best_lb

    return best_lb >= c_target, best_lb


@njit(cache=True)
def _box_certify_cell_cctr(mu_center, d, delta_q, c_target,
                             V_table, lam_table, valid_mask):
    """CCTR — Convex-Combination Trust-Region (Tier 2.5).

    Closes the minimax gap by computing a sound LB on
        min_{delta in cell} max_W TV_W(mu_c + delta).

    Approach: pick top-2 windows W1, W2 by TV_W(mu_c). Build the CONVEX
    COMBINATION at alpha=0.5:
        f_pair(delta) := 0.5*(TV_W1(mu_c+delta) + TV_W2(mu_c+delta))
                      = t_pair + g_pair . delta + delta^T A_pair delta
    where t_pair = 0.5*(TV_W1(mu_c) + TV_W2(mu_c)), g_pair = 0.5*(g_W1 + g_W2),
    A_pair = 0.5*(scale_W1 A_W1 + scale_W2 A_W2).

    By minimax inequality:
        max_W TV_W(mu) >= 0.5*(TV_W1(mu) + TV_W2(mu)) for all mu,
        => min_cell max_W TV_W >= min_cell 0.5*(TV_W1 + TV_W2).
    The latter is computed via trust-region with BADTR per-mode bounds:
    eigh(P A_pair P), project g_pair, solve BADTR.

    Useful when the cell-min for individual windows is NOT achieved at the
    same delta* — convex combination forces a "compromise" delta that's worse
    for both, raising the LB above any single window's bound.

    Returns (cert, lb): sound LB on min over cell of max_W TV_W(mu).
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    # Cell box (clipped)
    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem = 1.0 - mu_center[i]
        if rem < hi_i:
            hi_i = rem
        lo[i] = lo_i
        hi[i] = hi_i
    R_sq = 0.0
    for i in range(d):
        a = lo[i] * lo[i]
        b = hi[i] * hi[i]
        if a > b:
            R_sq += a
        else:
            R_sq += b

    # Step 1: scan windows, find top-2 by TV_W(mu_c)
    tv1 = -1e300
    ell1 = 0
    s1 = 0
    tv2 = -1e300
    ell2 = 0
    s2 = 0

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_off = ell - 2
        for s in range(conv_len - ell + 2):
            if valid_mask[ell, s] == 0:
                continue
            tv_W = 0.0
            for i in range(d):
                jl = s - i
                jh = s + s_hi_off - i
                if jl < 0:
                    jl = 0
                if jh > d - 1:
                    jh = d - 1
                acc = 0.0
                for j in range(jl, jh + 1):
                    acc += mu_center[j]
                tv_W += mu_center[i] * acc
            tv_W *= scale
            if tv_W > tv1:
                tv2 = tv1
                ell2 = ell1
                s2 = s1
                tv1 = tv_W
                ell1 = ell
                s1 = s
            elif tv_W > tv2:
                tv2 = tv_W
                ell2 = ell
                s2 = s

    # If we couldn't find 2 windows, return failure (shouldn't happen for valid d)
    if tv2 < -1e299:
        return False, 0.0

    scale1 = two_d / np.float64(ell1)
    scale2 = two_d / np.float64(ell2)
    s_hi_off1 = ell1 - 2
    s_hi_off2 = ell2 - 2

    # Step 2: build A_pair = 0.5 * (scale1 * A_W1 + scale2 * A_W2) (d x d)
    A_pair = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        # Window 1 contribution
        jl1 = s1 - i
        jh1 = s1 + s_hi_off1 - i
        if jl1 < 0: jl1 = 0
        if jh1 > d - 1: jh1 = d - 1
        for j in range(jl1, jh1 + 1):
            A_pair[i, j] += 0.5 * scale1
        # Window 2 contribution
        jl2 = s2 - i
        jh2 = s2 + s_hi_off2 - i
        if jl2 < 0: jl2 = 0
        if jh2 > d - 1: jh2 = d - 1
        for j in range(jl2, jh2 + 1):
            A_pair[i, j] += 0.5 * scale2

    # Step 3: tilde_A_pair = P A_pair P, P = I - 11^T/d
    # tilde_A[i,j] = A[i,j] - row_means[i] - col_means[j] + grand_mean
    row_means = np.zeros(d)
    col_means = np.zeros(d)
    for i in range(d):
        for j in range(d):
            row_means[i] += A_pair[i, j]
            col_means[j] += A_pair[i, j]
    grand_sum = 0.0
    for i in range(d):
        row_means[i] /= np.float64(d)
        grand_sum += row_means[i]
    for j in range(d):
        col_means[j] /= np.float64(d)
    grand_mean = grand_sum / np.float64(d)

    tilde_A_pair = np.empty((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            tilde_A_pair[i, j] = A_pair[i, j] - row_means[i] - col_means[j] + grand_mean

    # Step 4: eigh of tilde_A_pair (Numba supports np.linalg.eigh)
    lam_pair, V_pair = np.linalg.eigh(tilde_A_pair)

    # Step 5: compute g_pair = 0.5 * (g_W1 + g_W2)
    g_pair = np.empty(d, dtype=np.float64)
    for i in range(d):
        # g_W1[i] = 2 * scale1 * sum_{j: s1 <= i+j <= s1+ell1-2} mu_center[j]
        jl = s1 - i
        jh = s1 + s_hi_off1 - i
        if jl < 0: jl = 0
        if jh > d - 1: jh = d - 1
        acc1 = 0.0
        for j in range(jl, jh + 1):
            acc1 += mu_center[j]
        g_W1_i = 2.0 * scale1 * acc1

        jl = s2 - i
        jh = s2 + s_hi_off2 - i
        if jl < 0: jl = 0
        if jh > d - 1: jh = d - 1
        acc2 = 0.0
        for j in range(jl, jh + 1):
            acc2 += mu_center[j]
        g_W2_i = 2.0 * scale2 * acc2

        g_pair[i] = 0.5 * (g_W1_i + g_W2_i)

    # Step 6: project g_pair onto V_pair eigenbasis (centered)
    mean_g = 0.0
    for i in range(d):
        mean_g += g_pair[i]
    mean_g /= np.float64(d)
    g_centered = np.empty(d, dtype=np.float64)
    for i in range(d):
        g_centered[i] = g_pair[i] - mean_g

    c_vec = np.empty(d, dtype=np.float64)
    for k in range(d):
        acc = 0.0
        for i in range(d):
            acc += V_pair[i, k] * g_centered[i]
        c_vec[k] = acc

    # Step 7: compute beta_modes for V_pair via the box
    beta_modes = np.empty(d, dtype=np.float64)
    for k in range(d):
        v_max = 0.0
        v_min = 0.0
        for i in range(d):
            vik = V_pair[i, k]
            if vik > 0.0:
                v_max += vik * hi[i]
                v_min += vik * lo[i]
            else:
                v_max += vik * lo[i]
                v_min += vik * hi[i]
        a_sq = v_max * v_max
        b_sq = v_min * v_min
        beta_k = a_sq if a_sq > b_sq else b_sq
        # Detect kernel direction
        if abs(lam_pair[k]) < 1e-13 and abs(c_vec[k]) < 1e-12:
            beta_modes[k] = 1e-300
        else:
            beta_modes[k] = beta_k

    # Step 8: BADTR with eigenvalues already containing all scaling
    # (lam_pair = eigenvalues of P (0.5*(scale1*A_W1 + scale2*A_W2)) P, no extra scale needed)
    qp_min = _trust_region_qp_lb_badtr(c_vec, lam_pair, beta_modes, R_sq, d)

    # t_pair = 0.5 * (TV_W1(mu_c) + TV_W2(mu_c))
    t_pair = 0.5 * (tv1 + tv2)

    lb_pair = t_pair + qp_min
    return lb_pair >= c_target, lb_pair


@njit(cache=True)
def _vertex_drop_clipped(grad, A_W, scale, lo, hi, d):
    """max over { lo <= delta <= hi, sum delta = 0 } of (-grad . delta - scale * delta^T A_W delta).

    EXACT (vertex enumeration of the simplex-constrained box). Sound.
    Returns >= 0 when 0 is feasible (it is, since lo <= 0 <= hi for our cells).
    """
    best = 0.0  # delta = 0 always feasible and gives value 0
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

            # f(delta) = -grad.delta - scale * delta^T A_W delta
            lin = 0.0
            for i in range(d):
                lin += grad[i] * delta[i]
            quad = 0.0
            for i in range(d):
                Ai_dot_delta = 0.0
                for j in range(d):
                    Ai_dot_delta += A_W[i, j] * delta[j]
                quad += delta[i] * Ai_dot_delta
            val = -lin - scale * quad
            if val > best:
                best = val
    return best


@njit(cache=True)
def _box_certify_cell_vertex(mu_center, d, delta_q, c_target):
    """EXACT vertex-enum box-cert for d <= 16.

    For each window W=(ell, s), compute a RIGOROUS lower bound on
    min_{mu in cell} TV_W(mu) via vertex enumeration. Take max over W.
    Certified iff that max >= c_target.

    Returns (certified, best_min_tv).
    """
    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo_i = -h
        if -mu_center[i] > lo_i:
            lo_i = -mu_center[i]
        hi_i = h
        rem_to_one = 1.0 - mu_center[i]
        if rem_to_one < hi_i:
            hi_i = rem_to_one
        lo[i] = lo_i
        hi[i] = hi_i

    best_min_tv = 0.0
    A_W = np.zeros((d, d), dtype=np.float64)
    grad = np.empty(d, dtype=np.float64)

    for ell in range(2, 2 * d + 1):
        scale = two_d / np.float64(ell)
        s_hi_offset = ell - 2

        for s in range(conv_len - ell + 2):
            # Build A_W (zero, then mark contributing pairs)
            for i in range(d):
                for j in range(d):
                    A_W[i, j] = 0.0
            s_hi_idx = s + s_hi_offset
            for i in range(d):
                j_lo = s - i
                j_hi = s_hi_idx - i
                if j_lo < 0:
                    j_lo = 0
                if j_hi > d - 1:
                    j_hi = d - 1
                for j in range(j_lo, j_hi + 1):
                    A_W[i, j] = 1.0

            # TV_W(mu*)  and grad = 2*scale * A_W mu*
            tv_center = 0.0
            for i in range(d):
                Ai_dot_mu = 0.0
                for j in range(d):
                    Ai_dot_mu += A_W[i, j] * mu_center[j]
                grad[i] = 2.0 * scale * Ai_dot_mu
                tv_center += mu_center[i] * Ai_dot_mu
            tv_center *= scale

            # vertex_drop = max over cell of (-grad.delta - scale * delta^T A_W delta)
            vertex_drop = _vertex_drop_clipped(grad, A_W, scale, lo, hi, d)

            min_tv_W = tv_center - vertex_drop  # rigorous LB on TV_W over the cell
            if min_tv_W > best_min_tv:
                best_min_tv = min_tv_W
            if best_min_tv >= c_target:
                return True, best_min_tv

    return best_min_tv >= c_target, best_min_tv



def _box_certify_cell_mccormick(mu_center, d, delta_q, c_target, ell_max=None):
    """McCormick LP fallback for d > VERTEX_ENUM_MAX_D.

    For each window W=(ell, s) we solve a McCormick LP relaxation of
        min_{mu in cell, sum mu = 1} TV_W(mu)
    which is a RIGOROUS LB on the true cell minimum (since McCormick relaxes
    each bilinear term mu_i*mu_j to its tightest convex/concave envelope).

    Slower than vertex enum (uses scipy.optimize.linprog), so dispatched only
    when vertex enum is infeasible (d > 16).

    Returns (certified, best_min_tv).
    """
    from scipy.optimize import linprog

    h = delta_q / 2.0
    conv_len = 2 * d - 1
    two_d = 2.0 * float(d)
    mu_star = np.asarray(mu_center, dtype=np.float64)
    lo = np.maximum(mu_star - h, 0.0)
    hi = np.minimum(mu_star + h, 1.0)

    if ell_max is None:
        ell_max = 2 * d

    best_min_tv = 0.0

    for ell in range(2, ell_max + 1):
        scale = two_d / float(ell)
        for s in range(conv_len - ell + 2):
            # W_pairs: (i,j) with i<=j, s <= i+j <= s+ell-2
            W_pairs = []
            for i in range(d):
                for j in range(i, d):
                    if s <= i + j <= s + ell - 2:
                        W_pairs.append((i, j))
            n_pairs = len(W_pairs)
            if n_pairs == 0:
                continue

            n_vars = d + n_pairs
            c_obj = np.zeros(n_vars, dtype=np.float64)
            for k, (i, j) in enumerate(W_pairs):
                c_obj[d + k] = scale if i == j else 2.0 * scale

            bounds = []
            for i in range(d):
                bounds.append((float(lo[i]), float(hi[i])))
            for (i, j) in W_pairs:
                bounds.append((lo[i] * lo[j], hi[i] * hi[j]))

            A_eq = np.zeros((1, n_vars), dtype=np.float64)
            A_eq[0, :d] = 1.0
            b_eq = np.array([1.0])

            A_ub_rows = []
            b_ub_vals = []
            for k, (i, j) in enumerate(W_pairs):
                w_idx = d + k
                li, hi_i = lo[i], hi[i]
                lj, hj = lo[j], hi[j]
                if i == j:
                    # tangent lower bounds (3): w >= 2*p*mu - p^2 for p in {mu*_i, lo_i, hi_i}
                    for tangent_p in (mu_star[i], li, hi_i):
                        row = np.zeros(n_vars)
                        row[w_idx] = -1.0
                        row[i] = 2.0 * tangent_p
                        A_ub_rows.append(row)
                        b_ub_vals.append(tangent_p * tangent_p)
                    # secant upper: w <= (lo+hi)*mu - lo*hi
                    row = np.zeros(n_vars)
                    row[w_idx] = 1.0
                    row[i] = -(li + hi_i)
                    A_ub_rows.append(row)
                    b_ub_vals.append(-li * hi_i)
                else:
                    # Lower envelopes
                    row = np.zeros(n_vars); row[w_idx] = -1.0; row[j] = li; row[i] = lj
                    A_ub_rows.append(row); b_ub_vals.append(li * lj)
                    row = np.zeros(n_vars); row[w_idx] = -1.0; row[j] = hi_i; row[i] = hj
                    A_ub_rows.append(row); b_ub_vals.append(hi_i * hj)
                    # Upper envelopes
                    row = np.zeros(n_vars); row[w_idx] = 1.0; row[j] = -hi_i; row[i] = -lj
                    A_ub_rows.append(row); b_ub_vals.append(-hi_i * lj)
                    row = np.zeros(n_vars); row[w_idx] = 1.0; row[j] = -li; row[i] = -hj
                    A_ub_rows.append(row); b_ub_vals.append(-li * hj)

            A_ub = np.asarray(A_ub_rows, dtype=np.float64)
            b_ub = np.asarray(b_ub_vals, dtype=np.float64)

            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if not result.success:
                # LP failed — refuse to certify this cell (sound: -inf is a no-op)
                return False, best_min_tv
            min_tv_W = float(result.fun)
            if min_tv_W > best_min_tv:
                best_min_tv = min_tv_W
            if best_min_tv >= c_target:
                return True, best_min_tv

    return best_min_tv >= c_target, best_min_tv


def _box_certify_cell(mu_center, d, delta_q, c_target):
    """Dispatch box-cert to vertex enum (d <= 16) or McCormick LP (d > 16).

    Both paths are RIGOROUS lower bounds on min_{mu in cell} max_W TV_W(mu).
    Returns (certified, best_min_tv).
    """
    if d <= VERTEX_ENUM_MAX_D:
        return _box_certify_cell_vertex(np.asarray(mu_center, dtype=np.float64),
                                         d, float(delta_q), float(c_target))
    return _box_certify_cell_mccormick(mu_center, d, delta_q, c_target)


@njit(parallel=True, cache=True)
def _box_certify_batch_vertex(batch_int, d, S, c_target,
                               cert_out, min_tv_out):
    """Parallel vertex-enum box-cert (no Tier 1). Kept for diagnostic comparison."""
    B = batch_int.shape[0]
    delta_q = 1.0 / np.float64(S)
    for b in prange(B):
        mu = np.empty(d, dtype=np.float64)
        for i in range(d):
            mu[i] = np.float64(batch_int[b, i]) / np.float64(S)
        cert, mt = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        cert_out[b] = 1 if cert else 0
        min_tv_out[b] = mt


@njit(parallel=True, cache=True)
def _box_certify_batch_adaptive(batch_int, d, S, c_target,
                                 cert_out, min_tv_out, tier_out):
    """ADAPTIVE 2-tier box-cert (Tier 1 Lipschitz LB, Tier 3 vertex enum).

    Tier 1 (cheap, sound): _phase1_lipschitz_lb. Certifies ~99% of cells.
    Tier 3 (expensive, exact): _box_certify_cell_vertex. Only invoked when
    Tier 1 fails. Soundness: Tier 1 LB <= true cell minimum, so cert at
    Tier 1 implies cert at Tier 3.

    cert_out[b] = 1 if cell certified at any tier, 0 if all tiers fail
    min_tv_out[b] = best LB on cell minimum across tiers reached
    tier_out[b]   = 1 (passed at Tier 1) or 3 (needed Tier 3) — diagnostic
    """
    B = batch_int.shape[0]
    delta_q = 1.0 / np.float64(S)
    for b in prange(B):
        mu = np.empty(d, dtype=np.float64)
        for i in range(d):
            mu[i] = np.float64(batch_int[b, i]) / np.float64(S)
        # Tier 1: fast Lipschitz LB
        cert1, lb1 = _phase1_lipschitz_lb(mu, d, delta_q, c_target)
        if cert1:
            cert_out[b] = 1
            min_tv_out[b] = lb1
            tier_out[b] = 1
            continue
        # Tier 3: exact vertex enumeration (only when Tier 1 fails)
        cert3, lb3 = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        cert_out[b] = 1 if cert3 else 0
        # Report the tighter LB (vertex is tighter than Tier 1)
        min_tv_out[b] = lb3 if lb3 > lb1 else lb1
        tier_out[b] = 3


@njit(parallel=True, cache=True)
def _box_certify_batch_adaptive_v2(batch_int, d, S, c_target,
                                    qdrop_table, V_table, lam_table, valid_mask,
                                    cert_out, min_tv_out, tier_out):
    """Speedup #2 ADAPTIVE 4-tier box-cert (with low-overhead fast path).

      Tier 1a: _phase1_lipschitz_lb (original; cheapest, no table overhead)
      Tier 1b: _phase1_lipschitz_lb_spec (spectral qd_c via qdrop_table)
      Tier 2:  _box_certify_cell_trust_region (closed-form QP via L2 ball)
      Tier 3:  _box_certify_cell_vertex (exact vertex enum)

    Each subsequent tier is strictly tighter (sound). The original Phase 1
    runs first to avoid table-lookup overhead on the ~99% of cells that
    already pass it; only failed cells pay the cost of tier 1b/2.

    Soundness: each LB is <= true cell-min, so any cert is sound. Both Tier 1a
    and Tier 1b report tier_out = 1 (they're "Phase 1 family").

    Inputs:
      qdrop_table[ell, s] = max(0, -lambda_min^V(A_W))
      V_table[ell, s, :, :] = eigenvectors of P A_W P
      lam_table[ell, s, :]  = eigenvalues
      valid_mask[ell, s]    = 1 if window valid
    """
    B = batch_int.shape[0]
    delta_q = 1.0 / np.float64(S)
    for b in prange(B):
        mu = np.empty(d, dtype=np.float64)
        for i in range(d):
            mu[i] = np.float64(batch_int[b, i]) / np.float64(S)
        # Tier 1a: original Phase 1 (no table overhead — fast path)
        cert1a, lb1a = _phase1_lipschitz_lb(mu, d, delta_q, c_target)
        if cert1a:
            cert_out[b] = 1
            min_tv_out[b] = lb1a
            tier_out[b] = 1
            continue
        # Tier 1b: spectral Phase 1 (tighter qd_c via qdrop_table)
        cert1b, lb1b = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop_table)
        if cert1b:
            cert_out[b] = 1
            best_lb = lb1b if lb1b > lb1a else lb1a
            min_tv_out[b] = best_lb
            tier_out[b] = 1
            continue
        # Tier 2: BADTR (per-eigenmode trust-region — strictly tighter than L_2 ball alone)
        cert2, lb2 = _box_certify_cell_badtr(
            mu, d, delta_q, c_target, V_table, lam_table, valid_mask)
        if cert2:
            cert_out[b] = 1
            best_lb = lb2
            if lb1b > best_lb: best_lb = lb1b
            if lb1a > best_lb: best_lb = lb1a
            min_tv_out[b] = best_lb
            tier_out[b] = 2
            continue
        # Tier 2.5: CCTR (convex-combination trust-region — closes minimax gap)
        cert25, lb25 = _box_certify_cell_cctr(
            mu, d, delta_q, c_target, V_table, lam_table, valid_mask)
        if cert25:
            cert_out[b] = 1
            best_lb = lb25
            if lb2 > best_lb: best_lb = lb2
            if lb1b > best_lb: best_lb = lb1b
            if lb1a > best_lb: best_lb = lb1a
            min_tv_out[b] = best_lb
            tier_out[b] = 2  # report as Tier 2 (still non-vertex)
            continue
        # Tier 3: exact vertex enum (Tier 1a, 1b, 2, and 2.5 all failed)
        cert3, lb3 = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        cert_out[b] = 1 if cert3 else 0
        best_lb = lb3
        if lb25 > best_lb: best_lb = lb25
        if lb2 > best_lb: best_lb = lb2
        if lb1b > best_lb: best_lb = lb1b
        if lb1a > best_lb: best_lb = lb1a
        min_tv_out[b] = best_lb
        tier_out[b] = 3


def run_box_certification(d_final, S, c_target, verbose=True,
                          fail_fast=False, max_show_failures=5):
    """EXHAUSTIVE box-certify every canonical (b <= rev(b)) composition's cell.

    Soundness (§1 of COARSE_CASCADE_PROVER_FIXES.md): a rigorous proof requires
    every cell to be certified, NOT a random sample. The composition space is
    deduped by reflection symmetry (Z2-canonical) since TV_W(rev(mu)) reflects
    to TV_W(mu) under (ell, s) -> (ell, 2d-ell-s), so cert(mu) <=> cert(rev(mu)).

    Compositions where any single bin exceeds x_cap = compute_xcap are skipped:
    such cells have TV >= c_target by self-conv alone (TV_W with ell=2, s=2i
    gives mu_i^2 * d >= c_target), and TV is monotone in mu_i so the entire
    cell — not just the grid point — has TV >= c_target.

    Returns (all_certified, n_failed, worst_min_tv, failure_compositions).
    """
    delta_q = 1.0 / S
    x_cap = compute_xcap(c_target, S, d_final)
    n_cert = 0
    n_fail = 0
    n_skipped_xcap = 0
    n_skipped_xcap_winner = 0  # cells where some mu_i > x_cap and entire cell >= c
    n_tier1 = 0  # cells certified by Phase 1 spectral
    n_tier2 = 0  # cells certified by trust-region QP (Tier 2)
    n_tier3 = 0  # cells that needed exact vertex enum
    worst_tv = 1e30
    failures = []
    t0 = time.time()
    n_total_seen = 0

    if verbose:
        print(f"\n  Box certification at d={d_final}, S={S}, "
              f"delta={delta_q:.4f}:")
        print(f"    EXHAUSTIVE over canonical compositions (no sampling).")
        if d_final > VERTEX_ENUM_MAX_D:
            print(f"    d={d_final} > {VERTEX_ENUM_MAX_D}: using McCormick LP "
                  f"(slower; one cell at a time).")

    use_vertex = d_final <= VERTEX_ENUM_MAX_D
    # Precompute spectral tables for Speedup #2 (Phase 1 spectral + trust-region).
    if use_vertex:
        if verbose:
            print(f"    Precomputing spectral tables for d={d_final}...")
        t_pre = time.time()
        qdrop_table = compute_qdrop_table(d_final)
        V_table, lam_table, valid_mask = compute_window_eigen_table(d_final)
        if verbose:
            print(f"    Spectral precompute done in {time.time()-t_pre:.2f}s.")

    last_progress_t = t0
    # Default batch_size (100K) is well-tuned. Larger batches stall the
    # parallel workers because the composition generator is single-threaded.
    for batch in generate_canonical_compositions_batched(d_final, S):
        # Filter batch: drop rows with any bin > x_cap (already TV >= c by self-conv).
        if x_cap < S:
            keep_mask = np.all(batch <= x_cap, axis=1)
            n_drop = int(np.sum(~keep_mask))
            n_skipped_xcap_winner += n_drop
            batch = batch[keep_mask]
        n_total_seen += batch.shape[0] + (n_drop if x_cap < S else 0)
        if batch.shape[0] == 0:
            continue

        # Periodic progress report (every 30s) for long-running runs
        now = time.time()
        if verbose and (now - last_progress_t) > 30.0:
            print(f"    [progress @ {now-t0:.0f}s] processed {n_cert + n_fail:,} cells, "
                  f"failed: {n_fail}, worst TV: {worst_tv:.6f}, "
                  f"Tier 1: {n_tier1:,} ({100*n_tier1/max(n_tier1+n_tier3,1):.1f}%)",
                  flush=True)
            last_progress_t = now

        if use_vertex:
            cert_out = np.zeros(batch.shape[0], dtype=np.int8)
            min_tv_out = np.zeros(batch.shape[0], dtype=np.float64)
            tier_out = np.zeros(batch.shape[0], dtype=np.int8)
            _box_certify_batch_adaptive_v2(
                batch, d_final, S, c_target,
                qdrop_table, V_table, lam_table, valid_mask,
                cert_out, min_tv_out, tier_out)
            n_tier1 += int(np.sum(tier_out == 1))
            n_tier2 += int(np.sum(tier_out == 2))
            n_tier3 += int(np.sum(tier_out == 3))
            for b in range(batch.shape[0]):
                if cert_out[b]:
                    n_cert += 1
                else:
                    n_fail += 1
                    if len(failures) < max_show_failures:
                        failures.append(batch[b].copy())
                if min_tv_out[b] < worst_tv:
                    worst_tv = float(min_tv_out[b])
            if fail_fast and n_fail > 0:
                break
        else:
            # McCormick path — Python loop, not Numba-parallel.
            for b in range(batch.shape[0]):
                mu = batch[b].astype(np.float64) / S
                cert, mt = _box_certify_cell_mccormick(
                    mu, d_final, delta_q, c_target)
                if cert:
                    n_cert += 1
                else:
                    n_fail += 1
                    if len(failures) < max_show_failures:
                        failures.append(batch[b].copy())
                if mt < worst_tv:
                    worst_tv = mt
                if fail_fast and n_fail > 0:
                    break
            if fail_fast and n_fail > 0:
                break

    elapsed = time.time() - t0
    total = n_cert + n_fail
    all_certified = (n_fail == 0)

    if verbose:
        print(f"    Cells certified: {n_cert:,}/{total:,} "
              f"({100*n_cert/max(total,1):.2f}%)")
        n_tiered = n_tier1 + n_tier2 + n_tier3
        if n_tiered > 0:
            tier1_pct = 100 * n_tier1 / max(n_tiered, 1)
            tier2_pct = 100 * n_tier2 / max(n_tiered, 1)
            print(f"    Tier 1 (spectral Phase 1): {n_tier1:,} cells "
                  f"({tier1_pct:.1f}%)  |  Tier 2 (trust-region): {n_tier2:,} "
                  f"({tier2_pct:.1f}%)  |  Tier 3 (vertex enum): {n_tier3:,}")
        if n_skipped_xcap_winner > 0:
            print(f"    Cells skipped (any bin > x_cap, "
                  f"trivially TV >= c_target): {n_skipped_xcap_winner:,}")
        print(f"    Worst QP min TV: {worst_tv:.6f} (need >= {c_target})")
        print(f"    Time: {elapsed:.1f}s "
              f"({total/max(elapsed,1e-6):.0f} cells/s)")
        if all_certified:
            print(f"    [BOX-CERT PASS] every cell certified — proof valid.")
        else:
            print(f"    [BOX-CERT FAIL] {n_fail} cells failed — proof INVALID.")
            for f in failures:
                print(f"      failed cell: {f.tolist()}")

    return all_certified, n_fail, worst_tv, failures


# =====================================================================
# Main cascade driver
# =====================================================================

def run_cascade(c_target=1.30, S=50, d_start=2, max_levels=5, verbose=True,
                auto_s_shift=True, fail_fast=True):
    """Run the full coarse cascade proof.

    Returns True if the proof succeeds (C_{1a} >= c_target).

    If auto_s_shift=True (default), S is auto-shifted to dodge the integer
    threshold lattice (where margin = 0 exactly). See s_shift_safe().

    If fail_fast=True (default), box certification stops at the first failed
    cell (the proof is invalid regardless, and stopping early saves time when
    debugging). Set False if you want a full diagnostic of all failures.
    """
    # §4: shift S off the integer-threshold lattice (best-effort; some lattice
    # hits are unavoidable for rational c with small denominator, but minimizing
    # them reduces the number of margin=0 cells the box-cert has to handle).
    if auto_s_shift:
        d_max = d_start * (2 ** max_levels)
        S_safe = s_shift_safe(c_target, S, d_max, strict=False)
        if S_safe != S:
            n_hits = len(count_lattice_offenders(c_target, S_safe, d_max))
            if verbose:
                print(f"  [s_shift_safe] S={S} -> S={S_safe} "
                      f"(remaining lattice hits: {n_hits})")
            S = S_safe
        else:
            n_hits = len(count_lattice_offenders(c_target, S, d_max))
            if n_hits > 0 and verbose:
                print(f"  [s_shift_safe] S={S} has {n_hits} unavoidable "
                      f"lattice hits for c_target={c_target} (rational c with "
                      f"small denominator).")

    if verbose:
        print("=" * 64)
        print(f"COARSE CASCADE PROVER: C_{{1a}} >= {c_target}")
        print("=" * 64)
        print(f"  Grid: S={S} (delta={1/S:.4f})")
        print(f"  Starting dimension: d={d_start}")
        print(f"  No correction term (refinement monotonicity)")
        print()

    t_total = time.time()

    # --- L0 ---
    d = d_start
    if verbose:
        print(f"  L0 (d={d}):")
    t0 = time.time()
    survivors, n_surv, n_tested = run_l0(d, S, c_target)
    elapsed = time.time() - t0

    if n_surv > 0:
        _canonicalize_inplace(survivors)
        survivors = dedup(survivors)
        n_surv = len(survivors)

    if verbose:
        print(f"    Tested: {n_tested:,}")
        print(f"    Survivors: {n_surv:,}")
        print(f"    Time: {elapsed:.2f}s")

    if n_surv == 0:
        if verbose:
            print(f"\n  GRID-POINT CONVERGENCE at L0 (d={d}).")
            print(f"  Running EXHAUSTIVE box certification...")
        all_certified, n_failed, worst_tv, _ = run_box_certification(
            d, S, c_target, verbose=verbose, fail_fast=fail_fast)
        total_time = time.time() - t_total
        if all_certified:
            if verbose:
                print(f"\n  {'=' * 60}")
                print(f"  RIGOROUS PROOF: C_{{1a}} >= {c_target}")
                print(f"  Method: coarse cascade (S={S}, d={d}) + exact box cert")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  {'=' * 60}")
            return True
        else:
            if verbose:
                print(f"\n  GRID-POINT proof obtained, but box certification "
                      f"FAILED on {n_failed} cell(s).")
                print(f"  Worst min TV: {worst_tv:.6f} (need >= {c_target}).")
                print(f"  C_{{1a}} >= {c_target} NOT proven (continuum gap).")
                print(f"  Total time: {total_time:.2f}s")
            return False

    # Save checkpoint
    np.save(f"data/coarse_L0_survivors_S{S}.npy", survivors)

    # --- L1+ ---
    for level in range(1, max_levels + 1):
        d_parent = d
        d = 2 * d_parent

        if verbose:
            print(f"\n  L{level} (d={d}):")

        t0 = time.time()
        survivors, n_surv, n_tested = run_cascade_level(
            survivors, d_parent, S, c_target, verbose=verbose)
        elapsed = time.time() - t0

        if n_surv > 0:
            _canonicalize_inplace(survivors)
            survivors = dedup(survivors)
            n_surv = len(survivors)

        if verbose:
            print(f"    Tested: {n_tested:,}")
            print(f"    Survivors (after dedup): {n_surv:,}")
            print(f"    Time: {elapsed:.2f}s")

        if n_surv == 0:
            if verbose:
                print(f"\n  GRID-POINT CONVERGENCE at L{level} (d={d}).")
                print(f"  All {n_tested:,} children pruned by TV >= {c_target}.")

            # Box certification (EXHAUSTIVE — sound)
            if verbose:
                print(f"\n  Running EXHAUSTIVE box certification...")
            all_certified, n_failed, worst_tv, _ = run_box_certification(
                d, S, c_target, verbose=verbose, fail_fast=fail_fast)

            total_time = time.time() - t_total
            if not all_certified:
                if verbose:
                    print(f"\n  GRID-POINT proof obtained, but box certification "
                          f"FAILED on {n_failed} cell(s).")
                    print(f"  Worst min TV: {worst_tv:.6f} "
                          f"(need >= {c_target}).")
                    print(f"  C_{{1a}} >= {c_target} NOT proven (continuum gap).")
                    print(f"  Total time: {total_time:.2f}s")
                return False

            if verbose:
                print(f"\n  {'=' * 60}")
                print(f"  RIGOROUS PROOF: C_{{1a}} >= {c_target}")
                print(f"  Method: coarse cascade (S={S}) + exact box cert")
                print(f"  Converged at d={d} (L{level})")
                print(f"  Worst cell min TV: {worst_tv:.6f}")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  {'=' * 60}")
            return True

        # Save checkpoint
        np.save(f"data/coarse_L{level}_survivors_S{S}.npy", survivors)

    total_time = time.time() - t_total
    if verbose:
        print(f"\n  Did not converge within {max_levels} levels.")
        print(f"  Survivors at d={d}: {n_surv:,}")
        print(f"  Total time: {total_time:.2f}s")
    return False


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coarse cascade prover for C_{1a}")
    parser.add_argument("--c_target", type=float, default=1.30,
                        help="Target lower bound (default: 1.30)")
    parser.add_argument("--S", type=int, default=50,
                        help="Grid resolution S (mass quantum = 1/S)")
    parser.add_argument("--d_start", type=int, default=2,
                        help="Starting dimension")
    parser.add_argument("--max_levels", type=int, default=5,
                        help="Maximum cascade levels")
    parser.add_argument("--no_s_shift", action="store_true",
                        help="Disable auto S-shift (keep S even if on integer lattice)")
    parser.add_argument("--n_threads", type=int, default=0,
                        help="Numba thread count (0 = auto/all cores)")
    parser.add_argument("--no_fail_fast", action="store_true",
                        help="Continue past first cell failure (slower but full diagnostic)")
    args = parser.parse_args()

    if args.n_threads > 0:
        numba.set_num_threads(args.n_threads)
    print(f"Numba threads: {numba.get_num_threads()}")

    # JIT warmup
    print("Warming up JIT...", end="", flush=True)
    t0 = time.time()
    _warmup_thr = compute_thresholds(1.3, 10, 4)
    _warmup_buf = np.empty((0, 4), dtype=np.int32)
    _l0_bnb_inner(np.int32(2), 4, 10, 5, _warmup_thr, _warmup_buf, True)
    _warmup_parent = np.array([3, 3, 2, 2], dtype=np.int32)
    _warmup_buf2 = np.empty((100, 8), dtype=np.int32)
    _warmup_thr2 = compute_thresholds(1.3, 10, 8)
    _cascade_child_bnb(_warmup_parent, 4, 10, 5, _warmup_thr2, _warmup_buf2)
    _canonicalize_inplace(np.array([[1, 2, 3, 4]], dtype=np.int32))
    # Warmup vertex-enum box-cert and parallel batch
    _warmup_mu = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    _box_certify_cell_vertex(_warmup_mu, 4, 0.1, 1.05)
    _phase1_lipschitz_lb(_warmup_mu, 4, 0.1, 1.05)
    # Speedup #2 warmup: spectral Phase 1 + trust-region cell-min
    _warmup_qdrop = compute_qdrop_table(4)
    _warmup_V, _warmup_lam, _warmup_valid = compute_window_eigen_table(4)
    _phase1_lipschitz_lb_spec(_warmup_mu, 4, 0.1, 1.05, _warmup_qdrop)
    _box_certify_cell_trust_region(_warmup_mu, 4, 0.1, 1.05,
                                    _warmup_V, _warmup_lam, _warmup_valid)
    _box_certify_cell_badtr(_warmup_mu, 4, 0.1, 1.05,
                              _warmup_V, _warmup_lam, _warmup_valid)
    _box_certify_cell_cctr(_warmup_mu, 4, 0.1, 1.05,
                             _warmup_V, _warmup_lam, _warmup_valid)
    _warmup_batch = np.array([[2, 3, 3, 2]], dtype=np.int32)
    _wcert = np.zeros(1, dtype=np.int8)
    _wmin = np.zeros(1, dtype=np.float64)
    _wtier = np.zeros(1, dtype=np.int8)
    _box_certify_batch_vertex(_warmup_batch, 4, 10, 1.05, _wcert, _wmin)
    _box_certify_batch_adaptive(_warmup_batch, 4, 10, 1.05, _wcert, _wmin, _wtier)
    _box_certify_batch_adaptive_v2(
        _warmup_batch, 4, 10, 1.05,
        _warmup_qdrop, _warmup_V, _warmup_lam, _warmup_valid,
        _wcert, _wmin, _wtier)
    print(f" done ({time.time()-t0:.1f}s)")

    os.makedirs("data", exist_ok=True)
    success = run_cascade(
        c_target=args.c_target,
        S=args.S,
        d_start=args.d_start,
        max_levels=args.max_levels,
        auto_s_shift=not args.no_s_shift,
        fail_fast=not args.no_fail_fast,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

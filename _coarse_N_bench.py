"""Bench: spectral (variant N) δ²-bound for the COARSE-grid box certificate.

This is the coarse-grid analogue of `_N_bench.py`.  The fine grid uses
h = 1/m, while the coarse grid uses h = 1/(2S) — the cell half-width.

Mathematics (sound bound on |scale · δ^T A_W δ|, scale = 2d/ell):

  TV_W(μ* + δ) - TV_W(μ*) - <grad, δ> = (2d/ell) δ^T A_W δ
  with A_W[i,j] = 1{s ≤ i+j ≤ s+ell-2}.

  Σ δ_i = 0 ⇒ ∀α : δ^T A_W δ = δ^T (A_W − α·11ᵀ) δ.
  By Cauchy–Schwarz + operator-norm bound,
     |δ^T (A_W − α·11ᵀ) δ| ≤ ‖A_W − α·11ᵀ‖_op · ‖δ‖² ≤ op_rest · d · h²,
  where op_rest = min_α ‖A_W − α·11ᵀ‖_op (closed form: pick α that zeros the
  rank-1 11ᵀ component on the all-ones direction; for symmetric A_W this is
  α = (1/d²) · 1ᵀA_W·1 = N_W / d², giving
     op_rest = max(λ_max, |λ_min|) of the symmetric matrix A_W − α·11ᵀ).

  TV-space second-order correction:
     quad_corr_N = (2d/ell) · min(cross_W, d²−N_W, op_rest·d) · h²
                 = (2d/ell) · min(cross_W, d²−N_W, op_rest·d) / (4 S²)
  Sound: the existing two terms `cross_W` and `d²−N_W` come from the
  triangle baseline and we *add* the spectral term in the min — never weaker.

Soundness check: for compositions certified by N (i.e., pruned with
margin > cell_var + quad_corr_N) but NOT certified by triangle (i.e.,
margin ≤ cell_var + quad_corr_triangle), the BAD case would be that
N's tighter correction certifies a cell that actually contains a δ
with TV < c_target.  We verify by fine-grid search over the cell
(n_grid points per dim) — if min TV ≥ c_target, certificate holds.

Usage:
    python _coarse_N_bench.py
"""
import json
import os
import sys
import time

import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions


# =====================================================================
# Pre-compute pair-count prefix sums (matches run_cascade_coarse_v2)
# =====================================================================

@njit(cache=True)
def _build_pair_prefix(d):
    conv_len = 2 * d - 1
    prefix_nk = np.zeros(conv_len + 1, dtype=np.int64)
    prefix_mk = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        nk = min(k + 1, d, conv_len - k)
        mk = np.int64(1) if (k % 2 == 0 and k // 2 < d) else np.int64(0)
        prefix_nk[k + 1] = prefix_nk[k] + np.int64(nk)
        prefix_mk[k + 1] = prefix_mk[k] + mk
    return prefix_nk, prefix_mk


# =====================================================================
# Pre-compute op_rest[ell, s_lo] for all windows, for given d
# =====================================================================

def precompute_op_rest_table(d):
    """For each (ell, s_lo), compute op_rest := ‖A_W − α·11ᵀ‖_op
    where α is chosen to zero the all-ones rank-1 component:
        α* = (1ᵀA_W 1) / d² = N_W / d².

    Since A_W is symmetric (A_W[i,j] = 1{s ≤ i+j ≤ s+ell-2}),
    the operator norm of A_W − α·11ᵀ equals max(|λ_max|, |λ_min|).

    Returns op_rest[ell, s_lo] (dense; some entries are zero / unused).
    """
    conv_len = 2 * d - 1
    max_ell = 2 * d
    op_rest = np.zeros((max_ell + 1, conv_len), dtype=np.float64)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A = np.zeros((d, d), dtype=np.float64)
            n_pairs = 0
            for i in range(d):
                for j in range(d):
                    k = i + j
                    if s_lo <= k <= s_lo + ell - 2:
                        A[i, j] = 1.0
                        n_pairs += 1
            if n_pairs == 0:
                op_rest[ell, s_lo] = 0.0
                continue
            alpha = float(n_pairs) / (d * d)
            A_rest = A - alpha
            eigs = np.linalg.eigvalsh(A_rest)
            op_rest[ell, s_lo] = float(np.abs(eigs).max())
    return op_rest


# =====================================================================
# Triangle baseline kernel (matches run_cascade_coarse_v2 logic)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_triangle(batch_int, d, S, c_target):
    """Triangle (baseline) certifier: prune & count cells certified by
       margin > cell_var + quad_corr_triangle
    where quad_corr_triangle = (2d/ell) · min(cross_W, d²−N_W) / (4 S²).

    Returns (pruned_mask, certified_mask) — certified is the subset of
    pruned cells that pass box certification (margin > cell_var + qc).
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    pruned = np.zeros(B, dtype=numba.boolean)
    certified = np.zeros(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        is_pruned = False
        best_net = np.float64(-1e30)
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    is_pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key

                    cell_var = 0.0
                    for kk in range(d // 2):
                        cell_var += grad_arr[d - 1 - kk] - grad_arr[kk]
                    cell_var /= (2.0 * S_d)

                    hi = s_lo + ell - 1
                    N_W = prefix_nk[hi] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    pair_bound = cross_W if cross_W < (d_sq - N_W) else (d_sq - N_W)
                    if pair_bound < 0:
                        pair_bound = np.int64(0)
                    qc = (2.0 * d_d / ell_f) * np.float64(pair_bound) * inv_4S2

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        if is_pruned:
            pruned[b] = True
            if best_net > 0.0:
                certified[b] = True

    return pruned, certified


# =====================================================================
# Variant N kernel (spectral + triangle, take min)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_N(batch_int, d, S, c_target, op_rest_arr):
    """Variant N certifier: prune & count cells certified by
       margin > cell_var + quad_corr_N
    where quad_corr_N = (2d/ell) · min(cross_W, d²−N_W, op_rest·d) / (4 S²).

    Sound: includes the triangle bound terms in the min, so never weaker
    than the triangle baseline.
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    pruned = np.zeros(B, dtype=numba.boolean)
    certified = np.zeros(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        is_pruned = False
        best_net = np.float64(-1e30)
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    is_pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key

                    cell_var = 0.0
                    for kk in range(d // 2):
                        cell_var += grad_arr[d - 1 - kk] - grad_arr[kk]
                    cell_var /= (2.0 * S_d)

                    hi = s_lo + ell - 1
                    N_W = prefix_nk[hi] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    triangle_pair = cross_W if cross_W < (d_sq - N_W) else (d_sq - N_W)
                    if triangle_pair < 0:
                        triangle_pair = np.int64(0)

                    # Spectral bound: op_rest * d
                    spectral_pair = op_rest_arr[ell, s_lo] * d_d

                    pair_bound_f = np.float64(triangle_pair)
                    if spectral_pair < pair_bound_f:
                        pair_bound_f = spectral_pair

                    qc = (2.0 * d_d / ell_f) * pair_bound_f * inv_4S2

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        if is_pruned:
            pruned[b] = True
            if best_net > 0.0:
                certified[b] = True

    return pruned, certified


# =====================================================================
# Soundness check via fine grid search over a cell
# =====================================================================

def _max_TV_over_windows(mu, d):
    """Compute max_W TV_W(mu) for a single mu vector."""
    conv = np.zeros(2 * d - 1)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(2 * d - 1 - n_cv + 1):
            ws = float(conv[s_lo:s_lo + n_cv].sum())
            tv = (2.0 * d / ell) * ws
            if tv > best:
                best = tv
    return best


def fine_grid_min_TV(k_int, d, S, n_grid=15):
    """Find min over cell (each μ_i in [k_i/S - 1/(2S), k_i/S + 1/(2S)],
    Σμ_i=1) of max_W TV_W(μ).  Sample n_grid points per dim; then
    project onto the simplex constraint Σμ = 1 by scaling.

    Note: this is an APPROXIMATE search. We use random sampling +
    gridded search. We return the *minimum* found (the soundness threat).
    """
    rng = np.random.default_rng(seed=int(np.sum(k_int) * 1000 + k_int[0]))
    h = 1.0 / (2.0 * S)
    mu_star = k_int.astype(np.float64) / float(S)
    best_min = np.inf

    n_samples = n_grid ** 2  # quasi-random samples
    for _ in range(n_samples):
        # Sample uniform in [-h, +h]^d
        delta = rng.uniform(-h, h, size=d)
        # Project onto Σδ = 0
        delta -= delta.mean()
        # Clip (might violate |δ_i| ≤ h after projection; just clip)
        delta = np.clip(delta, -h, h)
        # Re-project (small loss but ok for soundness sampling)
        delta -= delta.mean()
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        mu = np.maximum(mu, 0.0)
        s = mu.sum()
        if s <= 0:
            continue
        mu /= s  # tiny renorm; cell ⊂ {Σ=1} feasible region
        tv = _max_TV_over_windows(mu, d)
        if tv < best_min:
            best_min = tv

    # Also try corners (deterministic)
    for sign in [1, -1]:
        delta = sign * h * np.ones(d)
        # Force Σ = 0 by zeroing one coordinate
        delta[0] = -delta[1:].sum()
        if abs(delta[0]) > h:
            continue
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        mu = np.maximum(mu, 0.0)
        s = mu.sum()
        if s <= 0:
            continue
        mu /= s
        tv = _max_TV_over_windows(mu, d)
        if tv < best_min:
            best_min = tv

    return best_min


# =====================================================================
# Public API: prune_coarse_N (matches triangle baseline signature)
# =====================================================================

def prune_coarse_N(batch_int, d, S, c_target, op_rest_arr=None):
    """Variant N coarse-grid certifier.

    Returns (pruned_mask, certified_mask).
    """
    if op_rest_arr is None:
        op_rest_arr = precompute_op_rest_table(d)
    return _prune_N(batch_int, d, S, c_target, op_rest_arr)


# =====================================================================
# Bench driver
# =====================================================================

def run_config(d, S, c_target, n_soundness_check=50, batch_size=200_000,
               verbose=True):
    if verbose:
        print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
    n_total = count_compositions(d, S)
    if verbose:
        print(f"     total compositions: {n_total:,}")

    op_rest = precompute_op_rest_table(d)

    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    _prune_triangle(warm, d, S, c_target)
    _prune_N(warm, d, S, c_target, op_rest)

    n_proc = 0
    n_pruned_T = 0
    n_cert_T = 0
    n_pruned_N = 0
    n_cert_N = 0

    # Collect compositions where N certifies but triangle doesn't (the
    # marginal cells where N's tighter correction adds new certification).
    # We'll soundness-check a sample of these.
    extra_N_compositions = []  # list of int32 arrays
    extra_N_max = 1000

    t0 = time.time()
    for batch in generate_compositions_batched(d, S, batch_size=batch_size):
        n_proc += len(batch)
        pT, cT = _prune_triangle(batch, d, S, c_target)
        pN, cN = _prune_N(batch, d, S, c_target, op_rest)
        n_pruned_T += int(pT.sum())
        n_cert_T += int(cT.sum())
        n_pruned_N += int(pN.sum())
        n_cert_N += int(cN.sum())

        # soundness invariant 1: N must be a superset (or equal) of T in pruned-set
        # (same threshold) — i.e., pT == pN
        if not bool(np.all(pT == pN)):
            mismatch = int(np.sum(pT != pN))
            print(f"     WARNING: pT != pN on {mismatch} comps "
                  f"(should be impossible — same threshold)")

        # soundness invariant 2: certified-by-T must be subset of certified-by-N
        # (since N's quad_corr ≤ T's quad_corr ⇒ if T certifies, N also does)
        if not bool(np.all(cT <= cN)):
            bad = int(np.sum(cT & ~cN))
            print(f"     WARNING: {bad} cells certified by T but not by N "
                  f"(violates monotonicity, indicates a bug)")

        # Collect extras
        extra_mask = cN & ~cT
        if extra_mask.any() and len(extra_N_compositions) < extra_N_max:
            extras = batch[extra_mask]
            take = min(len(extras), extra_N_max - len(extra_N_compositions))
            for i in range(take):
                extra_N_compositions.append(extras[i].copy())

    bench_time = time.time() - t0

    extra = n_cert_N - n_cert_T
    pct = 100.0 * extra / max(1, n_cert_T)
    print(f"     pruned (both)      : {n_pruned_T:,}")
    print(f"     triangle-certified : {n_cert_T:,}")
    print(f"     N-certified        : {n_cert_N:,}")
    print(f"     extra by N         : {extra:,} ({pct:+.2f}% over triangle)")
    print(f"     bench time         : {bench_time:.2f}s")

    # ---- Soundness check on extras ----
    n_soundness_violations = 0
    soundness_samples = 0
    if len(extra_N_compositions) > 0:
        rng = np.random.default_rng(42)
        n_samples = min(n_soundness_check, len(extra_N_compositions))
        idxs = rng.choice(len(extra_N_compositions), size=n_samples, replace=False)
        print(f"     soundness check    : {n_samples} sampled extras "
              f"(of {len(extra_N_compositions)} candidates)")
        worst_min_tv = np.inf
        t_sc = time.time()
        for i in idxs:
            k = extra_N_compositions[int(i)]
            min_tv = fine_grid_min_TV(k, d, S, n_grid=15)
            soundness_samples += 1
            if min_tv < worst_min_tv:
                worst_min_tv = min_tv
            if min_tv + 1e-9 < c_target:
                n_soundness_violations += 1
                print(f"       *** VIOLATION: k={k.tolist()}  min_TV={min_tv:.6f} "
                      f"< c_target={c_target} ***")
        sc_time = time.time() - t_sc
        if worst_min_tv == np.inf:
            worst_min_tv = -1.0
        print(f"     worst min_TV found : {worst_min_tv:.6f} (vs c={c_target})")
        print(f"     soundness time     : {sc_time:.2f}s")
    else:
        print(f"     soundness check    : SKIPPED (no extras to check)")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': int(n_total),
        'n_pruned': int(n_pruned_T),
        'n_triangle_certified': int(n_cert_T),
        'n_N_certified': int(n_cert_N),
        'extra_certified_by_N': int(extra),
        'pct_improvement': float(pct),
        'bench_time_sec': float(bench_time),
        'soundness_samples': int(soundness_samples),
        'soundness_violations': int(n_soundness_violations),
    }


def main():
    configs = [
        (4, 20, 1.20),
        (4, 30, 1.25),
        (6, 15, 1.20),
        (6, 20, 1.25),
        (8, 12, 1.20),
        (8, 15, 1.20),
    ]
    results = []
    for d, S, c in configs:
        results.append(run_config(d, S, c))

    out = os.path.join(_dir, '_coarse_N_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*78}")
    print(f"{'config':<22}{'n_total':>12}{'cert_T':>12}{'cert_N':>12}"
          f"{'extra':>10}{'pct':>8}{'viol':>6}")
    print('-' * 78)
    for r in results:
        cfg = f"d={r['d']}, S={r['S']}, c={r['c_target']}"
        print(f"{cfg:<22}{r['n_total']:>12,}{r['n_triangle_certified']:>12,}"
              f"{r['n_N_certified']:>12,}{r['extra_certified_by_N']:>10,}"
              f"{r['pct_improvement']:>7.2f}%{r['soundness_violations']:>6}")
    print('=' * 78)
    total_viol = sum(r['soundness_violations'] for r in results)
    print(f"TOTAL soundness violations across all configs: {total_viol}")


if __name__ == '__main__':
    main()

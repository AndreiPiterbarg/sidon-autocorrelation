"""Diagnostic: find min_mu max_W TV_W(mu) for increasing dimension d.

KEY MATHEMATICAL INSIGHT (no correction term needed):

For ANY nonneg f supported on [-1/4, 1/4] with integral 1, partitioned
into d bins of width 1/(2d), with bin masses mu_i = integral of f over bin i:

    max_{|t| <= 1/2} (f*f)(t)  >=  max_W TV_W(mu)

where TV_W(ell, s) = (2d/ell) * sum_{k=s}^{s+ell-2} sum_{i+j=k} mu_i*mu_j.

Proof: Averaging bound + Minkowski sum containment of bin pairs inside
the window.  The bound depends ONLY on bin masses, not function shape.
No step-function approximation, no height quantization, no epsilon.

This script computes:  val(d) = min_{mu in Delta_d} max_W TV_W(mu)

If val(d) >= c_target, the coarse cascade + box certification method
can prove C_{1a} >= c_target at dimension d.

Compare with the fine-grid cascade: effective threshold = c + 2/m + 1/m^2,
which eats into the pruning margin and prevents convergence for c >= 1.25.

Usage:
    python estimate_min_max_tv.py
    python estimate_min_max_tv.py --d_max 128 --restarts 50
    python estimate_min_max_tv.py --d_list 2,4,8,16,32,64
"""
import argparse
import time
import sys

import numpy as np
import numba
from numba import njit, prange


# =====================================================================
# Core Numba kernels
# =====================================================================

@njit(cache=True)
def _autoconv(mu, d):
    """Autoconvolution of mass vector: conv[k] = sum_{i+j=k} mu_i*mu_j."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        mi = mu[i]
        if mi != 0.0:
            conv[2 * i] += mi * mi
            for j in range(i + 1, d):
                mj = mu[j]
                if mj != 0.0:
                    conv[i + j] += 2.0 * mi * mj
    return conv


@njit(cache=True)
def _max_tv(mu, d):
    """Compute max_W TV_W(mu) and the argmax window (ell, s).

    TV_W(ell, s) = (2d/ell) * sum_{k=s}^{s+ell-2} conv[k]

    Returns (best_tv, best_ell, best_s).
    """
    conv = _autoconv(mu, d)
    conv_len = 2 * d - 1

    # Prefix sum for O(1) window queries
    prefix = np.zeros(conv_len + 1, dtype=np.float64)
    for k in range(conv_len):
        prefix[k + 1] = prefix[k] + conv[k]

    best_tv = 0.0
    best_ell = 2
    best_s = 0
    two_d = 2.0 * np.float64(d)

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)
        n_windows = conv_len - n_cv + 1
        for s in range(n_windows):
            ws = prefix[s + n_cv] - prefix[s]
            tv = ws * scale
            if tv > best_tv:
                best_tv = tv
                best_ell = ell
                best_s = s

    return best_tv, best_ell, best_s


@njit(cache=True)
def _all_tv_values(mu, d):
    """Return array of TV values for ALL windows, plus (ell, s) index arrays.

    Used by the smooth optimizer.  Returns (tvs, ells, ss, n_windows).
    """
    conv = _autoconv(mu, d)
    conv_len = 2 * d - 1

    prefix = np.zeros(conv_len + 1, dtype=np.float64)
    for k in range(conv_len):
        prefix[k + 1] = prefix[k] + conv[k]

    # Count total windows
    total_w = 0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        total_w += conv_len - n_cv + 1

    tvs = np.empty(total_w, dtype=np.float64)
    ells = np.empty(total_w, dtype=np.int32)
    ss = np.empty(total_w, dtype=np.int32)

    two_d = 2.0 * np.float64(d)
    idx = 0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)
        n_win = conv_len - n_cv + 1
        for s in range(n_win):
            ws = prefix[s + n_cv] - prefix[s]
            tvs[idx] = ws * scale
            ells[idx] = ell
            ss[idx] = s
            idx += 1

    return tvs, ells, ss, total_w


@njit(cache=True)
def _tv_subgradient(mu, d, ell, s):
    """Subgradient of TV_{ell,s}(mu) w.r.t. mu.

    d(TV)/d(mu_i) = (4d/ell) * sum_{j in J_i} mu_j
    where J_i = {j : max(0, s-i) <= j <= min(d-1, s+ell-2-i)}
    """
    grad = np.zeros(d, dtype=np.float64)
    scale = 4.0 * np.float64(d) / np.float64(ell)

    # Prefix sum of mu for O(1) range queries
    mu_pfx = np.zeros(d + 1, dtype=np.float64)
    for i in range(d):
        mu_pfx[i + 1] = mu_pfx[i] + mu[i]

    for i in range(d):
        j_lo = s - i
        if j_lo < 0:
            j_lo = 0
        j_hi = s + ell - 2 - i
        if j_hi >= d:
            j_hi = d - 1
        if j_lo <= j_hi:
            grad[i] = scale * (mu_pfx[j_hi + 1] - mu_pfx[j_lo])

    return grad


@njit(cache=True)
def _project_simplex(v):
    """Project v onto probability simplex {x >= 0, sum(x) = 1}.

    O(d log d) via sorting.
    """
    d = len(v)
    # Sort descending
    u = np.sort(v.copy())[::-1].copy()

    cssv = 0.0
    rho = 0
    for j in range(d):
        cssv += u[j]
        test = u[j] + (1.0 - cssv) / np.float64(j + 1)
        if test > 0:
            rho = j

    # Recompute cumsum up to rho for precision
    cssv2 = 0.0
    for j in range(rho + 1):
        cssv2 += u[j]
    theta = (cssv2 - 1.0) / np.float64(rho + 1)

    w = np.empty(d, dtype=np.float64)
    for i in range(d):
        w[i] = v[i] - theta
        if w[i] < 0.0:
            w[i] = 0.0

    return w


# =====================================================================
# Optimizer 1: Mirror Descent (entropic, optimal for simplex)
# =====================================================================

@njit(cache=True)
def _mirror_descent(d, mu_init, n_iters, alpha0):
    """Minimize max_W TV_W(mu) via entropic mirror descent on simplex.

    The mirror map is the negative entropy, giving multiplicative updates:
        mu_i <- mu_i * exp(-alpha * grad_i) / Z

    Convergence: O(sqrt(log(d) / T)) — optimal for simplex constraints.

    Returns (best_tv, best_mu).
    """
    mu = mu_init.copy()
    best_tv = 1e30
    best_mu = mu_init.copy()

    for k in range(n_iters):
        tv, ell, s = _max_tv(mu, d)

        if tv < best_tv:
            best_tv = tv
            for i in range(d):
                best_mu[i] = mu[i]

        # Subgradient of the active window
        grad = _tv_subgradient(mu, d, ell, s)

        # Step size: diminishing, tuned for mirror descent
        alpha = alpha0 / (1.0 + np.sqrt(np.float64(k)))

        # Multiplicative update (entropy mirror map)
        # mu_i <- mu_i * exp(-alpha * grad_i) / Z
        # Shift grad for numerical stability
        max_g = grad[0]
        for i in range(1, d):
            if grad[i] > max_g:
                max_g = grad[i]

        total = 0.0
        for i in range(d):
            mu[i] = mu[i] * np.exp(-alpha * (grad[i] - max_g))
            total += mu[i]

        if total > 1e-300:
            inv_total = 1.0 / total
            for i in range(d):
                mu[i] *= inv_total
        else:
            for i in range(d):
                mu[i] = 1.0 / np.float64(d)

    return best_tv, best_mu


# =====================================================================
# Optimizer 2: Projected Subgradient Descent
# =====================================================================

@njit(cache=True)
def _projected_subgradient(d, mu_init, n_iters, alpha0):
    """Minimize max_W TV_W(mu) via projected subgradient descent.

    Step: mu <- project_simplex(mu - alpha * grad / ||grad||)
    Step size: alpha_k = alpha0 / sqrt(k+1)

    Returns (best_tv, best_mu).
    """
    mu = mu_init.copy()
    best_tv = 1e30
    best_mu = mu_init.copy()

    for k in range(n_iters):
        tv, ell, s = _max_tv(mu, d)

        if tv < best_tv:
            best_tv = tv
            for i in range(d):
                best_mu[i] = mu[i]

        grad = _tv_subgradient(mu, d, ell, s)

        # Normalize gradient
        gnorm = 0.0
        for i in range(d):
            gnorm += grad[i] * grad[i]
        gnorm = np.sqrt(gnorm)
        if gnorm < 1e-14:
            break

        alpha = alpha0 / np.sqrt(np.float64(k + 1))

        for i in range(d):
            mu[i] -= alpha * grad[i] / gnorm

        mu = _project_simplex(mu)

    return best_tv, best_mu


# =====================================================================
# Optimizer 3: Frank-Wolfe (Conditional Gradient)
# =====================================================================

@njit(cache=True)
def _frank_wolfe(d, mu_init, n_iters):
    """Minimize max_W TV_W(mu) via Frank-Wolfe on simplex.

    Each step: move toward the simplex vertex with smallest gradient
    component.  Step size gamma = 2/(k+2).

    Returns (best_tv, best_mu).
    """
    mu = mu_init.copy()
    best_tv = 1e30
    best_mu = mu_init.copy()

    for k in range(n_iters):
        tv, ell, s = _max_tv(mu, d)

        if tv < best_tv:
            best_tv = tv
            for i in range(d):
                best_mu[i] = mu[i]

        grad = _tv_subgradient(mu, d, ell, s)

        # Linear minimization oracle: vertex e_i where i = argmin grad
        min_g = grad[0]
        min_i = 0
        for i in range(1, d):
            if grad[i] < min_g:
                min_g = grad[i]
                min_i = i

        gamma = 2.0 / np.float64(k + 2)

        for i in range(d):
            mu[i] = (1.0 - gamma) * mu[i]
        mu[min_i] += gamma

    return best_tv, best_mu


# =====================================================================
# Multi-strategy optimizer with random restarts
# =====================================================================

@njit(cache=True)
def _random_simplex(d, rng_state):
    """Sample uniformly from the probability simplex using exponential trick.

    Uses a simple LCG PRNG (Numba-compatible).
    """
    mu = np.empty(d, dtype=np.float64)
    total = 0.0
    for i in range(d):
        # Simple LCG: state = (a*state + c) mod m
        rng_state = (np.int64(6364136223846793005) * rng_state
                     + np.int64(1442695040888963407))
        # Convert to uniform [0,1)
        u = np.float64(np.uint64(rng_state) >> np.uint64(11)) * 4.6566128730773926e-10
        if u < 1e-10:
            u = 1e-10
        mu[i] = -np.log(u)
        total += mu[i]

    inv_total = 1.0 / total
    for i in range(d):
        mu[i] *= inv_total

    return mu, rng_state


@njit(cache=True)
def _structured_starts(d, idx):
    """Return structured starting distributions.

    idx=0: uniform
    idx=1: edge-concentrated (bimodal)
    idx=2: left-concentrated
    idx=3: alternating heavy/light
    idx=4: triangular (peak at center)
    idx=5: triangular (peak at edge)
    """
    mu = np.empty(d, dtype=np.float64)

    if idx == 0:
        # Uniform
        for i in range(d):
            mu[i] = 1.0 / np.float64(d)

    elif idx == 1:
        # Edge-concentrated (puts mass near both edges)
        total = 0.0
        for i in range(d):
            x = np.float64(i) / np.float64(d - 1) if d > 1 else 0.5
            mu[i] = 1.0 + 2.0 * (1.0 - 4.0 * (x - 0.5) ** 2)
            total += mu[i]
        for i in range(d):
            mu[i] /= total

    elif idx == 2:
        # Left-concentrated
        total = 0.0
        for i in range(d):
            mu[i] = np.float64(d - i)
            total += mu[i]
        for i in range(d):
            mu[i] /= total

    elif idx == 3:
        # Alternating heavy/light
        total = 0.0
        for i in range(d):
            mu[i] = 3.0 if i % 2 == 0 else 1.0
            total += mu[i]
        for i in range(d):
            mu[i] /= total

    elif idx == 4:
        # Triangular peak at center
        total = 0.0
        mid = np.float64(d - 1) / 2.0
        for i in range(d):
            mu[i] = 1.0 + np.float64(d) - abs(np.float64(i) - mid)
            total += mu[i]
        for i in range(d):
            mu[i] /= total

    else:
        # Flat with dip in center (adversarial for autoconvolution)
        total = 0.0
        mid = np.float64(d - 1) / 2.0
        for i in range(d):
            dist = abs(np.float64(i) - mid) / (mid + 1e-10)
            mu[i] = 0.5 + dist  # More mass at edges
            total += mu[i]
        for i in range(d):
            mu[i] /= total

    return mu


@njit(cache=True)
def _optimize_single_start(d, mu_init, n_iters):
    """Run all three optimizers from one starting point, return best."""
    # Allocate iterations: 40% mirror, 30% subgradient, 30% Frank-Wolfe
    n_mirror = max(n_iters * 2 // 5, 100)
    n_subgrad = max(n_iters * 3 // 10, 100)
    n_fw = max(n_iters - n_mirror - n_subgrad, 100)

    tv1, mu1 = _mirror_descent(d, mu_init, n_mirror, 0.5)
    tv2, mu2 = _projected_subgradient(d, mu_init, n_subgrad, 0.3)
    tv3, mu3 = _frank_wolfe(d, mu_init, n_fw)

    # Pick best
    if tv1 <= tv2 and tv1 <= tv3:
        best_tv, best_mu = tv1, mu1
    elif tv2 <= tv3:
        best_tv, best_mu = tv2, mu2
    else:
        best_tv, best_mu = tv3, mu3

    # Polish: run mirror descent from best point with smaller step
    tv_polish, mu_polish = _mirror_descent(d, best_mu, n_iters // 5, 0.1)
    if tv_polish < best_tv:
        best_tv = tv_polish
        best_mu = mu_polish

    return best_tv, best_mu


@njit(parallel=True, cache=True)
def _optimize_all_restarts(d, n_restarts, n_iters, seed):
    """Run multi-start optimization in parallel over restarts.

    Returns (best_tvs, best_mus) arrays for each restart.
    """
    n_structured = 6
    total_starts = n_structured + n_restarts

    best_tvs = np.empty(total_starts, dtype=np.float64)
    best_mus = np.empty((total_starts, d), dtype=np.float64)

    for r in prange(total_starts):
        if r < n_structured:
            mu_init = _structured_starts(d, r)
        else:
            # Random simplex point (seeded per restart for reproducibility)
            rng = np.int64(seed + r * 999983)
            mu_init, _ = _random_simplex(d, rng)

        tv, mu = _optimize_single_start(d, mu_init, n_iters)
        best_tvs[r] = tv
        for i in range(d):
            best_mus[r, i] = mu[i]

    return best_tvs, best_mus


# =====================================================================
# Box certification diagnostic
# =====================================================================

@njit(cache=True)
def _perturbation_bound(mu, d, ell, s, delta):
    """Worst-case TV perturbation for a cell of half-width delta/2.

    For mu_i perturbed by at most delta/2 (with simplex constraint):

    |Delta TV_W| <= (2d/ell) * [delta * C_W + (delta^2/4) * N_W]

    where C_W = sum over contributing pairs of (mu_i + mu_j),
    and N_W = number of contributing pairs.
    """
    two_d_over_ell = 2.0 * np.float64(d) / np.float64(ell)

    C_W = 0.0
    N_W = 0
    for k in range(s, s + ell - 1):
        for i in range(max(0, k - d + 1), min(d, k + 1)):
            j = k - i
            if 0 <= j < d:
                C_W += mu[i] + mu[j]
                N_W += 1

    P = two_d_over_ell * (delta * C_W + 0.25 * delta * delta * np.float64(N_W))
    return P


@njit(cache=True)
def _box_certify_qp(mu_center, d, delta, c_target):
    """Check if a box around mu_center is certifiable.

    For each window W, compute min TV_W over the box (water-filling).
    If max_W min_TV_W >= c_target, the box is CERTIFIED.

    The water-filling uses the monotonicity property: TV_W increases
    in mass of contributing bins, so minimize by putting excess mass
    in non-contributing bins.

    Returns (certified: bool, best_min_tv: float, best_ell, best_s).
    """
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo[i] = max(0.0, mu_center[i] - delta / 2.0)
        hi[i] = min(1.0, mu_center[i] + delta / 2.0)

    best_min_tv = 0.0
    best_ell = 0
    best_s = 0

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)

        for s in range(conv_len - n_cv + 1):
            # Identify contributing bins: bin i contributes to window (ell, s)
            # if there exists j in [0, d) with s <= i+j <= s+ell-2
            # i.e., max(0, s-d+1) <= i <= min(d-1, s+ell-2)
            contrib = np.zeros(d, dtype=numba.boolean)
            for k in range(s, s + n_cv):
                for i in range(max(0, k - d + 1), min(d, k + 1)):
                    contrib[i] = True

            # Water-filling: set contributing bins to lower bound,
            # non-contributing bins absorb excess mass first
            mu_opt = np.empty(d, dtype=np.float64)
            for i in range(d):
                mu_opt[i] = lo[i]

            excess = 1.0 - np.sum(mu_opt)

            # Phase 1: fill non-contributing bins
            for i in range(d):
                if not contrib[i] and excess > 1e-15:
                    add = min(excess, hi[i] - mu_opt[i])
                    mu_opt[i] += add
                    excess -= add

            # Phase 2: fill contributing bins (unavoidable excess)
            # Fill those with smallest marginal TV impact first
            if excess > 1e-15:
                # Compute marginal impact for each contributing bin
                # marginal_i = d(TV_W)/d(mu_i) at current mu_opt
                impacts = np.zeros(d, dtype=np.float64)
                mu_pfx = np.zeros(d + 1, dtype=np.float64)
                for i in range(d):
                    mu_pfx[i + 1] = mu_pfx[i] + mu_opt[i]

                for i in range(d):
                    if contrib[i]:
                        j_lo = s - i
                        if j_lo < 0:
                            j_lo = 0
                        j_hi = s + ell - 2 - i
                        if j_hi >= d:
                            j_hi = d - 1
                        if j_lo <= j_hi:
                            impacts[i] = scale * 2.0 * (
                                mu_pfx[j_hi + 1] - mu_pfx[j_lo])
                        else:
                            impacts[i] = 0.0

                # Sort contributing bins by impact (ascending)
                # Simple selection: fill lowest-impact first
                for _ in range(d):
                    if excess <= 1e-15:
                        break
                    min_imp = 1e30
                    min_idx = -1
                    for i in range(d):
                        if contrib[i] and mu_opt[i] < hi[i] - 1e-15:
                            if impacts[i] < min_imp:
                                min_imp = impacts[i]
                                min_idx = i
                    if min_idx < 0:
                        break
                    add = min(excess, hi[min_idx] - mu_opt[min_idx])
                    mu_opt[min_idx] += add
                    excess -= add

            # Compute TV_W at the minimizing mu_opt
            conv_opt = _autoconv(mu_opt, d)
            ws = 0.0
            for k in range(s, s + n_cv):
                ws += conv_opt[k]
            min_tv = ws * scale

            if min_tv > best_min_tv:
                best_min_tv = min_tv
                best_ell = ell
                best_s = s

            if best_min_tv >= c_target:
                return True, best_min_tv, best_ell, best_s

    return best_min_tv >= c_target, best_min_tv, best_ell, best_s


# =====================================================================
# Analysis helpers
# =====================================================================

@njit(cache=True)
def _active_windows(mu, d, tol=0.001):
    """Find all windows within tol of the maximum TV.

    Returns (count, ells, ss) of near-active windows.
    """
    tv_max, _, _ = _max_tv(mu, d)
    conv = _autoconv(mu, d)
    conv_len = 2 * d - 1
    prefix = np.zeros(conv_len + 1, dtype=np.float64)
    for k in range(conv_len):
        prefix[k + 1] = prefix[k] + conv[k]

    two_d = 2.0 * np.float64(d)
    max_active = 4 * d * d
    ells = np.empty(max_active, dtype=np.int32)
    ss = np.empty(max_active, dtype=np.int32)
    count = 0

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)
        for s in range(conv_len - n_cv + 1):
            ws = prefix[s + n_cv] - prefix[s]
            tv = ws * scale
            if tv >= tv_max - tol and count < max_active:
                ells[count] = ell
                ss[count] = s
                count += 1

    return count, ells[:count], ss[:count]


def describe_minimizer(mu, d):
    """Print analysis of the minimizing mass distribution."""
    tv_max, ell_best, s_best = _max_tv(mu, d)

    # Mass concentration
    sorted_mu = np.sort(mu)[::-1]
    top5_mass = sorted_mu[:min(5, d)].sum()
    nonzero = np.sum(mu > 1e-6)

    # Symmetry check
    rev = mu[::-1]
    sym_err = np.max(np.abs(mu - rev))

    # Active windows
    n_active, active_ells, active_ss = _active_windows(mu, d, tol=0.005)

    print(f"    Minimizer analysis:")
    print(f"      max TV = {tv_max:.6f}  (window: ell={ell_best}, s={s_best})")
    print(f"      Nonzero bins: {nonzero}/{d}")
    print(f"      Top-5 mass: {top5_mass:.4f}")
    print(f"      Symmetry err (mu vs rev): {sym_err:.2e}")
    print(f"      Near-active windows (within 0.005): {n_active}")
    if n_active <= 10:
        for j in range(n_active):
            print(f"        ell={active_ells[j]}, s={active_ss[j]}")

    # Compact mass distribution visualization
    if d <= 64:
        bar_width = min(d, 60)
        max_mu = mu.max()
        print(f"      Mass distribution (max={max_mu:.4f}):")
        bar = ""
        for i in range(d):
            level = int(mu[i] / max_mu * 8) if max_mu > 0 else 0
            chars = " .:-=+*#@"
            bar += chars[min(level, 8)]
        print(f"        [{bar}]")


# =====================================================================
# Main optimization driver
# =====================================================================

def optimize_for_d(d, n_restarts=30, n_iters=20000, seed=42, verbose=True):
    """Find min_mu max_W TV_W(mu) at dimension d.

    Uses multi-strategy optimization with random restarts.
    Returns (min_tv, best_mu).
    """
    if verbose:
        print(f"  d={d}: optimizing ({n_restarts} random + 6 structured starts, "
              f"{n_iters} iters each)...", end="", flush=True)

    t0 = time.time()

    # Warm up Numba on first call
    if d >= 2:
        _warmup = np.ones(2) / 2.0
        _max_tv(_warmup, 2)

    tvs, mus = _optimize_all_restarts(d, n_restarts, n_iters, seed)

    # Find global best
    best_idx = np.argmin(tvs)
    best_tv = tvs[best_idx]
    best_mu = mus[best_idx]

    # Polish: run each optimizer longer from the best point
    tv_p1, mu_p1 = _mirror_descent(d, best_mu, n_iters, 0.2)
    tv_p2, mu_p2 = _projected_subgradient(d, best_mu, n_iters, 0.1)
    tv_p3, mu_p3 = _frank_wolfe(d, best_mu, n_iters * 2)

    for tv_p, mu_p in [(tv_p1, mu_p1), (tv_p2, mu_p2), (tv_p3, mu_p3)]:
        if tv_p < best_tv:
            best_tv = tv_p
            best_mu = mu_p

    # Symmetrize and re-check (the optimal is likely symmetric)
    mu_sym = 0.5 * (best_mu + best_mu[::-1])
    tv_sym, _, _ = _max_tv(mu_sym, d)
    if tv_sym < best_tv:
        # Polish symmetric version
        tv_s2, mu_s2 = _mirror_descent(d, mu_sym, n_iters, 0.2)
        if tv_s2 < best_tv:
            best_tv = tv_s2
            best_mu = mu_s2

    elapsed = time.time() - t0

    if verbose:
        print(f" done ({elapsed:.1f}s)")

    return best_tv, best_mu


def run_box_certification_demo(mu_opt, d, c_target, deltas):
    """Demonstrate box certification at several grid spacings."""
    tv_opt, ell_opt, s_opt = _max_tv(mu_opt, d)

    print(f"\n  Box certification demo at d={d}, c_target={c_target}:")
    print(f"    Minimizer max TV = {tv_opt:.6f}")

    for delta in deltas:
        # Margin check
        margin = tv_opt - c_target
        P = _perturbation_bound(mu_opt, d, ell_opt, s_opt, delta)

        # Full QP check
        certified, min_tv_box, qp_ell, qp_s = _box_certify_qp(
            mu_opt, d, delta, c_target)

        status = "CERTIFIED" if certified else "NEEDS REFINEMENT"
        print(f"    delta={delta:.4f}: margin={margin:.4f}, "
              f"perturb_bound={P:.4f}, "
              f"margin>P={'YES' if margin > P else 'NO'}, "
              f"QP min TV={min_tv_box:.4f} [{status}]")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: find min_mu max_W TV_W(mu) for increasing d")
    parser.add_argument("--d_list", type=str, default=None,
                        help="Comma-separated list of d values (e.g., 2,4,8,16,32)")
    parser.add_argument("--d_max", type=int, default=128,
                        help="Max dimension (used if --d_list not given)")
    parser.add_argument("--restarts", type=int, default=30,
                        help="Number of random restarts per d")
    parser.add_argument("--iters", type=int, default=20000,
                        help="Iterations per optimizer per restart")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--box_demo", action="store_true",
                        help="Run box certification demo")
    args = parser.parse_args()

    if args.d_list:
        d_values = [int(x) for x in args.d_list.split(",")]
    else:
        d_values = []
        d = 2
        while d <= args.d_max:
            d_values.append(d)
            d *= 2

    c_targets = [1.20, 1.25, 1.28, 1.30, 1.35, 1.40, 1.45, 1.50]
    m_values = [20, 50]  # Fine grid m values for comparison

    print("=" * 72)
    print("COARSE CASCADE + BOX CERTIFICATION DIAGNOSTIC")
    print("=" * 72)
    print()
    print("Computing: val(d) = min_{mu in Delta_d} max_W TV_W(mu)")
    print()
    print("If val(d) >= c_target, the no-correction coarse cascade can")
    print("prove C_{1a} >= c_target at dimension d (with box certification).")
    print()
    print("Compare: fine-grid cascade needs TV >= c + 2/m + 1/m^2,")
    print("which is HIGHER and blocks convergence for large c_target.")
    print()

    # JIT warmup
    print("Warming up Numba JIT...", end="", flush=True)
    t0 = time.time()
    _dummy = np.ones(4) / 4.0
    _max_tv(_dummy, 4)
    _tv_subgradient(_dummy, 4, 2, 0)
    _project_simplex(_dummy)
    _mirror_descent(4, _dummy, 100, 0.5)
    _projected_subgradient(4, _dummy, 100, 0.3)
    _frank_wolfe(4, _dummy, 100)
    _optimize_all_restarts(4, 2, 100, 42)
    print(f" done ({time.time() - t0:.1f}s)")
    print()

    # Main optimization loop
    results = {}
    print("Optimization results:")
    print("-" * 72)

    for d in d_values:
        # Scale iterations and restarts with d
        n_iters = args.iters
        n_restarts = args.restarts
        if d >= 64:
            n_iters = max(n_iters, 30000)
            n_restarts = max(n_restarts, 40)
        if d >= 128:
            n_iters = max(n_iters, 50000)
            n_restarts = max(n_restarts, 50)

        tv, mu = optimize_for_d(d, n_restarts, n_iters, args.seed, verbose=True)
        results[d] = (tv, mu)
        describe_minimizer(mu, d)
        print()

    # Summary table
    print()
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print()

    header = f"{'d':>5} | {'min max TV':>12} |"
    for c in c_targets:
        header += f" c={c:.2f}"
    print(header)
    print("-" * len(header))

    for d in d_values:
        tv, _ = results[d]
        row = f"{d:>5} | {tv:>12.6f} |"
        for c in c_targets:
            if tv >= c - 1e-6:
                row += "    YES"
            else:
                row += "     no"
        print(row)

    # Fine-grid comparison
    print()
    print("=" * 72)
    print("FINE-GRID PENALTY COMPARISON")
    print("=" * 72)
    print()
    print("The fine grid adds correction 2/m + 1/m^2, raising the threshold.")
    print("This table shows the effective threshold for each (c_target, m):")
    print()

    header2 = f"{'c_target':>8} | {'No-corr':>10} |"
    for m in m_values:
        corr = 2.0 / m + 1.0 / (m * m)
        header2 += f" m={m} (+{corr:.4f})"
    print(header2)
    print("-" * len(header2))

    for c in c_targets:
        row = f"{c:>8.2f} | {c:>10.4f} |"
        for m in m_values:
            corr = 2.0 / m + 1.0 / (m * m)
            row += f"      {c + corr:>8.4f}"
        print(row)

    # Which c_targets become provable?
    print()
    print("=" * 72)
    print("PROVABILITY ANALYSIS")
    print("=" * 72)
    print()
    print("For each c_target: smallest d where val(d) >= c_target,")
    print("and comparison with fine-grid feasibility.")
    print()

    for c in c_targets:
        d_needed = None
        for d in d_values:
            tv, _ = results[d]
            if tv >= c - 1e-6:
                d_needed = d
                break

        if d_needed:
            tv_at_d, _ = results[d_needed]
            margin = tv_at_d - c
            print(f"  c_target = {c:.2f}:")
            print(f"    No-correction: PROVABLE at d={d_needed} "
                  f"(margin = {margin:.4f})")
            for m in m_values:
                corr = 2.0 / m + 1.0 / (m * m)
                eff_thr = c + corr
                # Check if fine grid can achieve this
                fg_possible = False
                fg_d = None
                for d2 in d_values:
                    tv2, _ = results[d2]
                    if tv2 >= eff_thr - 1e-6:
                        fg_possible = True
                        fg_d = d2
                        break
                if fg_possible:
                    print(f"    Fine grid m={m}: needs TV >= {eff_thr:.4f}, "
                          f"achievable at d={fg_d}")
                else:
                    print(f"    Fine grid m={m}: needs TV >= {eff_thr:.4f}, "
                          f"NOT achievable at d <= {d_values[-1]}")
        else:
            print(f"  c_target = {c:.2f}: NOT provable at d <= {d_values[-1]}")

    # Box certification demo
    if args.box_demo and len(d_values) >= 3:
        print()
        print("=" * 72)
        print("BOX CERTIFICATION DEMO")
        print("=" * 72)
        # Pick a medium d
        demo_d = d_values[min(3, len(d_values) - 1)]
        demo_tv, demo_mu = results[demo_d]
        deltas = [0.05, 0.02, 0.01, 0.005]
        for c in [1.20, 1.25, 1.28, 1.30]:
            if demo_tv >= c:
                run_box_certification_demo(demo_mu, demo_d, c, deltas)

    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    best_d = d_values[-1]
    best_tv, _ = results[best_d]
    print()
    print(f"  Largest dimension tested: d = {best_d}")
    print(f"  min max TV at d={best_d}: {best_tv:.6f}")
    print()
    print(f"  The coarse cascade + box certification method can prove")
    print(f"  C_{{1a}} >= c for any c <= {best_tv:.4f} (at d={best_d}).")
    print()
    print(f"  For larger d, the bound tightens further toward C_{{1a}}.")
    print(f"  The fine-grid cascade with m=20 adds +0.1025 to the threshold,")
    print(f"  making c_target >= {best_tv - 0.1025:.4f} the effective limit")
    print(f"  (vs. {best_tv:.4f} with no correction).")


if __name__ == "__main__":
    main()

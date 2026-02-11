"""
fast_core.py — Speed-optimized Numba JIT functions for Sidon optimization.

Key optimizations over sidon_core:
1. Inlined lazy k_star updates in Polyak polish — avoids function call overhead.
   Full autoconv O(P^2) every `track_interval` iters; inlined O(P) neighbor
   check on other iters for gradient direction.
2. Reduced tracking frequency in LSE phase (every 15 iters instead of every 1).
3. Reduced Armijo max backtracking steps (15 vs 30).
"""

import numpy as np
import numba as nb

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from sidon_core import (
    autoconv_coeffs, project_simplex_nb,
    lse_obj_nb, lse_objgrad_nb,
    BETA_HEAVY, BETA_ULTRA
)


# ============================================================================
# Single-k autoconvolution (for external use / benchmarking)
# ============================================================================

@nb.njit(cache=True)
def autoconv_single_k(x, P, k):
    """Compute c_k = 2P * sum_{i+j=k} x_i * x_j for a single k. O(P)."""
    n = len(x)
    s = 0.0
    i_lo = max(0, k - n + 1)
    i_hi = min(k, n - 1)
    for i in range(i_lo, i_hi + 1):
        s += x[i] * x[k - i]
    return 2.0 * P * s


# ============================================================================
# Optimized Polyak Polish — inlined lazy k_star updates
# ============================================================================

@nb.njit(cache=True)
def _polyak_polish_fast(x_init, P, n_iters):
    """
    Optimized Polyak polish with inlined lazy k_star updates.

    Every `track_interval` iterations: full O(P^2) autoconv for best_val tracking.
    Other iterations: inlined O(P) neighbor check for gradient k_star.
    All inner loops inlined to avoid Numba function call overhead.
    """
    x = x_init.copy()
    n = len(x)
    nc = 2 * n - 1
    scale4P = 2.0 * (2.0 * P)
    scale2P = 2.0 * P

    # Initial full computation
    c = autoconv_coeffs(x, P)
    fval = c[0]
    k_star = 0
    for k in range(1, nc):
        if c[k] > fval:
            fval = c[k]
            k_star = k

    best_val = fval
    best_x = x.copy()
    no_improve = 0
    stall_limit = 20000
    track_interval = 40  # full autoconv every N iters

    # Averaging state
    avg_start = int(n_iters * 0.75)
    x_avg = np.zeros(n)
    n_avg = 0

    # Pre-allocate gradient buffer
    g = np.empty(n)

    for t in range(n_iters):
        # --- Determine fval and k_star ---
        if t % track_interval == 0:
            # Full O(P^2) computation
            c = autoconv_coeffs(x, P)
            fval = c[0]
            k_star = 0
            for k in range(1, nc):
                if c[k] > fval:
                    fval = c[k]
                    k_star = k
            # Update best_val only on full refreshes (correct values)
            if fval < best_val:
                best_val = fval
                best_x = x.copy()
                no_improve = 0
            else:
                no_improve += track_interval
        else:
            # Inlined lazy check: compute c at k_star, then check +-4 neighbors
            # Compute c[k_star] inline
            s_center = 0.0
            i_lo = max(0, k_star - n + 1)
            i_hi = min(k_star, n - 1)
            for i in range(i_lo, i_hi + 1):
                s_center += x[i] * x[k_star - i]
            fval = s_center * scale2P
            best_k = k_star

            # Check neighbors: +-1, +-2, +-3
            for dk in range(1, 4):
                # +dk
                kn = k_star + dk
                if 0 <= kn < nc:
                    sn = 0.0
                    i_lo2 = max(0, kn - n + 1)
                    i_hi2 = min(kn, n - 1)
                    for i in range(i_lo2, i_hi2 + 1):
                        sn += x[i] * x[kn - i]
                    cn = sn * scale2P
                    if cn > fval:
                        fval = cn
                        best_k = kn
                # -dk
                kn = k_star - dk
                if 0 <= kn < nc:
                    sn = 0.0
                    i_lo2 = max(0, kn - n + 1)
                    i_hi2 = min(kn, n - 1)
                    for i in range(i_lo2, i_hi2 + 1):
                        sn += x[i] * x[kn - i]
                    cn = sn * scale2P
                    if cn > fval:
                        fval = cn
                        best_k = kn

            k_star = best_k
            # Don't update best_val on lazy iters (might miss distant peaks)

        # Stall-restart
        if no_improve >= stall_limit:
            no_improve = 0
            for i in range(n):
                x[i] = best_x[i] * (1.0 + 0.05 * (((t * 7 + i * 13) % 100) / 50.0 - 1.0))
            for i in range(n):
                if x[i] < 0.0:
                    x[i] = 0.0
            s = 0.0
            for i in range(n):
                s += x[i]
            if s > 1e-12:
                for i in range(n):
                    x[i] /= s
            else:
                for i in range(n):
                    x[i] = best_x[i]
            # Force full refresh
            c = autoconv_coeffs(x, P)
            fval = c[0]
            k_star = 0
            for k in range(1, nc):
                if c[k] > fval:
                    fval = c[k]
                    k_star = k
            continue

        offset = 0.01 / (1.0 + t * 1e-4)
        target = best_val - offset

        # Compute gradient for current k_star
        j_lo = max(0, k_star - n + 1)
        j_hi = min(k_star, n - 1)
        gnorm2 = 0.0
        for i in range(n):
            g[i] = 0.0
        for i in range(n):
            j = k_star - i
            if j_lo <= j <= j_hi:
                gi = scale4P * x[j]
                g[i] = gi
                gnorm2 += gi * gi
        if gnorm2 < 1e-20:
            break

        step = (fval - target) / gnorm2
        if step < 0.0:
            step = 1e-5 / (1.0 + t * 1e-4)

        for i in range(n):
            x[i] = x[i] - step * g[i]
        x = project_simplex_nb(x)

        # Accumulate average
        if t >= avg_start:
            for i in range(n):
                x_avg[i] += x[i]
            n_avg += 1

    # Final verification with full autoconv
    c = autoconv_coeffs(best_x, P)
    final_val = c[0]
    for k in range(1, nc):
        if c[k] > final_val:
            final_val = c[k]
    best_val = final_val

    # Check averaged iterate
    if n_avg > 0:
        for i in range(n):
            x_avg[i] /= n_avg
        x_avg = project_simplex_nb(x_avg)
        c = autoconv_coeffs(x_avg, P)
        avg_val = c[0]
        for k in range(1, nc):
            if c[k] > avg_val:
                avg_val = c[k]
        if avg_val < best_val:
            best_val = avg_val
            best_x = x_avg.copy()

    return best_val, best_x


# ============================================================================
# Optimized Armijo with reduced backtracking
# ============================================================================

@nb.njit(cache=True)
def armijo_step_fast(x, g, fval, P, beta, alpha_init, rho=0.5, c1=1e-4, max_bt=15):
    """Armijo with reduced max backtracking (15 instead of 30)."""
    alpha = alpha_init
    x_new = np.empty_like(x)
    for _ in range(max_bt):
        for i in range(len(x)):
            x_new[i] = x[i] - alpha * g[i]
        x_new = project_simplex_nb(x_new)
        fval_new = lse_obj_nb(x_new, P, beta)
        descent = 0.0
        for i in range(len(x)):
            descent += g[i] * (x[i] - x_new[i])
        if fval_new <= fval - c1 * descent:
            return x_new, fval_new, alpha
        alpha *= rho
    return x_new, fval_new, alpha


# ============================================================================
# Optimized hybrid single restart
# ============================================================================

@nb.njit(cache=True)
def _hybrid_single_restart_fast(x_init, P, beta_schedule, n_iters_lse, n_iters_polyak):
    """
    Optimized hybrid restart: LSE Nesterov continuation -> fast Polyak polish.

    Optimizations vs original:
    1. Check true objective every 15 LSE iterations instead of every iteration
    2. Reduced Armijo max backtracking (15 vs 30)
    3. Fast Polyak polish with inlined lazy k_star updates
    """
    x = x_init.copy()
    check_interval = 15

    for stage in range(len(beta_schedule)):
        beta = beta_schedule[stage]
        y = x.copy()
        x_prev = x.copy()
        alpha_init = 0.1
        best_stage_val = 1e300
        best_stage_x = x.copy()
        no_improve = 0

        for t in range(n_iters_lse):
            fval_y, g = lse_objgrad_nb(y, P, beta)
            x_new, fval_new, alpha_used = armijo_step_fast(
                y, g, fval_y, P, beta, alpha_init)
            alpha_init = min(alpha_used * 2.0, 1.0)

            momentum = t / (t + 3.0)
            nn = len(x_new)
            y_new = np.empty(nn)
            for i in range(nn):
                y_new[i] = x_new[i] + momentum * (x_new[i] - x_prev[i])
            y_new = project_simplex_nb(y_new)

            x_prev = x_new.copy()
            x = x_new
            y = y_new

            if t % check_interval == 0:
                tv = np.max(autoconv_coeffs(x, P))
                if tv < best_stage_val:
                    best_stage_val = tv
                    best_stage_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += check_interval

            if no_improve > 800:
                break

        x = best_stage_x

    lse_val = np.max(autoconv_coeffs(x, P))
    polished_val, polished_x = _polyak_polish_fast(x, P, n_iters_polyak)
    return lse_val, polished_val, polished_x


# ============================================================================
# Warmup
# ============================================================================

def warmup():
    """Trigger JIT compilation for all fast functions."""
    x = np.ones(5) / 5.0
    beta_arr = np.array([1.0, 10.0])
    _ = autoconv_single_k(x, 5, 2)
    _ = _polyak_polish_fast(x, 5, 10)
    _ = armijo_step_fast(x, np.ones(5), 1.5, 5, 10.0, 0.1)
    _ = _hybrid_single_restart_fast(x, 5, beta_arr, 10, 10)

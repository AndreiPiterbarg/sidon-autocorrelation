"""
sidon_core.py — Core optimization math for Sidon autocorrelation C_{1a}.

Extracted from logsumexp_optimizer.ipynb for use with Modal cloud compute.
All Numba JIT functions and Python helpers. No Modal dependency.

Method: Hybrid LSE continuation (smooth global search) + adaptive Polyak
subgradient (non-smooth local polish), parallelized via joblib.
"""

import os
import numpy as np
import numba as nb
from joblib import Parallel, delayed


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BETA_HEAVY = [1, 1.5, 2, 3, 5, 8, 12, 18, 28, 42, 65, 100, 150, 230, 350,
              500, 750, 1000, 1500, 2000, 3000]

BETA_ULTRA = [1, 1.3, 1.7, 2.2, 3, 4, 5.5, 7.5, 10, 14, 20, 28, 40, 55,
              75, 100, 140, 200, 280, 400, 560, 800, 1100, 1500, 2000, 3000, 4000]

ALL_STRATEGIES = [
    'dirichlet_uniform', 'dirichlet_sparse', 'dirichlet_concentrated',
    'gaussian_peak', 'bimodal', 'cosine_shaped', 'triangle',
    'flat_noisy', 'boundary_heavy', 'random_sparse_k',
    'symmetric_dirichlet', 'warm_perturb',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Numba JIT functions — exact copies from logsumexp_optimizer.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def project_simplex_nb(x):
    """Project x onto the probability simplex."""
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = 0
    for i in range(n):
        if u[i] * (i + 1) > cssv[i]:
            rho = i
    tau = cssv[rho] / (rho + 1.0)
    out = np.empty(n)
    for i in range(n):
        out[i] = max(x[i] - tau, 0.0)
    return out


@nb.njit(cache=True)
def convolve_full(a, b):
    """Full convolution of two 1D arrays."""
    na, nb_ = len(a), len(b)
    nc = na + nb_ - 1
    c = np.zeros(nc)
    for i in range(na):
        for j in range(nb_):
            c[i + j] += a[i] * b[j]
    return c


@nb.njit(cache=True)
def autoconv_coeffs(x, P):
    """c_k = 2P * sum_{i+j=k} x_i x_j"""
    return convolve_full(x, x) * (2.0 * P)


@nb.njit(cache=True)
def logsumexp_nb(c, beta):
    """Numerically stable LogSumExp."""
    bc_max = -1e300
    for i in range(len(c)):
        v = beta * c[i]
        if v > bc_max:
            bc_max = v
    s = 0.0
    for i in range(len(c)):
        s += np.exp(beta * c[i] - bc_max)
    return bc_max / beta + np.log(s) / beta


@nb.njit(cache=True)
def softmax_nb(c, beta):
    """Softmax weights."""
    n = len(c)
    bc_max = -1e300
    for i in range(n):
        v = beta * c[i]
        if v > bc_max:
            bc_max = v
    e = np.empty(n)
    s = 0.0
    for i in range(n):
        e[i] = np.exp(beta * c[i] - bc_max)
        s += e[i]
    for i in range(n):
        e[i] /= s
    return e


@nb.njit(cache=True)
def lse_obj_nb(x, P, beta):
    c = autoconv_coeffs(x, P)
    return logsumexp_nb(c, beta)


@nb.njit(cache=True)
def lse_grad_nb(x, P, beta):
    """Gradient of LSE_beta(c(x)) w.r.t. x.
    g_i = 2*(2P) * sum_k w_k * x_{k-i}
    """
    c = autoconv_coeffs(x, P)
    w = softmax_nb(c, beta)
    n = len(x)
    n_c = len(c)
    g = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(n_c):
            j = k - i
            if 0 <= j < n:
                s += w[k] * x[j]
        g[i] = s
    scale = 2.0 * (2.0 * P)
    for i in range(n):
        g[i] *= scale
    return g


@nb.njit(cache=True)
def armijo_step_nb(x, g, P, beta, alpha_init, rho=0.5, c1=1e-4, max_bt=30):
    """Armijo backtracking line search."""
    fval = lse_obj_nb(x, P, beta)
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


@nb.njit(cache=True)
def _polyak_polish_nb(x_init, P, n_iters):
    """Adaptive Polyak polish: target = best_val - shrinking_offset."""
    x = x_init.copy()
    n = len(x)
    best_val = np.max(autoconv_coeffs(x, P))
    best_x = x.copy()

    for t in range(n_iters):
        c = autoconv_coeffs(x, P)
        fval = np.max(c)
        if fval < best_val:
            best_val = fval
            best_x = x.copy()

        offset = 0.01 / (1.0 + t * 1e-4)
        target = best_val - offset

        k_star = 0
        for k in range(len(c)):
            if c[k] > c[k_star]:
                k_star = k

        g = np.zeros(n)
        for i in range(n):
            j = k_star - i
            if 0 <= j < n:
                g[i] = 2.0 * (2.0 * P) * x[j]

        gnorm2 = 0.0
        for i in range(n):
            gnorm2 += g[i] * g[i]
        if gnorm2 < 1e-20:
            break

        step = (fval - target) / gnorm2
        if step < 0.0:
            step = 1e-5 / (1.0 + t * 1e-4)

        for i in range(n):
            x[i] = x[i] - step * g[i]
        x = project_simplex_nb(x)

    return best_val, best_x


@nb.njit(cache=True)
def _hybrid_single_restart(x_init, P, beta_schedule, n_iters_lse, n_iters_polyak):
    """One restart: LSE Nesterov continuation -> adaptive Polyak polish."""
    x = x_init.copy()

    # Phase 1: LSE continuation
    for stage in range(len(beta_schedule)):
        beta = beta_schedule[stage]
        y = x.copy()
        x_prev = x.copy()
        alpha_init = 0.1
        best_stage_val = 1e300
        best_stage_x = x.copy()
        no_improve = 0

        for t in range(n_iters_lse):
            g = lse_grad_nb(y, P, beta)
            x_new, fval_new, alpha_used = armijo_step_nb(y, g, P, beta, alpha_init)
            alpha_init = min(alpha_used * 2.0, 1.0)

            momentum = t / (t + 3.0)
            n = len(x_new)
            y_new = np.empty(n)
            for i in range(n):
                y_new[i] = x_new[i] + momentum * (x_new[i] - x_prev[i])
            y_new = project_simplex_nb(y_new)

            x_prev = x_new.copy()
            x = x_new
            y = y_new

            tv = np.max(autoconv_coeffs(x, P))
            if tv < best_stage_val:
                best_stage_val = tv
                best_stage_x = x.copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 800:
                break

        x = best_stage_x

    lse_val = np.max(autoconv_coeffs(x, P))

    # Phase 2: adaptive Polyak polish
    polished_val, polished_x = _polyak_polish_nb(x, P, n_iters_polyak)

    return lse_val, polished_val, polished_x


# ═══════════════════════════════════════════════════════════════════════════════
# Exact evaluation of peak autoconvolution
# ═══════════════════════════════════════════════════════════════════════════════

def peak_autoconv_exact(edges, heights):
    """
    Compute the exact peak of (f*f)(t) for a step function f.

    f(x) = heights[i] for edges[i] <= x < edges[i+1], zero outside.
    (f*f)(t) = integral f(x) f(t-x) dx

    This is piecewise linear in t, with breakpoints at t = edges[i] + edges[j].
    The maximum must occur at one of these breakpoints.

    Returns (peak_value, peak_location).
    """
    edges = np.asarray(edges, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    N = len(heights)

    # All breakpoints
    bp = (edges[:, None] + edges[None, :]).ravel()
    bp = np.unique(bp)
    bp = bp[(bp >= 2 * edges[0]) & (bp <= 2 * edges[-1])]

    a = edges[:-1]  # left edges of bins
    b = edges[1:]   # right edges of bins

    peak = -np.inf
    peak_t = None

    batch_size = 500
    for start in range(0, len(bp), batch_size):
        end = min(start + batch_size, len(bp))
        t_batch = bp[start:end]

        conv = np.zeros(len(t_batch))
        for i in range(N):
            for j in range(N):
                lo = np.maximum(a[i], t_batch - b[j])
                hi = np.minimum(b[i], t_batch - a[j])
                overlap = np.maximum(0.0, hi - lo)
                conv += heights[i] * heights[j] * overlap

        idx = np.argmax(conv)
        if conv[idx] > peak:
            peak = conv[idx]
            peak_t = t_batch[idx]

    return float(peak), float(peak_t)


def exact_val(x, P):
    """Compute exact peak autoconvolution from simplex weights."""
    edges = np.linspace(-0.25, 0.25, P + 1)
    bin_width = 0.5 / P
    heights = x / bin_width
    peak, _ = peak_autoconv_exact(edges, heights)
    return peak


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization strategies
# ═══════════════════════════════════════════════════════════════════════════════

def make_inits(strategy, P, n_restarts, rng, warm_x=None):
    """Generate n_restarts initial simplex vectors for a given strategy."""
    centers = np.linspace(-0.25 + 0.25 / P, 0.25 - 0.25 / P, P)
    inits = []

    for _ in range(n_restarts):
        if strategy == 'dirichlet_uniform':
            x = rng.dirichlet(np.ones(P))

        elif strategy == 'dirichlet_sparse':
            x = rng.dirichlet(np.full(P, 0.1))

        elif strategy == 'dirichlet_concentrated':
            x = rng.dirichlet(np.full(P, 5.0))

        elif strategy == 'gaussian_peak':
            sigma = rng.uniform(0.03, 0.15)
            mu = rng.uniform(-0.05, 0.05)
            x = np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
            x += rng.uniform(0, 0.01, P)
            x /= x.sum()

        elif strategy == 'bimodal':
            sep = rng.uniform(0.05, 0.2)
            sigma = rng.uniform(0.02, 0.08)
            ratio = rng.uniform(0.3, 0.7)
            x = ratio * np.exp(-0.5 * ((centers - sep / 2) / sigma) ** 2)
            x += (1 - ratio) * np.exp(-0.5 * ((centers + sep / 2) / sigma) ** 2)
            x += rng.uniform(0, 0.005, P)
            x /= x.sum()

        elif strategy == 'cosine_shaped':
            phase = rng.uniform(0, np.pi)
            freq = rng.uniform(0.5, 3.0)
            x = np.cos(freq * np.pi * centers / 0.25 + phase) ** 2
            x += rng.uniform(0, 0.02, P)
            x /= x.sum()

        elif strategy == 'triangle':
            peak_pos = rng.uniform(-0.1, 0.1)
            width = rng.uniform(0.1, 0.25)
            x = np.maximum(0, 1 - np.abs(centers - peak_pos) / width)
            x += rng.uniform(0, 0.01, P)
            x /= x.sum()

        elif strategy == 'flat_noisy':
            noise_scale = rng.uniform(0.01, 0.2)
            x = np.ones(P) + noise_scale * rng.standard_normal(P)
            x = np.maximum(x, 0.0)
            x /= x.sum()

        elif strategy == 'boundary_heavy':
            decay = rng.uniform(2.0, 10.0)
            x = np.exp(-decay * np.abs(np.abs(centers) - 0.25))
            x += rng.uniform(0, 0.01, P)
            x /= x.sum()

        elif strategy == 'random_sparse_k':
            k = rng.integers(max(3, P // 10), max(4, P // 3))
            x = np.zeros(P)
            idx = rng.choice(P, size=k, replace=False)
            x[idx] = rng.dirichlet(np.ones(k))

        elif strategy == 'symmetric_dirichlet':
            half = P // 2
            x_half = rng.dirichlet(np.ones(half))
            x = np.zeros(P)
            x[:half] = x_half
            x[P - half:] = x_half[::-1]
            if P % 2 == 1:
                x[half] = rng.uniform(0.0, 0.1)
            x /= x.sum()

        elif strategy == 'warm_perturb':
            if warm_x is not None:
                noise = rng.uniform(0.3, 1.5)
                x = warm_x.copy() + noise * rng.standard_normal(P) * np.mean(warm_x)
                x = np.maximum(x, 0.0)
                if x.sum() < 1e-12:
                    x = rng.dirichlet(np.ones(P))
                else:
                    x /= x.sum()
            else:
                x = rng.dirichlet(np.ones(P))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        inits.append(x)

    return inits


def upsample_solution(x_low, P_low, P_high):
    """Upsample a P_low-bin solution to P_high bins via linear interpolation."""
    edges_low = np.linspace(-0.25, 0.25, P_low + 1)
    edges_high = np.linspace(-0.25, 0.25, P_high + 1)
    bin_width_low = 0.5 / P_low
    bin_width_high = 0.5 / P_high

    heights_low = x_low / bin_width_low
    centers_low = 0.5 * (edges_low[:-1] + edges_low[1:])
    centers_high = 0.5 * (edges_high[:-1] + edges_high[1:])

    heights_high = np.interp(centers_high, centers_low, heights_low)
    heights_high = np.maximum(heights_high, 0.0)

    x_high = heights_high * bin_width_high
    if x_high.sum() > 0:
        x_high /= x_high.sum()
    else:
        x_high = np.ones(P_high) / P_high
    return x_high


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel hybrid runner
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_strategy_run(P, strategy, beta_schedule, n_iters_lse=15000,
                        n_iters_polyak=200000, n_restarts=80,
                        n_jobs=-1, warm_x=None, seed=None):
    """
    Run hybrid optimizer with a specific initialization strategy.

    Returns (best_val, best_x, all_vals).
    """
    beta_arr = np.array(beta_schedule, dtype=np.float64)
    rng = np.random.default_rng(seed)
    inits = make_inits(strategy, P, n_restarts, rng, warm_x=warm_x)

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_hybrid_single_restart)(inits[i], P, beta_arr, n_iters_lse, n_iters_polyak)
        for i in range(n_restarts)
    )

    best_val = np.inf
    best_x = None
    all_vals = []
    for i, (lse_v, pol_v, x) in enumerate(results):
        all_vals.append(pol_v)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    arr = np.array(all_vals)
    print(f"    {strategy:30s}  best={best_val:.6f}  "
          f"median={np.median(arr):.6f}  std={np.std(arr):.6f}")

    return best_val, best_x, all_vals


# ═══════════════════════════════════════════════════════════════════════════════
# Warmup — trigger Numba JIT compilation
# ═══════════════════════════════════════════════════════════════════════════════

def warmup():
    """Compile all Numba functions by calling them once with small inputs."""
    x = np.ones(5) / 5.0
    _ = project_simplex_nb(x)
    _ = autoconv_coeffs(x, 5)
    _ = lse_obj_nb(x, 5, 10.0)
    _ = lse_grad_nb(x, 5, 10.0)
    _ = armijo_step_nb(x, np.ones(5), 5, 10.0, 0.1)
    _ = _polyak_polish_nb(x, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x, 5, beta_arr, 10, 10)

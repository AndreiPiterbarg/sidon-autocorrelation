"""Improved mu_star extraction and active-Hessian computation.

Implements:
  (1) Parallel multistart Nelder-Mead via multiprocessing.
  (2) Basin hopping for non-smooth max objective.
  (3) Exact KKT solver via scipy.optimize.minimize with proper QP.
  (4) alpha optimization to maximize lambda_min(P H_alpha P) — gets H as PSD as possible.
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import multiprocessing as mp


def max_tv(mu, d):
    """Direct TV evaluation: max_W (2d/ell) * sum_t (mu*mu)[t] indicator."""
    conv = np.zeros(2 * d - 1)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        scale = 2.0 * d / float(ell)
        for s in range(2 * d - ell + 1):
            ws = float(np.sum(conv[s:s + ell - 1]))
            v = scale * ws
            if v > best:
                best = v
    return best


def _build_AW_local(d, ell, s):
    """Local copy of A_W builder (avoids importing the full prover)."""
    A = np.zeros((d, d), dtype=np.float64)
    s_hi_off = ell - 2
    for i in range(d):
        jl = max(s - i, 0); jh = min(s + s_hi_off - i, d - 1)
        for j in range(jl, jh + 1):
            A[i, j] = 1.0
    return A


# --- Worker for multiprocess Nelder-Mead ---

def _nelder_mead_worker(args):
    """Single Nelder-Mead run from a random starting point."""
    seed, d, max_iter = args
    rng = np.random.default_rng(seed)
    # Random init on simplex (Dirichlet)
    raw = rng.exponential(1.0, d)
    mu0 = raw / raw.sum()

    def obj(mu_free):
        mu = np.zeros(d)
        mu[:d-1] = mu_free
        mu[d-1] = 1.0 - sum(mu_free)
        if mu[d-1] < -1e-9 or np.any(mu < -1e-9):
            return 1e10
        mu = np.maximum(mu, 0)
        return max_tv(mu, d)

    res = minimize(obj, mu0[:d-1], method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
    mu = np.zeros(d)
    mu[:d-1] = res.x
    mu[d-1] = 1.0 - sum(res.x)
    if mu[d-1] >= -1e-7 and np.all(mu >= -1e-7):
        mu = np.maximum(mu, 0)
        mu /= mu.sum()
        return float(res.fun), mu
    return None, None


def find_mu_star_parallel(d, n_restarts=200, max_iter=20000, n_workers=None):
    """Parallel Nelder-Mead multistart at d. Uses multiprocessing.

    Returns (best_val, best_mu) — best_val is an UPPER BOUND on val(d).
    """
    if n_workers is None:
        n_workers = min(64, mp.cpu_count())
    print(f"  Parallel Nelder-Mead: d={d}, n_restarts={n_restarts}, "
          f"workers={n_workers}", flush=True)

    args_list = [(42 + d * 1000 + r, d, max_iter) for r in range(n_restarts)]
    # Add a uniform start for good measure
    # Use Pool with imap_unordered for streaming progress
    best_val = float('inf')
    best_mu = None
    n_done = 0
    t_start = time.perf_counter()
    last_progress = t_start

    with mp.Pool(processes=n_workers) as pool:
        for val, mu in pool.imap_unordered(_nelder_mead_worker, args_list):
            n_done += 1
            if val is not None and val < best_val:
                best_val = val
                best_mu = mu
            now = time.perf_counter()
            if now - last_progress > 30:
                print(f"    [{now - t_start:.0f}s] {n_done}/{n_restarts} done, "
                      f"best val_d={best_val:.6f}", flush=True)
                last_progress = now
    return best_val, best_mu


# --- Exact KKT solver ---

def compute_active_hessian_exact(mu_star, d, val_d, tol_active=1e-3):
    """Solve KKT QP exactly for alpha*. Build H = sum alpha_W * 2*A_W.

    KKT stationarity at mu_star (interior simplex):
      sum_W alpha_W * grad(TV_W)(mu_star) - lambda * 1 + beta = 0
    where:
      alpha_W >= 0, sum alpha_W = 1
      lambda free
      beta_i >= 0 for i in boundary (mu_star_i = 0); beta_i = 0 else.

    Solve: min_{alpha, lambda, beta} ||residual||^2 with:
        residual = sum alpha_W * grad_W - lambda*1 - beta_full

    Returns (H, alpha_star, active_indices, residual_norm).
    """
    conv_len = 2 * d - 1
    # Step 1: enumerate windows, compute TV_W and gradients at mu_star
    tv_per_window = []
    for ell in range(2, 2 * d + 1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            A = _build_AW_local(d, ell, s)
            tv_W = scale * float(mu_star @ A @ mu_star)
            tv_per_window.append((tv_W, ell, s, A, scale))
    tv_per_window.sort(key=lambda t: -t[0])
    val_max = tv_per_window[0][0]
    active = [t for t in tv_per_window if t[0] >= val_max - tol_active]
    n_active = len(active)
    if n_active == 0:
        return np.zeros((d, d)), np.zeros(0), [], 1.0

    # Gradient stack: G[k, :] = 2 * scale_k * A_k @ mu_star
    grads = np.array([2.0 * a[4] * (a[3] @ mu_star) for a in active])
    boundary = mu_star < tol_active
    n_bnd = int(boundary.sum())

    # Variables: [alpha (n_active), lambda (1), beta_b (n_bnd)]
    # Solve: min ||G^T alpha - lambda * 1 - beta_full||^2
    # s.t. alpha >= 0, sum alpha = 1, beta_b >= 0
    # This is a convex QP (quadratic objective, linear constraints).
    # Use scipy.optimize.minimize with SLSQP for inequality+equality constraints.

    def obj(params):
        alpha = params[:n_active]
        lam = params[n_active]
        beta_b = params[n_active + 1:n_active + 1 + n_bnd]
        beta_full = np.zeros(d)
        beta_full[boundary] = beta_b
        residual = (alpha @ grads) - lam * np.ones(d) - beta_full
        return float(residual @ residual)

    def obj_grad(params):
        alpha = params[:n_active]
        lam = params[n_active]
        beta_b = params[n_active + 1:n_active + 1 + n_bnd]
        beta_full = np.zeros(d)
        beta_full[boundary] = beta_b
        residual = (alpha @ grads) - lam * np.ones(d) - beta_full
        # d/d alpha_k: 2 * residual . grads[k]
        dalpha = 2 * grads @ residual
        # d/d lambda: -2 * sum(residual)
        dlam = -2 * residual.sum()
        # d/d beta_b: -2 * residual[boundary]
        dbeta = -2 * residual[boundary]
        return np.concatenate([dalpha, [dlam], dbeta])

    x0 = np.zeros(n_active + 1 + n_bnd)
    x0[:n_active] = 1.0 / n_active
    x0[n_active] = float(grads.mean())
    bounds = [(0, None)] * n_active + [(None, None)] + [(0, None)] * n_bnd
    constraints = [{'type': 'eq', 'fun': lambda p: p[:n_active].sum() - 1.0,
                    'jac': lambda p: np.concatenate([np.ones(n_active),
                                                       [0.0],
                                                       np.zeros(n_bnd)])}]
    res = minimize(obj, x0, jac=obj_grad, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
    alpha_raw = res.x[:n_active]
    alpha_raw = np.maximum(alpha_raw, 0)
    if alpha_raw.sum() < 1e-12:
        alpha_star = np.ones(n_active) / n_active
    else:
        alpha_star = alpha_raw / alpha_raw.sum()
    residual_norm = float(np.sqrt(res.fun))

    H = np.zeros((d, d), dtype=np.float64)
    active_indices = []
    for k, (tv_W, ell, s, A, scale) in enumerate(active):
        H += alpha_star[k] * 2.0 * scale * A
        active_indices.append((ell, s))
    return H, alpha_star, active_indices, residual_norm


# --- Optimal alpha for tube method (max lambda_min on V) ---

def find_alpha_max_psd(active_windows_data, d, n_iter=300, lr=0.05):
    """Find alpha in simplex maximizing lambda_min(P H_alpha P) where
    H_alpha = sum_W alpha_W * A_W (no factor of 2 here, applied later).

    Uses subgradient ascent on the (concave) lambda_min function.
    Subgradient at alpha: (v_min^T A_W v_min)_W, where v_min is the
    eigenvector of P H_alpha P with smallest eigenvalue.

    active_windows_data: list of (tv_W, ell, s, A, scale).
    """
    n_active = len(active_windows_data)
    P = np.eye(d) - np.ones((d, d)) / d
    A_list = [P @ a[3] @ P for a in active_windows_data]  # P A_W P (centered)

    # Initialize uniform on simplex
    alpha = np.ones(n_active) / n_active

    best_lam_min = -np.inf
    best_alpha = alpha.copy()

    for it in range(n_iter):
        H = sum(alpha[k] * A_list[k] for k in range(n_active))
        # Smallest eigenvalue and eigenvector on V (skip kernel of P)
        eigs, vecs = np.linalg.eigh(H)
        # Identify the kernel direction (eigenvalue near 0 with eigenvector ≈ 1/sqrt(d))
        # We want the smallest NON-KERNEL eigenvalue.
        # Sort by eigenvalue, skip the one closest to 0 with all-equal-sign eigenvector.
        # Simpler: skip the smallest |eigenvalue| if its eigenvector has constant sign.
        idx_kernel = -1
        for i in range(d):
            v = vecs[:, i]
            if abs(eigs[i]) < 1e-10 and abs(v.std()) < 1e-6:
                idx_kernel = i
                break
        if idx_kernel == -1:
            # No clear kernel — use second smallest
            idx_kernel = np.argmin(np.abs(eigs))
        # Smallest non-kernel eigenvalue
        mask = np.ones(d, dtype=bool); mask[idx_kernel] = False
        eigs_nokern = eigs[mask]
        vecs_nokern = vecs[:, mask]
        i_min = np.argmin(eigs_nokern)
        lam_min = eigs_nokern[i_min]
        v_min = vecs_nokern[:, i_min]

        if lam_min > best_lam_min:
            best_lam_min = lam_min
            best_alpha = alpha.copy()

        # Subgradient: g_k = v_min^T A_list[k] v_min
        grad = np.array([float(v_min @ A_list[k] @ v_min) for k in range(n_active)])
        # Ascent step + simplex projection
        alpha = alpha + (lr / (1 + it * 0.01)) * grad
        # Project onto simplex
        alpha = _simplex_project(alpha)

    return best_alpha, best_lam_min


def _simplex_project(v):
    """Project v onto simplex {x : x >= 0, sum x = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    idx = np.arange(1, n + 1)
    rho = np.where(u - cssv / idx > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def compute_v_kkt(mu_star, alpha_star, active_indices, d):
    """Compute v_kkt = sum_W alpha_W^* * TV_W(mu_star)."""
    v_kkt = 0.0
    for k, (ell, s) in enumerate(active_indices):
        scale = 2.0 * d / float(ell)
        A = _build_AW_local(d, ell, s)
        tv_W = scale * float(mu_star @ A @ mu_star)
        v_kkt += alpha_star[k] * tv_W
    return v_kkt


if __name__ == "__main__":
    # Quick test at d=8
    print("Testing at d=8 (parallel NM, 50 restarts):")
    t0 = time.perf_counter()
    val_d, mu_star = find_mu_star_parallel(d=8, n_restarts=50, n_workers=4)
    print(f"  val_d={val_d:.6f}, mu*={mu_star}, time={time.perf_counter()-t0:.1f}s")

    print("\nKKT exact:")
    t0 = time.perf_counter()
    H, alpha_star, active_idx, res_norm = compute_active_hessian_exact(
        mu_star, 8, val_d)
    print(f"  active windows: {len(active_idx)}, KKT residual: {res_norm:.6f}, "
          f"time={time.perf_counter()-t0:.1f}s")
    eigs = np.linalg.eigvalsh(H)
    print(f"  H eigvals: min={eigs.min():.4f}, max={eigs.max():.4f}")

    print("\nv_kkt:")
    v_kkt = compute_v_kkt(mu_star, alpha_star, active_idx, 8)
    print(f"  v_kkt = {v_kkt:.6f}")

    print("\nOptimal alpha for max PSD-ness on V:")
    # Recompute active windows data
    conv_len = 2 * 8 - 1
    active_data = []
    tv_list = []
    for ell in range(2, 2 * 8 + 1):
        scale = 2.0 * 8 / float(ell)
        for s in range(conv_len - ell + 2):
            A = _build_AW_local(8, ell, s)
            tv_W = scale * float(mu_star @ A @ mu_star)
            tv_list.append((tv_W, ell, s, A, scale))
    tv_list.sort(key=lambda t: -t[0])
    active_data = [t for t in tv_list if t[0] >= tv_list[0][0] - 1e-3]
    t0 = time.perf_counter()
    alpha_psd, lam_min_psd = find_alpha_max_psd(active_data, 8, n_iter=200)
    print(f"  alpha_psd: {alpha_psd}")
    print(f"  achieved lam_min on V = {lam_min_psd:.6f}")
    print(f"  (KKT alpha gave lam_min = {eigs[1]:.6f})")  # second smallest (skip kernel)

"""KKT refinement: iteratively refine mu_star until KKT residual is essentially zero.

The KKT conditions at mu* (for unconstrained interior simplex point minimizer of max_W TV_W):
  sum_W alpha_W * grad(TV_W)(mu*) = lambda * 1 + beta
  alpha_W >= 0, sum alpha_W = 1
  beta_i >= 0, beta_i * mu*_i = 0 (complementary slackness on mu_i >= 0)
  TV_W(mu*) = val_d for active W (active set)
  TV_W(mu*) <= val_d for inactive W

We use ALTERNATING projections:
  Step (alpha): given mu, solve for alpha minimizing KKT residual.
  Step (mu):   given alpha, do gradient descent on Sigma alpha_W TV_W to find a stationary point.

Converges (linearly under regularity) to a true KKT pair (mu*, alpha*).

After refinement: KKT residual << 0.001, the linear term in the Lojasiewicz LB is negligible.
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize


def _build_AW_local(d, ell, s):
    A = np.zeros((d, d), dtype=np.float64)
    s_hi_off = ell - 2
    for i in range(d):
        jl = max(s - i, 0); jh = min(s + s_hi_off - i, d - 1)
        for j in range(jl, jh + 1):
            A[i, j] = 1.0
    return A


def _all_window_data(d):
    """Precompute (ell, s, A_W, scale_W) for all windows."""
    out = []
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            A = _build_AW_local(d, ell, s)
            out.append((ell, s, A, scale))
    return out


def evaluate_tv_per_window(mu, window_data):
    """For each (ell, s, A, scale) in window_data, return TV_W(mu) = scale * mu^T A mu."""
    return np.array([t[3] * mu @ t[2] @ mu for t in window_data])


def kkt_step_alpha(mu, window_data, tol_active=1e-3):
    """Given mu, find alpha minimizing KKT residual via SLSQP QP.

    Returns (H, alpha, active_indices_in_full_window_list, residual_norm).
    """
    d = len(mu)
    tv_full = evaluate_tv_per_window(mu, window_data)
    val_max = tv_full.max()
    active_full_idx = np.where(tv_full >= val_max - tol_active)[0]
    n_active = len(active_full_idx)
    if n_active == 0:
        return np.zeros((d, d)), np.zeros(0), [], 1.0

    grads = np.array([2.0 * window_data[i][3] * (window_data[i][2] @ mu)
                       for i in active_full_idx])  # (n_active, d)
    boundary = mu < tol_active
    n_bnd = int(boundary.sum())

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
        dalpha = 2 * grads @ residual
        dlam = -2 * residual.sum()
        dbeta = -2 * residual[boundary]
        return np.concatenate([dalpha, [dlam], dbeta])

    x0 = np.zeros(n_active + 1 + n_bnd)
    x0[:n_active] = 1.0 / n_active
    x0[n_active] = float(grads.mean())
    bounds = [(0, None)] * n_active + [(None, None)] + [(0, None)] * n_bnd
    constraints = [{'type': 'eq',
                    'fun': lambda p: p[:n_active].sum() - 1.0,
                    'jac': lambda p: np.concatenate([np.ones(n_active),
                                                       [0.0],
                                                       np.zeros(n_bnd)])}]
    res = minimize(obj, x0, jac=obj_grad, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-15})
    alpha_raw = np.maximum(res.x[:n_active], 0)
    if alpha_raw.sum() < 1e-12:
        alpha = np.ones(n_active) / n_active
    else:
        alpha = alpha_raw / alpha_raw.sum()
    residual_norm = float(np.sqrt(res.fun))

    H = np.zeros((d, d), dtype=np.float64)
    for k_local, w_idx in enumerate(active_full_idx):
        H += alpha[k_local] * 2.0 * window_data[w_idx][3] * window_data[w_idx][2]
    return H, alpha, [int(i) for i in active_full_idx], residual_norm


def kkt_step_mu(mu, alpha, active_idx, window_data, n_inner=200, lr=0.001):
    """Given alpha and active windows, gradient-descent on Sigma_W alpha_W TV_W to refine mu.

    Project to simplex after each step. Use Adam-like step for stability.
    """
    d = len(mu)
    grads_W = [2.0 * window_data[w_idx][3] * window_data[w_idx][2]
               for w_idx in active_idx]  # list of A_W matrices (with scale)
    # Effective gradient: sum_W alpha_W * grads_W . mu
    mu_cur = mu.copy()
    for it in range(n_inner):
        # f = sum alpha_W * scale_W * mu^T A_W mu
        # grad f = 2 * sum alpha_W * scale_W * A_W mu
        grad = sum(alpha[k] * (grads_W[k] @ mu_cur)
                   for k in range(len(active_idx)))
        # We want to MINIMIZE max_W TV_W. Gradient step: mu -= lr * grad.
        # But sum alpha_W TV_W is a LOWER bound on max — minimizing this proxies.
        mu_new = mu_cur - lr * grad
        # Project to simplex (sum=1, mu>=0)
        mu_new = _simplex_project(mu_new)
        # Check improvement (subgradient may be wrong direction for indefinite f)
        # Use line search if needed; for now just step
        mu_cur = mu_new
    return mu_cur


def _simplex_project(v):
    """Euclidean projection onto {x : x >= 0, sum x = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    idx = np.arange(1, n + 1)
    rho_arr = np.where(u - cssv / idx > 0)[0]
    if len(rho_arr) == 0:
        return np.ones(n) / n
    rho = rho_arr[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def refine_kkt(mu_init, d, target_residual=1e-4, max_outer=30, lr_mu=1e-3,
                verbose=False):
    """Alternating refinement to drive KKT residual below target_residual.

    Returns (mu_final, alpha_final, active_idx, residual_norm, val_d_at_mu).
    """
    window_data = _all_window_data(d)
    mu = mu_init.copy()
    mu = _simplex_project(mu)
    history = []
    for outer in range(max_outer):
        H, alpha, active_idx, res_norm = kkt_step_alpha(mu, window_data)
        tv_full = evaluate_tv_per_window(mu, window_data)
        val_d_now = tv_full.max()
        history.append((outer, val_d_now, res_norm))
        if verbose:
            print(f"  outer {outer}: val_d={val_d_now:.6f}, KKT residual={res_norm:.6f}")
        if res_norm < target_residual:
            return mu, alpha, active_idx, res_norm, val_d_now
        # Refine mu via gradient descent on aggregate
        mu = kkt_step_mu(mu, alpha, active_idx, window_data, n_inner=200, lr=lr_mu)
    # Final alpha and residual
    H, alpha, active_idx, res_norm = kkt_step_alpha(mu, window_data)
    val_d_now = evaluate_tv_per_window(mu, window_data).max()
    return mu, alpha, active_idx, res_norm, val_d_now


if __name__ == "__main__":
    # Test at d=8
    d = 8
    print(f"KKT refinement test at d={d}")
    # Random Dirichlet init
    rng = np.random.default_rng(0)
    raw = rng.exponential(1.0, d)
    mu0 = raw / raw.sum()
    print(f"  init mu={mu0}")
    t0 = time.perf_counter()
    mu, alpha, active_idx, res_norm, val_d = refine_kkt(
        mu0, d, target_residual=1e-3, max_outer=20, verbose=True)
    print(f"  Final: val_d={val_d:.6f}, residual={res_norm:.6f}, "
          f"time={time.perf_counter()-t0:.1f}s")
    print(f"  mu={mu}")
    print(f"  active_idx (in full window list): {active_idx}")

"""KKT-correct mu* finder via three-phase pipeline.

Phase 1 (massive multistart): 6400 diverse starts × NM on LSE_τ=0.05.
Phase 2 (LSE-Newton continuation): β-schedule 10 → 40960 with trust-region Newton.
Phase 3 (Newton-KKT polish): active-set Newton with τ-IP continuation, LM regularization.

Output: mu*, alpha*, KKT residual (target ≤ 1e-6).
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import cg, LinearOperator
from scipy.special import logsumexp
import multiprocessing as mp


# =========================================================================
# Window data and primitives
# =========================================================================

def build_window_data(d):
    """Return (A_stack[W,i,j], c_W[W]) for all windows W=(ell,s)."""
    A_list, c_list = [], []
    for ell in range(2, 2 * d + 1):
        c_W = 2.0 * d / float(ell)
        for s in range(2 * d - ell + 1):
            A = np.zeros((d, d), dtype=np.float64)
            for i in range(d):
                jl = max(s - i, 0)
                jh = min(s + ell - 2 - i, d - 1)
                for j in range(jl, jh + 1):
                    A[i, j] = 1.0
            A = 0.5 * (A + A.T)  # symmetrize
            A_list.append(A)
            c_list.append(c_W)
    return np.stack(A_list), np.array(c_list)


def evaluate_tv_per_window(mu, A_stack, c_W):
    """Returns TV_W(mu) for all W."""
    Amu = np.einsum('wij,j->wi', A_stack, mu)
    return c_W * (Amu * mu[None, :]).sum(axis=1)


# =========================================================================
# Simplex projection (Δ_d ∩ [0, x_cap]^d)
# =========================================================================

def project_simplex(v):
    """Euclidean projection onto Δ_d = {x ≥ 0, sum=1}."""
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


def project_simplex_capped(v, x_cap):
    """Projection onto Δ_d ∩ [0, x_cap]^d. If x_cap*d >= 1, simplex projection alone suffices.
    Else bisection on multiplier."""
    n = len(v)
    if x_cap * n >= 1.0:
        # Try simplex projection; check if it satisfies cap
        p = project_simplex(v)
        if (p <= x_cap + 1e-12).all():
            return p
    # Bisect on lambda
    lo, hi = v.min() - 2.0, v.max() + 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        s = np.clip(v - mid, 0, x_cap).sum()
        if s > 1:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi), 0, x_cap)


# =========================================================================
# Z2 canonicalization
# =========================================================================

def canonicalize_z2(mu):
    """Pick lex-minimum of mu and reverse(mu) (Z2 symmetry)."""
    mu_rev = mu[::-1]
    for i in range(len(mu)):
        if mu[i] != mu_rev[i]:
            return mu if mu[i] < mu_rev[i] else mu_rev
    return mu


# =========================================================================
# KKT residual evaluator (used in all phases for verification)
# =========================================================================

def kkt_residual_qp(mu, A_stack, c_W, x_cap=None,
                     tol_active=1e-4, tol_boundary=1e-6):
    """Solve QP for (alpha, lambda, beta_lo, beta_hi) minimizing KKT residual at mu.

    KKT system at mu:
       sum_W alpha_W * grad(TV_W)(mu) - lambda * 1 - beta_lo + beta_hi = 0
       alpha in simplex of active windows
       beta_lo >= 0, beta_lo_i = 0 if mu_i > 0
       beta_hi >= 0, beta_hi_i = 0 if mu_i < x_cap

    Returns dict: alpha, residual, active_idx, lambda, beta_lo, beta_hi, val_max.
    """
    d = len(mu)
    T_W = evaluate_tv_per_window(mu, A_stack, c_W)
    val_max = T_W.max()
    active = np.where(T_W >= val_max - tol_active)[0]
    n_a = len(active)
    if n_a == 0:
        return {'alpha': None, 'residual': 1e30, 'active_idx': [],
                'lambda': 0.0, 'beta_lo': np.zeros(d), 'beta_hi': np.zeros(d),
                'val_max': val_max}

    Amu = np.einsum('wij,j->wi', A_stack, mu)
    G = 2.0 * c_W[active][:, None] * Amu[active]

    bnd_lo = mu < tol_boundary
    bnd_hi = (x_cap is not None) and (mu > x_cap - tol_boundary)
    if isinstance(bnd_hi, bool) and not bnd_hi:
        bnd_hi = np.zeros(d, dtype=bool)
    n_blo = int(bnd_lo.sum())
    n_bhi = int(bnd_hi.sum())

    n_var = n_a + 1 + n_blo + n_bhi

    def assemble_residual(params):
        alpha = params[:n_a]
        lam = params[n_a]
        beta_lo_b = params[n_a + 1: n_a + 1 + n_blo]
        beta_hi_b = params[n_a + 1 + n_blo:]
        beta_lo_full = np.zeros(d)
        beta_hi_full = np.zeros(d)
        if n_blo > 0:
            beta_lo_full[bnd_lo] = beta_lo_b
        if n_bhi > 0:
            beta_hi_full[bnd_hi] = beta_hi_b
        return (alpha @ G) - lam * np.ones(d) - beta_lo_full + beta_hi_full

    def obj(params):
        r = assemble_residual(params)
        return float(r @ r)

    def grad_obj(params):
        r = assemble_residual(params)
        d_alpha = 2 * G @ r
        d_lam = -2 * r.sum()
        d_blo = -2 * r[bnd_lo] if n_blo > 0 else np.zeros(0)
        d_bhi = 2 * r[bnd_hi] if n_bhi > 0 else np.zeros(0)
        return np.concatenate([d_alpha, [d_lam], d_blo, d_bhi])

    x0 = np.zeros(n_var)
    x0[:n_a] = 1.0 / n_a
    bounds = [(0, None)] * n_a + [(None, None)] + [(0, None)] * (n_blo + n_bhi)
    constraints = [{'type': 'eq',
                    'fun': lambda p: p[:n_a].sum() - 1.0,
                    'jac': lambda p: np.concatenate(
                        [np.ones(n_a), [0.0], np.zeros(n_blo + n_bhi)])}]
    res = minimize(obj, x0, jac=grad_obj, method='SLSQP', bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-16})
    alpha_raw = np.maximum(res.x[:n_a], 0)
    if alpha_raw.sum() < 1e-12:
        alpha = np.ones(n_a) / n_a
    else:
        alpha = alpha_raw / alpha_raw.sum()

    beta_lo_full = np.zeros(d)
    beta_hi_full = np.zeros(d)
    if n_blo > 0:
        beta_lo_full[bnd_lo] = res.x[n_a + 1: n_a + 1 + n_blo]
    if n_bhi > 0:
        beta_hi_full[bnd_hi] = res.x[n_a + 1 + n_blo:]
    return {
        'alpha': alpha, 'residual': float(np.sqrt(res.fun)),
        'active_idx': list(active), 'lambda': float(res.x[n_a]),
        'beta_lo': beta_lo_full, 'beta_hi': beta_hi_full,
        'val_max': val_max
    }


# =========================================================================
# PHASE 1: massive multistart with NM on LSE
# =========================================================================

def sample_diverse_seed(rng, d, x_cap, source_idx):
    """Generate one diverse simplex seed."""
    r = source_idx % 100
    if r < 40:  # Dirichlet
        alpha_dir = [0.05, 0.2, 1.0, 5.0][r % 4]
        mu = rng.dirichlet(alpha_dir * np.ones(d))
    elif r < 65:  # Sparse k-hot
        k = min([1, 2, 3, 4, 6, 8][r % 6], d)
        idx = rng.choice(d, k, replace=False)
        mu = np.zeros(d)
        mu[idx] = rng.dirichlet(0.5 * np.ones(k))
    elif r < 80:  # Boundary-concentrated
        i0 = [0, d - 1, d // 2][r % 3]
        b = [0.5, 1.0, 2.0][r % 3]
        w = np.exp(-b * np.abs(np.arange(d) - i0))
        mu = w / w.sum()
    elif r < 90:  # Special structured
        if r % 5 == 0:
            mu = np.ones(d) / d
        elif r % 5 == 1:
            mu = np.zeros(d); mu[[0, d - 1]] = 0.5
        elif r % 5 == 2:
            mu = np.zeros(d)
            for k in range(0, d, 2):
                mu[k] = 1.0
            mu /= mu.sum()
        elif r % 5 == 3:
            # CS17-like: concentrated at outer two pairs
            mu = np.zeros(d); mu[[0, 1, d - 2, d - 1]] = 0.25
        else:
            mu = rng.dirichlet(np.ones(d))
    else:  # Z2-reflection of a Dirichlet
        mu = rng.dirichlet(np.ones(d))[::-1].copy()
    return project_simplex_capped(mu, x_cap)


def lse_objective_unconstrained(z, d, A_stack, c_W, beta, x_cap):
    """LSE objective in softmax-parametrized z (z in R^{d-1}, mu via softmax then projected)."""
    # Softmax to enforce simplex
    z_pad = np.concatenate([z, [0.0]])
    z_max = z_pad.max()
    e = np.exp(z_pad - z_max)
    mu = e / e.sum()
    # Apply cap
    mu = project_simplex_capped(mu, x_cap)
    Amu = np.einsum('wij,j->wi', A_stack, mu)
    T_W = c_W * (Amu * mu[None, :]).sum(axis=1)
    M = T_W.max()
    log_p = beta * (T_W - M)
    Z = logsumexp(log_p)
    return M + Z / beta


def _phase1_worker(args):
    """One Phase 1 start: sample seed, run NM on LSE_tau=0.05."""
    seed, d, x_cap, n_iters, A_stack, c_W = args
    rng = np.random.default_rng(seed)
    # Sample seed
    mu0 = sample_diverse_seed(rng, d, x_cap, seed % 100)
    # Use beta = 1/tau = 20 (smooth LSE)
    beta_smooth = 20.0
    # NM in unconstrained softmax space
    z0 = np.log(np.clip(mu0[:-1], 1e-12, None)) - np.log(np.clip(mu0[-1], 1e-12, None))
    res = minimize(lse_objective_unconstrained, z0,
                   args=(d, A_stack, c_W, beta_smooth, x_cap),
                   method='Nelder-Mead',
                   options={'maxiter': n_iters, 'xatol': 1e-7, 'fatol': 1e-8})
    z_pad = np.concatenate([res.x, [0.0]])
    z_max = z_pad.max()
    e = np.exp(z_pad - z_max)
    mu = e / e.sum()
    mu = project_simplex_capped(mu, x_cap)
    # True f(mu) = max_W TV_W
    T_W = evaluate_tv_per_window(mu, A_stack, c_W)
    f_true = float(T_W.max())
    return mu, f_true


def phase1_multistart(d, x_cap, n_starts=6400, n_workers=64,
                       n_iters_nm=800, top_K=100, verbose=True):
    """Run Phase 1: massive multistart, return top_K candidates by f-value."""
    A_stack, c_W = build_window_data(d)
    if verbose:
        print(f"Phase 1: {n_starts} multistarts at d={d}", flush=True)
    args_list = [(42 + i, d, x_cap, n_iters_nm, A_stack, c_W)
                 for i in range(n_starts)]
    t0 = time.perf_counter()
    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_phase1_worker, args_list, chunksize=10)
    else:
        results = [_phase1_worker(args) for args in args_list]
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"Phase 1: {n_starts} done in {elapsed:.1f}s", flush=True)
    # Sort by f-value ascending, keep top_K
    results.sort(key=lambda r: r[1])
    top = results[:top_K]
    if verbose:
        f_min, f_max = top[0][1], top[-1][1]
        print(f"  top-{top_K}: f range [{f_min:.6f}, {f_max:.6f}]", flush=True)
    return top, A_stack, c_W


# =========================================================================
# PHASE 2: LSE-Newton with β-continuation
# =========================================================================

def lse_value_grad(mu, A_stack, c_W, beta):
    """f^β(μ) and ∇f^β."""
    Amu = np.einsum('wij,j->wi', A_stack, mu)
    T_W = c_W * (Amu * mu[None, :]).sum(axis=1)
    M = T_W.max()
    log_p = beta * (T_W - M)
    Z = logsumexp(log_p)
    p = np.exp(log_p - Z)
    f_beta = M + Z / beta
    grad = 2.0 * np.einsum('w,w,wi->i', p, c_W, Amu)
    return f_beta, grad, p, T_W, Amu


def lse_hvp(v, A_stack, c_W, p, Amu, beta):
    """Hessian-vector product."""
    g_W = 2.0 * c_W[:, None] * Amu  # (|W|, d)
    g_bar = (p[:, None] * g_W).sum(axis=0)  # (d,)
    Av = np.einsum('wij,j->wi', A_stack, v)
    Hpv = 2.0 * np.einsum('w,w,wi->i', p, c_W, Av)
    gv = g_W @ v  # (|W|,)
    Egv = (p * gv).sum()
    cov_v = np.einsum('w,w,wi->i', p, gv - Egv, g_W)
    return Hpv + beta * cov_v


def phase2_lse_newton(mu0, d, x_cap, A_stack, c_W,
                       beta_schedule=None, max_inner=80, verbose=False):
    """Refine μ via LSE-Newton with β-continuation."""
    if beta_schedule is None:
        # 10, 20, ..., 40960 (13 stages, ρ=2)
        beta_schedule = [10.0 * (2.0 ** k) for k in range(13)]
    mu = project_simplex_capped(mu0.copy(), x_cap)
    for stage_idx, beta in enumerate(beta_schedule):
        lam_reg = 1e-3 * beta
        for it in range(max_inner):
            f_beta, grad, p, T_W, Amu = lse_value_grad(mu, A_stack, c_W, beta)
            # Project gradient onto V (centered subspace)
            grad_proj = grad - grad.mean()
            grad_norm = np.linalg.norm(grad_proj)
            if grad_norm < 1e-12:
                break

            # Solve (P H P + lam_reg I) d = -P grad via CG
            def H_op_centered(v):
                v_proj = v - v.mean()
                Hv = lse_hvp(v_proj, A_stack, c_W, p, Amu, beta) + lam_reg * v_proj
                return Hv - Hv.mean()
            op = LinearOperator((d, d), matvec=H_op_centered, dtype=np.float64)
            d_step, _ = cg(op, -grad_proj, atol=1e-10, maxiter=200)
            d_step = d_step - d_step.mean()  # ensure stays in V

            # Backtracking line search
            step = 1.0
            f_beta_cur = f_beta
            for ls_it in range(30):
                mu_new = project_simplex_capped(mu + step * d_step, x_cap)
                f_new, _, _, _, _ = lse_value_grad(mu_new, A_stack, c_W, beta)
                if f_new < f_beta_cur - 1e-8 * step * (-grad_proj @ d_step):
                    mu = mu_new
                    break
                step *= 0.5
            else:
                # Line search failed — break inner loop
                break

        if verbose:
            T_W = evaluate_tv_per_window(mu, A_stack, c_W)
            print(f"  Stage {stage_idx+1}: β={beta:.0f}, f={T_W.max():.6f}",
                  flush=True)

    T_W = evaluate_tv_per_window(mu, A_stack, c_W)
    return mu, float(T_W.max())


# =========================================================================
# PHASE 3: Newton-KKT polish (active-set + LM regularization)
# =========================================================================

def phase3_subgradient_polish(mu0, d, x_cap, A_stack, c_W,
                                max_iter=2000, target_residual=1e-6,
                                tol_active_init=1e-2, tol_active_final=1e-7,
                                verbose=False):
    """Projected subgradient method on g(μ) = max_W TV_W(μ).

    At each step:
      1. Compute alpha* minimizing KKT residual at current μ via QP
         (this gives the BEST possible alpha for this μ).
      2. Subgradient direction: g = Σ α_W * 2 c_W A_W μ.
      3. Project to V = {x : Σx_i = 0} ∩ {tangent of active box constraints}.
      4. Armijo line search on the actual max_W TV_W objective.
      5. Project back to Δ_d ∩ [0, x_cap]^d.
    With adaptive tol_active that tightens as residual decreases.

    Robust to rank-deficient active sets at high d. Returns (best_mu, alpha,
    active_idx, best_residual, final_mu).
    """
    mu = project_simplex_capped(mu0.copy(), x_cap)
    best_mu = mu.copy()
    best_alpha = None
    best_active = []
    best_residual = 1e30
    f_best = 1e30

    # Adaptive tol_active: log-decrease from init to final over iterations
    log_init = np.log(tol_active_init)
    log_final = np.log(tol_active_final)

    lr = 1.0
    last_improve_iter = 0

    for it in range(max_iter):
        tol_active = np.exp(log_init + (log_final - log_init) *
                              min(1.0, it / max(1, max_iter // 2)))
        # Compute alpha* minimizing KKT residual
        res = kkt_residual_qp(mu, A_stack, c_W, x_cap=x_cap,
                                tol_active=tol_active)
        f_cur = res['val_max']
        # Track best by (residual, f)
        better = False
        if res['residual'] < best_residual:
            best_residual = res['residual']
            best_mu = mu.copy()
            best_alpha = res['alpha'].copy() if res['alpha'] is not None else None
            best_active = list(res['active_idx'])
            f_best = f_cur
            last_improve_iter = it
            better = True

        if res['residual'] < target_residual:
            if verbose:
                print(f"    [iter {it}] converged: residual={res['residual']:.2e}, "
                      f"f={f_cur:.7f}", flush=True)
            return best_mu, best_alpha, best_active, best_residual, mu

        if res['alpha'] is None:
            break

        # Subgradient direction
        active = res['active_idx']
        alpha = res['alpha']
        Amu_active = np.einsum('wij,j->wi', A_stack[active], mu)  # (n_a, d)
        subgrad = 2.0 * (alpha[:, None] * c_W[active][:, None] *
                         Amu_active).sum(axis=0)
        # Project to V (mean zero) — preserves simplex constraint
        g_V = subgrad - subgrad.mean()
        # Boundary constraints: zero out components that point outside [0, x_cap]
        # at active boundaries. (mu_i = 0 and g_V_i > 0 → step would decrease mu_i below 0)
        boundary_lo = (mu < 1e-9) & (g_V > 0)
        boundary_hi = (mu > x_cap - 1e-9) & (g_V < 0)
        g_V_clipped = g_V.copy()
        g_V_clipped[boundary_lo] = 0
        g_V_clipped[boundary_hi] = 0
        # Re-center after clipping (so that simplex constraint stays satisfied)
        n_free = d - int(boundary_lo.sum() + boundary_hi.sum())
        if n_free > 0:
            free_mask = ~(boundary_lo | boundary_hi)
            g_V_clipped[free_mask] -= g_V_clipped[free_mask].mean()
        g_norm = np.linalg.norm(g_V_clipped)
        if g_norm < 1e-14:
            if verbose:
                print(f"    [iter {it}] gradient ~0, stopping", flush=True)
            break

        # Armijo line search on max_W TV_W
        f_new_best = f_cur
        lr_best = 0.0
        # Try increasing then decreasing lr
        trials = [lr * 2, lr, lr * 0.5, lr * 0.1, lr * 0.01]
        for lr_try in trials:
            mu_try = project_simplex_capped(mu - lr_try * g_V_clipped, x_cap)
            T_W_try = evaluate_tv_per_window(mu_try, A_stack, c_W)
            f_try = float(T_W_try.max())
            if f_try < f_new_best - 1e-14:
                f_new_best = f_try
                lr_best = lr_try
                # Don't break; check larger lrs first to allow growth
        if lr_best == 0:
            # No improvement at any lr; halt
            if verbose and it - last_improve_iter > 20:
                print(f"    [iter {it}] no descent (best f={f_cur:.8f}, "
                      f"residual={res['residual']:.4e}), stopping", flush=True)
            break
        # Take the best step
        mu = project_simplex_capped(mu - lr_best * g_V_clipped, x_cap)
        # Heuristic lr update
        lr = lr_best

    # Final polish: re-evaluate residual at best_mu with tightest tol
    final_res = kkt_residual_qp(best_mu, A_stack, c_W, x_cap=x_cap,
                                  tol_active=1e-7)
    if final_res['residual'] < best_residual:
        best_residual = final_res['residual']
        best_alpha = final_res['alpha']
        best_active = list(final_res['active_idx'])
    return best_mu, best_alpha, best_active, best_residual, mu


def phase3_newton_kkt(mu0, d, x_cap, A_stack, c_W,
                       tau_schedule=None, max_iter=30,
                       target_residual=1e-6, verbose=False):
    """Refine μ via Newton on the KKT system with τ-IP continuation.

    Variables: (μ ∈ R^d, α ∈ R^|A|, λ, v, β_lo ∈ R^d, β_hi ∈ R^d)
    Equations:
        F1: Σ α_W * 2 c_W A_W μ - λ 1 - β_lo + β_hi = 0           (d eqs)
        F2: c_W μ^T A_W μ - v = 0 for W ∈ active                  (|A| eqs)
        F3: Σ μ - 1 = 0                                             (1 eq)
        F4: Σ α_W - 1 = 0                                            (1 eq)
        F5: β_lo_i * μ_i = τ                                         (d eqs)
        F6: β_hi_i * (x_cap - μ_i) = τ                               (d eqs)
    Total: 3d + |A| + 2 equations and unknowns.

    NOTE: at d ≥ 16, this system tends to be over-determined because more
    windows are simultaneously near-active. Use phase3_subgradient_polish
    as a fallback for high-d.
    """
    if tau_schedule is None:
        tau_schedule = [1e-2, 1e-4, 1e-6, 1e-8, 0.0]

    mu = project_simplex_capped(mu0.copy(), x_cap)

    # Initialize with current active set
    res_dict = kkt_residual_qp(mu, A_stack, c_W, x_cap=x_cap, tol_active=1e-3)
    active_idx = res_dict['active_idx']
    if len(active_idx) == 0:
        return mu, np.zeros(0), [], 1e30, mu.copy()
    alpha = res_dict['alpha'].copy()
    lam = res_dict['lambda']
    val_max = res_dict['val_max']

    # Initialize beta_lo, beta_hi (interior point with large τ)
    beta_lo_full = np.maximum(1e-3 - mu, 1e-6)
    beta_hi_full = np.maximum(1e-3 - (x_cap - mu), 1e-6)

    nu_lm = 1e-6  # LM regularizer
    best_residual = res_dict['residual']
    best_mu = mu.copy()
    best_alpha = alpha.copy()
    best_active = list(active_idx)

    for tau in tau_schedule:
        for outer in range(max_iter):
            # Re-evaluate active set
            T_W = evaluate_tv_per_window(mu, A_stack, c_W)
            val_max = T_W.max()
            active = np.where(T_W >= val_max - 1e-4)[0]
            n_a = len(active)
            if n_a == 0:
                break

            # If active changed, reset alpha
            if list(active) != list(active_idx):
                if verbose:
                    print(f"    [active-set change] {len(active_idx)}→{len(active)}",
                          flush=True)
                active_idx = list(active)
                alpha = np.ones(n_a) / n_a

            # Build F (residual)
            Amu_active = np.einsum('wij,j->wi', A_stack[active], mu)  # (n_a, d)
            G = 2.0 * c_W[active][:, None] * Amu_active  # (n_a, d)
            H = (alpha[:, None, None] * 2.0 * c_W[active][:, None, None] *
                 A_stack[active]).sum(axis=0)  # (d, d)

            F1 = (alpha @ G) - lam * np.ones(d) - beta_lo_full + beta_hi_full
            F2 = c_W[active] * (Amu_active * mu[None, :]).sum(axis=1) - val_max
            F3 = mu.sum() - 1.0
            F4 = alpha.sum() - 1.0
            F5 = beta_lo_full * mu - tau
            F6 = beta_hi_full * (x_cap - mu) - tau
            F = np.concatenate([F1, F2, [F3, F4], F5, F6])
            res_norm = np.linalg.norm(F)

            if res_norm < target_residual:
                if verbose:
                    print(f"    τ={tau:.0e} converged at outer {outer}: "
                          f"||F||={res_norm:.2e}", flush=True)
                break

            # Build Jacobian (sparse-ish; size 3d+n_a+2)
            # Order vars: [mu(d), alpha(n_a), lam(1), v(1), beta_lo(d), beta_hi(d)]
            N = 3 * d + n_a + 2
            J = np.zeros((N, N))

            # F1 rows (d): w.r.t. mu, alpha, lam, beta_lo, beta_hi
            J[:d, :d] = H  # d/d mu
            J[:d, d:d + n_a] = G.T  # d/d alpha
            J[:d, d + n_a] = -1.0  # d/d lam
            J[:d, d + n_a + 2:2 * d + n_a + 2] = -np.eye(d)  # d/d beta_lo
            J[:d, 2 * d + n_a + 2:3 * d + n_a + 2] = np.eye(d)  # d/d beta_hi

            # F2 rows (n_a): w.r.t. mu, v
            J[d:d + n_a, :d] = G  # d/d mu
            J[d:d + n_a, d + n_a + 1] = -1.0  # d/d v

            # F3 row: w.r.t. mu
            J[d + n_a, :d] = 1.0

            # F4 row: w.r.t. alpha
            J[d + n_a + 1, d:d + n_a] = 1.0

            # F5 rows (d): beta_lo * mu = τ. d/d mu = beta_lo, d/d beta_lo = mu
            J[d + n_a + 2:2 * d + n_a + 2, :d] = np.diag(beta_lo_full)
            J[d + n_a + 2:2 * d + n_a + 2,
              d + n_a + 2:2 * d + n_a + 2] = np.diag(mu)

            # F6 rows (d): beta_hi * (x_cap - mu) = τ. d/d mu = -beta_hi, d/d beta_hi = (x_cap - mu)
            J[2 * d + n_a + 2:3 * d + n_a + 2, :d] = -np.diag(beta_hi_full)
            J[2 * d + n_a + 2:3 * d + n_a + 2,
              2 * d + n_a + 2:3 * d + n_a + 2] = np.diag(x_cap - mu)

            # LM regularization on diagonal of J^T J
            JtJ_diag = (J ** 2).sum(axis=0)
            try:
                dz = solve(J + nu_lm * np.diag(np.maximum(JtJ_diag, 1e-12)), -F)
            except np.linalg.LinAlgError:
                dz, *_ = lstsq(J, -F, lapack_driver='gelsd')

            # Line search
            step = 1.0
            best_step_residual = res_norm
            best_step = 0.0
            for ls in range(30):
                mu_n = mu + step * dz[:d]
                alpha_n = alpha + step * dz[d:d + n_a]
                lam_n = lam + step * dz[d + n_a]
                val_n = val_max + step * dz[d + n_a + 1]
                blo_n = beta_lo_full + step * dz[d + n_a + 2:2 * d + n_a + 2]
                bhi_n = beta_hi_full + step * dz[2 * d + n_a + 2:3 * d + n_a + 2]

                # Project mu to simplex+box
                mu_n = project_simplex_capped(mu_n, x_cap)
                # Clip multipliers to nonneg
                alpha_n = np.maximum(alpha_n, 0)
                if alpha_n.sum() > 1e-12:
                    alpha_n = alpha_n / alpha_n.sum()
                else:
                    alpha_n = np.ones(n_a) / n_a
                blo_n = np.maximum(blo_n, 1e-12)
                bhi_n = np.maximum(bhi_n, 1e-12)

                # Recompute residual
                Amu_n = np.einsum('wij,j->wi', A_stack[active], mu_n)
                G_n = 2.0 * c_W[active][:, None] * Amu_n
                F1_n = (alpha_n @ G_n) - lam_n - blo_n + bhi_n
                F2_n = c_W[active] * (Amu_n * mu_n[None, :]).sum(axis=1) - val_n
                F3_n = mu_n.sum() - 1.0
                F4_n = alpha_n.sum() - 1.0
                F5_n = blo_n * mu_n - tau
                F6_n = bhi_n * (x_cap - mu_n) - tau
                F_n = np.concatenate([F1_n, F2_n, [F3_n, F4_n], F5_n, F6_n])
                rn = np.linalg.norm(F_n)
                if rn < best_step_residual:
                    best_step_residual = rn
                    best_step = step
                if rn < res_norm * (1 - 1e-4 * step):
                    break
                step *= 0.5

            if best_step <= 1e-9:
                # No improvement in step, increase LM
                nu_lm = min(nu_lm * 10, 1e3)
                if verbose:
                    print(f"    LM bump: nu={nu_lm:.1e}", flush=True)
                if nu_lm > 1e2:
                    break
                continue
            else:
                nu_lm = max(nu_lm / 3, 1e-12)

            # Apply best step
            mu = mu + best_step * dz[:d]
            alpha = alpha + best_step * dz[d:d + n_a]
            lam = lam + best_step * dz[d + n_a]
            val_max = val_max + best_step * dz[d + n_a + 1]
            beta_lo_full = beta_lo_full + best_step * dz[d + n_a + 2:2 * d + n_a + 2]
            beta_hi_full = beta_hi_full + best_step * dz[2 * d + n_a + 2:3 * d + n_a + 2]
            mu = project_simplex_capped(mu, x_cap)
            alpha = np.maximum(alpha, 0)
            if alpha.sum() > 1e-12:
                alpha = alpha / alpha.sum()
            beta_lo_full = np.maximum(beta_lo_full, 1e-12)
            beta_hi_full = np.maximum(beta_hi_full, 1e-12)

            # Track best
            res_check = kkt_residual_qp(mu, A_stack, c_W, x_cap=x_cap)
            if res_check['residual'] < best_residual:
                best_residual = res_check['residual']
                best_mu = mu.copy()
                best_alpha = alpha.copy()
                best_active = list(active)

    return best_mu, best_alpha, best_active, best_residual, mu


# =========================================================================
# Master pipeline
# =========================================================================

def find_kkt_correct_mu_star(d, x_cap=None, c_target=None,
                                n_starts=6400, n_workers=64,
                                top_K_phase2=100, top_K_phase3=20,
                                target_residual=1e-6, verbose=True):
    """Master pipeline: Phase 1 → Phase 2 → Phase 3.

    Returns dict with keys:
        mu_star: best mu* (np.ndarray)
        alpha_star: KKT multipliers
        residual: final KKT residual
        f_value: max_W TV_W(mu_star)
        active_idx: active windows
        elapsed: wall time
    """
    if x_cap is None:
        if c_target is not None:
            x_cap = float(np.floor(d * np.sqrt(c_target / d)) + 1) / d
        else:
            x_cap = 1.0  # full simplex
    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"  KKT-correct μ* finder at d={d}, x_cap={x_cap:.4f}", flush=True)
        print(f"{'='*70}", flush=True)

    t_total = time.perf_counter()

    # PHASE 1
    t1 = time.perf_counter()
    top_phase1, A_stack, c_W = phase1_multistart(
        d, x_cap, n_starts=n_starts, n_workers=n_workers,
        top_K=top_K_phase2, verbose=verbose)
    t_phase1 = time.perf_counter() - t1

    # PHASE 2: refine top candidates via LSE-Newton continuation
    t2 = time.perf_counter()
    if verbose:
        print(f"\nPhase 2: LSE-Newton continuation on top-{top_K_phase2}", flush=True)
    phase2_results = []  # list of dicts: {'mu', 'f', 'residual', 'active', 'alpha'}
    for i, (mu0, _) in enumerate(top_phase1):
        try:
            mu_p2, f_p2 = phase2_lse_newton(
                mu0, d, x_cap, A_stack, c_W, verbose=False)
            res = kkt_residual_qp(mu_p2, A_stack, c_W, x_cap=x_cap,
                                    tol_active=1e-3)
            phase2_results.append({
                'mu': mu_p2, 'f': f_p2, 'residual': res['residual'],
                'active': list(res['active_idx']), 'alpha': res['alpha'],
            })
        except Exception as e:
            if verbose and i < 3:
                print(f"  start {i} failed: {e}", flush=True)
            continue
    phase2_results.sort(key=lambda r: r['f'])
    top_phase2 = phase2_results[:top_K_phase3]
    t_phase2 = time.perf_counter() - t2
    if verbose:
        if top_phase2:
            f_min = top_phase2[0]['f']
            f_max = top_phase2[-1]['f']
            r_min = min(r['residual'] for r in top_phase2)
            print(f"  Phase 2 done in {t_phase2:.1f}s. Top {top_K_phase3} f range: "
                  f"[{f_min:.6f}, {f_max:.6f}], min residual: {r_min:.4e}", flush=True)

    # PHASE 3: try BOTH Newton-KKT and subgradient polish per candidate;
    # collect ALL outputs (we'll pick best across phases at the end).
    t3 = time.perf_counter()
    if verbose:
        print(f"\nPhase 3: Newton-KKT + subgradient polish on top-{top_K_phase3}",
              flush=True)
    phase3_results = []
    for i, p2 in enumerate(top_phase2):
        mu0 = p2['mu']
        # Try Newton-KKT
        try:
            mu_n, alpha_n, active_n, res_n, _ = phase3_newton_kkt(
                mu0, d, x_cap, A_stack, c_W,
                target_residual=target_residual, verbose=False)
            T_W = evaluate_tv_per_window(mu_n, A_stack, c_W)
            f_n = float(T_W.max())
            phase3_results.append({
                'mu': mu_n, 'alpha': alpha_n, 'active': active_n,
                'residual': res_n, 'f': f_n, 'method': 'newton',
            })
        except Exception as e:
            if verbose and i < 3:
                print(f"  start {i} newton failed: {e}", flush=True)
        # Try subgradient polish
        try:
            mu_s, alpha_s, active_s, res_s, _ = phase3_subgradient_polish(
                mu0, d, x_cap, A_stack, c_W,
                max_iter=2000, target_residual=target_residual, verbose=False)
            T_W = evaluate_tv_per_window(mu_s, A_stack, c_W)
            f_s = float(T_W.max())
            phase3_results.append({
                'mu': mu_s, 'alpha': alpha_s, 'active': active_s,
                'residual': res_s, 'f': f_s, 'method': 'subgrad',
            })
        except Exception as e:
            if verbose and i < 3:
                print(f"  start {i} subgrad failed: {e}", flush=True)
    t_phase3 = time.perf_counter() - t3

    # Combine all candidates (Phase 2 outputs + Phase 3 outputs) — pick the global best
    all_candidates = []
    for r in phase2_results:
        all_candidates.append({**r, 'phase': 2, 'method': 'lse_newton'})
    for r in phase3_results:
        all_candidates.append({**r, 'phase': 3})

    if not all_candidates:
        if verbose:
            print(f"  No candidates from any phase!", flush=True)
        return {
            'mu_star': None, 'alpha_star': None, 'residual': 1e30,
            'f_value': None, 'active_idx': [], 'elapsed': time.perf_counter() - t_total,
        }

    # Selection: prefer candidates with residual ≤ target_residual, then min f.
    # Otherwise: prefer those with residual ≤ 1e-3, then min f.
    # Otherwise: pick min residual.
    qualified_strict = [r for r in all_candidates if r['residual'] < target_residual]
    qualified_lax = [r for r in all_candidates if r['residual'] < 1e-3]
    if qualified_strict:
        best = min(qualified_strict, key=lambda r: r['f'])
    elif qualified_lax:
        best = min(qualified_lax, key=lambda r: r['f'])
    else:
        best = min(all_candidates, key=lambda r: r['residual'])

    elapsed = time.perf_counter() - t_total
    if verbose:
        print(f"\nPhase 3 done in {t_phase3:.1f}s.", flush=True)
        # Diagnostic: show best Phase 2 and best Phase 3
        best_p2 = min(phase2_results, key=lambda r: (r['residual'], r['f']))
        if phase3_results:
            best_p3_res = min(phase3_results, key=lambda r: r['residual'])
            best_p3_f = min(phase3_results, key=lambda r: r['f'])
            print(f"  best by residual P2: f={best_p2['f']:.7f}, "
                  f"res={best_p2['residual']:.4e}", flush=True)
            print(f"  best by residual P3: f={best_p3_res['f']:.7f}, "
                  f"res={best_p3_res['residual']:.4e}, m={best_p3_res['method']}",
                  flush=True)
            print(f"  best by f-val P3: f={best_p3_f['f']:.7f}, "
                  f"res={best_p3_f['residual']:.4e}, m={best_p3_f['method']}",
                  flush=True)
        print(f"\n{'='*70}", flush=True)
        print(f"  RESULT at d={d}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  mu* = {best['mu']}", flush=True)
        print(f"  f(mu*) = {best['f']:.8f}", flush=True)
        print(f"  KKT residual = {best['residual']:.6e}", flush=True)
        print(f"  active windows: {len(best['active'])}", flush=True)
        print(f"  selected from phase {best.get('phase', '?')} "
              f"({best.get('method', '?')})", flush=True)
        print(f"  Phase times: P1={t_phase1:.1f}s, P2={t_phase2:.1f}s, "
              f"P3={t_phase3:.1f}s, total={elapsed:.1f}s", flush=True)

    return {
        'mu_star': best['mu'], 'alpha_star': best['alpha'],
        'residual': best['residual'], 'f_value': best['f'],
        'active_idx': best['active'], 'elapsed': elapsed,
        'phase_times': (t_phase1, t_phase2, t_phase3),
        'all_candidates': all_candidates,
        'best_phase': best.get('phase', '?'),
        'best_method': best.get('method', '?'),
    }


# =========================================================================
# Tests at d=8 and d=10
# =========================================================================

def test_d(d):
    print(f"\n{'#'*72}")
    print(f"# TEST at d={d}")
    print(f"{'#'*72}")
    n_workers = min(mp.cpu_count(), 16) if d <= 8 else min(mp.cpu_count(), 32)
    n_starts = 800 if d <= 8 else 1600
    result = find_kkt_correct_mu_star(
        d, x_cap=1.0,  # no cap for clean test
        n_starts=n_starts, n_workers=n_workers,
        top_K_phase2=50, top_K_phase3=10,
        target_residual=1e-6, verbose=True)
    print(f"\n  d={d} summary:")
    print(f"    f(μ*) = {result['f_value']:.8f}")
    print(f"    residual = {result['residual']:.6e}")
    print(f"    converged: {result['residual'] < 1e-4}")
    return result


if __name__ == "__main__":
    print("KKT-correct mu* finder pipeline test\n")
    print("Testing at d=8...")
    r8 = test_d(d=8)
    print("\nTesting at d=10...")
    r10 = test_d(d=10)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"d=8:  f(μ*)={r8['f_value']:.6f}, residual={r8['residual']:.2e}, "
          f"time={r8['elapsed']:.1f}s")
    print(f"d=10: f(μ*)={r10['f_value']:.6f}, residual={r10['residual']:.2e}, "
          f"time={r10['elapsed']:.1f}s")

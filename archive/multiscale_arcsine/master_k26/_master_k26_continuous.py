"""Continuous-mixture multi-scale arcsine optimization.

K(x) = sum_{i=1..N} lambda_i * K_arc(x; delta_i),  delta_i in (0, DELTA].
K_hat(xi) = sum_i lambda_i * J_0(pi * delta_i * xi)^2.

We pre-discretise delta_i on a fine uniform grid and search for the
probability measure nu = (lambda_1, ..., lambda_N) maximising M_cert(nu).

Methods:
  (1) Random Dirichlet multi-start (10^5 trials).
  (2) Frank-Wolfe with finite-difference gradient (linearise; jump to best
      atom, then perform line search).
  (3) Softmax-parametrised projected gradient ascent (Adam-style).
  (4) Pairwise coordinate descent over (lambda_i, lambda_j) - golden-section.

Saves results to _master_k26_continuous.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.special import j0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, MV_COEFFS, N_QP, U, mv_master_M_cert  # noqa: E402


# ----------------------- Pre-computation -----------------------------

QP_XI = np.arange(1, N_QP + 1) / U   # length 119


def _Cij_exact(d_i, d_j, eps_rel=1e-10):
    """C_ij = int_{-inf}^{inf} J_0(pi d_i xi)^2 J_0(pi d_j xi)^2 dxi.

    Equals 2 * int_0^inf since integrand is even.  This is the (i,j) entry
    of the Gram-like matrix s.t. K_2 = sum_ij lambda_i lambda_j C_ij.
    """
    def integrand(xi):
        a = j0(np.pi * d_i * xi)
        b = j0(np.pi * d_j * xi)
        return (a * b) ** 2
    # Adaptive scheme.  J_0(x)^2 oscillates; split into [0, 50] and [50, inf].
    I1, _ = quad(integrand, 0, 50, limit=500, epsrel=eps_rel)
    I2, _ = quad(integrand, 50, np.inf, limit=500, epsrel=eps_rel)
    return 2.0 * (I1 + I2)


def precompute_atoms(N_grid: int, verbose=False, cache_dir=None):
    """Wrapped: tries cached load first."""
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f"_master_k26_cache_N{N_grid}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return d["deltas"], d["C"], d["Kh_qp"], d["Kh_1"]
    deltas, C, Kh_qp, Kh_1 = _precompute_atoms_uncached(N_grid, verbose=verbose)
    if cache_dir is not None:
        np.savez(cache_path, deltas=deltas, C=C, Kh_qp=Kh_qp, Kh_1=Kh_1)
    return deltas, C, Kh_qp, Kh_1


def _precompute_atoms_uncached(N_grid: int, verbose=False):
    """Pre-compute kernels for each delta_i = (i/N_grid) * DELTA, i = 1..N_grid.

    Uses EXACT C_ij via scipy.quad (adaptive), so K_2 = lambda^T C lambda is
    high-precision (eliminates the XI_MAX tail-truncation bias).

    Returns:
        deltas:    shape (N,)
        C:         shape (N, N)    K_2 = lambda^T @ C @ lambda
        Kh_qp:     shape (N, N_QP) = J_0(pi delta_i (j/U))^2 for j = 1..119
        Kh_1:      shape (N,)      = J_0(pi delta_i * 1)^2
    """
    deltas = np.linspace(DELTA / N_grid, DELTA, N_grid)
    Kh_qp = np.empty((N_grid, N_QP), dtype=np.float64)
    Kh_1 = np.empty(N_grid, dtype=np.float64)
    for i, d in enumerate(deltas):
        Kh_qp[i] = j0(np.pi * d * QP_XI) ** 2
        Kh_1[i] = j0(np.pi * d * 1.0) ** 2
    # C_ij matrix
    C = np.empty((N_grid, N_grid), dtype=np.float64)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_grid):
            for j in range(i, N_grid):
                C[i, j] = _Cij_exact(deltas[i], deltas[j])
                C[j, i] = C[i, j]
            if verbose and i % 10 == 0:
                print(f"    C row {i+1}/{N_grid}")
    return deltas, C, Kh_qp, Kh_1


# ----------------------- Evaluator -----------------------------------

def eval_nu(lambdas, C, Kh_qp, Kh_1, mv_coeffs=MV_COEFFS):
    """Compute M_cert for nu = lambdas (sum = 1, >=0).

    K_2 = lambda^T C lambda  (exact).
    """
    K_2 = float(lambdas @ C @ lambdas)
    k_1 = float(lambdas @ Kh_1)
    kh_qp = lambdas @ Kh_qp       # (N_QP,)
    if np.any(kh_qp < 1e-20):
        return None, k_1, K_2, None
    S_1 = float(np.sum((mv_coeffs ** 2) / kh_qp))
    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    return M_cert, k_1, K_2, S_1


# ----------------------- Random Dirichlet multi-start -----------------

def random_dirichlet_search(C, Kh_qp, Kh_1, n_trials, alpha_list,
                            seed=0, sparse_support_sizes=None):
    """Sample probability vectors from various Dirichlet alphas and uniform
    sparse-support distributions; return best."""
    rng = np.random.default_rng(seed)
    N = Kh_1.shape[0]
    best_M = -np.inf
    best_lam = None
    best_kk = None
    per_trial = max(1, n_trials // (len(alpha_list)
                                    + (len(sparse_support_sizes or [])
                                       if sparse_support_sizes else 0)))
    for alpha in alpha_list:
        a = np.full(N, alpha)
        for _ in range(per_trial):
            lam = rng.dirichlet(a)
            M, k1, K2, S1 = eval_nu(lam, C, Kh_qp, Kh_1)
            if M is not None and M > best_M:
                best_M = M
                best_lam = lam.copy()
                best_kk = (k1, K2, S1)
    if sparse_support_sizes:
        for k in sparse_support_sizes:
            for _ in range(per_trial):
                idx = rng.choice(N, size=k, replace=False)
                w = rng.dirichlet(np.ones(k))
                lam = np.zeros(N)
                lam[idx] = w
                M, k1, K2, S1 = eval_nu(lam, C, Kh_qp, Kh_1)
                if M is not None and M > best_M:
                    best_M = M
                    best_lam = lam.copy()
                    best_kk = (k1, K2, S1)
    return best_lam, best_M, best_kk


# ----------------------- Frank-Wolfe ----------------------------------

def frank_wolfe(lam0, C, Kh_qp, Kh_1, n_iters=200, fd_step=1e-4,
                tol=1e-9, verbose=False):
    """Frank-Wolfe over the probability simplex.

    Gradient g_i = d M_cert / d lambda_i via finite differences (one-sided).
    LP step: vertex e_{i*} where i* = argmax g_i, projected back; line search
    on [0,1] along p = e_{i*} - lam.
    """
    N = Kh_1.shape[0]
    lam = lam0.copy()
    M_curr, _, _, _ = eval_nu(lam, C, Kh_qp, Kh_1)
    if M_curr is None:
        M_curr = -np.inf
    history = [M_curr]
    for it in range(n_iters):
        # Finite-difference gradient w.r.t. lambda_i, holding the simplex
        # constraint: g_i = M(lam + h*(e_i - lam)) - M(lam) / h. Equivalent
        # to directional derivative toward e_i.
        g = np.empty(N)
        for i in range(N):
            lam_p = (1 - fd_step) * lam
            lam_p[i] += fd_step
            Mp, _, _, _ = eval_nu(lam_p, C, Kh_qp, Kh_1)
            g[i] = (Mp - M_curr) / fd_step if Mp is not None else -np.inf
        i_star = int(np.argmax(g))
        if g[i_star] < tol:
            break
        # Line search on s in [0, 1]: lam_new = (1 - s) * lam + s * e_{i_star}
        def neg_M(s):
            lam_s = (1 - s) * lam
            lam_s[i_star] += s
            M, _, _, _ = eval_nu(lam_s, C, Kh_qp, Kh_1)
            return -M if M is not None else np.inf
        res = minimize_scalar(neg_M, bounds=(0.0, 1.0), method='bounded',
                              options={'xatol': 1e-6})
        s_best = res.x
        M_new = -res.fun
        if not (M_new > M_curr + 1e-10):
            break
        lam = (1 - s_best) * lam
        lam[i_star] += s_best
        M_curr = M_new
        history.append(M_curr)
        if verbose:
            print(f"  FW it {it}: i*={i_star}, s={s_best:.4f}, M={M_curr:.6f}")
    return lam, M_curr, history


# ----------------------- Pairwise coordinate descent ------------------

def pairwise_cd(lam0, C, Kh_qp, Kh_1, n_sweeps=10, verbose=False):
    """For each (i, j) pair, optimize the mass split between them holding
    the rest fixed.  Mass m = lam[i] + lam[j]; line search t in [0, m] for
    new lam[i].
    """
    lam = lam0.copy()
    M_curr, _, _, _ = eval_nu(lam, C, Kh_qp, Kh_1)
    N = Kh_1.shape[0]
    active = list(np.where(lam > 1e-8)[0])
    # Include some inactive atoms occasionally.
    for sweep in range(n_sweeps):
        improved = False
        # Always consider all active atoms; also random inactive ones.
        candidates = list(set(active) | set(np.argsort(-lam)[:max(20, len(active))]))
        for i in candidates:
            for j in candidates:
                if i >= j:
                    continue
                m = lam[i] + lam[j]
                if m < 1e-9:
                    continue

                def neg_M(t):
                    if t < 0 or t > m:
                        return np.inf
                    lam_t = lam.copy()
                    lam_t[i] = t
                    lam_t[j] = m - t
                    M, _, _, _ = eval_nu(lam_t, C, Kh_qp, Kh_1)
                    return -M if M is not None else np.inf
                res = minimize_scalar(neg_M, bounds=(0.0, m), method='bounded',
                                      options={'xatol': 1e-7})
                M_new = -res.fun
                if M_new > M_curr + 1e-9:
                    lam[i] = res.x
                    lam[j] = m - res.x
                    M_curr = M_new
                    improved = True
        active = list(np.where(lam > 1e-8)[0])
        if verbose:
            print(f"  CD sweep {sweep}: M={M_curr:.6f}, support={len(active)}")
        if not improved:
            break
    return lam, M_curr


# ----------------------- Softmax projected GA -------------------------

def softmax_grad_ascent(lam0, C, Kh_qp, Kh_1, n_iters=200, lr=0.05,
                       fd_step=1e-4, verbose=False):
    """Softmax parametrisation: lambda_i = exp(z_i) / sum exp(z_j).
    Adam-style gradient ascent on z.
    """
    N = Kh_1.shape[0]
    # Initialise z from lam0 (add eps to avoid -inf)
    eps = 1e-12
    z = np.log(lam0 + eps)
    z -= z.mean()
    m_a, v_a = np.zeros(N), np.zeros(N)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    def lam_of(z):
        zc = z - z.max()
        e = np.exp(zc)
        return e / e.sum()
    M_curr, _, _, _ = eval_nu(lam_of(z), C, Kh_qp, Kh_1)
    best_M = M_curr
    best_z = z.copy()
    for t in range(1, n_iters + 1):
        lam = lam_of(z)
        # FD gradient on z.
        g = np.empty(N)
        for i in range(N):
            zp = z.copy()
            zp[i] += fd_step
            Mp, _, _, _ = eval_nu(lam_of(zp), C, Kh_qp, Kh_1)
            g[i] = (Mp - M_curr) / fd_step if Mp is not None else 0.0
        # Adam update (ascent: z += lr * adam(g))
        m_a = beta1 * m_a + (1 - beta1) * g
        v_a = beta2 * v_a + (1 - beta2) * g * g
        m_hat = m_a / (1 - beta1 ** t)
        v_hat = v_a / (1 - beta2 ** t)
        z = z + lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        M_curr, _, _, _ = eval_nu(lam_of(z), C, Kh_qp, Kh_1)
        if M_curr is not None and M_curr > best_M:
            best_M = M_curr
            best_z = z.copy()
        if verbose and t % 20 == 0:
            print(f"  GA it {t}: M={M_curr:.6f} (best {best_M:.6f})")
    return lam_of(best_z), best_M


# ----------------------- Driver ---------------------------------------

def support_summary(lam, deltas, thresh=1e-4):
    idx = np.where(lam > thresh)[0]
    atoms = sorted([(float(deltas[i]), float(lam[i])) for i in idx],
                   key=lambda t: -t[1])
    return atoms


def optimize_nu_grid(N, n_iters=200, methods=None, seed=0, verbose=True):
    """Master driver.

    Returns dict with best nu, M_cert, and per-method histories.
    """
    if methods is None:
        methods = ["random", "fw", "cd", "ga"]
    print(f"=== Pre-computing {N} atoms on delta grid (DELTA = {DELTA}) ===")
    t0 = time.time()
    deltas, C, Kh_qp, Kh_1 = precompute_atoms(N, verbose=verbose, cache_dir=REPO)
    print(f"  done in {time.time() - t0:.2f}s; "
          f"C shape={C.shape}, Kh_qp shape={Kh_qp.shape}")

    results = {}
    # --- Baseline: pure DELTA ---
    lam_pure = np.zeros(N)
    lam_pure[-1] = 1.0
    M_pure, _, _, _ = eval_nu(lam_pure, C, Kh_qp, Kh_1)
    print(f"\nBaseline pure-DELTA (atom {N-1}, delta={deltas[-1]:.5f}): "
          f"M={M_pure:.6f}")
    results["pure_DELTA"] = {"M_cert": float(M_pure), "atoms": [(float(deltas[-1]), 1.0)]}

    # --- K26 2-atom seed ---
    # delta_1 = 0.138 = deltas[-1]; delta_2 = 0.055
    j_seed = int(np.argmin(np.abs(deltas - 0.055)))
    lam_k26 = np.zeros(N)
    lam_k26[-1] = 0.9312
    lam_k26[j_seed] = 1 - 0.9312
    M_k26, _, _, _ = eval_nu(lam_k26, C, Kh_qp, Kh_1)
    print(f"K26 seed (delta_2 = {deltas[j_seed]:.5f}, lambda_1=0.9312): "
          f"M={M_k26:.6f}")
    results["k26_seed"] = {"M_cert": float(M_k26),
                            "atoms": support_summary(lam_k26, deltas)}

    overall_best_M = M_k26
    overall_best_lam = lam_k26.copy()
    overall_best_method = "k26_seed"

    # --- Method 1: random Dirichlet multi-start ---
    if "random" in methods:
        print(f"\n--- Random Dirichlet multi-start (n_trials={n_iters * 50}) ---")
        t0 = time.time()
        lam_r, M_r, _ = random_dirichlet_search(
            C, Kh_qp, Kh_1, n_iters * 50,
            alpha_list=[0.05, 0.1, 0.3, 1.0, 3.0],
            sparse_support_sizes=[2, 3, 4, 5, 8],
            seed=seed)
        print(f"  best M = {M_r:.6f} in {time.time() - t0:.1f}s")
        results["random"] = {"M_cert": float(M_r),
                              "atoms": support_summary(lam_r, deltas)}
        if M_r > overall_best_M:
            overall_best_M, overall_best_lam = M_r, lam_r.copy()
            overall_best_method = "random"

    # --- Method 2: Frank-Wolfe from K26 seed ---
    if "fw" in methods:
        print(f"\n--- Frank-Wolfe from K26 seed (n_iters={n_iters}) ---")
        t0 = time.time()
        lam_fw, M_fw, hist_fw = frank_wolfe(
            overall_best_lam.copy(), C, Kh_qp, Kh_1, n_iters=n_iters,
            verbose=verbose)
        print(f"  best M = {M_fw:.6f} in {time.time() - t0:.1f}s, "
              f"history len={len(hist_fw)}")
        results["fw"] = {"M_cert": float(M_fw),
                          "atoms": support_summary(lam_fw, deltas),
                          "n_steps": len(hist_fw)}
        if M_fw > overall_best_M:
            overall_best_M, overall_best_lam = M_fw, lam_fw.copy()
            overall_best_method = "fw"

    # --- Method 3: pairwise coordinate descent ---
    if "cd" in methods:
        print("\n--- Pairwise coordinate descent (refine current best) ---")
        t0 = time.time()
        lam_cd, M_cd = pairwise_cd(overall_best_lam.copy(), C, Kh_qp,
                                   Kh_1, n_sweeps=8, verbose=verbose)
        print(f"  best M = {M_cd:.6f} in {time.time() - t0:.1f}s, "
              f"support={len(support_summary(lam_cd, deltas))}")
        results["cd"] = {"M_cert": float(M_cd),
                          "atoms": support_summary(lam_cd, deltas)}
        if M_cd > overall_best_M:
            overall_best_M, overall_best_lam = M_cd, lam_cd.copy()
            overall_best_method = "cd"

    # --- Method 4: softmax gradient ascent ---
    if "ga" in methods:
        print(f"\n--- Softmax gradient ascent (n_iters={n_iters // 2}) ---")
        t0 = time.time()
        # Reseed with uniform to escape current basin
        lam_ga0 = overall_best_lam.copy() + 1e-3
        lam_ga0 /= lam_ga0.sum()
        lam_ga, M_ga = softmax_grad_ascent(
            lam_ga0, C, Kh_qp, Kh_1, n_iters=n_iters // 2,
            lr=0.05, verbose=verbose)
        print(f"  best M = {M_ga:.6f} in {time.time() - t0:.1f}s")
        results["ga"] = {"M_cert": float(M_ga),
                          "atoms": support_summary(lam_ga, deltas)}
        if M_ga > overall_best_M:
            overall_best_M, overall_best_lam = M_ga, lam_ga.copy()
            overall_best_method = "ga"

    # --- Final polish: pairwise CD on overall_best ---
    print("\n--- Final polish: pairwise CD on overall best ---")
    lam_polish, M_polish = pairwise_cd(overall_best_lam.copy(), C, Kh_qp,
                                       Kh_1, n_sweeps=10, verbose=verbose)
    if M_polish > overall_best_M:
        overall_best_M, overall_best_lam = M_polish, lam_polish.copy()
        overall_best_method = "polish_cd"

    final_atoms = support_summary(overall_best_lam, deltas, thresh=1e-5)
    print(f"\n=== OVERALL BEST: M_cert = {overall_best_M:.6f} "
          f"(method = {overall_best_method}) ===")
    print(f"Support size: {len(final_atoms)}")
    for d, lam in final_atoms:
        print(f"  delta = {d:.5f},   lambda = {lam:.6f}")

    return {
        "N_grid": N,
        "DELTA": DELTA,
        "deltas": [float(d) for d in deltas],
        "baseline_pure_DELTA_M": float(M_pure),
        "k26_seed_M": float(M_k26),
        "overall_best_M_cert": float(overall_best_M),
        "overall_best_method": overall_best_method,
        "overall_best_atoms": final_atoms,
        "support_size": len(final_atoms),
        "per_method_results": results,
        "improvement_over_k26": float(overall_best_M - M_k26),
        "improvement_over_pure_arcsine": float(overall_best_M - M_pure),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=50, help="atom grid size")
    p.add_argument("--n_iters", type=int, default=200)
    p.add_argument("--methods", default="random,fw,cd,ga")
    p.add_argument("--out", default="_master_k26_continuous.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = optimize_nu_grid(args.N, n_iters=args.n_iters, methods=methods,
                           seed=args.seed, verbose=not args.quiet)
    outpath = os.path.join(REPO, args.out)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()

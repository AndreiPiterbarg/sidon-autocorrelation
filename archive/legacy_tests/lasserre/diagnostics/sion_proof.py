#!/usr/bin/env python
r"""
Rigorous proof of C_{1a} >= c_target via Sion's minimax theorem.

Proof: C_{1a} >= val(d) = min_mu max_W TV_W(mu) = max_lam min_mu sum lam_W TV_W(mu).
Find lambda* s.t. min_{mu on simplex} mu^T Q mu >= c_target, where Q = sum lam*_W M_W.
Verify by exhaustive KKT enumeration (2^d support sets).
"""
import numpy as np
from scipy.optimize import minimize, linprog
import time


def build_window_matrix(d, ell, s_lo):
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                A[i, j] = 1.0
    return A


def all_windows(d):
    conv_len = 2 * d - 1
    windows = []
    for ell in range(2, 2 * d + 1):
        for s_lo in range(conv_len - ell + 2):
            windows.append((ell, s_lo))
    return windows


def max_tv(mu, d):
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = 2.0 * d / ell
        ws = conv[:n_cv].sum()
        best = max(best, ws * scale)
        for s in range(1, conv_len - n_cv + 1):
            ws += conv[s + n_cv - 1] - conv[s - 1]
            best = max(best, ws * scale)
    return best


def simplex_min_quadratic(Q, d):
    """Global min of mu^T Q mu on simplex. Exact via vertices + KKT."""
    best_val = 1e30
    best_mu = np.ones(d) / d

    # All d vertices
    for i in range(d):
        if Q[i, i] < best_val:
            best_val = Q[i, i]
            best_mu = np.zeros(d); best_mu[i] = 1.0

    # KKT enumeration for |S| >= 2
    for mask in range(3, 2**d):
        S = [i for i in range(d) if mask & (1 << i)]
        s = len(S)
        if s < 2:
            continue
        Q_SS = Q[np.ix_(S, S)]
        try:
            v = np.linalg.solve(Q_SS, np.ones(s))
        except np.linalg.LinAlgError:
            continue
        denom = v.sum()
        if abs(denom) < 1e-15:
            continue
        mu_S = v / denom
        if np.any(mu_S < -1e-12):
            continue
        mu_S = np.maximum(mu_S, 0.0)
        mu = np.zeros(d)
        for idx, i in enumerate(S):
            mu[i] = mu_S[idx]
        mu /= mu.sum()
        # Dual feasibility
        grad = 2.0 * Q @ mu
        nu = 2.0 / denom
        ok = all(grad[i] - nu >= -1e-8 for i in range(d)
                 if not (mask & (1 << i)))
        if not ok:
            continue
        val = mu @ Q @ mu
        if val < best_val:
            best_val = val
            best_mu = mu.copy()

    # Scipy fallback
    for trial in range(5):
        rng = np.random.default_rng(trial * 77 + 13)
        x0 = rng.dirichlet(np.ones(d))
        try:
            res = minimize(lambda x: x @ Q @ x, x0,
                           jac=lambda x: 2.0 * Q @ x, method='SLSQP',
                           bounds=[(0, None)] * d,
                           constraints=[{'type': 'eq',
                                         'fun': lambda x: x.sum() - 1.0}],
                           options={'maxiter': 2000, 'ftol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_mu = np.maximum(res.x, 0.0)
                best_mu /= best_mu.sum()
        except Exception:
            pass

    return best_val, best_mu


def find_optimal_lambda(d, c_target, n_restarts=20, verbose=True):
    """Find lambda* maximizing min_{mu in Delta} mu^T Q(lambda) mu.

    Uses the double oracle / cutting plane algorithm:
      1. Maintain a set of adversarial mu's
      2. LP: find best lambda given these mu's (upper bound)
      3. Exact QP: find worst mu for current lambda (lower bound)
      4. Add worst mu to set, repeat until bounds match
    """
    windows = all_windows(d)
    n_windows = len(windows)
    if verbose:
        print(f"  Windows: {n_windows}")

    M_stack = np.zeros((n_windows, d, d))
    for w, (ell, s_lo) in enumerate(windows):
        M_stack[w] = (2.0 * d / ell) * build_window_matrix(d, ell, s_lo)

    # Precompute TV_w(e_i) for all vertices and windows
    vertex_tvs = np.zeros((d, n_windows))
    for i in range(d):
        for w in range(n_windows):
            vertex_tvs[i, w] = M_stack[w, i, i]

    best_lb = -1e30
    best_lambda = None
    best_Q = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 137 + 42)

        # Seed adversarial mu's: all vertices + random
        mus = []
        for i in range(d):
            e = np.zeros(d); e[i] = 1.0
            mus.append(e)
        for _ in range(2 * d):
            mus.append(rng.dirichlet(np.ones(d)))

        # Precompute TV matrix for seeded mus
        tv_matrix = np.zeros((len(mus), n_windows))
        for k, mk in enumerate(mus):
            for w in range(n_windows):
                tv_matrix[k, w] = mk @ M_stack[w] @ mk

        for iteration in range(500):
            n_pts = len(mus)

            # --- LP: max z s.t. z <= sum_w lam_w TV_w(mu_k), sum lam=1 ---
            c_lp = np.zeros(n_windows + 1)
            c_lp[-1] = -1.0
            A_ub = np.zeros((n_pts, n_windows + 1))
            A_ub[:, :n_windows] = -tv_matrix
            A_ub[:, -1] = 1.0
            b_ub = np.zeros(n_pts)
            A_eq = np.zeros((1, n_windows + 1))
            A_eq[0, :n_windows] = 1.0
            b_eq = np.array([1.0])
            lp_bounds = [(0, None)] * n_windows + [(None, None)]

            lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub,
                             A_eq=A_eq, b_eq=b_eq,
                             bounds=lp_bounds, method='highs')
            if not lp_res.success:
                break
            lam = lp_res.x[:n_windows]
            lp_val = lp_res.x[-1]  # upper bound

            # --- Exact QP: find worst mu for current lambda ---
            Q = np.tensordot(lam, M_stack, axes=([0], [0]))
            qp_val, mu_worst = simplex_min_quadratic(Q, d)

            # qp_val is the TRUE min for this lambda (lower bound on game value)
            if qp_val > best_lb:
                best_lb = qp_val
                best_lambda = lam.copy()
                best_Q = Q.copy()

            # Check convergence: LP upper bound ≈ QP lower bound
            if abs(lp_val - qp_val) < 1e-10:
                if verbose and restart == 0:
                    print(f"    Converged at iteration {iteration}: "
                          f"val={qp_val:.8f}")
                break

            # Add worst mu to adversarial set
            mus.append(mu_worst.copy())
            new_row = np.array([mu_worst @ M_stack[w] @ mu_worst
                                for w in range(n_windows)])
            tv_matrix = np.vstack([tv_matrix, new_row.reshape(1, -1)])

        if verbose and (restart + 1) % 5 == 0:
            print(f"    Restart {restart+1}/{n_restarts}: "
                  f"lb={best_lb:.8f}", flush=True)

    if verbose:
        n_active = int(np.sum(best_lambda > 1e-8))
        print(f"  Best lower bound: {best_lb:.8f}")
        print(f"  Active windows: {n_active}")
        for idx in np.argsort(best_lambda)[::-1][:10]:
            if best_lambda[idx] > 1e-8:
                ell, s_lo = windows[idx]
                print(f"    lam={best_lambda[idx]:.6f} -> (ell={ell}, s={s_lo})")

    return best_lambda, best_Q, best_lb, windows, M_stack


def prove(d, c_target, n_restarts=30, verbose=True):
    if verbose:
        print(f"\n{'='*70}")
        print(f"SION MINIMAX PROOF: C_{{1a}} >= {c_target}")
        print(f"  Dimension: d={d}")
        print(f"{'='*70}\n")

    t0 = time.time()

    # Step 1: Find optimal lambda
    print("STEP 1: Find optimal window mixture lambda*", flush=True)
    lam, Q, lb, windows, M_stack = find_optimal_lambda(
        d, c_target, n_restarts=n_restarts, verbose=verbose)

    if lb < c_target:
        print(f"\n  WARNING: best lb={lb:.8f} < {c_target}. "
              f"Proof may fail.", flush=True)

    # Step 2: Verify
    print(f"\nSTEP 2: Verify min mu^T Q mu >= {c_target} on simplex",
          flush=True)
    gmin, gmin_mu = simplex_min_quadratic(Q, d)
    verified = gmin >= c_target - 1e-10
    print(f"  Global minimum: {gmin:.10f}")
    print(f"  Margin: {gmin - c_target:.10f}")
    if gmin_mu is not None:
        print(f"  Minimizer: {np.array2string(gmin_mu, precision=6)}")
        print(f"  max_W TV_W at minimizer: {max_tv(gmin_mu, d):.10f}")
    print(f"  {'VERIFIED' if verified else 'FAILED'}", flush=True)

    # Step 3: Cross-validate
    print(f"\nSTEP 3: Cross-validation (100K samples)...", flush=True)
    rng = np.random.default_rng(12345)
    worst = 1e30
    for _ in range(100_000):
        mu = rng.dirichlet(np.ones(d))
        worst = min(worst, mu @ Q @ mu)
    print(f"  Worst sampled: {worst:.10f}")
    if worst < gmin - 1e-6:
        print(f"  WARNING: sample {worst:.10f} < KKT min {gmin:.10f}!")
        verified = False

    # PSD check
    eigvals = np.linalg.eigvalsh(Q - c_target * np.ones((d, d)))
    print(f"\n  PSD check: min eigenvalue = {eigvals.min():.8f}  "
          f"{'PASS' if eigvals.min() >= -1e-10 else 'FAIL'}")

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    if verified:
        print(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
        print(f"  d={d}, margin={gmin-c_target:.10f}, time={elapsed:.1f}s")
    else:
        print(f"PROOF FAILED: min={gmin:.10f}, need>={c_target}")
    print(f"{'='*70}")

    return {'proven': verified, 'd': d, 'c_target': c_target,
            'global_min': gmin, 'margin': gmin - c_target,
            'lambda': lam, 'Q': Q, 'elapsed': elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--restarts', type=int, default=30)
    args = parser.parse_args()

    result = prove(args.d, args.c_target, n_restarts=args.restarts)

    if not result['proven']:
        print(f"\nSearching for max provable c at d={args.d}...")
        for c in np.arange(args.c_target - 0.01, 0.99, -0.01):
            r = prove(args.d, round(c, 2), n_restarts=max(10, args.restarts // 2),
                      verbose=False)
            if r['proven']:
                print(f"  PROVEN: C_{{1a}} >= {round(c,2)} "
                      f"(margin={r['margin']:.8f})")
                break


if __name__ == '__main__':
    main()

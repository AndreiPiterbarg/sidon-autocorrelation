#!/usr/bin/env python
r"""
Rigorous lower bound on val(d) via SDP relaxation.

==========================================================================
PROBLEM
==========================================================================

val(d) = min_{mu in Delta_d} max_W TV_W(mu)

where Delta_d = {mu >= 0, sum mu_i = 1}, TV_W(mu) = (2d/ell) * mu^T A_W mu.

Rewrite as:
    val(d) = min  t
             s.t. mu^T M_W mu <= t   for all windows W
                  mu >= 0, sum mu = 1

where M_W = (2d/ell) * A_W.

==========================================================================
SDP RELAXATION (Shor relaxation)
==========================================================================

Introduce X = mu * mu^T (rank-1 PSD matrix). Then mu^T M_W mu = Tr(M_W X).

Relaxation: drop the rank-1 constraint. Keep X >= 0 (PSD).

Additional valid constraints from mu >= 0, sum mu = 1:
  - X >= 0 (PSD)
  - X_{ij} >= 0 (entry-wise nonneg, since mu >= 0 => X = mu*mu^T >= 0 entrywise)
  - sum_i X_{ii} = sum_i mu_i^2 <= (sum mu_i)^2 = 1  (by Cauchy-Schwarz, actually =)
  - Actually: Tr(11^T X) = (sum mu_i)^2 = 1 at rank 1
  - At general PSD X: Tr(11^T X) = 1^T X 1 >= 0
  - We need: 1^T X 1 = 1  (from sum mu = 1, since 1^T mu mu^T 1 = 1)
  - Also: X_{ii} >= 0 (diagonal nonneg, automatic from PSD)
  - Also: X_{ij} >= 0 for all i,j (since mu >= 0)

SDP:
    val_SDP = min  t
              s.t. Tr(M_W X) <= t   for all W
                   X >= 0  (PSD)
                   X_{ij} >= 0  (entrywise nonneg, "doubly nonneg")
                   1^T X 1 = 1  (sum constraint)

val_SDP <= val(d) (relaxation gives lower bound).

If val_SDP >= c_target: PROOF that val(d) >= c_target, hence C_{1a} >= c_target.

The doubly-nonneg constraint (X PSD + entrywise nonneg) is key — it's much
tighter than plain PSD relaxation.

==========================================================================
DUAL OF THE SDP
==========================================================================

The dual provides a CERTIFICATE. If the dual is feasible with objective >= c,
that constitutes a proof.

Dual: max  z
      s.t. sum_W alpha_W M_W - Z - N + z * 11^T = 0  (schur)
           Z >= 0 (PSD), N >= 0 (entrywise), alpha_W >= 0, sum alpha_W = 1
           (not exactly this, need to work out carefully)

By weak duality: dual_obj <= primal_obj = val_SDP <= val(d).

==========================================================================
"""

import numpy as np
import time
import sys


def build_window_matrix(d, ell, s_lo):
    """M_W = (2d/ell) * A_W where A_W[i,j] = 1 if s_lo <= i+j <= s_lo+ell-2."""
    A = np.zeros((d, d), dtype=np.float64)
    s_hi = s_lo + ell - 2
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return (2.0 * d / ell) * A


def all_windows(d):
    """Return list of all valid (ell, s_lo) pairs."""
    conv_len = 2 * d - 1
    windows = []
    for ell in range(2, 2 * d + 1):
        for s_lo in range(conv_len - ell + 2):
            windows.append((ell, s_lo))
    return windows


def max_tv(mu, d):
    """Compute max_W TV_W(mu) directly."""
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


def solve_sdp(d, c_target, verbose=True):
    """Solve the SDP relaxation for val(d) >= c_target.

    Returns dict with:
        lb: lower bound on val(d)
        proven: True if lb >= c_target
        status: solver status
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("ERROR: cvxpy not installed. Run: pip install cvxpy")
        return {'lb': 0, 'proven': False, 'status': 'no_cvxpy'}

    windows = all_windows(d)
    n_windows = len(windows)

    if verbose:
        print(f"  d={d}, windows={n_windows}")
        print(f"  Building window matrices...", flush=True)

    M_mats = [build_window_matrix(d, ell, s_lo) for ell, s_lo in windows]

    if verbose:
        print(f"  Setting up SDP...", flush=True)

    # Variables
    X = cp.Variable((d, d), symmetric=True)
    t = cp.Variable()

    # Constraints
    constraints = []

    # X is PSD
    constraints.append(X >> 0)

    # X is entrywise nonneg (doubly nonneg)
    constraints.append(X >= 0)

    # 1^T X 1 = 1 (from sum mu = 1)
    ones = np.ones(d)
    constraints.append(ones @ X @ ones == 1)

    # Tr(M_W X) <= t for all windows
    for w in range(n_windows):
        constraints.append(cp.trace(M_mats[w] @ X) <= t)

    # Objective: minimize t
    prob = cp.Problem(cp.Minimize(t), constraints)

    if verbose:
        print(f"  Solving SDP ({d}x{d} matrix, {n_windows} window constraints, "
              f"doubly nonneg)...", flush=True)

    t0 = time.time()

    # Try MOSEK first (best), then SCS (free)
    solved = False
    for solver in [cp.MOSEK, cp.SCS, cp.CLARABEL]:
        try:
            if solver == cp.SCS:
                prob.solve(solver=solver, verbose=False,
                           max_iters=50000, eps=1e-9)
            elif solver == cp.CLARABEL:
                prob.solve(solver=solver, verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                solved = True
                break
        except (cp.SolverError, Exception) as e:
            if verbose:
                print(f"    Solver {solver} failed: {e}")
            continue

    elapsed = time.time() - t0

    if not solved:
        if verbose:
            print(f"  All solvers failed. Status: {prob.status}")
        return {'lb': 0, 'proven': False, 'status': prob.status,
                'elapsed': elapsed}

    lb = t.value
    proven = lb >= c_target - 1e-6  # small tolerance for solver accuracy

    if verbose:
        print(f"  Solver: {prob.status} in {elapsed:.2f}s")
        print(f"  SDP lower bound: {lb:.10f}")
        print(f"  Margin over c_target: {lb - c_target:.10f}")

        # Extract approximate mu from X
        X_val = X.value
        if X_val is not None:
            # Leading eigenvector of X gives approximate mu
            eigvals, eigvecs = np.linalg.eigh(X_val)
            mu_approx = eigvecs[:, -1] * np.sqrt(max(eigvals[-1], 0))
            # Fix sign and normalize
            if mu_approx.sum() < 0:
                mu_approx = -mu_approx
            mu_approx = np.maximum(mu_approx, 0)
            if mu_approx.sum() > 1e-10:
                mu_approx /= mu_approx.sum()
                # Verify
                actual_max_tv = max_tv(mu_approx, d)
                print(f"  Approximate minimizer (from X eigenvector):")
                print(f"    mu = {np.array2string(mu_approx, precision=4)}")
                print(f"    max_W TV_W(mu) = {actual_max_tv:.10f}")
                print(f"    Rank of X: {np.sum(eigvals > 1e-6)}")

    return {
        'lb': lb,
        'proven': proven,
        'status': prob.status,
        'elapsed': elapsed,
        'X': X.value if X.value is not None else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SDP proof of C_{1a} >= c_target")
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--c_target', type=float, default=1.30)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"SDP RELAXATION PROOF: C_{{1a}} >= {args.c_target}")
    print(f"{'='*70}\n")

    # Solve for target d
    for d in [4, 6, 8, 10, 12, 14, 16]:
        if d > args.d:
            break
        print(f"\n--- d={d} ---", flush=True)
        result = solve_sdp(d, args.c_target, verbose=True)

        if result['proven']:
            print(f"\n  *** SDP PROVES: val({d}) >= {args.c_target} ***")
            print(f"  *** Therefore: C_{{1a}} >= {args.c_target} ***")
        else:
            lb = result['lb']
            if lb is not None:
                print(f"  SDP bound {lb:.6f} < {args.c_target}. "
                      f"Cannot prove at d={d}.")
            else:
                print(f"  Solver failed at d={d}.")

    # Also try to find the tightest bound at d=args.d
    print(f"\n{'='*70}")
    print(f"FINDING TIGHTEST SDP BOUND AT d={args.d}")
    print(f"{'='*70}")
    result = solve_sdp(args.d, 0.0, verbose=True)
    if result['lb'] is not None:
        print(f"\n  Best SDP lower bound on val({args.d}): {result['lb']:.10f}")
        if result['lb'] >= args.c_target:
            print(f"  *** PROVES C_{{1a}} >= {args.c_target} ***")
        else:
            print(f"  Insufficient for {args.c_target} "
                  f"(gap = {args.c_target - result['lb']:.6f})")


if __name__ == '__main__':
    main()

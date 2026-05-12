"""Bochner phase attack: upper-bound ||f||_2^2 using g = f*f >= 0 PHASE info.

This is Path A §5.3.c, identified by all 4 deployed agents (2026-05) as the
single untried direction that escapes the chain (†) blocker for asymmetric f.

Setup
-----
Let f >= 0, supp f c [-1/4, 1/4], int f = 1. Periodise at period 1; let
    f_hat(j) = z_j e^{i theta_j},  z_j >= 0,  theta_j in R.
Then for g = f * f, real-valued and >= 0,
    g_hat(j) = (f_hat(j))^2 = z_j^2 (cos 2 theta_j + i sin 2 theta_j).

Define R_j = Re g_hat(j) = z_j^2 cos 2 theta_j,  I_j = Im g_hat(j).
Then |g_hat(j)| = z_j^2.

Variables (j = 1..N):  R_j, I_j, z_j^2    [+ implied  R_0 = z_0^2 = 1, I_0 = 0]

Constraints
-----------
1) SOC:  z_j^2 >= sqrt(R_j^2 + I_j^2)              ("|g_hat(j)| = z_j^2")
2) Lemma 1 (Multi-moment derivation):  z_j^2 <= mu(M),  mu(M) = M sin(pi/M)/pi.
3) Peak constraint:  g(0) = 1 + 2 sum_{j>=1} R_j = M  (peak normalization;
   we WLOG translate so the peak of g is at t=0).
4) Bochner of f >= 0:  Toeplitz [f_hat(i-j)] is HERMITIAN-PSD.
   Note: this constrains f_hat directly, NOT g_hat. To use, we'd lift to
   z_j, theta_j; nonconvex unless we use a relaxation. We use a WEAKER
   Bochner constraint on g instead.
5) Bochner of g >= 0:  Toeplitz [g_hat(i-j)] = [R_{i-j} + i I_{i-j}] is
   HERMITIAN-PSD (for i, j = 0..N).  This is the (Hermitian) Toeplitz-PSD
   constraint on (R_j, I_j).
6) g <= M pointwise:  g(t) <= M for all t in [-1/2, 1/2]. Equivalently:
   M - g >= 0, hence (M)*delta - g has all-PSD Toeplitz.
   In Fourier: Toeplitz [M * delta_{ij} - g_hat(i-j)] is HERMITIAN-PSD.

Objective: maximize  K = ||f||_2^2 = sum_{all j} z_j^2 = 1 + 2 sum_{j>=1} z_j^2.

Truncation: we use a finite N. The truncation drops z_j^2 for |j| > N, so
the SDP optimum is an UPPER BOUND on the truncated K. To upper-bound the
TRUE K = sum_{all j} z_j^2, we add a tail bound:

    K  <=  1 + 2 sum_{j=1..N} z_j^2  +  2 * tail(N, M)

where tail(N, M) is an explicit upper bound on sum_{j>N} z_j^2. Cheap
candidates:
  (a) tail(N, M) <= sum_{j>N} mu(M) = INFINITY — trivially useless.
  (b) tail(N, M) bounded by the "uncovered Plancherel mass" via separate
      argument.  Future work; for now we report the truncated version
      with the caveat.

For the BREAKEVEN check at M=1.276 (need K < 2.08):
  Truncated K = 1 + 2 sum_{j=1..N} z_j^2.
  If even WITHOUT tail, the truncated K exceeds 2.08, sharper-Markov is dead.
  If truncated K << 2.08, we have margin to absorb any tail bound.

For now we report truncated K; the tail handling is a separate module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BochnerPhaseResult:
    M: float
    N: int
    K_truncated: float
    z2_values: np.ndarray             # z_j^2 for j = 1..N
    R_values: np.ndarray              # R_j for j = 1..N
    I_values: np.ndarray              # I_j for j = 1..N
    breakeven: float
    path_alive: bool
    solver_status: str
    wall_s: float


def mu_M(M: float) -> float:
    return M * math.sin(math.pi / M) / math.pi


def breakeven_K(M: float) -> float:
    return 1.0 + (M - 1.0) / mu_M(M)


def solve_bochner_phase_sdp(M: float, N: int,
                            solver: str = "CLARABEL",
                            verbose: bool = False) -> BochnerPhaseResult:
    """Compute upper bound on truncated K = 1 + 2 sum_{j=1..N} z_j^2.

    Uses cvxpy with complex Hermitian Toeplitz PSD encoding.

    Returns truncated K (without tail). For full ||f||_2^2 bound, add tail.
    """
    import time
    import cvxpy as cp

    t0 = time.time()
    mu = mu_M(M)

    # Variables: R_j, I_j (real), z_j^2 (real, nonneg) for j = 1..N.
    R = cp.Variable(N)            # R_j for j=1..N
    I_ = cp.Variable(N)
    z2 = cp.Variable(N, nonneg=True)

    # Build the Toeplitz Hermitian matrix T of size (N+1) x (N+1) for g_hat.
    # T[i, j] = g_hat(i - j), with g_hat(0) = 1, g_hat(j) = R_{|j|} + i sign(j) I_{|j|}
    # for j != 0.
    # We encode complex Hermitian PSD using cvxpy's complex SDP.
    T_real = cp.Variable((N + 1, N + 1), symmetric=True)
    T_imag = cp.Variable((N + 1, N + 1))
    # Anti-symmetric for imaginary part: T_imag[i,j] = -T_imag[j,i].
    cons = [T_imag + T_imag.T == 0]    # T_imag is anti-symmetric

    # Toeplitz structure: T[i,j] depends only on i - j.
    # diagonal (i - j = 0): T_real = 1, T_imag = 0
    for i in range(N + 1):
        cons.append(T_real[i, i] == 1)
        # T_imag[i, i] = 0 follows from anti-symmetry

    # off-diagonals (i - j = k, k = 1..N): T_real[i, i-k] = R_k, T_imag[i, i-k] = -I_k
    # (since g_hat(-k) = conj(g_hat(k)) = R_k - i I_k, so T[i, i-k] = g_hat(-(i-(i-k))) = g_hat(-k))
    # Wait: T[i, j] = g_hat(i - j). For i > j: T[i, j] = g_hat(i - j) = R_{i-j} + i I_{i-j}.
    # For i < j: T[i, j] = g_hat(i - j) = g_hat(-(j-i)) = conj(g_hat(j-i)) = R_{j-i} - i I_{j-i}.
    for i in range(N + 1):
        for j in range(N + 1):
            k = abs(i - j)
            if k == 0:
                continue   # already handled
            if i > j:
                cons.append(T_real[i, j] == R[k - 1])
                cons.append(T_imag[i, j] == I_[k - 1])
            else:  # i < j
                cons.append(T_real[i, j] == R[k - 1])
                cons.append(T_imag[i, j] == -I_[k - 1])

    # Hermitian PSD: the (2N+2) x (2N+2) real matrix
    # [[T_real, -T_imag], [T_imag, T_real]] must be PSD.
    big = cp.bmat([[T_real, -T_imag], [T_imag, T_real]])
    cons.append(big >> 0)

    # Bochner of g <= M (i.e. M - g >= 0): Toeplitz of M*delta - g_hat is PSD.
    # M * delta(0) - g_hat(0) = M - 1.   For k != 0:  -g_hat(k) = -R_k - i I_k.
    # Toeplitz S: S[i, j] = (M)*delta_{i,j} - g_hat(i-j)
    S_real = cp.Variable((N + 1, N + 1), symmetric=True)
    S_imag = cp.Variable((N + 1, N + 1))
    cons.append(S_imag + S_imag.T == 0)
    for i in range(N + 1):
        cons.append(S_real[i, i] == M - 1)
    for i in range(N + 1):
        for j in range(N + 1):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(S_real[i, j] == -R[k - 1])
                cons.append(S_imag[i, j] == -I_[k - 1])
            else:
                cons.append(S_real[i, j] == -R[k - 1])
                cons.append(S_imag[i, j] == I_[k - 1])
    big_S = cp.bmat([[S_real, -S_imag], [S_imag, S_real]])
    cons.append(big_S >> 0)

    # SOC: z_j^2 >= sqrt(R_j^2 + I_j^2)  for each j
    for j in range(N):
        cons.append(cp.norm(cp.hstack([R[j], I_[j]]), 2) <= z2[j])

    # Lemma 1 cap: z_j^2 <= mu(M)
    cons.append(z2 <= mu)

    # Peak normalization: 1 + 2 * sum R = M
    cons.append(1 + 2 * cp.sum(R) == M)

    # Objective: max sum z_j^2 (truncated)
    obj = cp.Maximize(cp.sum(z2))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)

    bk = breakeven_K(M)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return BochnerPhaseResult(
            M=M, N=N,
            K_truncated=float("inf"),
            z2_values=np.array([]),
            R_values=np.array([]),
            I_values=np.array([]),
            breakeven=bk, path_alive=False,
            solver_status=prob.status,
            wall_s=time.time() - t0,
        )
    sum_z2 = float(prob.value)
    K_trunc = 1.0 + 2.0 * sum_z2
    return BochnerPhaseResult(
        M=M, N=N,
        K_truncated=K_trunc,
        z2_values=np.asarray(z2.value).flatten(),
        R_values=np.asarray(R.value).flatten(),
        I_values=np.asarray(I_.value).flatten(),
        breakeven=bk,
        path_alive=(K_trunc < bk),
        solver_status=prob.status,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.275, 1.276, 1.280, 1.290, 1.300, 1.378])
    p.add_argument("--N", type=int, nargs="*", default=[8, 16, 24, 32])
    p.add_argument("--solver", type=str, default="CLARABEL")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("=== Bochner phase attack: upper bound on ||f||_2^2 (truncated) ===")
    print()
    print(f"{'N':>3} {'M':>7} {'breakeven':>11} {'K_trunc':>11} "
          f"{'max z_j^2':>11} {'verdict':>14} {'wall':>7}")
    print("-" * 75)
    for N in args.N:
        for M in args.M:
            r = solve_bochner_phase_sdp(M, N, solver=args.solver,
                                        verbose=args.verbose)
            v = "PATH ALIVE" if r.path_alive else "path dead"
            max_z2 = float(np.max(r.z2_values)) if len(r.z2_values) else float("nan")
            print(f"{N:>3d} {M:>7.4f} {r.breakeven:>11.4f} "
                  f"{r.K_truncated:>11.4f} {max_z2:>11.4f} "
                  f"{v:>14} {r.wall_s:>7.1f}")

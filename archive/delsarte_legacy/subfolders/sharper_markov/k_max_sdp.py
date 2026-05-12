"""K_max(M) SDP: rigorous upper bound on ||f||_2^2 = sum z_n^2 subject to
||f*f||_inf <= M, for asymmetric f with supp [-1/4, 1/4], int f = 1.

This implements Approach 3.4 from the synthesis of 4 agents (2026-05).
The empirical conjecture (consistent with all known evidence; not yet proven
analytically): K_max(M) <= M for M < 2. If proved by this SDP,
unblocks Attack 1's chain (Section 5.3 of path_a_unconditional_holder/derivation.md)
to give:

  c*(M) = (1 + mu(M)(K_max(M) - 1)) / M
  At M=1.276 with K_max=M: c* = 0.838  =>  C_{1a} >= 1.424 unconditional.

SDP setup (frequency-domain Lasserre level-2 lift)
--------------------------------------------------
Variables (for j = 1..N):
    c_j      = Re f_hat(j)            (free real)
    s_j      = Im f_hat(j)            (free real, with s_0 = 0)
    cc_j     = c_j^2                  (Lasserre lift)
    ss_j     = s_j^2
    cs_j     = c_j s_j
    z2_j     = c_j^2 + s_j^2 = |f_hat(j)|^2  (= cc_j + ss_j)

Derived (linear in lifted variables):
    R_j = c_j^2 - s_j^2 = cc_j - ss_j   = Re g_hat(j) where g = f*f
    I_j = 2 c_j s_j     = 2 * cs_j      = Im g_hat(j)

Constraints:
  (Lasserre L2 per frequency j)  PSD on M_j(c, s) =
      [[1,  c_j,  s_j ],
       [c_j, cc_j, cs_j],
       [s_j, cs_j, ss_j]]
   This forces cc_j >= c_j^2, ss_j >= s_j^2, cc_j*ss_j >= cs_j^2 etc.
   At rank-1 (the actual f_hat) these are equalities.

  (Bochner of f >= 0)   Hermitian Toeplitz [f_hat(i-j)] (size N+1) PSD.
   In real-imaginary blocks: [[T_re, -T_im], [T_im, T_re]] PSD with
   T_re symmetric, T_im antisymmetric, T_re[i,i]=1, T_im[i,i]=0,
   T_re[i,j]=c_{|i-j|}, T_im[i,j]=sign(i-j)*s_{|i-j|}.

  (Bochner of g >= 0)   Hermitian Toeplitz [g_hat(i-j)] PSD.
   T_re_g[i,j] = R_{|i-j|}, T_im_g[i,j] = sign(i-j)*I_{|i-j|}, diag = 1.

  (g <= M pointwise = Bochner of M - g)   Toeplitz of (M*delta - g_hat) PSD.
   T_re[i,i] = M - 1, T_re[i,j]=-R, T_im[i,j]=-sign*I.

  (Lemma 2.14 / Lemma 1)   z2_j <= mu(M) for j = 1..N.

  (Peak of g equal to M, i.e. g(0)=M)   1 + 2 sum_j R_j = M.

  (Normalization)         z_0 = c_0 = 1, s_0 = 0  (encoded as fixed entries).

Objective: max  K_truncated = 1 + 2 sum_{j=1..N} z2_j.

Tail handling: omitted (truncated). For rigorous full bound, add tail.
For empirical purposes (testing if K <= M conjecture), truncated K is the right
quantity to observe.

Note: this is a CONVEX SDP — the Lasserre L2 lift makes everything linear in
the lifted variables. Solver: cvxpy with Clarabel/MOSEK.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KMaxResult:
    M: float
    N: int
    K_truncated: float
    z2_values: np.ndarray
    c_values: np.ndarray
    s_values: np.ndarray
    cc_values: np.ndarray
    ss_values: np.ndarray
    cs_values: np.ndarray
    rank1_residual_max: float    # max gap (cc - c^2, ss - s^2) — rank-1 deviation
    breakeven: float
    sym_breakeven: float          # (1 + (M-1)/mu) — the (†)-chain breakeven
    conjecture_check: float       # K_truncated / M  (conjectured <= 1)
    solver_status: str
    wall_s: float


def mu_M(M: float) -> float:
    return M * math.sin(math.pi / M) / math.pi


def breakeven_K(M: float) -> float:
    return 1.0 + (M - 1.0) / mu_M(M)


def solve_k_max_sdp(M: float, N: int,
                    solver: str = "CLARABEL",
                    verbose: bool = False) -> KMaxResult:
    """SDP for max K_truncated = 1 + 2 sum z2_j s.t. all constraints.

    Variables c_j, s_j, cc_j=c_j^2, ss_j=s_j^2, cs_j=c_j s_j, z2_j=cc+ss
    for j=1..N (j=0 fixed at c_0=1, s_0=0 trivially).
    """
    import cvxpy as cp

    t0 = time.time()
    mu = mu_M(M)

    # ---- variables
    c = cp.Variable(N)            # c_j for j=1..N (real)
    s = cp.Variable(N)            # s_j
    cc = cp.Variable(N, nonneg=True)
    ss = cp.Variable(N, nonneg=True)
    cs = cp.Variable(N)
    z2 = cc + ss                  # z_j^2 = c_j^2 + s_j^2
    R = cc - ss                   # Re g_hat(j)
    I_ = 2 * cs                   # Im g_hat(j)

    cons = []

    # ---- Lasserre L2 PSD per frequency
    # M_j = [[1, c_j, s_j], [c_j, cc_j, cs_j], [s_j, cs_j, ss_j]] PSD
    for j in range(N):
        Mj = cp.bmat([
            [cp.reshape(cp.Constant(1.0), (1, 1)),
             cp.reshape(c[j], (1, 1)), cp.reshape(s[j], (1, 1))],
            [cp.reshape(c[j], (1, 1)),
             cp.reshape(cc[j], (1, 1)), cp.reshape(cs[j], (1, 1))],
            [cp.reshape(s[j], (1, 1)),
             cp.reshape(cs[j], (1, 1)), cp.reshape(ss[j], (1, 1))],
        ])
        cons.append(Mj >> 0)

    # ---- Lemma 2.14: z2_j <= mu(M)
    cons.append(z2 <= mu)

    # ---- Peak constraint: 1 + 2 sum R_j = M  (g(0) = M)
    cons.append(1 + 2 * cp.sum(R) == M)

    # ---- Bochner of f: Toeplitz [f_hat(i-j)] (N+1) Hermitian-PSD.
    # f_hat(0) = 1, f_hat(j) = c_j + i s_j, f_hat(-j) = c_j - i s_j.
    # T_re[i, j] = Re f_hat(i-j), T_im[i, j] = Im f_hat(i-j).
    # T_re symmetric (since c is real even), T_im antisymmetric.
    NN = N + 1
    T_re_f = cp.Variable((NN, NN), symmetric=True)
    T_im_f = cp.Variable((NN, NN))
    cons.append(T_im_f + T_im_f.T == 0)   # antisymmetric
    for i in range(NN):
        cons.append(T_re_f[i, i] == 1)
    for i in range(NN):
        for j in range(NN):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(T_re_f[i, j] == c[k - 1])
                cons.append(T_im_f[i, j] == s[k - 1])
            else:
                cons.append(T_re_f[i, j] == c[k - 1])
                cons.append(T_im_f[i, j] == -s[k - 1])
    # Hermitian PSD via 2N+2 real lift:
    big_f = cp.bmat([[T_re_f, -T_im_f], [T_im_f, T_re_f]])
    cons.append(big_f >> 0)

    # ---- Bochner of g (g >= 0): Toeplitz [g_hat(i-j)] PSD.
    T_re_g = cp.Variable((NN, NN), symmetric=True)
    T_im_g = cp.Variable((NN, NN))
    cons.append(T_im_g + T_im_g.T == 0)
    for i in range(NN):
        cons.append(T_re_g[i, i] == 1)
    for i in range(NN):
        for j in range(NN):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(T_re_g[i, j] == R[k - 1])
                cons.append(T_im_g[i, j] == I_[k - 1])
            else:
                cons.append(T_re_g[i, j] == R[k - 1])
                cons.append(T_im_g[i, j] == -I_[k - 1])
    big_g = cp.bmat([[T_re_g, -T_im_g], [T_im_g, T_re_g]])
    cons.append(big_g >> 0)

    # ---- Bochner of M - g: Toeplitz of (M*delta - g_hat) PSD.
    T_re_S = cp.Variable((NN, NN), symmetric=True)
    T_im_S = cp.Variable((NN, NN))
    cons.append(T_im_S + T_im_S.T == 0)
    for i in range(NN):
        cons.append(T_re_S[i, i] == M - 1)
    for i in range(NN):
        for j in range(NN):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(T_re_S[i, j] == -R[k - 1])
                cons.append(T_im_S[i, j] == -I_[k - 1])
            else:
                cons.append(T_re_S[i, j] == -R[k - 1])
                cons.append(T_im_S[i, j] == I_[k - 1])
    big_S = cp.bmat([[T_re_S, -T_im_S], [T_im_S, T_re_S]])
    cons.append(big_S >> 0)

    # ---- Objective
    obj = cp.Maximize(cp.sum(z2))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)

    bk = breakeven_K(M)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return KMaxResult(
            M=M, N=N,
            K_truncated=float("inf"),
            z2_values=np.array([]),
            c_values=np.array([]), s_values=np.array([]),
            cc_values=np.array([]), ss_values=np.array([]),
            cs_values=np.array([]),
            rank1_residual_max=float("inf"),
            breakeven=bk, sym_breakeven=bk,
            conjecture_check=float("inf"),
            solver_status=prob.status,
            wall_s=time.time() - t0,
        )
    sum_z2 = float(prob.value)
    K_trunc = 1.0 + 2.0 * sum_z2
    c_v = np.asarray(c.value).flatten()
    s_v = np.asarray(s.value).flatten()
    cc_v = np.asarray(cc.value).flatten()
    ss_v = np.asarray(ss.value).flatten()
    cs_v = np.asarray(cs.value).flatten()
    z2_v = cc_v + ss_v
    rank1_resid = max(
        float(np.max(cc_v - c_v ** 2)),
        float(np.max(ss_v - s_v ** 2)),
    )
    return KMaxResult(
        M=M, N=N,
        K_truncated=K_trunc,
        z2_values=z2_v, c_values=c_v, s_values=s_v,
        cc_values=cc_v, ss_values=ss_v, cs_values=cs_v,
        rank1_residual_max=rank1_resid,
        breakeven=bk, sym_breakeven=bk,
        conjecture_check=K_trunc / M,
        solver_status=prob.status,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.276, 1.30, 1.50, 2.0])
    p.add_argument("--N", type=int, nargs="*", default=[8, 12])
    p.add_argument("--solver", type=str, default="CLARABEL")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("=== K_max SDP: bound ||f||_2^2 with f_hat Bochner-PSD ===")
    print()
    print(f"{'N':>3} {'M':>7} {'breakeven':>11} {'K_trunc':>11} "
          f"{'K/M':>7} {'rank1_res':>11} {'wall':>7}")
    print("-" * 75)
    for N in args.N:
        for M in args.M:
            r = solve_k_max_sdp(M, N, solver=args.solver, verbose=args.verbose)
            print(f"{N:>3d} {M:>7.4f} {r.breakeven:>11.4f} "
                  f"{r.K_truncated:>11.4f} {r.conjecture_check:>7.4f} "
                  f"{r.rank1_residual_max:>11.4e} {r.wall_s:>7.1f}")
    print()
    print("Conjecture: K_max(M) <= M for M < 2")
    print("If K/M < 1 always: closes Attack 1 unconditionally => C_1a >= 1.42")

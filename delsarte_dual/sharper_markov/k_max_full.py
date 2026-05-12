"""K_max(M) SDP with FULL admissibility (Hausdorff support + Bochner + Lasserre L2).

Builds on k_max_sdp.py by adding the SUPPORT constraint supp f c [-1/4, 1/4]
via Hausdorff moment PSD (mirror of `moment_constraints.py` C-block in cvxpy).

This closes the loophole where the basic SDP allowed (c, s) = 0 with positive
lifted moments (cc, ss) — the support constraint forces (c, s) and (cc, ss)
to be linked through a real measure on [-1/4, 1/4].

Variables
---------
   m_k       = int x^k f(x) dx   for k = 0, 1, ..., K_max         (m_0 = 1)
   c_j, s_j  = Re/Im f_hat(j)    for j = 1, ..., N
   cc_j, ss_j, cs_j  (Lasserre L2 lifts of c_j^2, s_j^2, c_j s_j)
   z2_j      = cc_j + ss_j      (= |f_hat(j)|^2 in rank-1)
   R_j, I_j  = cc_j - ss_j, 2 cs_j  (= Re/Im g_hat(j))

Constraints
-----------
   (Hausdorff support [-1/4, 1/4]):
     Hankel H1[i, j] = m_{i+j}             PSD (size K_max/2 + 1)
     Hankel H2[i, j] = (1/16) m_{i+j} - m_{i+j+2}   PSD

   (Fourier-to-moment, truncated at degree K_max with error bound):
     c_j = sum_k (real part of (-2 pi i j)^k / k!) * m_k +/- err_j
     s_j = sum_k (imag part of (-2 pi i j)^k / k!) * m_k +/- err_j
     err_j = sum_{k > K_max} |(-2 pi j)^k / k!| * (1/4)^k

   (Lasserre L2 PSD per j):  3x3 PSD on (1, c_j, s_j, cc_j, cs_j, ss_j).

   (Bochner of f >= 0):  (N+1) Hermitian-Toeplitz on (c_j, s_j) PSD.

   (Bochner of g >= 0 and M - g >= 0):  Toeplitz on (R_j, I_j) PSD,
   and shifted version PSD.

   (Lemma 2.14):  z2_j <= mu(M).

   (Peak):  1 + 2 sum R_j = M.

   (Normalization):  m_0 = 1, c_0 = 1, s_0 = 0.

Objective: max sum z2_j  =>  K_truncated = 1 + 2 sum z2_j.

This is large at d>10, but for the SCREENING question (is K_max(M) ≤ M?)
small N=4..6 should suffice if the f_hat Bochner + Hausdorff is enough.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KMaxFullResult:
    M: float
    N: int
    K_max_order: int       # K_max moment truncation
    K_truncated: float
    z2_values: np.ndarray
    moments: np.ndarray
    breakeven: float
    K_over_M: float
    solver_status: str
    wall_s: float


def mu_M(M: float) -> float:
    return M * math.sin(math.pi / M) / math.pi


def breakeven_K(M: float) -> float:
    return 1.0 + (M - 1.0) / mu_M(M)


def _fourier_to_moment_coeffs(j: int, K_max: int):
    """Return (real_coeffs, imag_coeffs, err) such that
       Re f_hat(j) = sum_{k=0..K_max} real_coeffs[k] * m_k +/- err
       Im f_hat(j) = sum_{k=0..K_max} imag_coeffs[k] * m_k +/- err

    Truncation error: |(-2 pi j)^k / k!| * (1/4)^k summed for k > K_max.
    """
    real_c = np.zeros(K_max + 1)
    imag_c = np.zeros(K_max + 1)
    factorial = 1.0
    for k in range(K_max + 1):
        if k > 0:
            factorial *= k
        # (-2 pi i j)^k / k!
        val_pow = (-2.0 * math.pi * j) ** k / factorial
        # split by k mod 4: 1, -i, -1, i
        m4 = k % 4
        if m4 == 0:
            real_c[k] = val_pow
        elif m4 == 1:
            imag_c[k] = -val_pow
        elif m4 == 2:
            real_c[k] = -val_pow
        else:
            imag_c[k] = val_pow

    # Truncation error: sum_{k > K_max} (2 pi j)^k / k! * (1/4)^k
    err = 0.0
    factorial = 1.0
    for k in range(1, K_max + 1):
        factorial *= k
    # K_max+1 onwards
    fact = factorial
    for k in range(K_max + 1, K_max + 100):
        fact *= k
        term = (2.0 * math.pi * j) ** k / fact * (0.25 ** k)
        err += term
        if term < 1e-30:
            break
    return real_c, imag_c, err


def solve_k_max_full(M: float, N: int, K_max: int = None,
                     solver: str = "CLARABEL",
                     verbose: bool = False) -> KMaxFullResult:
    """K_max SDP with Hausdorff + Bochner-f + Bochner-g + L2 lift."""
    import cvxpy as cp

    t0 = time.time()
    mu = mu_M(M)
    if K_max is None:
        K_max = max(2 * N + 2, 12)        # need enough moments for Fourier accuracy

    # -- variables
    m = cp.Variable(K_max + 1)
    c = cp.Variable(N)
    s = cp.Variable(N)
    cc = cp.Variable(N, nonneg=True)
    ss = cp.Variable(N, nonneg=True)
    cs = cp.Variable(N)
    z2 = cc + ss
    R = cc - ss
    I_ = 2 * cs

    cons = []

    # -- Normalisation
    cons.append(m[0] == 1)

    # -- Hausdorff PSD (support [-1/4, 1/4])
    # Hankel H1: H1[i, j] = m_{i+j} PSD
    # Use as many rows as fit: i, j in 0..H_dim-1, where H_dim is chosen so
    # that 2*H_dim - 2 <= K_max
    H_dim = (K_max + 2) // 2
    H1 = cp.Variable((H_dim, H_dim), symmetric=True)
    for i in range(H_dim):
        for j in range(H_dim):
            cons.append(H1[i, j] == m[i + j])
    cons.append(H1 >> 0)

    # H2: H2[i, j] = (1/16) m_{i+j} - m_{i+j+2}
    # Need indices up to 2*(H_dim-1)+2 = 2 H_dim, so K_max >= 2 H_dim
    H2_dim = (K_max - 2) // 2 + 1
    if H2_dim >= 1:
        H2 = cp.Variable((H2_dim, H2_dim), symmetric=True)
        for i in range(H2_dim):
            for j in range(H2_dim):
                k = i + j
                if k + 2 <= K_max:
                    cons.append(H2[i, j] == (1.0 / 16.0) * m[k] - m[k + 2])
                else:
                    cons.append(H2[i, j] == 0)
        cons.append(H2 >> 0)

    # -- Fourier-to-moment map (truncated)
    # For each j = 1..N: c_j = real-coeffs . m, s_j = imag-coeffs . m, with
    # truncation error bounded by err_j.
    for j in range(1, N + 1):
        real_c, imag_c, err = _fourier_to_moment_coeffs(j, K_max)
        # |c_j - sum real_c * m_k| <= err
        cons.append(c[j - 1] - real_c @ m <= err)
        cons.append(c[j - 1] - real_c @ m >= -err)
        cons.append(s[j - 1] - imag_c @ m <= err)
        cons.append(s[j - 1] - imag_c @ m >= -err)

    # -- Lasserre L2 PSD per j
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

    # -- Lemma 2.14
    cons.append(z2 <= mu)

    # -- MO 2004 Lemma 2.17 (strong form):  c_2 - 2 c_1 + 1 <= 0
    if N >= 2:
        cons.append(c[1] - 2 * c[0] + 1 <= 0)

    # -- Peak constraint
    cons.append(1 + 2 * cp.sum(R) == M)

    # -- Bochner of f
    NN = N + 1
    T_re_f = cp.Variable((NN, NN), symmetric=True)
    T_im_f = cp.Variable((NN, NN))
    cons.append(T_im_f + T_im_f.T == 0)
    for i in range(NN):
        cons.append(T_re_f[i, i] == 1)
    for i in range(NN):
        for j in range(NN):
            k = abs(i - j)
            if k == 0:
                continue
            sign = 1 if i > j else -1
            cons.append(T_re_f[i, j] == c[k - 1])
            cons.append(T_im_f[i, j] == sign * s[k - 1])
    big_f = cp.bmat([[T_re_f, -T_im_f], [T_im_f, T_re_f]])
    cons.append(big_f >> 0)

    # -- Bochner of g (Toeplitz of g_hat PSD)
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
            sign = 1 if i > j else -1
            cons.append(T_re_g[i, j] == R[k - 1])
            cons.append(T_im_g[i, j] == sign * I_[k - 1])
    big_g = cp.bmat([[T_re_g, -T_im_g], [T_im_g, T_re_g]])
    cons.append(big_g >> 0)

    # -- Bochner of M - g
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
            sign = 1 if i > j else -1
            cons.append(T_re_S[i, j] == -R[k - 1])
            cons.append(T_im_S[i, j] == -sign * I_[k - 1])
    big_S = cp.bmat([[T_re_S, -T_im_S], [T_im_S, T_re_S]])
    cons.append(big_S >> 0)

    # -- Objective
    obj = cp.Maximize(cp.sum(z2))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)

    bk = breakeven_K(M)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return KMaxFullResult(
            M=M, N=N, K_max_order=K_max,
            K_truncated=float("inf"),
            z2_values=np.array([]),
            moments=np.array([]),
            breakeven=bk,
            K_over_M=float("inf"),
            solver_status=prob.status,
            wall_s=time.time() - t0,
        )
    sum_z2 = float(prob.value)
    K_trunc = 1.0 + 2.0 * sum_z2
    return KMaxFullResult(
        M=M, N=N, K_max_order=K_max,
        K_truncated=K_trunc,
        z2_values=np.asarray(z2.value).flatten(),
        moments=np.asarray(m.value).flatten(),
        breakeven=bk,
        K_over_M=K_trunc / M,
        solver_status=prob.status,
        wall_s=time.time() - t0,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.276, 1.30, 1.50])
    p.add_argument("--N", type=int, nargs="*", default=[4, 6])
    p.add_argument("--K_max", type=int, default=None)
    p.add_argument("--solver", type=str, default="CLARABEL")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("=== K_max SDP with FULL admissibility (Hausdorff + Bochner-f) ===")
    print()
    print(f"{'N':>3} {'K_max':>6} {'M':>7} {'breakeven':>11} "
          f"{'K_trunc':>11} {'K/M':>7} {'wall':>7}")
    print("-" * 75)
    for N in args.N:
        K_max = args.K_max if args.K_max else max(2 * N + 4, 12)
        for M in args.M:
            r = solve_k_max_full(M, N, K_max=K_max,
                                 solver=args.solver, verbose=args.verbose)
            print(f"{N:>3d} {K_max:>6d} {M:>7.4f} {r.breakeven:>11.4f} "
                  f"{r.K_truncated:>11.4f} {r.K_over_M:>7.4f} {r.wall_s:>7.1f}")
    print()
    print("Conjecture: K_max(M) ≤ M for M < 2 (gives C_1a >= 1.42 unconditional)")

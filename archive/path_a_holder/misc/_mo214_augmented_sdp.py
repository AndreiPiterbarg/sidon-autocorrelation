"""Quick augmented MO 2.14 SDP: minimize M with full Bochner-on-f and Bochner-on-g.

PROBLEM:
  C_{1a} := inf ||f*f||_inf over f >= 0, supp f ⊂ [-1/4, 1/4], ∫f = 1.
  MO 2.14: |hat f(j)|^2 <= mu(M) := M sin(pi/M)/pi  with M = ||f*f||_inf.

GOAL: tighten by setting up a (small N <= 5) SDP minimizing M with constraints:
  - hat f(0) = 1
  - z_j := |hat f(j)|^2 <= mu(M)            (MO 2.14)
  - Bochner-PSD: Toeplitz [hat f(j-k)] >= 0   (Bochner: f >= 0)
  - Bochner-PSD: Toeplitz [hat g(j-k)] >= 0   (g = f*f >= 0)
  - Bochner-PSD: Toeplitz [(M*delta - hat g)(j-k)] >= 0  (M - g >= 0)
  - g(0) = sum_j hat g(j) >= K >= 2   (Cauchy-Schwarz lower bound K = ||f||_2^2 >= 2)
    (since g(0) = (f*f)(0) = int f(x) f(-x) dx is NOT K, but we use a different
     equality below)

Variables: real Lasserre lift c_j = Re hat f(j), s_j = Im hat f(j),
           plus z_j = c_j^2 + s_j^2 lifts (so the MO 2.14 constraint is linear in z).
We use cc_j, ss_j, cs_j as in k_max_sdp.py.
"""
from __future__ import annotations
import math
import numpy as np
import cvxpy as cp


def mu(M: float) -> float:
    return M * math.sin(math.pi / M) / math.pi


def build_sdp(M: float, N: int):
    """Returns (problem, variables) for given fixed M, N. Feasibility: status optimal => M is feasible."""
    muM = mu(M)
    c = cp.Variable(N)
    s = cp.Variable(N)
    cc = cp.Variable(N, nonneg=True)
    ss = cp.Variable(N, nonneg=True)
    cs = cp.Variable(N)
    z2 = cc + ss             # z_j = |hat f(j)|^2
    R = cc - ss              # Re hat g(j) = c^2 - s^2
    I_ = 2 * cs              # Im hat g(j)
    cons = []
    # Lasserre L2 PSD per j on (1, c_j, s_j) lift
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
    # MO 2.14
    cons.append(z2 <= muM)
    # K = ||f||_2^2 = 1 + 2 sum z_j  (Parseval; integers index Fourier coefs)
    K_expr = 1.0 + 2.0 * cp.sum(z2)
    # Cauchy-Schwarz: K >= (int f)^2 / |supp| = 1 / (1/2) = 2  (supp width 1/2 since f on [-1/4,1/4])
    cons.append(K_expr >= 2.0)
    # Bochner of f
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
            if i > j:
                cons.append(T_re_f[i, j] == c[k - 1])
                cons.append(T_im_f[i, j] == s[k - 1])
            else:
                cons.append(T_re_f[i, j] == c[k - 1])
                cons.append(T_im_f[i, j] == -s[k - 1])
    big_f = cp.bmat([[T_re_f, -T_im_f], [T_im_f, T_re_f]])
    cons.append(big_f >> 0)
    # Bochner of g
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
    # Bochner of (M - g)
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
    # Trivial objective for feasibility
    return cp.Problem(cp.Minimize(0), cons), z2


def build_sdp_with_K_upper(M: float, N: int, K_upper: float):
    """Same SDP plus K = 1 + 2 sum z_j <= K_upper (symmetric-style closure)."""
    prob, z2 = build_sdp(M, N)
    prob.constraints.append(1.0 + 2.0 * cp.sum(z2) <= K_upper)
    return prob, z2


def is_feasible(M: float, N: int, K_upper: float = None) -> bool:
    if K_upper is None:
        prob, _ = build_sdp(M, N)
    else:
        prob, _ = build_sdp_with_K_upper(M, N, K_upper)
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        try:
            prob.solve(solver="SCS", verbose=False)
        except Exception:
            return False
    return prob.status in ("optimal", "optimal_inaccurate")


def bisect_min_M(N: int, lo: float = 1.001, hi: float = 1.40, tol: float = 1e-4,
                 K_mode: str = "K_geq_2"):
    """Bisection: smallest M for which the augmented SDP is feasible.
    K_mode: 'K_geq_2' (default; only K>=2 used in build_sdp), or
            'K_leq_M' (additionally constrain K <= M, the symmetric closure)."""
    def feas(M):
        if K_mode == "K_geq_2":
            return is_feasible(M, N)
        elif K_mode == "K_leq_M":
            return is_feasible(M, N, K_upper=M)
        else:
            raise ValueError(K_mode)

    if not feas(hi):
        return None
    if feas(lo):
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feas(mid):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    print("=== Augmented MO 2.14 + Bochner SDP — minimize M ===")
    print()
    print("Mode 1: only K >= 2 (rigorous CS bound on K = ||f||_2^2)")
    print(f"{'N':>3} {'min feasible M':>16} {'mu(M)':>10}")
    print("-" * 40)
    for N in [3, 4, 5, 6]:
        M_lb = bisect_min_M(N, lo=1.001, hi=1.40, tol=5e-5, K_mode="K_geq_2")
        if M_lb is None:
            print(f"{N:>3d} {'INFEAS @ 1.40':>16}")
        else:
            print(f"{N:>3d} {M_lb:>16.5f} {mu(M_lb):>10.5f}")
    print()
    print("Mode 2: also K <= M (symmetric closure / K=M conjecture; NOT rigorous in general)")
    print(f"{'N':>3} {'min feasible M':>16} {'mu(M)':>10}")
    print("-" * 40)
    for N in [3, 4, 5, 6]:
        M_lb = bisect_min_M(N, lo=1.001, hi=1.50, tol=5e-5, K_mode="K_leq_M")
        if M_lb is None:
            print(f"{N:>3d} {'INFEAS @ 1.50':>16}")
        else:
            print(f"{N:>3d} {M_lb:>16.5f} {mu(M_lb):>10.5f}")
    print()


if __name__ == "__main__":
    main()

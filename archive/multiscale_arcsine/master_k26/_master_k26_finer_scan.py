"""Finer scan around the K26 alt-point to find the true float optimum."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq
import math

DELTA = 0.138
U = 0.5 + DELTA
N_QP = 119
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
MV = np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


def K2(deltas, lambdas, split=10.0):
    def f(xi):
        s = 0.0
        for lam, d in zip(lambdas, deltas):
            s += lam * j0(np.pi * d * xi) ** 2
        return s * s
    v1, _ = quad(f, 0.0, split, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, split, np.inf, limit=800, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def K_arr(xi, deltas, lambdas):
    out = np.zeros_like(np.atleast_1d(xi), dtype=float)
    for lam, d in zip(lambdas, deltas):
        out += lam * j0(np.pi * d * xi) ** 2
    return out


def Mcert(k_1, K_2_val, S_1):
    if K_2_val <= 1 + 2 * k_1 * k_1:
        return None
    a = (4.0 / U) / S_1
    target = 2.0 / U + a
    rad2 = K_2_val - 1 - 2 * k_1 * k_1

    def sup_R(M):
        if M <= 1.0:
            return float("-inf")
        mu_ = M * math.sin(math.pi / M) / math.pi
        ys2 = (k_1 ** 2) * (M - 1) / (K_2_val - 1)
        ys = math.sqrt(max(0.0, ys2))
        if ys <= mu_:
            return M + 1 + math.sqrt((M - 1) * (K_2_val - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float("inf")
        return M + 1 + 2 * mu_ * k_1 + math.sqrt(rad1 * rad2)
    try:
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-12)
    except Exception:
        return None


qp_xi = np.arange(1, N_QP + 1) / U
best = (-np.inf, None, None)
for d2 in np.arange(0.040, 0.075, 0.0025):
    for l1 in np.arange(0.91, 0.96, 0.005):
        deltas = [DELTA, float(d2)]
        lambdas = [float(l1), 1.0 - float(l1)]
        k1 = sum(la * j0(np.pi * d * 1.0) ** 2 for la, d in zip(lambdas, deltas))
        k2 = K2(deltas, lambdas)
        kqp = K_arr(qp_xi, deltas, lambdas)
        S1 = float(np.sum(MV ** 2 / kqp))
        m = Mcert(k1, k2, S1)
        if m is not None and m > best[0]:
            best = (m, float(d2), float(l1))
        if m is not None and m > 1.2800:
            print(f"  d2={d2:.4f} l1={l1:.3f}  M={m:.6f}  (over 1.28)")
print(f"\nFinest float optimum: M={best[0]:.6f} at d2={best[1]}, l1={best[2]}")

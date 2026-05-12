"""Cross-check: replicate the arb-based rigorous S_1 upper bound for the
v4 threescale-N200 best certificate.

This verifies that S_1 = sum_{j=1}^{200} a_j^2 / K_hat(j/u) computed in arb
with rounded coefficients (10^12 denom) actually matches 29.84090655472349.
"""
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import flint
from flint import arb

DELTA1_Q = Fraction(138, 1000)
DELTA2_Q = Fraction(55, 1000)
DELTA3_Q = Fraction(25, 1000)
LAM1_Q = Fraction(85, 100)
LAM2_Q = Fraction(10, 100)
LAM3_Q = Fraction(5, 100)
U_Q = Fraction(1, 2) + DELTA1_Q
RATIONAL_DEN = 10**12
N_MODES = 200


def Q_arb(q):
    return arb(q.numerator) / arb(q.denominator)


def K_hat_arb(xi, deltas_arb, lambdas_arb, PI):
    out = arb(0)
    for lam, d in zip(lambdas_arb, deltas_arb):
        j0_val = (PI * d * xi).bessel_j(0)
        out = out + lam * j0_val * j0_val
    return out


def main():
    flint.ctx.prec = 256
    PI = arb.pi()
    deltas_arb = [Q_arb(DELTA1_Q), Q_arb(DELTA2_Q), Q_arb(DELTA3_Q)]
    lambdas_arb = [Q_arb(LAM1_Q), Q_arb(LAM2_Q), Q_arb(LAM3_Q)]
    U = Q_arb(U_Q)

    # Run QP independently to get the rounded coefficients
    import cvxpy as cp
    from scipy.special import j0 as scipy_j0
    u_f = float(U_Q)
    w = np.zeros(N_MODES)
    for j in range(1, N_MODES + 1):
        xi_j = j / u_f
        v = 0.0
        for lam, d in zip([0.85, 0.10, 0.05],
                          [0.138, 0.055, 0.025]):
            v += lam * scipy_j0(math.pi * d * xi_j) ** 2
        w[j - 1] = v
    n_grid = max(5001, 4 * N_MODES + 1)
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, N_MODES))
    for j in range(1, N_MODES + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)

    a = cp.Variable(N_MODES)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver="MOSEK", verbose=False)
    a_value = np.asarray(a.value).flatten()
    print(f"  QP status = {prob.status}")
    print(f"  unrounded S_1 = {float(np.sum(a_value ** 2 / w)):.12f}")

    # Round to Fraction(10^12)
    coeffs_q = [Fraction(int(round(av * RATIONAL_DEN)), RATIONAL_DEN) for av in a_value]
    coeffs_arb = [Q_arb(q) for q in coeffs_q]

    # Compute S_1 in arb
    inv_u = arb(1) / U
    total = arb(0)
    t0 = time.time()
    for j, a_arb in enumerate(coeffs_arb, start=1):
        xi = arb(j) * inv_u
        kh = K_hat_arb(xi, deltas_arb, lambdas_arb, PI)
        term = (a_arb * a_arb) / kh
        total = total + term
    print(f"  arb S_1 upper = {float(total.upper()):.12f}")
    print(f"  arb S_1 mid   = {float(total.mid()):.12f}")
    print(f"  arb S_1 rad   = {float(total.rad()):.4e}")
    print(f"  time = {time.time() - t0:.1f}s")

    # Compare to v4 reported
    s1_v4 = 29.84090655472349
    diff_upper = float(total.upper()) - s1_v4
    diff_mid = float(total.mid()) - s1_v4
    print(f"\n  v4 reported S_1_upper = {s1_v4:.12f}")
    print(f"  W9 arb upper diff     = {diff_upper:.4e}")
    print(f"  W9 arb mid   diff     = {diff_mid:.4e}")
    if abs(diff_upper) < 1e-9:
        print(f"  *** EXACT MATCH (within arb precision) ***")
    elif abs(diff_upper) < 1e-3:
        print(f"  Close match (likely due to QP solver tolerance)")


if __name__ == "__main__":
    main()

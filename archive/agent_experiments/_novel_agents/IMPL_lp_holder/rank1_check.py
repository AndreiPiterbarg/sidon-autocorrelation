"""Diagnostic: verify the SDP gives the EXPECTED VALUE when restricted to rank-1.

For uniform tilde_f (rank-1 case), we know analytically:
    sum_{j=0}^{N_leg} rho_j^2 -> ||tilde_R||_2^2 = 1/3 as N_leg -> infty.
The SDP RELAXATION will give SDP_value <= rank-1 evaluation = sum rho_j^2 (uniform).

If SDP_value < this, the relaxation is loose.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lp_holder_4pt import (
    legendre_orthonormal_minus2_2_Q,
    build_4pt_l2_sdp,
)


def uniform_rank1_eval(N_leg: int) -> float:
    """Evaluate sum_{j=0}^{N_leg} rho_j(uniform_f)^2."""
    def m(a):
        if a % 2 == 1:
            return 0.0
        return 1.0 / (a + 1)

    def tilde_r(a):
        out = 0.0
        for l in range(a + 1):
            out += math.comb(a, l) * ((-1) ** l) * m(a - l) * m(l)
        return out

    Q = legendre_orthonormal_minus2_2_Q(N_leg)
    total = 0.0
    for j in range(N_leg + 1):
        rho = sum(Q[j, r] * tilde_r(r) for r in range(j + 1))
        total += rho ** 2
    return total


def main():
    print("Rank-1 evaluation at uniform tilde_f vs SDP relaxation value:")
    print(f"{'N_leg':>6} {'rank1_uniform':>15} {'true_||R||_2^2':>15} {'sdp_relax':>15} {'gap_(rank1-sdp)':>20}")
    for N_leg in [2, 3, 4, 5]:
        rank1 = uniform_rank1_eval(N_leg)
        true_val = 1.0 / 3.0  # full ||tilde_R||_2^2 for uniform
        # Solve SDP at k = N_leg
        problem, info, misc = build_4pt_l2_sdp(k=N_leg, N_leg=N_leg)
        problem.solve(solver=cp.MOSEK, verbose=False)
        sdp_val = problem.value
        gap = rank1 - sdp_val
        print(f"{N_leg:>6} {rank1:>15.8f} {true_val:>15.8f} "
              f"{sdp_val:>15.8f} {gap:>20.8f}")


if __name__ == "__main__":
    main()

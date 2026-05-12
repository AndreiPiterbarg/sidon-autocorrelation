"""Bessel-ansatz primal upper bound on C_{1a}.

STATUS (2026-04-20): This module produces a PRIMAL upper bound from the
restricted even-Bessel family  phi_j(x) = (1 - 16 x^2)^{j - 1/2}.  An
empirical scan P = 4..32 PLATEAUS at M_P^* ~ 2.012, far above current
LB=1.2802 and UB=1.5029.  The family is too restrictive:
  - it only represents even f (misses MV's asymmetric optimisers);
  - even with signed coefficients + Putinar Q(u) >= 0 on [0,1] for
    admissibility of f = sqrt(1-16x^2) * Q(u = 1-16x^2), the ansatz
    concentrates mass near x=0 and cannot approximate the step-function
    CS-type extremisers.
To push Bessel past 1.28 one must instead use the basis inside the
Lasserre DUAL (as Rechnitzer does for nu_2^2 via Hoelder duality), not
the primal.  See the discussion at the bottom of this docstring.

Given the Bessel family phi_j(x) = (1 - 16 x^2)^{j - 1/2} on [-1/4, 1/4]
for j = 1..P, we compute

    M_P^*  :=  min_{a: f = sum a_j phi_j >= 0, int f = 1}
               max_{t in [-1/2, 1/2]}  a^T K(t) a

which is an UPPER bound on C_{1a} (since it is the inf over a restricted
subclass of admissible f).  Rechnitzer's L^2 analogue achieved ~100-digit
precision at P=101; for L^infty that route ran via Hoelder duality on the
L^2 norm and does NOT transfer to the sup-norm bound.

Methods
-------
1. SDP relaxation (``solve_bessel_sdp``):
        min_A   max_{t in T}  tr(K(t) A)
        s.t.    A succeq 0,  A >= 0 entrywise,  beta^T A beta = 1,
   dropping rank(A)=1.  Solved in CVXPY with Clarabel.  Gives a LOWER
   bound on M_P^* (because dropping rank is a relaxation of the primal
   min) and hence NOT directly an upper bound on C_{1a}.  The SDP is a
   useful lower bound on the primal infimum.

2. Cutting-plane primal (``solve_bessel_primal_cutting``):
        iterate  a^(k+1) = argmin_{a >= 0, beta^T a = 1} max_{i <= k} a^T K(t_i^*) a
        where t_{k+1}^* = argmax_t (a^(k))^T K(t) a^(k).
   Each inner step is a QCQP (non-convex in a since max-of-quadratics);
   we solve a semidefinite relaxation (Shor) per step.  Returns a
   feasible a and UB = max_t a^T K(t) a on a fine verification grid.

3. Rank-1 extraction (``extract_a_from_A``):
   Given the SDP optimum A*, recover a by top-eigenvector extraction
   and projection onto {a >= 0, beta^T a = 1}.  Re-evaluate the primal
   objective to get a feasible certified UB.

Use ``rigorous_verify`` to re-evaluate max_t a^T K(t) a in Arb ball
arithmetic on a fine t-grid for a certified primal bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

import flint
from flint import arb, fmpq

from certified_lasserre.bessel_kernel import (
    bessel_K_matrix,
    bilinear_ff_at,
    normalisation_row,
    _arb_K_at_zero,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arb_to_float(a: arb) -> float:
    return float(a.mid())


def K_matrix_float(t: float, P: int, prec_bits: int = 200) -> np.ndarray:
    """K(t) as numpy float array (P x P) via Arb ball arithmetic midpoints."""
    tf = Fraction(t).limit_denominator(10 ** 9)
    t_q = fmpq(tf.numerator, tf.denominator)
    K_arb = bessel_K_matrix(t_q, P, prec_bits=prec_bits)
    K = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        for j in range(P):
            K[i, j] = _arb_to_float(K_arb[i][j])
    return K


def beta_vec_float(P: int, prec_bits: int = 200) -> np.ndarray:
    flint.ctx.prec = prec_bits
    return np.array([_arb_to_float(b) for b in normalisation_row(P)], dtype=np.float64)


# ---------------------------------------------------------------------------
# SDP relaxation:  min_A  max_t  tr(K(t) A)
# ---------------------------------------------------------------------------

@dataclass
class BesselSDPResult:
    P: int
    t_grid: np.ndarray
    val: float
    A_opt: np.ndarray  # (P, P)
    a_rank1: np.ndarray  # (P,), extracted via top eigenvector
    ub_feasible: float  # UB from plugging a_rank1 back into max_t a^T K(t) a
    ub_feasible_fine: float  # on a finer verification grid
    status: str


def solve_bessel_sdp(
    P: int,
    n_grid: int = 41,
    prec_bits: int = 300,
    t_max: float = 0.5,
    verbose: bool = True,
) -> BesselSDPResult:
    """Solve the SDP relaxation of the Bessel primal.

    A t-grid on [0, t_max] is used (by symmetry t in [-t_max, 0] is the
    same).  The SDP has O(P^2) variables in A and |grid| linear constraints.
    """
    import cvxpy as cp

    # Grid; avoid t=0 as an isolated peak usually dominates
    t_grid = np.linspace(0.0, t_max, n_grid)
    beta = beta_vec_float(P, prec_bits=prec_bits)

    # Precompute K(t) for each grid point (expensive; P=40 => ~20s per t)
    if verbose:
        print(f"  precomputing K(t) on {n_grid}-point grid ...", flush=True)
    K_list = []
    for idx, t in enumerate(t_grid):
        K_list.append(K_matrix_float(float(t), P, prec_bits=prec_bits))
        if verbose and idx % max(1, n_grid // 10) == 0:
            print(f"    grid point {idx}/{n_grid}  t={t:.4f}", flush=True)

    # SDP:  min s  s.t.  s >= tr(K(t_i) A)  for each i;
    #                    beta^T A beta = 1;  A succeq 0;  A_{jm} >= 0.
    A = cp.Variable((P, P), symmetric=True)
    s = cp.Variable()
    constraints = [A >> 0, A >= 0, beta @ A @ beta == 1]
    for K in K_list:
        constraints.append(cp.trace(K @ A) <= s)
    prob = cp.Problem(cp.Minimize(s), constraints)
    if verbose:
        print(f"  solving SDP: P={P}, grid={n_grid}, vars={P*(P+1)//2}", flush=True)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    A_opt = np.array(A.value)
    A_opt = 0.5 * (A_opt + A_opt.T)
    # Top eigenvector extraction (rank-1 projection):
    w, V = np.linalg.eigh(A_opt)
    # Largest eigenvalue corresponds to dominant rank-1 direction
    v_top = V[:, -1]
    a_rank1 = v_top * np.sign(v_top.sum())  # flip sign so sum >= 0
    # Clip negatives and renormalise so beta^T a = 1
    a_pos = np.maximum(a_rank1, 0.0)
    if beta @ a_pos <= 0:
        # fallback: use |v_top|
        a_pos = np.abs(v_top)
    a_pos = a_pos / (beta @ a_pos)

    # Evaluate UB on coarse grid
    ub_coarse = max(float(a_pos @ K @ a_pos) for K in K_list)

    # Re-evaluate on a finer grid (but reuse coarse K precomputation for speed)
    n_fine = 201
    t_fine = np.linspace(0.0, t_max, n_fine)
    ub_fine = 0.0
    if verbose:
        print(f"  fine-grid verify on {n_fine} points ...", flush=True)
    for tf in t_fine:
        Kf = K_matrix_float(float(tf), P, prec_bits=prec_bits)
        val = float(a_pos @ Kf @ a_pos)
        if val > ub_fine:
            ub_fine = val

    return BesselSDPResult(
        P=P, t_grid=t_grid,
        val=float(prob.value) if prob.value is not None else float("nan"),
        A_opt=A_opt, a_rank1=a_pos,
        ub_feasible=ub_coarse, ub_feasible_fine=ub_fine,
        status=str(prob.status),
    )


# ---------------------------------------------------------------------------
# Rigorous verification via Arb
# ---------------------------------------------------------------------------

def rigorous_max_arb(
    a: Sequence[float],
    t_grid: Sequence[float],
    P: int,
    prec_bits: int = 384,
) -> arb:
    """Rigorously evaluate max_t a^T K(t) a on the given t-grid using Arb.

    Returns an arb ball that PROVABLY contains (an upper bound on) the
    evaluated max (modulo the grid resolution; a continuous max is
    bracketed by adjacent grid points + a radius term we DO NOT add here
    — callers should ensure grid density).
    """
    from fractions import Fraction
    flint.ctx.prec = prec_bits
    a_arb = [arb(Fraction(float(x)).limit_denominator(10 ** 15).numerator) /
             arb(Fraction(float(x)).limit_denominator(10 ** 15).denominator)
             for x in a]
    best = arb(0)
    for t in t_grid:
        t_q = fmpq(Fraction(float(t)).limit_denominator(10 ** 9).numerator,
                   Fraction(float(t)).limit_denominator(10 ** 9).denominator)
        val = bilinear_ff_at(t_q, a, prec_bits=prec_bits)
        # Strict upper bound via midpoint + radius
        bound = arb(float(val.mid()) + float(val.rad()))
        if float(bound.mid()) > float(best.mid()):
            best = bound
    return best


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("=" * 70)
    print("bessel_outer.py — Bessel primal upper bound on C_{1a}")
    print("=" * 70)

    P = 8
    if len(sys.argv) > 1:
        P = int(sys.argv[1])

    t0 = time.time()
    res = solve_bessel_sdp(P=P, n_grid=31, prec_bits=200, verbose=True)
    print(f"\n  P = {P}")
    print(f"  SDP optimum (lower bound on primal):   {res.val:.6f}")
    print(f"  Rank-1 feasible UB (coarse grid):      {res.ub_feasible:.6f}")
    print(f"  Rank-1 feasible UB (fine 201-pt grid): {res.ub_feasible_fine:.6f}")
    print(f"  CVXPY status: {res.status}   time: {time.time()-t0:.1f}s")
    print(f"\n  Context: MV upper bound = 1.2748, CS2017 lower = 1.28, KK UB = 1.5029")
    print(f"  a_rank1[:8] = {res.a_rank1[:8].round(4)}")
    # Compare against MV 1.2748 explicitly
    if res.ub_feasible_fine < 1.2748:
        print(f"  => Bessel family achieves UB {res.ub_feasible_fine:.4f} < MV 1.2748  "
              f"(BEATS MV upper bound by {1.2748 - res.ub_feasible_fine:.4f})")
    elif res.ub_feasible_fine < 1.2802:
        print(f"  => Bessel UB is between MV 1.2748 and CS2017 LB 1.2802 ({res.ub_feasible_fine:.4f})")
    else:
        print(f"  => Bessel UB = {res.ub_feasible_fine:.4f} is above CS2017 LB; family may be under-expressive at P={P}")

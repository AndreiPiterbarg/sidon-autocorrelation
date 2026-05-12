"""Toeplitz-PSD probe on the MV multimoment dual.

Question
--------
The MV-multimoment dual (delsarte_dual/mv_multimoment.py) certifies a lower
bound on M = ||f*f||_inf via the master inequality (MM-10):

    2/u + a  <=  M + 1 + 2 sum_{j in N} y_j k_j
                  + sqrt(M - 1 - 2 sum y_j^2) * sqrt(K2 - 1 - 2 sum k_j^2)

where y_j = z_j^2 = |hat f(j)|^2, j = 1..n_max, k_j = hat_K(j), and the
admissibility constraints used by the rigorous certifier are:

  (B)  0 <= y_j <= mu(M) := M sin(pi/M)/pi      pointwise (bathtub / MO 2.14).

This probe asks: does adding the Bochner / Toeplitz-PSD constraint on the
sequence (y_0, y_1, ..., y_{n_max}) with y_0 = |hat f(0)|^2 = 1, namely

  (T)  Toep(y) := [y_{|j-k|}]_{j,k=0..n_max}  >=  0  (PSD),

tighten the certified lower bound on M beyond MV's value (~ 1.2748 / 1.2802)?

The sequence n -> |hat f(n)|^2 IS positive-definite (it is the Fourier
transform of f *~ f, the autocorrelation, which is non-negative), so (T) is a
valid constraint. The question is whether it is *active* on top of (B).

Procedure
---------
We bisect M on [1.27, 1.31].  At each test M, we maximise the master-inequality
RHS subject to (B) + (T).  The constraint y_0 = 1 closes the Toeplitz.  We
solve via cvxpy (Clarabel).  The objective is concave in y (linear gain plus
sqrt of a concave affine of y), so the maximisation over the convex set (box +
PSD) is a convex SDP (treated as a semidefinite-constrained concave-program;
we recast the sqrt term via a rotated SOC).

If the maximum RHS is < target = 2/u + a, then M is FEASIBLE (i.e. the master
inequality is violated, so our M cannot occur — call M an INFEASIBLE candidate
for f, hence a CERTIFIED lower bound on C_{1a}).  We bisect for the smallest
feasible M (rigorous numerical lower bound).

Comparison: we run the same bisection (a) WITHOUT (T) (pointwise / MV value)
and (b) WITH (T) (Bochner-augmented), at n_max = 4 and 5.  The difference
M_with - M_without is the value Bochner adds in this dual.
"""
from __future__ import annotations

import math
import sys
from typing import List, Tuple

import cvxpy as cp
import mpmath as mp
import numpy as np

# Make the local delsarte_dual package importable
sys.path.insert(0, "C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon")

from delsarte_dual.mv_multimoment import (
    k_values as mv_k_values,
    mu_of_M,
)
from delsarte_dual.mv_bound import (
    MV_DELTA,
    MV_K2_BOUND_OVER_DELTA,
    MV_U,
    MV_COEFFS_119,
    gain_parameter,
    min_G_on_0_quarter,
    S1_sum,
)


# ---------------------------------------------------------------------------
# Target  2/u + a   from MV
# ---------------------------------------------------------------------------
def mv_target() -> float:
    """Compute target = 2/u + a using MV's delta = 0.6079, with the MV
    rigorous values for u and the gain a as in mv_multimoment.py /
    delsarte_dual/mv_bound.py."""
    min_G, _ = min_G_on_0_quarter()
    S1 = S1_sum()
    a_gain = float(gain_parameter(min_G, S1))
    target = 2.0 / float(MV_U) + a_gain
    return target


# ---------------------------------------------------------------------------
# Toeplitz matrix construction (symbolic in cvxpy)
# ---------------------------------------------------------------------------
def cvxpy_toeplitz(y_full: cp.Expression, n: int) -> cp.Expression:
    """Build the (n+1)x(n+1) symmetric Toeplitz matrix with diagonal y_full[0]
    and j-th off-diagonals filled with y_full[j].

    y_full is a 1-D cvxpy variable/expression of length n+1.
    Returns an (n+1)x(n+1) symmetric matrix expression (Hermitian since real)."""
    rows = []
    for i in range(n + 1):
        row = [y_full[abs(i - j)] for j in range(n + 1)]
        rows.append(cp.hstack(row))
    return cp.vstack(rows)


# ---------------------------------------------------------------------------
# Master-inequality RHS as a cvxpy *concave* expression in y (lifted)
# ---------------------------------------------------------------------------
def rhs_concave(M: float, y_vec: cp.Variable, k_vec: np.ndarray, K2: float) -> cp.Expression:
    """RHS = M + 1 + 2 <k, y> + sqrt(M - 1 - 2 ||y||^2) * sqrt(K2 - 1 - 2 ||k||^2).

    Variables:  y_vec is a length-n_max nonneg cvxpy variable (y_j = z_j^2).
    Constants:  M, k_vec, K2.
    The sqrt(K2 - 1 - 2 ||k||^2) factor is constant; the other sqrt is concave
    in y because (M - 1 - 2 ||y||^2) is a concave (quadratic-in-y, with a
    minus-sum-of-squares term).  cvxpy can model sqrt(concave_nonneg) as
    geo_mean([concave_nonneg, 1]) (concave atom), or via the rotated-SOC
    epigraph.  We use cp.sqrt on a Boolean-affine concave argument exposed via
    cp.Variable rad1 with the constraint rad1 <= M - 1 - 2 sum_sq(y).
    """
    sum_sq_k = float(np.sum(k_vec ** 2))
    rad2 = K2 - 1.0 - 2.0 * sum_sq_k
    if rad2 <= 0:
        # square-root tail is imaginary => master inequality vacuous; signal
        # by returning -inf via a placeholder (caller treats infeasible).
        return None
    sqrt_rad2 = math.sqrt(rad2)

    # rad1 = M - 1 - 2 ||y||^2  (must be >= 0 for the sqrt step to apply)
    # We expose rad1 as a variable and constrain rad1 <= M - 1 - 2 ||y||^2,
    # then add sqrt_rad2 * sqrt(rad1).  Since sqrt is concave-monotone, this
    # is a valid relaxation when we MAXIMISE: the maximiser pushes rad1 up to
    # equality.
    rad1 = cp.Variable(nonneg=True)
    aux_constraint = (rad1 <= M - 1.0 - 2.0 * cp.sum_squares(y_vec))
    expr = M + 1.0 + 2.0 * (k_vec @ y_vec) + sqrt_rad2 * cp.sqrt(rad1)
    return expr, [aux_constraint]


# ---------------------------------------------------------------------------
# The SDP at fixed M
# ---------------------------------------------------------------------------
def max_rhs_at_M(
    M: float,
    n_max: int,
    delta: float = float(MV_DELTA),
    K2: float = float(MV_K2_BOUND_OVER_DELTA) / float(MV_DELTA),
    use_psd: bool = True,
) -> Tuple[float, np.ndarray]:
    """Solve max RHS s.t. (B) [+ (T) if use_psd], at fixed M.  Returns
    (rhs_value, y_array).  Returns (-inf, None) if infeasible/numerical fail.
    """
    k_vec = np.array([float(mv_k_values(n_max, delta=delta)[j]) for j in range(n_max)])
    mu = float(mu_of_M(M))

    y = cp.Variable(n_max, nonneg=True)
    constraints = [y <= mu]

    if use_psd:
        # Build full y_full = [1, y_1, ..., y_{n_max}] for Toeplitz.
        y_full = cp.hstack([np.array([1.0]), y])
        T = cvxpy_toeplitz(y_full, n_max)
        constraints.append(T >> 0)
        # Symmetrize (cvxpy needs symmetric for >> 0).
        # T is constructed symmetric by indexing [|i-j|], so this is automatic.

    rhs_expr_pack = rhs_concave(M, y, k_vec, K2)
    if rhs_expr_pack is None:
        return float("-inf"), None
    rhs_expr, aux_cons = rhs_expr_pack
    constraints.extend(aux_cons)

    prob = cp.Problem(cp.Maximize(rhs_expr), constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            prob.solve(solver=cp.SCS)
        except Exception:
            return float("-inf"), None

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return float("-inf"), None
    return float(prob.value), np.asarray(y.value).flatten()


# ---------------------------------------------------------------------------
# Bisection on M
# ---------------------------------------------------------------------------
def bisect_for_M(
    target: float,
    n_max: int,
    use_psd: bool,
    M_lo: float = 1.05,
    M_hi: float = 1.40,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    """Bisect for the LARGEST M for which max RHS(M) <= target.

    Reasoning: the master inequality says target <= RHS(M).  If at M_test
    we maximise RHS over (B)[+(T)] and get max_RHS < target, then NO admissible
    f can have ||f*f||_inf = M_test, so M_test is a certified lower bound on
    C_{1a}.  RHS(M) is monotonically increasing in M (linear gain + sqrt
    of an increasing arg minus z_j^2 term), so the largest such M is the
    sharpest dual lower bound the inequality gives.
    """
    rhs_lo, _ = max_rhs_at_M(M_lo, n_max, use_psd=use_psd)
    rhs_hi, _ = max_rhs_at_M(M_hi, n_max, use_psd=use_psd)
    print(f"    bracket: M_lo={M_lo:.5f} rhs={rhs_lo:.6f}  target={target:.6f}  (rhs<target: {rhs_lo < target})")
    print(f"    bracket: M_hi={M_hi:.5f} rhs={rhs_hi:.6f}  target={target:.6f}  (rhs<target: {rhs_hi < target})")
    if not (rhs_lo < target):
        return float("nan"), rhs_lo  # M_lo already too large
    if rhs_hi < target:
        return M_hi, rhs_hi          # whole bracket gives a bound; expand M_hi
    # Now rhs_lo < target <= rhs_hi.  Bisect for the crossover.
    while M_hi - M_lo > tol:
        M_mid = 0.5 * (M_lo + M_hi)
        rhs_mid, _ = max_rhs_at_M(M_mid, n_max, use_psd=use_psd)
        if rhs_mid < target:
            M_lo = M_mid
        else:
            M_hi = M_mid
    rhs_at, _ = max_rhs_at_M(M_lo, n_max, use_psd=use_psd)
    return M_lo, rhs_at


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    target = mv_target()
    print(f"MV target  2/u + a = {target:.6f}")
    print(f"MV K2 = ||K||_2^2 = {float(MV_K2_BOUND_OVER_DELTA)/float(MV_DELTA):.6f}")
    print()

    # Recompute the WITHOUT-PSD bound at multiple n_max to anchor MV's value.
    for n_max in (4, 5):
        print(f"--- n_max = {n_max} ---")
        print("[Pointwise (B) only -- this should reproduce ~1.2748]")
        M_no, rhs_no = bisect_for_M(target, n_max, use_psd=False)
        print(f"   M_no_PSD = {M_no:.6f}  (RHS at M_no = {rhs_no:.6f})")

        # Inspect optimizer for diagnostics
        rhs_v, y_v = max_rhs_at_M(M_no, n_max, use_psd=False)
        if y_v is not None:
            mu = float(mu_of_M(M_no))
            print(f"   y* (no-PSD) = {y_v}, mu(M)={mu:.6f}, y/mu = {y_v / mu}")

        print("[Pointwise (B) + Bochner Toeplitz PSD (T)]")
        M_psd, rhs_psd = bisect_for_M(target, n_max, use_psd=True)
        print(f"   M_PSD   = {M_psd:.6f}  (RHS at M_PSD = {rhs_psd:.6f})")
        print(f"   delta   = {M_psd - M_no:+.6e}")

        rhs_v, y_v = max_rhs_at_M(M_psd, n_max, use_psd=True)
        if y_v is not None:
            full = np.concatenate([[1.0], y_v])
            T = np.array([[full[abs(i-j)] for j in range(n_max+1)] for i in range(n_max+1)])
            eigs = np.linalg.eigvalsh(T)
            mu = float(mu_of_M(M_psd))
            print(f"   y* (PSD)    = {y_v}, mu(M)={mu:.6f}, y/mu = {y_v / mu}")
            print(f"   Toeplitz eigvals = {eigs}")
            print(f"   min eigval = {eigs.min():.3e}  (active iff ~ 0)")
        print()

    # Diagnostic: does Toeplitz EVER bite at MV's optimum y* = lambda * k?
    print("--- Direct check: is MV's interior optimum y* = lambda*k Bochner-feasible? ---")
    M_test = 1.274838
    K2 = float(MV_K2_BOUND_OVER_DELTA) / float(MV_DELTA)
    lam = math.sqrt((M_test - 1.0) / (K2 - 1.0))
    print(f"   M={M_test}, lambda = sqrt((M-1)/(K2-1)) = {lam:.6f}")
    for n_max in (4, 5, 8, 12, 16):
        kv = np.array([float(mv_k_values(n_max, delta=float(MV_DELTA))[j]) for j in range(n_max)])
        y_star = lam * kv
        full = np.concatenate([[1.0], y_star])
        T = np.array([[full[abs(i-j)] for j in range(n_max+1)] for i in range(n_max+1)])
        eigs = np.linalg.eigvalsh(T)
        mu = float(mu_of_M(M_test))
        viol_box = (y_star.max() - mu)
        print(f"   n_max={n_max}: y*[0..3]={y_star[:4]}, max y/mu={y_star.max()/mu:.4f}, min Toep eig={eigs.min():.3e}")


if __name__ == "__main__":
    main()

"""Bilevel optimiser over the G-coefficients for the MV dual bound.

Part B.5.  Outer layer: choose (a_0, ..., a_n) to maximise M*.  Inner
layer: ``certified_forbidden_max`` on fixed G.

The outer problem is a low-dimensional NON-CONVEX optimisation (the inner
layer is continuous but not convex in a_j), so we use SciPy's ``minimize``
with L-BFGS-B and finite-difference gradients.

Warm start
----------
MV's 119 coefficients are the natural warm start.  When ``n`` differs we
pad / truncate MV's coefficient list.  A user-supplied ``a_init`` overrides.

Caveats
-------
* Each outer evaluation triggers an inner bisection (~100 inner M samples,
  each calling ``max_rhs_over_z`` which is analytic / O(n_max^2)).  At
  ``n = 119`` outer steps cost a few seconds each.
* The gradient is numerical (central differences at 1e-5).  For ``n = 480``
  the outer layer needs ~hours to converge; run ``sweep.py`` on the pod.
* For paranoia: after the final G* is found, re-run inner at higher
  precision for certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    MV_COEFFS_119, MV_DELTA, MV_U, MV_K2_BOUND_OVER_DELTA,
    MVMultiMomentBound,
)
from .forbidden_region import certified_forbidden_max


@dataclass
class OptimiseResult:
    M_cert_float: float
    a_opt: List[float]
    n_inner_evals: int
    history: List[float]


def _evaluate_G(a_vec, delta, u, K2_bod, N, use_mo):
    """Compute the certified M* for a candidate G coefficient vector."""
    a_mpf = [mpf(float(a)) for a in a_vec]
    bound = MVMultiMomentBound(
        delta=delta, u=u, G_coeffs=a_mpf,
        K2_bound_over_delta=K2_bod, N=N,
        n_grid_minG=4001,  # faster for outer loop
    )
    # Quick guard: if gain a_gain <= 0 then the MV bound is vacuous.
    if bound.a_gain <= 0:
        return -1.0
    res = certified_forbidden_max(
        bound, use_mo=use_mo, M_lo=mpf("1.0001"), M_hi=mpf("1.40"),
        tol=mpf("1e-6"), max_iter=60,
    )
    return float(res.M_cert)


def optimise_G(
    a_init: Sequence[float] = None,
    delta=MV_DELTA, u=MV_U,
    K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
    N: int = 1,
    use_mo: bool = False,
    n: int = None,
    method: str = "L-BFGS-B",
    max_iter: int = 40,
    verbose: bool = True,
) -> OptimiseResult:
    """Maximise M*(a) over a in R^n by local search.

    Default warm start is MV's 119 coefficients (optionally truncated/padded
    to length ``n``).  If ``n`` is None uses ``len(a_init) or 119``.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise RuntimeError("scipy required for optimise_G")

    if a_init is None:
        a_init = [float(a) for a in MV_COEFFS_119]
    a_init = list(a_init)
    if n is None:
        n = len(a_init)
    if len(a_init) < n:
        a_init = a_init + [0.0] * (n - len(a_init))
    else:
        a_init = a_init[:n]

    history: List[float] = []
    eval_counter = {"n": 0}

    def objective(a_arr):
        eval_counter["n"] += 1
        M = _evaluate_G(a_arr, delta, u, K2_bound_over_delta, N, use_mo)
        history.append(M)
        if verbose and eval_counter["n"] % 5 == 0:
            print(f"    iter {eval_counter['n']:3d}: M* = {M:.6f}")
        return -M  # minimise negative

    res = minimize(
        objective,
        x0=np.array(a_init),
        method=method,
        options={"maxiter": max_iter, "disp": verbose},
    )
    return OptimiseResult(
        M_cert_float=-float(res.fun),
        a_opt=[float(x) for x in res.x],
        n_inner_evals=eval_counter["n"],
        history=history,
    )


if __name__ == "__main__":
    mp.mp.dps = 25
    print("=" * 70)
    print("optimizer.py — short self-test (few iterations only)")
    print("=" * 70)
    # Very short run: 10 iterations starting from MV's coefficients.
    res = optimise_G(N=2, n=119, max_iter=5, verbose=True)
    print(f"\n  M* after 5 L-BFGS-B iters = {res.M_cert_float:.6f}")
    print(f"  inner evals = {res.n_inner_evals}")

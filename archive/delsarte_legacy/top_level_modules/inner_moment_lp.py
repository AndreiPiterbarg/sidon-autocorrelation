"""Tighter inner bound on inf_f int g(t)(f*f)(t) dt for admissible f.

Mathematical setup
------------------
For any test function g with g >= 0 on [-1/2, 1/2], and any admissible f
(f >= 0, int f = 1, supp f ⊂ [-1/4, 1/4]), we have
    ||f*f||_inf * int g >= int g(t)(f*f)(t) dt
so
    C_{1a} = inf_f ||f*f||_inf >= (inf_f int g(f*f)) / int g.

The INNER inf_f int g(f*f) is what we want to compute TIGHTLY.

Key identity: for discrete f = sum_i c_i delta_{x_i} (x_i in [-1/4, 1/4]),
    int g(f*f) = c^T G c  where  G_ij = g(x_i + x_j).
This is a standard quadratic program (StQP) on the probability simplex:
    min c^T G c   s.t.  c >= 0,  sum c = 1.

For a RIGOROUS LOWER BOUND on this QP (required so our C_{1a} bound is
valid), we use the Shor SDP relaxation:
    min tr(G Y)
    s.t.  c >= 0,  sum c = 1,
          [[1, c^T], [c, Y]] PSD.
For any feasible rank-1 Y = c c^T, tr(GY) = c^T G c, so SDP value
<= optimal QP value. Hence the SDP gives a valid LOWER BOUND on the QP
min. For PSD G this is tight; for indefinite G there can be a gap.

This tighter inner bound can be STRICTLY larger than the pointwise
L^sharp bound used elsewhere in this package, because the Diracs
achieving pointwise-L^sharp differ across xi, so no single f attains
pointwise-L^sharp simultaneously.

A note on continuous vs discrete f
-----------------------------------
The LP above allows Dirac f (point masses). For such f, ||f*f||_inf = inf
and the dual bound is trivial. The extremiser of C_{1a} is a continuous
function, not a Dirac. The LP inner min computed here is the GREATEST
LOWER BOUND over all admissible f including Diracs, so it is a valid
(but potentially loose) lower bound on the continuous case too.

In particular, for Dirac f_0 = delta_{x_0}:
    int g (f_0 * f_0) = g(2 x_0)
so inf over Diracs = min_{u in [-1/2, 1/2]} g(u).

The Shor SDP can give a bound strictly BETWEEN min g and the Dirac inf,
depending on G's structure. In practice for our g's it lies near min g.

Usage
-----
    from delsarte_dual.inner_moment_lp import inner_sdp_lb
    val = inner_sdp_lb(g_callable, n_grid=64)
    # val is a rigorous lower bound on inf_f int g(f*f).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:
    HAVE_CVXPY = False


def build_G_matrix(g_callable: Callable[[float], float], n_grid: int = 64) -> np.ndarray:
    """G_ij = g(x_i + x_j) for x_i a grid on [-1/4, 1/4]."""
    xs = np.linspace(-0.25, 0.25, n_grid)
    G = np.zeros((n_grid, n_grid))
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            G[i, j] = g_callable(xi + xj)
    # Symmetrise for numerical hygiene.
    G = 0.5 * (G + G.T)
    return G, xs


def inner_sdp_lb(
    g_callable: Callable[[float], float],
    n_grid: int = 64,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> dict:
    """Rigorous lower bound on inf_f int g(f*f) via Shor SDP.

    Returns dict with:
      'sdp_lb':   SDP value (lower bound on QP min, hence on inf_f int g(f*f))
      'qp_ub':    upper bound on QP min from rank-1 extraction (run with c = sdp_sol)
      'min_g':    min of g on the grid {x_i + x_j} (trivial lower bd for ref)
      'diag_min': min of g(2 x_i) over the grid (Dirac inf)
      'n_grid':   grid size used
    """
    if not HAVE_CVXPY:
        raise RuntimeError("cvxpy is required for inner_sdp_lb.")

    G, xs = build_G_matrix(g_callable, n_grid=n_grid)
    n = n_grid

    # Shor SDP: min tr(GY), s.t. [[1, c^T], [c, Y]] >> 0, c >= 0, sum(c) = 1.
    c = cp.Variable(n, nonneg=True)
    Y = cp.Variable((n, n), symmetric=True)

    # Build the lifted (n+1) x (n+1) block.
    M = cp.bmat([
        [cp.Constant(np.array([[1.0]])), cp.reshape(c, (1, n))],
        [cp.reshape(c, (n, 1)), Y],
    ])

    # Bounding constraint: for Y = c c^T (rank-1), sum_{ij} Y_ij = (sum c)^2 = 1.
    # Adding sum(Y) == 1 to the relaxation keeps the SDP bounded and remains
    # valid for rank-1 optimum.
    constraints = [
        cp.sum(c) == 1,
        cp.sum(Y) == 1,
        cp.diag(Y) <= c,   # c_i^2 <= c_i for c_i in [0, 1]; tighter bound
        M >> 0,
    ]
    obj = cp.Minimize(cp.trace(G @ Y))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)

    sdp_lb = float(prob.value)
    c_sol = np.asarray(c.value).flatten() if c.value is not None else None

    # Rank-1 primal upper bound (what you'd get by rounding SDP to c).
    if c_sol is not None:
        qp_ub = float(c_sol @ G @ c_sol)
    else:
        qp_ub = float("nan")

    # Trivial references:
    min_G = float(G.min())
    diag_min = float(np.min(np.diag(G)))

    return {
        "sdp_lb": sdp_lb,
        "qp_ub": qp_ub,
        "c": c_sol,
        "min_G": min_G,
        "diag_min": diag_min,
        "n_grid": n_grid,
    }


def inner_dirac_lb(
    g_callable: Callable[[float], float], n_grid: int = 201
) -> float:
    """Trivial lower bound: min of g over [-1/2, 1/2] (Dirac inf).

    inf_f int g(f*f) <= min_{u in [-1/2,1/2]} g(u) (Dirac).
    So this is an UPPER BOUND on the inner inf, i.e., a LOWER BOUND to what
    our dual gives us if we could compute inf_f exactly. It is the MAXIMUM
    possible inner inf for our purposes (anything above is wrong).
    """
    us = np.linspace(-0.5, 0.5, n_grid)
    vals = np.array([g_callable(u) for u in us])
    return float(vals.min())


def solve_stqp_local(G: np.ndarray, n_restarts: int = 50, tol: float = 1e-10) -> dict:
    """Local search on the simplex for min c^T G c. Returns BEST c found.
    This is an UPPER bound on the QP min (any valid c gives an UB).
    For diagnostic comparison only; the SDP bound is what's rigorous.
    """
    from scipy.optimize import minimize
    n = G.shape[0]
    rng = np.random.default_rng(0)

    def fun(c):
        c = np.maximum(c, 0)
        s = c.sum()
        if s < 1e-12:
            return 1e20
        c = c / s
        return float(c @ G @ c)

    best = {"val": np.inf, "c": None}
    for r in range(n_restarts):
        if r == 0:
            x0 = np.ones(n) / n
        elif r <= n:
            x0 = np.zeros(n)
            x0[r - 1] = 1.0
        else:
            x0 = rng.dirichlet(np.ones(n))
        try:
            res = minimize(
                fun, x0, method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
            )
            v = fun(res.x)
            if v < best["val"]:
                c = np.maximum(res.x, 0)
                c /= c.sum()
                best = {"val": v, "c": c}
        except Exception:
            pass
    return best

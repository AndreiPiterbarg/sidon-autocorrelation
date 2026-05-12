"""Numerical evaluator for the dual functional

    M(g)  =   inf_{f >= 0, supp f c [-1/4, 1/4], int f = 1}
                 [ int int g(x+y) f(x) f(y) dx dy ]
                 / int g

For a fixed candidate g >= 0 supported on [-1/2, 1/2], M(g) is a *lower
bound* on C_{1a}.  This module computes it numerically by discretizing
f at d points on [-1/4, 1/4] and solving the resulting QP

    min_{mu in Delta_d}  mu^T G mu     where G[i,j] = g(x_i + x_j)

If G is PSD (i.e. g has nonneg Fourier transform, by Bochner), the QP
is convex and solved exactly via cvxpy.  Otherwise we use Frank-Wolfe
with multiple random restarts.

NOTE: this module is for *numerical exploration only*.  Rigorous
verification of M(g) >= 1.2805 requires Putinar SOS and rational
rounding (see build_sdp.py and verify_sos.py downstream).

Convention: x grid is uniform in [-1/4, 1/4] with d points; t grid for
the int g integration is uniform in [-1/2, 1/2] with n_t points
(default 4001).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import simpson


# ---------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------

def make_x_grid(d: int, x_lo: float = -0.25, x_hi: float = 0.25) -> np.ndarray:
    """Uniform d-point grid on [x_lo, x_hi]."""
    return np.linspace(x_lo, x_hi, d)


def kernel_matrix(g_func: Callable[[float], float],
                  x: np.ndarray) -> np.ndarray:
    """Build G[i,j] = g(x_i + x_j) on an x-grid of length d.

    Vectorised: g_func should accept a numpy array.
    """
    d = len(x)
    sums = x[:, None] + x[None, :]   # (d, d) matrix of x_i + x_j
    return g_func(sums)


def integrate_g(g_func: Callable[[float], float],
                t_lo: float = -0.5, t_hi: float = 0.5,
                n: int = 4001) -> float:
    """int_{t_lo}^{t_hi} g(t) dt via Simpson's rule (high-precision numeric)."""
    t = np.linspace(t_lo, t_hi, n)
    vals = g_func(t)
    return float(simpson(vals, t))


# ---------------------------------------------------------------------
# QP solvers on Delta_d
# ---------------------------------------------------------------------

def solve_qp_shor(G: np.ndarray, solver: str = "CLARABEL",
                  rlt: bool = True) -> Tuple[float, np.ndarray]:
    """Shor SDP relaxation of  min mu^T G mu  s.t. mu in Delta_d.

    Lift Y = [1; mu][1; mu]^T (size (d+1) x (d+1) PSD).
    Constraints:
      Y[0,0] = 1
      Y[0, 1:] = mu  >= 0
      sum_i Y[0, 1+i] = 1
      Y[1+i, 1+j] = mu_i mu_j; nonnegative; row-sums = mu_i (RLT, optional).

    Returns (lower_bound, mu_recovered) where lower_bound <= true min.
    For PSD G this is tight; for non-PSD G it is a relaxation.
    """
    import cvxpy as cp
    d = G.shape[0]
    n = d + 1
    Y = cp.Variable((n, n), PSD=True)
    mu = Y[0, 1:]
    XX = Y[1:, 1:]
    constr = [
        Y[0, 0] == 1,
        cp.sum(mu) == 1,
        mu >= 0,
        XX >= 0,                # entrywise non-negativity
        cp.diag(XX) >= 0,
    ]
    if rlt:
        # RLT: sum_j Y[1+i, 1+j] = mu_i (since mu_i sum_j mu_j = mu_i)
        for i in range(d):
            constr.append(cp.sum(XX[i, :]) == mu[i])
    obj = cp.Minimize(cp.trace(G @ XX))
    prob = cp.Problem(obj, constr)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Shor SDP failed: status={prob.status}")
    mu_val = np.asarray(mu.value).flatten()
    return float(prob.value), mu_val


def solve_qp_symmetric(G: np.ndarray, solver: str = "CLARABEL"
                       ) -> Tuple[float, np.ndarray]:
    """min mu^T G mu  s.t. mu in Delta_d AND mu_i = mu_{d-1-i} (symmetric).

    On the symmetric subspace, the Toeplitz-like sum-kernel G(x+y) is PSD
    (because Bochner's theorem applies to ghat >= 0 -> sum-kernel PSD on
    symmetric vectors).  This gives a *fast OVER-estimate* of M(g)
    suitable for screening but NOT a proof.  For rigorous bound, use
    solve_qp_shor instead.

    Returns (val, mu) where val is the inf restricted to symmetric mu.
    Note: val >= true inf, so val/int_g is *not* a lower bound on C_{1a}.
    """
    import cvxpy as cp
    d = G.shape[0]
    if d % 2 == 0:
        # Even d: pair up (0, d-1), (1, d-2), ..., free indices [0, d/2)
        m = d // 2
        z = cp.Variable(m, nonneg=True)
        # mu_i = z_i for i < m; mu_{d-1-i} = z_i.
        # sum mu = 2 * sum z = 1 -> sum z = 1/2.
        # mu^T G mu = sum_{i,j} G[i,j] mu_i mu_j
        # Build the reduced kernel: G_red[a, b] = G[a, b] + G[a, d-1-b]
        #                                       + G[d-1-a, b] + G[d-1-a, d-1-b]
        G_red = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                G_red[a, b] = (G[a, b] + G[a, d - 1 - b]
                               + G[d - 1 - a, b] + G[d - 1 - a, d - 1 - b])
        # Make symmetric and add small jitter for PSD (Bochner says it
        # *should* be PSD on this subspace; numerical rounding may give
        # tiny negatives).
        G_red = 0.5 * (G_red + G_red.T)
        eigs = np.linalg.eigvalsh(G_red)
        if eigs.min() < 0:
            G_red = G_red + (-eigs.min() + 1e-12) * np.eye(m)
            shift = -eigs.min() + 1e-12
        else:
            shift = 0.0
        constr = [cp.sum(z) == 0.5]
        obj = cp.Minimize(cp.quad_form(z, cp.psd_wrap(G_red)))
        prob = cp.Problem(obj, constr)
        prob.solve(solver=solver)
        val_shifted = float(prob.value)
        # Recover val without the shift by recomputing
        z_val = np.asarray(z.value).flatten()
        mu_val = np.zeros(d)
        mu_val[:m] = z_val
        mu_val[m:] = z_val[::-1]
        val = float(mu_val @ G @ mu_val)
        return val, mu_val
    else:
        # Odd d: middle index is fixed (mu_{d/2}); pair the others.
        m = (d - 1) // 2
        z = cp.Variable(m, nonneg=True)
        c = cp.Variable(nonneg=True)   # middle weight
        # sum mu = 2 sum z + c = 1.
        # Build constraint:
        constr = [2 * cp.sum(z) + c == 1]
        # Build reduced quadratic. mu_i for i<m is z_i; mu_m=c; mu_{d-1-i}=z_i.
        # Construct full mu in cvxpy via concatenation:
        full_mu = cp.hstack([z, cp.reshape(c, (1,)), z[::-1]])
        # Obj
        Gs = 0.5 * (G + G.T)
        # ensure PSD by shift
        eigs = np.linalg.eigvalsh(Gs)
        if eigs.min() < 0:
            shift_full = -eigs.min() + 1e-12
            Gs = Gs + shift_full * np.eye(d)
        else:
            shift_full = 0.0
        obj = cp.Minimize(cp.quad_form(full_mu, cp.psd_wrap(Gs)))
        prob = cp.Problem(obj, constr)
        prob.solve(solver=solver)
        z_val = np.asarray(z.value).flatten()
        c_val = float(c.value)
        mu_val = np.concatenate([z_val, [c_val], z_val[::-1]])
        val = float(mu_val @ G @ mu_val)
        return val, mu_val


# Backwards-compatible alias for tests/clients
solve_qp_convex = solve_qp_shor


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto Delta_d (sorted-pivot algorithm)."""
    d = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u - cssv / np.arange(1, d + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def solve_qp_frank_wolfe(G: np.ndarray, n_starts: int = 20,
                         max_iters: int = 1000,
                         seed: int = 0) -> Tuple[float, np.ndarray]:
    """Multi-restart Frank-Wolfe for non-PSD QP on Delta_d.

    Result is a *local* min; not a true min for non-convex G.  Use as a
    fast upper bound on the true infimum.
    """
    rng = np.random.default_rng(seed)
    d = G.shape[0]
    best_val = np.inf
    best_mu = None
    for s in range(n_starts):
        # Random Dirichlet start
        mu = rng.dirichlet(np.ones(d))
        prev_val = np.inf
        for it in range(max_iters):
            grad = 2 * (G @ mu)
            # Linear min on Delta_d: vertex with smallest grad
            j = int(np.argmin(grad))
            v = np.zeros(d)
            v[j] = 1.0
            # Line search (quadratic, exact)
            # f(mu + alpha (v - mu)) = (mu + alpha d)^T G (mu + alpha d)
            #   = mu^T G mu + 2 alpha d^T G mu + alpha^2 d^T G d
            # min in alpha in [0, 1]: alpha* = -(d^T G mu) / (d^T G d) if positive
            d_dir = v - mu
            num = -d_dir @ G @ mu
            den = d_dir @ G @ d_dir
            if den > 1e-15:
                alpha = max(0.0, min(1.0, num / den))
            else:
                alpha = 2.0 / (it + 2)
            mu_new = mu + alpha * d_dir
            mu_new = project_simplex(mu_new)
            val = mu_new @ G @ mu_new
            if abs(val - prev_val) < 1e-12:
                mu = mu_new
                break
            prev_val = val
            mu = mu_new
        if val < best_val:
            best_val = val
            best_mu = mu.copy()
    return float(best_val), best_mu


# ---------------------------------------------------------------------
# Top-level: compute M(g)
# ---------------------------------------------------------------------

def M_g(g_func: Callable[[np.ndarray], np.ndarray],
        d: int = 400,
        x_lo: float = -0.25, x_hi: float = 0.25,
        t_lo: float = -0.5, t_hi: float = 0.5,
        n_starts: int = 20,
        force_method: Optional[str] = None,
        verbose: bool = False) -> dict:
    """Compute M(g) numerically.

    Returns dict with:
      'M'           : the bound (numerator / int_g)
      'numerator'   : inf_mu mu^T G mu
      'int_g'       : int g
      'd'           : grid size
      'x'           : grid points
      'mu_star'     : argmin mu (a numerical witness)
      'method'      : 'convex' or 'frank-wolfe'
      'is_psd'      : whether G is PSD (Bochner-admissible)
      'min_eig_G'   : minimum eigenvalue of G

    force_method in {'convex', 'frank-wolfe'} overrides automatic choice.
    """
    x = make_x_grid(d, x_lo, x_hi)
    G = kernel_matrix(g_func, x)
    int_g = integrate_g(g_func, t_lo, t_hi)

    Gs = 0.5 * (G + G.T)
    eigs = np.linalg.eigvalsh(Gs)
    min_eig = float(eigs.min())
    is_psd = min_eig > -1e-9

    if force_method == 'convex' or (force_method is None and is_psd):
        method = 'convex'
        num, mu_star = solve_qp_convex(G)
    else:
        method = 'frank-wolfe'
        num, mu_star = solve_qp_frank_wolfe(G, n_starts=n_starts)

    if int_g <= 0:
        raise ValueError(f"int_g = {int_g}, must be > 0")

    M = num / int_g
    if verbose:
        print(f"  d={d}, int_g={int_g:.6f}, min_eig(G)={min_eig:.3e}, method={method}")
        print(f"  numerator={num:.6f}, M={M:.6f}")
    return dict(M=M, numerator=num, int_g=int_g, d=d, x=x,
                mu_star=mu_star, method=method, is_psd=is_psd,
                min_eig_G=min_eig)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("M(g) evaluator self-test")
    print("=" * 70)

    # Test 1: g identically zero -> int_g = 0 -> error.
    # Test 2: g = constant 1 on [-1/2, 1/2]:
    #   int g = 1.
    #   int int g(x+y) f f = 1 * (int f)^2 = 1 (since g==1 wherever x+y in
    #     supp(g), which contains all x+y in [-1/2, 1/2]).
    #   So M = 1.
    print("\nTest 1: g = 1 identically on [-1/2, 1/2]")
    g_const = lambda t: np.ones_like(t) if hasattr(t, 'shape') else 1.0
    res = M_g(g_const, d=200, force_method='convex', verbose=True)
    print(f"  expected M = 1.0, got M = {res['M']:.6f}")
    assert abs(res['M'] - 1.0) < 1e-3, f"M = {res['M']} != 1.0"

    # Test 3: g = indicator of [-h, h] for small h > 0:
    #   int g = 2h.
    #   int int g(x+y) f f = int int 1_{|x+y| <= h} f f dx dy
    #                      = (f * f) integrated over [-h, h]
    #   For uniform f on [-1/4, 1/4]: f = 2 on [-1/4, 1/4].
    #   f*f is a triangle: (f*f)(t) = 4 * (1/2 - |t|) for |t| <= 1/2.
    #   int_{-h}^{h} (f*f)(t) dt = int_{-h}^{h} 4(1/2 - |t|) dt
    #                             = 4 (h - h^2)
    #   So M = 4(h - h^2) / (2h) = 2(1 - h).
    #   For h -> 0: M -> 2.
    #   But also: f = uniform isn't necessarily the inf!
    #   inf_f int int g f f might be smaller.
    print("\nTest 2: g = 1_{|t| <= 0.05}")
    h = 0.05
    g_ind = lambda t: ((t >= -h) & (t <= h)).astype(np.float64) \
        if hasattr(t, 'shape') else (1.0 if -h <= t <= h else 0.0)
    res = M_g(g_ind, d=200, verbose=True)
    print(f"  M = {res['M']:.6f}")
    print(f"  is_psd = {res['is_psd']}")
    print(f"  upper bound (from uniform f) = {2 * (1 - h):.6f}")
    # Indicator kernel G[i,j] = 1_{|x_i+x_j| <= h} is generally NOT PSD,
    # but we still get a numeric M. Should be < 2(1-h).

    # Test 4: g = (1/2 - |t|)_+ (triangle, equal to (f_uniform * f_uniform) up to scale)
    # This is the autoconv of uniform_{[-1/4, 1/4]} (after scaling).
    # int g = (1/2)^2 = 1/4 ... let me actually compute.
    # int_{-1/2}^{1/2} (1/2 - |t|) dt = 2 * int_0^{1/2} (1/2 - t) dt = 2 * 1/8 = 1/4.
    # int int g(x+y) f f for f = uniform_{[-1/4, 1/4]} (f = 2):
    #   = int int (1/2 - |x+y|) * 4 dx dy ... over [-1/4, 1/4]^2
    #   = 4 * (int int (1/2 - |x+y|) dx dy)
    # We can compute that integral but skip; just check M numerically.
    print("\nTest 3: g = (1/2 - |t|)_+ (triangular)")
    g_tri = lambda t: np.maximum(0.5 - np.abs(t), 0)
    res = M_g(g_tri, d=200, verbose=True)
    print(f"  M = {res['M']:.6f}")
    # Triangular g has nonneg Fourier transform (it's a sinc^2). PSD!
    print(f"  is_psd = {res['is_psd']}")

    print("\nself-test OK")

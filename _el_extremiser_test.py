"""
Numerical test of EL stationarity + symmetry of approximate C_{1a} extremiser.

Discretise f as N-cell histogram on [-1/4, 1/4], cell width w=1/(2N), heights
v_i / w. Then (f*f) on cell-aligned grid gives a triangular convolution; in
particular, on the t-grid t_k = k * w/2 (or rather k * w on cell-mid grid),
(f*f)(t_k) = (1/w) * sum_i v_i * v_{k-i}  (when k is the integer cell offset
from the support range).

Concretely: define u_k = sum_{i+j=k} v_i v_j for k=0..2N-2, then for
integer-offset shifts the autoconvolution at the cell-boundary x = (k - N + 1) * w
equals u_k / w. (Continuous (f*f) is piecewise-linear between these knots.)

Convex problem: min M s.t. u_k / w <= M for all k, u_k = (v*v)_k, v >= 0,
sum v_i = 1. This is non-convex in v BUT becomes a QCQP. We instead solve it
via a convex moment relaxation by optimising directly with scipy SLSQP and
use multiple random starts.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, linprog

def conv_v(v):
    """u_k = sum_{i+j=k} v_i v_j = (v*v)_k for k = 0..2N-2."""
    return np.convolve(v, v)

def piecewise_linear_max_ff(v, w):
    """Continuous (f*f)(t) for f a step function with heights v/w on cells.
    The continuous autoconv is piecewise-linear with knots at t = (k - (N-1)) * w
    for k = 0..2N-2 (cell-boundary knots), and the values at knots are (v*v)_k / w.
    Wait: more carefully, with f piecewise-constant on cells of width w, f*f
    is piecewise-linear on intervals of width w. The maximum is attained at a
    knot. So max (f*f) = max_k (v*v)_k / w.
    """
    return conv_v(v).max() / w

def M_value(v, w):
    return piecewise_linear_max_ff(v, w)

def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho_idx = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if len(rho_idx) == 0:
        return np.maximum(v, 0) / max(np.sum(np.maximum(v, 0)), 1e-30)
    rho = rho_idx[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)

def optimise_lp_iterate(N, n_iter=500, lr=0.01, seed=0):
    """Frank-Wolfe / projected subgradient on max_k (v*v)_k / w."""
    w = 1.0 / (2 * N)
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.5, 1.5, N)
    v = project_simplex(v)
    for k in range(n_iter):
        u = conv_v(v)
        # Smoothed: use logsumexp gradient with growing beta
        beta = 50 + 200 * (k / n_iter)
        u_max = u.max()
        e = np.exp(beta * (u - u_max))
        p = e / e.sum()
        # gradient of sum_k p_k * (v*v)_k w.r.t. v_i = 2 * sum_k p_k * v_{k-i}
        grad = 2 * np.convolve(p, v[::-1])  # cross-correlation; produces full length 2N-1+N-1 = 3N-2
        # Need indices such that sum_k p_k * v_{k-i} for i=0..N-1
        # np.convolve(p, v[::-1]) at output index m corresponds to sum_n p[n] * v[::-1][m-n] = sum_n p[n] * v[N-1-(m-n)]
        # We want sum_k p_k * v_{k-i}, i.e. let v_{k-i} index. With p of length 2N-1 and v of length N, full conv has length 3N-2.
        # Use np.correlate(p, v, 'full'): c[m] = sum_n p[n+m-(N-1)] * v[n]; for m in [0, 3N-3]
        # Equivalently c[m] = sum_n p[n + m - (N-1)] v[n]. We want sum_k p[k] v[k - i] = sum_n p[n+i] v[n] (k = n+i).
        # So m - (N-1) = i, i.e. m = N-1+i. Then grad[i] = 2 * c[N-1+i].
        c = np.correlate(p, v, mode='full')
        grad = np.array([2 * c[N - 1 + i] for i in range(N)])
        v = project_simplex(v - lr * grad)
    return v, w, M_value(v, w)

def optimise_scipy(N, seed=0, multistart=5):
    """Minimise max_k (v*v)_k / w using SLSQP. Use t-variable LP-style:
    min M s.t. (v*v)_k <= M*w for all k, sum v = 1, v >= 0.
    Decision vars: x = (v_0, ..., v_{N-1}, M).
    Quadratic constraints (v*v)_k - w*M <= 0.
    """
    w = 1.0 / (2 * N)

    def cons_qp(x, k):
        v = x[:N]
        M = x[N]
        return w * M - np.convolve(v, v)[k]

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - 1.0}]
    for k in range(2 * N - 1):
        cons.append({'type': 'ineq', 'fun': cons_qp, 'args': (k,)})
    bounds = [(0, None)] * N + [(0, None)]

    rng = np.random.default_rng(seed)
    best_M = np.inf
    best_v = None
    for s in range(multistart):
        x0 = np.concatenate([rng.uniform(0.5, 2.0, N), [3.0]])
        x0[:N] = project_simplex(x0[:N])
        x0[N] = M_value(x0[:N], w) * 1.01
        res = minimize(lambda x: x[N], x0, method='SLSQP', constraints=cons,
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success and res.x[N] < best_M:
            v = res.x[:N]
            v = np.maximum(v, 0)
            v /= v.sum()
            M = M_value(v, w)
            if M < best_M:
                best_M = M
                best_v = v
    return best_v, w, best_M

def symmetry_score(v):
    return np.max(np.abs(v - v[::-1])) / max(v.max(), 1e-30)

def active_set(v, w, tol=2e-3):
    u = conv_v(v) / w
    M = u.max()
    mask = u >= M * (1 - tol)
    # knot positions: (f*f) has knots at t = (k - (N-1)) * w for k = 0..2N-2
    N = len(v)
    t_knots = (np.arange(2 * N - 1) - (N - 1)) * w
    return t_knots[mask], u[mask], M

def el_residual_atom(v, w, t_atoms, sigma_w):
    """Compute (f * sigma)(x_i) on the cells x_i (cell midpoints) of supp f.
    f piecewise-constant on cells of width w with height v_i/w.
    (f*sigma)(x) = sum_a sigma_w[a] * f(t_atoms[a] - x).
    """
    N = len(v)
    # cell midpoints in [-1/4, 1/4]
    x_mid = -1/4 + w * (np.arange(N) + 0.5)
    out = np.zeros(N)
    for ta, wa in zip(t_atoms, sigma_w):
        # f(ta - x_mid): which cell?
        y = ta - x_mid  # array of length N
        idx = np.floor((y + 1/4) / w).astype(int)
        valid = (idx >= 0) & (idx < N)
        contrib = np.zeros(N)
        contrib[valid] = v[idx[valid]] / w
        out += wa * contrib
    # Restrict to supp f: cells where v > eps
    supp = v > 1e-7 * v.max()
    return x_mid[supp], out[supp], v[supp]

if __name__ == "__main__":
    print("="*72)
    print("EL test: minimise ||f*f||_inf over step-function f with N pieces.")
    print("Reference: C_{1a} ~ 1.2802 (lower bound, CS 2017); upper bound 1.5029.")
    print("="*72)
    for N in [4, 6, 8, 12, 16, 24]:
        v, w, M = optimise_scipy(N, multistart=12)
        if v is None:
            print(f"N={N}: SLSQP failed; skip")
            continue
        sym = symmetry_score(v)
        t_act, g_act, _ = active_set(v, w, tol=3e-3)
        # Symmetrised candidate
        v_sym = 0.5 * (v + v[::-1])
        v_sym /= v_sym.sum()
        M_sym = M_value(v_sym, w)
        print(f"\nN={N:3d}: M = {M:.5f}  sym_score = {sym:.4f}  M_symmetrised = {M_sym:.5f}")
        with np.printoptions(precision=3, suppress=True):
            print(f"  v        = {v}")
            print(f"  v_rev    = {v[::-1]}")
            print(f"  active t = {np.round(t_act, 4)} (count {len(t_act)})")
        # EL test: try sigma = uniform on active knots
        if len(t_act) >= 1:
            sw = np.ones(len(t_act)) / len(t_act)
            x_supp, fs_vals, v_supp = el_residual_atom(v, w, t_act, sw)
            mean = fs_vals.mean()
            rel_std = fs_vals.std() / max(abs(mean), 1e-12)
            print(f"  (f*sigma)|supp f: mean={mean:.4f}, rel std={rel_std:.4f}")

    print("\n" + "="*72)
    print("Symmetric subclass: same problem with v_i = v_{N-1-i} forced.")
    print("Expected reference: M_sym >= 1.42401 by Path A symmetric theorem.")
    print("="*72)

    def optimise_symmetric(N, multistart=8, seed=0):
        w = 1.0 / (2 * N)
        # half-space v has dim ceil(N/2)
        half = (N + 1) // 2
        def expand(z):
            v = np.zeros(N)
            v[:half] = z
            v[N - half:] = z[::-1] if N % 2 == 0 else np.concatenate([z, z[-2::-1]])[N-half:]
            # easier: build by symmetry
            v = np.zeros(N)
            for i in range(N):
                v[i] = z[min(i, N - 1 - i)]
            return v
        def neg_grad(z):
            v = expand(z)
            return M_value(v, w)
        rng = np.random.default_rng(seed)
        best_M = np.inf
        best_v = None
        for s in range(multistart):
            x0 = np.concatenate([rng.uniform(0.5, 2.0, half), [3.0]])
            x0_v = expand(x0[:half]); x0_v = x0_v / x0_v.sum()
            x0[:half] = x0_v[:half]
            x0[half] = M_value(x0_v, w) * 1.01
            def cons_qp(x, k):
                z = x[:half]
                v = expand(z)
                M = x[half]
                return w * M - np.convolve(v, v)[k]
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(expand(x[:half])) - 1.0}]
            for k in range(2 * N - 1):
                cons.append({'type': 'ineq', 'fun': cons_qp, 'args': (k,)})
            bounds = [(0, None)] * half + [(0, None)]
            res = minimize(lambda x: x[half], x0, method='SLSQP', constraints=cons,
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                z = res.x[:half]; v = expand(z); v = np.maximum(v, 0); v /= v.sum()
                M = M_value(v, w)
                if M < best_M:
                    best_M = M; best_v = v
        return best_v, w, best_M

    for N in [4, 6, 8, 12, 16, 24]:
        v, w, M = optimise_symmetric(N, multistart=12)
        if v is None: continue
        with np.printoptions(precision=3, suppress=True):
            print(f"N={N}: M_sym = {M:.5f}  v = {v}")

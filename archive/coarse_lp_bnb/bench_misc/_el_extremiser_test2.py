"""
Extension: also test EL on N=20-32 with better solver, and check whether the
symmetric subclass extremiser (post-rearrangement) is structurally different
from the general extremiser.

Key check: compare M_general (asymmetric class) vs M_symmetric (forced even).
If M_general < M_symmetric strictly, the extremiser cannot be even.
"""
import numpy as np
from scipy.optimize import minimize

def conv_v(v):
    return np.convolve(v, v)

def M_value(v, w):
    return conv_v(v).max() / w

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

def opt_general(N, multistart=20, seed=0):
    """min M s.t. (v*v)_k <= w*M, v >= 0, sum v = 1."""
    w = 1.0 / (2 * N)
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for s in range(multistart):
        # diverse seeds
        if s == 0:
            v0 = np.ones(N) / N  # indicator
        elif s == 1:
            v0 = np.linspace(0.1, 1, N); v0 /= v0.sum()  # ramp
        elif s == 2:
            v0 = np.exp(-((np.arange(N) - N/2) / (N/4))**2); v0 /= v0.sum()  # Gaussian
        else:
            v0 = rng.uniform(0, 2, N); v0 = project_simplex(v0)

        x0 = np.concatenate([v0, [M_value(v0, w) * 1.01]])

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - 1.0}]
        for k in range(2 * N - 1):
            def cf(x, kk=k):
                return w * x[N] - np.convolve(x[:N], x[:N])[kk]
            cons.append({'type': 'ineq', 'fun': cf})
        bounds = [(0, None)] * N + [(0, None)]
        try:
            res = minimize(lambda x: x[N], x0, method='SLSQP', constraints=cons,
                           bounds=bounds, options={'maxiter': 800, 'ftol': 1e-11})
            v = np.maximum(res.x[:N], 0); s_v = v.sum()
            if s_v > 1e-9:
                v /= s_v
                M = M_value(v, w)
                if M < best[0]:
                    best = (M, v)
        except Exception:
            pass
    return best[1], w, best[0]

def opt_symmetric(N, multistart=20, seed=0):
    """Force v_i = v_{N-1-i}."""
    w = 1.0 / (2 * N)
    half = (N + 1) // 2
    rng = np.random.default_rng(seed)
    best = (np.inf, None)

    def expand(z):
        v = np.zeros(N)
        for i in range(N):
            v[i] = z[min(i, N - 1 - i)]
        return v

    for s in range(multistart):
        if s == 0:
            z0 = np.ones(half)
        elif s == 1:
            z0 = np.linspace(0.1, 1, half)
        elif s == 2:
            z0 = np.linspace(1, 0.1, half)
        elif s == 3:
            # Two-spike symmetric guess
            z0 = np.zeros(half); z0[0] = 1.0; z0[-1] = 1.0
        else:
            z0 = rng.uniform(0, 2, half)
        v0 = expand(z0); v0 = v0 / v0.sum()
        z0 = v0[:half]
        x0 = np.concatenate([z0, [M_value(v0, w) * 1.01]])

        cons = [{'type': 'eq', 'fun': lambda x: np.sum(expand(x[:half])) - 1.0}]
        for k in range(2 * N - 1):
            def cf(x, kk=k):
                v = expand(x[:half])
                return w * x[half] - np.convolve(v, v)[kk]
            cons.append({'type': 'ineq', 'fun': cf})
        bounds = [(0, None)] * half + [(0, None)]
        try:
            res = minimize(lambda x: x[half], x0, method='SLSQP', constraints=cons,
                           bounds=bounds, options={'maxiter': 800, 'ftol': 1e-11})
            z = np.maximum(res.x[:half], 0); v = expand(z); s_v = v.sum()
            if s_v > 1e-9:
                v /= s_v
                M = M_value(v, w)
                if M < best[0]:
                    best = (M, v)
        except Exception:
            pass
    return best[1], w, best[0]

print("="*72)
print("Comparison: M_general vs M_symmetric for varying N")
print("Reference: C_{1a} known >= 1.2802 (CS 2017), upper 1.5029")
print("Path A: symmetric class >= 1.42401")
print("="*72)
print(f"{'N':>4} | {'M_general':>10} | {'M_sym':>10} | {'gap M_sym - M_gen':>20}")
print("-" * 60)
for N in [4, 6, 8, 10, 12, 16, 20, 24, 32]:
    vg, w, Mg = opt_general(N, multistart=15)
    vs, w2, Ms = opt_symmetric(N, multistart=15)
    print(f"{N:>4} | {Mg:10.5f} | {Ms:10.5f} | {Ms - Mg:>20.5f}")
    # Spot-check: how close is general optimum to being even?
    if vg is not None:
        sym_score = np.max(np.abs(vg - vg[::-1])) / max(vg.max(), 1e-30)
        print(f"       general v sym_score = {sym_score:.4f}; nonzero cells = {(vg > 1e-6).sum()}")

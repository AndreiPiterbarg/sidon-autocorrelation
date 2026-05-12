"""Debug the E-L extremizer search."""
import numpy as np
from el_extremizer import autoconv, autocorr, stats, project, grad_J, el_search, C_TARGET

# Sanity: indicator f = 2 * 1_{[-1/4,1/4]}, expected M=2, c_emp=2/3
N = 121
dx = 0.5/(N-1)
f = 2.0 * np.ones(N)
M, L2sq, L1, g = stats(f, dx)
print(f"Indicator f=2 on [-1/4,1/4]: M={M:.4f}  L2sq={L2sq:.4f}  L1={L1:.4f}  c_emp={L2sq/M:.4f}")
print(f"  expected M=2, L1=1, c_emp = 2/3 = {2/3:.4f}")

# Sanity: triangle-ish from MV: narrow Gaussian
x = np.linspace(-0.25, 0.25, N)
sigma = 0.05
f = np.exp(-x*x/(2*sigma*sigma))
f = project(f, dx)
M, L2sq, L1, g = stats(f, dx)
print(f"Gaussian sigma=0.05: M={M:.4f}  c_emp={L2sq/M:.4f}  L1={L1:.4f}")

# Test grad_J: compare numerical vs analytic
f0 = project(np.exp(-x*x/0.01), dx)
M0, L2_0, _, _ = stats(f0, dx)
J0 = L2_0
g_an = grad_J(f0, dx)
# Numerical
eps = 1e-6
ig = N // 2
fp = f0.copy()
fp[ig] += eps
fp = project(fp, dx)  # warning: projection skews finite-diff
# better: don't project, just check raw J = sum(autoconv(f)^2)*dx
def Jraw(ff):
    g = autoconv(ff, dx)
    return float((g*g).sum()*dx)
J_plus = Jraw(f0 + eps*np.eye(N)[ig])
J_minus = Jraw(f0 - eps*np.eye(N)[ig])
g_num_ig = (J_plus - J_minus) / (2*eps)
print(f"grad_J check: analytic[{ig}]={g_an[ig]:.4f}  numerical={g_num_ig:.4f}")

# Try a simpler search with more debug
print("\n=== Simpler search: small N, sym, M_cap=1.5 ===")
def simple_search(N, M_cap, n_steps, lr, alpha_pen, seed, sym=True):
    rng = np.random.default_rng(seed)
    dx = 0.5/(N-1)
    x = np.linspace(-0.25, 0.25, N)
    sigma = 0.04 + 0.02*rng.uniform()
    f = np.exp(-x*x/(2*sigma*sigma))
    if sym: f = 0.5*(f+f[::-1])
    f = project(f, dx)
    M0, L2_0, _, _ = stats(f, dx)
    print(f"  init: M={M0:.4f}  c_emp={L2_0/M0:.4f}")
    for it in range(n_steps):
        M, L2sq, _, g = stats(f, dx)
        gJ = grad_J(f, dx)
        # penalty grad
        from scipy.signal import fftconvolve
        excess = np.maximum(g - M_cap, 0.0)
        full = fftconvolve(excess, f[::-1]) * dx
        gP = 4.0 * full[N-1:N-1+N]
        grad = gJ - alpha_pen * gP
        f = f + lr * grad
        if sym: f = 0.5*(f+f[::-1])
        f = project(f, dx)
        if it % 500 == 0:
            print(f"  it={it} M={M:.4f}  c_emp={L2sq/M:.4f}  ||grad||={np.linalg.norm(grad):.2e}")
    M, L2sq, L1, g = stats(f, dx)
    print(f"  FINAL: M={M:.4f}  c_emp={L2sq/M:.4f}  L1={L1:.4f}")
    return f, M, L2sq

simple_search(101, 1.5, 3000, 0.01, 200.0, 1, sym=True)
print("\n=== try M_cap=1.378 ===")
simple_search(101, 1.378, 3000, 0.01, 500.0, 1, sym=True)
print("\n=== try M_cap=2 (Holder ceiling) ===")
simple_search(101, 2.0, 3000, 0.01, 100.0, 1, sym=True)
print("\n=== try M_cap=1.378 narrow init ===")
simple_search(101, 1.378, 3000, 0.005, 1000.0, 5, sym=True)

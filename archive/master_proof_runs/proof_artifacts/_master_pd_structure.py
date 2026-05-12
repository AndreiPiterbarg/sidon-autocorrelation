"""
Analyze the minimizer g* for the restricted problem and extrapolate to continuum.
Also test some closed-form candidates and a higher-resolution run.
"""
import numpy as np
from _master_pd_compute import optimize, ratio, conv_full

# Closed-form candidates on [0, 1/2]:
def candidate_indicator(N, L=0.5):
    """g = 1 on [0, 1/2]."""
    dx = L / N
    g = np.ones(N)
    g *= 1.0 / (g.sum() * dx)
    return g, dx

def candidate_triangle_full(N, L=0.5):
    """g = triangle peaked at 1/4."""
    dx = L / N
    x = np.linspace(0, 1, N)
    g = 1 - np.abs(2*x - 1)
    g = np.maximum(g, 1e-12)
    g *= 1.0 / (g.sum() * dx)
    return g, dx

def candidate_halfcosine(N, L=0.5):
    """g = cos(pi*(x-1/4)/.5) on [0, 1/2]."""
    dx = L / N
    x = np.linspace(0, L, N)
    g = np.cos(np.pi*(x - L/2)/L)
    g = np.maximum(g, 0)
    g *= 1.0 / (g.sum() * dx)
    return g, dx

def candidate_raisedcos(N, L=0.5, alpha=2):
    """g = (1 - cos(2pi x/L))^alpha-ish."""
    dx = L / N
    x = np.linspace(0, L, N)
    g = np.sin(np.pi*x/L)**alpha
    g *= 1.0 / (g.sum() * dx)
    return g, dx

candidates = {
    "indicator": candidate_indicator,
    "triangle":  candidate_triangle_full,
    "halfcos":   candidate_halfcosine,
    "sin^2":     lambda N,L=0.5: candidate_raisedcos(N,L,2),
    "sin^4":     lambda N,L=0.5: candidate_raisedcos(N,L,4),
}

print("Closed-form candidates R(g) for restricted problem:")
N = 400
for name, fn in candidates.items():
    g, dx = fn(N)
    R = ratio(g, dx)
    print(f"  {name:12s}  R = {R:.6f}")

# Extrapolation: finer N
print("\nConvergence with N (uniform init, projected GD):")
Rs = {}
for N in [100, 200, 400, 800]:
    R, g_opt, _ = optimize(N=N, L=0.5, n_iter=8000, lr=0.01, seed=2, init='uniform')
    print(f"  N = {N:5d}   R* = {R:.7f}")
    Rs[N] = R

# Richardson-like extrapolation: assume R(N) = R_inf + c/N
import numpy as np
Ns = np.array(list(Rs.keys()))
Rvals = np.array([Rs[n] for n in Ns])
A = np.column_stack([np.ones_like(Ns, dtype=float), 1.0/Ns])
coef, *_ = np.linalg.lstsq(A, Rvals, rcond=None)
R_inf = coef[0]
print(f"\nExtrapolated R_inf (continuum): {R_inf:.6f}")

# Save best g shape (normalized) so we know what extremizer looks like
import json
best_N = max(Rs)
R_best, g_best, _ = optimize(N=best_N, L=0.5, n_iter=12000, lr=0.005, seed=2, init='uniform')
# Save shape (coarsened to 50 pts)
idx = np.linspace(0, best_N-1, 50).astype(int)
shape = g_best[idx] / g_best.max()
out = {
    "R_per_N": Rs,
    "R_inf_extrapolated": float(R_inf),
    "g_shape_sample_50pts (x in [0,0.5])": shape.tolist(),
    "g_max_location_frac": float(np.argmax(g_best)/best_N),
}
with open(r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_master_pd_structure.json","w") as f:
    json.dump(out, f, indent=2)
print("Saved _master_pd_structure.json")
print(f"\nLocation of max(g*) at fraction of [0,1/2]: {np.argmax(g_best)/best_N:.3f}")

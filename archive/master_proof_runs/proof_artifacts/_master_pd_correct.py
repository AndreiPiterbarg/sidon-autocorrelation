"""
CORRECT-normalization restricted C_{1a} computation.

Problem: f >= 0, supp f subset [-1/4, 1/4], int f = 1, minimize sup_{|t|<=1/2}(f*f)(t).
Restrict to PD f: f = g * g_tilde, g >= 0, supp g subset [-1/8, 1/8], ||g||_1 = 1.
For PD f, sup(f*f) = (f*f)(0) = ||f||_2^2 = ||g*g_tilde||_2^2.

So restricted C_{1a}^{PD} = inf_{g >= 0, supp g <= [-1/8,1/8], ||g||_1=1}  ||g*g_tilde||_2^2.

Equivalently parametrize g on [0, 1/4] (translate; only ratio matters via translation
invariance of autocorrelation), width L = 1/4.
"""
import numpy as np
from _master_pd_compute import optimize, ratio, conv_full

# Sanity: indicator f = 2 * 1_[-1/4,1/4], integrate to 1, gives (f*f)(0) = 2.
# In our g-parametrization: g = c * 1_{[0,L]} on width L=1/4. ||g||_1 = c*L = 1, so c = 1/L = 4.
# f = g*g_tilde is triangle of width 2L = 1/2 centered at 0, peak c^2 * L = 16/4 = 4.
# Then ||f||_2^2 = int triangle^2 ... let's compute directly.
L = 0.25
N = 1000
dx = L/N
g = np.ones(N) * (1.0 / L)  # ||g||_1 = 1
A = conv_full(g, g[::-1], dx)
print(f"Indicator-g: ||g||_1 = {g.sum()*dx:.4f}, ||A||_2^2 = {(A*A).sum()*dx:.6f}")
print(f"This should equal restricted ratio = ||f||_2^2 with ||f||_1=1.\n")

# Indicator gives 2.66... in this normalization? Compute analytically:
# g = 4 on [0, 1/4], 0 elsewhere. A(t) = (g*g_tilde)(t) = int g(s)g(s-t)ds for |t|<=1/4
#   = 16 * (1/4 - |t|) for |t|<=1/4.
# ||A||_2^2 = 2 * int_0^{1/4} (16(1/4-t))^2 dt = 2 * 256 * (1/4)^3/3 = 2*256/192 = 256/96 = 8/3.
print(f"Analytic ||A||_2^2 for indicator: 8/3 = {8/3:.6f}")
print()

# Now optimize with correct L=1/4
print("=== Optimizing restricted C_1a^PD with correct normalization ===")
print("(L = 1/4 for g support, ||g||_1 = 1, minimize ||g*g_tilde||_2^2)\n")

results = {}
for N in [100, 200, 400, 800, 1600]:
    R, g_opt, _ = optimize(N=N, L=0.25, n_iter=8000, lr=0.005, seed=1, init='uniform')
    print(f"  N = {N:5d}   R* = {R:.7f}")
    results[N] = R

# Extrapolate
Ns = np.array(sorted(results.keys()))
Rvals = np.array([results[n] for n in Ns])
A = np.column_stack([np.ones_like(Ns, dtype=float), 1.0/Ns])
coef, *_ = np.linalg.lstsq(A, Rvals, rcond=None)
R_inf = coef[0]
print(f"\nExtrapolated continuum: C_1a^PD = {R_inf:.6f}")
print(f"Unrestricted MV LB: 1.2748")
print(f"CS17 UB (unrestricted): 1.5029")

if R_inf > 1.5029:
    print(f"\n=> Restricted UB ({R_inf:.4f}) is ABOVE the unrestricted UB ({1.5029}).")
    print("   Tells us extremizer is non-PD (PD subclass costs more).")
elif R_inf > 1.2748:
    print(f"\n=> Restricted UB ({R_inf:.4f}) lies in [MV_LB, CS17_UB]. Consistent.")
    print("   The PD class achieves something between MV and CS17.")
else:
    print(f"\n=> CONTRADICTION: restricted infimum ({R_inf:.4f}) below MV LB ({1.2748}).")
    print("   This would refute MV — investigate!")

# Save
import json
out = {
    "definition": "f >= 0, supp f subset [-1/4,1/4], int f = 1, sup_{|t|<=1/2}(f*f)",
    "restricted_class": "additionally hat f >= 0; parametrize f = g*g_tilde, g>=0, supp g subset [-1/8,1/8]",
    "for_PD_f_sup_equals": "(f*f)(0) = ||f||_2^2 = ||g*g_tilde||_2^2 with ||g||_1=1",
    "R_per_N": {int(k):float(v) for k,v in results.items()},
    "C1a_PD_continuum": float(R_inf),
    "indicator_g_value": float((A*A).sum()*dx if False else 8/3),  # 2.6667
    "MV_unrestricted_LB": 1.2748,
    "CS17_unrestricted_UB": 1.5029,
}
with open(r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_master_pd_correct.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSaved _master_pd_correct.json")

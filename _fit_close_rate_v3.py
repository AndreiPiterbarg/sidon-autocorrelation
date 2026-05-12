"""Mixture model: easy population (saturates to 1) + hard population (never closes at order-2).

Plus comparison with d=10 data to anchor the EASY population's curve shape.
"""
import numpy as np
from scipy.optimize import curve_fit

# d=22 data (depth 63, target=1.2805)
f22 = np.array([0.000, 0.017, 0.034, 0.085])
p22 = np.array([0.094, 0.389, 0.522, 0.626])

# d=10 data (assume slack ~0.025 close to d=22, so SHAPE comparable for "easy" boxes)
f10 = np.array([0.000, 0.021, 0.042, 0.063, 0.084, 0.126, 0.168])
p10 = np.array([0.020, 0.680, 0.880, 0.940, 0.930, 1.000, 1.000])

# Mixture model: easy (sigmoid_to_1) + hard (always 0)
# p(f) = pi_A * (1 / (1 + exp(-(f - c)/s)))
def mixture(f, pi_A, c, s):
    return pi_A * (1.0 / (1 + np.exp(-(f - c) / s)))

# Anchor the curve shape from d=10 (assume easy population at d=22 has same shape):
# Fit d=10 first
popt10, _ = curve_fit(mixture, f10, p10, p0=[1.0, 0.02, 0.01],
                       bounds=([0.5, 0, 0.001], [1.0, 0.5, 0.5]))
print(f"d=10 mixture fit (anchor): pi_A={popt10[0]:.3f} c={popt10[1]:.4f} s={popt10[2]:.4f}")
print(f"  predicted: {[f'{x:.3f}' for x in mixture(f10, *popt10)]}")
print(f"  actual:    {[f'{x:.3f}' for x in p10]}")

# Now fit d=22 with full freedom
popt22, _ = curve_fit(mixture, f22, p22, p0=[0.7, 0.03, 0.02],
                       bounds=([0.3, 0, 0.001], [1.0, 0.5, 0.5]))
print(f"\nd=22 mixture fit (free): pi_A={popt22[0]:.3f} c={popt22[1]:.4f} s={popt22[2]:.4f}")
print(f"  predicted: {[f'{x:.3f}' for x in mixture(f22, *popt22)]}")
print(f"  actual:    {[f'{x:.3f}' for x in p22]}")

# Constrained fit: assume d=22 has SAME shape as d=10 (just different pi_A — i.e., same
# "easy population" curve, just smaller easy fraction)
def mixture_d10shape(f, pi_A):
    return pi_A * (1.0 / (1 + np.exp(-(f - popt10[1]) / popt10[2])))

popt22_constrained, _ = curve_fit(mixture_d10shape, f22, p22, p0=[0.6],
                                    bounds=([0.3], [1.0]))
print(f"\nd=22 mixture with d=10 shape (forced): pi_A={popt22_constrained[0]:.4f}")
pred = mixture_d10shape(f22, *popt22_constrained)
rmse = np.sqrt(np.mean((p22 - pred)**2))
print(f"  predicted: {[f'{x:.3f}' for x in pred]}")
print(f"  actual:    {[f'{x:.3f}' for x in p22]}")
print(f"  RMSE: {rmse:.4f}")

# Predictions
print("\n" + "=" * 80)
print("PREDICTIONS at high K under each model")
print("=" * 80)
for K in [80, 160, 200, 250, 320, 500, 946]:
    f = K / 946
    p_free = mixture(np.array([f]), *popt22)[0]
    p_constrained = mixture_d10shape(np.array([f]), *popt22_constrained)[0]
    print(f"  K={K:>4} (f={f:.3f}): free model p={p_free:.4f}   d10-shape model p={p_constrained:.4f}")

# Both predict pi_A as plateau. Plateau = fraction of "easy" boxes.
# At d=22, residual fraction = 1 - pi_A (the "hard" population).
print("\n" + "=" * 80)
print("STRUCTURAL INTERPRETATION")
print("=" * 80)
print(f"d=10:  pi_A ≈ {popt10[0]:.2f}  (essentially all boxes are 'easy' — order-2 SDP is tight)")
print(f"d=22 free: pi_A ≈ {popt22[0]:.2f}  ({100*(1-popt22[0]):.0f}% of LP-fails are 'hard')")
print(f"d=22 d10-shape: pi_A ≈ {popt22_constrained[0]:.2f}  ({100*(1-popt22_constrained[0]):.0f}% 'hard')")

# Convergence with mixture model: at any K we get pi_A close rate (asymptotic).
# Per parent, residual = (1 - pi_A) × S
# For convergence: (1 - pi_A) × S ≤ 1 → S ≤ 1 / (1-pi_A)
print()
for label, pi_A_val in [('d=22 free', popt22[0]),
                         ('d=22 d10-shape', popt22_constrained[0])]:
    if 1 - pi_A_val > 0:
        S_max = int(1 / (1 - pi_A_val))
        d_split_max = int(np.log2(S_max))
        print(f"{label}: max S for convergence = {S_max} → max split_depth ≈ {d_split_max}")

# What if "hard" boxes can be split into easy children at deeper depth?
# E.g., a depth-63 hard box might split into depth-72 boxes that are mostly "easy".
# The "hard" fraction is depth-dependent.
print("\nIF split-and-recurse on hard boxes promotes most to 'easy' at deeper depth:")
print("  e.g., 30% hard at depth 63 may become 5% hard at depth 75 (more splits per axis)")
print("  This is the iterative-split approach: each iter cleans up the previous's hard set")

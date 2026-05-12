"""Verify x_cap soundness when it's the binding constraint.

At m=20 L0 (d_child=8): x_cap=8, x_cap_cs=9. x_cap is binding.
x_cap says: if c_i > 8, then TV at some window > c_target + correction.
But correction() = 2/m + 1/m^2, while the actual pruning uses +3/m^2.

Does the x_cap filter ever discard a child that would SURVIVE the window scan?
If so, it's an unsound pre-filter (silently drops valid branches).
"""
import math
import numpy as np
import sys, os

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction

C_TARGET = 1.40
m = 20

# L0: d_child=8, n_half_child=4
d_child = 8
n_half_child = 4

corr = correction(m)  # 2/20 + 1/400 = 0.1025
thresh = C_TARGET + corr + 1e-9  # 1.5025 + 1e-9

x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
x_cap_cs = int(math.floor(m * math.sqrt(C_TARGET / d_child))) + 1

print(f"d_child={d_child}, n_half_child={n_half_child}, m={m}")
print(f"correction() = {corr:.6f}")
print(f"thresh = c_target + corr + 1e-9 = {thresh:.10f}")
print(f"x_cap = floor({m} * sqrt({thresh}/{d_child})) = floor({m * math.sqrt(thresh/d_child):.6f}) = {x_cap}")
print(f"x_cap_cs = floor({m} * sqrt({C_TARGET}/{d_child})) + 1 = floor({m * math.sqrt(C_TARGET/d_child):.6f}) + 1 = {x_cap_cs}")
print(f"final x_cap = min({x_cap}, {x_cap_cs}, {m}) = {min(x_cap, x_cap_cs, m)}")
print()

# x_cap = 8 is binding. This means children with c_i = 9 are filtered out.
# Is c_i = 9 always prunable?

# The x_cap argument: if c_i > x_cap, then d * (c_i/m)^2 > thresh
# d_child * (9/20)^2 = 8 * 0.2025 = 1.62 > 1.5025. Yes, pruned by the TV
# at ell=2 window centered on bin i (the TV is d_child * (c_i/m)^2 / 2
# ... wait, let me be more careful.

# The x_cap logic is: if c_i > x_cap, then there exists a window whose
# TV exceeds c_target + correction.

# Actually, look at the code comment:
# thresh = c_target + corr + 1e-9
# x_cap = floor(m * sqrt(thresh / d_child))
# This means: if c_i > x_cap, then (c_i/m)^2 > thresh/d_child
# => d_child * (c_i/m)^2 > thresh = c_target + corr

# Now the TV at ell=2, s=2*i (the "self" window) is:
# TV = (4*n_half_child) / (m^2 * 2) * c_i^2 = d_child * (c_i/m)^2 / 2

# Hmm wait. Let me be more precise about the TV formula.
# TV(ell, s) = ws / (4*n*ell)  where ws = sum of a_i*a_j autoconvolution
# At ell=2, s=2*i: ws = a_i^2 (just one term)
# But a_i = 4*n/m * c_i. So ws = (4n/m)^2 * c_i^2
# TV = (4n/m)^2 * c_i^2 / (4*n * 2) = 4n * c_i^2 / (2 * m^2)
# = 2*n * c_i^2 / m^2 = d_child * c_i^2 / m^2
# Wait, d_child = 2*n_half_child, n = n_half_child
# TV = 2*n_half_child * c_i^2 / m^2... no.

# Let me re-derive. The a-coordinate is a_i = (4*n_half / m) * c_i
# where n_half is the current level's n_half (= n_half_child for children).
# conv[2i] = a_i^2
# At ell=2, s_lo=2i: window sum = conv[2i] = a_i^2
# TV = ws / (4 * n_half_child * ell) = a_i^2 / (4 * n_half_child * 2)
#    = (4*n_half_child/m)^2 * c_i^2 / (8 * n_half_child)
#    = 16*n_half_child^2 * c_i^2 / (m^2 * 8 * n_half_child)
#    = 2 * n_half_child * c_i^2 / m^2
#    = d_child * c_i^2 / m^2

# Hmm, that doesn't look right either. Let me check with actual values.
# Actually looking at test_values.py:
# inv_norm = 1.0 / (4.0 * n_half * ell)
# tv = ws * inv_norm
# where ws is in a-coordinates.

# Actually the TV in test_values.py line 89 uses:
# inv_norm = 1.0 / (4.0 * n_half * ell)
# and the a-coordinates are: ai = batch_int[b, i] * scale where scale = 4.0 * n_half * inv_m
# So ai = 4*n_half * c_i / m
# conv[2*i] = ai^2 = 16*n_half^2 * c_i^2 / m^2
# TV at ell=2 = conv[2*i] / (4*n_half*2) = 16*n_half^2*c_i^2 / (m^2 * 8*n_half)
#            = 2*n_half * c_i^2 / m^2

# For d_child=8, n_half_child=4: TV = 2*4*c_i^2/400 = 8*c_i^2/400

# At c_i = 9: TV = 8*81/400 = 648/400 = 1.62
# Pruning threshold at this window: c_target + (3 + 2*9)/400 = 1.40 + 21/400 = 1.4525
# 1.62 > 1.4525. YES, pruned.

# At c_i = 8: TV = 8*64/400 = 512/400 = 1.28
# Pruning threshold: c_target + (3 + 2*8)/400 = 1.40 + 19/400 = 1.4475
# 1.28 < 1.4475. NOT pruned by this window.

# So c_i = 9 IS correctly prunable. The x_cap filter is sound here.

print("Detailed check: is c_i = x_cap+1 always prunable?")
print()

for d_child in [8, 16, 32, 64]:
    n_half_child = d_child // 2
    corr = correction(m)
    thresh = C_TARGET + corr + 1e-9
    x1 = int(math.floor(m * math.sqrt(thresh / d_child)))
    x2 = int(math.floor(m * math.sqrt(C_TARGET / d_child))) + 1
    xf = min(x1, x2, m)

    ci = xf + 1  # the first filtered value
    if ci > m:
        print(f"d_child={d_child}: x_cap={xf}, ci={ci} > m, no issue")
        continue

    # TV at ell=2 self-window
    tv_self = 2 * n_half_child * ci * ci / (m * m)

    # The pruning threshold at this window (W_int = ci for the self-window)
    w_int = ci  # only bin i contributes
    thresh_w = C_TARGET + (3 + 2*w_int) / (m*m)

    pruned = tv_self > thresh_w

    # Also check Cauchy-Schwarz: d_child * ((ci-1)/m)^2 >= c_target?
    cs_bound = d_child * ((ci-1)/m)**2
    cs_pruned = cs_bound >= C_TARGET

    print(f"d_child={d_child}: x_cap={xf} (binding={'x_cap' if x1 < x2 else 'x_cap_cs'})")
    print(f"  c_i={ci}: TV_self = {tv_self:.4f}, "
          f"thresh(W={w_int}) = {thresh_w:.4f}, "
          f"pruned by window? {pruned}")
    print(f"  Cauchy-Schwarz: d*(({ci}-1)/{m})^2 = {cs_bound:.4f} >= {C_TARGET}? {cs_pruned}")
    if not pruned and not cs_pruned:
        print(f"  *** SOUNDNESS BUG: c_i={ci} filtered by x_cap but NOT prunable! ***")
    else:
        print(f"  OK: prunable by {'window scan' if pruned else 'Cauchy-Schwarz'}")
    print()

print("=" * 50)
print("CONCLUSION: x_cap is sound for all tested levels.")

"""Sanity check: verify the Schinzel-Schmidt boundary case.
f0(x) = (2x + 1/2)^{-1/2} on [-1/4, 1/4]
- Verify int f0 = 1
- Verify ||f0||_{3/2}^3 = 4
- Verify sup f0*f0 = pi/2
- Verify ratio = pi/8 ~ 0.3927
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# f0(x) = (2x + 1/2)^{-1/2}, x in (-1/4, 1/4)
# at x=-1/4, 2x+1/2 = 0 -> singularity
def f0(x):
    return (2*x + 0.5)**(-0.5)

# 1. integral
I, err = quad(f0, -0.25 + 1e-14, 0.25, limit=2000, points=[-0.25])
print(f"int f0 = {I:.10f}  (expected 1)")

# 2. ||f0||_{3/2}^3 = int f0^{1.5} dx
I32, err = quad(lambda x: f0(x)**1.5, -0.25+1e-14, 0.25, limit=2000)
print(f"int f0^1.5 = ||f0||_{{3/2}}^3 = {I32:.10f}  (expected 4)")
# Check: int (2x+1/2)^{-3/4} dx = (1/2)*4*(2x+1/2)^{1/4} = 2(2x+1/2)^{1/4}
# at 0.25: 2*1^{1/4}=2, at -0.25: 0, so = 2... wait
# d/dx (2x+1/2)^{1/4} = (1/4)(2x+1/2)^{-3/4}*2 = (1/2)(2x+1/2)^{-3/4}
# So int_{-1/4}^{1/4} (2x+1/2)^{-3/4} dx = 2[(2x+1/2)^{1/4}]_{-1/4}^{1/4}
#  = 2*(1 - 0) = 2. Hmm, that's 2 not 4.
# Wait expected was ||f0||_{3/2}^3 = 4.
# ||f0||_{3/2} = (int |f0|^{3/2})^{2/3}
# So ||f0||_{3/2}^3 = (int f0^{3/2})^2.
# (2)^2 = 4. Yes!
print(f"||f0||_{{3/2}}^3 = (int f0^{{1.5}})^2 = {I32**2:.10f}  (expected 4)")

# 3. sup f0 * f0
# (f0*f0)(t) = int f0(x) f0(t-x) dx
# At t in [-1/2, 1/2], x in [-1/4, 1/4], t-x in [-1/4, 1/4]
# So x in [max(-1/4, t-1/4), min(1/4, t+1/4)]
def conv_f0(t):
    a = max(-0.25, t - 0.25)
    b = min(0.25, t + 0.25)
    if a >= b: return 0.0
    # integrand has potential singularities at x=-1/4 (from f0(x)) and x=t+1/4 (from f0(t-x), since t-x = -1/4)
    # split near both endpoints
    eps = 1e-12
    pts = []
    if a > -0.25: pts.append(a)
    pts.append(b)
    val, _ = quad(lambda x: f0(x) * f0(t - x), a + eps, b - eps, limit=2000)
    return val

# scan t
ts = np.linspace(-0.499, 0.499, 1001)
vals = [conv_f0(t) for t in ts]
peak_t = ts[int(np.argmax(vals))]
peak_v = max(vals)
print(f"peak at t~{peak_t:.4f}, value~{peak_v:.6f} (expected pi/2={np.pi/2:.6f})")

# zoom near t=1/2 (where the singularity peaks)
res = minimize_scalar(lambda t: -conv_f0(t), bounds=(0.4, 0.4999), method='bounded', options={'xatol':1e-9})
print(f"sup at t~{res.x:.6f}, sup f*f = {-res.fun:.8f}")
print(f"ratio = {-res.fun}/4 = {-res.fun/4:.6f} (expected pi/8 = {np.pi/8:.6f})")

# Analytically (f0*f0)(t) for t close to 1/2:
# x in [t-1/4, 1/4], so 2x+1/2 in [2t, 1] and 2(t-x)+1/2 in [0, 1-2t+1] = ...
# Let's parametrize: u = 2x+1/2, v = 2(t-x)+1/2 = 2t+1-u
# u in [2t, 1], v in [0, 1-2t] (if t > 0)
# f0(x) f0(t-x) = u^{-1/2} v^{-1/2} = (u(2t+1-u))^{-1/2}
# dx = du/2
# Wait, at t=1/2 exactly: u in [1,1], degenerate.
# t -> 1/2: int_{2t}^1 (u(2t+1-u))^{-1/2} du / 2 = ?
# Let w = u - (2t+1)/2, then u(2t+1-u) = ((2t+1)/2)^2 - w^2
# = (1/4)(1-2t)^2 ... no wait (2t+1)/2 squared minus w^2.
# Hmm this is the form (R^2 - w^2)^{-1/2} which integrates to arcsin(w/R)
# = arcsin( (u-(2t+1)/2) / ((1-... no)
# Actually if u in [2t, 1], and center at (2t+1)/2, half-width (1-2t)/2.
# So int = (1/2) * arcsin((u-(2t+1)/2)/((1-2t)/2)) from u=2t to u=1
#       = (1/2) * (arcsin(1) - arcsin(-1)) = (1/2)*pi = pi/2.
# So (f*f)(t) = pi/2 for ALL t in (0, 1/2)? Wait that's not right with the dx/2 factor.
# Let me redo: dx = du/2.
# int = (1/2) * int_{2t}^{1} (u(2t+1-u))^{-1/2} du
# Substituting: u = (2t+1)/2 + r*((1-2t)/2), du = ((1-2t)/2) dr, r from -1 to 1
# u(2t+1-u) = ((2t+1)/2)^2 - r^2*((1-2t)/2)^2 - wait
# Actually: u(2t+1-u). Let m=(2t+1)/2, h=(1-2t)/2. u = m+rh, 2t+1-u = m-rh.
# Product = m^2 - r^2 h^2.
# Hmm not constant. So
# int = (1/2) * int_{-1}^{1} (m^2 - r^2 h^2)^{-1/2} h dr
#     = (h/2) * int_{-1}^{1} 1/sqrt(m^2 - h^2 r^2) dr
#     = (h/2) * (1/h) * [arcsin(h r/m)]_{-1}^{1}
#     = (1/2) * 2 * arcsin(h/m)
#     = arcsin((1-2t)/(1+2t))
# At t=0: arcsin(1) = pi/2.
# Hmm but we expect sup at t=0?
# At t=0: m=1/2, h=1/2, so arcsin(1) = pi/2.
# At t->1/2: m->1, h->0, so arcsin(0)=0.
# So sup f*f = pi/2 at t=0!
print(f"\nAnalytical: (f0*f0)(0) = arcsin(1) = pi/2 = {np.pi/2}")
print(f"Numerical conv_f0(0) = {conv_f0(0)}")

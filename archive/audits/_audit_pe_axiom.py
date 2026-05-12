"""Adversarial probe for cs_eq1_pointeval axiom.

We try to construct admissible f (nonneg, support in (-1/4, 1/4), int f = 1)
where (g*g)(t_q) > R(f) + correction(c, q), violating the Lean axiom.

Key parameters: n=2, m=20, q=3 (lattice point t_3 = 0).
For this q, pointeval_correction = (2*W_int + n_bins)/(4*n*m^2) where
W_int = sum c, n_bins = 4 (all bins contribute since q=3 is the center).
"""
import numpy as np
from scipy.integrate import quad


def bin_masses(f, n=2):
    bins = []
    delta = 1.0 / (4*n)
    for i in range(2*n):
        a = -0.25 + i*delta
        b = -0.25 + (i+1)*delta
        m, _ = quad(f, a, b, limit=5000)
        bins.append(m)
    return bins


def canonical_disc(masses, n=2, m_param=20):
    S = 4*n*m_param
    total = sum(masses)
    if total == 0:
        return [0]*(2*n)
    cum = [0.0]
    for k in range(1, 2*n+1):
        cum.append(sum(masses[:k]) / total)
    target = [c*S for c in cum]
    disc_cum = [int(np.floor(t)) for t in target]
    c = []
    d = 2*n
    for i in range(d):
        if i+1 < d:
            c.append(disc_cum[i+1] - disc_cum[i])
        else:
            c.append(S - disc_cum[i])
    return c


def pointeval_value(c, q, n=2, m=20):
    val = 0.0
    d = 2*n
    for i in range(d):
        for j in range(d):
            if i+j == q:
                val += (c[i]/m) * (c[j]/m)
    return (1.0/(4*n)) * val


def pointeval_correction(c, q, n=2, m=20):
    d = 2*n
    lo = max(0, q-d+1)
    hi = min(q, d-1)
    if lo > hi:
        return 0.0
    W_int = sum(c[i] for i in range(lo, hi+1))
    n_bins = hi+1 - lo
    return (2*W_int + n_bins) / (4*n*m**2)


def r_f_estimate(f, n_samples=400):
    def ff(x):
        v, _ = quad(lambda t: f(t)*f(x-t), -0.25, 0.25, limit=10000)
        return v
    xs = np.linspace(-0.499, 0.499, n_samples)
    return max([ff(x) for x in xs])


def test_axiom(f_factory, label):
    """Test whether axiom holds for f produced by f_factory()."""
    f = f_factory
    I, _ = quad(f, -0.25, 0.25, limit=5000)
    if I <= 0:
        return None
    # Normalize
    def fn(x):
        return f(x) / I
    Rf = r_f_estimate(fn)
    bm = bin_masses(fn)
    c = canonical_disc(bm)
    print(f"\n=== {label} ===")
    print(f"  bin masses (norm): {[f'{x:.4f}' for x in bm]}")
    print(f"  c = {c}, sum = {sum(c)}")
    print(f"  R(f) = {Rf:.4f}")
    for q in range(7):
        pv = pointeval_value(c, q)
        corr = pointeval_correction(c, q)
        violation = pv - Rf - corr
        flag = "VIOLATION!" if violation > 0 else ""
        print(f"  q={q} t={-0.5 + (q+1)/8.0:.4f}: pv={pv:.4f}, corr={corr:.4f}, "
              f"pv-R(f)-corr = {violation:+.4f} {flag}")
    return Rf, c


# ------------------- Test cases -------------------

# Case 1: Uniform f (boundary case, axiom tight)
def f_uniform(x):
    return 2.0 if -0.25 < x < 0.25 else 0.0
test_axiom(f_uniform, "Uniform f = 2")

# Case 2: Anti-symmetric oscillation
for eps in [1.0, 1.5, 1.9]:
    for N in [100, 1000]:
        def f_sin(x, eps=eps, N=N):
            if -0.25 < x < 0.25:
                v = 2.0 + eps*np.sin(2*np.pi*N*x)
                return max(0, v)
            return 0.0
        test_axiom(f_sin, f"f = 2 + {eps}sin({N}x)")

# Case 3: Concentrate near boundaries, measure central bins
def f_outer(x):
    if -0.25 < x < -0.125: return 4.0
    if 0.125 < x < 0.25: return 4.0
    return 0.0  # bin 1, 2 EMPTY
test_axiom(f_outer, "f concentrated at outer bins")

# Case 4: Concentrate near center
def f_center(x):
    if -0.125 < x < 0.125: return 4.0
    return 0.0
test_axiom(f_center, "f concentrated at center")

# Case 5: TWO-POINT-SPIKES (gives high (f*f)(0))
def f_spikes(x):
    if -0.005 < x < 0.005: return 100.0
    return 0.0
test_axiom(f_spikes, "Narrow spike at 0")

# Case 6: Anti-spike at 0 (mass at far ends)
def f_two_spikes(x):
    if 0.245 < x < 0.25: return 100.0
    if -0.25 < x < -0.245: return 100.0
    return 0.0
test_axiom(f_two_spikes, "Two spikes at ends")

# Case 7: Concentrate in bin 1 boundary, sharp (the critical one for q=3)
# We want (g*g)(0) large but (f*f)(0) small.
# (g*g)(0) for c=(c_0, c_1, c_2, c_3) with i+j=3: c_0 c_3 + c_1 c_2 + c_2 c_1 + c_3 c_0.
# = 2(c_0 c_3 + c_1 c_2).  Maximize when c is concentrated.
# Try f with mass = 0.5 in bin 1 and 0.5 in bin 2 (so c=(0,80,80,0)).
# (g*g)(0) = (1/8)·(2·80·80)/400 = 12800/3200 = 4.0.
# Then for f to have (f*f)(0) small: split bin 1 and bin 2 mass into spikes far from 0.
# bin 1 spike at -1/8 + 0.001 = -0.124, bin 2 spike at 1/8 - 0.001 = 0.124.
def f_outer_inner(x):
    if -0.124 < x < -0.123: return 500.0  # mass 0.5
    if 0.123 < x < 0.124: return 500.0  # mass 0.5
    return 0.0
test_axiom(f_outer_inner, "Spikes near bin1/bin2 boundaries (away from 0)")

# Case 8: Spikes in bin 1 / bin 2 BOTH at -0.001/+0.001 (near 0)
def f_inner_spikes(x):
    if -0.001 < x < 0: return 500.0
    if 0 < x < 0.001: return 500.0
    return 0.0
test_axiom(f_inner_spikes, "Spikes near 0 (gives huge (f*f)(0))")

# Case 9: Trial — mass in bin 1 spread to far-from-0 side, bin 2 spread to far-from-0 side
# bin 1 = (-1/8, 0): place spike at -1/8 + 1/200 = -0.115
# bin 2 = (0, 1/8): place spike at 1/8 - 1/200 = 0.115
def f_far_from_0(x):
    if -0.116 < x < -0.115: return 500.0  # mass 0.5
    if 0.115 < x < 0.116: return 500.0   # mass 0.5
    return 0.0
test_axiom(f_far_from_0, "Mass in bins 1,2 but far from t=0")

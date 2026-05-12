"""Test whether (f̃*f̃)(t_q) ≤ ‖f*f‖_∞ for general f.

If TRUE, the Lean axiom is paper-derivable via the chain:
  (g*g)(t_q) ≤ (f̃*f̃)(t_q) + correction(c, q)    [C&S Lemma 3 refinement (1) per-point]
  (f̃*f̃)(t_q) ≤ ‖f*f‖_∞ = R(f)                  [the gap question]
  Combined: (g*g)(t_q) ≤ R(f) + correction.

If FALSE, the axiom is suspect.

We construct f with bin masses (m_0, m_1, m_2, m_3), bin avgs a_i = (4n)·m_i,
and compare (f̃*f̃)(t_q) to max over x of (f*f)(x).
"""
import numpy as np
from scipy.integrate import quad

n = 2
delta = 1.0/(4*n)
bin_lo = [-0.25 + i*delta for i in range(2*n)]
bin_hi = [-0.25 + (i+1)*delta for i in range(2*n)]


def f_tilde_eval(x, masses):
    """Step function f̃ with heights a_i = (4n)·m_i at bin i."""
    if x < -0.25 or x >= 0.25:
        return 0.0
    i = int(np.floor((x + 0.25)/delta))
    if 0 <= i < 2*n:
        return (4*n) * masses[i]
    return 0.0


def step_conv(masses, x):
    """(f̃*f̃)(x) for step function with bin masses given."""
    v, _ = quad(lambda t: f_tilde_eval(t, masses)*f_tilde_eval(x-t, masses),
                -0.25, 0.25, limit=1000)
    return v


def general_conv(f, x):
    v, _ = quad(lambda t: f(t)*f(x-t), -0.25, 0.25, limit=10000)
    return v


def max_conv(f, n_samples=400):
    xs = np.linspace(-0.499, 0.499, n_samples)
    return max([general_conv(f, x) for x in xs])


def test_gap(f, label):
    """Test whether (f̃*f̃)(t_q) ≤ ‖f*f‖_∞ for the bin-avg of f."""
    I, _ = quad(f, -0.25, 0.25, limit=5000)
    if I <= 0:
        return
    def fn(x):
        return f(x)/I
    # Bin masses
    masses = []
    for i in range(2*n):
        m, _ = quad(fn, bin_lo[i], bin_hi[i], limit=2000)
        masses.append(m)
    Rf = max_conv(fn)
    # Lattice points and (f̃*f̃)
    print(f"\n=== {label} ===")
    print(f"  bin masses = {[f'{m:.4f}' for m in masses]}")
    print(f"  bin avgs a = {[f'{(4*n)*m:.4f}' for m in masses]}")
    print(f"  R(f) = ‖f*f‖_∞ ≈ {Rf:.4f}")
    for q in range(2*(2*n) - 1):
        t_q = -0.5 + (q+1)/(4*n)
        ft_val = step_conv(masses, t_q)
        # Also compute pointeval_value = step conv at t_q via bin avgs
        a_arr = [(4*n)*m for m in masses]
        pv_a = (1.0/(4*n)) * sum(
            a_arr[i]*a_arr[j] for i in range(2*n) for j in range(2*n) if i+j==q
        )
        gap = ft_val - Rf
        flag = "GAP!" if gap > 0 else ""
        print(f"  q={q} t={t_q:.4f}: (f̃*f̃)(t)={ft_val:.4f}, sum-version={pv_a:.4f}, "
              f"(f̃*f̃)-R(f) = {gap:+.4f} {flag}")


# Test cases:
# 1. Step function (should be exact: f̃ = f → all gaps zero)
def f_step(x):
    if -0.25 < x < -0.125: return 4.0
    if 0.125 < x < 0.25: return 4.0
    return 0.0
test_gap(f_step, "Step f = (4,0,0,4)")

# 2. Asymmetric f within bins to push (f̃*f̃)(t_q) > ‖f*f‖_∞:
# Put mass in bin 1 at left edge (close to -1/8), mass in bin 2 at right edge (close to 1/8).
# Bin avgs a = (0, A, A, 0) for A = 4·m, large.
# (f̃*f̃)(0) = (1/8)·(2·A·A) = A²/4 if mass = 0.5 each: A=2 each so f̃*f̃(0) = 1.
# But (f*f)(0) = ∫f(t)f(-t) dt: mass in bin 1 close to -1/8 paired with mass in bin 2 close to 1/8 — distance > 0, so (f*f)(0) is small.
# Mass in bin 1 at far-left (-1/8 + h) with -t = +1/8-h; mass in bin 2 must be at +1/8-h.
# f(t)f(-t) is non-zero only when both t in bin 1 and -t in bin 2 ⇒ t ∈ (-1/8, 0) AND -t ∈ (0, 1/8), so t ∈ (-1/8, 0).
# If mass concentrated at t = -1/8 + h_1 in bin 1, and at t' = 1/8 - h_2 in bin 2:
# (f*f)(0) = ∫f(t)f(-t)dt. Both delta-like spikes at -1/8+h_1 and 1/8-h_2. f(-t)= f at t reflected is at 1/8-h_1. For f(-t) to align with the bin 2 spike at 1/8-h_2, we need 1/8-h_1 = 1/8-h_2 ⇒ h_1=h_2.
# So concentration at t=-1/8+h in bin 1 AND t=1/8-h in bin 2 gives (f*f)(0) = δ²·something. Otherwise zero.

def f_concentrated(x):
    # Bin 1 (-1/8, 0): place mass 0.5 at x ≈ -0.124 (very narrow)
    if -0.124 < x < -0.123: return 500.0
    # Bin 2 (0, 1/8): place mass 0.5 at x ≈ 0.124
    if 0.123 < x < 0.124: return 500.0
    return 0.0
test_gap(f_concentrated, "Concentrated f: spikes at -0.123, 0.123 in bins 1,2")

# 3. Same idea, bins 1 and 2 are SHIFTED: bin 1 spike at boundary of bin 0/bin 1.
def f_concentrated_2(x):
    if -0.124 < x < -0.123: return 500.0  # bin 1
    if 0.001 < x < 0.002: return 500.0    # bin 2 near 0
    return 0.0
test_gap(f_concentrated_2, "Asymmetric: bin 1 far, bin 2 close to 0")

# 4. Bin masses (1, 0, 0, 1) (uniform A=4 in bins 0 and 3, zero in bins 1, 2)
# f̃*f̃ at t=0: i+j=3 pairs: a_0 a_3 + a_3 a_0 = 32; pv = (1/8)·32 = 4.
# But mass is at far ends: actual (f*f)(0) = ?  Both pulses at distance 1/2. (f*f)(0) = ∫f(t)f(-t) dt.
# f(t)=4 on (-1/4,-1/8) ⇒ f(-t)=f at 1/8 to 1/4: f(-t)=4 there. Integrand =16, area = 1/8. (f*f)(0) = 2.
# So (f̃*f̃)(0)=4 (because step), but (f*f)(0)=2 (different).
# Wait f IS the step function in this case... so f=f̃, (f*f)=(f̃*f̃).

# Let's try f within bin 0 concentrated at left, within bin 3 concentrated at right.
def f_extreme(x):
    if -0.249 < x < -0.245: return 125.0  # mass 0.5 at far left
    if 0.245 < x < 0.249: return 125.0    # mass 0.5 at far right
    return 0.0
test_gap(f_extreme, "Extreme: spikes at -0.247, 0.247 (bins 0, 3)")

# 5. SAME bin masses (0.5, 0, 0, 0.5) but NOT step: mass concentrated at INNER edges of bins 0, 3
def f_inner(x):
    if -0.126 < x < -0.125: return 500.0  # bin 0 right edge: mass 0.5
    if 0.125 < x < 0.126: return 500.0    # bin 3 left edge: mass 0.5
    return 0.0
test_gap(f_inner, "Inner edges: spikes at ±0.125 (bins 0, 3)")

# 6. Try maximizing (f̃*f̃)(t_q) − R(f):
# Need bin-avg a such that (1/(4n))·sum_{i+j=q} a_i a_j is large at q,
# but f within bins arranged so (f*f)(x) is small at all x.
# Key insight: ∫(f*f)dx = (∫f)² = 1.  If (f*f)(0) is small, mass must be concentrated elsewhere.
# (f̃*f̃) and (f*f) have SAME L^1 norm: ∫(f̃*f̃) = (∫f̃)² = (∫f)² = 1.
# So if (f*f) has its mass spread, ‖f*f‖_∞ is small; but (f̃*f̃) also has total mass 1.
# Q: can max(f̃*f̃) > max(f*f)?

# Try: f in bin 1 = Dirac at -0.001 (near 0), bin 2 = Dirac at +0.001 (near 0).
def f_inner_to_boundary(x):
    if -0.002 < x < -0.001: return 1000.0  # bin 1, near 0, mass 1
    if 0.001 < x < 0.002: return 1000.0    # bin 2, near 0, mass 1
    return 0.0
test_gap(f_inner_to_boundary, "Spikes near 0 (bin 1 and 2)")

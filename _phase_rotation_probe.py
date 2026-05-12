"""
Phase-rotation gap probe.

For f >= 0 on [-1/4, 1/4] with int f = 1, define
    R_f = f * f_tilde  (autocorrelation, even, max = K = ||f||_2^2)
    M(f) = max_{|t| <= 1/2} (f * f)(t)  (convolution, |t|<=1/2 contains support)
    K - M = "phase-rotation gap" (>= 0, =0 iff f symmetric about some center)

Sidon constant: C_{1a} = inf_f M(f).  Bracket 1.2802 <= C_{1a} <= 1.5029.

We probe whether K - M is controlled by phase functionals of theta(xi) = arg(f_hat).
We compute several candidate phase functionals and check correlation with K - M.

All FFT-based on a fine grid; resolution N >> 1.
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq

# ---- grid / FFT setup -----------------------------------------------------
# f supported on [-1/4, 1/4] (length 1/2)
# f*f supported on [-1/2, 1/2] (length 1)
# Use a working interval [-1, 1] (length 2) with periodic FFT — no wrap because supp(f*f) << period.
L = 2.0          # period
N = 1 << 15      # 32768 grid points
dx = L / N
xs = (np.arange(N) - N // 2) * dx   # -1, ..., 1-dx (centered)

# integer indices for support
def in_support_quarter(x):
    return (x >= -0.25 - 1e-12) & (x <= 0.25 + 1e-12)

mask_q = in_support_quarter(xs)

def normalize(f):
    """Force supp f subset [-1/4, 1/4], integral 1, nonneg."""
    f = np.maximum(f, 0.0)
    f = np.where(mask_q, f, 0.0)
    s = f.sum() * dx
    if s <= 0:
        return None
    return f / s

def conv(f, g):
    """f * g via FFT (linear convolution OK since supports << period)."""
    F = fft(np.fft.ifftshift(f))
    G = fft(np.fft.ifftshift(g))
    return np.fft.fftshift(np.real(ifft(F * G))) * dx

def K_M_and_phase(f):
    """Return (K, M, phase data dict) for the given f sampled on xs."""
    # autocorrelation R_f(t) = (f * f_tilde)(t), where f_tilde(x) = f(-x).
    f_tilde = f[::-1]
    Rf = conv(f, f_tilde)
    # convolution f*f
    ff = conv(f, f)
    # restrict measurement interval |t| <= 1/2 (where supp lives)
    mask_half = (xs >= -0.5 - 1e-12) & (xs <= 0.5 + 1e-12)
    K_meas = float(np.max(Rf[mask_half]))
    M = float(np.max(ff[mask_half]))
    K = float(np.sum(f * f) * dx)  # ||f||_2^2 (analytic K)
    # f-hat on the FFT grid:
    F = np.fft.fftshift(fft(np.fft.ifftshift(f))) * dx     # continuous-FT approx
    # frequency grid:
    xi = np.fft.fftshift(fftfreq(N, d=dx))
    return K, M, K_meas, dict(F=F, xi=xi, Rf=Rf, ff=ff)

# ---- phase functionals ----------------------------------------------------

def phase_functionals(F, xi, K, weight_kind="arcsine"):
    """Compute various phase-deviation functionals.

    F[k] = f_hat(xi[k]) on a continuous-frequency grid.
    We define theta(xi) = arg(F(xi)), |F(xi)|^2 = autocorr spectrum.
    """
    mag2 = (np.abs(F) ** 2).real
    theta = np.angle(F)
    # 2 theta is the rotation that turns |F|^2 into F^2: F^2 = |F|^2 e^{i 2 theta}
    sin2theta = np.sin(2 * theta)
    cos2theta = np.cos(2 * theta)
    sinT = np.sin(theta)

    # frequency weights
    # arcsine k_hat: K(t) = (2/pi) * 1/sqrt(1-(2t)^2) on |t|<1/2 has hat
    # we use a flat weight on |xi|<=Xi with Xi to be chosen
    Xi = 6.0  # bandwidth where Bessel coefs of arcsine are large enough to matter
    w_flat = (np.abs(xi) <= Xi).astype(float)
    # Riemann-Lebesgue tail correction:
    # mass-weighted: just |F|^2 (Plancherel for f weighted by f)
    # we want functionals that vanish for symmetric f (theta even => 2theta even)
    # but the sign of sin(2 theta) can flip; we use squared/abs versions.

    # P1 = int |sin(2 theta)|^2 |F|^2 d xi (basic phase deviation, weighted by spectrum)
    integrand1 = sin2theta**2 * mag2
    P1 = float(np.trapezoid(integrand1, xi))

    # P2 = int (1 - cos(2 theta)) |F|^2 d xi  (= Re((F)^2 - |F|^2) etc.)
    # 1 - cos(2 theta) = 2 sin^2(theta).
    integrand2 = (1.0 - cos2theta) * mag2
    P2 = float(np.trapezoid(integrand2, xi))

    # P3 = int |F^2 - |F|^2|^2 d xi = int |F|^4 |1 - e^{-i 2 theta}|^2 d xi
    # = int |F|^4 * 2(1 - cos 2 theta) d xi
    integrand3 = (np.abs(F**2 - mag2)) ** 2
    P3 = float(np.trapezoid(integrand3, xi))

    # P4 = int Im(F^2) over xi (signed, integral of imaginary part) — could be 0 by AS.
    P4 = float(np.trapezoid(np.imag(F**2), xi))

    # P5 = max over t of (R_f(t) - Re(f*f)(t))?  We compute via spectrum:
    # R_f - Re(f*f) hat = |F|^2 - Re(F^2) = |F|^2 (1 - cos 2 theta)
    # so its inverse FT integrand is real even.
    # Bound on its sup: ||(|F|^2 (1 - cos 2 theta))||_1 / (2 pi)? Just use L1 of the integrand.
    P5 = float(np.trapezoid(mag2 * (1 - cos2theta), xi))   # = P2

    # P6 = int |sin 2 theta|^2 / (1 + xi^2) d xi  (smooth cutoff)
    P6 = float(np.trapezoid(sin2theta**2 / (1 + xi**2), xi))

    # P7 = sup over t of (R_f - f*f) (the actual value we want to bound!)
    # by Parseval-type:  R_f(t) - f*f(t) has FT |F|^2 - F^2, real part = |F|^2(1 - cos 2 th)
    diff = None  # computed externally if needed

    return dict(P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, P6=P6)


# ---- candidate parametric f's --------------------------------------------

rng = np.random.default_rng(7)

def f_indicator():
    """Symmetric: 2 * 1_{[-1/4,1/4]}.  K = ||f||_2 = 2."""
    f = np.where(mask_q, 1.0, 0.0)
    return normalize(f)

def f_skewed_bump(a, b):
    """Triangular density on [-a,b] (a,b in [0,1/4])."""
    f = np.zeros_like(xs)
    # peak at 0, linear to -a and to b
    inL = (xs >= -a) & (xs <= 0)
    inR = (xs >= 0) & (xs <= b)
    f[inL] = 1.0 - (-xs[inL]) / a if a > 0 else 0.0
    f[inR] = 1.0 - xs[inR] / b if b > 0 else 0.0
    return normalize(f)

def f_two_bump(c1, c2, w1, w2, h1, h2):
    """Two Gaussian bumps inside [-1/4,1/4], potentially asymmetric."""
    f = h1 * np.exp(-((xs - c1) / w1) ** 2) + h2 * np.exp(-((xs - c2) / w2) ** 2)
    return normalize(f)

def f_skew_normal(loc, scale, alpha):
    """Truncated skew-normal-style: phi(z) * Phi(alpha z), z = (x-loc)/scale."""
    z = (xs - loc) / scale
    phi = np.exp(-0.5 * z**2)
    Phi = 0.5 * (1 + np.erf(alpha * z / np.sqrt(2))) if False else 0.5 * (1 + np.tanh(alpha * z / 1.5))
    f = phi * Phi
    return normalize(f)

def f_three_atoms(p, q, r, x1, x2, x3, w):
    """Three narrow Gaussians, possibly asymmetric atoms."""
    f = (p * np.exp(-((xs - x1)/w)**2)
         + q * np.exp(-((xs - x2)/w)**2)
         + r * np.exp(-((xs - x3)/w)**2))
    return normalize(f)

def f_random_smooth(scale=0.05, K=10):
    """Random nonneg smooth on [-1/4,1/4]."""
    centers = rng.uniform(-0.24, 0.24, size=K)
    weights = rng.exponential(scale=1.0, size=K)
    widths = rng.uniform(0.01, 0.06, size=K)
    f = np.zeros_like(xs)
    for c, w, ww in zip(centers, weights, widths):
        f += w * np.exp(-((xs - c) / ww) ** 2)
    return normalize(f)

# ---- run probe ------------------------------------------------------------

def asymmetry_score(f):
    """Simple asymmetry: |f - f(-)|_1 / |f|_1."""
    return float(np.sum(np.abs(f - f[::-1])) * dx) / 2.0

def trial(name, f):
    if f is None:
        return None
    K, M, K_meas, data = K_M_and_phase(f)
    Pdict = phase_functionals(data["F"], data["xi"], K)
    asym = asymmetry_score(f)
    return dict(name=name, K=K, K_meas=K_meas, M=M, gap=K - M, asym=asym, **Pdict)


cases = []

# 1. symmetric reference
cases.append(("symm_indicator", f_indicator()))
cases.append(("symm_triangle", f_skewed_bump(0.25, 0.25)))
cases.append(("symm_gauss_centered", f_two_bump(0.0, 0.0, 0.08, 0.08, 1.0, 0.0)))

# 2. triangle skew sweep
for r in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18]:
    a = max(0.01, 0.25 - r)
    b = 0.25
    cases.append((f"tri_skew_r{r:.2f}", f_skewed_bump(a, b)))

# 3. two-bump asymmetric sweep
for delta in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for hr in [0.5, 0.8, 1.0, 1.5, 2.0]:
        c1, c2 = -delta, delta
        cases.append((f"twobump_d{delta:.2f}_hr{hr:.1f}", f_two_bump(c1, c2, 0.03, 0.03, 1.0, hr)))

# 4. skew-normal sweep
for alpha in [-3, -1, 0, 1, 2, 4, 8]:
    cases.append((f"skewN_a{alpha}", f_skew_normal(0.0, 0.07, alpha)))

# 5. three-atom (asym)
for trip in [(0.5, 0.3, 0.2, -0.18, 0.0, 0.20),
             (0.4, 0.4, 0.4, -0.20, -0.05, 0.18),
             (0.3, 0.5, 0.2, -0.22, 0.10, 0.22),
             (0.2, 0.4, 0.4, -0.22, 0.05, 0.20)]:
    cases.append((f"tri_atom_{trip[3]:+.2f}_{trip[4]:+.2f}_{trip[5]:+.2f}",
                  f_three_atoms(*trip[:3], *trip[3:6], 0.015)))

# 6. random smooth ensemble
for k in range(10):
    cases.append((f"rand_smooth_{k:02d}", f_random_smooth()))

# 7. extreme single-atom shifted (close to boundary)
for c in [-0.20, -0.10, 0.0, 0.10, 0.20]:
    cases.append((f"atom_c{c:+.2f}", f_two_bump(c, c, 0.03, 0.03, 1.0, 0.0)))

# Run trials
results = []
for name, f in cases:
    r = trial(name, f)
    if r is not None:
        results.append(r)

# ---- output ---------------------------------------------------------------
print(f"# Phase-rotation probe; N = {N}, dx = {dx:.2e}")
print(f"{'name':32s} {'K':>9s} {'M':>9s} {'gap':>9s} {'asym':>7s} {'P1':>9s} {'P2':>9s} {'P3':>9s} {'P6':>9s}")
for r in results:
    print(f"{r['name']:32s} {r['K']:9.4f} {r['M']:9.4f} {r['gap']:9.4f} {r['asym']:7.3f} "
          f"{r['P1']:9.4e} {r['P2']:9.4e} {r['P3']:9.4e} {r['P6']:9.4e}")

# ---- correlation analysis ------------------------------------------------
import numpy as np

def corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.std() < 1e-12 or y.std() < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

gaps = [r["gap"] for r in results]
for key in ("P1", "P2", "P3", "P6"):
    vals = [r[key] for r in results]
    c = corr(gaps, vals)
    # also for nontrivial gap > 1e-4
    big = [(g, v) for g, v in zip(gaps, vals) if g > 1e-4]
    c_big = corr([g for g, _ in big], [v for _, v in big]) if len(big) > 3 else float("nan")
    # and ratio gap / P
    ratios = [g / v if v > 1e-12 else float("inf") for g, v in zip(gaps, vals) if g > 1e-4]
    print(f"{key}: corr_all={c:.3f}  corr_gap>1e-4={c_big:.3f}  "
          f"min(gap/{key})={min(ratios) if ratios else float('nan'):.3e}  "
          f"max(gap/{key})={max(ratios) if ratios else float('nan'):.3e}")

# best single-functional bound: largest c such that gap <= c * P_j for all
print()
print("# Largest constant c with gap(f) <= c * P_j(f) for all asym cases (gap > 1e-4):")
for key in ("P1", "P2", "P3", "P6"):
    vals = [(r["gap"], r[key]) for r in results if r["gap"] > 1e-4 and r[key] > 0]
    if not vals: continue
    cs = [g / v for g, v in vals]
    print(f"  c_min({key}) = {min(cs):.3e}, c_max({key}) = {max(cs):.3e}, ratio = {max(cs)/min(cs):.2e}")

# ---- save raw data for further analysis ----------------------------------
import json
with open("phase_rotation_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print("\n# wrote phase_rotation_results.json")

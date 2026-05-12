"""K - M gap audit for Sidon C_{1a}.

For each near-extremizer / test pdf f on [-1/4, 1/4] with int f = 1, compute
    K = ||f||_2^2,
    M = ||f*f||_inf,
    K - M.
And asymmetry data:
    mean   = int x f(x) dx
    var    = int (x - mean)^2 f
    skew   = int (x - mean)^3 f / var^{3/2}    (Pearson skewness)
    abs3   = int |x - mean|^3 f
    fphi   = int |hat f(xi)|^2 |arg hat f(xi)|^2 dxi   (phase functional)
    fphi1  = int |hat f(xi)|^2 |arg hat f(xi)| dxi     (sin^2 phase)
Then look for an inequality K - M <= Phi(asymmetry data).

Uses purely numeric (FFT) machinery; the BL witness uses its exact integer
heights from coeffBL.txt; MV's near-extremizer uses the 119-cosine recipe at
u = 0.638. Step-function autoconvolutions are evaluated exactly piecewise-
linearly to avoid grid bias.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# Generic numeric K, M, asymmetry on a fine grid
# =============================================================================

def _normalize(f, dx):
    s = f.sum() * dx
    return f / s


def _conv_full(f, dx):
    # (f*f) on length-2N grid; spacing dx; returns autoconvolution values
    return np.convolve(f, f) * dx


def _stats_from_grid(x, f):
    dx = x[1] - x[0]
    f = _normalize(f, dx)
    K = float(np.sum(f * f) * dx)            # ||f||_2^2

    g = _conv_full(f, dx)                     # f*f at spacing dx, on 2N-1 points
    M = float(g.max())                        # ||f*f||_inf

    # Asymmetry data
    mean = float(np.sum(x * f) * dx)
    var  = float(np.sum((x - mean) ** 2 * f) * dx)
    if var > 0:
        skew = float(np.sum((x - mean) ** 3 * f) * dx / var ** 1.5)
    else:
        skew = 0.0
    abs3 = float(np.sum(np.abs(x - mean) ** 3 * f) * dx)

    # Fourier-phase functionals on a fine xi-grid
    # f hat(xi) = int e^{-2 pi i xi x} f(x) dx
    # We use a discrete Fourier integral on a wide xi range.
    n = len(x)
    Nfft = 1
    while Nfft < 4 * n:
        Nfft <<= 1
    # zero-pad f onto the larger interval [-Lx_pad, Lx_pad] then DFT
    M_pad = Nfft
    pad = np.zeros(M_pad)
    Lx = x[-1] - x[0]
    # place x onto centered indices
    start = (M_pad - n) // 2
    pad[start:start + n] = f
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pad))) * dx
    xi = np.fft.fftshift(np.fft.fftfreq(M_pad, d=dx))
    mag2 = (F.real ** 2 + F.imag ** 2)
    phase = np.angle(F)
    # Restrict to the "useful" band where mag2 dominates (avoid tail noise)
    band = mag2 > 1e-10 * mag2.max()
    dxi = xi[1] - xi[0]
    fphi  = float(np.sum(mag2[band] * (phase[band] ** 2)) * dxi)
    fphi1 = float(np.sum(mag2[band] * np.abs(phase[band])) * dxi)

    return {
        "K": K, "M": M, "K-M": K - M,
        "mean": mean, "var": var, "skew": skew, "abs3": abs3,
        "fphi": fphi, "fphi1": fphi1,
    }


# =============================================================================
# Step-function autoconvolution: exact piecewise-linear M
# =============================================================================

def stats_step_function(v, scale_to_quarter=True):
    """v = (v_0,...,v_{N-1}) heights of a unit-interval step function on [0,N).

    If ``scale_to_quarter`` True, rescale to support [-1/4, 1/4].
    Returns dict with K, M, K-M, asymmetry data (computed on the *rescaled* f).

    Exact M comes from the BL formula: f*f piecewise linear with nodes at
    integers, so max is at a node. After rescaling to support length 1/2,
    K_new = K_old * (N / (1/2)) = 2 N * K_old, etc.
    """
    v = np.asarray(v, dtype=np.float64)
    N = len(v)
    # On unit-spaced grid:
    # int f = sum v_n; ||f||_2^2 = sum v_n^2
    # f*f piecewise linear on [j, j+1] with values L_j at j, L_{j+1} at j+1
    L = np.convolve(v, v)  # length 2N-1, L_j for j=0,..,2N-2
    sumv = v.sum()

    # Normalize on unit grid first: f_norm = f / sumv (so int f = 1 on [0, N])
    K_unit = float((v ** 2).sum() / sumv ** 2)        # ||f_norm||_2^2 on [0,N]
    M_unit = float(L.max() / sumv ** 2)               # ||f*f||_inf on [0,N]

    if scale_to_quarter:
        # Support length L_old = N -> L_new = 1/2.  s = L_new / L_old = 1/(2N).
        # f_new(x) = (1/s) f_old(x/s); int f_new = 1.
        # ||f_new||_2^2 = (1/s) ||f_old||_2^2
        # ||f_new * f_new||_inf = (1/s) ||f_old * f_old||_inf
        s = 1.0 / (2 * N)
        K = K_unit / s   # = 2N * K_unit
        M = M_unit / s   # = 2N * M_unit
    else:
        K, M = K_unit, M_unit

    # Asymmetry: build a rescaled grid for the step function
    # Place midpoints of each step on [-1/4, 1/4]
    if scale_to_quarter:
        mid = (np.arange(N) + 0.5) / (2 * N) - 0.25       # in [-1/4, 1/4]
    else:
        mid = np.arange(N) + 0.5
    f_dist = v / sumv   # PMF of midpoints (approx of pdf weight)
    # Continuous version: f_pdf(x) = (v_n / sumv) * (1/step_width)
    step_width = (1.0 / (2 * N)) if scale_to_quarter else 1.0
    f_pdf_at_mid = f_dist / step_width  # pdf height on each step

    # Mean, var, skew using exact step-function moments
    # int x f = sum_n v_n/sumv * mid_n  (exact since within a step, x has mean=mid)
    mean = float((mid * f_dist).sum())
    # int x^2 f over step = mid^2 + step_width^2/12 (var of uniform)
    second = float((f_dist * (mid ** 2 + step_width ** 2 / 12.0)).sum())
    var = second - mean ** 2
    # int (x - mean)^3 f -- expand: E[(X-mean)^3] for X uniform on step + offset
    # E[(X-mean)^3] = (mid - mean)^3 + 3 (mid - mean) * step_width^2/12 (the
    # cube of uniform on [-h/2, h/2] integrates to 0).
    third_central = float(
        (f_dist * ((mid - mean) ** 3 + (mid - mean) * step_width ** 2 / 4.0)).sum()
    )
    if var > 0:
        skew = third_central / var ** 1.5
    else:
        skew = 0.0
    abs3 = float((f_dist * np.abs(mid - mean) ** 3).sum())  # midpoint approx

    # Fourier-phase via DFT on midpoints (sufficient resolution)
    Nfft = max(8192, 8 * N)
    pad = np.zeros(Nfft)
    start = (Nfft - N) // 2
    pad[start:start + N] = f_pdf_at_mid
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pad))) * step_width
    xi = np.fft.fftshift(np.fft.fftfreq(Nfft, d=step_width))
    mag2 = F.real ** 2 + F.imag ** 2
    phase = np.angle(F)
    band = mag2 > 1e-10 * mag2.max()
    dxi = xi[1] - xi[0]
    fphi  = float(np.sum(mag2[band] * (phase[band] ** 2)) * dxi)
    fphi1 = float(np.sum(mag2[band] * np.abs(phase[band])) * dxi)

    return {
        "K": K, "M": M, "K-M": K - M,
        "mean": mean, "var": var, "skew": skew, "abs3": abs3,
        "fphi": fphi, "fphi1": fphi1,
        "_N": N,
    }


# =============================================================================
# Test pdfs
# =============================================================================

def make_grid(n=100001, half_width=0.25):
    return np.linspace(-half_width, half_width, n)


def gaussian_symmetric(sigma=0.08):
    x = make_grid()
    f = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return x, f


def gaussian_skewed(sigma=0.08, alpha=4.0):
    """Skew-normal restricted to [-1/4, 1/4]."""
    x = make_grid()
    pdf = np.exp(-(x ** 2) / (2 * sigma ** 2))
    cdf = 0.5 * (1 + np.tanh(alpha * x / np.sqrt(2)))  # smooth approx
    f = pdf * cdf
    return x, f


def double_gaussian(sigma=0.04, sep=0.08, h_ratio=1.0):
    x = make_grid()
    f = (np.exp(-((x - sep) ** 2) / (2 * sigma ** 2))
         + h_ratio * np.exp(-((x + sep) ** 2) / (2 * sigma ** 2)))
    return x, f


def triangle_symmetric():
    x = make_grid()
    f = np.maximum(0.0, 0.25 - np.abs(x))
    return x, f


def asymmetric_two_bumps(h1=1.0, h2=0.6, sigma=0.04, x1=-0.10, x2=+0.13):
    x = make_grid()
    f = (h1 * np.exp(-((x - x1) ** 2) / (2 * sigma ** 2))
         + h2 * np.exp(-((x - x2) ** 2) / (2 * sigma ** 2)))
    return x, f


def asymmetric_ramp(power=1.5):
    x = make_grid()
    f = np.maximum(0.0, x + 0.25) ** power
    return x, f


def indicator():
    x = make_grid()
    f = np.ones_like(x)
    return x, f


# =============================================================================
# MV near-extremizer (119-cosine on [-1/4, 1/4], u = 0.638)
# =============================================================================

def mv_extremizer():
    from delsarte_dual.restricted_holder.conditional_bound import (
        MV_COEFFS_119_STR,
    )
    coeffs = np.array([float(s) for s in MV_COEFFS_119_STR])
    u = 0.638
    x = make_grid(n=200001)
    G = np.zeros_like(x)
    for j, aj in enumerate(coeffs, start=1):
        G += aj * np.cos(2 * np.pi * j * x / u)
    # G should be >= 0 on [-1/4, 1/4] (it's the MV polynomial)
    G = np.maximum(G, 0.0)
    return x, G


# =============================================================================
# Boyer-Li witness from coeffBL.txt (575 integer heights)
# =============================================================================

def boyer_li_witness():
    p = ROOT / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"
    txt = p.read_text().strip().strip("{}")
    v = np.array([float(s.strip()) for s in txt.split(",")], dtype=np.float64)
    return v


# =============================================================================
# Main
# =============================================================================

def main():
    rows = []

    print("Computing MV near-extremizer ...")
    x, f = mv_extremizer()
    s = _stats_from_grid(x, f)
    s["name"] = "MV-119cos"
    rows.append(s)

    print("Computing Boyer-Li witness (575 steps, exact step-function M)...")
    v = boyer_li_witness()
    print(f"  N = {len(v)}")
    s = stats_step_function(v, scale_to_quarter=True)
    s["name"] = "BL-575step"
    rows.append(s)

    print("Computing symmetric Gaussian, double Gaussian, triangle, indicator ...")
    for name, (x, f) in [
        ("Gaussian-sym",      gaussian_symmetric()),
        ("Triangle-sym",      triangle_symmetric()),
        ("Indicator",         indicator()),
        ("DblGauss-sym",      double_gaussian(h_ratio=1.0)),
    ]:
        s = _stats_from_grid(x, f)
        s["name"] = name
        rows.append(s)

    print("Computing asymmetric pdfs ...")
    asyms = [
        ("Skew-Gauss-a2",     gaussian_skewed(alpha=2.0)),
        ("Skew-Gauss-a4",     gaussian_skewed(alpha=4.0)),
        ("Skew-Gauss-a8",     gaussian_skewed(alpha=8.0)),
        ("DblGauss-1:0.6",    double_gaussian(h_ratio=0.6)),
        ("DblGauss-1:0.3",    double_gaussian(h_ratio=0.3)),
        ("Two-bumps-asym",    asymmetric_two_bumps()),
        ("Ramp-1.5",          asymmetric_ramp(power=1.5)),
        ("Ramp-3.0",          asymmetric_ramp(power=3.0)),
    ]
    for name, (x, f) in asyms:
        s = _stats_from_grid(x, f)
        s["name"] = name
        rows.append(s)

    # Print table
    print()
    print("=" * 116)
    print(f"{'name':16s} {'K':>9s} {'M':>9s} {'K-M':>9s} {'|mean|':>8s} "
          f"{'sqrt(var)':>9s} {'|skew|':>8s} {'abs3':>9s} {'fphi':>8s} {'fphi1':>8s}")
    print("-" * 116)
    for r in rows:
        print(f"{r['name']:16s} {r['K']:9.4f} {r['M']:9.4f} {r['K-M']:9.4f} "
              f"{abs(r['mean']):8.4f} {math.sqrt(max(r['var'],0)):9.4f} "
              f"{abs(r['skew']):8.4f} {r['abs3']:9.5f} "
              f"{r['fphi']:8.4f} {r['fphi1']:8.4f}")
    print("=" * 116)

    # Try fitting K - M ~ c * (asymmetry feature) for asymmetric f
    # Identify asymmetric rows: |skew| > 0.05 OR |mean| > 0.005
    print("\nLooking for upper bound: K - M <= c * Phi(asym) over asymmetric rows.")
    asym_rows = [r for r in rows if abs(r["skew"]) > 0.05 or abs(r["mean"]) > 0.005]
    print(f"  asymmetric rows: {[r['name'] for r in asym_rows]}")

    print("\nCorrelation tests over asymmetric rows: K-M vs feature, ratio (K-M)/feature")
    for feat_name, feat_lambda in [
        ("|skew|",         lambda r: abs(r["skew"])),
        ("skew^2",         lambda r: r["skew"] ** 2),
        ("|mean|",         lambda r: abs(r["mean"])),
        ("mean^2",         lambda r: r["mean"] ** 2),
        ("abs3",           lambda r: r["abs3"]),
        ("fphi",           lambda r: r["fphi"]),
        ("fphi1",          lambda r: r["fphi1"]),
    ]:
        vals = []
        for r in asym_rows:
            phi = feat_lambda(r)
            if phi > 1e-12:
                vals.append((r["K-M"], phi, r["K-M"] / phi, r["name"]))
        if vals:
            ratios = np.array([v[2] for v in vals])
            print(f"  {feat_name:12s}: ratio (K-M)/Phi: "
                  f"min={ratios.min():.4f}  max={ratios.max():.4f}  "
                  f"max/min={ratios.max()/max(ratios.min(),1e-12):.2f}")

    # Sym sanity
    print("\nSymmetric pdfs: should have K - M = K - K = 0 (M = K trivially)")
    for r in rows:
        if abs(r["skew"]) <= 0.05 and abs(r["mean"]) <= 0.005:
            print(f"  {r['name']:16s}: K-M = {r['K-M']:9.5f}  "
                  f"(M = {r['M']:.4f}, K = {r['K']:.4f})")

    # Extra diagnostic: check whether K - M is upper-bounded by K - 1.2802 in
    # all asymmetric cases (the structure that would imply M >= 1.2802 if K >= 2).
    print("\nSidon-relevant check: does K - M <= K - 1.2802 (i.e., M >= 1.2802)?")
    print("(Any 'NO' would already show some asymmetric f goes below 1.2802.)")
    for r in rows:
        verdict = "YES (M>=1.2802)" if r["M"] >= 1.2802 else "NO  (M<1.2802!)"
        print(f"  {r['name']:16s}: M = {r['M']:9.4f}   {verdict}")


if __name__ == "__main__":
    main()

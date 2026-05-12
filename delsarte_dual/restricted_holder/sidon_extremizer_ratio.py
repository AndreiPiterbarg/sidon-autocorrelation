"""Empirical Holder ratio c_emp for MV's near-extremizer.

For MV's 119-cosine test function G_{u,n}(x) = sum_{j=1}^{119} a_j cos(2 pi j x/u),
which is positive on [-1/4, 1/4], we take the corresponding "primal" pdf

    f(x) = G(x) / Z   on [-1/4, 1/4],   Z = int_{-1/4}^{1/4} G(x) dx,

and compute the Holder ratio

    c_emp := ||f*f||_2^2  /  (||f*f||_inf * ||f*f||_1).

This is the empirical Holder constant achieved by the Sidon-regime
extremizer in MV's framework. By Parseval:

    ||f*f||_2^2 = sum_{n in Z} |hat f(n)|^4,
    ||f*f||_1   = (hat f(0))^2 = 1   (f is a pdf).

For real, even f, ||f*f||_inf = (f*f)(0) = ||f||_2^2 = sum_n |hat f(n)|^2.

If c_emp << 1, this is strong empirical evidence that the restricted Holder
hypothesis Hyp_R(c, M_max) holds for MV-class extremizers, since the most
"L^2-tight" pdfs in the Sidon regime would give c approaching the true
extremum c_* (an unknown constant in (log 16/pi, 1)).

CLI:  python -m delsarte_dual.restricted_holder.sidon_extremizer_ratio
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
from mpmath import mpf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delsarte_dual.restricted_holder.conditional_bound import (  # noqa: E402
    MV_COEFFS_119_STR,
)


def _hi_prec_coeffs(dps):
    mp.mp.dps = dps
    return [mp.mpf(s) for s in MV_COEFFS_119_STR]


def _sinc_quarter(xi):
    """Return int_{-1/4}^{1/4} e^{i 2 pi xi x} dx = sin(pi xi/2)/(pi xi).

    Real-valued; defined as 1/2 at xi = 0 by continuity.
    """
    if xi == 0:
        return mpf(1) / 2
    return mp.sin(mp.pi * xi / 2) / (mp.pi * xi)


def hat_f_coeffs(coeffs, u, Nmax: int):
    """Fourier coefficients hat f(n) for n = 0, ..., Nmax.

    f(x) = G(x)/Z restricted to [-1/4, 1/4] with G(x) = sum_j a_j cos(2 pi j x/u).

    int_{-1/4}^{1/4} cos(2 pi j x/u) e^{-i 2 pi n x} dx
        = (1/2) [sinc((j/u - n)/2) + sinc((j/u + n)/2)]

    where sinc(t) = sin(pi t)/(pi t) and we set sinc(0) := 1.

    Returns the *normalized* coefficients (so hat f(0) = 1), and Z.
    """
    u = mpf(u)
    half = mpf(1) / 2

    def integral(j, n):
        # int_{-1/4}^{1/4} cos(2 pi j x/u) e^{-i 2 pi n x} dx
        a = mpf(j) / u - mpf(n)
        b = mpf(j) / u + mpf(n)
        return half * (_sinc_quarter(a) + _sinc_quarter(b))

    raw = []
    for n in range(0, Nmax + 1):
        s = mpf(0)
        for j, aj in enumerate(coeffs, start=1):
            s += aj * integral(j, n)
        raw.append(s)

    Z = raw[0]
    if Z == 0:
        raise ValueError("Z = int G on [-1/4,1/4] = 0; cannot normalize.")
    hat_f = [r / Z for r in raw]
    return hat_f, Z


def f_at_zero(coeffs, u):
    """f(0) = G(0)/Z = sum_j a_j / Z."""
    u = mpf(u)
    G0 = sum(mpf(a) for a in coeffs)
    # Z = int_{-1/4}^{1/4} G = sum_j a_j * sin(pi j/(2u))/(pi j/u)
    pi = mp.pi
    Z = mpf(0)
    for j, aj in enumerate(coeffs, start=1):
        Z += aj * mp.sin(pi * j / (2 * u)) / (pi * j / u)
    return G0 / Z


def f_squared_l2(coeffs, u, Nmax: int = 4000):
    """||f||_2^2 = sum_n |hat f(n)|^2 (Parseval)."""
    hat_f, _ = hat_f_coeffs(coeffs, u, Nmax)
    s = hat_f[0] * hat_f[0]
    for v in hat_f[1:]:
        s += 2 * v * v  # symmetric pair n, -n
    return s


def holder_ratio(coeffs=None, u=None, Nmax: int = 4000, dps: int = 40):
    """Compute c_emp = ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1) for f = G/Z.

    Returns dict with intermediate values:
        ff_inf:  ||f*f||_inf  (= (f*f)(0) = ||f||_2^2 by Parseval, since f is even)
        ff_1:    ||f*f||_1    (= 1, by f is a pdf)
        ff_l22:  ||f*f||_2^2  (= sum_n |hat f(n)|^4)
        c_emp:   the Holder ratio.
        Nmax:    Fourier truncation used.
    """
    mp.mp.dps = dps
    if coeffs is None:
        coeffs = _hi_prec_coeffs(dps)
    if u is None:
        u = mp.mpf("0.638")
    hat_f, Z = hat_f_coeffs(coeffs, u, Nmax)

    # Parseval sums (f is real and even, so hat f(-n) = hat f(n))
    # ||f||_2^2  = sum_n |hat f(n)|^2 = hat f(0)^2 + 2 sum_{n>=1} hat f(n)^2
    # ||f*f||_2^2 = sum_n |hat f(n)|^4 = hat f(0)^4 + 2 sum_{n>=1} hat f(n)^4
    f_l22 = hat_f[0] ** 2
    ff_l22 = hat_f[0] ** 4
    for v in hat_f[1:]:
        f_l22 += 2 * v ** 2
        ff_l22 += 2 * v ** 4

    ff_inf = f_l22  # (f*f)(0) = ||f||_2^2 for even f (autoconvolution at origin)
    ff_1 = mpf(1)   # ||f*f||_1 = ||f||_1^2 = 1 for a pdf
    c_emp = ff_l22 / (ff_inf * ff_1)
    return {
        "ff_inf": ff_inf,
        "ff_1": ff_1,
        "ff_l22": ff_l22,
        "c_emp": c_emp,
        "Z": Z,
        "Nmax": Nmax,
    }


def main():
    print("=" * 72)
    print("Empirical Holder ratio c_emp for MV's 119-cosine near-extremizer")
    print("=" * 72)
    res = holder_ratio(Nmax=4000, dps=40)
    print(f"  Z (= int_{{-1/4,1/4}} G dx)   = {mp.nstr(res['Z'], 20)}")
    print(f"  ||f*f||_inf  (= ||f||_2^2)  = {mp.nstr(res['ff_inf'], 20)}")
    print(f"  ||f*f||_1                  = {mp.nstr(res['ff_1'], 20)}")
    print(f"  ||f*f||_2^2                = {mp.nstr(res['ff_l22'], 20)}")
    print(f"  c_emp = ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1)")
    print(f"        = {mp.nstr(res['c_emp'], 20)}")
    print(f"  Fourier truncation Nmax = {res['Nmax']}")
    print()
    if res["c_emp"] < mpf("0.99"):
        print(f"  --> c_emp = {mp.nstr(res['c_emp'], 6)} < 0.99: strong evidence")
        print(f"      that Hyp_R holds for MV-class extremizers.")
    else:
        print(f"  --> c_emp = {mp.nstr(res['c_emp'], 6)} >= 0.99: Hyp_R is borderline")
        print(f"      for MV-class extremizers; conditional theorem at risk.")


if __name__ == "__main__":
    main()

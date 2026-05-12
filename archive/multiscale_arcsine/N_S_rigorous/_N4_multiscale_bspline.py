"""Agent N4: Multi-scale B-spline kernels with CLOSED-FORM K_2.

Motivation (alternative-path)
----------------------------
K26 multi-scale arcsine numerically hits M_cert ~ 1.290 but the cross-Bessel
integral I(d1,d2) = ∫ J_0(π d1 ξ)^2 · J_0(π d2 ξ)^2 dξ has no known closed
form, so the rigorous Minkowski bound on K_2 is too loose to deduce a
rigorous M_cert > MV.

B-spline kernels SIDESTEP this entirely:
  K_hat(ξ; δ, n) = sinc(π δ ξ / n)^(2n)
is the FT of the cardinal B-spline of order 2n scaled to width δ.  This is
the 2n-fold convolution of the box of width δ/n with unit mass, hence a
piecewise polynomial of degree 2n-1 supported on [-δ, δ].

Multi-scale ansatz:
  K_hat(ξ) = Σ_i λ_i sinc(π δ_i ξ / n_i)^(2 n_i),       Σ λ_i = 1, λ_i ≥ 0.
Bochner-positive (sum of non-negative powers of sinc^{2n}).

By Parseval:
  K_2 = ∫ K_hat(ξ)^2 dξ
      = Σ_{i,j} λ_i λ_j  C_{ij},   C_{ij} = ∫ B_{2 n_i}(x; δ_i / n_i) ·
                                                  B_{2 n_j}(x; δ_j / n_j) dx
where B_{2n}(x; h) is the 2n-fold auto-conv of the unit-mass box of width h.

EACH C_{ij} is the inner product of two compactly-supported piecewise
polynomials and is a RATIONAL number in (δ_i/n_i, δ_j/n_j) (when the
ratio is rational).  We compute it exactly via sympy.

Comparison points
-----------------
Helper-arcsine M_cert = 1.26990 (helper's fixed-G calibration).
K23 single-scale B-spline n=1 (= triangle / sinc^2 K_hat):
  M_cert = 1.21612.
K23 n=2:  M_cert = 1.15804  (worse: pulled down by S_1 blow-up at j=119/U
near sinc zero).
K23 n>=3: M_cert = None (helper QP S_1 = inf at exact sinc zero
crossings).

Hypothesis: multi-scale B-spline mixing different (δ_i, n_i) breaks the
QP zero alignment and may give a workable finite M_cert.  Even if
M_cert < 1.27481, the RIGOROUS bound is attainable via exact piecewise-
polynomial K_2.

This script computes M_cert under the helper's *fixed-G* assumption (the
re-opt G would be larger; this is conservative numerically but matches the
calibration baseline 1.26990).
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    MV_COEFFS,
    N_QP,
    U,
    mv_master_M_cert,
)


# =====================================================================
# Part 1.  Closed-form K_2 (Parseval / piecewise-polynomial convolution).
# =====================================================================
import sympy as sp


def _box_pp(h):
    """Return cardinal box of width 2h centred at 0 as a tuple
    (breakpoints, polys) representing a piecewise-polynomial with unit mass.

    Breakpoints: list of sympy.Rational of length k+1.
    Polys: list of sympy polynomial expressions (in symbol x) of length k.
    Each poly is active on (breakpoints[i], breakpoints[i+1]).

    Box of width 2h, mass 1: value 1/(2h) on [-h, h], else 0.
    """
    h = sp.Rational(h)
    x = sp.symbols('x')
    return (
        [-h, h],
        [sp.Rational(1) / (2 * h)],
        x,
    )


def _pp_conv(pp_a, pp_b):
    """Convolve two compactly-supported piecewise-polynomials.

    Inputs: (bps_a, polys_a, x), (bps_b, polys_b, x_b).
    Output: (bps_c, polys_c, x).

    Method: for each pair (interval_a, interval_b), the convolution of two
    poly pieces over their respective intervals is computed symbolically.
    The resulting piecewise polynomial in `t = x` is the sum over all such
    pairs, evaluated at each output interval.

    For simplicity & robustness here we compute by:
      1.  enumerate breakpoints of c: all sums of a-bps + b-bps.
      2.  on each c-piece, integrate symbolically over the overlap region.
    """
    bps_a, polys_a, x = pp_a
    bps_b, polys_b, x_b = pp_b
    t, s = sp.symbols('t s')
    # All possible c-breakpoints
    c_bps = sorted({sp.simplify(a + b) for a in bps_a for b in bps_b})

    polys_c = []
    # The convolution: (a*b)(t) = ∫ a(s) b(t-s) ds  over s in [bps_a[0], bps_a[-1]]
    for i in range(len(c_bps) - 1):
        # Take midpoint of c-piece to determine which pieces of a,b overlap
        t_lo = c_bps[i]
        t_hi = c_bps[i + 1]
        t_mid = (t_lo + t_hi) / 2
        # Sum contributions from each (k, l) pair of pieces in a, b
        # respectively where a's piece [bps_a[k], bps_a[k+1]] and
        # b's piece (in s' = t - s coords) at the chosen t_mid.
        total = sp.Rational(0)
        for k in range(len(polys_a)):
            a_lo, a_hi = bps_a[k], bps_a[k + 1]
            a_poly = polys_a[k].subs(x, s)
            for l in range(len(polys_b)):
                b_lo, b_hi = bps_b[l], bps_b[l + 1]
                b_poly = polys_b[l].subs(x_b, t - s)
                # Active region in s: overlap of [a_lo, a_hi] with [t-b_hi, t-b_lo]
                s_lo = sp.Max(a_lo, t_mid - b_hi)
                s_hi = sp.Min(a_hi, t_mid - b_lo)
                # If overlap is empty at t_mid, skip
                if sp.simplify(s_hi - s_lo) <= 0:
                    continue
                # Now we know on the c-piece (t_lo, t_hi), the s-bounds
                # are linear in t.  Recompute with symbolic t.
                s_lo_t = sp.Piecewise((a_lo, a_lo > t - b_hi), (t - b_hi, True))
                # Simplification: at the midpoint, s_lo = max(a_lo, t - b_hi).
                # Use whichever is achieved at the midpoint.
                if sp.simplify(a_lo - (t_mid - b_hi)) >= 0:
                    s_lo_sym = a_lo
                else:
                    s_lo_sym = t - b_hi
                if sp.simplify(a_hi - (t_mid - b_lo)) <= 0:
                    s_hi_sym = a_hi
                else:
                    s_hi_sym = t - b_lo
                integrand = sp.expand(a_poly * b_poly)
                contrib = sp.integrate(integrand, (s, s_lo_sym, s_hi_sym))
                total = total + contrib
        polys_c.append(sp.expand(total.subs(t, x)))

    return (c_bps, polys_c, x)


def cardinal_bspline_pp(n, h):
    """Cardinal B-spline of order n: n-fold auto-conv of box of width 2h
    (centred at 0), unit mass.  Returns piecewise-polynomial (bps, polys, x).

    Support: [-n h, n h].  Polynomial degree: n-1.
    """
    if n == 1:
        return _box_pp(h)
    pp = _box_pp(h)
    for _ in range(n - 1):
        pp = _pp_conv(pp, _box_pp(h))
    return pp


def pp_innerprod(pp_a, pp_b):
    """Compute ∫_R a(x) b(x) dx exactly (rational) for two pp.

    Output: sympy Rational.
    """
    bps_a, polys_a, x = pp_a
    bps_b, polys_b, x_b = pp_b
    # The product is piecewise poly on the union of breakpoints.
    bps_all = sorted({sp.Rational(p) for p in (bps_a + bps_b)})
    # Clip to common support
    lo = max(bps_a[0], bps_b[0])
    hi = min(bps_a[-1], bps_b[-1])
    bps_all = [p for p in bps_all if lo <= p <= hi]
    if len(bps_all) < 2:
        return sp.Rational(0)
    total = sp.Rational(0)
    for i in range(len(bps_all) - 1):
        x_lo = bps_all[i]
        x_hi = bps_all[i + 1]
        x_mid = (x_lo + x_hi) / 2
        # Find which piece of a is active on (x_lo, x_hi)
        ka = next(
            (k for k in range(len(polys_a))
             if bps_a[k] <= x_mid < bps_a[k + 1]),
            None,
        )
        kb = next(
            (k for k in range(len(polys_b))
             if bps_b[k] <= x_mid < bps_b[k + 1]),
            None,
        )
        if ka is None or kb is None:
            continue
        a_p = polys_a[ka]
        b_p = polys_b[kb].subs(x_b, x)
        integrand = sp.expand(a_p * b_p)
        contrib = sp.integrate(integrand, (x, x_lo, x_hi))
        total = total + contrib
    return sp.nsimplify(total, rational=True)


def cross_integral_closed_form(delta_i, n_i, delta_j, n_j, verbose=False):
    """Compute  C_{ij} = ∫ sinc(π δ_i ξ / n_i)^(2 n_i) · sinc(π δ_j ξ / n_j)^(2 n_j) dξ

    By Parseval, this equals ∫ B_{2 n_i}(x; δ_i / n_i) · B_{2 n_j}(x; δ_j / n_j) dx
    where B_{2n}(x; h) is the 2n-fold auto-conv of the unit-mass box of
    width 2 (h/2) = h.

    Wait — let's be careful with the convention.  sinc(π a ξ) is the FT of
    the box of width 2/(2π)/... actually:
        FT[ (1/a) * 1_{[-a/2, a/2]} ](ξ) = sinc(π a ξ).
    Equivalently the unit-mass box of width a, called rect_a, has
    FT[rect_a](ξ) = sinc(π a ξ).  So sinc(π a ξ)^(2n) is the FT of the
    2n-fold convolution of rect_a, which is the cardinal B-spline of
    order 2n with scale a.

    Cardinal B-spline of order m with scale a: m-fold auto-conv of
    rect_a (the unit-mass indicator on [-a/2, a/2]).  Support:
    [-m a / 2, m a / 2].  Polynomial degree: m - 1.

    Our sinc-power is sinc(π δ ξ / n)^(2n) ↔ rect_{δ/n}^{*2n}, which has
    support [-2n · (δ/n) / 2, ·] = [-δ, δ].

    So C_{ij} = ∫ B_i(x) B_j(x) dx where B_i is m=2n_i-fold auto-conv
    of unit-mass box of width δ_i/n_i, supp [-δ_i, δ_i].
    """
    h_i = sp.Rational(delta_i) / sp.Rational(n_i)
    h_j = sp.Rational(delta_j) / sp.Rational(n_j)
    # cardinal_bspline_pp(m, h) returns m-fold auto-conv of unit-mass box
    # of half-width h, i.e. mass-1 indicator on [-h, h] of value 1/(2h).
    # So we need the half-width = (δ_i/n_i)/2 to get a unit-mass box of
    # full width δ_i/n_i.
    half_h_i = h_i / 2
    half_h_j = h_j / 2
    m_i = 2 * n_i
    m_j = 2 * n_j
    if verbose:
        print(f"  Computing B-spline of order {m_i} with half-width {half_h_i}")
    pp_i = cardinal_bspline_pp(m_i, half_h_i)
    if verbose:
        print(f"  Computing B-spline of order {m_j} with half-width {half_h_j}")
    pp_j = cardinal_bspline_pp(m_j, half_h_j)
    C_ij = pp_innerprod(pp_i, pp_j)
    return C_ij


# =====================================================================
# Part 2.  Numerical evaluator for M_cert (matches helper conventions).
# =====================================================================

def K_hat_msbspline(xi, scales, lambdas):
    """K_hat(ξ) = Σ_i λ_i sinc(π δ_i ξ / n_i)^(2 n_i)."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for (delta, n), lam in zip(scales, lambdas):
        # numpy sinc(z) = sin(pi z)/(pi z), so to get sin(pi a ξ)/(pi a ξ)
        # we pass a*ξ as argument.
        z = delta * xi / n
        out = out + lam * np.sinc(z) ** (2 * n)
    return out


# Xi grid for K_2 numerical sanity check (closed-form is the rigorous answer).
N_XI = 80001
XI_MAX = 800.0
_XI = np.linspace(0.0, XI_MAX, N_XI)
_DXI = _XI[1] - _XI[0]


def K_2_closed_form(scales, lambdas):
    """K_2 = Σ_{i,j} λ_i λ_j C_{ij}, where C_{ij} via pp inner product."""
    n_scales = len(scales)
    # Cache cross integrals
    cache = {}
    for i in range(n_scales):
        for j in range(i, n_scales):
            di, ni = scales[i]
            dj, nj = scales[j]
            cache[(i, j)] = cross_integral_closed_form(di, ni, dj, nj)
            if i != j:
                cache[(j, i)] = cache[(i, j)]
    K_2 = sp.Rational(0)
    for i in range(n_scales):
        for j in range(n_scales):
            K_2 = K_2 + sp.Rational(lambdas[i]) * sp.Rational(lambdas[j]) * cache[(i, j)]
    return float(K_2), cache


def K_2_numerical(scales, lambdas):
    """K_2 = ∫ K_hat(ξ)^2 dξ via dense trapezoidal."""
    Kh = K_hat_msbspline(_XI, scales, lambdas)
    K_2_pos = float(np.trapezoid(Kh ** 2, dx=_DXI))
    return 2.0 * K_2_pos


def evaluate_msbspline(scales, lambdas, label, use_closed_form=True, verbose=True):
    """Compute M_cert for K_hat = Σ_i λ_i sinc(π δ_i ξ / n_i)^(2 n_i)."""
    scales = [(float(d), int(n)) for d, n in scales]
    lambdas = np.asarray(lambdas, dtype=float)
    if not np.isclose(lambdas.sum(), 1.0, atol=1e-10):
        raise ValueError(f"lambdas must sum to 1, got {lambdas.sum()}")
    if np.any(lambdas < -1e-12):
        raise ValueError(f"lambdas must be non-negative, got {lambdas}")
    if any(d > DELTA + 1e-12 for d, _ in scales):
        raise ValueError(f"all δ_i must be <= DELTA={DELTA}, got {scales}")

    # k_1 = K_hat(1)
    k_1 = float(K_hat_msbspline(np.array([1.0]), scales, lambdas)[0])

    # K_2: closed-form (rational) + numerical sanity.
    cf_info = None
    K_2_num = K_2_numerical(scales, lambdas)
    if use_closed_form:
        try:
            K_2_cf, cache = K_2_closed_form(scales, lambdas)
            cf_info = {"K_2_closed_form_float": float(K_2_cf),
                       "K_2_numerical": K_2_num,
                       "diff": float(K_2_cf - K_2_num),
                       "C_ij_cache": {str(k): str(v) for k, v in cache.items()}}
            K_2 = float(K_2_cf)
        except Exception as e:
            cf_info = {"closed_form_error": str(e)}
            K_2 = K_2_num
    else:
        K_2 = K_2_num

    # S_1 = Σ a_j^2 / K_hat(j/U)
    if MV_COEFFS is None:
        return {"label": label, "M_cert": None, "reason": "no MV coeffs"}
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_msbspline(qp_xi, scales, lambdas)
    if np.any(kh_qp < 1e-15):
        return {"label": label, "M_cert": None,
                "reason": "K_hat(j/U) ~ 0 at some j (sinc zero)",
                "k_1": k_1,
                "K_2": K_2,
                "kh_qp_min": float(np.min(kh_qp)),
                "kh_qp_min_idx": int(np.argmin(kh_qp))}
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    out = {
        "label": label,
        "scales": scales,
        "lambdas": lambdas.tolist(),
        "k_1": k_1,
        "K_2": K_2,
        "K_2_numerical": K_2_num,
        "S_1": S_1,
        "M_cert": M_cert,
        "beats_MV": (M_cert is not None and M_cert > 1.27481),
        "beats_127": (M_cert is not None and M_cert > 1.270),
        "closed_form_info": cf_info,
    }
    if verbose:
        Mtxt = f"{M_cert:.5f}" if M_cert is not None else "None"
        K2cf = f"{cf_info['K_2_closed_form_float']:.5f}" if (
            cf_info and "K_2_closed_form_float" in cf_info) else "n/a"
        print(f"[{label}] k_1={k_1:.5f}  K_2={K_2:.4f} (cf={K2cf})  "
              f"S_1={S_1:.2e}  M_cert={Mtxt}  beats_MV={out['beats_MV']}")
    return out


# =====================================================================
# Part 3.  Sanity check: closed-form K_2 vs numerical, on simple cases.
# =====================================================================

def sanity_closed_form():
    """Verify K_2_closed_form matches K_2_numerical for single-scale B-spline."""
    print("\n=== Sanity: closed-form K_2 vs numerical ===")
    sanity_cases = [
        # (scales, lambdas, label)
        ([(0.138, 1)], [1.0], "n=1, delta=DELTA"),
        ([(0.138, 2)], [1.0], "n=2, delta=DELTA"),
        ([(0.100, 1)], [1.0], "n=1, delta=0.1"),
        ([(0.138, 1), (0.07, 1)], [0.5, 0.5], "mix n1=n2=1"),
        ([(0.138, 2), (0.10, 1)], [0.6, 0.4], "mix n1=2,n2=1"),
    ]
    results = []
    for scales, lambdas, label in sanity_cases:
        try:
            res = evaluate_msbspline(scales, lambdas, label, use_closed_form=True,
                                     verbose=True)
            cf = res.get("closed_form_info") or {}
            if "K_2_closed_form_float" in cf:
                print(f"  closed form = {cf['K_2_closed_form_float']:.10f}")
                print(f"  numerical    = {cf['K_2_numerical']:.10f}")
                print(f"  diff         = {cf['diff']:.3e}")
            results.append(res)
        except Exception as e:
            print(f"  [{label}] failed: {e}")
            results.append({"label": label, "error": str(e)})
    return results


# =====================================================================
# Part 4.  Multi-scale B-spline parameter sweep.
# =====================================================================

def sweep_two_component(delta_1, n_1, delta_2_list, n_2_list, lambda_1_list):
    """Sweep (δ_2, n_2, λ_1) for fixed (δ_1, n_1)."""
    results = []
    best = {"M_cert": -np.inf, "config": None}
    for n_2 in n_2_list:
        for d_2 in delta_2_list:
            for lam in lambda_1_list:
                label = (f"d1={delta_1:.4f},n1={n_1},"
                         f"d2={d_2:.4f},n2={n_2},l1={lam:.2f}")
                try:
                    res = evaluate_msbspline(
                        [(delta_1, n_1), (d_2, n_2)],
                        [lam, 1.0 - lam],
                        label,
                        use_closed_form=False,  # speed: defer to top-K
                        verbose=False,
                    )
                except Exception as e:
                    res = {"label": label, "M_cert": None, "error": str(e)}
                rec = {
                    "delta_1": float(delta_1),
                    "n_1": int(n_1),
                    "delta_2": float(d_2),
                    "n_2": int(n_2),
                    "lambda_1": float(lam),
                    "M_cert": res.get("M_cert"),
                    "k_1": res.get("k_1"),
                    "K_2": res.get("K_2"),
                    "S_1": res.get("S_1"),
                    "reason": res.get("reason"),
                }
                results.append(rec)
                M = res.get("M_cert")
                if M is not None and M > best["M_cert"]:
                    best = {
                        "M_cert": float(M),
                        "config": rec,
                    }
    return results, best


def main():
    print("=" * 78)
    print("Agent N4: Multi-scale B-spline kernels with closed-form K_2")
    print("=" * 78)

    # --- Closed-form sanity ---
    t0 = time.time()
    sanity = sanity_closed_form()
    print(f"  (sanity took {time.time() - t0:.1f}s)")

    # --- Single-scale baselines on helper (for calibration). ---
    print("\n=== Single-scale baselines ===")
    print("(Compare to K23: n=1 -> 1.21612, n=2 -> 1.15804, n>=3 -> None.)")
    single_base = {}
    for n in [1, 2]:
        res = evaluate_msbspline([(DELTA, n)], [1.0],
                                 f"single n={n}",
                                 use_closed_form=False,
                                 verbose=True)
        single_base[n] = res

    # --- TWO-COMPONENT SWEEP ---
    print("\n=== Two-component sweep (different scales / orders) ===")
    # First sweep: same n=1 (triangle), vary delta_2 and lambda_1.  This avoids
    # sinc^(2n) zero alignment issues for n>=2.
    print("\n--- Sweep A: n_1 = n_2 = 1 (triangle K_hat = sinc^2), vary delta ---")
    delta_2_list_A = [0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    lambda_1_list_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    res_A, best_A = sweep_two_component(
        delta_1=DELTA, n_1=1,
        delta_2_list=delta_2_list_A,
        n_2_list=[1],
        lambda_1_list=lambda_1_list_A,
    )
    print(f"Sweep A best: {best_A}")

    # Second sweep: n_1 = 1, n_2 = 2.
    print("\n--- Sweep B: n_1 = 1, n_2 = 2 (mixed orders) ---")
    res_B, best_B = sweep_two_component(
        delta_1=DELTA, n_1=1,
        delta_2_list=[0.04, 0.06, 0.08, 0.10, 0.12, 0.13],
        n_2_list=[2],
        lambda_1_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    print(f"Sweep B best: {best_B}")

    # Third sweep: n_1 = 2, n_2 = 1 (Hann mixed with triangle).
    print("\n--- Sweep C: n_1 = 2, n_2 = 1 ---")
    res_C, best_C = sweep_two_component(
        delta_1=DELTA, n_1=2,
        delta_2_list=[0.04, 0.06, 0.08, 0.10, 0.12, 0.13],
        n_2_list=[1],
        lambda_1_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    print(f"Sweep C best: {best_C}")

    # Fourth sweep: tiny-delta_2 "QP filler" with HIGH lambda_1.
    # Mechanism: tiny delta_2 makes sinc(pi delta_2 xi)^(2n) ~ 1 over the
    # QP frequency range [1/U, 119/U], filling in the deep nulls of the
    # main triangle, dramatically lowering S_1.
    print("\n--- Sweep D: tiny delta_2 'QP filler', high lambda_1 ---")
    res_D, best_D = sweep_two_component(
        delta_1=DELTA, n_1=1,
        delta_2_list=[0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02,
                      0.025, 0.03, 0.04, 0.05],
        n_2_list=[1, 2, 3, 4],
        lambda_1_list=[0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98],
    )
    print(f"Sweep D best: {best_D}")

    # Pick overall best.
    overall_best = max(
        [best_A, best_B, best_C, best_D],
        key=lambda b: b["M_cert"] if b["M_cert"] is not None else -np.inf,
    )

    # --- Refine the overall best with closed-form K_2 ---
    print("\n=== Refining overall best with CLOSED-FORM K_2 ===")
    best_cfg = overall_best.get("config")
    if best_cfg is not None:
        d_1, n_1 = best_cfg["delta_1"], best_cfg["n_1"]
        d_2, n_2 = best_cfg["delta_2"], best_cfg["n_2"]
        l_1 = best_cfg["lambda_1"]
        # Convert to rational deltas for exact closed-form
        # (sympy.Rational accepts floats; use limit_denominator for cleanness)
        d_1_r = float(Fraction(d_1).limit_denominator(10000))
        d_2_r = float(Fraction(d_2).limit_denominator(10000))
        print(f"  rationalised: d_1 -> {d_1_r}, d_2 -> {d_2_r}")
        best_refined = evaluate_msbspline(
            [(d_1_r, n_1), (d_2_r, n_2)],
            [l_1, 1.0 - l_1],
            f"refined: d1={d_1_r},n1={n_1},d2={d_2_r},n2={n_2},l1={l_1}",
            use_closed_form=True,
            verbose=True,
        )
    else:
        best_refined = None

    # --- THREE-COMPONENT sweep (if 2-component gives improvement). ---
    three_results = []
    three_best = None
    if (overall_best["M_cert"] is not None and
            overall_best["M_cert"] > 1.21):  # any sub-MV improvement
        print("\n=== Three-component exploration ===")
        # Anchor on (delta_1, n_1) = (DELTA, 1) (always the best anchor),
        # vary second QP-filler scale & third intermediate scale.
        d_1, n_1 = DELTA, 1
        # 3-component grid: anchor + tiny-delta filler + intermediate scale.
        triples = []
        for d_2 in [0.005, 0.01, 0.02]:
            for n_2 in [1, 2, 3]:
                for d_3 in [0.05, 0.10, 0.13]:
                    for n_3 in [1, 2]:
                        for l_1 in [0.90, 0.94, 0.96]:
                            for l_2 in [0.01, 0.02, 0.04, 0.06]:
                                l_3 = 1.0 - l_1 - l_2
                                if l_3 <= 0 or l_3 > 0.5:
                                    continue
                                triples.append((d_2, n_2, d_3, n_3, l_1, l_2, l_3))
        print(f"  exploring {len(triples)} three-component configurations...")
        for d_2, n_2, d_3, n_3, l_1, l_2, l_3 in triples:
            label = (f"3c: d2={d_2},n2={n_2},d3={d_3},n3={n_3},"
                     f"l1={l_1},l2={l_2},l3={l_3:.3f}")
            try:
                res = evaluate_msbspline(
                    [(d_1, n_1), (d_2, n_2), (d_3, n_3)],
                    [l_1, l_2, l_3],
                    label,
                    use_closed_form=False,
                    verbose=False,
                )
            except Exception as e:
                res = {"M_cert": None, "error": str(e)}
            rec = {
                "delta_1": d_1, "n_1": n_1,
                "delta_2": d_2, "n_2": n_2,
                "delta_3": d_3, "n_3": n_3,
                "lambda_1": l_1, "lambda_2": l_2, "lambda_3": l_3,
                "M_cert": res.get("M_cert"),
                "k_1": res.get("k_1"),
                "K_2": res.get("K_2"),
                "S_1": res.get("S_1"),
            }
            three_results.append(rec)
            if three_best is None or (
                    rec["M_cert"] is not None
                    and rec["M_cert"] > (three_best["M_cert"] or -np.inf)):
                three_best = rec
        print(f"Three-component best: {three_best}")

    # --- Save. ---
    out = {
        "family": "multi-scale B-spline (sum of sinc^{2n} K_hat)",
        "helper_arcsine_M_cert_baseline": 1.26990,
        "helper_baseline_single_n1": single_base.get(1, {}).get("M_cert"),
        "helper_baseline_single_n2": single_base.get(2, {}).get("M_cert"),
        "MV_published": 1.27481,
        "sanity_closed_form": [
            {k: v for k, v in r.items() if k != "closed_form_info"}
            for r in sanity if "label" in r
        ],
        "sweep_A": {"results": res_A, "best": best_A},
        "sweep_B": {"results": res_B, "best": best_B},
        "sweep_C": {"results": res_C, "best": best_C},
        "sweep_D_QP_filler": {"results": res_D, "best": best_D},
        "overall_best": overall_best,
        "best_refined_with_closed_form": best_refined,
        "three_component_results": three_results,
        "three_component_best": three_best,
        "beats_MV_1_2748": (overall_best["M_cert"] is not None
                            and overall_best["M_cert"] > 1.27481),
        "beats_127": (overall_best["M_cert"] is not None
                      and overall_best["M_cert"] > 1.270),
    }
    outpath = os.path.join(REPO, "_N4_multiscale_bspline.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: (str(x) if isinstance(x, sp.Rational)
                                     else float(x) if hasattr(x, "item") else x))
    print(f"\nWrote {outpath}")
    print(f"\nFINAL OVERALL BEST M_cert = {overall_best['M_cert']}")
    if best_refined is not None and best_refined.get("M_cert") is not None:
        print(f"With closed-form K_2: M_cert = {best_refined['M_cert']:.5f}")
        cf = best_refined.get("closed_form_info") or {}
        if "K_2_closed_form_float" in cf:
            print(f"  K_2 (closed form, exact rational): "
                  f"{cf['K_2_closed_form_float']:.10f}")
            print(f"  K_2 (numerical sanity): {cf['K_2_numerical']:.10f}")
            print(f"  diff: {cf['diff']:.3e}")
    return out


if __name__ == "__main__":
    main()

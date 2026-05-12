"""Numerical exploration accompanying the analytical investigation of ||f^*||_infty
for the Sidon optimum.  This file serves as a working notebook supporting the
companion analytical report.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np


# =====================================================================
# Lemma 1 (LB): For any f >= 0, supp(f) subset [-1/4, 1/4], int f = 1:
#   ||f||_2^2 >= 2          (Cauchy-Schwarz)
#   ||f||_infty >= ||f||_2^2 / ||f||_1 = ||f||_2^2 >= 2
#
# Therefore ||f^*||_infty >= 2 unconditionally.
# This is the LB.  The UB is the open question.
# =====================================================================

def verify_lb_from_cauchy_schwarz():
    """Numerical sanity: for various f, verify ||f||_infty >= ||f||_2^2 >= 2."""
    print("=== LB verification: ||f||_infty >= ||f||_2^2 >= 2 ===")
    # Discretize [-1/4, 1/4] at high resolution
    N = 10000
    xs = np.linspace(-0.25, 0.25, N + 1)
    dx = xs[1] - xs[0]

    cases = {
        "uniform": np.ones_like(xs) * 2.0,
        "left-shifted": np.where(xs <= 0, 4.0, 0.0),
        "two-stack": np.where(np.abs(xs) <= 0.125, 4.0, 0.0),
        "triangle-up": 8.0 * np.maximum(0.25 - np.abs(xs), 0) - 0.0,  # approximately ramp
        "asym-step": np.where(xs <= -0.1, 0.0, np.where(xs <= 0.05, 6.6667, 0.0)),
    }
    for name, f in cases.items():
        # Renormalize
        Z = np.trapz(f, xs)
        f = f / Z
        l2sq = np.trapz(f**2, xs)
        linfty = np.max(f)
        # Compute (f*f) and its max
        ff = np.convolve(f, f, mode='full') * dx
        ff_max = np.max(ff)
        check_cs = linfty >= l2sq - 1e-9
        check_l2 = l2sq >= 2 - 1e-3  # numerical slack
        check_supff = ff_max <= linfty + 1e-3
        print(f"  {name:>12s}: ||f||_inf={linfty:.4f}, ||f||_2^2={l2sq:.4f}, sup(f*f)={ff_max:.4f}  "
              f"[CS:{check_cs}, L2>=2:{check_l2}, sup<=Linf:{check_supff}]")


# =====================================================================
# Truncation/Mollification analysis
#
# Given any f with int f = 1, supp f subset [-1/4, 1/4], sup(f*f) = C:
# Construct f' = (f rescaled to support [-(1/4 - eps), 1/4 - eps]) * eta_eps
# where eta_eps is uniform mollifier on [-eps, eps].
#
# Properties (proof in companion report):
#   sup(f' * f') <= (1 - 4 eps)^{-1} * C
#   ||f'||_infty <= ||f||_2 / sqrt(2 eps (1 - 4 eps))
#
# For C in [1.28, 1.51] and ||f||_2 in [sqrt 2, sqrt(C)*?]: numerical bounds.
# =====================================================================

def truncation_bound(C: float, l2: float, eps: float) -> Tuple[float, float]:
    """Given f with sup(f*f)=C, ||f||_2 = l2, eps in (0, 1/4):
    return (sup(f'*f') upper bound, ||f'||_infty upper bound)."""
    sigma = 1 - 4 * eps
    if sigma <= 0:
        return (math.inf, math.inf)
    sup_bound = C / sigma
    linf_bound = l2 / math.sqrt(2 * eps * sigma)
    return sup_bound, linf_bound


def explore_truncation_tradeoff():
    """For each f's (l2, C), find the best eps for trade-off."""
    print("\n=== Truncation/mollification tradeoff ===")
    print("(Goal: small ||f'||_inf at modest cost to sup(f'*f') > C)")
    for C, l2 in [(1.2802, math.sqrt(2)), (1.31, math.sqrt(2)), (1.5, math.sqrt(2)),
                   (1.5, 1.5), (1.5, 2.0)]:
        print(f"\n  Source f: sup(f*f) = C = {C:.4f},  ||f||_2 = {l2:.4f} (||f||_2^2 = {l2**2:.4f})")
        for eps in [0.01, 0.02, 0.05, 0.1, 0.125, 0.15]:
            sup_b, linf_b = truncation_bound(C, l2, eps)
            print(f"    eps={eps:.3f}: sup(f'*f') <= {sup_b:.4f}, ||f'||_inf <= {linf_b:.4f}")


# =====================================================================
# Examine known constructions for ||f||_infty
# =====================================================================

def known_construction_uniform():
    """Uniform on [-1/4, 1/4]:  f = 2,  sup(f*f) = 2,  ||f||_inf = 2."""
    return {"f_inf": 2.0, "sup_ff": 2.0, "name": "uniform"}


def known_construction_two_step(c1: float, c2: float, w1: float, w2: float):
    """Two-step asymmetric:  f = c1 on [a, a+w1], c2 on [a+w1, a+w1+w2],
    centered so support fits [-1/4, 1/4].  Compute sup(f*f) and ||f||_inf.
    """
    # Place support centered: total width = w1 + w2, anchor = -(w1+w2)/2
    if w1 + w2 > 0.5 + 1e-9:
        return None
    if not (c1 * w1 + c2 * w2) > 0:
        return None
    # Renormalize to int = 1
    Z = c1 * w1 + c2 * w2
    c1n, c2n = c1 / Z, c2 / Z
    f_inf = max(c1n, c2n)
    # f*f computation: convolution of two-step with itself
    # f = c1n * 1_[a, a+w1] + c2n * 1_[a+w1, a+w1+w2], a = -(w1+w2)/2
    a = -(w1 + w2) / 2
    # Discrete convolution at high resolution
    NN = 5000
    xs = np.linspace(-0.5, 0.5, 2 * NN + 1)
    f = np.zeros_like(xs)
    f[(xs >= a) & (xs <= a + w1)] = c1n
    f[(xs > a + w1) & (xs <= a + w1 + w2)] = c2n
    dx = xs[1] - xs[0]
    ff = np.convolve(f, f, mode='same') * dx
    sup_ff = np.max(ff)
    # Check int f
    int_f = np.trapz(f, xs)
    return {"f_inf": f_inf, "sup_ff": sup_ff, "int_f": int_f, "params": (c1, c2, w1, w2)}


def explore_two_step_constructions():
    """Sweep two-step constructions, find ones with sup(f*f) close to known UB."""
    print("\n=== Two-step constructions ===")
    print(f"{'c1':>5s} {'c2':>5s} {'w1':>6s} {'w2':>6s} {'||f||_inf':>10s} {'sup(f*f)':>10s}")
    best = None
    for c1 in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for c2 in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for w1 in [0.10, 0.15, 0.2, 0.25, 0.3]:
                for w2 in [0.10, 0.15, 0.2, 0.25, 0.3]:
                    r = known_construction_two_step(c1, c2, w1, w2)
                    if r and abs(r["int_f"] - 1.0) < 1e-2 and r["sup_ff"] < 2.0:
                        if best is None or r["sup_ff"] < best["sup_ff"]:
                            best = r
                            print(f"  {c1:>5.2f} {c2:>5.2f} {w1:>6.3f} {w2:>6.3f} "
                                  f"{r['f_inf']:>10.4f} {r['sup_ff']:>10.4f}  <- new best")
    print(f"\n  Best two-step found: sup(f*f) = {best['sup_ff']:.4f}, ||f||_inf = {best['f_inf']:.4f}")


def explore_three_step_constructions():
    """Three-step f for tighter sup(f*f), examine ||f||_inf."""
    print("\n=== Three-step constructions ===")
    best = None
    cnt = 0
    NN = 3000
    xs = np.linspace(-0.5, 0.5, 2 * NN + 1)
    dx = xs[1] - xs[0]
    for c1 in np.linspace(0.2, 4.0, 12):
        for c2 in np.linspace(0.2, 4.0, 12):
            for c3 in np.linspace(0.2, 4.0, 12):
                for w1 in [0.08, 0.12, 0.16]:
                    for w2 in [0.10, 0.15, 0.2]:
                        for w3 in [0.08, 0.12, 0.16]:
                            if w1 + w2 + w3 > 0.5 + 1e-9:
                                continue
                            Z = c1 * w1 + c2 * w2 + c3 * w3
                            if Z < 1e-6:
                                continue
                            cn = [c1/Z, c2/Z, c3/Z]
                            f_inf = max(cn)
                            a = -(w1 + w2 + w3) / 2
                            f = np.zeros_like(xs)
                            f[(xs >= a) & (xs <= a + w1)] = cn[0]
                            f[(xs > a + w1) & (xs <= a + w1 + w2)] = cn[1]
                            f[(xs > a + w1 + w2) & (xs <= a + w1 + w2 + w3)] = cn[2]
                            ff = np.convolve(f, f, mode='same') * dx
                            sup_ff = np.max(ff)
                            cnt += 1
                            if best is None or sup_ff < best["sup_ff"]:
                                best = {"f_inf": f_inf, "sup_ff": sup_ff,
                                        "cn": cn, "ws": (w1, w2, w3)}
    print(f"  Searched {cnt} three-step configs.")
    print(f"  Best three-step: sup(f*f) = {best['sup_ff']:.4f}, "
          f"||f||_inf = {best['f_inf']:.4f}, c_n = {best['cn']}, w = {best['ws']}")


if __name__ == "__main__":
    verify_lb_from_cauchy_schwarz()
    explore_truncation_tradeoff()
    explore_two_step_constructions()
    explore_three_step_constructions()

"""Master K26-Hybrid: Convex combinations of *different* Bochner-admissible
kernel families, each of compact support [-delta_i, delta_i] (delta_i <= DELTA),
each normalised so int K_i = 1 (equivalently K_hat_i(0) = 1).

Construction
============
    K(x) = sum_i  lambda_i  *  K_i(x; delta_i, family_i)
    sum lambda_i = 1,  lambda_i >= 0,  delta_i <= DELTA.

By convex combination Bochner OK iff each K_hat_i >= 0. We enforce that by
choice of family. Each family's K_hat_i is given in closed form so we
evaluate via the K_hat-side pipeline (same as agent K26 multi-scale arcsine).

Catalog of Bochner-admissible families with support in [-delta, delta]
----------------------------------------------------------------------
Let z = pi * delta * xi (so z(0) = 0 and K_hat(0) = 1 in each case).

 1. ARCSINE  (a.k.a. MV's choice).  K = (1/delta) * (beta * beta)(x/delta)
    where beta is the arcsine density on (-1/2, 1/2).
        K_hat(xi) = J_0(pi delta xi)^2 = J_0(z)^2.

 2. TRIANGLE / FEJER.  K = autoconv of box[-delta/2, delta/2] normalised so
    int K = 1.   This is the triangle on [-delta, delta] with peak 1/delta.
        K_hat(xi) = sinc^2(z) where sinc(z) = sin(z)/z, sinc(0)=1.

 3. B-SPLINE-B3 / sinc^4 (cubic spline).  K = triangle * triangle / delta,
    or equivalently autoconv of triangle on [-delta/2, delta/2]. Support
    [-delta, delta]; smoother than triangle.
        K_hat(xi) = sinc^4(z/2).

 4. EPANECHNIKOV / parabolic.  K(x) = (3/(4*delta)) (1 - (x/delta)^2)_+,
    support [-delta, delta]. Bochner OK (Epanechnikov kernel; positive FT).
        K_hat(xi) = 3 (sin(z) - z cos(z)) / z^3.  (Real, positive; > 0.)

 5. SEMICIRCLE / Wigner.  K(x) = (2/(pi*delta^2)) sqrt(delta^2 - x^2)_+,
    support [-delta, delta]. Bochner OK.
        K_hat(xi) = 2 J_1(z) / z.

 6. RAISED COSINE (Hann window).  K(x) = (1/delta)(1 + cos(pi x/delta))/2,
    x in [-delta, delta], else 0. K = phi*phi where phi has support
    [-delta/2, delta/2]... no, the *kernel* itself supported on [-delta, delta]
    is the raised cosine.  Need Bochner check. K_hat(xi) involves:
        ah_hat(xi) ~ sin(z) (z^2 / (z^2 - pi^2)) * (something).  Use numerics.
    SKIP: Bochner-positivity not obvious; we drop this family for safety.

 6'. We instead use COSINE-LIFTED ARCSINE: omit; covered by mix arcsine + tri.

All families above have closed-form K_hat with no zeros at j/U for j = 1..119
when delta is chosen near DELTA = 0.138 (verified numerically per evaluation).

Pipeline
========
Reuse the K_hat-side evaluator from _agent_K26_multiscale_arcsine.py but
generalise K_hat to a sum over (lambda_i, family_i, delta_i) triples.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.special import j0, j1

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    MV_COEFFS,
    N_QP,
    U,
    mv_master_M_cert,
)

# ---------------------------------------------------------------------------
# Xi grid for K_2 integration (Parseval: K_2 = int K_hat^2 dxi).
# Match K26 settings.
# ---------------------------------------------------------------------------
N_XI = 40001
XI_MAX = 600.0
_XI = np.linspace(0.0, XI_MAX, N_XI)
_DXI = _XI[1] - _XI[0]


# ---------------------------------------------------------------------------
# K_hat closed forms for each family.  Each takes (xi, delta) and returns
# K_hat values.  Each is normalised so K_hat(0) = 1.  Each is >= 0.
# ---------------------------------------------------------------------------
def Khat_arcsine(xi, delta):
    z = np.pi * delta * xi
    return j0(z) ** 2


def _sinc(z):
    # sinc(z) = sin(z)/z, sinc(0) = 1.
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-14
    out[nz] = np.sin(z[nz]) / z[nz]
    return out


def Khat_triangle(xi, delta):
    z = np.pi * delta * xi
    return _sinc(z) ** 2


def Khat_bspline3(xi, delta):
    z = np.pi * delta * xi
    return _sinc(z / 2.0) ** 4


def Khat_epanechnikov(xi, delta):
    """K(x) = (3/(4 delta)) (1 - (x/delta)^2)_+ on [-delta, delta].
    K_hat(xi) = 3 (sin(z) - z cos(z)) / z^3,  z = pi delta xi.
    Real, positive (and -> 1 as z -> 0).
    """
    z = np.pi * delta * xi
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-6
    zz = z[nz]
    out[nz] = 3.0 * (np.sin(zz) - zz * np.cos(zz)) / (zz ** 3)
    # Taylor at z = 0: 1 - z^2/10 + z^4/280 - ...
    z0 = z[~nz]
    out[~nz] = 1.0 - z0 ** 2 / 10.0 + z0 ** 4 / 280.0
    return out


def Khat_semicircle(xi, delta):
    """K(x) = (2/(pi delta^2)) sqrt(delta^2 - x^2)_+ on [-delta, delta].
    K_hat(xi) = 2 J_1(z) / z,  z = pi delta xi.
    """
    z = np.pi * delta * xi
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-8
    out[nz] = 2.0 * j1(z[nz]) / z[nz]
    # Taylor at 0:  2 J_1(z)/z -> 1 - z^2/8 + z^4/192 - ...
    z0 = z[~nz]
    out[~nz] = 1.0 - z0 ** 2 / 8.0 + z0 ** 4 / 192.0
    return out


FAMILIES = {
    "arcsine":      Khat_arcsine,
    "triangle":     Khat_triangle,
    "bspline3":     Khat_bspline3,
    "epanechnikov": Khat_epanechnikov,
    "semicircle":   Khat_semicircle,
}


# ---------------------------------------------------------------------------
# K_hat for a mixture.
# ---------------------------------------------------------------------------
def K_hat_mixture(xi, components):
    """components: list of (family_name, delta, lambda).  sum lambda = 1.
    Returns K_hat(xi) = sum_i lambda_i * K_hat_{family_i}(xi; delta_i).
    """
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for fam, delta, lam in components:
        out = out + lam * FAMILIES[fam](xi, delta)
    return out


def evaluate_mixture(components, label, verbose=True):
    """Evaluate M_cert for a mixture kernel given by components.

    components: list of (family_name, delta, lambda) tuples.
    """
    lams = np.array([c[2] for c in components])
    deltas = np.array([c[1] for c in components])
    if not np.isclose(lams.sum(), 1.0):
        raise ValueError(f"lambdas must sum to 1, got {lams.sum()}: {label}")
    if np.any(lams < -1e-12):
        raise ValueError(f"lambdas must be non-negative, got {lams}: {label}")
    if np.any(deltas > DELTA + 1e-12):
        raise ValueError(f"all delta_i must be <= DELTA={DELTA}, got {deltas}")

    k_1 = float(K_hat_mixture(np.array([1.0]), components)[0])

    # K_2 = 2 * int_0^inf K_hat^2 dxi  (K_hat even).
    Kh = K_hat_mixture(_XI, components)
    if np.any(Kh < -1e-10):
        return {"label": label, "M_cert": None,
                "reason": f"K_hat negative on grid: min={Kh.min()}"}
    K_2_pos = np.trapezoid(Kh ** 2, dx=_DXI)
    K_2 = float(2.0 * K_2_pos)

    if MV_COEFFS is None:
        return {"label": label, "M_cert": None, "reason": "no MV coeffs"}
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_mixture(qp_xi, components)
    if np.any(kh_qp < 1e-18):
        return {"label": label, "M_cert": None,
                "reason": f"K_hat(j/U)~0 at some j: min={kh_qp.min()}"}
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    out = {
        "label": label,
        "components": [(c[0], float(c[1]), float(c[2])) for c in components],
        "k_1": k_1,
        "K_2": K_2,
        "S_1": S_1,
        "M_cert": M_cert,
        "beats_MV_1.2748": (M_cert is not None and M_cert > 1.2748),
        "beats_K26_1.28013": (M_cert is not None and M_cert > 1.28013),
    }
    if verbose:
        Mtxt = f"{M_cert:.5f}" if M_cert is not None else "None"
        print(f"[{label}] k_1={k_1:.5f} K_2={K_2:.4f} S_1={S_1:.3f} "
              f"M_cert={Mtxt}")
    return out


# ---------------------------------------------------------------------------
# Sanity: pure single-family kernels at delta = DELTA.
# ---------------------------------------------------------------------------
def sanity_pure():
    print("--- Sanity: pure single-family at delta = DELTA ---")
    out = {}
    for fam in FAMILIES:
        res = evaluate_mixture([(fam, DELTA, 1.0)], f"pure-{fam}")
        out[fam] = res
    return out


# ---------------------------------------------------------------------------
# Pairwise hybrid sweep.
# Pair (fam_A, fam_B) at (delta_A = DELTA, delta_B variable, lambda_A variable).
# ---------------------------------------------------------------------------
def hybrid_pair_sweep(fam_A, fam_B, delta_B_list, lambda_A_list,
                      delta_A=DELTA, verbose=True):
    results = []
    best = {"M_cert": -np.inf}
    for dB in delta_B_list:
        for lA in lambda_A_list:
            components = [(fam_A, delta_A, lA), (fam_B, dB, 1.0 - lA)]
            label = f"{fam_A}@{delta_A:.3f}/lA={lA:.2f} + {fam_B}@{dB:.3f}"
            res = evaluate_mixture(components, label, verbose=False)
            rec = {
                "fam_A": fam_A, "fam_B": fam_B,
                "delta_A": float(delta_A), "delta_B": float(dB),
                "lambda_A": float(lA),
                **{k: v for k, v in res.items()
                   if k not in ("label", "components")},
            }
            results.append(rec)
            M = res.get("M_cert")
            if M is not None and M > best["M_cert"]:
                best = {"M_cert": float(M),
                        "fam_A": fam_A, "fam_B": fam_B,
                        "delta_A": float(delta_A), "delta_B": float(dB),
                        "lambda_A": float(lA)}
    if verbose:
        if best.get("M_cert", -np.inf) > -np.inf:
            print(f"  best ({fam_A}+{fam_B}): M_cert={best['M_cert']:.5f} "
                  f"at delta_B={best['delta_B']}, lambda_A={best['lambda_A']}")
        else:
            print(f"  ({fam_A}+{fam_B}): no valid M_cert in sweep")
    return results, best


# ---------------------------------------------------------------------------
# Refined search around a coarse best.
# ---------------------------------------------------------------------------
def refine_pair(best, delta_A=DELTA):
    fam_A, fam_B = best["fam_A"], best["fam_B"]
    d0 = best["delta_B"]; l0 = best["lambda_A"]
    dB_lo = max(0.04, d0 - 0.02)
    dB_hi = min(DELTA - 1e-4, d0 + 0.02)
    dB_list = list(np.round(np.linspace(dB_lo, dB_hi, 9), 5))
    l_lo = max(0.05, l0 - 0.10)
    l_hi = min(0.95, l0 + 0.10)
    lA_list = list(np.round(np.linspace(l_lo, l_hi, 9), 4))
    print(f"  refine around delta_B={d0}, lambda_A={l0}: "
          f"{len(dB_list)} x {len(lA_list)}")
    return hybrid_pair_sweep(fam_A, fam_B, dB_list, lA_list, delta_A,
                             verbose=True)


# ---------------------------------------------------------------------------
# Three-component sweep (mixture of 3 different families).
# ---------------------------------------------------------------------------
def three_way_search(top_pair_best, third_families):
    fam_A, fam_B = top_pair_best["fam_A"], top_pair_best["fam_B"]
    dA = top_pair_best["delta_A"]; dB = top_pair_best["delta_B"]
    lA = top_pair_best["lambda_A"]
    # Reserve some mass for fam_C; reshape:
    results = []
    best = {"M_cert": top_pair_best["M_cert"]}
    delta_C_list = [0.05, 0.08, 0.10, 0.12, 0.13]
    lC_list = [0.05, 0.10, 0.15, 0.20, 0.30]
    print(f"--- 3-way: base ({fam_A}, {fam_B}), add third family ---")
    for fam_C in third_families:
        if fam_C in (fam_A, fam_B):
            continue
        for dC in delta_C_list:
            for lC in lC_list:
                # Redistribute: keep ratio lA:(1-lA) between A and B for the
                # remaining 1 - lC mass.
                lAm = lA * (1.0 - lC)
                lBm = (1.0 - lA) * (1.0 - lC)
                comps = [(fam_A, dA, lAm), (fam_B, dB, lBm), (fam_C, dC, lC)]
                label = f"3w:{fam_A}@{dA:.2f}({lAm:.2f})+{fam_B}@{dB:.2f}({lBm:.2f})+{fam_C}@{dC:.2f}({lC:.2f})"
                res = evaluate_mixture(comps, label, verbose=False)
                rec = {
                    "fam_A": fam_A, "fam_B": fam_B, "fam_C": fam_C,
                    "delta_A": dA, "delta_B": dB, "delta_C": float(dC),
                    "lambda_A": lAm, "lambda_B": lBm, "lambda_C": float(lC),
                    **{k: v for k, v in res.items()
                       if k not in ("label", "components")},
                }
                results.append(rec)
                M = res.get("M_cert")
                if M is not None and M > best["M_cert"]:
                    best = {"M_cert": float(M), "fam_A": fam_A,
                            "fam_B": fam_B, "fam_C": fam_C,
                            "delta_A": dA, "delta_B": dB,
                            "delta_C": float(dC),
                            "lambda_A": lAm, "lambda_B": lBm,
                            "lambda_C": float(lC)}
                    print(f"  *NEW 3w best* M={M:.5f}  ({label})")
    return results, best


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("Master K26-Hybrid: convex combos of DIFFERENT Bochner-admissible kernels")
    print(f"DELTA={DELTA}  U={U}  N_XI={N_XI}  XI_MAX={XI_MAX}")
    print(f"Baseline arcsine M_cert  ~= 1.27499 (K26 sanity)")
    print(f"Target to beat (K26)     >= 1.28013 (multi-scale arcsine)")
    print("=" * 78)

    sanity = sanity_pure()
    print()

    # Pairs of distinct families.
    fam_list = list(FAMILIES.keys())
    pairs = [(a, b) for a in fam_list for b in fam_list if a != b]
    # We always put fam_A at delta=DELTA (the "anchor") and sweep delta_B < DELTA.
    # 5 families => 5*4 = 20 ordered pairs.  Both orderings are tested (A,B) and (B,A)
    # because the anchor differs.
    print(f"Will sweep {len(pairs)} ordered pairs (anchor at delta=DELTA).")

    delta_B_list = [0.05, 0.07, 0.09, 0.10, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135]
    lambda_A_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    pair_results = {}
    pair_bests = {}
    global_best = {"M_cert": -np.inf}
    print("--- Pairwise hybrid sweeps (coarse) ---")
    for fam_A, fam_B in pairs:
        print(f"\n>>> Pair: anchor={fam_A}@{DELTA}, scan={fam_B}@delta_B ...")
        res, best = hybrid_pair_sweep(fam_A, fam_B, delta_B_list, lambda_A_list,
                                      verbose=True)
        pair_results[(fam_A, fam_B)] = res
        pair_bests[(fam_A, fam_B)] = best
        if best.get("M_cert", -np.inf) > global_best["M_cert"]:
            global_best = best

    print()
    print("=" * 78)
    print(f"GLOBAL COARSE BEST: {global_best}")
    print("=" * 78)

    # Refine the top 3 pairs.
    pairs_sorted = sorted(pair_bests.items(),
                          key=lambda kv: kv[1].get("M_cert", -np.inf),
                          reverse=True)
    refined = {}
    refined_bests = {}
    for (fa, fb), bst in pairs_sorted[:3]:
        if bst.get("M_cert", -np.inf) > -np.inf:
            print(f"\n--- Refine pair ({fa}, {fb}) "
                  f"coarse M={bst['M_cert']:.5f} ---")
            rr, rb = refine_pair(bst)
            refined[(fa, fb)] = rr
            refined_bests[(fa, fb)] = rb
            if rb.get("M_cert", -np.inf) > global_best["M_cert"]:
                global_best = rb

    print()
    print("=" * 78)
    print(f"GLOBAL BEST after refinement: {global_best}")
    print("=" * 78)

    # 3-way only if we found a competitive pair.
    three_way_results, three_way_best = [], None
    if global_best.get("M_cert", -np.inf) > 1.270:
        three_way_results, three_way_best = three_way_search(
            global_best, list(FAMILIES.keys()))
        if three_way_best.get("M_cert", -np.inf) > global_best["M_cert"]:
            global_best = three_way_best

    elapsed = time.time() - t0
    print()
    print("=" * 78)
    print(f"FINAL GLOBAL BEST: {global_best}")
    print(f"K26 reference   = 1.28013")
    print(f"MV reference    = 1.2748")
    print(f"Elapsed: {elapsed:.1f} s")
    print("=" * 78)

    # Serialize.
    def jsonable(x):
        if isinstance(x, dict):
            return {str(k): jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [jsonable(v) for v in x]
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x

    out = {
        "DELTA": DELTA, "U": U, "N_XI": N_XI, "XI_MAX": XI_MAX,
        "families": list(FAMILIES.keys()),
        "sanity_pure": jsonable(sanity),
        "pair_coarse_bests": jsonable({f"{a}+{b}": v
                                        for (a, b), v in pair_bests.items()}),
        "pair_refined_bests": jsonable({f"{a}+{b}": v
                                         for (a, b), v in refined_bests.items()}),
        "three_way_best": jsonable(three_way_best),
        "global_best": jsonable(global_best),
        "elapsed_s": elapsed,
        "beats_K26_1_28013": (global_best.get("M_cert", -np.inf) > 1.28013),
        "beats_MV_1_2748":   (global_best.get("M_cert", -np.inf) > 1.2748),
    }
    outpath = os.path.join(REPO, "_master_k26_hybrid.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {outpath}")
    return out


if __name__ == "__main__":
    main()

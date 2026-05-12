"""Agent K31: Beurling-Selberg-Vaaler extremal kernels for MV C_{1a}.

Tests several phi(x) constructions inspired by the Beurling-Selberg-Vaaler
classical extremal theory (Vaaler, BAMS 1985; Carneiro-Vaaler 2010; Selberg
unpublished 1974).

PROBLEM CONTEXT.  The MV master inequality uses phi : R -> R, supp(phi) in
[-delta/2, delta/2], phi >= 0, K = phi * phi supported in [-delta, delta],
K_hat = (phi_hat)^2 >= 0 (Bochner automatic).  MV's arcsine baseline gives
M_cert ~ 1.270 with k_1 = 0.909, K_2 = 4.254.

THEORY.  Among entire functions of exponential type 2*pi, Vaaler 1985 gives
the "elementary" non-negative one:
    K(z) = (sin(pi z) / (pi z))^2     (Fejer kernel, the sinc^2)
and the not-non-negative Beurling sign-uncertainty function:
    B(z) = (sin(pi z)/pi)^2 * { sum_{n=0}^inf 1/(z-n)^2
                              - sum_{n=-inf}^{-1} 1/(z-n)^2 + 2/z }
    H(z) = (sin(pi z)/pi)^2 * { sum_{n} sgn(n)/(z-n)^2 + 2/z }
    B(z) = H(z) + K(z),   -B(-z) = H(z) - K(z)
The Selberg majorant of indicator chi_{[a,b]} is:
    C(z) = (1/2) * (B(b - z) + B(z - a))     bandlimited, >= chi_{[a,b]}
The Selberg minorant of chi_{[a,b]}:
    c(z) = (1/2) * (-B(z - b) - B(a - z))    bandlimited, <= chi_{[a,b]}

For OUR problem we need phi(x) supported in [-delta/2, delta/2] in SPACE.
By Paley-Wiener, this means phi_hat is entire of exp type pi*delta, NOT
that phi itself is bandlimited.  So we cannot directly use Selberg
majorants/minorants AS phi (they are bandlimited).  Instead:

  (i) Use Vaaler's positive bandlimited K(z) = sinc^2(pi z) (or rescaled
      and truncated) as a CANDIDATE phi.  Since K decays in space, truncation
      to [-delta/2, delta/2] is a controlled error.

  (ii) Use the dual: phi_hat(xi) = (Selberg majorant evaluated at delta*xi/2)
      is entire of type pi*delta, so its inverse FT phi(x) IS supported in
      [-delta/2, delta/2].  But the Selberg majorant is NOT non-negative,
      so phi(x) may go negative.  We try B(z), H(z), K(z) and combinations
      as phi_hat, then check if phi(x) >= 0 on its support.

  (iii) Use products: phi(x) = K(a*x) * K(b*x), still non-negative,
      bandlimited.  Truncate to [-delta/2, delta/2].

This script implements (i), (ii), (iii) and reports k_1, K_2, S_1, M_cert.

OUTPUT: _agent_K31_selberg_extremal_result.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    MV_COEFFS,
    N_QP,
    U,
    evaluate_phi,
    evaluate_K_directly,
    mv_master_M_cert,
    reference_arcsine_value,
)


# ---------------------------------------------------------------------------
# Sinc-squared / Fejer / Vaaler K(z) primitive.
# ---------------------------------------------------------------------------
def sinc_sq(z: np.ndarray) -> np.ndarray:
    """K(z) = (sin(pi z) / (pi z))^2, Vaaler's elementary positive function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    mask_zero = (np.abs(z) < 1e-14)
    out[mask_zero] = 1.0
    nz = ~mask_zero
    pz = np.pi * z[nz]
    out[nz] = (np.sin(pz) / pz) ** 2
    return out


# ---------------------------------------------------------------------------
# Beurling B(z), H(z) on real x via truncated series.
# B(z) = (sin pi z / pi)^2 * { sum_{n=0}^N 1/(z-n)^2 - sum_{n=-N}^{-1} 1/(z-n)^2 + 2/z }
# H(z) = (sin pi z / pi)^2 * { sum_{n=-N..N, n != 0} sgn(n)/(z-n)^2 + 2/z }
# ---------------------------------------------------------------------------
def _shift_off_integer(z: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """If z is very close to an integer, nudge by eps so that 1/(z-n)^2 stays
    finite.  Since (sin pi z)^2 -> 0 simultaneously as z -> n, the product
    B(z) is well-defined; we trade exact integer evaluation for a tiny
    numerical perturbation.

    For our use, z is a continuous variable on a quadrature grid, so the
    chance of hitting an exact integer is essentially zero; this is a safety
    net only.
    """
    z = np.asarray(z, dtype=float)
    nearest = np.round(z)
    near_mask = np.abs(z - nearest) < eps
    z_safe = np.where(near_mask, nearest + eps, z)
    return z_safe


def beurling_B(z: np.ndarray, N: int = 200) -> np.ndarray:
    """Beurling's function B(z) majorizing sgn(z), entire of exp type 2 pi.

    B(z) = (sin pi z / pi)^2 * { sum_{n=0}^inf 1/(z-n)^2
                                - sum_{n=-inf}^{-1} 1/(z-n)^2 + 2/z }

    Implemented via truncated series.  We nudge inputs away from exact
    integers (where the prefactor (sin pi z)^2 cancels the 1/(z-n)^2 pole).
    """
    z = _shift_off_integer(np.asarray(z, dtype=float))
    s = np.zeros_like(z)
    # n = 0: positive sum contributes 1/z^2; no contribution from negative sum
    s = s + 1.0 / (z * z)
    # n = 1..N: positive sum contributes 1/(z-n)^2
    # n = -N..-1: negative sum (subtracted) contributes -1/(z-n)^2 = -1/(z+|n|)^2
    for n in range(1, N + 1):
        s = s + 1.0 / ((z - n) ** 2) - 1.0 / ((z + n) ** 2)
    # +2/z term
    s = s + 2.0 / z
    pref = (np.sin(np.pi * z) / np.pi) ** 2
    out = pref * s
    return out


def beurling_H(z: np.ndarray, N: int = 200) -> np.ndarray:
    """H(z) = (sin pi z / pi)^2 * (sum_{n != 0} sgn(n)/(z-n)^2 + 2/z).

    H is the symmetric Beurling: H(-z) = -H(z) (odd).  Interpolates sgn(x).
    """
    z = _shift_off_integer(np.asarray(z, dtype=float))
    s = np.zeros_like(z)
    for n in range(1, N + 1):
        s = s + 1.0 / ((z - n) ** 2) - 1.0 / ((z + n) ** 2)
    s = s + 2.0 / z
    pref = (np.sin(np.pi * z) / np.pi) ** 2
    return pref * s


def selberg_majorant_indicator(z: np.ndarray, a: float, b: float,
                                N: int = 200) -> np.ndarray:
    """Selberg majorant C(z) of chi_{[a,b]}: (1/2)(B(b-z) + B(z-a)).

    Bandlimited entire of exp type 2*pi.  Satisfies C(x) >= chi_{[a,b]}(x).
    Integral: int (C - chi_{[a,b]}) dx = 1.
    """
    return 0.5 * (beurling_B(b - z, N) + beurling_B(z - a, N))


def selberg_minorant_indicator(z: np.ndarray, a: float, b: float,
                                N: int = 200) -> np.ndarray:
    """Selberg minorant c(z) of chi_{[a,b]}: (1/2)(-B(z-b) - B(a-z)).

    Bandlimited entire of exp type 2*pi.  Satisfies c(x) <= chi_{[a,b]}(x).
    Note: equals -B(-(b-z))/2 - B(-(z-a))/2 + ...

    Standard form: minorant = -C(z; complement).
    Using -B(-x) <= sgn(x): we get -B(z-b) <= sgn(b-z)
    and similar for left side.  Then chi_{[a,b]} = (1/2)(sgn(b-z)+sgn(z-a))
    so minorant = (1/2)(-B(z-b) + (-B(a-z))) ...
    """
    return 0.5 * (-beurling_B(z - b, N) + (-beurling_B(a - z, N)))


# ---------------------------------------------------------------------------
# Phi constructions.
# ---------------------------------------------------------------------------

def phi_truncated_sinc_sq(scale: float):
    """Phi(x) = sinc^2(pi * scale * x) on |x| <= DELTA/2, else 0.

    Vaaler's elementary positive function K(z) = sinc^2(pi z), rescaled and
    truncated.  scale = 1/DELTA puts the first null of sinc at the support
    boundary.  Larger scale -> faster decay -> closer to (Vaaler-)extremal.
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        return np.where(in_supp, sinc_sq(scale * x), 0.0)
    return phi


def phi_truncated_sinc_quad(scale: float):
    """Phi(x) = sinc^4(pi * scale * x) on |x| <= DELTA/2.

    K(z)^2 = sinc^4(pi z) is non-negative bandlimited of type 4 pi (entire);
    its inverse FT is bicubic B-spline.  After truncation it remains
    non-negative.
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        return np.where(in_supp, sinc_sq(scale * x) ** 2, 0.0)
    return phi


def phi_vaaler_combination(a: float, b: float, c: float):
    """Phi(x) = a * K(2x/delta) + b * K(2x/delta * c) on |x|<=DELTA/2.

    Convex (or weighted) combination of two sinc^2 kernels at different scales.
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        u = 2.0 * x / DELTA
        val = a * sinc_sq(u) + b * sinc_sq(c * u)
        val = np.maximum(val, 0.0)
        return np.where(in_supp, val, 0.0)
    return phi


def phi_selberg_majorant_rescaled(alpha: float):
    """Phi(x) = Selberg majorant of chi_{[-alpha, alpha]} evaluated at u = 2x/delta,
    truncated to [-delta/2, delta/2].

    The Selberg majorant of chi_{[-alpha,alpha]} is bandlimited and >= 1 on
    [-alpha, alpha], with oscillating tails.  Truncation to [-1, 1] (in u)
    gives a positive bump-like function IF alpha < 1.  In x coords this is
    [-delta/2, delta/2].
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        u = 2.0 * x / DELTA
        val = selberg_majorant_indicator(u, -alpha, alpha, N=300)
        val = np.maximum(val, 0.0)
        return np.where(in_supp, val, 0.0)
    return phi


def phi_beurling_diff(scale: float):
    """Phi(x) = (1/2)(B(scale * x + 1) - B(scale * x - 1)) on |x| <= delta/2.

    Construction: B(z) is the Beurling function majorizing sgn.  Then
    B(z+1) - B(z-1) is "Beurling's majorant of indicator[-1,1]" up to a
    factor, and is generally non-negative (it's B applied to both sides of
    a symmetric interval).  Specifically C(z) for [-1,1] = (1/2)(B(1-z) + B(z+1)).
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        z = scale * x
        # Selberg majorant of [-1,1] at z = scale * x
        val = 0.5 * (beurling_B(1.0 - z, N=300) + beurling_B(z + 1.0, N=300))
        val = np.maximum(val, 0.0)
        return np.where(in_supp, val, 0.0)
    return phi


def phi_carneiro_vaaler_gauss(t: float, scale: float):
    """Phi(x) = Gaussian-subordinated truncated Vaaler kernel.

    Carneiro-Vaaler 2010 ("Gaussian subordination for Beurling-Selberg")
    constructs extremal majorants by integrating Beurling against Gaussian:
    F_t(z) = int B(z * sqrt(t/u)) * (heat kernel) du.

    For a numerical surrogate that captures the Gaussian-smoothed Beurling
    extremal structure, we use:
        phi(x) = exp(-t * x^2 / DELTA^2) * sinc_sq(scale * x)
    truncated to [-DELTA/2, DELTA/2].  Always positive.
    """
    def phi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        in_supp = (np.abs(x) <= DELTA / 2.0 + 1e-14)
        z = scale * x
        gauss = np.exp(-t * (x / DELTA) ** 2)
        val = gauss * sinc_sq(z)
        return np.where(in_supp, val, 0.0)
    return phi


# ---------------------------------------------------------------------------
# Main runner.
# ---------------------------------------------------------------------------

def run_one(label: str, phi_fn, extra: dict | None = None) -> dict:
    t0 = time.time()
    try:
        r = evaluate_phi(phi_fn, label=label, verbose=False)
    except Exception as e:
        r = {"label": label, "M_cert": None, "reason": f"exception: {e}"}
    r["wall_sec"] = time.time() - t0
    if extra is not None:
        r.update(extra)
    M = r.get("M_cert")
    Mtxt = f"{M:.5f}" if M is not None else f"None({r.get('reason','?')})"
    k_1 = r.get("k_1")
    K_2 = r.get("K_2")
    S_1 = r.get("S_1")
    if k_1 is not None and K_2 is not None and S_1 is not None:
        print(f"  [{label:50s}] k_1={k_1:.4f}  K_2={K_2:.3f}  "
              f"S_1={S_1:.2e}  M_cert={Mtxt}")
    else:
        print(f"  [{label:50s}] M_cert={Mtxt}")
    return r


def main() -> None:
    print("=" * 78)
    print("Agent K31: Beurling-Selberg-Vaaler extremal kernels")
    print(f"DELTA = {DELTA}")
    print("=" * 78)

    out: dict = {"delta": DELTA}

    # --- Reference: arcsine ---------------------------------------------------
    print("\n--- Reference: arcsine MV baseline ---")
    ref = reference_arcsine_value()
    out["reference_arcsine"] = ref
    M_arc = ref.get("M_cert")
    print(f"  arcsine: M_cert = {M_arc:.5f}, k_1={ref['k_1']:.4f}, K_2={ref['K_2']:.4f}")

    results: list[dict] = []

    # --- Family A: truncated Vaaler sinc^2 (the canonical positive case) ----
    print("\n--- A: Phi = sinc^2(pi * scale * x) truncated to [-DELTA/2, DELTA/2] ---")
    for scale in [1.0 / DELTA, 2.0 / DELTA, 3.0 / DELTA, 4.0 / DELTA,
                   5.0 / DELTA, 6.0 / DELTA, 8.0 / DELTA, 10.0 / DELTA]:
        label = f"A_sinc2_scale={scale * DELTA:.1f}/DELTA"
        r = run_one(label, phi_truncated_sinc_sq(scale),
                    extra={"family": "A_sinc_sq_trunc",
                           "scale": float(scale),
                           "scale_x_DELTA": float(scale * DELTA)})
        results.append(r)

    # --- Family B: truncated sinc^4 (Vaaler K^2) -----------------------------
    print("\n--- B: Phi = sinc^4(pi * scale * x) (Vaaler K^2 truncated) ---")
    for scale in [1.0 / DELTA, 2.0 / DELTA, 3.0 / DELTA, 4.0 / DELTA,
                   5.0 / DELTA, 6.0 / DELTA]:
        label = f"B_sinc4_scale={scale * DELTA:.1f}/DELTA"
        r = run_one(label, phi_truncated_sinc_quad(scale),
                    extra={"family": "B_sinc_quad_trunc",
                           "scale": float(scale),
                           "scale_x_DELTA": float(scale * DELTA)})
        results.append(r)

    # --- Family C: two-scale Vaaler combination ------------------------------
    print("\n--- C: Phi = a sinc^2(u) + b sinc^2(c*u),  u = 2x/DELTA ---")
    combos = [
        (1.0, 0.0, 1.0),  # pure sinc_sq at u-scale 1
        (0.7, 0.3, 2.0),
        (0.5, 0.5, 2.0),
        (0.3, 0.7, 2.0),
        (0.5, 0.5, 3.0),
        (0.3, 0.7, 3.0),
        (0.2, 0.8, 4.0),
    ]
    for (a, b, c) in combos:
        label = f"C_combo_a={a:.1f}_b={b:.1f}_c={c:.1f}"
        r = run_one(label, phi_vaaler_combination(a, b, c),
                    extra={"family": "C_combo", "a": a, "b": b, "c": c})
        results.append(r)

    # --- Family D: Selberg majorant of [-alpha, alpha] rescaled --------------
    print("\n--- D: Phi = Selberg-majorant of chi_{[-alpha,alpha]} (in u=2x/DELTA) ---")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        label = f"D_selberg_maj_alpha={alpha:.1f}"
        r = run_one(label, phi_selberg_majorant_rescaled(alpha),
                    extra={"family": "D_selberg_majorant", "alpha": alpha})
        results.append(r)

    # --- Family E: Beurling's Selberg majorant at general scale --------------
    print("\n--- E: Phi = (1/2)(B(1-scale*x) + B(scale*x+1)) (Selberg majorant of [-1,1]) ---")
    for scale in [1.0 / DELTA, 2.0 / DELTA, 3.0 / DELTA, 4.0 / DELTA, 5.0 / DELTA,
                   1.5 / DELTA, 2.5 / DELTA, 3.5 / DELTA]:
        label = f"E_selb_maj_[-1,1]_scale={scale * DELTA:.2f}/DELTA"
        r = run_one(label, phi_beurling_diff(scale),
                    extra={"family": "E_selb_maj_pm1", "scale": float(scale),
                           "scale_x_DELTA": float(scale * DELTA)})
        results.append(r)

    # --- Family F: Gaussian-subordinated (Carneiro-Vaaler 2010 surrogate) ----
    print("\n--- F: Phi = Gaussian * sinc^2 (Carneiro-Vaaler-flavoured) ---")
    for t in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for scale in [2.0 / DELTA, 3.0 / DELTA, 4.0 / DELTA]:
            label = f"F_CV_t={t:.1f}_scale={scale*DELTA:.1f}/DELTA"
            r = run_one(label, phi_carneiro_vaaler_gauss(t, scale),
                        extra={"family": "F_carneiro_vaaler", "t": t,
                               "scale": float(scale),
                               "scale_x_DELTA": float(scale * DELTA)})
            results.append(r)

    out["results"] = results

    # --- Summary -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    finite = [r for r in results if r.get("M_cert") is not None]
    if finite:
        best = max(finite, key=lambda r: r["M_cert"])
        out["best"] = {
            "label": best.get("label"),
            "M_cert": best["M_cert"],
            "k_1": best.get("k_1"),
            "K_2": best.get("K_2"),
            "S_1": best.get("S_1"),
            "family": best.get("family"),
        }
        print(f"\nBest M_cert: {best['M_cert']:.5f} from [{best.get('label')}]")
        print(f"  k_1 = {best.get('k_1'):.5f}  K_2 = {best.get('K_2'):.4f}  "
              f"S_1 = {best.get('S_1'):.2e}")
        print(f"  Arcsine baseline:  M_cert = {M_arc:.5f}  "
              f"k_1 = {ref['k_1']:.5f}  K_2 = {ref['K_2']:.4f}")
        improve = best["M_cert"] - M_arc
        print(f"  Improvement over arcsine: {improve:+.5f}")
        out["improvement_over_arcsine"] = improve
        out["beats_arcsine_numerical"] = best["M_cert"] > M_arc
        out["beats_MV_1_2748"] = best["M_cert"] > 1.27481
    else:
        print("\nNo finite M_cert.")
        out["best"] = None
        out["beats_arcsine_numerical"] = False
        out["beats_MV_1_2748"] = False

    # --- Top-5 table ---------------------------------------------------------
    if finite:
        sorted_res = sorted(finite, key=lambda r: r["M_cert"], reverse=True)
        print("\nTop 10 by M_cert:")
        print(f"  {'label':50s}  {'M_cert':>9s}  {'k_1':>7s}  "
              f"{'K_2':>8s}  {'S_1':>9s}")
        for r in sorted_res[:10]:
            print(f"  {r.get('label',''):50s}  {r['M_cert']:9.5f}  "
                  f"{r.get('k_1', 0):7.4f}  {r.get('K_2', 0):8.3f}  "
                  f"{r.get('S_1', 0):9.2e}")

    out_path = os.path.join(REPO, "_agent_K31_selberg_extremal_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

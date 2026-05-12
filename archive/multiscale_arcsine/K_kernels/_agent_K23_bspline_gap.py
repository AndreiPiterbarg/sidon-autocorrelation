"""Agent K23: fill the B-spline auto-conv gap (n in {1, 4, 6, 7, 8, 10, 15, 20}).

Uses the shared `_kernel_probe_helper.evaluate_phi` (FIXED MV G-coefficients).
phi for order n is the n-fold convolution of the box of width delta/n, so
supp(phi) = [-delta/2, delta/2] and K = phi * phi has K_hat = sinc(pi delta xi/n)^{2n}.

For each n we precompute phi on a fine uniform x-grid by repeated
np.convolve, then expose phi via np.interp on the theta-grid the helper uses.

Bochner is automatic because K_hat = (phi_hat)^2.  We still record
bochner_min for sanity.

Prior (K10 entries via re-optimised MV-QP pipeline in
delsarte_dual/grid_bound_alt_kernel/REPORT.md):
    n=2 -> M_cert = 1.2107
    n=3 -> M_cert = 1.2030
    n=5 -> M_cert = 1.1864
These re-optimise the G QP per kernel; the helper here keeps MV's fixed G
so absolute numbers differ but the family/trend comparison is internally
consistent.  We report helper M_cert on its own scale plus a reference
arcsine value computed with the same helper for calibration.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (
    DELTA,
    MV_COEFFS,
    N_QP,
    N_THETA,
    N_XI,
    U,
    XI_MAX_OVER_DELTA,
    evaluate_phi,
    mv_master_M_cert,
    reference_arcsine_value,
)


def make_bspline_phi(n: int, npts_phi: int = 200001):
    """Return a callable phi(x) for the cardinal B-spline of order n,
    scaled to have support [-DELTA/2, DELTA/2].

    Computed by n-fold numerical convolution of the box on the SAME fine
    uniform x-grid; the result is then exposed via np.interp.
    """
    if n < 1:
        raise ValueError("n >= 1 required")
    # Slight margin so the convolution support fits cleanly
    half = DELTA / 2.0 + 1e-4
    x_fine = np.linspace(-half, half, npts_phi)
    dx = x_fine[1] - x_fine[0]
    box_half = DELTA / (2.0 * n)
    # Indicator on [-box_half, box_half] normalised to unit mass
    box = (np.abs(x_fine) <= box_half).astype(float)
    box_int = np.trapezoid(box, x_fine)
    if box_int <= 0:
        raise RuntimeError("Box integral 0 — increase npts_phi")
    box /= box_int

    phi_vals = box.copy()
    for _ in range(n - 1):
        phi_vals = np.convolve(phi_vals, box, mode="same") * dx

    # Clip tiny numerical negatives
    phi_vals = np.maximum(phi_vals, 0.0)

    def phi(x: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x), x_fine, phi_vals, left=0.0, right=0.0)

    return phi


def _diagnostic_bound(phi_fn, n: int, label: str) -> dict:
    """When the helper bails (phi_hat(j/u) ~ 0 at some j), compute k_1, K_2,
    bochner_min, and the LIST of j's that hit (near-)zero phi_hat^2 — useful
    for explaining the failure mode and bounding what the MV inequality with
    a different QP could do.
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
    dth = theta[1] - theta[0]
    x_t = (DELTA / 2.0) * np.sin(theta)
    cos_t = np.cos(theta)
    phi_vals = np.asarray(phi_fn(x_t), dtype=float)
    phi_vals = np.maximum(phi_vals, 0.0)
    w = phi_vals * (DELTA / 2.0) * cos_t
    Z = np.trapezoid(w, dx=dth)
    w = w / Z

    def phi_hat(xi):
        xi = np.atleast_1d(xi)
        cos_mat = np.cos(2 * np.pi * xi[:, None] * x_t[None, :])
        return cos_mat @ w * dth

    ns = np.arange(1, 201)
    ph_int = phi_hat(ns)
    kn = ph_int ** 2
    k_1 = float(kn[0])
    bmin = float(kn.min())

    # K_2 via int phi_hat^4 dxi
    xi_grid = np.linspace(0.0, XI_MAX_OVER_DELTA / DELTA, N_XI)
    dxi = xi_grid[1] - xi_grid[0]
    batch = 1000
    K2_pos = 0.0
    for s in range(0, N_XI, batch):
        chunk = xi_grid[s:s + batch]
        ph = phi_hat(chunk)
        K2_pos += np.sum(ph ** 4) * dxi
    K_2 = 2.0 * K2_pos

    # Find which j cause phi_hat(j/u)^2 ~ 0
    qp_xi = np.arange(1, N_QP + 1) / U
    ph_qp = phi_hat(qp_xi)
    ph_qp2 = ph_qp ** 2
    bad_js = [int(j) for j in np.where(ph_qp2 < 1e-12)[0] + 1]
    return {
        "k_1": k_1,
        "K_2": float(K_2),
        "bochner_min": bmin,
        "zero_qp_js": bad_js,
        "phi_hat_jpu_min_abs": float(np.min(np.abs(ph_qp))),
    }


def main() -> None:
    out: dict = {}

    # Calibration: helper's own arcsine M_cert (fixed MV coefficients)
    print("=== Reference: arcsine (helper calibration) ===")
    out["__reference_arcsine__"] = reference_arcsine_value()
    print()

    targets = [1, 4, 6, 7, 8, 10, 15, 20]
    results = []
    for n in targets:
        t0 = time.time()
        phi_fn = make_bspline_phi(n)
        res = evaluate_phi(phi_fn, label=f"K23_bspline_n={n}")
        if res.get("M_cert") is None:
            extras = _diagnostic_bound(phi_fn, n, label=res["label"])
            res.update(extras)
        res["n"] = n
        res["wall_sec"] = time.time() - t0
        results.append(res)

    # Also include known prior n=2,3,5 with the same helper for an
    # internally consistent monotonicity check.
    for n in [2, 3, 5]:
        t0 = time.time()
        phi_fn = make_bspline_phi(n)
        res = evaluate_phi(phi_fn, label=f"K23_bspline_n={n}_recheck")
        if res.get("M_cert") is None:
            extras = _diagnostic_bound(phi_fn, n, label=res["label"])
            res.update(extras)
        res["n"] = n
        res["wall_sec"] = time.time() - t0
        results.append(res)

    results.sort(key=lambda r: r["n"])
    out["results"] = results

    # Summary
    summary = []
    for r in results:
        summary.append(
            {
                "n": r["n"],
                "M_cert": r.get("M_cert"),
                "k_1": r.get("k_1"),
                "K_2": r.get("K_2"),
                "S_1": r.get("S_1"),
                "bochner_min": r.get("bochner_min"),
                "zero_qp_js": r.get("zero_qp_js"),
                "phi_hat_jpu_min_abs": r.get("phi_hat_jpu_min_abs"),
                "reason": r.get("reason"),
            }
        )
    out["summary"] = summary

    finite = [r for r in results if r.get("M_cert") is not None]
    if finite:
        best = max(finite, key=lambda r: r["M_cert"])
        out["best_M_cert"] = best["M_cert"]
        out["best_n"] = best["n"]
        out["beats_1p260"] = best["M_cert"] > 1.260
        out["beats_MV"] = best["M_cert"] > 1.27481
    else:
        out["best_M_cert"] = None

    json_path = os.path.join(REPO, "_agent_K23_bspline_gap_result.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {json_path}")

    print("\n=== B-spline auto-conv M_cert vs n ===")
    print(f"{'n':>3}  {'M_cert':>14}  {'k_1':>9}  {'K_2':>9}  {'S_1':>14}")
    for s in summary:
        mc = s["M_cert"]
        mc_s = f"{mc:.5f}" if mc is not None else f"NONE ({s.get('reason','?')})"
        k1 = s.get("k_1")
        K2 = s.get("K_2")
        S1 = s.get("S_1")
        k1_s = f"{k1:.5f}" if isinstance(k1, (float, int)) else "NA"
        K2_s = f"{K2:.4f}" if isinstance(K2, (float, int)) else "NA"
        S1_s = f"{S1:.3e}" if isinstance(S1, (float, int)) else "NA"
        print(f"{s['n']:>3}  {mc_s:>14}  {k1_s:>9}  {K2_s:>9}  {S1_s:>14}")
    if out.get("best_M_cert") is not None:
        print(
            f"\nBest n = {out['best_n']}, M_cert = {out['best_M_cert']:.5f} "
            f"(beats 1.260: {out['beats_1p260']}, beats MV 1.27481: {out['beats_MV']})"
        )


if __name__ == "__main__":
    main()

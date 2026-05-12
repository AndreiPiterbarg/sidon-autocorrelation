"""Agent K24: Tukey (tapered cosine) window auto-conv probe for MV M_cert.

phi defined on x in [-DELTA/2, DELTA/2]:
    Let L = DELTA/2, r = alpha * L.
    phi(x) = 1                                  if |x| <= L - r       (flat top)
           = 0.5 (1 + cos(pi * (|x| - (L-r)) / r))   if L-r < |x| < L (cosine taper)
           = 0                                  if |x| >= L

alpha = 0  --> pure box on [-L, L] (no taper)
alpha = 1  --> pure cosine/Hann (full taper)
0 < alpha < 1 interpolates.

K = phi * phi has K_hat = (phi_hat)^2 >= 0, Bochner-OK by construction.
"""
from __future__ import annotations

import json
import os

import numpy as np

from _kernel_probe_helper import DELTA, evaluate_phi


def phi_tukey(x: np.ndarray, alpha: float) -> np.ndarray:
    """Tukey window of given alpha on [-DELTA/2, DELTA/2]."""
    L = DELTA / 2.0
    ax = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(ax)
    if alpha <= 0.0:
        # Pure box on [-L, L]
        out[ax < L] = 1.0
        return out
    r = alpha * L
    flat_top = L - r
    # flat region
    flat_mask = ax <= flat_top
    out[flat_mask] = 1.0
    # taper region
    taper_mask = (ax > flat_top) & (ax < L)
    if r > 0:
        out[taper_mask] = 0.5 * (1.0 + np.cos(np.pi * (ax[taper_mask] - flat_top) / r))
    return out


def sweep(alphas: list[float]) -> list[dict]:
    results = []
    for a in alphas:
        label = f"tukey_alpha={a:.3f}"
        try:
            res = evaluate_phi(lambda x, _a=a: phi_tukey(x, _a), label)
        except Exception as exc:
            res = {"label": label, "M_cert": None, "error": str(exc)}
        res["alpha"] = a
        results.append(res)
    return results


def best_alpha(results: list[dict]):
    candidates = [r for r in results if r.get("M_cert") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["M_cert"])


def main():
    print("=== Tukey window sweep (coarse) ===")
    alphas_coarse = [round(0.1 * i, 2) for i in range(11)]
    coarse = sweep(alphas_coarse)

    best = best_alpha(coarse)
    out = {
        "coarse_alphas": alphas_coarse,
        "coarse_results": coarse,
        "best_coarse": best,
    }

    if best is not None and best["M_cert"] is not None and best["M_cert"] > 1.250:
        print(f"\n=== Refined sweep around alpha={best['alpha']} (step 0.01) ===")
        a0 = best["alpha"]
        lo = max(0.0, a0 - 0.1)
        hi = min(1.0, a0 + 0.1)
        # 21 points centred near best, step ~0.01
        n_fine = int(round((hi - lo) / 0.01)) + 1
        fine_alphas = [round(lo + i * 0.01, 4) for i in range(n_fine)]
        fine = sweep(fine_alphas)
        out["fine_alphas"] = fine_alphas
        out["fine_results"] = fine
        out["best_fine"] = best_alpha(fine)

    # Overall best across coarse + fine
    all_results = list(coarse) + list(out.get("fine_results", []))
    overall = best_alpha(all_results)
    out["best_overall"] = overall

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_agent_K24_tukey_result.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)

    print(f"\nWrote {json_path}")
    if overall is not None:
        print(f"Best: alpha={overall['alpha']}  M_cert={overall['M_cert']}  "
              f"k_1={overall.get('k_1')}  K_2={overall.get('K_2')}")
    else:
        print("No valid M_cert across sweep.")


if __name__ == "__main__":
    main()

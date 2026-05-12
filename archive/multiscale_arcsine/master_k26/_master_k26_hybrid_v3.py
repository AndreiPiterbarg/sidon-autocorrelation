"""Master K26-Hybrid v3: Fine refinement around K26 + epsilon * triangle.

Best from v2: K26-base + eps=0.01 * triangle@delta=0.05  =>  M=1.28016.
That's barely above K26's 1.28013. We refine the (eps, delta_C) grid AND
re-optimize (lambda_1, delta_2) within K26 with the triangle perturbation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _master_k26_hybrid import (  # noqa: E402
    DELTA, evaluate_mixture,
)


def make_mix(l1, d2, l2_a, fam_C, dC, eps_C):
    """Mixture: arcsine@DELTA(l1) + arcsine@d2(l2_a) + fam_C@dC(eps_C).
    With l1 + l2_a + eps_C = 1 enforced.
    """
    # Re-scale so that lambdas sum to 1.
    s = l1 + l2_a + eps_C
    return [("arcsine", DELTA, l1 / s),
            ("arcsine", d2, l2_a / s),
            (fam_C, dC, eps_C / s)]


def main():
    t0 = time.time()
    print("=" * 78)
    print("Master K26-Hybrid v3: fine refine arcsine x2 + small triangle")
    print("=" * 78)

    # K26 base.
    K26 = evaluate_mixture(
        [("arcsine", DELTA, 0.9312), ("arcsine", 0.055, 0.0688)],
        "K26-base")
    print(f"K26 base M_cert = {K26['M_cert']:.6f}")

    best = {"M_cert": K26["M_cert"], "config": "K26-base"}
    results = []

    print("\n--- Coarse: arcsine(l1)@DELTA + arcsine(d2)@l2 + tri(eps)@dC ---")
    # Grids.
    l1_list = [0.91, 0.92, 0.93, 0.94]
    d2_list = [0.045, 0.05, 0.055, 0.06, 0.065, 0.07]
    eps_list = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    dC_list = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]
    fam_C_list = ["triangle", "bspline3"]

    count = 0
    for fam_C in fam_C_list:
        for l1 in l1_list:
            for d2 in d2_list:
                for eps in eps_list:
                    for dC in dC_list:
                        l2 = 1.0 - l1 - eps
                        if l2 < 0.001:
                            continue
                        comps = [("arcsine", DELTA, l1),
                                 ("arcsine", d2, l2),
                                 (fam_C, dC, eps)] if eps > 0 else [
                                 ("arcsine", DELTA, l1),
                                 ("arcsine", d2, 1.0 - l1)]
                        label = f"l1={l1},d2={d2},eps={eps},{fam_C}@{dC}"
                        res = evaluate_mixture(comps, label, verbose=False)
                        count += 1
                        rec = {"l1": l1, "d2": d2, "eps": eps,
                               "fam_C": fam_C, "dC": dC,
                               **{k: v for k, v in res.items()
                                  if k not in ("label", "components")}}
                        results.append(rec)
                        M = res.get("M_cert")
                        if M is not None and M > best["M_cert"]:
                            best = {"M_cert": float(M),
                                    "config": label,
                                    "l1": l1, "d2": d2, "eps": eps,
                                    "fam_C": fam_C, "dC": dC}
                            print(f"  *NEW* M={M:.6f}  {label}")

    print(f"\nEvaluated {count} configurations")
    print(f"Best: {best}")
    print(f"K26 reference: 1.28013")
    print(f"Improvement over K26: {best['M_cert'] - K26['M_cert']:+.6f}")
    print(f"Beats K26: {best['M_cert'] > K26['M_cert'] + 1e-6}")

    out = {
        "K26_base_M_cert": float(K26["M_cert"]),
        "best": best,
        "improvement_over_K26": float(best["M_cert"] - K26["M_cert"]),
        "beats_K26": best["M_cert"] > K26["M_cert"] + 1e-6,
        "evaluated_count": count,
        "elapsed_s": time.time() - t0,
    }
    outpath = os.path.join(REPO, "_master_k26_hybrid_v3.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nWrote {outpath}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

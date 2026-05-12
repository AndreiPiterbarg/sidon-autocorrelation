"""Master K26-Hybrid v2: extend K26's multi-scale arcsine optimum by adding a
small mass of a DIFFERENT family. Also test 3-family pure mixtures.

Base: K26 best = arcsine@0.138 (lambda=0.9312) + arcsine@0.055 (lambda=0.0688)
  -> M_cert ~= 1.28013

Test 1: K26 base + epsilon * fam_C@delta_C for fam_C in {triangle, bspline3,
        epanechnikov, semicircle}, scanning delta_C and epsilon.
Test 2: 3-family pure mixtures with all distinct families.
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
    DELTA, FAMILIES, evaluate_mixture,
)


def main():
    t0 = time.time()
    print("=" * 78)
    print("Master K26-Hybrid v2: K26 base + epsilon * other family")
    print("=" * 78)

    # K26 base.
    K26_DELTA_2 = 0.055
    K26_LAMBDA_1 = 0.9312
    K26_LAMBDA_2 = 1.0 - K26_LAMBDA_1

    # Confirm K26 base value.
    base = evaluate_mixture(
        [("arcsine", DELTA, K26_LAMBDA_1),
         ("arcsine", K26_DELTA_2, K26_LAMBDA_2)],
        "K26-base (2-arcsine)",
    )
    print(f"\nK26 base M_cert = {base['M_cert']:.5f}  (expected ~1.28013)")

    print("\n--- K26 base + epsilon * fam_C@delta_C ---")
    results = []
    best = {"M_cert": base["M_cert"]}
    eps_list = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    dC_list = [0.05, 0.07, 0.09, 0.10, 0.115, 0.125, 0.135]
    for fam_C in ["triangle", "bspline3", "epanechnikov", "semicircle"]:
        for dC in dC_list:
            for eps in eps_list:
                if eps == 0.0 and fam_C != "triangle":
                    continue  # eps=0 is redundant
                comps = [
                    ("arcsine", DELTA, K26_LAMBDA_1 * (1 - eps)),
                    ("arcsine", K26_DELTA_2, K26_LAMBDA_2 * (1 - eps)),
                    (fam_C, dC, eps),
                ]
                label = (f"K26 + {eps:.2f}*{fam_C}@{dC:.3f}")
                res = evaluate_mixture(comps, label, verbose=False)
                rec = {"fam_C": fam_C, "delta_C": float(dC), "eps": float(eps),
                       **{k: v for k, v in res.items()
                          if k not in ("label", "components")}}
                results.append(rec)
                M = res.get("M_cert")
                if M is not None and M > best["M_cert"]:
                    best = {"M_cert": float(M), "fam_C": fam_C,
                            "delta_C": float(dC), "eps": float(eps)}
                    print(f"  *NEW BEST* M={M:.5f}  {label}")

    print(f"\nBest after K26+eps scan: {best}")

    print("\n--- 3-family pure mixtures (3 distinct families) ---")
    three_results = []
    three_best = {"M_cert": -np.inf}
    fams = list(FAMILIES.keys())
    # Anchor: arcsine at DELTA. Other two: distinct.
    fam_A = "arcsine"
    dA = DELTA
    other_fams = [f for f in fams if f != fam_A]
    delta_list = [0.05, 0.07, 0.09, 0.11, 0.125, 0.135]
    lam_grid = [
        (0.85, 0.10, 0.05), (0.85, 0.05, 0.10),
        (0.80, 0.15, 0.05), (0.80, 0.10, 0.10), (0.80, 0.05, 0.15),
        (0.75, 0.15, 0.10), (0.75, 0.10, 0.15),
        (0.70, 0.20, 0.10), (0.70, 0.15, 0.15), (0.70, 0.10, 0.20),
        (0.90, 0.05, 0.05),
        (0.93, 0.04, 0.03),
    ]
    for fam_B in other_fams:
        for fam_C in other_fams:
            if fam_C <= fam_B:  # unordered pair, B != C
                continue
            for dB in delta_list:
                for dC in delta_list:
                    if dC == dB:
                        continue
                    for lA, lB, lC in lam_grid:
                        comps = [(fam_A, dA, lA), (fam_B, dB, lB),
                                 (fam_C, dC, lC)]
                        label = (f"{fam_A}@{dA:.3f}({lA:.2f})"
                                 f"+{fam_B}@{dB:.3f}({lB:.2f})"
                                 f"+{fam_C}@{dC:.3f}({lC:.2f})")
                        res = evaluate_mixture(comps, label, verbose=False)
                        rec = {"fam_A": fam_A, "fam_B": fam_B, "fam_C": fam_C,
                               "delta_A": dA, "delta_B": float(dB),
                               "delta_C": float(dC),
                               "lambda_A": lA, "lambda_B": lB, "lambda_C": lC,
                               **{k: v for k, v in res.items()
                                  if k not in ("label", "components")}}
                        three_results.append(rec)
                        M = res.get("M_cert")
                        if M is not None and M > three_best["M_cert"]:
                            three_best = {"M_cert": float(M),
                                          "fam_A": fam_A, "fam_B": fam_B,
                                          "fam_C": fam_C,
                                          "delta_A": dA, "delta_B": float(dB),
                                          "delta_C": float(dC),
                                          "lambda_A": lA, "lambda_B": lB,
                                          "lambda_C": lC}
                            print(f"  *3-fam BEST* M={M:.5f}  {label}")

    print(f"\n3-family best: {three_best}")

    final_best = best if best["M_cert"] >= three_best["M_cert"] else three_best
    print(f"\nFINAL v2 best: {final_best}")
    print(f"K26 reference: 1.28013")
    print(f"Beats K26: {final_best['M_cert'] > 1.28013}")

    out = {
        "K26_base_M_cert": float(base["M_cert"]),
        "K26_eps_scan_best": best,
        "three_family_best": three_best,
        "final_v2_best": final_best,
        "beats_K26_1_28013": final_best["M_cert"] > 1.28013,
        "elapsed_s": time.time() - t0,
    }
    outpath = os.path.join(REPO, "_master_k26_hybrid_v2.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nWrote {outpath}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

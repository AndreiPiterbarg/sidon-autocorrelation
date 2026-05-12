"""Master K26-Hybrid v4: finer refinement near (l1=0.93, d2=0.055, eps=0.005,
triangle@dC=0.04). Push dC smaller and eps grid finer.
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


def main():
    t0 = time.time()
    print("=" * 78)
    print("Master K26-Hybrid v4: very fine sweep around v3 best")
    print("=" * 78)

    # v3 best:
    best = {"M_cert": -np.inf}
    results = []
    l1_list  = [0.920, 0.925, 0.930, 0.935, 0.940]
    d2_list  = [0.050, 0.053, 0.055, 0.057, 0.060]
    eps_list = [0.0, 0.002, 0.005, 0.008, 0.010, 0.015]
    dC_list  = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
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
                        if eps > 0:
                            comps = [("arcsine", DELTA, l1),
                                     ("arcsine", d2, l2),
                                     (fam_C, dC, eps)]
                        else:
                            comps = [("arcsine", DELTA, l1),
                                     ("arcsine", d2, 1.0 - l1)]
                        label = f"l1={l1},d2={d2},eps={eps},{fam_C}@{dC}"
                        res = evaluate_mixture(comps, label, verbose=False)
                        count += 1
                        M = res.get("M_cert")
                        rec = {"l1": l1, "d2": d2, "eps": eps,
                               "fam_C": fam_C, "dC": dC,
                               "M_cert": (None if M is None else float(M))}
                        results.append(rec)
                        if M is not None and M > best.get("M_cert", -np.inf):
                            best = {"M_cert": float(M),
                                    "config": label,
                                    "l1": l1, "d2": d2, "eps": eps,
                                    "fam_C": fam_C, "dC": dC}
                            print(f"  *NEW* M={M:.6f}  {label}")

    print(f"\nEvaluated {count} configurations")
    print(f"Best: {best}")
    print(f"K26 ref 1.28013   Improv = {best['M_cert'] - 1.280133:+.6f}")

    # Now mass-redistribution: maybe eps should consume from d2 side, not d1.
    # Re-confirm the gain genuinely came from the triangle vs an arcsine of same dC.
    # Cross-check: replace triangle@dC with arcsine@dC at same parameters.
    print("\n--- Cross-check: replace triangle with arcsine at same (dC, eps) ---")
    if best["M_cert"] > -np.inf:
        b = best
        comps_cmp = [("arcsine", DELTA, b["l1"]),
                     ("arcsine", b["d2"], 1.0 - b["l1"] - b["eps"]),
                     ("arcsine", b["dC"], b["eps"])]
        res_cmp = evaluate_mixture(comps_cmp, "arcsine-only-replacement",
                                   verbose=True)
        diff = best["M_cert"] - (res_cmp["M_cert"] if res_cmp["M_cert"] else 0)
        print(f"Hybrid (with triangle) - arcsine-only at same params: "
              f"{diff:+.6f}")

    out = {
        "best": best,
        "K26_ref": 1.280133,
        "improvement_over_K26": float(best["M_cert"] - 1.280133),
        "evaluated_count": count,
        "elapsed_s": time.time() - t0,
    }
    outpath = os.path.join(REPO, "_master_k26_hybrid_v4.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nWrote {outpath}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Agent K22: cosine-power kernel sweep for MV C_{1a} lower bound.

phi(x) = C * cos^p(pi x / DELTA) for x in [-DELTA/2, DELTA/2].
- p = 2 reproduces Hann (M_cert ~ 1.20 in prior sweeps).
- Higher p concentrates phi near x=0; phi_hat decay structure changes (sum of
  shifted sincs in closed form).
- Fractional p uses sign-preserving abs: cos(.)^p := sign(cos)*|cos|^p, but on
  [-DELTA/2, DELTA/2] we have cos(pi x/DELTA) in [0,1], so this is just |cos|^p.

Outputs _agent_K22_cosine_power_result.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, evaluate_phi  # noqa: E402


def phi_p(x: np.ndarray, p: float) -> np.ndarray:
    """Unnormalised cosine-power kernel for autoconv.

    On [-DELTA/2, DELTA/2], cos(pi x/DELTA) lies in [0, 1] (vanishes at +-DELTA/2).
    For fractional p we use |cos|^p to be safe under floating noise.
    """
    c = np.cos(np.pi * x / DELTA)
    # avoid 0^p edge issue; clip negatives that arise from tiny float noise
    c = np.clip(c, 0.0, 1.0)
    return c ** p


def run_sweep(ps, tag: str) -> list[dict]:
    results = []
    for p in ps:
        label = f"cospow_p={p}"
        try:
            r = evaluate_phi(lambda x, _p=p: phi_p(x, _p), label, verbose=True)
        except Exception as e:
            r = {"label": label, "p": float(p), "error": str(e), "M_cert": None}
        r["p"] = float(p)
        r["tag"] = tag
        results.append(r)
    return results


def main():
    integer_ps = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    fractional_ps = [2.5, 3.5, 4.5]

    print("=" * 60)
    print("INTEGER + FRACTIONAL cos^p sweep (coarse)")
    print("=" * 60)
    coarse = run_sweep(integer_ps + fractional_ps, tag="coarse")

    # Find best by M_cert
    valid = [r for r in coarse if r.get("M_cert") is not None]
    valid.sort(key=lambda r: r["M_cert"], reverse=True)
    best = valid[0] if valid else None

    refined = []
    if best is not None and best["M_cert"] > 1.260:
        bp = best["p"]
        print()
        print("=" * 60)
        print(f"REFINED fractional sweep around best p={bp} (step 0.1)")
        print("=" * 60)
        lo = max(0.5, bp - 1.0)
        hi = bp + 1.0
        # step 0.1; avoid re-running exact same p as in coarse
        fine_ps = np.round(np.arange(lo, hi + 1e-9, 0.1), 2).tolist()
        # skip exact dupes
        already = {round(r["p"], 2) for r in coarse}
        fine_ps = [p for p in fine_ps if round(p, 2) not in already]
        refined = run_sweep(fine_ps, tag="refined")

    all_results = coarse + refined
    all_valid = [r for r in all_results if r.get("M_cert") is not None]
    all_valid.sort(key=lambda r: r["M_cert"], reverse=True)
    overall_best = all_valid[0] if all_valid else None

    payload = {
        "delta": DELTA,
        "n_results": len(all_results),
        "coarse_count": len(coarse),
        "refined_count": len(refined),
        "best": overall_best,
        "all_results": all_results,
    }
    out_path = os.path.join(REPO, "_agent_K22_cosine_power_result.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Wrote: {out_path}")
    if overall_best is not None:
        print(f"OVERALL BEST: p={overall_best['p']}  M_cert={overall_best['M_cert']:.5f}  "
              f"k_1={overall_best['k_1']:.5f}  K_2={overall_best['K_2']:.4f}")
    else:
        print("No valid M_cert in sweep.")


if __name__ == "__main__":
    main()

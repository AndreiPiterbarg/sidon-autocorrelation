"""Stall-scaling experiment driver.

Five experiments (cf. user prompt):
  E1: fixed margin 0.017, d sweep ∈ {6,7,8,9,10}, find threshold d.
  E2: stuck-region trend at d ∈ {12, 14, 16} with margin 0.017.
  E3: at d=10, sweep target ∈ {1.20, 1.21, 1.215, 1.220} (varying margin).
  E4: d=10, t=1.21 with min_box_width ∈ {1e-10, 1e-15}.
  E5: d=10, t=1.21 with INTERVAL_BNB_JOINT_DEPTH ∈ {default(20), 10}.

Writes JSON results incrementally to data/stall_scaling_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from interval_bnb.parallel import parallel_branch_and_bound

OUT = _HERE / "data" / "stall_scaling_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _save(results):
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _run_one(label, d, target, *,
             time_budget_s=120,
             init_split_depth=12,
             min_box_width=1e-10,
             workers=8,
             joint_depth=None):
    """Run one BnB job. Returns a result dict."""
    if joint_depth is not None:
        os.environ["INTERVAL_BNB_JOINT_DEPTH"] = str(joint_depth)
    else:
        os.environ.pop("INTERVAL_BNB_JOINT_DEPTH", None)

    # target as Fraction for rigor.
    if isinstance(target, str):
        tq = Fraction(target)
    else:
        # Convert float to Fraction with 6-digit precision.
        tq = Fraction(target).limit_denominator(10**6)

    t0 = time.time()
    print(f"\n[exp] {label}  d={d}  target={tq}  (~{float(tq):.6f})  "
          f"budget={time_budget_s}s  joint_depth={joint_depth}  "
          f"min_w={min_box_width:.0e}  init_depth={init_split_depth}",
          flush=True)
    r = parallel_branch_and_bound(
        d=d, target_c=tq,
        workers=workers,
        init_split_depth=init_split_depth,
        min_box_width=min_box_width,
        time_budget_s=time_budget_s,
        verbose=False,
    )
    out = {
        "label": label,
        "d": d,
        "target": float(tq),
        "target_q": str(tq),
        "init_split_depth": init_split_depth,
        "min_box_width": min_box_width,
        "joint_depth": joint_depth,
        "workers": workers,
        "time_budget_s": time_budget_s,
        "success": bool(r["success"]),
        "elapsed_s": r["elapsed_s"],
        "total_nodes": r["total_nodes"],
        "coverage_fraction": r["coverage_fraction"],
        "max_depth": r.get("max_depth", 0),
    }
    print(f"[exp] {label}  -> success={out['success']}  "
          f"coverage={100*out['coverage_fraction']:.4f}%  "
          f"nodes={out['total_nodes']}  elapsed={out['elapsed_s']:.1f}s",
          flush=True)
    return out


def main():
    results = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S")}, "runs": []}

    # ----- E1: d sweep at margin 0.017 ------------------------------
    # User's reference: margin 0.017 means target = val(d) - 0.017.
    # Empirical values (CS bound + MV) for val(d):
    #   val(6) ~= 1.176  -> tgt 1.159
    #   val(7) ~= 1.193  -> tgt 1.176   (interpolation)
    #   val(8) ~= 1.201  -> tgt 1.184   (matches user's data)
    #   val(9) ~= 1.219  -> tgt 1.202   (interpolation)
    #   val(10) ~= 1.225 -> tgt 1.208   (matches user's data)
    # We use the same target convention as the user's empirical row.
    e1_targets = {
        6:  1.159,
        7:  1.176,
        8:  1.184,
        9:  1.202,
        10: 1.208,
    }
    for d in [6, 7, 8, 9, 10]:
        r = _run_one(f"E1_d{d}", d=d, target=e1_targets[d],
                     time_budget_s=120, init_split_depth=12, workers=8)
        results["runs"].append(r); _save(results)

    # ----- E2: trend at d=12, 14, 16, margin 0.017 ------------------
    # val(12) ~= 1.27072 -> tgt 1.254
    # val(14) ~= 1.28396 -> tgt 1.267
    # val(16) ~= 1.295   -> tgt 1.278
    e2_targets = {12: 1.254, 14: 1.267, 16: 1.278}
    for d in [12, 14, 16]:
        # init_split_depth scaling: d=12 -> 12, d=14 -> 14, d=16 -> 16 is too
        # heavy on starter; keep at 14 and rely on time_budget.
        depth = 12 if d == 12 else 14
        r = _run_one(f"E2_d{d}", d=d, target=e2_targets[d],
                     time_budget_s=180, init_split_depth=depth, workers=8)
        results["runs"].append(r); _save(results)

    # ----- E3: d=10 margin sweep ------------------------------------
    for tgt in [1.200, 1.210, 1.215, 1.220]:
        r = _run_one(f"E3_d10_t{tgt:.3f}", d=10, target=tgt,
                     time_budget_s=120, init_split_depth=12, workers=8)
        results["runs"].append(r); _save(results)

    # ----- E4: min_box_width effect at d=10 t=1.21 ------------------
    for mw in [1e-10, 1e-15]:
        r = _run_one(f"E4_d10_mw{mw:.0e}", d=10, target=1.210,
                     time_budget_s=120, init_split_depth=12, workers=8,
                     min_box_width=mw)
        results["runs"].append(r); _save(results)

    # ----- E5: joint-face depth effect at d=10 t=1.21 ---------------
    for jd in [20, 10]:
        r = _run_one(f"E5_d10_jd{jd}", d=10, target=1.210,
                     time_budget_s=120, init_split_depth=12, workers=8,
                     joint_depth=jd)
        results["runs"].append(r); _save(results)

    results["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(results)
    print("\n[exp] ALL DONE -> ", OUT)


if __name__ == "__main__":
    main()

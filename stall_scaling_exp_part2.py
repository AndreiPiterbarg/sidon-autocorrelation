"""Continuation of stall_scaling_exp.py — picks up at E2_d14 onward.

Skips E1 + E2_d12 (already in data/stall_scaling_results.json).
Reduces init_split_depth back to 12 for d=14, d=16 (was 14, possibly causing
worker startup hang on Windows spawn context).
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


def _load():
    if OUT.exists():
        with open(OUT) as f:
            return json.load(f)
    return {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S")}, "runs": []}


def _save(results):
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _run_one(label, d, target, *,
             time_budget_s=120,
             init_split_depth=12,
             min_box_width=1e-10,
             workers=8,
             joint_depth=None):
    if joint_depth is not None:
        os.environ["INTERVAL_BNB_JOINT_DEPTH"] = str(joint_depth)
    else:
        os.environ.pop("INTERVAL_BNB_JOINT_DEPTH", None)

    if isinstance(target, str):
        tq = Fraction(target)
    else:
        tq = Fraction(target).limit_denominator(10**6)

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
    results = _load()
    done_labels = {r["label"] for r in results["runs"]}
    print(f"[exp] resuming; already-done: {sorted(done_labels)}", flush=True)

    # ----- E2 continuation (d=14, d=16; init_depth=12 instead of 14) ---
    e2_targets = {14: 1.267, 16: 1.278}
    for d in [14, 16]:
        label = f"E2_d{d}"
        if label in done_labels:
            print(f"[exp] skip {label} (done)", flush=True); continue
        r = _run_one(label, d=d, target=e2_targets[d],
                     time_budget_s=180, init_split_depth=12, workers=8)
        results["runs"].append(r); _save(results)

    # ----- E3: d=10 margin sweep ------------------------------------
    for tgt in [1.200, 1.210, 1.215, 1.220]:
        label = f"E3_d10_t{tgt:.3f}"
        if label in done_labels:
            print(f"[exp] skip {label} (done)", flush=True); continue
        r = _run_one(label, d=10, target=tgt,
                     time_budget_s=120, init_split_depth=12, workers=8)
        results["runs"].append(r); _save(results)

    # ----- E4: min_box_width effect at d=10 t=1.21 ------------------
    for mw in [1e-10, 1e-15]:
        label = f"E4_d10_mw{mw:.0e}"
        if label in done_labels:
            print(f"[exp] skip {label} (done)", flush=True); continue
        r = _run_one(label, d=10, target=1.210,
                     time_budget_s=120, init_split_depth=12, workers=8,
                     min_box_width=mw)
        results["runs"].append(r); _save(results)

    # ----- E5: joint-face depth at d=10 t=1.21 ----------------------
    for jd in [20, 10]:
        label = f"E5_d10_jd{jd}"
        if label in done_labels:
            print(f"[exp] skip {label} (done)", flush=True); continue
        r = _run_one(label, d=10, target=1.210,
                     time_budget_s=120, init_split_depth=12, workers=8,
                     joint_depth=jd)
        results["runs"].append(r); _save(results)

    results["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(results)
    print("\n[exp] ALL DONE -> ", OUT, flush=True)


if __name__ == "__main__":
    main()

"""Benchmark script: d=12 t=1.15 and d=12 t=1.2, serial.

Prints per-case wall-time, nodes_processed, leaves_certified, leaves_split.
Used to compare baseline vs. H2 (dual-cert McCormick) / H3 (exact rational LP).
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.parallel import parallel_branch_and_bound


def run_one(d: int, target: str, workers: int, max_nodes: int = 500_000_000,
            time_budget_s: float = 3600.0) -> dict:
    target_q = Fraction(target)
    t0 = time.time()
    res = parallel_branch_and_bound(
        d=d, target_c=target_q, workers=workers,
        init_split_depth=10, verbose=True,
        max_nodes=max_nodes, time_budget_s=time_budget_s,
    )
    elapsed = time.time() - t0
    return {
        "d": d,
        "target": target,
        "workers": workers,
        "success": res["success"],
        "elapsed_s": elapsed,
        "total_nodes": res["total_nodes"],
        "total_leaves_certified": res["total_leaves_certified"],
        "max_depth": res["max_depth"],
        "coverage_fraction": res["coverage_fraction"],
    }


def main():
    workers = int(os.environ.get("BENCH_WORKERS", "32"))
    d = int(os.environ.get("BENCH_D", "12"))
    targets = os.environ.get("BENCH_TARGETS", "1.25").split(",")
    budget = float(os.environ.get("BENCH_TIME_BUDGET_S", "21600"))  # 6h default
    h2_ratio = os.environ.get("INTERVAL_BNB_H2_RATIO", "1.1")
    print(f"[bench] d={d} H2 ratio={h2_ratio} (<1 = H2 on, >=1 = H2 off)")
    out = []
    for target in targets:
        print(f"[bench] running d={d} target={target} workers={workers}")
        r = run_one(d, target, workers=workers, time_budget_s=budget)
        print(f"[bench] d={d} t={target}: success={r['success']} "
              f"elapsed={r['elapsed_s']:.1f}s nodes={r['total_nodes']} "
              f"cert={r['total_leaves_certified']}")
        r["h2_ratio"] = h2_ratio
        out.append(r)
    tag = os.environ.get("BENCH_TAG", "default")
    path = os.path.join(_REPO, "data", f"bench_d{d}_{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()

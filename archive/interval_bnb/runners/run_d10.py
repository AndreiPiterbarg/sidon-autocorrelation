"""d=10 pipeline validation run.

val(10) ~= 1.24137 (not a record, but a realistic stress test).
Certifying val(10) >= 1.24 validates the full pipeline and gives
measured tree sizes to extrapolate from.

Writes results to interval_bnb/run_d10.log and interval_bnb/tree_d10.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bnb import branch_and_bound  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="interval BnB d=10 pipeline run")
    ap.add_argument("--target", type=str, default="1.24",
                    help="target c to certify (float or exact rational)")
    ap.add_argument("--max_nodes", type=int, default=30_000_000)
    ap.add_argument("--time_budget_s", type=float, default=3600.0)
    ap.add_argument("--method", choices=["natural", "autoconv", "mccormick", "combined"],
                    default="combined")
    ap.add_argument("--log_out", type=str,
                    default=os.path.join(_HERE, "run_d10.log"))
    ap.add_argument("--stats_out", type=str,
                    default=os.path.join(_HERE, "tree_d10.json"))
    args = ap.parse_args()

    target = Fraction(args.target)
    print(f"[run_d10] certifying val(10) >= {target}  method={args.method}")
    t0 = time.time()
    res = branch_and_bound(
        d=10, target_c=target, verbose=True,
        log_every=50_000, time_budget_s=args.time_budget_s,
        max_nodes=args.max_nodes, method=args.method,
    )
    elapsed = time.time() - t0
    print(f"[run_d10] SUCCESS={res.success} elapsed={elapsed:.1f}s  "
          f"nodes={res.stats.nodes_processed}")
    with open(args.stats_out, "w") as fh:
        d = res.stats.to_dict()
        d["d"] = 10
        d["target_c"] = target
        d["target_rational"] = str(res.target_q)
        d["success"] = res.success
        json.dump(d, fh, indent=2, default=str)
    print(f"[run_d10] stats written to {args.stats_out}")


if __name__ == "__main__":
    main()

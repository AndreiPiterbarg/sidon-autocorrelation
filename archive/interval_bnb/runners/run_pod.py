"""Pod-side BnB driver with d/target arguments.

Invoked by deploy_interval_bnb_pod.py launch -- --d 10 --target 1.24.
Writes data/interval_bnb_d{D}_t{T}.log and tree_d{D}.json into the
pod's data/ directory, which is fetched by the deploy script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bnb import branch_and_bound  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--max_nodes", type=int, default=200_000_000)
    ap.add_argument("--time_budget_s", type=float, default=7 * 86400.0)
    ap.add_argument("--min_box_width", type=float, default=1e-10)
    ap.add_argument("--method", default="combined")
    ap.add_argument("--log_every", type=int, default=200_000)
    ap.add_argument("--log_dir", type=str, default=os.path.join(_REPO, "data"))
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = serial, >1 = work-stealing multiprocessing")
    ap.add_argument("--init_split_depth", type=int, default=10,
                    help="log2 of starter sub-boxes seeded into the queue")
    ap.add_argument("--pull_batch_max", type=int, default=64,
                    help="max boxes per shared-queue put/get")
    ap.add_argument("--donate_threshold_floor", type=int, default=16,
                    help="min local stack size before worker donates")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    tag = f"d{args.d}_t{args.target.replace('.', 'p')}"
    log_path = os.path.join(args.log_dir, f"interval_bnb_{tag}.log")
    stats_path = os.path.join(args.log_dir, f"interval_bnb_{tag}.json")

    # Double-output: to terminal AND to the log file.
    log_fh = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, *streams): self.s = streams
        def write(self, x):
            for s in self.s: s.write(x); s.flush()
        def flush(self):
            for s in self.s: s.flush()

    sys.stdout = _Tee(sys.__stdout__, log_fh)

    from fractions import Fraction as _F
    target = _F(args.target)
    print(f"[run_pod] d={args.d}  target={target}  method={args.method}  "
          f"workers={args.workers}")
    print(f"[run_pod] max_nodes={args.max_nodes}  time_budget={args.time_budget_s}s")
    t0 = time.time()

    if args.workers <= 1:
        res = branch_and_bound(
            d=args.d, target_c=target, verbose=True,
            log_every=args.log_every, time_budget_s=args.time_budget_s,
            max_nodes=args.max_nodes, min_box_width=args.min_box_width,
            method=args.method,
        )
        elapsed = time.time() - t0
        stats = res.stats.to_dict()
        stats["success"] = res.success
        stats["target_rational"] = str(res.target_q)
    else:
        from interval_bnb.parallel import parallel_branch_and_bound
        result = parallel_branch_and_bound(
            d=args.d, target_c=target,
            init_split_depth=args.init_split_depth,
            workers=args.workers,
            verbose=True,
            max_nodes=args.max_nodes,
            min_box_width=args.min_box_width,
            pull_batch_max=args.pull_batch_max,
            donate_threshold_floor=args.donate_threshold_floor,
            time_budget_s=args.time_budget_s,
        )
        elapsed = time.time() - t0
        stats = dict(result)
        stats["target_rational"] = str(target)

    stats["d"] = args.d
    stats["target_c"] = target
    stats["elapsed_s"] = elapsed
    print(f"[run_pod] FINAL success={stats['success']} elapsed={elapsed:.1f}s")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"[run_pod] stats -> {stats_path}")


if __name__ == "__main__":
    main()

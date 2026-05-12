#!/usr/bin/env python3.11
"""Farkas-certified Lasserre sweep targeted at val(d) > 1.28+.

Launched via:
    python -m cpupod launch scripts/farkas_push_sweep.py

For each (d, order, t_test) in the preconfigured plan, run farkas_certify_cg
(column-generation Farkas at fixed t).  Records every result (CERTIFIED,
NOT_CERTIFIED, FEAS) to stdout and stops a given d's loop at the first CERT.

Tuning vs. the existing scripts/push_1_28.sh:
  - nthreads=64 (previously 16)
  - max_denom_S = 10**14 (previously 10**12): tightens residual ~100x, widens
    the "margin gap" so Cholesky rounding wins by a larger amount
  - eig_margin = 1e-10
  - n_add_per_iter = 30 (up from 25): fewer CG outer iterations
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.set_int_max_str_digits(10**7)
except AttributeError:
    pass

from certified_lasserre.farkas_cg import farkas_certify_cg


OUT_DIR = Path("data/farkas_push_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plan: (d, order, descending_t_values).
# Stop each row at first CERTIFIED.
PLAN = [
    # Order 3 at moderate d — most likely to certify high thresholds.
    (10, 3, [1.28, 1.275, 1.27, 1.265, 1.26, 1.25, 1.24, 1.23, 1.22]),
    (12, 3, [1.29, 1.285, 1.28, 1.275, 1.27, 1.265, 1.26, 1.255, 1.25]),
    (14, 3, [1.29, 1.285, 1.282, 1.28, 1.275, 1.27, 1.265, 1.26]),
    # Order 2 at large d — cheap but wider gap.  Sanity check the trajectory.
    (16, 2, [1.25, 1.22, 1.20, 1.18, 1.16]),
    (20, 2, [1.25, 1.22, 1.20, 1.18]),
]


def run_one(d: int, order: int, t_test: float) -> dict:
    t0 = time.time()
    try:
        res, _ = farkas_certify_cg(
            d=d, order=order, t_test=t_test,
            max_cg_iters=50, n_add_per_iter=30,
            max_denom_S=10**14, max_denom_mu=10**14,
            eig_margin=1e-10, nthreads=64,
            verbose=True,
        )
        row = {
            "d": d, "order": order, "t_test": t_test,
            "status": res.status,
            "mu0": res.mu0_float,
            "residual_l1": res.residual_l1_float,
            "margin": res.safety_margin_float,
            "time": time.time() - t0,
            "notes": res.notes,
        }
        if res.status == "CERTIFIED":
            row["lb_rig_decimal"] = res.lb_rig_decimal
            row["lb_rig_num_den"] = [int(res.lb_rig.numerator),
                                     int(res.lb_rig.denominator)]
            path = OUT_DIR / f"d{d}_o{order}_t{t_test:.5f}.json"
            path.write_text(json.dumps(row, indent=2))
    except Exception as exc:
        row = {
            "d": d, "order": order, "t_test": t_test,
            "status": f"ERR:{type(exc).__name__}",
            "error": str(exc)[:400],
            "time": time.time() - t0,
        }
    return row


def main():
    total_t0 = time.time()
    summary = []
    for d, order, t_list in PLAN:
        print(f"\n{'#'*60}", flush=True)
        print(f"# d={d} order={order} sweeping t values (desc)", flush=True)
        print(f"#"*60, flush=True)
        for t in t_list:
            print(f"\n--- d={d} o={order} t={t:.4f} ---", flush=True)
            row = run_one(d, order, t)
            summary.append(row)
            print(f"    => status={row['status']}  "
                  f"margin={row.get('margin', 0):+.2e}  "
                  f"time={row['time']:.1f}s",
                  flush=True)
            if row["status"] == "CERTIFIED":
                print(f"*** CERTIFIED val({d}) > {t} at order {order} ***",
                      flush=True)
                break  # don't waste time on lower t
            if row["status"].startswith("ERR"):
                print(f"    error, skipping rest of d={d}: {row.get('error','')}",
                      flush=True)
                break

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    total_dt = time.time() - total_t0

    # Print human-readable summary.
    print(f"\n{'='*60}", flush=True)
    print(f"SWEEP SUMMARY  total_time={total_dt:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)
    best = None
    for row in summary:
        status = row["status"]
        d, o, t = row["d"], row["order"], row["t_test"]
        margin = row.get("margin", 0)
        time_s = row["time"]
        print(f"  d={d:2d} o={o} t={t:.4f}  {status:15s}  margin={margin:+.2e}  t={time_s:.0f}s",
              flush=True)
        if status == "CERTIFIED":
            if best is None or t > best[2]:
                best = (d, o, t)
    if best is not None:
        print(f"\nBEST CERTIFIED: val({best[0]}) > {best[2]} at order={best[1]}",
              flush=True)
    else:
        print(f"\nNo bound certified. Highest-attempted: see per-(d,t) status above.",
              flush=True)


if __name__ == "__main__":
    main()

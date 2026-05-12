"""Order-3 dense trajectory at d ∈ {4, 6, 8, 10} — local laptop run.

For each d, runs Farkas-certify with bisection at order=3 / bandwidth=d-1
(dense). Records the certified lower bound, total wall, peak RSS, and
the SDP problem dimensions.

Output: lasserre/trajectory/results/order3_d{d}_trajectory.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lasserre.trajectory.run_trajectory import measure_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, nargs='+', default=[4, 6, 8, 10])
    ap.add_argument('--time_budget_s', type=float, default=1500.0,
                    help='wall budget per d')
    ap.add_argument('--results_dir', type=str,
                    default=str(Path(_HERE) / 'results'))
    args = ap.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for d in args.d:
        out_path = Path(args.results_dir) / f'order3_d{d}_trajectory.json'
        # bandwidth = d-1 → dense
        b = d - 1
        try:
            r = measure_d(
                d=d, order=3, bandwidth=b,
                t_lo=1.05, t_hi=1.30,
                bisect_tol=1e-3,
                time_budget_s=args.time_budget_s,
            )
        except KeyboardInterrupt:
            print(f'[abort] d={d} interrupted', flush=True)
            raise

        with open(out_path, 'w') as f:
            json.dump(asdict(r), f, indent=2)
        print(f'\n[saved] {out_path}', flush=True)
        print(f'  status:                {r.status}')
        print(f'  largest_certified_t:   {r.largest_certified_t}')
        print(f'  total_wall_s:          {r.total_wall_s:.1f}')
        print(f'  peak_rss_overall_gb:   {r.peak_rss_overall_gb:.2f}')
        if r.status == 'OOM_OR_SOLVER_FAILURE':
            print('  ===> stopping further d (OOM/failure) <===', flush=True)
            break


if __name__ == '__main__':
    main()

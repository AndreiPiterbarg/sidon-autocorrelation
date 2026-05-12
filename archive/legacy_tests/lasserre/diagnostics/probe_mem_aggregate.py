#!/usr/bin/env python
"""Aggregate probe_mem.py outputs into a compact memory table.

Usage:
    python tests/probe_mem_aggregate.py data/probe_d12
    python tests/probe_mem_aggregate.py data/probe_d12 --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dir', help='Directory containing probe_*.json files')
    p.add_argument('--csv', type=str, default=None)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, '*.json')))
    if not files:
        print(f'No *.json in {args.dir}', file=sys.stderr)
        return 1

    rows = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f'[skip] {f}: {e}', file=sys.stderr)
            continue
        name = os.path.basename(f).replace('.json', '')
        rows.append({
            'name': name,
            'd': r.get('d'),
            'threads': r.get('threads'),
            'z2': r.get('z2_full'),
            'upper_loc': r.get('upper_loc'),
            'solve_form': r.get('solve_form'),
            'tol': r.get('tol'),
            'build_only': r.get('build_only'),
            'windows': r.get('active_windows_limit'),
            'peak_gb': r.get('peak_rss_gb'),
            'total_s': r.get('total_wall_s'),
            'precompute_s': r.get('phase_times_s', {}).get('precompute_s'),
            'build_s': r.get('phase_times_s', {}).get('build_s'),
            'optimize_s': r.get('phase_times_s', {}).get('optimize_s'),
            'verdict': r.get('verdict'),
            'lam': r.get('lambda_star'),
        })

    # Print wide table.
    hdr = f"{'name':<25} {'d':>3} {'thr':>4} {'z2':>3} {'uloc':>4} " \
          f"{'sf':<6} {'tol':>7} {'BO':>3} {'wins':>5} " \
          f"{'peak_GB':>8} {'build_s':>8} {'opt_s':>8} " \
          f"{'verdict':<8} {'lam':>10}"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['name']:<25} {r['d']!s:>3} {r['threads']!s:>4} "
              f"{str(r['z2'])[:3]:>3} {str(r['upper_loc'])[:4]:>4} "
              f"{str(r['solve_form'])[:6]:<6} "
              f"{r['tol']:>7.0e} {str(r['build_only'])[:3]:>3} "
              f"{str(r['windows'])[:5]:>5} "
              f"{(r['peak_gb'] or 0):>8.2f} "
              f"{(r['build_s'] or 0):>8.1f} "
              f"{(r['optimize_s'] or 0):>8.1f} "
              f"{str(r['verdict'])[:8]:<8} "
              f"{(r['lam'] if r['lam'] == r['lam'] else 0):>10.4e}")

    if args.csv:
        import csv
        cols = list(rows[0].keys()) if rows else []
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nCSV -> {args.csv}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

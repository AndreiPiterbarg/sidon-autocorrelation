"""AGENT G — orchestration probe.

(a) Count canonical compositions at various (d, S) that the cascade might use.
(b) Time v6 certify_composition on representative cells.

This is RESEARCH-ONLY (no production edits).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from math import comb

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))

from compositions import generate_canonical_compositions_batched

import _coarse_bnb_v6 as v6


def count_canonical(d, S, cap=2_000_000):
    """Count canonical compositions of S into d non-negative bins."""
    total_raw = comb(S + d - 1, d - 1)
    if total_raw > 50_000_000:
        return {'d': d, 'S': S, 'raw': int(total_raw),
                'canon_estimated': int(total_raw // 2 + 1),
                'canon_exact': None,
                'note': 'raw too large; ~half (palindromic exception)'}
    cnt = 0
    for batch in generate_canonical_compositions_batched(d, S, batch_size=200_000):
        cnt += batch.shape[0]
        if cnt > cap:
            return {'d': d, 'S': S, 'raw': int(total_raw),
                    'canon_estimated': cnt, 'canon_exact': None,
                    'note': f'aborted at cap={cap}'}
    return {'d': d, 'S': S, 'raw': int(total_raw),
            'canon_exact': cnt, 'canon_estimated': cnt}


def time_one_cell(d, S, c_int, c_target, max_depth=2):
    windows = v6.build_all_windows(d)
    bundle = v6.get_bundle(windows)
    c = np.asarray(c_int, dtype=np.float64)
    cell = v6.Cell.from_integer_composition(c, S)
    t0 = time.perf_counter()
    r = v6.cert_cell(cell, windows, c_target, max_depth=max_depth,
                       bundle=bundle)
    dt = time.perf_counter() - t0
    return {'d': d, 'S': S, 'c': list(c_int), 'c_target': c_target,
            'certified': bool(r.certified), 'tier': r.tier_used,
            'depth_used': int(r.depth_used), 'bound': float(getattr(r, 'bound', 0.0) or 0.0),
            'seconds': round(dt, 4)}


def main():
    out = {'counts': [], 'timings': []}

    # ---- (a) counts ----
    sizes = [
        (2, 20), (2, 40), (2, 64),
        (4, 20), (4, 40), (4, 80), (4, 160),
        (6, 15), (6, 30), (6, 60),
        (8, 10), (8, 16), (8, 24),
        (10, 10), (10, 20),
    ]
    for (d, S) in sizes:
        try:
            r = count_canonical(d, S, cap=500_000)
        except Exception as e:
            r = {'d': d, 'S': S, 'error': str(e)}
        out['counts'].append(r)
        print(f"  d={d:>2} S={S:>3}  raw={r.get('raw'):>12}  canon={r.get('canon_exact') or r.get('canon_estimated'):>12}  {r.get('note','')}")

    # ---- (b) timings ----
    # Representative cells, mostly the "uniform" hard ones (most borderline).
    cases = [
        # d=2: easy, just for baseline
        (2, 20, (10, 10), 1.281),
        # d=4 uniform — the canonical hard cell at S=80
        (4, 80, (20, 20, 20, 20), 1.281),
        (4, 80, (19, 21, 21, 19), 1.281),
        (4, 40, (10, 10, 10, 10), 1.281),
        # d=6 uniform
        (6, 30, (5, 5, 5, 5, 5, 5), 1.281),
        # d=8 uniform
        (8, 16, (2, 2, 2, 2, 2, 2, 2, 2), 1.281),
        (8, 24, (3, 3, 3, 3, 3, 3, 3, 3), 1.281),
        # d=10 uniform
        (10, 20, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 1.281),
    ]
    print("\nTimings (one-cell v6 cert):")
    for (d, S, c, ct) in cases:
        try:
            t = time_one_cell(d, S, c, ct, max_depth=2)
        except Exception as e:
            t = {'d': d, 'S': S, 'c': list(c), 'c_target': ct, 'error': str(e)}
        out['timings'].append(t)
        print(f"  d={d:>2} S={S:>3} c={t.get('c')!s:<30} -> "
              f"cert={t.get('certified')} tier={t.get('tier')} "
              f"{t.get('seconds','?')}s")

    with open(os.path.join(_dir, '_audit_G_probe.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nWrote _audit_G_probe.json")


if __name__ == '__main__':
    main()

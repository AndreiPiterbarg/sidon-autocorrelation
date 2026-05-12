"""AGENT G — does d=4 (20,20,20,20) close when refined to d=8 children?"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))

import _coarse_bnb_v6 as v6
from _coarse_bnb_v2_orchestrator import enumerate_children, canonical_form


def refine_and_certify(parent_c, S, c_target, max_depth=3):
    d_parent = len(parent_c)
    d_child = 2 * d_parent
    children = enumerate_children(np.asarray(parent_c, dtype=np.int64))
    # canon dedupe
    seen, uniq = set(), []
    for c in children:
        k = canonical_form(c)
        if k not in seen:
            seen.add(k); uniq.append(c)
    windows = v6.build_all_windows(d_child)
    bundle = v6.get_bundle(windows)
    t0 = time.perf_counter()
    closed = 0; survivors = []
    tiers = {}
    for c in uniq:
        cell = v6.Cell.from_integer_composition(c.astype(np.float64), S)
        r = v6.cert_cell(cell, windows, c_target, max_depth=max_depth,
                          bundle=bundle)
        if r.certified:
            closed += 1
            tiers[r.tier_used] = tiers.get(r.tier_used, 0) + 1
        else:
            survivors.append(c.tolist())
    dt = time.perf_counter() - t0
    return {
        'parent': list(parent_c), 'S': S, 'c_target': c_target,
        'children_raw': len(children), 'children_canon': len(uniq),
        'closed': closed, 'open': len(survivors), 'tiers': tiers,
        'seconds': round(dt, 2), 'open_examples': survivors[:10],
    }


def main():
    out = []
    cases = [
        ((20, 20, 20, 20), 80, 1.281, 3),
        ((19, 21, 21, 19), 80, 1.281, 3),
        ((10, 10), 20, 1.281, 3),  # d=2 -> d=4, expect (5,5,5,5) hard children
        # extra: smaller S?
        ((10, 10, 10, 10), 40, 1.281, 3),
    ]
    for (c, S, ct, md) in cases:
        try:
            r = refine_and_certify(c, S, ct, md)
            print(f"parent={c} S={S} -> {r['closed']}/{r['children_canon']} closed "
                  f"in {r['seconds']}s  tiers={r['tiers']}  open={r['open']}")
            if r['open_examples']:
                print(f"   first open child: {r['open_examples'][0]}")
        except Exception as e:
            r = {'parent': list(c), 'S': S, 'error': str(e)}
            print(f"  ERR: {e}")
        out.append(r)

    with open(os.path.join(_dir, '_audit_G_probe2.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()

"""Re-verify the 1.25 proof with B35 DISABLED.

B35 in `_coarse_bnb_v4.py::is_cell_empty` is unsound (counter-example in
`_b35_audit.py`).  At the d=8 cascade root level it never fired (verified).
This script re-runs the d=16 refinement of the 4 stragglers using only the
sound B34 empty test, to confirm the 1.25 result holds without B35.

If all 7920 children close, the 1.25 result is independently verified
without depending on the unsound B35 test.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
from itertools import product

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v4 as v4

# Monkey-patch is_cell_empty to ONLY use B34 (the sound tests).
_orig_is_cell_empty = v4.is_cell_empty


def is_cell_empty_b34_only(cell, eps: float = 1e-12) -> bool:
    """Sound: cell empty iff Σlo > 1 OR Σhi < 1.  Drops B35."""
    if cell.lo.sum() > 1.0 + eps:
        return True
    if cell.hi.sum() < 1.0 - eps:
        return True
    return False


v4.is_cell_empty = is_cell_empty_b34_only
print("[patch] is_cell_empty replaced with B34-only version.", flush=True)


stragglers = [
    [4, 1, 0, 0, 2, 2, 2, 5],
    [4, 1, 0, 0, 3, 1, 2, 5],
    [4, 1, 1, 0, 2, 2, 1, 5],
    [4, 1, 1, 0, 2, 2, 2, 4],
]


def gen_children(parent):
    d_p = len(parent)
    opts = [[(a, p - a) for a in range(p + 1)] for p in parent]
    children = []
    for combo in product(*opts):
        ch = np.zeros(2 * d_p, dtype=np.int64)
        for i, (a, b) in enumerate(combo):
            ch[2 * i] = a
            ch[2 * i + 1] = b
        children.append(ch)
    return children


def canonical_dedup(children):
    seen = set()
    out = []
    for c in children:
        rc = tuple(c[::-1])
        ct = tuple(c)
        key = ct if ct <= rc else rc
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main():
    d_child = 16
    S = 16
    c_target = 1.25

    print(f"\n=== Re-verify 1.25 d=16 refinement, B35 DISABLED ===", flush=True)
    print(f"  d_child={d_child}, S={S}, c={c_target}", flush=True)

    all_children = []
    for parent in stragglers:
        ch = gen_children(parent)
        all_children.extend(ch)
    unique = canonical_dedup(all_children)
    print(f"  total raw children: {len(all_children)}", flush=True)
    print(f"  unique canonical:   {len(unique)}", flush=True)

    windows = v4.build_all_windows(d_child)
    v4.get_sdp_template(d_child)
    v4.get_joint_template(d_child, 4)
    print(f"  templates warmed", flush=True)

    t0 = time.time()
    counts = {'B1': 0, 'empty': 0, 'F': 0, 'L': 0, 'L_joint': 0, 'split': 0}
    open_cells = []
    log_every = max(1, len(unique) // 20)

    for k, c in enumerate(unique):
        if k > 0 and k % log_every == 0:
            elapsed = time.time() - t0
            n_done = sum(counts.values())
            print(f"  [{k:>5}/{len(unique)}] closed={n_done}  open={len(open_cells)}  "
                  f"elapsed={elapsed:.1f}s  "
                  f"(B1={counts['B1']} F={counts['F']} L={counts['L']} "
                  f"Lj={counts['L_joint']} sp={counts['split']})",
                  flush=True)
        try:
            r = v4.certify_composition(c.astype(np.float64), S, d_child, c_target,
                                          windows=windows, max_depth=3)
        except Exception as e:
            open_cells.append(c)
            continue
        if r.certified:
            counts[r.tier_used] = counts.get(r.tier_used, 0) + 1
        else:
            open_cells.append(c)

    elapsed = time.time() - t0
    print(f"\n=== Final tally (B35 disabled) ===", flush=True)
    print(f"  closed:  B1={counts['B1']}  empty={counts['empty']}  "
          f"F={counts['F']}  L={counts['L']}  Lj={counts['L_joint']}  split={counts['split']}",
          flush=True)
    print(f"  open:    {len(open_cells)}", flush=True)
    print(f"  elapsed: {elapsed:.1f}s", flush=True)
    if not open_cells:
        print(f"\n  *** 1.25 RESULT INDEPENDENTLY VERIFIED WITHOUT B35 ***", flush=True)
    else:
        print(f"\n  !!! {len(open_cells)} cells unclosed with B35 disabled — investigate !!!",
              flush=True)
        for c in open_cells[:5]:
            print(f"    open: {c.tolist()}", flush=True)


if __name__ == '__main__':
    main()

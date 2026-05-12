"""Refine the 4 d=8 stragglers to d=16 children, certify each child."""
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
from itertools import product

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import _coarse_bnb_v4 as v4
from _d16_F_bench import _prune_coarse_count_cell


stragglers = [
    [4, 1, 0, 0, 2, 2, 2, 5],
    [4, 1, 0, 0, 3, 1, 2, 5],
    [4, 1, 1, 0, 2, 2, 1, 5],
    [4, 1, 1, 0, 2, 2, 2, 4],
]


def gen_children(parent):
    """All children at d_child=2*d_parent. Each parent bin c_i splits into
    (a, c_i-a) for a ∈ [0, c_i]. Returns list of np.array."""
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
    """Keep one of (c, rev(c)) per pair."""
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
    S = 16  # mass conserved
    c_target = 1.25

    print(f"\n=== Refine 4 stragglers to d={d_child}, c={c_target} ===", flush=True)
    print(f"  S={S} (mass conserved when d doubles)", flush=True)

    # JIT warm
    warm = np.zeros((1, d_child), dtype=np.int32)
    warm[0, 0] = S
    _prune_coarse_count_cell(warm, d_child, S, c_target)

    # Generate + dedup children
    t0 = time.time()
    all_children = []
    for parent in stragglers:
        ch = gen_children(parent)
        print(f"  parent c={parent}: {len(ch)} raw children", flush=True)
        all_children.extend(ch)
    unique = canonical_dedup(all_children)
    print(f"  total raw: {len(all_children)}, unique canonical: {len(unique)}  "
          f"[{time.time()-t0:.2f}s]", flush=True)

    # Numba F pass
    print(f"\n  Numba F pass at d={d_child}...", flush=True)
    t0 = time.time()
    batch = np.array(unique, dtype=np.int32)
    survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
        batch, d_child, S, c_target)
    n_grid_p = int((~survived).sum())
    n_grid_s = int(survived.sum())
    n_uncert = int(n_neg)
    print(f"  grid-pruned:      {n_grid_p}", flush=True)
    print(f"  grid-survivors:   {n_grid_s}", flush=True)
    print(f"  cell-uncertain:   {n_uncert}  (residue for v4)", flush=True)
    print(f"  min_net:          {min_net:.6f}", flush=True)
    print(f"  time:             {time.time()-t0:.2f}s", flush=True)

    # v4 on residue
    if n_uncert == 0 and n_grid_s == 0:
        print(f"\n  *** ALL CHILDREN CLOSED BY NUMBA F AT d={d_child}, c={c_target} ***", flush=True)
        print(f"  *** Combined with earlier d=8 result: full proof at this (d, S) ***", flush=True)
        return

    if n_grid_s > 0:
        print(f"\n  Grid-survivors at d={d_child}: NEED d=32 refine; bailing.", flush=True)
        return

    residue_mask = (~survived) & neg_mask
    residue = [batch[i].astype(np.int64).copy() for i in np.where(residue_mask)[0]]
    print(f"\n  Running v4 cell-cert on {len(residue)} residue cells at d={d_child}...",
          flush=True)
    windows = v4.build_all_windows(d_child)
    v4.get_sdp_template(d_child)
    v4.get_joint_template(d_child, 4)

    t0 = time.time()
    counts = {'B1': 0, 'empty': 0, 'F': 0, 'L': 0, 'L_joint': 0, 'split': 0}
    open_cells = []
    for k, c in enumerate(residue):
        if k > 0 and k % max(1, len(residue) // 20) == 0:
            print(f"    [{k}/{len(residue)}] closed={sum(counts.values())} "
                  f"open={len(open_cells)} elapsed={time.time()-t0:.1f}s", flush=True)
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
    print(f"\n  v4 SUMMARY:", flush=True)
    print(f"    closed: B1={counts['B1']} F={counts['F']} L={counts['L']} "
          f"L_joint={counts['L_joint']} split={counts['split']}", flush=True)
    print(f"    still open: {len(open_cells)}", flush=True)
    print(f"    elapsed: {elapsed:.1f}s", flush=True)
    if open_cells:
        for c in open_cells[:5]:
            print(f"    open: {c.tolist()}", flush=True)
    else:
        print(f"\n  *** ALL CHILDREN CLOSED AT d=16 ***", flush=True)
        print(f"  *** Combined with d=8 result: PROOF OF C_{{1a}} >= 1.25 COMPLETE ***", flush=True)


if __name__ == '__main__':
    main()

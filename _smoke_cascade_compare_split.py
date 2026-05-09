"""Compare cascade WITH vs WITHOUT split-cell SDP at L0 (d=2) and L1 (d=4).

Standalone driver that mirrors the cascade's chain F+FN+Q+QN+L (with optional
SP) but enumerates compositions directly to avoid Windows multiproc handle
limits in `run_cascade.py`.

Pipeline:
  L0: enumerate canonical compositions of length d0=2*n_half summing to
      S0=4*n_half*m.  Apply F+FN+Q+QN+L to get L0-survivors.
      WITHOUT SP: report count.
      WITH SP:    apply split-cell SDP to L0 survivors, report count.

  L1: for each L0 SP-survivor (parent), generate refinement children
      (length-2d compositions c' with c'_{2i} + c'_{2i+1} = c_i for parent c).
      Apply F+FN+Q+QN+L on the union of children.
      WITHOUT SP: report count.
      WITH SP:    apply split-cell SDP, report count.

Usage:  python _smoke_cascade_compare_split.py
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_canonical_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows
from _L_bench import _build_A_matrices
from post_filters import (apply_FN_filter, apply_Q_filter, apply_QN_filter,
                          apply_L_filter, apply_split_cell_filter_parallel)


def run_chain(survivors, n_half, m, c_target, label, with_sp=False, n_workers=2):
    """Run F-survivors -> +FN -> +Q -> +QN -> +L (-> +SP if with_sp).
    Returns dict with per-stage counts and times."""
    out = {'label': label, 'n_in': int(len(survivors)),
            'with_sp': with_sp}
    if len(survivors) == 0:
        return out

    t = time.time()
    fn = apply_FN_filter(survivors, n_half, m, c_target)
    out['n_FN'] = int(len(fn)); out['t_FN'] = round(time.time() - t, 3)

    t = time.time()
    q = apply_Q_filter(fn, n_half, m, c_target)
    out['n_Q'] = int(len(q)); out['t_Q'] = round(time.time() - t, 3)

    t = time.time()
    qn = apply_QN_filter(q, n_half, m, c_target)
    out['n_QN'] = int(len(qn)); out['t_QN'] = round(time.time() - t, 3)

    t = time.time()
    if len(qn) > 0:
        L = apply_L_filter(qn, n_half, m, c_target, solver='MOSEK')
    else:
        L = qn
    out['n_L'] = int(len(L)); out['t_L'] = round(time.time() - t, 3)
    out['L_survivors'] = L

    if with_sp:
        t = time.time()
        if len(L) > 0:
            sp = apply_split_cell_filter_parallel(
                L, n_half, m, c_target, n_workers=n_workers,
                max_d=10, early_terminate=True)
        else:
            sp = L
        out['n_SP'] = int(len(sp)); out['t_SP'] = round(time.time() - t, 3)
        out['SP_survivors'] = sp
    else:
        out['n_SP'] = out['n_L']
        out['SP_survivors'] = L
    return out


def refine_to_d_children(parents, d_child):
    """Generate length-d_child children of length-d_parent parents.

    Each child c' satisfies c'_{2i} + c'_{2i+1} = c_i for the parent c.
    Children are not deduplicated by symmetry; we just yield the full set.
    """
    if len(parents) == 0:
        return np.empty((0, d_child), dtype=np.int32)
    d_parent = parents.shape[1]
    assert d_child == 2 * d_parent
    out = []
    for c in parents:
        # For each i, choose c'_{2i} in [0, c_i], c'_{2i+1} = c_i - c'_{2i}
        ranges = [range(int(c[i]) + 1) for i in range(d_parent)]
        for vals in product(*ranges):
            child = np.empty(d_child, dtype=np.int32)
            for i in range(d_parent):
                child[2 * i] = vals[i]
                child[2 * i + 1] = int(c[i]) - vals[i]
            out.append(child)
    return np.vstack(out) if out else np.empty((0, d_child), dtype=np.int32)


def main():
    n_half = 1
    m = 20
    c_target = 1.281
    n_workers = 2

    print(f'=== Cascade comparison: n_half={n_half}, m={m}, c_target={c_target} ===\n')

    # ----- L0: d=2 -----
    d0 = 2 * n_half
    S0 = 4 * n_half * m
    print(f'--- L0: d={d0}, S={S0} ---')
    full = []
    for batch in generate_canonical_compositions_batched(d0, S0, batch_size=200_000):
        full.append(batch.astype(np.int32, copy=False))
    full = np.vstack(full)
    print(f'  canonical comps: {len(full)}')

    warm = np.zeros((1, d0), dtype=np.int32); warm[0,0] = 2*m
    prune_F(warm, n_half, m, c_target)

    sF = prune_F(full, n_half, m, c_target)
    F0 = full[sF]
    print(f'  F-survivors:     {len(F0)}')

    res_no = run_chain(F0, n_half, m, c_target, 'L0_no_SP', with_sp=False, n_workers=n_workers)
    print(f'  WITHOUT SP: F={res_no["n_in"]} -> FN={res_no["n_FN"]} -> Q={res_no["n_Q"]}'
          f' -> QN={res_no["n_QN"]} -> L={res_no["n_L"]}'
          f'  (wall total {sum(res_no.get(k,0) for k in ("t_FN","t_Q","t_QN","t_L")):.2f}s)')

    res_sp = run_chain(F0, n_half, m, c_target, 'L0_with_SP', with_sp=True, n_workers=n_workers)
    print(f'  WITH SP:    F={res_sp["n_in"]} -> FN={res_sp["n_FN"]} -> Q={res_sp["n_Q"]}'
          f' -> QN={res_sp["n_QN"]} -> L={res_sp["n_L"]} -> SP={res_sp["n_SP"]}'
          f'  (SP wall {res_sp.get("t_SP", 0):.2f}s)')

    extra_l0 = res_sp['n_L'] - res_sp['n_SP']
    print(f'  L0 extra prunes from SP: {extra_l0}\n')

    # ----- L1: d=4, children of L0 SP-survivors (use no-SP survivors so the
    #          comparison is fair: SP would only ADD prunes, never remove
    #          parents that should be examined further) -----
    parents = res_no['L_survivors']
    d1 = 2 * d0
    print(f'--- L1: d={d1}, children of {len(parents)} L0 L-survivors ---')
    children = refine_to_d_children(parents, d1)
    print(f'  total children:  {len(children)}')
    sF = prune_F(children, n_half * 2, m, c_target)
    F1 = children[sF]
    print(f'  F-survivors:     {len(F1)}')

    n_half_child = n_half * 2  # at L1, n_half doubles too (d_child = 2*n_half_child)
    res_no1 = run_chain(F1, n_half_child, m, c_target, 'L1_no_SP', with_sp=False, n_workers=n_workers)
    print(f'  WITHOUT SP: F={res_no1["n_in"]} -> FN={res_no1["n_FN"]} -> Q={res_no1["n_Q"]}'
          f' -> QN={res_no1["n_QN"]} -> L={res_no1["n_L"]}'
          f'  (wall {sum(res_no1.get(k,0) for k in ("t_FN","t_Q","t_QN","t_L")):.2f}s)')

    res_sp1 = run_chain(F1, n_half_child, m, c_target, 'L1_with_SP', with_sp=True, n_workers=n_workers)
    print(f'  WITH SP:    F={res_sp1["n_in"]} -> FN={res_sp1["n_FN"]} -> Q={res_sp1["n_Q"]}'
          f' -> QN={res_sp1["n_QN"]} -> L={res_sp1["n_L"]} -> SP={res_sp1["n_SP"]}'
          f'  (SP wall {res_sp1.get("t_SP", 0):.2f}s)')

    extra_l1 = res_sp1['n_L'] - res_sp1['n_SP']
    print(f'  L1 extra prunes from SP: {extra_l1}')

    summary = {
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target,
                    'n_workers': n_workers},
        'L0': {
            'd': d0, 'S': S0,
            'total_canonical': int(len(full)),
            'F_survivors': int(len(F0)),
            'no_SP': {k: v for k, v in res_no.items()
                      if k != 'L_survivors' and k != 'SP_survivors'},
            'with_SP': {k: v for k, v in res_sp.items()
                        if k != 'L_survivors' and k != 'SP_survivors'},
            'extra_prunes_from_SP': int(extra_l0),
        },
        'L1': {
            'd': d1, 'parents': int(len(parents)),
            'total_children': int(len(children)),
            'F_survivors': int(len(F1)),
            'no_SP': {k: v for k, v in res_no1.items()
                      if k != 'L_survivors' and k != 'SP_survivors'},
            'with_SP': {k: v for k, v in res_sp1.items()
                        if k != 'L_survivors' and k != 'SP_survivors'},
            'extra_prunes_from_SP': int(extra_l1),
        },
    }

    out_path = os.path.join(_HERE, '_smoke_cascade_compare_split.json')
    with open(out_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f'\nWrote {out_path}')

    print(f'\n=== FINAL ===')
    print(f'  L0: {res_no["n_L"]} L-survivors -> {res_sp["n_SP"]} SP-survivors  (extra prunes: {extra_l0})')
    print(f'  L1: {res_no1["n_L"]} L-survivors -> {res_sp1["n_SP"]} SP-survivors  (extra prunes: {extra_l1})')


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()
    main()

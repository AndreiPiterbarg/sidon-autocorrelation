"""Sweep cascade convergence at (n_half=1, m=20) over c_target values.

For each c_target in {1.30, 1.32, 1.35, 1.40, 1.45}:
  L0: enumerate canonical comps (d=2, S=80) -> F+FN+Q+QN+L (no SP) -> count survivors.
  L1: refine 2-4 sampled L0 survivors into d=4 children -> F+FN+Q+QN+L -> count survivors per parent.
  Mark "converging" if L1 avg per-parent survivors < (L0 survivors) at parent.

Mirrors `_smoke_cascade_compare_split.py` to avoid Windows multiproc handle issues.
n_workers capped at 2 to avoid Pool startup contention.
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
from post_filters import (apply_FN_filter, apply_Q_filter, apply_QN_filter,
                          apply_L_filter, apply_split_cell_filter_parallel)


C_TARGETS = [1.30, 1.32, 1.35, 1.40, 1.45]
N_HALF = 1
M = 20
N_WORKERS = 2
L1_PARENT_SAMPLE = 4   # sample up to this many L0 survivors for L1 expansion


def run_chain(survivors, n_half, m, c_target, with_sp=False, n_workers=2):
    """F-survivors -> +FN -> +Q -> +QN -> +L (-> +SP optional).
    Returns dict with counts and intermediate survivors."""
    out = {'n_in': int(len(survivors)), 'with_sp': with_sp}
    if len(survivors) == 0:
        for k in ('n_FN','n_Q','n_QN','n_L','n_SP'):
            out[k] = 0
        out['L_survivors'] = survivors
        out['SP_survivors'] = survivors
        return out

    t0 = time.time()
    fn = apply_FN_filter(survivors, n_half, m, c_target)
    out['n_FN'] = int(len(fn)); out['t_FN'] = round(time.time() - t0, 3)

    t0 = time.time()
    q = apply_Q_filter(fn, n_half, m, c_target) if len(fn) > 0 else fn
    out['n_Q'] = int(len(q)); out['t_Q'] = round(time.time() - t0, 3)

    t0 = time.time()
    qn = apply_QN_filter(q, n_half, m, c_target) if len(q) > 0 else q
    out['n_QN'] = int(len(qn)); out['t_QN'] = round(time.time() - t0, 3)

    t0 = time.time()
    L = apply_L_filter(qn, n_half, m, c_target, solver='MOSEK') if len(qn) > 0 else qn
    out['n_L'] = int(len(L)); out['t_L'] = round(time.time() - t0, 3)
    out['L_survivors'] = L

    if with_sp and len(L) > 0:
        t0 = time.time()
        try:
            sp = apply_split_cell_filter_parallel(
                L, n_half, m, c_target, n_workers=n_workers,
                max_d=10, early_terminate=True)
        except Exception as e:
            print(f'    SP exception: {e}', flush=True)
            sp = L
        out['n_SP'] = int(len(sp)); out['t_SP'] = round(time.time() - t0, 3)
        out['SP_survivors'] = sp
    else:
        out['n_SP'] = out['n_L']; out['SP_survivors'] = L
    return out


def refine_to_d_children(parents, d_child):
    """Given length-d_parent compositions `parents`, generate length-d_child=2*d_parent
    children where c'_{2i}+c'_{2i+1} = parent[i]."""
    if len(parents) == 0:
        return np.empty((0, d_child), dtype=np.int32)
    d_parent = parents.shape[1]
    assert d_child == 2 * d_parent
    out = []
    for c in parents:
        ranges = [range(int(c[i]) + 1) for i in range(d_parent)]
        for vals in product(*ranges):
            child = np.empty(d_child, dtype=np.int32)
            for i in range(d_parent):
                child[2*i] = vals[i]
                child[2*i+1] = int(c[i]) - vals[i]
            out.append(child)
    return np.vstack(out) if out else np.empty((0, d_child), dtype=np.int32)


def sweep_one(c_target, full_l0, n_half, m, n_workers=2, parent_sample=4, rng=None):
    """Run F+FN+Q+QN+L at L0 + sample-based L1 for one c_target."""
    if rng is None:
        rng = np.random.default_rng(20260509)
    d0 = 2 * n_half
    print(f'\n=== c_target={c_target} ===', flush=True)

    # ---- L0 ----
    t = time.time()
    sF = prune_F(full_l0, n_half, m, c_target)
    F0 = full_l0[sF]
    print(f'  L0: F-survivors={len(F0)} (of {len(full_l0)})  '
          f'wall_F={time.time()-t:.2f}s', flush=True)
    res0 = run_chain(F0, n_half, m, c_target, with_sp=False, n_workers=n_workers)
    print(f'  L0 chain: F={res0["n_in"]} -> FN={res0["n_FN"]} -> Q={res0["n_Q"]}'
          f' -> QN={res0["n_QN"]} -> L={res0["n_L"]}'
          f'  wall={sum(res0.get(k,0) for k in ("t_FN","t_Q","t_QN","t_L")):.2f}s',
          flush=True)
    n_l0 = res0['n_L']

    parents = res0['L_survivors']

    # ---- L1: sample up to `parent_sample` parents ----
    l1_per_parent = []
    if n_l0 == 0:
        print(f'  L1: skipped (L0=0)', flush=True)
        return {
            'c_target': c_target,
            'L0_survivors': int(n_l0),
            'L1_per_parent': [],
            'L1_total_survivors': 0,
            'L1_avg_survivors': 0.0,
            'parents_sampled': 0,
            'L0_chain': {k:v for k,v in res0.items()
                          if k not in ('L_survivors','SP_survivors')},
        }

    n_parents_sample = min(parent_sample, n_l0)
    if n_l0 > n_parents_sample:
        idx = rng.choice(n_l0, n_parents_sample, replace=False)
        sample_parents = parents[idx]
    else:
        sample_parents = parents

    n_half_child = 2 * n_half  # n_half doubles at next level (d_child = 2 * n_half_child = 4)
    d1 = 2 * d0
    print(f'  L1: d_child={d1}, sampling {len(sample_parents)} of {n_l0} parents',
          flush=True)
    for i, p in enumerate(sample_parents):
        t = time.time()
        children_p = refine_to_d_children(p[None,:], d1)
        sF1 = prune_F(children_p, n_half_child, m, c_target)
        F1p = children_p[sF1]
        res_p = run_chain(F1p, n_half_child, m, c_target,
                          with_sp=False, n_workers=n_workers)
        wall_p = time.time() - t
        rec = {
            'parent_idx': int(i),
            'parent': p.tolist(),
            'n_children': int(len(children_p)),
            'n_F': int(len(F1p)),
            'n_FN': int(res_p['n_FN']),
            'n_Q': int(res_p['n_Q']),
            'n_QN': int(res_p['n_QN']),
            'n_L': int(res_p['n_L']),
            'wall_total_sec': round(wall_p, 2),
        }
        l1_per_parent.append(rec)
        print(f'    parent[{i}]={p.tolist()} children={rec["n_children"]} '
              f'-> F={rec["n_F"]} -> FN={rec["n_FN"]} -> Q={rec["n_Q"]}'
              f' -> QN={rec["n_QN"]} -> L={rec["n_L"]}  wall={wall_p:.2f}s',
              flush=True)

    total_l1 = sum(r['n_L'] for r in l1_per_parent)
    avg_l1 = total_l1 / max(1, len(l1_per_parent))
    print(f'  L1 summary: total={total_l1}, avg_per_parent={avg_l1:.2f}', flush=True)

    return {
        'c_target': c_target,
        'L0_survivors': int(n_l0),
        'L1_per_parent': l1_per_parent,
        'L1_total_survivors': int(total_l1),
        'L1_avg_survivors': float(avg_l1),
        'parents_sampled': int(len(sample_parents)),
        'L0_chain': {k:v for k,v in res0.items()
                      if k not in ('L_survivors','SP_survivors')},
    }


def main():
    n_half = N_HALF
    m = M
    n_workers = N_WORKERS
    d0 = 2 * n_half
    S0 = 4 * n_half * m

    print(f'=== c_target sweep: n_half={n_half}, m={m}, d0={d0}, S0={S0} ===',
          flush=True)
    print(f'   c_targets={C_TARGETS}, n_workers={n_workers}', flush=True)

    # Enumerate canonical comps once.
    print(f'\n--- enumerate canonical comps (d={d0}, S={S0}) ---', flush=True)
    t = time.time()
    full = []
    for batch in generate_canonical_compositions_batched(d0, S0, batch_size=200_000):
        full.append(batch.astype(np.int32, copy=False))
    full = np.vstack(full)
    print(f'  total canonical: {len(full)}  wall={time.time()-t:.2f}s', flush=True)

    # Warm prune_F numba JIT.
    warm = np.zeros((1, d0), dtype=np.int32); warm[0,0] = 2*m
    prune_F(warm, n_half, m, 1.5)

    rng = np.random.default_rng(20260509)
    results = []
    t_total = time.time()
    for c in C_TARGETS:
        r = sweep_one(c, full, n_half, m,
                       n_workers=n_workers,
                       parent_sample=L1_PARENT_SAMPLE,
                       rng=rng)
        # Convergence verdict heuristic.
        l0 = r['L0_survivors']
        avg = r['L1_avg_survivors']
        if l0 == 0:
            r['converging'] = True
            r['note'] = 'L0=0; chain closes immediately'
        elif avg == 0:
            r['converging'] = True
            r['note'] = 'L1 avg per parent = 0; chain converges'
        elif avg < 1.0:
            r['converging'] = True
            r['note'] = f'L1 avg per parent={avg:.2f} < 1; chain converges (geometric decay)'
        elif avg < l0:
            # Heuristic: if L1 per-parent average is less than L0 total, that's necessary
            # but not sufficient. Mark "marginal" if 1 <= avg < l0/4.
            if avg < l0 / 4:
                r['converging'] = True
                r['note'] = f'L1 avg={avg:.2f} < L0/4={l0/4:.2f}; likely converging'
            else:
                r['converging'] = False
                r['note'] = f'L1 avg={avg:.2f} >= L0/4={l0/4:.2f}; marginal/diverging'
        else:
            r['converging'] = False
            r['note'] = f'L1 avg={avg:.2f} >= L0={l0}; diverging'
        print(f'  >>> c={c}: L0={l0}, L1_avg={avg:.2f}, converging={r["converging"]} ({r["note"]})\n',
              flush=True)
        results.append(r)

    summary = {
        'config': {'n_half': n_half, 'm': m, 'd0': d0, 'S0': S0,
                    'c_targets': C_TARGETS, 'n_workers': n_workers,
                    'L1_parent_sample': L1_PARENT_SAMPLE},
        'L0_total_canonical': int(len(full)),
        'results': results,
        'wall_total_sec': round(time.time() - t_total, 2),
    }

    out_path = os.path.join(_HERE, '_smoke_c_target_sweep.json')
    with open(out_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f'\nWrote {out_path}', flush=True)

    # Identify convergence threshold.
    print(f'\n=== TABLE ===', flush=True)
    print(f'{"c_target":>10}  {"L0":>6}  {"L1_avg":>8}  {"converging":>10}', flush=True)
    threshold = None
    for r in results:
        c = r['c_target']
        print(f'{c:>10.3f}  {r["L0_survivors"]:>6d}  {r["L1_avg_survivors"]:>8.2f}'
              f'  {str(r["converging"]):>10s}', flush=True)
        if r['converging'] and threshold is None:
            threshold = c
    print(f'\nThreshold (lowest c that converges): {threshold}', flush=True)


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()
    main()

"""Coarse cascade orchestrator using _coarse_bnb_v2 as the cell-certifier.

For each (d, S, c_target):
  L0: enumerate all canonical compositions of S into d bins.
      For each, call cert_cell from v2.
      Survivors = those NOT certified.
  L_k+1: refine each L_k survivor to children at d_{k+1} = 2·d_k,
         maintaining Σc = S (mass conservation, each parent bin
         splits into two child bins summing to the parent value).
         Each child is then cell-certified.

Termination:
  - All cells certified at some level → proof of C_{1a} ≥ c_target at this S.
  - Survivors at max_level → cascade did not converge in budget.

Heavy logging at each level.

NEVER PALINDROMIC. Canonical enumeration via reversal symmetry only.
"""
from __future__ import annotations
import os, sys, time, logging, json
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
from itertools import product

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _coarse_bnb_v2 import (
    Cell, WindowData, build_all_windows, certify_composition,
    cert_cell,
)
from compositions import generate_canonical_compositions_batched


def _log(msg=""):
    print(msg, flush=True)


def enumerate_children(parent_c: np.ndarray) -> list:
    """All valid children of parent_c at d_child = 2·d_parent.

    Each parent bin c_i splits into (c_{2i}, c_{2i+1}) with c_{2i}+c_{2i+1}=c_i.
    Returns list of np.array of length d_child.
    """
    d_p = len(parent_c)
    # For each parent bin, enumerate splits
    splits = []
    for ci in parent_c:
        ci_int = int(ci)
        splits.append([(a, ci_int - a) for a in range(ci_int + 1)])
    # Cartesian product
    children = []
    for combo in product(*splits):
        child = np.zeros(2 * d_p, dtype=np.int64)
        for i, (a, b) in enumerate(combo):
            child[2 * i] = a
            child[2 * i + 1] = b
        children.append(child)
    return children


def canonical_form(c: np.ndarray) -> tuple:
    """Lex-smaller of (c, reversed c) as a tuple, for deduplication."""
    rc = c[::-1]
    if tuple(c) <= tuple(rc):
        return tuple(c)
    return tuple(rc)


def run_orchestrator(d0: int, S: int, c_target: float, max_levels: int = 4,
                       max_cells_per_level: int = 1_000_000,
                       time_budget_s: float = 600.0,
                       max_bnb_depth: int = 3,
                       sample_only: int = 0):
    """Run cascade starting at d=d0, refining up to max_levels.

    If sample_only > 0, instead of full enumeration at L0, sample that many
    compositions (useful for very large d where full enum is infeasible).
    """
    t_start = time.time()
    _log(f"{'='*72}")
    _log(f"COARSE BnB v2 ORCHESTRATOR")
    _log(f"  d0={d0}  S={S}  c_target={c_target}")
    _log(f"  max_levels={max_levels}  max_cells_per_level={max_cells_per_level:,}")
    _log(f"  time_budget={time_budget_s}s  max_bnb_depth={max_bnb_depth}")
    _log(f"  sample_only={sample_only} (0 = full enumeration)")
    _log(f"{'='*72}")

    d_curr = d0
    S_curr = S
    windows = build_all_windows(d_curr)

    # L0: enumerate
    _log(f"\n--- L0: d={d_curr}, S={S_curr} ---")
    t0 = time.time()
    if sample_only > 0:
        # Sample
        _log(f"  Sampling {sample_only} compositions (no full enumeration)...")
        rng = np.random.default_rng(20260511)
        l0_cells = []
        for _ in range(sample_only):
            u = rng.dirichlet(np.ones(d_curr) * rng.uniform(0.3, 3.0))
            c = np.round(u * S_curr).astype(np.int64)
            c[-1] = S_curr - c[:-1].sum()
            if c.min() < 0:
                continue
            l0_cells.append(c)
        l0_cells = l0_cells[:sample_only]
        _log(f"  Sampled {len(l0_cells)} compositions  [{time.time()-t0:.2f}s]")
    else:
        _log(f"  Enumerating canonical compositions...")
        l0_cells = []
        seen = set()
        for batch in generate_canonical_compositions_batched(
                d_curr, S_curr, batch_size=200_000):
            for row in batch:
                c = np.asarray(row, dtype=np.int64)
                # generate_canonical already returns canonical; just dedupe defensively
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    l0_cells.append(c)
                if len(l0_cells) >= max_cells_per_level:
                    break
            if len(l0_cells) >= max_cells_per_level:
                break
        _log(f"  Enumerated {len(l0_cells):,} canonical compositions  "
             f"[{time.time()-t0:.2f}s]")
        if len(l0_cells) >= max_cells_per_level:
            _log(f"  WARNING: hit max_cells cap of {max_cells_per_level:,}")

    # Process L0
    _log(f"\n  Certifying {len(l0_cells):,} L0 cells via v2 (F→Q→L→L_joint→split)...")
    t1 = time.time()
    survivors = []
    counts = {'F': 0, 'Q': 0, 'L': 0, 'L_joint': 0, 'split': 0, 'open': 0}
    for k, c in enumerate(l0_cells):
        if time.time() - t_start > time_budget_s:
            _log(f"  TIME BUDGET EXCEEDED at L0 cell {k}/{len(l0_cells)}")
            break
        if k % max(1, len(l0_cells) // 20) == 0 and k > 0:
            elapsed = time.time() - t1
            rate = k / elapsed if elapsed > 0 else 0
            _log(f"    L0 {k:>6}/{len(l0_cells):,}  ({100*k/len(l0_cells):.1f}%) "
                 f"  closed: F={counts['F']} Q={counts['Q']} L={counts['L']} "
                 f"Lj={counts['L_joint']} split={counts['split']}  open={counts['open']} "
                 f"  rate={rate:.1f}/s")
        c_float = c.astype(np.float64)
        try:
            r = cert_cell(Cell.from_integer_composition(c_float, S_curr),
                           windows, c_target, max_depth=max_bnb_depth,
                           solver='MOSEK')
        except Exception as e:
            _log(f"    L0 cell {k} ERROR: {e}")
            survivors.append(c)
            counts['open'] += 1
            continue
        if r.certified:
            counts[r.tier_used] = counts.get(r.tier_used, 0) + 1
        else:
            survivors.append(c)
            counts['open'] += 1
    t_L0 = time.time() - t1
    _log(f"\n  L0 RESULT: closed {sum(counts[k] for k in ('F','Q','L','L_joint','split'))} "
         f"  open={len(survivors)}  [{t_L0:.1f}s]")
    _log(f"    by tier: F={counts['F']} Q={counts['Q']} L={counts['L']} "
         f"L_joint={counts['L_joint']} split={counts['split']}")

    if not survivors:
        _log(f"\n*** L0 TERMINATES at d={d_curr}, S={S_curr}, c={c_target} ***")
        _log(f"    All cells certified. C_{{1a}} >= {c_target} via this cascade.")
        return {'status': 'L0_terminate', 'level': 0, 'd_final': d_curr,
                 'total_time': time.time() - t_start,
                 'L0_counts': counts}

    # L1+: refine and process
    for level in range(1, max_levels + 1):
        d_curr = 2 * d_curr
        windows = build_all_windows(d_curr)
        _log(f"\n--- L{level}: d={d_curr} (refining {len(survivors):,} L{level-1} parents) ---")
        if time.time() - t_start > time_budget_s:
            _log(f"  TIME BUDGET EXCEEDED before L{level}")
            break
        # Generate children
        t_gen = time.time()
        all_children = []
        for parent in survivors:
            ch = enumerate_children(parent)
            all_children.extend(ch)
            if len(all_children) > max_cells_per_level:
                _log(f"  L{level} CHILD COUNT EXCEEDED {max_cells_per_level:,} "
                     f"after {len(all_children)} children from "
                     f"{survivors.index(parent)+1}/{len(survivors)} parents")
                break
        # Deduplicate via canonical
        seen = set()
        unique_children = []
        for c in all_children:
            key = canonical_form(c)
            if key not in seen:
                seen.add(key)
                unique_children.append(c)
        _log(f"  L{level} children: {len(all_children):,} raw, "
             f"{len(unique_children):,} canonical  [{time.time()-t_gen:.1f}s]")

        # Certify
        t1 = time.time()
        survivors = []
        counts = {'F': 0, 'Q': 0, 'L': 0, 'L_joint': 0, 'split': 0, 'open': 0}
        for k, c in enumerate(unique_children):
            if time.time() - t_start > time_budget_s:
                _log(f"  TIME BUDGET EXCEEDED at L{level} cell {k}/{len(unique_children)}")
                break
            if k % max(1, len(unique_children) // 20) == 0 and k > 0:
                _log(f"    L{level} {k:>7}/{len(unique_children):,} "
                     f"closed={sum(counts[t] for t in ('F','Q','L','L_joint','split'))} "
                     f"open={counts['open']}")
            try:
                r = cert_cell(Cell.from_integer_composition(c.astype(np.float64), S_curr),
                               windows, c_target, max_depth=max_bnb_depth,
                               solver='MOSEK')
            except Exception as e:
                survivors.append(c)
                counts['open'] += 1
                continue
            if r.certified:
                counts[r.tier_used] = counts.get(r.tier_used, 0) + 1
            else:
                survivors.append(c)
                counts['open'] += 1
        _log(f"\n  L{level} RESULT: closed {sum(counts[t] for t in ('F','Q','L','L_joint','split'))} "
             f"  open={len(survivors)}  [{time.time()-t1:.1f}s]")
        _log(f"    by tier: F={counts['F']} Q={counts['Q']} L={counts['L']} "
             f"L_joint={counts['L_joint']} split={counts['split']}")
        if not survivors:
            _log(f"\n*** L{level} TERMINATES at d={d_curr}, S={S_curr}, c={c_target} ***")
            _log(f"    Cascade converged. C_{{1a}} >= {c_target} via {level+1} levels.")
            return {'status': f'L{level}_terminate', 'level': level,
                     'd_final': d_curr, 'total_time': time.time() - t_start}

    _log(f"\n--- FINAL ---")
    _log(f"  Cascade did NOT terminate. Survivors at L{max_levels}: {len(survivors):,}")
    _log(f"  Total time: {time.time() - t_start:.1f}s")
    return {'status': 'NOT_CONVERGED', 'level': max_levels,
             'survivors_remaining': len(survivors),
             'total_time': time.time() - t_start}


# =====================================================================
# Driver
# =====================================================================

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d0', type=int, default=2)
    ap.add_argument('--S', type=int, default=8)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--max_levels', type=int, default=3)
    ap.add_argument('--max_cells', type=int, default=200_000)
    ap.add_argument('--time_budget', type=float, default=300.0)
    ap.add_argument('--max_bnb_depth', type=int, default=3)
    ap.add_argument('--sample', type=int, default=0,
                     help='If > 0, sample this many L0 compositions instead of full enum')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    result = run_orchestrator(
        d0=args.d0, S=args.S, c_target=args.c_target,
        max_levels=args.max_levels,
        max_cells_per_level=args.max_cells,
        time_budget_s=args.time_budget,
        max_bnb_depth=args.max_bnb_depth,
        sample_only=args.sample,
    )
    if args.out:
        with open(args.out, 'w') as fp:
            json.dump(result, fp, indent=2)
    _log(f"\nResult: {result}")

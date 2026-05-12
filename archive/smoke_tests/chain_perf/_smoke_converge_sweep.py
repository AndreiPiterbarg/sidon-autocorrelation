"""Smoke convergence sweep for the FINE cascade.

Identifies the smallest (n_half, m) configuration where the cascade
ACTUALLY CONVERGES at c_target=1.281, by exhaustively running the
L0 chain (F+FN+Q+QN+L) on every config in the sweep, then probing
a tiny L1 sample on configs with non-zero L0 survivors to determine
expansion behavior.

Sweep: m in {10,20,30,50,100} cross n_half in {1,2}.
Wall budget < 30 min total.  n_workers capped at 2 (Windows multiproc safety).

Output: prints a table at the end with
    (n_half, m, d0, L0_total, L0_F+FN+Q+QN, L0_L, L1_avg_F, L1_avg_L,
     est_L1_total, verdict)
and a recommendation.
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, ROOT)

from pruning import correction
from run_cascade import process_parent_fused, run_level0
from post_filters import (apply_FN_filter_parallel,
                          apply_Q_filter_parallel,
                          apply_QN_filter_parallel,
                          apply_L_filter_parallel)


C_TARGET = 1.281
N_WORKERS = 2  # Windows safety
L1_SAMPLE_PARENTS = 2  # tiny — just enough to estimate expansion


def n_full_compositions(d, S):
    return comb(S + d - 1, d - 1)


def run_l0_chain(n_half, m, c_target, n_workers=N_WORKERS):
    """Run the full L0 chain F+FN+Q+QN+L.  Returns (n_F_chain,n_L,survivors,
    walls, vacuous, l0_total)."""
    d0 = 2 * n_half
    S0 = 4 * n_half * m
    l0_total = n_full_compositions(d0, S0)
    corr = correction(m, n_half)
    if c_target + corr >= 1.5029:
        return {
            'n_F_chain': None, 'n_L': None, 'survivors': None,
            'wall_l0_FFNQ_sec': None, 'wall_L_sec': None,
            'vacuous': True, 'l0_total': l0_total,
            'corr': corr, 'd0': d0,
        }

    t0 = time.time()
    r0 = run_level0(n_half, m, c_target, verbose=False, use_F=True,
                    use_Q=True, use_L=False)
    wall_l0_FFNQ = time.time() - t0
    survivors = r0['survivors']
    n_FFNQ = int(r0['n_survivors'])

    # Apply L (Shor SDP via direct-MOSEK) sequentially over what survived.
    n_L = n_FFNQ
    wall_L = 0.0
    if n_FFNQ > 0:
        tL = time.time()
        try:
            survivors = apply_L_filter_parallel(
                survivors, n_half, m, c_target,
                solver='MOSEK', n_workers=n_workers)
            n_L = int(len(survivors))
        except Exception as e:
            print(f"    [L0 L-filter EXC: {e}]", flush=True)
        wall_L = time.time() - tL

    return {
        'n_F_chain': n_FFNQ, 'n_L': n_L, 'survivors': survivors,
        'wall_l0_FFNQ_sec': wall_l0_FFNQ,
        'wall_L_sec': wall_L,
        'vacuous': False, 'l0_total': l0_total,
        'corr': corr, 'd0': d0,
    }


def probe_l1_expansion(survivors, n_half, m, c_target,
                       sample_n=L1_SAMPLE_PARENTS, n_workers=N_WORKERS,
                       per_parent_budget=120):
    """For configs with non-zero L0 survivors, sample sample_n parents,
    expand once via process_parent_fused → F → FN → Q → QN → L.  Return
    average per-parent counts.

    Returns dict with:
        n_l0: parent count
        sample_size
        avg_children, avg_F, avg_FN, avg_Q, avg_QN, avg_L
        est_total_children, est_total_F, ..., est_total_L
        wall_total_sec
    """
    n_l0 = int(len(survivors))
    if n_l0 == 0:
        return {'n_l0': 0, 'sample_size': 0,
                'avg_children': 0, 'avg_F': 0, 'avg_FN': 0,
                'avg_Q': 0, 'avg_QN': 0, 'avg_L': 0,
                'est_total_children': 0,
                'est_total_F': 0, 'est_total_FN': 0,
                'est_total_Q': 0, 'est_total_QN': 0,
                'est_total_L': 0,
                'wall_total_sec': 0.0}

    rng = np.random.default_rng(seed=42)
    sample_size = min(sample_n, n_l0)
    if n_l0 > sample_size:
        idx = rng.choice(n_l0, sample_size, replace=False)
        sample = survivors[idx]
    else:
        sample = survivors

    d_parent = int(survivors.shape[1])
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    per_parent = []
    t0 = time.time()
    for i, parent in enumerate(sample):
        elapsed = time.time() - t0
        if elapsed > per_parent_budget * sample_size:
            print(f"    [L1 budget exhausted after {i}/{sample_size} parents]",
                  flush=True)
            break
        tp = time.time()
        try:
            surv_F, n_children = process_parent_fused(
                parent, m, c_target, n_half_child,
                use_flat_threshold=False, use_F=True, use_Q=False,
                skip_sdp_cert=True)
        except Exception as e:
            print(f"    [parent {i} F EXC: {e}]", flush=True)
            continue
        n_F = int(len(surv_F))
        try:
            surv_FN = apply_FN_filter_parallel(surv_F, n_half_child, m, c_target)
        except Exception:
            surv_FN = surv_F
        n_FN = int(len(surv_FN))
        try:
            surv_Q = apply_Q_filter_parallel(
                surv_FN, n_half_child, m, c_target, n_workers=n_workers)
        except Exception:
            surv_Q = surv_FN
        n_Q = int(len(surv_Q))
        try:
            surv_QN = apply_QN_filter_parallel(
                surv_Q, n_half_child, m, c_target, n_workers=n_workers)
        except Exception:
            surv_QN = surv_Q
        n_QN = int(len(surv_QN))
        try:
            surv_L = apply_L_filter_parallel(
                surv_QN, n_half_child, m, c_target,
                solver='MOSEK', n_workers=n_workers)
        except Exception as e:
            print(f"    [parent {i} L EXC: {e}]", flush=True)
            surv_L = surv_QN
        n_L = int(len(surv_L))
        wall_p = time.time() - tp
        per_parent.append({
            'children': int(n_children),
            'F': n_F, 'FN': n_FN, 'Q': n_Q, 'QN': n_QN, 'L': n_L,
            'wall_sec': round(wall_p, 2),
        })
        print(f"      L1 parent[{i+1}/{sample_size}]: children={n_children:,}  "
              f"F={n_F:,} -> FN={n_FN:,} -> Q={n_Q:,} -> QN={n_QN:,} -> "
              f"L={n_L:,}  wall={wall_p:.1f}s",
              flush=True)
        if wall_p > per_parent_budget:
            print(f"      [parent wall {wall_p:.1f}s > budget {per_parent_budget}s, "
                  f"stopping further L1 parents]", flush=True)
            break
    wall_total = time.time() - t0

    n_done = len(per_parent)
    if n_done == 0:
        return {'n_l0': n_l0, 'sample_size': 0,
                'avg_children': 0, 'avg_F': 0, 'avg_FN': 0,
                'avg_Q': 0, 'avg_QN': 0, 'avg_L': 0,
                'est_total_children': 0,
                'est_total_F': 0, 'est_total_FN': 0,
                'est_total_Q': 0, 'est_total_QN': 0,
                'est_total_L': 0,
                'wall_total_sec': round(wall_total, 2)}

    avg_children = sum(p['children'] for p in per_parent) / n_done
    avg_F = sum(p['F'] for p in per_parent) / n_done
    avg_FN = sum(p['FN'] for p in per_parent) / n_done
    avg_Q = sum(p['Q'] for p in per_parent) / n_done
    avg_QN = sum(p['QN'] for p in per_parent) / n_done
    avg_L = sum(p['L'] for p in per_parent) / n_done

    return {
        'n_l0': n_l0, 'sample_size': n_done,
        'avg_children': avg_children, 'avg_F': avg_F, 'avg_FN': avg_FN,
        'avg_Q': avg_Q, 'avg_QN': avg_QN, 'avg_L': avg_L,
        'est_total_children': avg_children * n_l0,
        'est_total_F': int(round(avg_F * n_l0)),
        'est_total_FN': int(round(avg_FN * n_l0)),
        'est_total_Q': int(round(avg_Q * n_l0)),
        'est_total_QN': int(round(avg_QN * n_l0)),
        'est_total_L': int(round(avg_L * n_l0)),
        'wall_total_sec': round(wall_total, 2),
    }


def verdict_for(L0_L, L1_est_L, vacuous):
    if vacuous:
        return 'VACUOUS'
    if L0_L == 0:
        return 'CLOSED_AT_L0'
    if L1_est_L == 0:
        return 'CLOSED_AT_L1_PROJECTED'
    if L1_est_L < L0_L:
        return 'POSSIBLY_CONVERGENT_L1<L0'
    return 'NONCONVERGENT_L1>=L0'


def main():
    sweep = []
    for n_half in (1, 2):
        for m in (10, 20, 30, 50, 100):
            sweep.append((n_half, m))

    out_rows = []
    t_global = time.time()
    print(f"\nSmoke convergence sweep at c_target={C_TARGET}, "
          f"n_workers={N_WORKERS}", flush=True)
    print("=" * 80, flush=True)

    for (n_half, m) in sweep:
        elapsed_global = time.time() - t_global
        if elapsed_global > 1700:  # ~28 min
            print(f"\nGlobal wall budget exhausted at {elapsed_global:.0f}s; "
                  f"skipping remaining configs.", flush=True)
            break
        d0 = 2 * n_half
        S0 = 4 * n_half * m
        l0_total = n_full_compositions(d0, S0)
        print(f"\n--- n_half={n_half}, m={m}, d0={d0}, "
              f"L0_total_compositions={l0_total:,} ---", flush=True)

        # Skip configs with absurd L0 enumeration cost.  At n_half=2, m=100
        # we'd have C(401, 3) ~ 10.8M compositions: tractable.  At n_half=2,
        # m=300 it would explode but we don't go that far in this sweep.
        if l0_total > 50_000_000:
            print(f"    [L0 too big to enumerate in smoke window: "
                  f"{l0_total:,}]", flush=True)
            out_rows.append({
                'n_half': n_half, 'm': m, 'd0': d0,
                'l0_total': l0_total,
                'wall_l0_sec': None,
                'l0_FFNQ': None,
                'l0_L': None,
                'l1_est_L': None,
                'verdict': 'L0_TOO_BIG',
                'wall_total_sec': 0.0,
            })
            continue

        t0_cfg = time.time()
        l0_res = run_l0_chain(n_half, m, C_TARGET)
        if l0_res['vacuous']:
            print(f"    VACUOUS: c_target+corr={C_TARGET + l0_res['corr']:.4f} "
                  f">= 1.5029", flush=True)
            out_rows.append({
                'n_half': n_half, 'm': m, 'd0': d0,
                'l0_total': l0_total,
                'wall_l0_sec': 0.0,
                'l0_FFNQ': 0, 'l0_L': 0,
                'l1_est_L': 0,
                'verdict': 'VACUOUS',
                'wall_total_sec': 0.0,
            })
            continue

        n_FFNQ = l0_res['n_F_chain']
        n_L = l0_res['n_L']
        wall_l0 = l0_res['wall_l0_FFNQ_sec'] + l0_res['wall_L_sec']
        print(f"    L0: F+FN+Q={n_FFNQ:,} -> L={n_L:,}  in {wall_l0:.1f}s "
              f"(F+FN+Q={l0_res['wall_l0_FFNQ_sec']:.1f}s, "
              f"L={l0_res['wall_L_sec']:.1f}s)", flush=True)

        l1_est_L = None
        l1_wall = 0.0
        if n_L == 0:
            verdict = 'CLOSED_AT_L0'
            l1_est_L = 0
            print(f"    => verdict: CLOSED_AT_L0", flush=True)
        else:
            # L1 probe.  For huge n_L, sample is small (still 2 parents).
            print(f"    L1 probe ({L1_SAMPLE_PARENTS} parents) ...", flush=True)
            try:
                l1 = probe_l1_expansion(
                    l0_res['survivors'], n_half, m, C_TARGET,
                    sample_n=L1_SAMPLE_PARENTS,
                    n_workers=N_WORKERS,
                    per_parent_budget=180,  # 3 min per parent
                )
                l1_est_L = l1['est_total_L']
                l1_wall = l1['wall_total_sec']
                avg_F = l1.get('avg_F', 0)
                avg_L = l1.get('avg_L', 0)
                print(f"    L1: avg_F={avg_F:.1f}, avg_L={avg_L:.1f}, "
                      f"est_total_L={l1_est_L:,}  in {l1_wall:.1f}s",
                      flush=True)
                verdict = verdict_for(n_L, l1_est_L, False)
                print(f"    => verdict: {verdict}", flush=True)
            except Exception as e:
                print(f"    L1 probe EXC: {e}", flush=True)
                l1 = None
                verdict = f'L1_ERROR'

        wall_total = time.time() - t0_cfg
        out_rows.append({
            'n_half': n_half, 'm': m, 'd0': d0,
            'l0_total': l0_total,
            'wall_l0_sec': round(wall_l0, 2),
            'l0_FFNQ': n_FFNQ,
            'l0_L': n_L,
            'l1_est_L': l1_est_L,
            'wall_l1_sec': round(l1_wall, 2),
            'verdict': verdict,
            'wall_total_sec': round(wall_total, 2),
        })

    # ---- Final table ----
    print("\n" + "=" * 100, flush=True)
    print("SMOKE CONVERGENCE SWEEP — RESULTS at c_target=1.281", flush=True)
    print("=" * 100, flush=True)
    hdr = (f"{'n_half':>6} {'m':>4} {'d0':>3} {'L0_total':>14} "
           f"{'L0_FFNQ':>10} {'L0_L':>10} {'L1_estL':>14} "
           f"{'wall':>7}  verdict")
    print(hdr, flush=True)
    for row in out_rows:
        l1_str = (f"{row['l1_est_L']:,}" if row.get('l1_est_L') is not None
                  else 'n/a')
        l0_FFNQ_str = (f"{row['l0_FFNQ']:,}"
                       if row['l0_FFNQ'] is not None else 'n/a')
        l0_L_str = (f"{row['l0_L']:,}" if row['l0_L'] is not None else 'n/a')
        wall_str = (f"{row.get('wall_total_sec', 0):.1f}s")
        print(f"{row['n_half']:>6} {row['m']:>4} {row['d0']:>3} "
              f"{row['l0_total']:>14,} {l0_FFNQ_str:>10} {l0_L_str:>10} "
              f"{l1_str:>14} {wall_str:>7}  {row['verdict']}",
              flush=True)

    # Recommendation
    print("\n" + "=" * 100, flush=True)
    closed_at_l0 = [r for r in out_rows if r['verdict'] == 'CLOSED_AT_L0']
    closed_l1 = [r for r in out_rows if r['verdict'] == 'CLOSED_AT_L1_PROJECTED']
    converging = [r for r in out_rows if r['verdict'] == 'POSSIBLY_CONVERGENT_L1<L0']
    nonconv = [r for r in out_rows if r['verdict'] == 'NONCONVERGENT_L1>=L0']

    print(f"L0-closed configs: {len(closed_at_l0)}", flush=True)
    print(f"L1-closed (projected): {len(closed_l1)}", flush=True)
    print(f"Converging (L1 < L0): {len(converging)}", flush=True)
    print(f"Non-convergent: {len(nonconv)}", flush=True)

    chosen = None
    rationale = ''
    if closed_at_l0:
        # Smallest m within smallest n_half
        chosen = min(closed_at_l0, key=lambda r: (r['n_half'], r['m']))
        rationale = 'CLOSED_AT_L0 (cascade trivially converges)'
    elif closed_l1:
        chosen = min(closed_l1, key=lambda r: (r['n_half'], r['m']))
        rationale = 'L1 projected to 0 survivors'
    elif converging:
        chosen = min(converging,
                     key=lambda r: (r['l1_est_L'] / max(1, r['l0_L']),
                                    r['n_half'], r['m']))
        rationale = 'L1 < L0 in projection (provisional convergence)'
    else:
        rationale = 'No config in this sweep demonstrably converges.'

    print("\nRecommendation:", flush=True)
    if chosen:
        print(f"   n_half={chosen['n_half']}, m={chosen['m']}, "
              f"d0={chosen['d0']}", flush=True)
        print(f"   L0_total={chosen['l0_total']:,}  "
              f"L0_L={chosen['l0_L']:,}  "
              f"L1_estL={chosen.get('l1_est_L')}  "
              f"wall={chosen.get('wall_total_sec', 0):.1f}s", flush=True)
        print(f"   ({rationale})", flush=True)
    else:
        print(f"   {rationale}", flush=True)

    out_path = os.path.join(HERE, '_smoke_converge_sweep_results.json')
    payload = {
        'c_target': C_TARGET,
        'n_workers': N_WORKERS,
        'l1_sample_parents': L1_SAMPLE_PARENTS,
        'rows': out_rows,
        'recommendation': {
            'chosen': chosen,
            'rationale': rationale,
        },
        'wall_total_sec': round(time.time() - t_global, 2),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  results saved -> {out_path}", flush=True)
    print(f"  total wall: {time.time() - t_global:.1f}s", flush=True)


if __name__ == '__main__':
    main()

"""Quick cascade feasibility probe.

Samples ~100 parents per level and estimates expansion factor.
Stops when expansion is clearly futile or survivors hit zero.

Usage:
    python tests/benchmark_sweep.py
    python tests/benchmark_sweep.py --m 20 --n_half 2 --c_target 1.40
    python tests/benchmark_sweep.py --sample 100 --use_flat_threshold
    python tests/benchmark_sweep.py --use_F
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import process_parent_fused, run_level0

C_UPPER = 1.5029


LEVEL_TIME_BUDGET_SEC = 600.0  # 10 minutes per cascade level (sample wall)

# Empirical Gray-code throughput on AMD EPYC 9354 (single thread, F enabled).
# At d_child = 8 we measured 24-48 M children/s; pick a conservative 30 M/s
# baseline.  Cost grows ~ d² (window scan dominates).
BASE_RATE_D8 = 30_000_000  # children/s at d_child = 8


def expected_rate(d_child):
    """Estimated children/s at given d_child (single-thread, F enabled)."""
    return BASE_RATE_D8 * (8.0 / max(8, d_child)) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--c_target', type=float, default=1.40)
    ap.add_argument('--sample', type=int, default=100,
                    help='parents to sample per level')
    ap.add_argument('--use_flat_threshold', action='store_true')
    ap.add_argument('--use_F', action='store_true',
                    help='Use variant F pruning (LP-tight Δ_BB).  Sound, '
                         '25-65%% additional pruning over W-refined.  '
                         'Mutually exclusive with --use_flat_threshold.')
    ap.add_argument('--level_time_sec', type=float, default=LEVEL_TIME_BUDGET_SEC,
                    help='Wall-clock budget per cascade level (the SAMPLE'
                         ' loop).  When exceeded, break and move to next '
                         'level with whatever survivors we have.  '
                         'Default %(default)s seconds (10 min).')
    ap.add_argument('--with_sdp', action='store_true',
                    help='Enable the CVXPY SDP parent cert.  Default is '
                         'SKIPPED (it costs 50-500ms per parent and clears '
                         '<1%% in practice).  Theorem-1 + LP cert always run.')
    args = ap.parse_args()

    if args.use_F and args.use_flat_threshold:
        ap.error('--use_F and --use_flat_threshold are mutually exclusive.')

    m, n_half, c_target = args.m, args.n_half, args.c_target
    sample_n = args.sample
    flat = args.use_flat_threshold
    use_F = args.use_F

    # Vacuity check
    corr = correction(m, n_half)
    if c_target + corr >= C_UPPER:
        print(f"VACUOUS: c_target={c_target} + correction={corr:.4f} "
              f"= {c_target+corr:.4f} >= {C_UPPER}")
        return

    d0 = 2 * n_half
    S0 = 4 * n_half * m
    n_compositions = count_compositions(d0, S0)
    print(f"Config: m={m}, n_half={n_half}, c_target={c_target}, "
          f"flat={flat}, use_F={use_F}")
    print(f"L0: d={d0}, S={S0}, compositions={n_compositions:,}")
    print(f"    correction={corr:.6f}, threshold={c_target+corr:.6f}")
    print()

    # --- L0: run fully ---
    t0 = time.time()
    result = run_level0(n_half, m, c_target, verbose=True,
                        use_flat_threshold=flat, use_F=use_F)
    survivors = result['survivors']
    n_surv = result['n_survivors']
    print(f"\nL0 done: {n_surv:,} survivors in {time.time()-t0:.1f}s")

    if n_surv == 0:
        print("PROVEN at L0!")
        return

    # --- Cascade levels ---
    level = 1
    while True:
        d_parent = survivors.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2
        n_parents = len(survivors)

        # Sample
        if n_parents > sample_n:
            idx = np.random.default_rng().choice(n_parents, sample_n,
                                                    replace=False)
            sample = survivors[idx]
        else:
            sample = survivors
        sample_n_actual = len(sample)

        # Pre-check: estimate children per parent from first few samples
        from pruning import correction as _corr
        _c = _corr(m, n_half_child)
        _thresh = c_target + _c + 1e-9
        _x_cap = int(math.floor(m * math.sqrt(4 * d_child * _thresh)))
        _x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
        _x_cap = min(_x_cap, _x_cap_cs)
        _B = sample.astype(np.int64)
        _lo = np.maximum(0, 2 * _B - _x_cap)
        _hi = np.minimum(2 * _B, _x_cap)
        _eff = np.maximum(_hi - _lo + 1, 0)
        _counts = np.prod(_eff, axis=1)
        _median_cpp = int(np.median(_counts))
        _est_total = float(_median_cpp) * n_parents

        print(f"\nL{level}: d_parent={d_parent} -> d_child={d_child}, "
              f"{n_parents:,} parents, sampling {sample_n_actual}")
        print(f"    x_cap={_x_cap}, median children/parent={_median_cpp:,}")
        print(f"    estimated total children: {_est_total:.2e}")
        print(f"    per-level wall budget: {args.level_time_sec:.0f}s "
              f"(processes parents until budget exhausted)")

        BUDGET = 160e12
        if _est_total > BUDGET:
            print(f"\n*** EXCEEDS BUDGET before processing: "
                  f"{_est_total:.2e} > {BUDGET:.0e} "
                  f"({_est_total/BUDGET:.0f}x over) ***")
            return
        print(f"    budget usage: {_est_total/BUDGET*100:.1f}%")

        # Sort by children count ascending: process the cheap ones first
        # so we maximize the # of completed parents within the time budget.
        next_level_surv = []
        _order = np.argsort(_counts)
        sample = sample[_order]
        _counts_sorted = _counts[_order]
        sample_n_actual = len(sample)
        print(f"    {sample_n_actual} parents queued (sorted by children, "
              f"lightest first)", flush=True)

        total_children_sampled = 0
        total_survivors_sampled = 0
        next_level_surv = []
        t0 = time.time()
        n_completed = 0
        budget_hit = False

        rate_at_d = expected_rate(d_child)
        for i, parent in enumerate(sample):
            elapsed_so_far = time.time() - t0
            remaining = args.level_time_sec - elapsed_so_far

            # Predictive skip: estimate how long this parent would take
            # from its children count and the empirical rate.  If it
            # would exceed remaining budget, skip it AND all subsequent
            # ones (sample is sorted ascending by children count).
            ch_predict = int(_counts_sorted[i])
            t_predict = ch_predict / max(1.0, rate_at_d)

            if n_completed > 0 and (elapsed_so_far > args.level_time_sec
                                      or t_predict > remaining):
                budget_hit = True
                n_skipped_remaining = sample_n_actual - i
                print(f"    *** TIME BUDGET HIT: completed "
                      f"{n_completed}/{sample_n_actual} parents in "
                      f"{elapsed_so_far:.0f}s.  Next parent has "
                      f"{ch_predict:,} children "
                      f"(est ~{t_predict:.0f}s, remaining "
                      f"~{remaining:.0f}s).  Skipping {n_skipped_remaining} "
                      f"remaining parents, moving to next level. ***",
                      flush=True)
                break

            t_p = time.time()
            print(f"    [{i+1}/{sample_n_actual}] "
                  f"starting parent ({ch_predict:,} children, "
                  f"est ~{t_predict:.1f}s, "
                  f"budget remaining {remaining:.0f}s)...", flush=True)
            surv_i, n_children_i = process_parent_fused(
                parent, m, c_target, n_half_child,
                use_flat_threshold=flat, use_F=use_F,
                skip_sdp_cert=not args.with_sdp)
            t_p_elapsed = time.time() - t_p
            total_children_sampled += n_children_i
            n_surv_i = len(surv_i)
            total_survivors_sampled += n_surv_i
            n_completed += 1
            if n_surv_i > 0 and sum(len(s) for s in next_level_surv) < sample_n:
                next_level_surv.append(surv_i)
            print(f"    [{i+1}/{sample_n_actual}] "
                  f"{n_children_i:,} children, {n_surv_i:,} surv  "
                  f"[{t_p_elapsed:.1f}s, total {elapsed_so_far+t_p_elapsed:.0f}s/{args.level_time_sec:.0f}s]",
                  flush=True)

        elapsed = time.time() - t0
        if n_completed == 0:
            print(f"  L{level}: no parents completed, stopping")
            return
        avg_children = total_children_sampled / n_completed
        avg_survivors = total_survivors_sampled / n_completed
        expansion = avg_survivors

        budget_str = ' [TIME-BUDGET CUT]' if budget_hit else ''
        print(f"  L{level} summary: {elapsed:.1f}s "
              f"({avg_children/max(1, elapsed/n_completed)/1e6:.0f}M children/s, "
              f"{n_completed}/{sample_n_actual} parents){budget_str}")
        print(f"    avg children/parent:  {avg_children:,.0f}")
        print(f"    avg survivors/parent: {avg_survivors:,.1f}")
        print(f"    expansion factor:     {expansion:.2f}x")

        est_total_survivors = int(expansion * n_parents)
        est_total_children = int(avg_children * n_parents)
        print(f"    estimated total survivors: {est_total_survivors:,}")
        print(f"    estimated total children:  {est_total_children:,}")

        if total_survivors_sampled == 0:
            print(f"\n*** ALL PRUNED at L{level}! Cascade converges. ***")
            return

        BUDGET = 160e12
        if est_total_children > BUDGET:
            print(f"\n*** EXCEEDS BUDGET: {est_total_children:.2e} total children "
                  f"at L{level} > {BUDGET:.0e} budget. "
                  f"({est_total_children/BUDGET:.0f}x over) ***")
            return

        print(f"    budget usage: {est_total_children/BUDGET*100:.1f}% "
              f"of {BUDGET:.0e}")

        survivors = np.vstack(next_level_surv)
        if len(survivors) > sample_n:
            survivors = survivors[:sample_n]

        level += 1


if __name__ == '__main__':
    main()

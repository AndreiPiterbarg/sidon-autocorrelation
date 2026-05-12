"""Estimate full cascade runtime for a given config.

Runs L0 fully, samples L1 parents, measures throughput, projects total time.
Designed to run on the target cloud CPU to get accurate timing.

Usage:
    python tests/estimate_runtime.py
    python tests/estimate_runtime.py --c_target 1.40 --cores 64
"""
import argparse
import sys
import os
import time
import math
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused


def children_counts(parents, m, d_child, c_target):
    n_half_child = d_child // 2
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    xc = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    xc_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    xc = min(xc, xc_cs)
    B = parents.astype(np.int64)
    lo = np.maximum(0, 2 * B - xc)
    hi = np.minimum(2 * B, xc)
    eff = np.maximum(hi - lo + 1, 0)
    return np.prod(eff, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--c_target', type=float, default=1.35)
    ap.add_argument('--cores', type=int, default=64)
    ap.add_argument('--l1_sample', type=int, default=50,
                    help='L1 parents to process for timing')
    ap.add_argument('--l2_sample', type=int, default=20,
                    help='L2 parents to process (lightest)')
    ap.add_argument('--l2_max_children', type=int, default=2_000_000_000,
                    help='Max children for an L2 parent to be processable')
    args = ap.parse_args()

    m, n_half, c_target = args.m, args.n_half, args.c_target
    cores = args.cores

    print(f"=== Runtime Estimate ===")
    print(f"Config: m={m}, n_half={n_half}, c_target={c_target}")
    print(f"Target: {cores} CPU cores")
    print()

    # --- L0 ---
    t0 = time.time()
    l0 = run_level0(n_half, m, c_target, verbose=False)
    l0_time = time.time() - t0
    l0_surv = l0['survivors']
    n0 = len(l0_surv)
    print(f"L0: {n0:,} survivors in {l0_time:.1f}s", flush=True)

    # --- L1: sample parents, measure throughput ---
    n_l1 = min(args.l1_sample, n0)
    rng = np.random.RandomState(42)
    idx = rng.choice(n0, n_l1, replace=False)

    l1_all = []
    l1_total_children = 0
    l1_total_surv = 0
    l1_total_time = 0.0
    nhc = 2 * n_half

    print(f"\nL1: processing {n_l1} parents...", flush=True)
    for i, ix in enumerate(idx):
        t1 = time.time()
        s, nc = process_parent_fused(l0_surv[ix], m, c_target, nhc)
        dt = time.time() - t1
        l1_total_children += nc
        l1_total_surv += len(s)
        l1_total_time += dt
        if len(s) > 0:
            l1_all.append(s)
        if (i + 1) % 10 == 0:
            rate = l1_total_children / l1_total_time
            print(f"  [{i+1}/{n_l1}] {rate/1e6:.1f}M children/sec, "
                  f"avg surv/parent={l1_total_surv/(i+1):,.0f}", flush=True)

    l1_rate = l1_total_children / l1_total_time  # children/sec on 1 core
    l1_avg_children = l1_total_children / n_l1
    l1_avg_surv = l1_total_surv / n_l1
    l1_est_total_children = l1_avg_children * n0
    l1_est_total_surv = l1_avg_surv * n0
    l1_est_time = l1_est_total_children / (l1_rate * cores)

    print(f"\nL1 measured:")
    print(f"  Throughput: {l1_rate/1e6:.2f}M children/sec (1 core)")
    print(f"  Avg children/parent: {l1_avg_children:,.0f}")
    print(f"  Avg survivors/parent: {l1_avg_surv:,.0f}")
    print(f"  Expansion: {l1_avg_surv:,.0f}x")
    print(f"L1 projected full run:")
    print(f"  Total children: {l1_est_total_children:.2e}")
    print(f"  Total survivors: {l1_est_total_surv:.2e}")
    print(f"  Time on {cores} cores: {l1_est_time/3600:.2f} hours")
    print(flush=True)

    if not l1_all:
        print("L1 produced 0 survivors — PROVEN at L1!")
        return

    l1_surv = np.vstack(l1_all)
    print(f"L1 survivor pool: {len(l1_surv):,} (from {n_l1} parents)")

    # --- L2: find processable parents, measure throughput + expansion ---
    d_child_l2 = 2 * l1_surv.shape[1]
    nhc_l2 = d_child_l2 // 2
    counts = children_counts(l1_surv, m, d_child_l2, c_target)

    print(f"\nL2 children/parent distribution:")
    for pct in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        val = int(np.percentile(counts, pct)) if pct < 100 else int(counts.max())
        print(f"  p{pct:>3}: {val:,}")

    feasible = np.where(counts <= args.l2_max_children)[0]
    feasible = feasible[np.argsort(counts[feasible])]
    n_feasible = len(feasible)
    n_l2 = min(args.l2_sample, n_feasible)

    if n_l2 == 0:
        print(f"\nNo L2 parents under {args.l2_max_children:,} children.")
        print(f"Lightest: {int(counts.min()):,}")
        print(f"Cannot estimate L2 time. Need GPU or bigger machine.")
        # Still project
        l2_median = float(np.median(counts))
        l2_est_total = l2_median * l1_est_total_surv
        print(f"\nL2 projected (using median children/parent):")
        print(f"  Median children/parent: {l2_median:.2e}")
        print(f"  Total children: {l2_est_total:.2e}")
        # Estimate L2 rate as L1 rate / scaling factor
        d_ratio = d_child_l2 / (2 * n_half * 2)  # ratio of d values
        l2_rate_est = l1_rate / (d_ratio ** 1.5)  # rough scaling
        l2_time_est = l2_est_total / (l2_rate_est * cores)
        print(f"  Estimated rate: {l2_rate_est/1e6:.2f}M/sec (1 core)")
        print(f"  Time on {cores} cores: {l2_time_est/3600:.1f} hours")
        return

    print(f"\nL2: {n_feasible} parents under {args.l2_max_children:,}, "
          f"processing {n_l2}...", flush=True)

    l2_total_children = 0
    l2_total_surv = 0
    l2_total_time = 0.0

    for i in range(n_l2):
        ix = feasible[i]
        t2 = time.time()
        s, nc = process_parent_fused(l1_surv[ix], m, c_target, nhc_l2)
        dt = time.time() - t2
        l2_total_children += nc
        l2_total_surv += len(s)
        l2_total_time += dt
        print(f"  L2 p{i}: {len(s):,}/{nc:,} ({dt:.1f}s)", flush=True)

    l2_rate = l2_total_children / l2_total_time if l2_total_time > 0 else l1_rate / 4
    l2_avg_surv = l2_total_surv / n_l2
    l2_median_children = float(np.median(counts))
    l2_est_total_children = l2_median_children * l1_est_total_surv
    l2_est_time = l2_est_total_children / (l2_rate * cores)

    print(f"\nL2 measured:")
    print(f"  Throughput: {l2_rate/1e6:.2f}M children/sec (1 core)")
    print(f"  Survivors: {l2_total_surv:,} / {l2_total_children:,} "
          f"({n_l2} parents)")
    print(f"  Expansion: {l2_avg_surv:.1f}x per parent")
    print(f"L2 projected full run:")
    print(f"  Median children/parent: {l2_median_children:.2e}")
    print(f"  Projected L2 parents: {l1_est_total_surv:.2e}")
    print(f"  Total children: {l2_est_total_children:.2e}")
    print(f"  Time on {cores} cores: {l2_est_time/3600:.1f} hours")
    print(flush=True)

    # --- Summary ---
    total_time = l0_time + l1_est_time + l2_est_time
    print(f"{'='*50}")
    print(f"TOTAL ESTIMATED TIME on {cores} cores:")
    print(f"  L0: {l0_time:.1f}s")
    print(f"  L1: {l1_est_time/3600:.2f}h")
    print(f"  L2: {l2_est_time/3600:.1f}h")
    print(f"  TOTAL: {total_time/3600:.1f} hours")
    print(f"{'='*50}")

    if l2_total_surv == 0:
        print(f"\nL2 expansion = 0 ({n_l2} parents, {l2_total_children:,} children)")
        print(f"CASCADE LIKELY CONVERGES AT L2.")
    else:
        l2_est_surv = l2_avg_surv * l1_est_total_surv
        print(f"\nL2 projected survivors: {l2_est_surv:.2e}")
        print(f"L3 would be needed.")


if __name__ == '__main__':
    main()

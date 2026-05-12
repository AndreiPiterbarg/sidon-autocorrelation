"""1-hour cloud estimate for c_target=1.35 proof runtime.

Measures actual throughput on the cloud CPU, then:
  1. Runs L0 fully (fast)
  2. Processes L1 parents in parallel for 30 minutes
  3. Collects L1 survivors, processes lightest L2 parents in parallel
  4. Projects total proof time from measured rates

Run on cloud pod:
    python tests/cloud_estimate.py
"""
import sys
import os

# CRITICAL: disable Numba internal parallelism so multiprocessing workers
# each use 1 core cleanly instead of all fighting for the same threads.
os.environ['NUMBA_NUM_THREADS'] = '1'

import time
import math
import multiprocessing
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction
from run_cascade import run_level0, process_parent_fused

M = 20
N_HALF = 2
C_TARGET = 1.35
CORES = multiprocessing.cpu_count()
# Use half physical cores to avoid overwhelming the pod
WORKERS = max(1, CORES // 4)
L1_TIME_BUDGET = 30 * 60
L2_TIME_BUDGET = 20 * 60
L2_MAX_CHILDREN = 5_000_000_000


def _process_l1_parent(parent):
    """Worker function for L1 multiprocessing."""
    s, nc = process_parent_fused(parent, M, C_TARGET, 2 * N_HALF)
    return s, nc


def _process_l2_parent(args):
    """Worker function for L2 multiprocessing."""
    parent, nhc = args
    s, nc = process_parent_fused(parent, M, C_TARGET, nhc)
    return s, nc


def children_counts(parents, m, d_child, c_target):
    n_half_child = d_child // 2
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    xc = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    xc_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    xc = max(min(xc, xc_cs), 0)
    B = parents.astype(np.int64)
    lo = np.maximum(0, 2 * B - xc)
    hi = np.minimum(2 * B, xc)
    eff = np.maximum(hi - lo + 1, 0)
    return np.prod(eff.astype(np.float64), axis=1), xc


def main():
    print(f"=" * 60)
    print(f"Cloud Runtime Estimate: c_target={C_TARGET}, m={M}, n_half={N_HALF}")
    print(f"CPU cores detected: {CORES}")
    print(f"L1 budget: {L1_TIME_BUDGET//60}min, L2 budget: {L2_TIME_BUDGET//60}min")
    print(f"=" * 60)
    print(flush=True)

    # --- L0 ---
    t0 = time.time()
    l0 = run_level0(N_HALF, M, C_TARGET, verbose=True)
    l0_time = time.time() - t0
    survivors = l0['survivors']
    n0 = len(survivors)
    print(f"\nL0: {n0:,} survivors in {l0_time:.1f}s\n", flush=True)

    # --- L1: process parents in parallel ---
    rng = np.random.RandomState(42)
    order = rng.permutation(n0)

    l1_total_children = 0
    l1_total_surv = 0
    l1_all_surv = []
    l1_n_done = 0
    l1_start = time.time()

    print(f"L1: processing parents with {WORKERS} workers "
          f"(budget {L1_TIME_BUDGET//60}min)...", flush=True)

    all_parents = [survivors[ix] for ix in order]
    with multiprocessing.Pool(WORKERS) as pool:
        for s, nc in pool.imap_unordered(_process_l1_parent, all_parents,
                                          chunksize=4):
            if time.time() - l1_start > L1_TIME_BUDGET:
                pool.terminate()
                break
            l1_total_children += nc
            l1_total_surv += len(s)
            l1_n_done += 1
            if len(s) > 0:
                l1_all_surv.append(s)

            if l1_n_done % 100 == 0:
                elapsed = time.time() - l1_start
                rate = l1_total_children / elapsed
                avg_surv = l1_total_surv / l1_n_done
                print(f"  [{l1_n_done} parents, {elapsed:.0f}s] "
                      f"{rate/1e6:.1f}M children/sec, "
                      f"avg surv/parent={avg_surv:,.0f}", flush=True)

    l1_elapsed = time.time() - l1_start
    l1_rate = l1_total_children / l1_elapsed  # children/sec (all cores via numba)
    l1_avg_children = l1_total_children / l1_n_done
    l1_avg_surv = l1_total_surv / l1_n_done
    l1_expansion = l1_avg_surv

    # Project full L1 run
    l1_total_children_full = l1_avg_children * n0
    l1_time_full = l1_total_children_full / l1_rate

    print(f"\nL1 RESULTS ({l1_n_done} parents in {l1_elapsed:.0f}s):")
    print(f"  Throughput: {l1_rate/1e6:.2f}M children/sec (all {CORES} cores)")
    print(f"  Avg children/parent: {l1_avg_children:,.0f}")
    print(f"  Avg survivors/parent: {l1_avg_surv:,.0f}")
    print(f"  Expansion: {l1_expansion:,.0f}x")
    print(f"  Projected full L1: {l1_total_children_full:.2e} children, "
          f"{l1_time_full/3600:.2f}h")
    print(flush=True)

    # --- L2: collect L1 survivors, process lightest ---
    if not l1_all_surv:
        print("L1 produced 0 survivors — PROVEN at L1!")
        return

    l1_surv = np.vstack(l1_all_surv)
    print(f"L1 survivor pool: {len(l1_surv):,} from {l1_n_done} parents")

    d_child_l2 = 2 * l1_surv.shape[1]
    counts_l2, xc_l2 = children_counts(l1_surv, M, d_child_l2, C_TARGET)

    print(f"\nL2 children/parent distribution (x_cap={xc_l2}):")
    for pct in [0, 1, 5, 10, 25, 50, 75, 90, 99]:
        print(f"  p{pct:>2}: {np.percentile(counts_l2, pct):,.0f}")
    print(f"  max: {counts_l2.max():,.0f}")

    # Filter and sort
    feasible = np.where(counts_l2 <= L2_MAX_CHILDREN)[0]
    feasible = feasible[np.argsort(counts_l2[feasible])]
    print(f"\n{len(feasible)} L2 parents under {L2_MAX_CHILDREN:,} "
          f"(of {len(l1_surv):,})", flush=True)

    if len(feasible) == 0:
        print("No processable L2 parents. Cannot measure L2.")
        # Still project
        l2_median = float(np.median(counts_l2))
        l2_projected_parents = l1_expansion * n0
        l2_total = l2_median * l2_projected_parents
        print(f"L2 projected: {l2_projected_parents:.2e} parents × "
              f"{l2_median:.2e} median children = {l2_total:.2e} total")
        print(f"L2 time estimate: {l2_total / l1_rate / 3600:.0f}h "
              f"(using L1 rate as lower bound)")
        return

    l2_total_children = 0
    l2_total_surv = 0
    l2_n_done = 0
    l2_start = time.time()
    nhc_l2 = d_child_l2 // 2

    print(f"\nL2: processing parents with {WORKERS} workers "
          f"(budget {L2_TIME_BUDGET//60}min)...", flush=True)

    l2_args = [(l1_surv[ix], nhc_l2) for ix in feasible]
    with multiprocessing.Pool(WORKERS) as pool:
        for s, nc in pool.imap_unordered(_process_l2_parent, l2_args,
                                          chunksize=1):
            if time.time() - l2_start > L2_TIME_BUDGET:
                pool.terminate()
                break
            l2_total_children += nc
            l2_total_surv += len(s)
            l2_n_done += 1
            elapsed = time.time() - l2_start
            rate = l2_total_children / elapsed if elapsed > 0 else 0
            print(f"  L2 p{l2_n_done-1}: {len(s):,}/{nc:,} "
                  f"[total: {l2_total_surv}/{l2_total_children:,}, "
                  f"{rate/1e6:.1f}M/s]", flush=True)

    l2_elapsed = time.time() - l2_start
    l2_rate = l2_total_children / l2_elapsed if l2_elapsed > 0 else l1_rate

    print(f"\nL2 RESULTS ({l2_n_done} parents in {l2_elapsed:.0f}s):")
    print(f"  Throughput: {l2_rate/1e6:.2f}M children/sec")
    print(f"  Survivors: {l2_total_surv:,} / {l2_total_children:,}")
    print(f"  Expansion: {l2_total_surv/l2_n_done:.1f}x per parent")

    # --- Project full proof time ---
    l2_projected_parents = l1_expansion * n0
    l2_median_children = float(np.median(counts_l2))
    l2_total_full = l2_median_children * l2_projected_parents
    l2_time_full = l2_total_full / l2_rate

    print(f"\n{'='*60}")
    print(f"FULL PROOF PROJECTION (c_target={C_TARGET}, m={M})")
    print(f"{'='*60}")
    print(f"L0: {n0:,} survivors, {l0_time:.1f}s")
    print(f"L1: {l1_total_children_full:.2e} children, {l1_time_full/3600:.2f}h "
          f"(rate={l1_rate/1e6:.1f}M/s)")
    print(f"    {l1_expansion:,.0f}x expansion -> {l2_projected_parents:.2e} L2 parents")
    print(f"L2: {l2_total_full:.2e} children, {l2_time_full/3600:.1f}h "
          f"(rate={l2_rate/1e6:.1f}M/s)")
    print(f"    Expansion: {l2_total_surv/max(l2_n_done,1):.1f}x per parent")
    total_hours = l0_time/3600 + l1_time_full/3600 + l2_time_full/3600
    print(f"\nTOTAL: {total_hours:.1f} hours on {CORES} cores")
    if l2_total_surv == 0:
        print(f"L2 EXPANSION = 0 from {l2_n_done} parents — cascade converges!")
        print(f"But full L2 run ({l2_total_full:.2e} children) takes {l2_time_full/3600:.1f}h")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

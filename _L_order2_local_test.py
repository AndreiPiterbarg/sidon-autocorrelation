"""Local order-1 vs order-2 Lasserre L-prune benchmark.

Picks ~100 F+Q-survivor compositions at d=8, n_half=4, m=10, c_target=1.281,
runs order-1 and order-2 L-prune on each, reports kill counts and wall times.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, prune_L_one, _detect_solver

N_HALF = 4
M = 10
C_TARGET = 1.281
TARGET_SURV = 100
ORDER2_TIME_BUDGET_S = 600.0  # cap order-2 wall

def main():
    d = 2 * N_HALF
    S_half = 2 * N_HALF * M
    solver = _detect_solver('MOSEK')
    print(f"solver = {solver}")
    if solver == 'NONE':
        print("No solver available; abort.")
        return
    windows, ell_int_sums = _build_windows(d)
    A_mats = _build_A_matrices(d, windows)
    sigmas = _enum_balanced_signs(d)
    print(f"d={d}, n_win={len(windows)}, n_sigma={len(sigmas)}")

    # Collect F+Q survivors
    survivors = []
    t0 = time.time()
    for half_batch in generate_compositions_batched(N_HALF, S_half, batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :N_HALF] = half_batch
        batch[:, N_HALF:] = half_batch[:, ::-1]
        sF = prune_F(batch, N_HALF, M, C_TARGET)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            c = batch[idx]
            if not prune_Q_one(c, windows, ell_int_sums, sigmas, N_HALF, M, C_TARGET, margin=1e-9):
                survivors.append(c.copy())
                if len(survivors) >= TARGET_SURV:
                    break
        if len(survivors) >= TARGET_SURV:
            break
    print(f"Collected {len(survivors)} F+Q survivors in {time.time()-t0:.2f}s")
    if not survivors:
        print("No survivors -- nothing to test.")
        return

    # Order 1
    print("\n=== Order 1 (Shor) ===")
    times1 = []
    pruned1 = 0
    statuses1 = {}
    t1 = time.time()
    for i, c in enumerate(survivors):
        ts = time.time()
        pr, st = prune_L_one(c, A_mats, windows, N_HALF, M, C_TARGET,
                              solver=solver, order=1)
        dt = time.time() - ts
        times1.append(dt)
        statuses1[st] = statuses1.get(st, 0) + 1
        if pr:
            pruned1 += 1
    print(f"  pruned: {pruned1}/{len(survivors)}")
    print(f"  median {np.median(times1)*1000:.1f}ms  p95 {np.percentile(times1,95)*1000:.1f}ms  max {np.max(times1)*1000:.1f}ms")
    print(f"  total wall: {time.time()-t1:.2f}s")
    print(f"  statuses: {statuses1}")

    # Order 2: run a small calibration first
    print("\n=== Order 2 (full Lasserre) ===")
    print("Calibrating with first 5 compositions...")
    times2 = []
    pruned2 = 0
    statuses2 = {}
    t2 = time.time()
    cal_n = min(5, len(survivors))
    for i in range(cal_n):
        ts = time.time()
        pr, st = prune_L_one(survivors[i], A_mats, windows, N_HALF, M, C_TARGET,
                              solver=solver, order=2)
        dt = time.time() - ts
        times2.append(dt)
        statuses2[st] = statuses2.get(st, 0) + 1
        if pr:
            pruned2 += 1
        print(f"  [{i+1}/{cal_n}] dt={dt:.2f}s pruned={pr} status={st}")

    median_so_far = float(np.median(times2))
    est_full = median_so_far * len(survivors)
    print(f"  median so far: {median_so_far:.2f}s -> est full {len(survivors)} comps = {est_full:.1f}s")

    if est_full > ORDER2_TIME_BUDGET_S:
        print(f"  Estimated wall {est_full:.1f}s > budget {ORDER2_TIME_BUDGET_S}s; stopping at {cal_n}.")
    else:
        for i in range(cal_n, len(survivors)):
            ts = time.time()
            pr, st = prune_L_one(survivors[i], A_mats, windows, N_HALF, M, C_TARGET,
                                  solver=solver, order=2)
            dt = time.time() - ts
            times2.append(dt)
            statuses2[st] = statuses2.get(st, 0) + 1
            if pr:
                pruned2 += 1
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{len(survivors)}] cum_pruned={pruned2} cum_wall={time.time()-t2:.1f}s")
            if time.time() - t2 > ORDER2_TIME_BUDGET_S:
                print(f"  Hit time budget at i={i+1}; stopping.")
                break

    n2 = len(times2)
    print(f"\n  Order-2 ran on {n2} comps")
    print(f"  pruned: {pruned2}/{n2}")
    print(f"  median {np.median(times2)*1000:.1f}ms  p95 {np.percentile(times2,95)*1000:.1f}ms  max {np.max(times2)*1000:.1f}ms")
    print(f"  total wall: {time.time()-t2:.2f}s")
    print(f"  statuses: {statuses2}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  config: d={d}, n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"  F+Q survivors tested: {len(survivors)}")
    print(f"  Order-1: pruned {pruned1}/{len(survivors)} ({100*pruned1/len(survivors):.1f}%)  median {np.median(times1)*1000:.1f}ms")
    print(f"  Order-2: pruned {pruned2}/{n2} ({100*pruned2/n2:.1f}%)  median {np.median(times2)*1000:.1f}ms")
    if n2 == len(survivors):
        extra = pruned2 - pruned1
        print(f"  Order-2 extra over Order-1: {extra} ({100*extra/len(survivors):.1f}%)")

    out = {
        'config': {'d': d, 'n_half': N_HALF, 'm': M, 'c_target': C_TARGET},
        'n_FQ_survivors': len(survivors),
        'order1': {'pruned': pruned1, 'n': len(survivors),
                   'median_ms': float(np.median(times1)*1000),
                   'p95_ms': float(np.percentile(times1,95)*1000),
                   'max_ms': float(np.max(times1)*1000),
                   'statuses': statuses1},
        'order2': {'pruned': pruned2, 'n': n2,
                   'median_ms': float(np.median(times2)*1000),
                   'p95_ms': float(np.percentile(times2,95)*1000),
                   'max_ms': float(np.max(times2)*1000),
                   'statuses': statuses2},
        'solver': solver,
    }
    with open(os.path.join(_dir, '_L_order2_local_test.json'), 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    np.save(os.path.join(_dir, '_L_order2_survivors.npy'),
            np.array(survivors, dtype=np.int32))
    print(f"\nSaved _L_order2_local_test.json and _L_order2_survivors.npy")

if __name__ == '__main__':
    main()

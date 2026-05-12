#!/usr/bin/env python
"""Order-3 Lasserre with ALL optimizations: sparse cliques + CG + upper loc + secant.

Streams every update for monitoring.
"""
import sys
import os
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70, flush=True)
print("ORDER-3 SPARSE LASSERRE — ALL OPTIMIZATIONS", flush=True)
print("=" * 70, flush=True)
print(f"Started: {time.strftime('%H:%M:%S')}", flush=True)

from lasserre_enhanced import solve_enhanced

val_d_precise = {
    4: 1.10233300,
    6: 1.17110285,
    8: 1.20464420,
    10: 1.24136874,
    12: 1.27071936,
}

results = {}

# Configs: (d, order, psd_mode, bandwidth, search, n_bisect, max_cg)
configs = [
    # Baselines (fast)
    (8,  2, 'full',   None, 'bisect', 12, 8,  "d=8 O2 full (baseline)"),
    # Order-3 sparse with varying bandwidth
    (8,  3, 'sparse', 4,    'bisect', 10, 6,  "d=8 O3 sparse bw=4"),
    (8,  3, 'sparse', 6,    'bisect', 10, 6,  "d=8 O3 sparse bw=6"),
    # Order-3 sparse + secant search
    (8,  3, 'sparse', 6,    'secant', 10, 6,  "d=8 O3 sparse bw=6 secant"),
    # Push to d=10
    (10, 2, 'full',   None, 'bisect', 12, 8,  "d=10 O2 full (baseline)"),
    (10, 3, 'sparse', 4,    'bisect', 10, 6,  "d=10 O3 sparse bw=4"),
    (10, 3, 'sparse', 6,    'bisect', 10, 6,  "d=10 O3 sparse bw=6"),
    # Push to d=12
    (12, 2, 'sparse', 6,    'bisect', 10, 6,  "d=12 O2 sparse bw=6 (baseline)"),
    (12, 3, 'sparse', 4,    'bisect', 8,  4,  "d=12 O3 sparse bw=4"),
    (12, 3, 'sparse', 6,    'bisect', 8,  4,  "d=12 O3 sparse bw=6"),
]

for d, order, psd, bw, search, nb, mcg, tag in configs:
    val_d = val_d_precise.get(d, 1.3)
    gap = val_d - 1.0

    print(f"\n>>> START [{tag}] at {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()

    try:
        kwargs = dict(
            d=d, c_target=val_d - 0.02,
            order=order, psd_mode=psd,
            search_mode=search,
            n_bisect=nb, max_cg_rounds=mcg,
            max_add_per_round=20,
            add_upper_loc=True,
            verbose=True,
        )
        if bw is not None:
            kwargs['sparse_bandwidth'] = bw

        r = solve_enhanced(**kwargs)
        lb = r['lb']
        gc = (lb - 1.0) / gap * 100
        elapsed = time.time() - t0
        results[tag] = (lb, gc, elapsed)

        print(f">>> RESULT [{tag}]: lb={lb:.10f} gc={gc:.2f}% "
              f"time={elapsed:.1f}s", flush=True)

    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        print(f">>> ERROR [{tag}]: {e} time={elapsed:.1f}s", flush=True)
        traceback.print_exc()
        results[tag] = None

# Final summary
print("\n" + "=" * 70, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"{'Tag':<35} {'lb':>14} {'gc%':>8} {'time':>8}", flush=True)
print("-" * 70, flush=True)

for d, order, psd, bw, search, nb, mcg, tag in configs:
    r = results.get(tag)
    if r is not None:
        lb, gc, elapsed = r
        print(f"{tag:<35} {lb:>14.10f} {gc:>7.2f}% {elapsed:>7.1f}s",
              flush=True)
    else:
        print(f"{tag:<35} {'FAILED':>14}", flush=True)

print(f"\nDone at {time.strftime('%H:%M:%S')}", flush=True)

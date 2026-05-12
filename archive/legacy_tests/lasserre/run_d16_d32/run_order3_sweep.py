#!/usr/bin/env python
"""Order-3 vs Order-2 Lasserre comparison sweep with streaming output.

Streams progress line-by-line so monitoring picks up every update.
"""
import sys
import os
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70, flush=True)
print("ORDER-3 vs ORDER-2 LASSERRE SWEEP", flush=True)
print("=" * 70, flush=True)
print(f"Started: {time.strftime('%H:%M:%S')}", flush=True)

from lasserre_enhanced import solve_enhanced

val_d_precise = {
    4: 1.10233300,
    6: 1.17110285,
    8: 1.20464420,
}

results = {}

for d in [4, 6, 8]:
    val_d = val_d_precise[d]
    gap = val_d - 1.0

    for order in [2, 3]:
        tag = f"d={d} order={order}"
        print(f"\n>>> START {tag} at {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()

        try:
            # Use fewer bisect/CG rounds for order=3 to keep runtime sane
            nb = 10 if order == 2 else 8
            mcg = 8 if order == 2 else 4
            r = solve_enhanced(
                d, c_target=val_d - 0.01,
                order=order, psd_mode='full',
                n_bisect=nb, max_cg_rounds=mcg,
                add_upper_loc=True, verbose=True,
            )
            lb = r['lb']
            gc = (lb - 1.0) / gap * 100
            elapsed = time.time() - t0
            results[(d, order)] = lb

            print(f">>> RESULT {tag}: lb={lb:.10f} "
                  f"gc={gc:.2f}% time={elapsed:.1f}s", flush=True)

        except Exception as e:
            elapsed = time.time() - t0
            print(f">>> ERROR {tag}: {e} time={elapsed:.1f}s", flush=True)
            results[(d, order)] = None

# Final summary
print("\n" + "=" * 70, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"{'d':>3} {'order':>5} {'lb':>14} {'gc%':>8} {'improvement':>12}", flush=True)
print("-" * 50, flush=True)

for d in [4, 6, 8]:
    val_d = val_d_precise[d]
    gap = val_d - 1.0
    lb2 = results.get((d, 2))
    lb3 = results.get((d, 3))
    for order in [2, 3]:
        lb = results.get((d, order))
        if lb is not None:
            gc = (lb - 1.0) / gap * 100
            imp = ""
            if order == 3 and lb2 is not None:
                imp = f"+{lb - lb2:.6f}"
            print(f"{d:>3} {order:>5} {lb:>14.10f} {gc:>7.2f}% {imp:>12}",
                  flush=True)
        else:
            print(f"{d:>3} {order:>5} {'FAILED':>14}", flush=True)

print(f"\nDone at {time.strftime('%H:%M:%S')}", flush=True)

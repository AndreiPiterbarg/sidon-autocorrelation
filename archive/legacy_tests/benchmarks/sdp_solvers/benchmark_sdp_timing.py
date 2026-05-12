#!/usr/bin/env python
"""Benchmark SDP solve times across configurations to predict runtime.

For each config, runs exactly 1 SDP solve and 1 CG round, measures wall time.
Then extrapolates total runtime based on expected CG rounds.
"""
import sys
import os
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre_enhanced import solve_enhanced
from lasserre_scalable import _precompute
from lasserre_fusion import enum_monomials

print("=" * 70, flush=True)
print("SDP TIMING BENCHMARK", flush=True)
print("=" * 70, flush=True)

# =====================================================================
# Step 1: Measure single-solve times by running 1 CG round only
# =====================================================================

configs = [
    # (d, order, psd_mode, bandwidth, label)
    # Known baselines
    (4,  2, 'full',   None, "d=4 O2 full"),
    (4,  3, 'full',   None, "d=4 O3 full"),
    (6,  2, 'full',   None, "d=6 O2 full"),
    (6,  3, 'full',   None, "d=6 O3 full"),
    (8,  2, 'full',   None, "d=8 O2 full"),
    # Sparse configs we need to predict
    (8,  3, 'sparse', 4,    "d=8 O3 sp4"),
    (8,  3, 'sparse', 6,    "d=8 O3 sp6"),
    (10, 2, 'full',   None, "d=10 O2 full"),
    (10, 3, 'sparse', 4,    "d=10 O3 sp4"),
    (10, 3, 'sparse', 6,    "d=10 O3 sp6"),
    (12, 2, 'sparse', 6,    "d=12 O2 sp6"),
    (12, 3, 'sparse', 4,    "d=12 O3 sp4"),
    (12, 3, 'sparse', 6,    "d=12 O3 sp6"),
    (16, 3, 'sparse', 4,    "d=16 O3 sp4"),
    (16, 3, 'sparse', 6,    "d=16 O3 sp6"),
]

results = {}

for d, order, psd, bw, label in configs:
    # Print problem size info
    if psd == 'full':
        n_basis = len(enum_monomials(d, order))
        n_loc = len(enum_monomials(d, order - 1)) if order >= 2 else 0
        size_str = f"moment={n_basis}x{n_basis} loc={n_loc}x{n_loc}"
    else:
        cs = bw + 1  # clique size
        n_cliques = d - bw
        cb = len(enum_monomials(cs, order))
        cb_loc = len(enum_monomials(cs, order - 1)) if order >= 2 else 0
        size_str = f"{n_cliques}cliques x {cb}x{cb} loc={cb_loc}x{cb_loc}"

    print(f"\n--- {label} [{size_str}] ---", flush=True)
    t0 = time.time()

    try:
        kwargs = dict(
            d=d, c_target=0.5,  # low target so it converges fast
            order=order, psd_mode=psd,
            search_mode='bisect',
            n_bisect=3,          # only 3 bisect probes (minimum to measure)
            max_cg_rounds=1,     # only 1 CG round
            max_add_per_round=10,
            add_upper_loc=True,
            verbose=False,
        )
        if bw is not None:
            kwargs['sparse_bandwidth'] = bw

        r = solve_enhanced(**kwargs)
        elapsed = time.time() - t0
        lb = r['lb']
        n_active = r.get('n_active_windows', r.get('n_cg_rounds', '?'))

        # Time breakdown: phase1 + 1 CG round (3 bisect solves + violation check)
        p1_time = r.get('p1_time', 0)
        p2_time = elapsed - p1_time if p1_time else elapsed

        results[label] = {
            'elapsed': elapsed,
            'p1_time': p1_time,
            'p2_time': p2_time,
            'per_solve_est': p2_time / 3.0,  # 3 bisect probes
            'lb': lb,
        }

        print(f"  total={elapsed:.2f}s  phase1={p1_time:.2f}s  "
              f"CG_round={p2_time:.2f}s  per_solve~{p2_time/3:.2f}s  "
              f"lb={lb:.6f}", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR after {elapsed:.1f}s: {e}", flush=True)
        results[label] = {'elapsed': elapsed, 'error': str(e)}


# =====================================================================
# Step 2: Extrapolate full run times
# =====================================================================

print("\n" + "=" * 70, flush=True)
print("EXTRAPOLATED FULL RUN TIMES", flush=True)
print("=" * 70, flush=True)
print(f"{'Config':<20} {'per_solve':>10} {'1 CG round':>12} "
      f"{'3 rounds':>10} {'6 rounds':>10}", flush=True)
print("-" * 70, flush=True)

for d, order, psd, bw, label in configs:
    r = results.get(label, {})
    ps = r.get('per_solve_est')
    if ps is not None:
        # Full run: n_bisect=10-12 probes per round
        round_time = ps * 12  # 12 bisect probes per round
        r3 = round_time * 3 + r.get('p1_time', 0)
        r6 = round_time * 6 + r.get('p1_time', 0)

        def fmt(s):
            if s < 60:
                return f"{s:.0f}s"
            elif s < 3600:
                return f"{s/60:.1f}m"
            else:
                return f"{s/3600:.1f}h"

        print(f"{label:<20} {fmt(ps):>10} {fmt(round_time):>12} "
              f"{fmt(r3):>10} {fmt(r6):>10}", flush=True)
    else:
        print(f"{label:<20} {'ERROR':>10}", flush=True)

print(f"\nDone at {time.strftime('%H:%M:%S')}", flush=True)

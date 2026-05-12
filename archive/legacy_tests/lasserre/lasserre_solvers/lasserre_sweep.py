#!/usr/bin/env python
"""L4 d=12 then L5 d=10 — measure gap closure scaling."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_fusion import solve_lasserre_fusion

val_d = {8: 1.205, 10: 1.241, 12: 1.271, 14: 1.284, 16: 1.319}

print("=" * 60)
print("LASSERRE GAP CLOSURE MEASUREMENT")
print("=" * 60, flush=True)

configs = [
    (12, 4, "L4 d=12 (Schur ~127GB)"),
    (10, 5, "L5 d=10 (Schur ~273GB)"),
]

for d, order, desc in configs:
    print(f"\n{'#'*60}")
    print(f"# {desc}")
    print(f"{'#'*60}", flush=True)
    try:
        r = solve_lasserre_fusion(d, 1.28, order=order, n_bisect=12)
        lb = r['lb']
        v = val_d.get(d, 0)
        gap = (v - lb) / (v - 1.0) * 100 if v > 1 else 0
        print(f"\n  => lb={lb:.6f}, val={v:.3f}, "
              f"gap_remaining={gap:.1f}%, "
              f"closure={100-gap:.1f}%", flush=True)
    except Exception as e:
        print(f"\n  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

print(f"\n{'='*60}", flush=True)

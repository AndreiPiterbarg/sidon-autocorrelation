#!/usr/bin/env python
"""Test val(16) > 1.30: run L0 at d0=16 with increasing S."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
from run_cascade import run_level0
from pruning import count_compositions

print("=" * 60)
print("val(16) TEST: does val(16) > 1.30?")
print("If survivors appear at large S, val(16) < 1.30")
print("=" * 60, flush=True)

for S in [10, 12, 15, 18, 20, 25, 30]:
    n = count_compositions(16, S)
    if n > 2_000_000_000:
        print(f"\nS={S}: {n:,} comps — SKIP", flush=True)
        continue
    t0 = time.time()
    r = run_level0(8.0, 20, 1.30, verbose=True, d0=16, coarse_S=S)
    elapsed = time.time() - t0
    surv = r['n_survivors']
    mn = r.get('min_cert_net')
    mn_s = f"{mn:.6f}" if mn is not None else "N/A"
    print(f"\n  S={S}: {n:,} comps, {surv} survivors, "
          f"net={mn_s}, {elapsed:.1f}s", flush=True)
    if surv > 0:
        print(f"\n  >>> val(16) < 1.30 — survivors at S={S}", flush=True)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60, flush=True)

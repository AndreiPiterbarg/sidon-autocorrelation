#!/usr/bin/env python3
"""Compare new GPU adaptive multi-level vs old CPU multi-level for c_target=1.2."""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))

from gpu import is_available, get_device_name, gpu_run_multi_level
from multilevel import run_multi_level

if not is_available():
    print("ERROR: No CUDA GPU found")
    sys.exit(1)

print(f"Device: {get_device_name()}")

results = {}

# =====================================================================
# Test 1: c_target=1.2, m=50 — GPU adaptive multi-level (new)
# =====================================================================
print(f"\n{'='*70}")
print("TEST 1: GPU adaptive multi-level, c_target=1.2, m=50")
print(f"{'='*70}")
t0 = time.time()
r = gpu_run_multi_level(n_start=3, n_max=24, m=50, c_target=1.2, verbose=True)
results['gpu_adaptive_1.2_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Test 2: c_target=1.2, m=50 — CPU multi-level (old baseline)
# =====================================================================
print(f"\n{'='*70}")
print("TEST 2: CPU multi-level (old), c_target=1.2, m=50")
print(f"{'='*70}")
t0 = time.time()
r = run_multi_level(n_start=3, n_max=24, m=50, c_target=1.2, verbose=True)
results['cpu_multilevel_1.2_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Test 3: c_target=1.25, m=50 — GPU adaptive multi-level
# =====================================================================
print(f"\n{'='*70}")
print("TEST 3: GPU adaptive multi-level, c_target=1.25, m=50")
print(f"{'='*70}")
t0 = time.time()
r = gpu_run_multi_level(n_start=3, n_max=24, m=50, c_target=1.25, verbose=True)
results['gpu_adaptive_1.25_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Test 4: c_target=1.25, m=50 — CPU multi-level (old baseline)
# =====================================================================
print(f"\n{'='*70}")
print("TEST 4: CPU multi-level (old), c_target=1.25, m=50")
print(f"{'='*70}")
t0 = time.time()
r = run_multi_level(n_start=3, n_max=24, m=50, c_target=1.25, verbose=True)
results['cpu_multilevel_1.25_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Test 5: c_target=1.28, m=50 — GPU adaptive multi-level
# =====================================================================
print(f"\n{'='*70}")
print("TEST 5: GPU adaptive multi-level, c_target=1.28, m=50")
print(f"{'='*70}")
t0 = time.time()
r = gpu_run_multi_level(n_start=3, n_max=24, m=50, c_target=1.28, verbose=True)
results['gpu_adaptive_1.28_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Test 6: c_target=1.28, m=50 — CPU multi-level (old baseline)
# =====================================================================
print(f"\n{'='*70}")
print("TEST 6: CPU multi-level (old), c_target=1.28, m=50")
print(f"{'='*70}")
t0 = time.time()
r = run_multi_level(n_start=3, n_max=24, m=50, c_target=1.28, verbose=True)
results['cpu_multilevel_1.28_m50'] = {
    'proven': r['proven'], 'elapsed': time.time() - t0,
    'level_stats': r['level_stats'], 'n_survivors': len(r['survivors'])
}

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print(f"{'='*70}")
for name, res in results.items():
    status = "PROVEN" if res['proven'] else f"NOT proven ({res['n_survivors']} surv)"
    print(f"  {name:40s}: {res['elapsed']:8.2f}s  {status}")
print(f"{'='*70}")

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
os.makedirs('data', exist_ok=True)
out_path = f'data/comparison_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {out_path}")

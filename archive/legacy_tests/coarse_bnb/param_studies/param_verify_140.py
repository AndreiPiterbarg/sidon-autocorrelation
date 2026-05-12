"""Verify convergence pattern at c_target=1.40 with larger samples.

The sampled cascade showed convergence at L4. But we need to verify
this isn't a sampling artifact. Use larger samples at each level.
"""
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
from run_cascade import (run_level0, process_parent_fused,
                         _canonicalize_inplace, _fast_dedup)


# =========================================================================
# Full L0 + Full L1 + Sampled L2 + Sampled L3 for c_target=1.40
# =========================================================================

C_TARGET = 1.40
N_HALF = 2
M = 20

print("=" * 80)
print(f"CONVERGENCE VERIFICATION: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
print("=" * 80)

# --- L0: Full ---
l0 = run_level0(N_HALF, M, C_TARGET, verbose=True)
current = l0['survivors']
print(f"\nL0: {len(current)} survivors (d={current.shape[1]})")

# --- L1: Full ---
d_parent = current.shape[1]
d_child = 2 * d_parent
n_half_child = d_child // 2

all_surv = []
total_children = 0
t0 = time.time()

for i, parent in enumerate(current):
    surv, nc = process_parent_fused(parent, M, C_TARGET, n_half_child)
    total_children += nc
    if len(surv) > 0:
        all_surv.append(surv)

elapsed = time.time() - t0
if all_surv:
    next_surv = np.vstack(all_surv)
    _canonicalize_inplace(next_surv)
    next_surv = _fast_dedup(next_surv)
else:
    next_surv = np.empty((0, d_child), dtype=np.int32)

print(f"\nL1: {len(current)} parents -> {len(next_surv)} unique survivors "
      f"(expansion={len(next_surv)/len(current):.1f}x)")
print(f"    {total_children:,} children total, {elapsed:.1f}s")
current = next_surv

# --- L2: Sample 500 parents ---
if len(current) > 0:
    d_parent = current.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    n_sample = min(500, len(current))
    rng = np.random.default_rng(42)
    if n_sample < len(current):
        idx = rng.choice(len(current), n_sample, replace=False)
        sample = current[idx]
    else:
        sample = current

    all_surv = []
    total_children = 0
    t0 = time.time()

    for i, parent in enumerate(sample):
        surv, nc = process_parent_fused(parent, M, C_TARGET, n_half_child)
        total_children += nc
        if len(surv) > 0:
            all_surv.append(surv)
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            n_s = sum(len(s) for s in all_surv)
            print(f"    L2 progress: {i+1}/{n_sample} parents, {n_s} survivors, {elapsed:.1f}s")

    elapsed = time.time() - t0
    n_surv_sample = sum(len(s) for s in all_surv)
    surv_per_parent = n_surv_sample / n_sample
    projected = int(surv_per_parent * len(current))

    # Also get unique survivors from sample
    if all_surv:
        sample_surv = np.vstack(all_surv)
        _canonicalize_inplace(sample_surv)
        sample_surv = _fast_dedup(sample_surv)
    else:
        sample_surv = np.empty((0, d_child), dtype=np.int32)

    print(f"\nL2: {len(current)} parents, sampled {n_sample}")
    print(f"    sample: {n_surv_sample} survivors ({len(sample_surv)} unique)")
    print(f"    surv/parent: {surv_per_parent:.1f}")
    print(f"    projected: {projected:,} (expansion={surv_per_parent:.1f}x)")
    print(f"    {total_children:,} children, {elapsed:.1f}s")

    current = sample_surv

    # --- L3: Sample from L2 survivors ---
    if len(current) > 0:
        d_parent = current.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        n_sample = min(200, len(current))
        if n_sample < len(current):
            idx = rng.choice(len(current), n_sample, replace=False)
            sample = current[idx]
        else:
            sample = current

        all_surv = []
        total_children = 0
        t0 = time.time()

        for i, parent in enumerate(sample):
            if time.time() - t0 > 300:  # 5 min budget
                print(f"    L3 timeout at {i}/{n_sample}")
                n_sample = i
                break
            surv, nc = process_parent_fused(parent, M, C_TARGET, n_half_child)
            total_children += nc
            if len(surv) > 0:
                all_surv.append(surv)
            if (i+1) % 20 == 0:
                elapsed = time.time() - t0
                n_s = sum(len(s) for s in all_surv)
                print(f"    L3 progress: {i+1}/{n_sample} parents, {n_s} survivors, {elapsed:.1f}s")

        elapsed = time.time() - t0
        n_surv_sample = sum(len(s) for s in all_surv)
        if n_sample > 0:
            surv_per_parent = n_surv_sample / n_sample
        else:
            surv_per_parent = 0
        projected = int(surv_per_parent * len(current))

        if all_surv:
            sample_surv = np.vstack(all_surv)
            _canonicalize_inplace(sample_surv)
            sample_surv = _fast_dedup(sample_surv)
        else:
            sample_surv = np.empty((0, d_child), dtype=np.int32)

        print(f"\nL3: {len(current)} parents (from L2 sample), sampled {n_sample}")
        print(f"    sample: {n_surv_sample} survivors ({len(sample_surv)} unique)")
        print(f"    surv/parent: {surv_per_parent:.1f}")
        print(f"    projected (from sample): {projected:,}")
        print(f"    {total_children:,} children, {elapsed:.1f}s")

        # --- L4: if any L3 survivors ---
        current = sample_surv
        if len(current) > 0:
            d_parent = current.shape[1]
            d_child = 2 * d_parent
            n_half_child = d_child // 2

            n_sample = min(100, len(current))
            if n_sample < len(current):
                idx = rng.choice(len(current), n_sample, replace=False)
                sample = current[idx]
            else:
                sample = current

            all_surv = []
            total_children = 0
            t0 = time.time()

            for i, parent in enumerate(sample):
                if time.time() - t0 > 300:
                    print(f"    L4 timeout at {i}/{n_sample}")
                    n_sample = i
                    break
                surv, nc = process_parent_fused(parent, M, C_TARGET, n_half_child)
                total_children += nc
                if len(surv) > 0:
                    all_surv.append(surv)

            elapsed = time.time() - t0
            n_surv_sample = sum(len(s) for s in all_surv)
            if n_sample > 0:
                surv_per_parent = n_surv_sample / n_sample
            print(f"\nL4: {len(current)} parents, sampled {n_sample}")
            print(f"    sample: {n_surv_sample} survivors")
            print(f"    surv/parent: {surv_per_parent:.1f}")
            print(f"    {total_children:,} children, {elapsed:.1f}s")
        else:
            print(f"\nL4: 0 parents -> CONVERGED at L3/L4!")


print("\n\n" + "=" * 80)
print("ANALYSIS: What do these expansion factors mean for compute?")
print("=" * 80)

print("""
If L0->L1 expansion = ~150x and L1->L2 expansion = ~130x:
  Full L2 set = 345 * 150 * 130 ~ 6.7 million survivors (at d=32)

If L2->L3 expansion drops to ~5x (as sampled):
  Full L3 set = 6.7M * 5 ~ 33 million survivors (at d=64)

If L3->L4 expansion drops to ~0 (as sampled):
  PROVEN!

The computational bottleneck:
  L2: ~50K parents * ~4400 children/parent = ~220M children to test
  L3: ~6.7M parents * ~56K children/parent = ~375 billion children to test

At GPU rate of 60K children/sec: L3 would take ~72 days on one GPU
At 64x H100 rate (~4M children/sec each): L3 takes ~27 minutes

This is WHY we need the GPU kernel!
""")

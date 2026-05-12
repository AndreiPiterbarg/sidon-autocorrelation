#!/bin/bash
# profile_bottlenecks.sh — Comprehensive GPU kernel profiling for bottleneck identification.
#
# Runs 4 tests on SMALL datasets (total runtime <5 minutes):
#   Test 1: ncu hardware metrics on 50 L2 parents (L2→L3, d_child=32)
#   Test 2: TRACE build — enumeration stats, QC hit rate (100 L2 parents)
#   Test 3: ncu hardware metrics on synthetic L3 parents (L3→L4, d_child=64)
#   Test 4: Parent analysis — children distribution, load imbalance prediction
#
# Usage:
#   cd /workspace/sidon-autocorrelation/gpu
#   chmod +x profile_bottlenecks.sh
#   ./profile_bottlenecks.sh
#
# Prerequisites:
#   - CUDA toolkit with ncu (Nsight Compute)
#   - checkpoint_L2_survivors.npy in ../data/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
PROFILE_DIR="${SCRIPT_DIR}/profile_results"
mkdir -p "$PROFILE_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  GPU Kernel Bottleneck Profiler"
echo "  Target: cascade_kernel (Sidon autocorrelation cascade prover)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 0: Prepare test datasets
# ──────────────────────────────────────────────────────────────────
echo "=== Step 0: Preparing test datasets ==="

python3 -c "
import numpy as np
import os, sys

data_dir = '${DATA_DIR}'
profile_dir = '${PROFILE_DIR}'

# Load L2 survivors for L2→L3 profiling (d_parent=16, d_child=32)
l2_path = os.path.join(data_dir, 'checkpoint_L2_survivors.npy')
if not os.path.exists(l2_path):
    print(f'ERROR: {l2_path} not found. Upload L2 checkpoint first.')
    sys.exit(1)

l2 = np.load(l2_path)
print(f'L2 survivors loaded: {l2.shape}')

# Small subset for ncu (50 parents — ncu replays are slow)
np.save(os.path.join(profile_dir, 'test_L2_50.npy'), l2[:50])
print(f'  Created test_L2_50.npy (50 parents, d=16)')

# Medium subset for TRACE (100 parents)
np.save(os.path.join(profile_dir, 'test_L2_100.npy'), l2[:100])
print(f'  Created test_L2_100.npy (100 parents, d=16)')

# Large subset for timing (1000 parents)
np.save(os.path.join(profile_dir, 'test_L2_1000.npy'), l2[:1000])
print(f'  Created test_L2_1000.npy (1000 parents, d=16)')

# === Synthetic L3 parents for d_child=64 profiling ===
# Real L3 survivors are m=20 distributed across d=32 bins.
# Generate realistic ones: sparse vectors summing to 20.
rng = np.random.default_rng(42)
d_parent_L3 = 32
m = 20
n_synthetic = 50

parents_L3 = np.zeros((n_synthetic, d_parent_L3), dtype=np.int32)
for i in range(n_synthetic):
    remaining = m
    # Distribute mass with most bins getting 0 or 1 (like real survivors)
    active_bins = rng.choice(d_parent_L3, size=min(8, d_parent_L3), replace=False)
    for j in active_bins[:-1]:
        if remaining <= 0:
            break
        val = rng.integers(0, min(remaining + 1, 4))
        parents_L3[i, j] = val
        remaining -= val
    if remaining > 0:
        parents_L3[i, active_bins[-1]] = remaining

np.save(os.path.join(profile_dir, 'test_L3_synth_50.npy'), parents_L3)
print(f'  Created test_L3_synth_50.npy (50 synthetic L3 parents, d=32)')

# 10 for ncu (very slow with d=64)
np.save(os.path.join(profile_dir, 'test_L3_synth_10.npy'), parents_L3[:10])
print(f'  Created test_L3_synth_10.npy (10 synthetic L3 parents, d=32)')

# === Parent analysis: children distribution ===
print()
print('=== Parent Analysis: Children-per-parent distribution (L2→L3) ===')

# For each parent, compute range product = number of children
m = 20
d_parent = 16
range_products = []
for i in range(min(len(l2), 10000)):
    parent = l2[i]
    product = 1
    for p in range(d_parent):
        a_p = int(parent[p])
        lo = max(0, a_p - m)  # conservative; actual lo/hi depend on half-mass
        hi = min(a_p, m)
        # Simplified: assume lo=0, hi=a_p for each parent bin
        # This overestimates but shows the range
        rng = a_p + 1  # each bin splits into (c, a_p - c) for c in [0, a_p]
        if rng > 1:
            product *= rng
    range_products.append(product)

rp = np.array(range_products, dtype=np.float64)
print(f'  Parents analyzed: {len(rp)}')
print(f'  Range products (children per parent):')
print(f'    min:    {rp.min():.0f}')
print(f'    median: {np.median(rp):.0f}')
print(f'    mean:   {rp.mean():.0f}')
print(f'    p95:    {np.percentile(rp, 95):.0f}')
print(f'    p99:    {np.percentile(rp, 99):.0f}')
print(f'    max:    {rp.max():.0f}')
print(f'  Parents with <=10 children: {(rp <= 10).sum()} ({(rp <= 10).mean()*100:.1f}%)')
print(f'  Parents with <=100 children: {(rp <= 100).sum()} ({(rp <= 100).mean()*100:.1f}%)')
print(f'  Parents with >10000 children: {(rp > 10000).sum()} ({(rp > 10000).mean()*100:.1f}%)')
# Compute load imbalance: if assigned round-robin to 132 SMs
n_sm = 132
n_blocks = min(len(rp), n_sm * 4)
total_work = rp.sum()
ideal_per_block = total_work / n_blocks
# Sort by work descending — heaviest parent determines runtime
rp_sorted = np.sort(rp)[::-1]
print(f'  Heaviest parent: {rp_sorted[0]:.0f} children')
print(f'  Ideal per-block (uniform): {ideal_per_block:.0f} children')
print(f'  Imbalance ratio: {rp_sorted[0] / max(ideal_per_block, 1):.1f}x')
" 2>&1 | tee "$PROFILE_DIR/test0_parent_analysis.txt"

echo ""
echo "=== Step 0 complete ==="
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 1: Build release + trace variants
# ──────────────────────────────────────────────────────────────────
echo "=== Step 1: Building kernel variants ==="

cd "$SCRIPT_DIR"

# Release build
echo "  Building RELEASE..."
./build.sh release 2>&1 | tail -3
cp cascade_prover "$PROFILE_DIR/cascade_prover_release"

# Trace build (adds enumeration completeness + per-parent stats)
echo "  Building TRACE..."
./build.sh trace 2>&1 | tail -3
cp cascade_prover "$PROFILE_DIR/cascade_prover_trace"

echo "  Build complete."
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 2: Baseline timing (release, 1000 parents, L2→L3)
# ──────────────────────────────────────────────────────────────────
echo "=== Step 2: Baseline timing — 1000 L2 parents, L2→L3 (d_child=32) ==="

"$PROFILE_DIR/cascade_prover_release" \
    "$PROFILE_DIR/test_L2_1000.npy" \
    "$PROFILE_DIR/output_timing.npy" \
    --d_parent 16 --m 20 --c_target 1.4 \
    --max_survivors 1000000 \
    2>&1 | tee "$PROFILE_DIR/test2_baseline_timing.txt"

echo ""

# ──────────────────────────────────────────────────────────────────
# Step 3: TRACE run — enumeration stats + QC effectiveness
# ──────────────────────────────────────────────────────────────────
echo "=== Step 3: TRACE run — 100 L2 parents (enumeration stats) ==="

"$PROFILE_DIR/cascade_prover_trace" \
    "$PROFILE_DIR/test_L2_100.npy" \
    "$PROFILE_DIR/output_trace.npy" \
    --d_parent 16 --m 20 --c_target 1.4 \
    --max_survivors 1000000 \
    2>&1 | tee "$PROFILE_DIR/test3_trace_output.txt"

echo ""
echo "  Parsing TRACE output..."
python3 -c "
import re

with open('${PROFILE_DIR}/test3_trace_output.txt') as f:
    lines = f.readlines()

total_tested = 0
total_expected = 0
total_skipped = 0
parent_children = []

for line in lines:
    m = re.search(r'tested=(\d+)\s+skipped=(\d+)\s+expected=(\d+)', line)
    if m:
        tested = int(m.group(1))
        skipped = int(m.group(2))
        expected = int(m.group(3))
        total_tested += tested
        total_expected += expected
        total_skipped += skipped
        parent_children.append(expected)

if total_expected > 0:
    print(f'TRACE Summary (100 parents, L2→L3, d_child=32):')
    print(f'  Total children expected: {total_expected:,}')
    print(f'  Total children tested:   {total_tested:,}')
    print(f'  Total skipped (subtree): {total_skipped:,}')
    print(f'  Subtree prune rate:      {total_skipped/max(total_expected,1)*100:.1f}%')

    import statistics
    if parent_children:
        print(f'  Children/parent distribution:')
        print(f'    min:    {min(parent_children):,}')
        print(f'    median: {statistics.median(parent_children):,.0f}')
        print(f'    mean:   {statistics.mean(parent_children):,.0f}')
        print(f'    max:    {max(parent_children):,}')
        print(f'    stdev:  {statistics.stdev(parent_children) if len(parent_children)>1 else 0:,.0f}')
else:
    print('  No TRACE output found — check kernel build.')
" 2>&1 | tee "$PROFILE_DIR/test3_trace_summary.txt"

echo ""

# ──────────────────────────────────────────────────────────────────
# Step 4: Nsight Compute profiling — hardware metrics
# ──────────────────────────────────────────────────────────────────
echo "=== Step 4: Nsight Compute profiling — 50 L2 parents ==="

# Check if ncu is available
NCU_BIN=""
if command -v ncu &>/dev/null; then
    NCU_BIN="ncu"
elif [ -x "/usr/local/cuda/bin/ncu" ]; then
    NCU_BIN="/usr/local/cuda/bin/ncu"
elif [ -x "/opt/nvidia/nsight-compute/ncu" ]; then
    NCU_BIN="/opt/nvidia/nsight-compute/ncu"
fi

if [ -z "$NCU_BIN" ]; then
    echo "  WARNING: ncu (Nsight Compute) not found. Skipping hardware profiling."
    echo "  Install: apt-get install -y nsight-compute"
    echo "  Or use: ncu from CUDA toolkit."

    # Fallback: use nvprof or basic CUDA events
    echo ""
    echo "  Falling back to cudaEvent-based profiling..."
    echo "  (Running release kernel and measuring wall-clock time only)"
else
    echo "  Using ncu at: $NCU_BIN"

    # Profile with key metrics sections
    # Using --replay-mode kernel to minimize overhead
    # Limiting to first kernel launch only
    echo ""
    echo "  --- SpeedOfLight + Occupancy ---"
    $NCU_BIN \
        --kernel-name "cascade_kernel" \
        --launch-count 1 \
        --section SpeedOfLight \
        --section Occupancy \
        "$PROFILE_DIR/cascade_prover_release" \
        "$PROFILE_DIR/test_L2_50.npy" \
        "$PROFILE_DIR/output_ncu_sol.npy" \
        --d_parent 16 --m 20 --c_target 1.4 \
        --max_survivors 100000 \
        2>&1 | tee "$PROFILE_DIR/test4_ncu_speedoflight.txt"

    echo ""
    echo "  --- Memory Workload Analysis ---"
    $NCU_BIN \
        --kernel-name "cascade_kernel" \
        --launch-count 1 \
        --section MemoryWorkloadAnalysis \
        "$PROFILE_DIR/cascade_prover_release" \
        "$PROFILE_DIR/test_L2_50.npy" \
        "$PROFILE_DIR/output_ncu_mem.npy" \
        --d_parent 16 --m 20 --c_target 1.4 \
        --max_survivors 100000 \
        2>&1 | tee "$PROFILE_DIR/test4_ncu_memory.txt"

    echo ""
    echo "  --- Warp State Statistics ---"
    $NCU_BIN \
        --kernel-name "cascade_kernel" \
        --launch-count 1 \
        --section WarpStateStatistics \
        --section SchedulerStatistics \
        "$PROFILE_DIR/cascade_prover_release" \
        "$PROFILE_DIR/test_L2_50.npy" \
        "$PROFILE_DIR/output_ncu_warp.npy" \
        --d_parent 16 --m 20 --c_target 1.4 \
        --max_survivors 100000 \
        2>&1 | tee "$PROFILE_DIR/test4_ncu_warpstate.txt"
fi

echo ""

# ──────────────────────────────────────────────────────────────────
# Step 4b: d_child=64 profiling (synthetic L3 parents, L3→L4)
# ──────────────────────────────────────────────────────────────────
echo "=== Step 4b: d_child=64 baseline timing — 50 synthetic L3 parents ==="

"$PROFILE_DIR/cascade_prover_release" \
    "$PROFILE_DIR/test_L3_synth_50.npy" \
    "$PROFILE_DIR/output_L3_timing.npy" \
    --d_parent 32 --m 20 --c_target 1.4 \
    --max_survivors 1000000 \
    2>&1 | tee "$PROFILE_DIR/test4b_d64_timing.txt"

echo ""

if [ -n "$NCU_BIN" ]; then
    echo "=== Step 4c: ncu SpeedOfLight+Occupancy at d_child=64 (10 parents) ==="

    $NCU_BIN \
        --kernel-name "cascade_kernel" \
        --launch-count 1 \
        --section SpeedOfLight \
        --section Occupancy \
        "$PROFILE_DIR/cascade_prover_release" \
        "$PROFILE_DIR/test_L3_synth_10.npy" \
        "$PROFILE_DIR/output_ncu_d64_sol.npy" \
        --d_parent 32 --m 20 --c_target 1.4 \
        --max_survivors 100000 \
        2>&1 | tee "$PROFILE_DIR/test4c_ncu_d64_speedoflight.txt"

    echo ""
    echo "  --- d=64 Warp State Statistics ---"
    $NCU_BIN \
        --kernel-name "cascade_kernel" \
        --launch-count 1 \
        --section WarpStateStatistics \
        --section SchedulerStatistics \
        "$PROFILE_DIR/cascade_prover_release" \
        "$PROFILE_DIR/test_L3_synth_10.npy" \
        "$PROFILE_DIR/output_ncu_d64_warp.npy" \
        --d_parent 32 --m 20 --c_target 1.4 \
        --max_survivors 100000 \
        2>&1 | tee "$PROFILE_DIR/test4c_ncu_d64_warpstate.txt"
fi

echo ""

# ──────────────────────────────────────────────────────────────────
# Step 4d: Shared memory analysis
# ──────────────────────────────────────────────────────────────────
echo "=== Step 4d: Shared memory occupancy analysis ==="

python3 -c "
print('=== Shared Memory Usage Analysis ===')
print()

# Static shared memory arrays (from cascade_kernel.cu)
arrays_d64 = {
    'parent_smem[32]':       32*4,
    'child_smem[64]':        64*4,
    'cursor_smem[32]':       32*4,
    'lo_smem[32]':           32*4,
    'hi_smem[32]':           32*4,
    'raw_conv_smem[127]':    127*4,
    'active_pos_smem[32]':   32*4,
    'radix_smem[32]':        32*4,
    'gc_a_smem[32]':         32*4,
    'gc_dir_smem[32]':       32*4,
    'gc_focus_smem[33]':     33*4,
    'prefix_conv_smem[128]': 128*4,   # UNUSED at d=64 (subtree pruning disabled)
    'prefix_tmp_smem[128]':  128*4,   # UNUSED at d=64
    'prefix_c_smem[65]':     65*4,    # UNUSED at d=64
    'surv_buf_smem[64*64]':  64*64*4, # 16 KB — survivor staging buffer
    'cmp_array_smem[64]':    64*4,
    'parent_prefix_smem[33]':33*4,    # UNUSED at d=64
    'qc_warp_sums_smem[2]':  2*4,
    'scalars (~20 vars)':    100,
}

unused_at_d64 = ['prefix_conv_smem[128]', 'prefix_tmp_smem[128]',
                 'prefix_c_smem[65]', 'parent_prefix_smem[33]']

total_static = sum(arrays_d64.values())
total_unused = sum(arrays_d64[k] for k in unused_at_d64)
surv_buf = arrays_d64['surv_buf_smem[64*64]']

print(f'Static shared memory per block:')
for name, size in sorted(arrays_d64.items(), key=lambda x: -x[1]):
    unused = ' ← UNUSED at d=64' if name in unused_at_d64 else ''
    big = ' ← LARGEST' if size == surv_buf else ''
    print(f'  {name:30s} {size:6d} B  ({size/1024:.1f} KB){unused}{big}')
print(f'  {\"TOTAL STATIC\":30s} {total_static:6d} B  ({total_static/1024:.1f} KB)')
print()

# Dynamic shared memory
d_child = 64
m = 20
ell_count = 2*d_child - 1  # 127
threshold_size = ell_count * (m+1) * 4
ell_order_size = ell_count * 4
total_dynamic = threshold_size + ell_order_size

print(f'Dynamic shared memory:')
print(f'  threshold_table[{ell_count}*{m+1}] {threshold_size:6d} B  ({threshold_size/1024:.1f} KB)')
print(f'  ell_order[{ell_count}]         {ell_order_size:6d} B  ({ell_order_size/1024:.1f} KB)')
print(f'  {\"TOTAL DYNAMIC\":30s} {total_dynamic:6d} B  ({total_dynamic/1024:.1f} KB)')
print()

grand_total = total_static + total_dynamic
print(f'GRAND TOTAL per block: {grand_total:,} B ({grand_total/1024:.1f} KB)')
print()

# H100 occupancy analysis
h100_smem = 228 * 1024  # 228 KB
h100_regs = 65536
h100_threads = 2048
h100_sms = 132
block_threads = 64

blocks_smem = h100_smem // grand_total
blocks_regs = h100_regs // (block_threads * 48)  # assume 48 regs/thread
blocks_threads = h100_threads // block_threads
blocks_actual = min(blocks_smem, blocks_regs, blocks_threads)

print(f'=== H100 Occupancy Analysis (current kernel, d=64) ===')
print(f'  Shared memory limit: {blocks_smem} blocks/SM  (228 KB / {grand_total/1024:.1f} KB)')
print(f'  Register limit:      {blocks_regs} blocks/SM  (65536 / {block_threads}*48)')
print(f'  Thread limit:        {blocks_threads} blocks/SM  (2048 / {block_threads})')
print(f'  ==> ACTUAL:          {blocks_actual} blocks/SM  (limited by shared memory)')
print(f'  Total concurrent:    {blocks_actual * h100_sms} blocks across {h100_sms} SMs')
print(f'  Warps per SM:        {blocks_actual * (block_threads // 32)}')
print(f'  Warps per scheduler: {blocks_actual * (block_threads // 32) / 4:.1f} (want ~20 for full latency hiding)')
print()

# Optimized: remove unused arrays + reduce surv_buf
for surv_cap in [64, 16, 4, 1]:
    opt_static = total_static - total_unused - surv_buf + surv_cap * d_child * 4
    opt_total = opt_static + total_dynamic
    opt_blocks = h100_smem // opt_total
    opt_blocks = min(opt_blocks, blocks_regs, blocks_threads)
    warps = opt_blocks * (block_threads // 32)
    speedup_est = opt_blocks / blocks_actual
    print(f'  SURV_CAP={surv_cap:2d}: smem={opt_total/1024:.1f}KB → {opt_blocks} blocks/SM → {warps} warps/SM → {speedup_est:.1f}x occupancy')

print()
print('=== BOTTLENECK SUMMARY ===')
print(f'  surv_buf_smem alone is {surv_buf/1024:.0f}KB = {surv_buf/grand_total*100:.0f}% of total shared memory')
print(f'  unused arrays at d=64: {total_unused/1024:.1f}KB = {total_unused/grand_total*100:.0f}% of total')
print(f'  Reducing both would increase occupancy by ~{min(h100_smem // ((total_static - total_unused - surv_buf + 4*d_child*4) + total_dynamic), blocks_regs, blocks_threads) / blocks_actual:.1f}x')
" 2>&1 | tee "$PROFILE_DIR/test4d_smem_analysis.txt"

echo ""

# ──────────────────────────────────────────────────────────────────
# Step 5: Compute-bound vs Memory-bound analysis
# ──────────────────────────────────────────────────────────────────
echo "=== Step 5: Theoretical analysis — compute vs memory bound ==="

python3 -c "
print('Theoretical Analysis: cascade_kernel hot loop')
print()

# Per-child operation counts (d_child=32, 1 warp)
d = 32
conv_len = 2*d - 1  # 63
m = 20

print('--- Per-child operation breakdown (d_child=32, 1 warp) ---')
print()

# Step 1: GC advance + child update (lane 0 only)
gc_ops = 15  # a few adds, compares, stores
print(f'Step 1 (GC advance):    ~{gc_ops} ops (lane 0 only)')

# Step 2: Conv update — each thread writes 1 conv entry
# Read child[lane], child[lane-1], compute delta_total, write conv[k1+lane]
conv_reads = 4  # child[lane], child[lane-1], conv[k1+lane], old values
conv_ops = 8    # multiply, subtract, add, compare
conv_writes = 1
print(f'Step 2 (Conv update):   ~{conv_reads} reads + {conv_ops} ops + {conv_writes} write per thread')

# Step 2.5: QC check (lane 0)
# Sum ~16 conv values, 1 threshold lookup, 1 compare
qc_sum_ops = conv_len // 2  # average qc_ell
print(f'Step 2.5 (QC, lane 0):  ~{qc_sum_ops} adds + 1 lookup + 1 compare')

# Step 3: Full scan (15% of children)
# Each thread handles ~2 ells
ells_per_thread = conv_len // d  # ~2
# Per ell: O(d) initial sum + O(d) sliding window
ops_per_ell = d + d  # initial sum + sliding
total_scan_ops = ells_per_thread * ops_per_ell
print(f'Step 3 (Full scan):     ~{total_scan_ops} ops per thread (only 15% of children)')

# Barriers
barriers_qc_hit = 3   # barrier 1 + 2 + 2.5
barriers_qc_miss = 4  # barrier 1 + 2 + 2.5 + 3
barrier_cost = 25      # cycles per __syncthreads (1 warp = cheap, just a memory fence)
print(f'Barriers:               {barriers_qc_hit} (QC hit) or {barriers_qc_miss} (QC miss)')
print(f'  Barrier cost (1 warp, d=32): ~{barrier_cost} cycles each')
print()

# Weighted per-child cycles
qc_hit_rate = 0.85
cycles_gc = 50
cycles_conv = 80
cycles_qc = 40
cycles_scan = 300
cycles_barrier = barrier_cost

total_qc_hit = cycles_gc + cycles_conv + cycles_qc + barriers_qc_hit * cycles_barrier
total_qc_miss = cycles_gc + cycles_conv + cycles_qc + cycles_scan + barriers_qc_miss * cycles_barrier
weighted = qc_hit_rate * total_qc_hit + (1 - qc_hit_rate) * total_qc_miss

print(f'--- Estimated cycles per child (d_child=32) ---')
print(f'  QC hit path:  {total_qc_hit} cycles ({qc_hit_rate*100:.0f}% of children)')
print(f'  QC miss path: {total_qc_miss} cycles ({(1-qc_hit_rate)*100:.0f}% of children)')
print(f'  Weighted avg: {weighted:.0f} cycles per child')
print()

# Now for d_child=64 (2 warps)
print('--- Estimated cycles per child (d_child=64, 2 warps) ---')
d64 = 64
barrier_cost_2w = 30  # __syncthreads with 2 warps is more expensive
cycles_gc_64 = 50     # same (lane 0 only)
cycles_conv_64 = 100  # same work, but 2 warps need barrier
cycles_qc_64 = 50     # lane 0 does more work (larger conv)
cycles_scan_64 = 400  # 2 ells per thread, each with O(64) work
barriers_64_hit = 3
barriers_64_miss = 4

total_64_hit = cycles_gc_64 + cycles_conv_64 + cycles_qc_64 + barriers_64_hit * barrier_cost_2w
total_64_miss = cycles_gc_64 + cycles_conv_64 + cycles_qc_64 + cycles_scan_64 + barriers_64_miss * barrier_cost_2w
weighted_64 = qc_hit_rate * total_64_hit + (1 - qc_hit_rate) * total_64_miss

print(f'  QC hit path:  {total_64_hit} cycles')
print(f'  QC miss path: {total_64_miss} cycles')
print(f'  Weighted avg: {weighted_64:.0f} cycles per child')
print()

# Throughput at H100 clock
h100_ghz = 1.83
children_per_sec_32 = h100_ghz * 1e9 / weighted * 132 * 4  # 132 SMs, ~4 blocks/SM
children_per_sec_64 = h100_ghz * 1e9 / weighted_64 * 132 * 6  # 132 SMs, ~6 blocks/SM

print(f'--- Throughput estimates (1x H100) ---')
print(f'  d=32: {children_per_sec_32/1e9:.2f} billion children/s')
print(f'  d=64: {children_per_sec_64/1e9:.2f} billion children/s')
print()

# L4 time estimate
l4_children = 7.4e12
l4_time_1h100 = l4_children / children_per_sec_64
l4_time_64h100 = l4_time_1h100 / 64
print(f'  L4 estimate (1x H100):  {l4_time_1h100/60:.1f} min')
print(f'  L4 estimate (64x H100): {l4_time_64h100/60:.1f} min')
print()

# Key bottleneck identification
print('═══════════════════════════════════════════════')
print('  KEY BOTTLENECK AREAS TO INVESTIGATE')
print('═══════════════════════════════════════════════')
print()
print('1. BARRIER OVERHEAD: ~25-30 cycles × 3-4 per child = 75-120 cycles')
print('   Fraction of per-child time: {:.0f}%'.format(barriers_64_hit * barrier_cost_2w / total_64_hit * 100))
print('   → Can we reduce barrier count or use warp-level sync?')
print()
print('2. FULL WINDOW SCAN: ~400 cycles (only 15% of children)')
print('   Fraction of weighted time: {:.0f}%'.format((1-qc_hit_rate) * cycles_scan_64 / weighted_64 * 100))
print('   → Can we improve QC hit rate or reduce scan cost?')
print()
print('3. QUICK-CHECK (lane-0 sequential): ~50 cycles')
print('   All other lanes IDLE during QC')
print('   Wasted warp-seconds: {:.0f}%'.format((d64-1)/d64 * cycles_qc_64 / total_64_hit * 100))
print('   → Parallelize QC across warp?')
print()
print('4. LOAD IMBALANCE: Parents vary 100x+ in children count')
print('   Atomic work-stealing helps but tail latency dominates')
print('   → Split heavy parents across multiple blocks?')
print()
print('5. OCCUPANCY vs SHARED MEMORY:')
print('   Static smem per block: ~30KB (conv, child, threshold, staging)')
print('   H100 max: 228KB/SM → ~7 blocks/SM')
print('   Actual occupancy limited by register count + barriers')
print()
" 2>&1 | tee "$PROFILE_DIR/test5_theoretical_analysis.txt"

echo ""

# ──────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  PROFILING COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $PROFILE_DIR/"
echo ""
echo "Files:"
ls -la "$PROFILE_DIR/"*.txt 2>/dev/null || echo "  (no text output files)"
echo ""
echo "To view results:"
echo "  cat $PROFILE_DIR/test5_theoretical_analysis.txt"
echo "  cat $PROFILE_DIR/test4_ncu_speedoflight.txt"
echo "  cat $PROFILE_DIR/test4_ncu_warpstate.txt"
echo ""

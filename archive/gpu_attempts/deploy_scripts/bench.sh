#!/bin/bash
# bench.sh — Benchmark the GPU cascade prover with custom parameters.
#
# Usage:
#   ./bench.sh                          # Run all available benchmark levels
#   ./bench.sh <d_parent> <m> <c_target> <parents.npy>  # Run specific config
#
# This script builds the kernel, runs benchmarks at each available level,
# and reports children/second throughput.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
PROVER="${SCRIPT_DIR}/cascade_prover"

echo "═══════════════════════════════════════════════════════════════"
echo "  Sidon Cascade GPU Prover — BENCHMARK MODE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build if needed
if [ ! -x "$PROVER" ]; then
    echo "Building kernel first..."
    cd "$SCRIPT_DIR" && ./build.sh release
    echo ""
fi

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.sm --format=csv,noheader
echo ""

# If custom args provided, run just that
if [ $# -ge 4 ]; then
    D_PARENT="$1"
    M="$2"
    C_TARGET="$3"
    PARENTS="$4"
    D_CHILD=$((2 * D_PARENT))
    OUTPUT="/tmp/bench_output.npy"

    echo "Custom benchmark: d_parent=$D_PARENT d_child=$D_CHILD m=$M c_target=$C_TARGET"
    echo "  Parents: $PARENTS"
    echo ""

    "$PROVER" "$PARENTS" "$OUTPUT" \
        --d_parent "$D_PARENT" --m "$M" --c_target "$C_TARGET" \
        --max_survivors 1000000

    rm -f "$OUTPUT"
    exit 0
fi

# Default: run benchmarks for n_half=3, m=15, c_target=1.35
M=15
C_TARGET=1.35

echo "Parameters: m=$M  c_target=$C_TARGET  (n_half=3 cascade)"
echo ""

# Level 1: L0→L1 (d_parent=6, d_child=12)
L0="${DATA_DIR}/bench_n3m15/checkpoint_L0_survivors.npy"
if [ -f "$L0" ]; then
    echo "──────────────────────────────────────────────────"
    echo "  Level 1: L0→L1  (d_parent=6, d_child=12)"
    echo "──────────────────────────────────────────────────"
    "$PROVER" "$L0" "/tmp/bench_L1.npy" \
        --d_parent 6 --m $M --c_target $C_TARGET \
        --max_survivors 1000000
    echo ""
else
    echo "SKIP L0→L1: $L0 not found"
fi

# Level 2: L1→L2 (d_parent=12, d_child=24)
L1="${DATA_DIR}/bench_n3m15/checkpoint_L1_survivors.npy"
if [ -f "$L1" ]; then
    echo "──────────────────────────────────────────────────"
    echo "  Level 2: L1→L2  (d_parent=12, d_child=24)"
    echo "──────────────────────────────────────────────────"
    "$PROVER" "$L1" "/tmp/bench_L2.npy" \
        --d_parent 12 --m $M --c_target $C_TARGET \
        --max_survivors 10000000
    echo ""
else
    echo "SKIP L1→L2: $L1 not found"
fi

# Level 3: L2→L3 (d_parent=24, d_child=48)
L2="${DATA_DIR}/bench_n3m15/checkpoint_L2_survivors.npy"
if [ -f "$L2" ]; then
    echo "──────────────────────────────────────────────────"
    echo "  Level 3: L2→L3  (d_parent=24, d_child=48)"
    echo "──────────────────────────────────────────────────"
    "$PROVER" "$L2" "/tmp/bench_L3.npy" \
        --d_parent 24 --m $M --c_target $C_TARGET \
        --max_survivors 200000000
    echo ""
else
    echo "SKIP L2→L3: $L2 not found"
fi

# Also benchmark with the existing n_half=2 data if available, for comparison
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Comparison: existing n_half=2, m=20, c_target=1.4 data"
echo "═══════════════════════════════════════════════════════════════"

L0_OLD="${DATA_DIR}/checkpoint_L0_survivors.npy"
if [ -f "$L0_OLD" ]; then
    echo "──────────────────────────────────────────────────"
    echo "  Level 1: L0→L1  (d_parent=4, d_child=8, m=20)"
    echo "──────────────────────────────────────────────────"
    "$PROVER" "$L0_OLD" "/tmp/bench_old_L1.npy" \
        --d_parent 4 --m 20 --c_target 1.4 \
        --max_survivors 500000
    echo ""
fi

L2_OLD="${DATA_DIR}/checkpoint_L2_survivors.npy"
if [ -f "$L2_OLD" ]; then
    # Use first 1000 parents for a quick test (avoid running the full 7.5M)
    echo "──────────────────────────────────────────────────"
    echo "  Level 3 (subset): L2→L3  (d_parent=16, d_child=32, m=20)"
    echo "  Using first 1000 parents only for benchmark"
    echo "──────────────────────────────────────────────────"
    python3 -c "
import numpy as np
p = np.load('$L2_OLD')
np.save('/tmp/bench_L2_subset.npy', p[:1000])
print(f'  Subset: {p[:1000].shape[0]} parents from {p.shape[0]} total')
" 2>/dev/null || echo "  (python3 not available for subsetting, skipping)"

    if [ -f "/tmp/bench_L2_subset.npy" ]; then
        "$PROVER" "/tmp/bench_L2_subset.npy" "/tmp/bench_old_L3.npy" \
            --d_parent 16 --m 20 --c_target 1.4 \
            --max_survivors 10000000
    fi
    echo ""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo "═══════════════════════════════════════════════════════════════"

# Cleanup temp files
rm -f /tmp/bench_*.npy

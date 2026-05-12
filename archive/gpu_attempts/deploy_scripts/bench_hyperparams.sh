#!/bin/bash
# bench_hyperparams.sh — Benchmark the optimized cascade kernel at L3→L4.
#
# Tests throughput with different numbers of parents to measure:
#   1. Kernel startup overhead vs steady-state throughput
#   2. Subtree pruning effectiveness (via skip counts in trace output)
#   3. Quick-check hit rate
#   4. Overall parents/second at d_child=64
#
# Usage:
#   ./bench_hyperparams.sh          # full benchmark suite
#   ./bench_hyperparams.sh quick    # small subset (1-2 min)
#
# Prerequisites:
#   - cascade_prover binary built (./build.sh release)
#   - checkpoint_L2_survivors.npy in ../data/ (for L2→L3 warmup)
#   - checkpoint_L3_survivors.npy in ../data/ (for L3→L4 benchmark)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
PROVER="${SCRIPT_DIR}/cascade_prover"
RESULTS_DIR="${DATA_DIR}/bench_results"

mkdir -p "$RESULTS_DIR"

if [ ! -x "$PROVER" ]; then
    echo "ERROR: $PROVER not found. Run ./build.sh first."
    exit 1
fi

MODE="${1:-full}"

M=20
C_TARGET=1.4
MAX_SURV=100000

echo "═══════════════════════════════════════════════════════"
echo "  Sidon Cascade GPU Prover — Hyperparameter Benchmark"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════════════════"
echo ""

# --- GPU info ---
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# --- Helper: extract a subset of parents ---
extract_parents() {
    local input="$1"
    local output="$2"
    local count="$3"
    python3 -c "
import numpy as np
p = np.load('$input')
np.save('$output', p[:$count])
print(f'  Extracted {min($count, len(p))} of {len(p)} parents')
"
}

# --- Helper: run one benchmark and extract timing ---
run_bench() {
    local label="$1"
    local parents="$2"
    local d_parent="$3"
    local max_surv="$4"
    local logfile="${RESULTS_DIR}/${label}.log"

    echo "--- $label ---"
    echo "  Parents: $parents"
    echo "  d_parent=$d_parent  m=$M  c_target=$C_TARGET"

    # Time the run
    local t_start=$(date +%s%N)
    "$PROVER" "$parents" "${RESULTS_DIR}/${label}_output.npy" \
        --d_parent "$d_parent" --m "$M" --c_target "$C_TARGET" \
        --max_survivors "$max_surv" 2>&1 | tee "$logfile"
    local t_end=$(date +%s%N)

    local elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    echo "  Wall time: ${elapsed_ms}ms"
    echo ""

    # Clean up output file (we only care about timing)
    rm -f "${RESULTS_DIR}/${label}_output.npy"
}

# ═══════════════════════════════════════════════════════════
# SECTION 1: L2→L3 (d_child=32) — baseline/warmup
# ═══════════════════════════════════════════════════════════

L2_PARENTS="${DATA_DIR}/checkpoint_L2_survivors.npy"
if [ -f "$L2_PARENTS" ]; then
    echo "=== SECTION 1: L2→L3 (d_child=32) ==="
    echo ""

    extract_parents "$L2_PARENTS" "${RESULTS_DIR}/l2_1k.npy" 1000
    run_bench "L2_L3_1k" "${RESULTS_DIR}/l2_1k.npy" 16 "$MAX_SURV"

    if [ "$MODE" = "full" ]; then
        extract_parents "$L2_PARENTS" "${RESULTS_DIR}/l2_10k.npy" 10000
        run_bench "L2_L3_10k" "${RESULTS_DIR}/l2_10k.npy" 16 "$MAX_SURV"
    fi

    rm -f "${RESULTS_DIR}/l2_1k.npy" "${RESULTS_DIR}/l2_10k.npy"
else
    echo "SKIP: L2→L3 (no checkpoint_L2_survivors.npy)"
    echo ""
fi

# ═══════════════════════════════════════════════════════════
# SECTION 2: L3→L4 (d_child=64) — the critical benchmark
# ═══════════════════════════════════════════════════════════

L3_PARENTS="${DATA_DIR}/checkpoint_L3_survivors.npy"
if [ -f "$L3_PARENTS" ]; then
    echo "=== SECTION 2: L3→L4 (d_child=64) ==="
    echo ""

    # Small runs to measure startup + per-parent cost
    for N in 10 100 1000; do
        extract_parents "$L3_PARENTS" "${RESULTS_DIR}/l3_${N}.npy" "$N"
        run_bench "L3_L4_${N}" "${RESULTS_DIR}/l3_${N}.npy" 32 "$MAX_SURV"
        rm -f "${RESULTS_DIR}/l3_${N}.npy"
    done

    if [ "$MODE" = "full" ]; then
        for N in 5000 10000; do
            extract_parents "$L3_PARENTS" "${RESULTS_DIR}/l3_${N}.npy" "$N"
            run_bench "L3_L4_${N}" "${RESULTS_DIR}/l3_${N}.npy" 32 "$MAX_SURV"
            rm -f "${RESULTS_DIR}/l3_${N}.npy"
        done
    fi
else
    echo "SKIP: L3→L4 (no checkpoint_L3_survivors.npy)"
    echo ""
fi

# ═══════════════════════════════════════════════════════════
# SECTION 3: Build with TRACE to measure subtree pruning
# ═══════════════════════════════════════════════════════════

echo "=== SECTION 3: Subtree pruning analysis (TRACE build) ==="
echo ""

# Build trace version
echo "Building TRACE version..."
cd "$SCRIPT_DIR"
./build.sh trace 2>&1 | tail -3
PROVER_TRACE="${SCRIPT_DIR}/cascade_prover"

if [ -f "$L3_PARENTS" ]; then
    # Run 10 parents with trace to see subtree pruning stats
    extract_parents "$L3_PARENTS" "${RESULTS_DIR}/l3_trace_10.npy" 10
    echo ""
    echo "--- L3→L4 TRACE (10 parents) ---"
    "$PROVER_TRACE" "${RESULTS_DIR}/l3_trace_10.npy" "${RESULTS_DIR}/trace_output.npy" \
        --d_parent 32 --m "$M" --c_target "$C_TARGET" \
        --max_survivors 10000 2>&1 | tee "${RESULTS_DIR}/trace_10.log"

    echo ""
    echo "=== Subtree pruning summary ==="
    grep -c "skipped=" "${RESULTS_DIR}/trace_10.log" || echo "  (no skip data)"
    grep "skipped=" "${RESULTS_DIR}/trace_10.log" | head -20

    rm -f "${RESULTS_DIR}/l3_trace_10.npy" "${RESULTS_DIR}/trace_output.npy"
fi

# Rebuild release version
echo ""
echo "Rebuilding RELEASE version..."
cd "$SCRIPT_DIR"
./build.sh release 2>&1 | tail -3

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo "  Results in: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Log files:"
ls -la "${RESULTS_DIR}"/*.log 2>/dev/null || echo "  (none)"

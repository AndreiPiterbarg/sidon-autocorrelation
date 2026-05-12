#!/bin/bash
# run_full_proof.sh — End-to-end GPU cascade proof.
#
# Generates L0 on CPU, then runs L1→L2→L3 on GPU.
# Stops early if any level produces 0 survivors (proof complete).
#
# Usage:
#   ./run_full_proof.sh                          # defaults: m=35, c_target=1.33
#   ./run_full_proof.sh --m 35 --c_target 1.33
#   ./run_full_proof.sh --m 20 --c_target 1.40 --max_level 4
#
# The proof is RIGOROUS if it converges at d_child <= m.
# For m=35: L1(d=8)✓ L2(d=16)✓ L3(d=32)✓ L4(d=64)✗

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Default parameters
M=35
C_TARGET=1.33
N_HALF=2
MAX_LEVEL=3    # L3 → d_child=32 ≤ m=35, rigorous
MAX_SURV_L1=500000
MAX_SURV_L2=10000000
MAX_SURV_L3=200000000
MAX_SURV_L4=500000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)        M="$2"; shift 2;;
        --c_target) C_TARGET="$2"; shift 2;;
        --n_half)   N_HALF="$2"; shift 2;;
        --max_level) MAX_LEVEL="$2"; shift 2;;
        --max_surv_l1) MAX_SURV_L1="$2"; shift 2;;
        --max_surv_l2) MAX_SURV_L2="$2"; shift 2;;
        --max_surv_l3) MAX_SURV_L3="$2"; shift 2;;
        *)          echo "Unknown arg: $1"; exit 1;;
    esac
done

PROVER="${SCRIPT_DIR}/cascade_prover"
if [ ! -x "$PROVER" ]; then
    echo "Error: $PROVER not found. Run ./build.sh first."
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  Sidon Cascade GPU Proof"
echo "  m=$M  c_target=$C_TARGET  n_half=$N_HALF  max_level=$MAX_LEVEL"
echo "  Rigorous if converges at d <= $M"
echo "══════════════════════════════════════════════════════════════"
echo ""

T_START=$(date +%s)

# ── Step 1: Generate L0 on CPU ──
echo "=== STEP 1: Generate L0 checkpoint (CPU) ==="
L0_FILE="${DATA_DIR}/checkpoint_L0_survivors.npy"

# Find python (RunPod images use python3 or python3.x)
PYTHON=""
for p in python3 python python3.13 python3.11 python3.10; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Error: No Python found."
    exit 1
fi
echo "  Using: $PYTHON"

$PYTHON "${SCRIPT_DIR}/generate_l0.py" \
    --m "$M" --c_target "$C_TARGET" --n_half "$N_HALF" \
    --output "$L0_FILE"

if [ ! -f "$L0_FILE" ]; then
    echo "Error: L0 generation failed."
    exit 1
fi
echo ""

# ── Step 2+: Run cascade levels on GPU ──
run_level() {
    local LEVEL=$1
    local D_PARENT=$2
    local PARENTS=$3
    local OUTPUT=$4
    local MAX_SURV=$5
    local D_CHILD=$((D_PARENT * 2))

    echo "=== STEP $((LEVEL+1)): Level $LEVEL (d_parent=$D_PARENT → d_child=$D_CHILD) ==="

    if [ ! -f "$PARENTS" ]; then
        echo "  Error: Parent file not found: $PARENTS"
        return 1
    fi

    # Check file is non-empty (has actual data rows)
    local FSIZE
    FSIZE=$(stat -c%s "$PARENTS" 2>/dev/null || stat -f%z "$PARENTS" 2>/dev/null)
    if [ "$FSIZE" -le 128 ]; then
        echo "  Parent file too small ($FSIZE bytes) — likely 0 parents."
        echo "  PROOF COMPLETE at previous level!"
        return 2
    fi

    "$PROVER" "$PARENTS" "$OUTPUT" \
        --d_parent "$D_PARENT" \
        --m "$M" \
        --c_target "$C_TARGET" \
        --max_survivors "$MAX_SURV"

    local RC=$?
    if [ $RC -ne 0 ]; then
        echo "  Kernel failed (exit code $RC)"
        return 1
    fi

    # Check if output exists and has survivors
    if [ ! -f "$OUTPUT" ]; then
        echo ""
        echo "  ╔═══════════════════════════════════════════════╗"
        if [ "$D_CHILD" -le "$M" ]; then
            echo "  ║  PROOF COMPLETE — 0 survivors at d=$D_CHILD    ║"
            echo "  ║  C_1a >= $C_TARGET  (RIGOROUS: d=$D_CHILD <= m=$M)  ║"
        else
            echo "  ║  0 survivors at d=$D_CHILD but d > m=$M          ║"
            echo "  ║  NOT RIGOROUS                                ║"
        fi
        echo "  ╚═══════════════════════════════════════════════╝"
        return 0
    fi

    echo "  Output: $OUTPUT"
    echo ""
    return 0
}

# Level 1: L0→L1 (d_parent=4, d_child=8)
if [ "$MAX_LEVEL" -ge 1 ]; then
    run_level 1 4 \
        "${DATA_DIR}/checkpoint_L0_survivors.npy" \
        "${DATA_DIR}/checkpoint_L1_survivors.npy" \
        "$MAX_SURV_L1"
    RC=$?
    if [ $RC -eq 2 ] || { [ $RC -eq 0 ] && [ ! -f "${DATA_DIR}/checkpoint_L1_survivors.npy" ]; }; then
        echo "Proof converged at L1 (d=8)."
        exit 0
    elif [ $RC -ne 0 ]; then
        exit 1
    fi
fi

# Level 2: L1→L2 (d_parent=8, d_child=16)
if [ "$MAX_LEVEL" -ge 2 ]; then
    run_level 2 8 \
        "${DATA_DIR}/checkpoint_L1_survivors.npy" \
        "${DATA_DIR}/checkpoint_L2_survivors.npy" \
        "$MAX_SURV_L2"
    RC=$?
    if [ $RC -eq 2 ] || { [ $RC -eq 0 ] && [ ! -f "${DATA_DIR}/checkpoint_L2_survivors.npy" ]; }; then
        echo "Proof converged at L2 (d=16)."
        exit 0
    elif [ $RC -ne 0 ]; then
        exit 1
    fi
fi

# Level 3: L2→L3 (d_parent=16, d_child=32)
if [ "$MAX_LEVEL" -ge 3 ]; then
    run_level 3 16 \
        "${DATA_DIR}/checkpoint_L2_survivors.npy" \
        "${DATA_DIR}/checkpoint_L3_survivors.npy" \
        "$MAX_SURV_L3"
    RC=$?
    if [ $RC -eq 2 ] || { [ $RC -eq 0 ] && [ ! -f "${DATA_DIR}/checkpoint_L3_survivors.npy" ]; }; then
        echo "Proof converged at L3 (d=32)."
        exit 0
    elif [ $RC -ne 0 ]; then
        exit 1
    fi
fi

# Level 4: L3→L4 (d_parent=32, d_child=64) — NOT rigorous for m=35
if [ "$MAX_LEVEL" -ge 4 ]; then
    echo ""
    echo "  WARNING: L4 (d=64) exceeds m=$M — proof would NOT be rigorous."
    echo ""
    run_level 4 32 \
        "${DATA_DIR}/checkpoint_L3_survivors.npy" \
        "${DATA_DIR}/checkpoint_L4_survivors.npy" \
        "$MAX_SURV_L4"
    RC=$?
    if [ $RC -eq 2 ] || { [ $RC -eq 0 ] && [ ! -f "${DATA_DIR}/checkpoint_L4_survivors.npy" ]; }; then
        echo "Proof converged at L4 (d=64) — but NOT rigorous (d > m=$M)."
        exit 0
    elif [ $RC -ne 0 ]; then
        exit 1
    fi
fi

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Total wall time: ${ELAPSED}s"
echo "  Cascade did NOT converge within $MAX_LEVEL levels."
echo "══════════════════════════════════════════════════════════════"

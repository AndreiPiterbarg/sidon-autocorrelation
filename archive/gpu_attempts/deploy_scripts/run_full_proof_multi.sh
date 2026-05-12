#!/bin/bash
# run_full_proof_multi.sh — End-to-end GPU cascade proof on multi-GPU cluster.
#
# Generates L0 on CPU, runs L1-L3 on GPU with chunked checkpointing.
# 4 GPUs process chunks in parallel. Crash-safe: restart resumes from last chunk.
#
# Usage:
#   ./run_full_proof_multi.sh
#   ./run_full_proof_multi.sh --m 35 --c_target 1.33 --ngpus 4
#   ./run_full_proof_multi.sh --m 35 --c_target 1.33 --ngpus 8 --chunk_size 2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"

# Default parameters
M=35
C_TARGET=1.33
N_HALF=2
MAX_LEVEL=3
NGPUS=4
# Chunk sizes per level (smaller = more checkpoints = safer)
# On 8x H100 SXM at ~25 parents/s, 500 parents ≈ 20s per chunk.
# Spot preemption loses at most 1 chunk per GPU = ~20s of work.
CHUNK_SIZE_L1=5000    # L1 is fast, bigger chunks fine
CHUNK_SIZE_L2=1000    # L2 is moderate
CHUNK_SIZE_L3=500     # L3 is slow, tiny chunks = minimal loss on preemption
MAX_SURV=5000000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)          M="$2"; shift 2;;
        --c_target)   C_TARGET="$2"; shift 2;;
        --n_half)     N_HALF="$2"; shift 2;;
        --max_level)  MAX_LEVEL="$2"; shift 2;;
        --ngpus)      NGPUS="$2"; shift 2;;
        --chunk_size) CHUNK_SIZE_L1="$2"; CHUNK_SIZE_L2="$2"; CHUNK_SIZE_L3="$2"; shift 2;;
        --data_dir)   DATA_DIR="$2"; shift 2;;
        *)            echo "Unknown arg: $1"; exit 1;;
    esac
done

PROVER="${SCRIPT_DIR}/cascade_prover"
if [ ! -x "$PROVER" ]; then
    echo "Error: $PROVER not found. Run ./build.sh first."
    exit 1
fi

# Find python
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

mkdir -p "$DATA_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  Sidon Cascade GPU Proof (Multi-GPU)"
echo "  m=$M  c_target=$C_TARGET  n_half=$N_HALF  max_level=$MAX_LEVEL"
echo "  GPUs: $NGPUS"
echo "  Rigorous if converges at d <= $M"
echo "══════════════════════════════════════════════════════════════"
echo ""

T_GLOBAL=$(date +%s)

# ── Step 1: Generate L0 on CPU ──
L0_FILE="${DATA_DIR}/checkpoint_L0_survivors.npy"
if [ -f "$L0_FILE" ]; then
    echo "=== STEP 1: L0 checkpoint exists, skipping ==="
    $PYTHON -c "import numpy as np; d=np.load('$L0_FILE'); print(f'  L0: {len(d)} survivors (shape {d.shape})')"
else
    echo "=== STEP 1: Generate L0 checkpoint (CPU) ==="
    $PYTHON "${SCRIPT_DIR}/generate_l0.py" \
        --m "$M" --c_target "$C_TARGET" --n_half "$N_HALF" \
        --output "$L0_FILE"
fi
echo ""

# Function to run one level with multi-GPU chunking
run_level_multi() {
    local LEVEL=$1
    local D_PARENT=$2
    local INPUT=$3
    local OUTPUT=$4
    local CHUNK_SIZE=$5
    local D_CHILD=$((D_PARENT * 2))
    local LEVEL_DIR="${DATA_DIR}/L${LEVEL}_run"

    echo "=== Level $LEVEL: d_parent=$D_PARENT → d_child=$D_CHILD ==="

    # Check if final merged output already exists
    if [ -f "$OUTPUT" ]; then
        echo "  Output already exists: $OUTPUT — skipping"
        $PYTHON -c "import numpy as np; d=np.load('$OUTPUT'); print(f'  {len(d)} survivors')"
        return 0
    fi

    if [ ! -f "$INPUT" ]; then
        echo "  Input not found: $INPUT"
        return 1
    fi

    # Check input size
    local N_PARENTS
    N_PARENTS=$($PYTHON -c "import numpy as np; print(len(np.load('$INPUT')))")
    if [ "$N_PARENTS" -eq 0 ]; then
        echo "  0 parents — proof COMPLETE at previous level!"
        return 2
    fi
    echo "  Parents: $N_PARENTS"

    # Run multi-GPU
    chmod +x "${SCRIPT_DIR}/run_multi_gpu.sh"
    "${SCRIPT_DIR}/run_multi_gpu.sh" "$INPUT" "$LEVEL_DIR" \
        --d_parent "$D_PARENT" \
        --m "$M" \
        --c_target "$C_TARGET" \
        --ngpus "$NGPUS" \
        --chunk_size "$CHUNK_SIZE" \
        --max_surv "$MAX_SURV"

    # Merge results
    echo ""
    echo "  Merging survivors..."
    $PYTHON "${SCRIPT_DIR}/merge_survivors.py" "$LEVEL_DIR" --output "$OUTPUT"

    # Check result
    if [ -f "$OUTPUT" ]; then
        local N_SURV
        N_SURV=$($PYTHON -c "import numpy as np; print(len(np.load('$OUTPUT')))")
        if [ "$N_SURV" -eq 0 ]; then
            echo ""
            echo "  ╔═══════════════════════════════════════════════╗"
            if [ "$D_CHILD" -le "$M" ]; then
                echo "  ║  PROOF COMPLETE — 0 survivors at d=$D_CHILD     ║"
                echo "  ║  C_1a >= $C_TARGET  (RIGOROUS: d=$D_CHILD <= m=$M)   ║"
            else
                echo "  ║  0 survivors at d=$D_CHILD but d > m=$M           ║"
                echo "  ║  NOT RIGOROUS                                 ║"
            fi
            echo "  ╚═══════════════════════════════════════════════╝"
            return 0
        fi
        echo "  L$LEVEL survivors: $N_SURV"
    else
        echo "  No output file — 0 survivors!"
        return 0
    fi
    echo ""
    return 0
}

# ── Run levels ──
if [ "$MAX_LEVEL" -ge 1 ]; then
    run_level_multi 1 4 \
        "${DATA_DIR}/checkpoint_L0_survivors.npy" \
        "${DATA_DIR}/checkpoint_L1_survivors.npy" \
        "$CHUNK_SIZE_L1"
    RC=$?
    if [ $RC -eq 2 ]; then exit 0; fi
    if [ $RC -ne 0 ]; then exit 1; fi

    N=$($PYTHON -c "import numpy as np; print(len(np.load('${DATA_DIR}/checkpoint_L1_survivors.npy')))" 2>/dev/null || echo "0")
    if [ "$N" -eq 0 ]; then
        echo "Proof converged at L1 (d=8)."
        exit 0
    fi
fi

if [ "$MAX_LEVEL" -ge 2 ]; then
    run_level_multi 2 8 \
        "${DATA_DIR}/checkpoint_L1_survivors.npy" \
        "${DATA_DIR}/checkpoint_L2_survivors.npy" \
        "$CHUNK_SIZE_L2"
    RC=$?
    if [ $RC -eq 2 ]; then exit 0; fi
    if [ $RC -ne 0 ]; then exit 1; fi

    N=$($PYTHON -c "import numpy as np; print(len(np.load('${DATA_DIR}/checkpoint_L2_survivors.npy')))" 2>/dev/null || echo "0")
    if [ "$N" -eq 0 ]; then
        echo "Proof converged at L2 (d=16)."
        exit 0
    fi
fi

if [ "$MAX_LEVEL" -ge 3 ]; then
    run_level_multi 3 16 \
        "${DATA_DIR}/checkpoint_L2_survivors.npy" \
        "${DATA_DIR}/checkpoint_L3_survivors.npy" \
        "$CHUNK_SIZE_L3"
    RC=$?
    if [ $RC -eq 2 ]; then exit 0; fi
    if [ $RC -ne 0 ]; then exit 1; fi

    N=$($PYTHON -c "import numpy as np; print(len(np.load('${DATA_DIR}/checkpoint_L3_survivors.npy')))" 2>/dev/null || echo "0")
    if [ "$N" -eq 0 ]; then
        echo "Proof converged at L3 (d=32)."
        exit 0
    fi
fi

T_END=$(date +%s)
ELAPSED=$((T_END - T_GLOBAL))
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Total wall time: ${ELAPSED}s ($((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m)"
echo "  Cascade did NOT converge within $MAX_LEVEL levels."
echo "══════════════════════════════════════════════════════════════"

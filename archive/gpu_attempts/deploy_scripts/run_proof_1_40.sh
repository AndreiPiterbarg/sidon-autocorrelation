#!/bin/bash
# run_proof_1_40.sh — Prove C_1a >= 1.40 using n_half=3, m=20 on GPU.
#
# Spot-instance safe: checkpoints after every chunk.
# Restart-safe: automatically resumes from last completed chunk.
#
# Cascade structure (n_half=3, m=20):
#   L0: d=6,  53130 compositions → ~5692 survivors    (CPU, <1s)
#   L1: d=12, d_parent=6  → d_child=12,  ~5.7K parents  (GPU, fast)
#   L2: d=24, d_parent=12 → d_child=24,  ~1.14M parents (GPU, ~1h)
#   L3: d=48, d_parent=24 → d_child=48,  ~86.67M parents (GPU, heavy)
#   L4: d=96, d_parent=48 → d_child=96,  ~57.01M parents (GPU, heavy)
#   L5: d=192, d_parent=96 → d_child=192 (insurance if L4 has survivors)
#
# Usage:
#   ./run_proof_1_40.sh                  # defaults
#   ./run_proof_1_40.sh --data_dir /persistent/sidon
#   ./run_proof_1_40.sh --ngpus 1        # single B200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"

# Proof parameters — DO NOT CHANGE (these define the proof)
M=20
C_TARGET=1.40
N_HALF=3

# Deployment parameters
NGPUS=1
MAX_LEVEL=5      # L4 expected to converge; L5 as insurance

# Chunk sizes: smaller = safer on spot, but more overhead.
# At ~1000 parents/sec on B200, 500 parents ≈ 0.5s per chunk.
CHUNK_SIZE_L1=5000     # L1 is tiny
CHUNK_SIZE_L2=5000     # L2 is moderate (~228 chunks at 1.14M parents)
CHUNK_SIZE_L3=1000     # L3 is heavy (~86.7K chunks at 86.67M parents)
CHUNK_SIZE_L4=1000     # L4 is heavy (~57K chunks at 57.01M parents)
CHUNK_SIZE_L5=500      # L5 insurance — d_child=384, very aggressive pruning
MAX_SURV=10000000      # Max survivors per chunk

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpus)      NGPUS="$2"; shift 2;;
        --data_dir)   DATA_DIR="$2"; shift 2;;
        --chunk_l3)   CHUNK_SIZE_L3="$2"; shift 2;;
        --chunk_l4)   CHUNK_SIZE_L4="$2"; shift 2;;
        --max_surv)   MAX_SURV="$2"; shift 2;;
        *)            echo "Unknown arg: $1"; exit 1;;
    esac
done

PROVER="${SCRIPT_DIR}/cascade_prover"
if [ ! -x "$PROVER" ]; then
    echo "Binary not found. Building..."
    cd "$SCRIPT_DIR"
    chmod +x build.sh
    ./build.sh release
    cd -
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
echo "  Sidon Cascade GPU Proof: C_1a >= 1.40"
echo "  Parameters: n_half=$N_HALF  m=$M  c_target=$C_TARGET"
echo "  GPU count: $NGPUS"
echo "  Data dir:  $DATA_DIR"
echo "  Levels:    L0(d=6) → L1(d=12) → L2(d=24) → L3(d=48) → L4(d=96) → L5(d=192)"
echo "══════════════════════════════════════════════════════════════"
echo ""

T_GLOBAL=$(date +%s)

# ── Step 1: Generate L0 on CPU ──
L0_FILE="${DATA_DIR}/checkpoint_L0_survivors.npy"
if [ -f "$L0_FILE" ]; then
    echo "=== L0: checkpoint exists, skipping ==="
    $PYTHON -c "import numpy as np; d=np.load('$L0_FILE'); print(f'  L0: {len(d)} survivors (shape {d.shape})')"
else
    echo "=== L0: Generating on CPU (n_half=$N_HALF, m=$M, c_target=$C_TARGET) ==="
    $PYTHON "${SCRIPT_DIR}/generate_l0.py" \
        --m "$M" --c_target "$C_TARGET" --n_half "$N_HALF" \
        --output "$L0_FILE"
fi
echo ""

# ── Generic level runner ──
run_gpu_level() {
    local LEVEL=$1
    local D_PARENT=$2
    local INPUT=$3
    local OUTPUT=$4
    local CHUNK_SIZE=$5
    local D_CHILD=$((D_PARENT * 2))
    local LEVEL_DIR="${DATA_DIR}/L${LEVEL}_run"

    echo "=== Level $LEVEL: d_parent=$D_PARENT → d_child=$D_CHILD ==="

    # Check if already complete
    if [ -f "$OUTPUT" ]; then
        local N_SURV
        N_SURV=$($PYTHON -c "import numpy as np; d=np.load('$OUTPUT'); print(len(d))")
        echo "  Output exists: $N_SURV survivors — skipping"
        if [ "$N_SURV" -eq 0 ]; then
            echo ""
            echo "  ╔══════════════════════════════════════════════════════╗"
            echo "  ║  PROOF COMPLETE — 0 survivors at d=$D_CHILD            ║"
            echo "  ║  C_1a >= $C_TARGET  (RIGOROUS: d=$D_CHILD <= m=$M)         ║"
            echo "  ╚══════════════════════════════════════════════════════╝"
            return 2
        fi
        echo ""
        return 0
    fi

    if [ ! -f "$INPUT" ]; then
        echo "  Input not found: $INPUT"
        return 1
    fi

    local N_PARENTS
    N_PARENTS=$($PYTHON -c "import numpy as np; print(len(np.load('$INPUT')))")
    if [ "$N_PARENTS" -eq 0 ]; then
        echo "  0 parents — proof COMPLETE at previous level!"
        # Save empty checkpoint
        $PYTHON -c "import numpy as np; np.save('$OUTPUT', np.empty((0, $D_CHILD), dtype=np.int32))"
        return 2
    fi
    echo "  Parents: $N_PARENTS  (chunks of $CHUNK_SIZE)"

    # Run multi-GPU with chunking
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
        echo "  L$LEVEL survivors: $N_SURV"
        if [ "$N_SURV" -eq 0 ]; then
            echo ""
            echo "  ╔══════════════════════════════════════════════════════╗"
            if [ "$D_CHILD" -le "$M" ]; then
                echo "  ║  PROOF COMPLETE — 0 survivors at d=$D_CHILD            ║"
                echo "  ║  C_1a >= $C_TARGET  (RIGOROUS)                         ║"
            else
                echo "  ║  0 survivors at d=$D_CHILD (d > m=$M)                  ║"
                echo "  ║  NOT YET RIGOROUS — see note below                 ║"
            fi
            echo "  ╚══════════════════════════════════════════════════════╝"
            return 2
        fi
    else
        echo "  No output file — 0 survivors!"
        $PYTHON -c "import numpy as np; np.save('$OUTPUT', np.empty((0, $D_CHILD), dtype=np.int32))"
        return 2
    fi
    echo ""
    return 0
}

# ── Run L1: d_parent=6, d_child=12 ──
run_gpu_level 1 6 \
    "${DATA_DIR}/checkpoint_L0_survivors.npy" \
    "${DATA_DIR}/checkpoint_L1_survivors.npy" \
    "$CHUNK_SIZE_L1"
RC=$?
if [ $RC -eq 2 ]; then
    echo "Proof converged at L1."
    exit 0
fi
if [ $RC -ne 0 ]; then exit 1; fi

# ── Run L2: d_parent=12, d_child=24 ──
run_gpu_level 2 12 \
    "${DATA_DIR}/checkpoint_L1_survivors.npy" \
    "${DATA_DIR}/checkpoint_L2_survivors.npy" \
    "$CHUNK_SIZE_L2"
RC=$?
if [ $RC -eq 2 ]; then
    echo "Proof converged at L2."
    exit 0
fi
if [ $RC -ne 0 ]; then exit 1; fi

# ── Run L3: d_parent=24, d_child=48 ──
run_gpu_level 3 24 \
    "${DATA_DIR}/checkpoint_L2_survivors.npy" \
    "${DATA_DIR}/checkpoint_L3_survivors.npy" \
    "$CHUNK_SIZE_L3"
RC=$?
if [ $RC -eq 2 ]; then
    echo "Proof converged at L3."
    exit 0
fi
if [ $RC -ne 0 ]; then exit 1; fi

# ── Run L4: d_parent=48, d_child=96 ──
# ── Run L4: d_parent=48, d_child=96 ──
run_gpu_level 4 48 \
    "${DATA_DIR}/checkpoint_L3_survivors.npy" \
    "${DATA_DIR}/checkpoint_L4_survivors.npy" \
    "$CHUNK_SIZE_L4"
RC=$?
if [ $RC -eq 2 ]; then
    echo "Proof converged at L4 (d=96)."
    T_END=$(date +%s)
    ELAPSED=$((T_END - T_GLOBAL))
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  PROVEN: C_1a >= 1.40  (total ${ELAPSED}s)"
    echo "══════════════════════════════════════════════════════════════"
    exit 0
fi
if [ $RC -ne 0 ]; then exit 1; fi

# ── Run L5 (insurance): d_parent=96, d_child=192 ──
run_gpu_level 5 96 \
    "${DATA_DIR}/checkpoint_L4_survivors.npy" \
    "${DATA_DIR}/checkpoint_L5_survivors.npy" \
    "$CHUNK_SIZE_L5"
RC=$?
if [ $RC -eq 2 ]; then
    echo "Proof converged at L5 (d=192)."
    T_END=$(date +%s)
    ELAPSED=$((T_END - T_GLOBAL))
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  PROVEN: C_1a >= 1.40  (total ${ELAPSED}s)"
    echo "══════════════════════════════════════════════════════════════"
    exit 0
fi

T_END=$(date +%s)
ELAPSED=$((T_END - T_GLOBAL))
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Total wall time: ${ELAPSED}s ($((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m)"
echo "══════════════════════════════════════════════════════════════"

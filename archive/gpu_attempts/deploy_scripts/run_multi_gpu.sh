#!/bin/bash
# run_multi_gpu.sh — Run one cascade level across multiple GPUs with checkpointing.
#
# Splits parents into small chunks, distributes across GPUs, saves after each chunk.
# If anything crashes, only the current chunk per GPU is lost — restart picks up
# where it left off by skipping chunks that already have output files.
#
# Usage:
#   ./run_multi_gpu.sh <parents.npy> <output_dir> --d_parent 16 --m 35 --c_target 1.33
#   ./run_multi_gpu.sh <parents.npy> <output_dir> --d_parent 16 --m 35 --c_target 1.33 --ngpus 4 --chunk_size 5000
#
# After completion, merge with:
#   python merge_survivors.py <output_dir> --output checkpoint_L3_survivors.npy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVER="${SCRIPT_DIR}/cascade_prover"

if [ ! -x "$PROVER" ]; then
    echo "Error: $PROVER not found. Run ./build.sh first."
    exit 1
fi

# Defaults
NGPUS=4
CHUNK_SIZE=5000
MAX_SURV_PER_CHUNK=5000000

# Parse args
PARENTS=""
OUTPUT_DIR=""
D_PARENT=""
M=""
C_TARGET=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --d_parent)   D_PARENT="$2"; shift 2;;
        --m)          M="$2"; shift 2;;
        --c_target)   C_TARGET="$2"; shift 2;;
        --ngpus)      NGPUS="$2"; shift 2;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2;;
        --max_surv)   MAX_SURV_PER_CHUNK="$2"; shift 2;;
        *)            POSITIONAL+=("$1"); shift;;
    esac
done

PARENTS="${POSITIONAL[0]:?Usage: $0 <parents.npy> <output_dir> --d_parent D --m M --c_target C}"
OUTPUT_DIR="${POSITIONAL[1]:?Usage: $0 <parents.npy> <output_dir> --d_parent D --m M --c_target C}"

if [ -z "$D_PARENT" ] || [ -z "$M" ] || [ -z "$C_TARGET" ]; then
    echo "Error: --d_parent, --m, and --c_target are required"
    exit 1
fi

# Detect available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: requested $NGPUS GPUs but only $AVAILABLE_GPUS available. Using $AVAILABLE_GPUS."
    NGPUS=$AVAILABLE_GPUS
fi

mkdir -p "$OUTPUT_DIR"
CHUNKS_DIR="${OUTPUT_DIR}/chunks"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$CHUNKS_DIR" "$LOGS_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  Multi-GPU Cascade Prover"
echo "  Parents:    $PARENTS"
echo "  Output:     $OUTPUT_DIR"
echo "  d_parent=$D_PARENT  m=$M  c_target=$C_TARGET"
echo "  GPUs: $NGPUS  chunk_size: $CHUNK_SIZE"
echo "══════════════════════════════════════════════════════════════"
echo ""

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

# Step 1: Split parents into chunks
echo "=== Splitting parents into chunks of $CHUNK_SIZE ==="
$PYTHON "${SCRIPT_DIR}/split_parents.py" "$PARENTS" \
    --chunk_size "$CHUNK_SIZE" \
    --output_dir "$CHUNKS_DIR"

# Count chunks
NCHUNKS=$(ls "$CHUNKS_DIR"/chunk_*.npy 2>/dev/null | wc -l)
echo ""
echo "Total chunks: $NCHUNKS across $NGPUS GPUs"
echo ""

# Step 2: Process chunks across GPUs
# Each GPU gets a worker that pulls from a shared work queue (filesystem-based).
# A chunk is "done" if its output file exists.

T_START=$(date +%s)

process_gpu() {
    local GPU_ID=$1
    local WORKER_ID=$2

    for CHUNK_FILE in "$CHUNKS_DIR"/chunk_*.npy; do
        CHUNK_NAME=$(basename "$CHUNK_FILE" .npy)
        OUTPUT_FILE="${OUTPUT_DIR}/output_${CHUNK_NAME}.npy"
        LOCK_FILE="${OUTPUT_DIR}/.lock_${CHUNK_NAME}"
        LOG_FILE="${LOGS_DIR}/${CHUNK_NAME}_gpu${GPU_ID}.log"

        # Skip if already done
        if [ -f "$OUTPUT_FILE" ] || [ -f "${OUTPUT_FILE%.npy}_empty.marker" ]; then
            continue
        fi

        # Atomic claim via mkdir (atomic on all filesystems)
        if ! mkdir "$LOCK_FILE" 2>/dev/null; then
            continue  # another GPU claimed it
        fi

        echo "[GPU $GPU_ID] Processing $CHUNK_NAME..."

        CUDA_VISIBLE_DEVICES=$GPU_ID "$PROVER" \
            "$CHUNK_FILE" "$OUTPUT_FILE" \
            --d_parent "$D_PARENT" \
            --m "$M" \
            --c_target "$C_TARGET" \
            --max_survivors "$MAX_SURV_PER_CHUNK" \
            > "$LOG_FILE" 2>&1

        RC=$?

        if [ $RC -ne 0 ]; then
            echo "[GPU $GPU_ID] FAILED on $CHUNK_NAME (exit code $RC). See $LOG_FILE"
            rmdir "$LOCK_FILE" 2>/dev/null
            continue
        fi

        # If prover produced no output file, it means 0 survivors — mark as done
        if [ ! -f "$OUTPUT_FILE" ]; then
            touch "${OUTPUT_FILE%.npy}_empty.marker"
        fi

        echo "[GPU $GPU_ID] Done $CHUNK_NAME"
    done
}

echo "=== Cleaning stale locks (from prior preemption) ==="
find "$OUTPUT_DIR" -name ".lock_*" -type d -exec rmdir {} \; 2>/dev/null || true
echo "  Stale locks cleared."

echo "=== Launching $NGPUS GPU workers ==="
PIDS=()
for ((i=0; i<NGPUS; i++)); do
    process_gpu "$i" "$i" &
    PIDS+=($!)
    echo "  Started worker on GPU $i (PID ${PIDS[$i]})"
done

# Wait for all workers with progress reporting
echo ""
echo "Processing... (progress updates every 30s)"
while true; do
    # Check if all workers are done
    ALL_DONE=true
    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            ALL_DONE=false
            break
        fi
    done

    DONE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 \( -name 'output_chunk_*.npy' -o -name '*_empty.marker' \) 2>/dev/null | wc -l)
    ELAPSED=$(($(date +%s) - T_START))
    PCT=$((DONE_COUNT * 100 / NCHUNKS))
    if [ "$ELAPSED" -gt 0 ] && [ "$DONE_COUNT" -gt 0 ]; then
        RATE=$(echo "scale=2; $DONE_COUNT / $ELAPSED * 3600" | bc 2>/dev/null || echo "?")
        REMAINING=$(( (NCHUNKS - DONE_COUNT) * ELAPSED / DONE_COUNT ))
        ETA_H=$((REMAINING / 3600))
        ETA_M=$(( (REMAINING % 3600) / 60 ))
        printf "\r  [%d/%d] (%d%%) chunks done  %s chunks/hr  ETA %dh%02dm  [%ds elapsed]    " \
            "$DONE_COUNT" "$NCHUNKS" "$PCT" "$RATE" "$ETA_H" "$ETA_M" "$ELAPSED"
    else
        printf "\r  [%d/%d] (%d%%) chunks done  [%ds elapsed]    " \
            "$DONE_COUNT" "$NCHUNKS" "$PCT" "$ELAPSED"
    fi

    if $ALL_DONE; then
        echo ""
        break
    fi

    sleep 30
done

# Wait for all PIDs to fully exit
for PID in "${PIDS[@]}"; do
    wait "$PID" 2>/dev/null || true
done

T_END=$(date +%s)
TOTAL_ELAPSED=$((T_END - T_START))
DONE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 \( -name 'output_chunk_*.npy' -o -name '*_empty.marker' \) 2>/dev/null | wc -l)
FAILED=$((NCHUNKS - DONE_COUNT))

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Multi-GPU run complete"
echo "  Chunks: $DONE_COUNT/$NCHUNKS done, $FAILED failed"
echo "  Wall time: ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 ))m)"
echo "══════════════════════════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED chunks failed. Re-run this script to retry them."
    echo "  (Completed chunks are skipped automatically.)"
fi

echo ""
echo "Next: merge results with:"
echo "  $PYTHON ${SCRIPT_DIR}/merge_survivors.py $OUTPUT_DIR --output <checkpoint.npy>"

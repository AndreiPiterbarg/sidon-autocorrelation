#!/bin/bash
# spot_runner.sh — Spot-safe proof runner for RunPod.
#
# Designed for SPOT instances that can be preempted at any time.
# - Stores ALL data on /runpod-volume/ (survives preemption + restart)
# - Auto-resumes from last completed chunk on restart
# - Small chunks (500 parents) = lose at most ~20s of work per GPU on preempt
# - Logs to volume so progress is never lost
#
# SETUP (run once after pod creation):
#   bash gpu/spot_runner.sh setup
#
# RUN (run after setup, or on every pod restart):
#   bash gpu/spot_runner.sh run
#
# To make this auto-run on pod restart, set it as the RunPod Docker command:
#   bash -c "cd /workspace/compact_sidon && bash gpu/spot_runner.sh run"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ALL data goes to the persistent volume — never the container disk
VOLUME="/runpod-volume"
DATA_DIR="${VOLUME}/sidon_data"
LOG_FILE="${DATA_DIR}/proof_run.log"

# Proof parameters
M=35
C_TARGET=1.33
MAX_LEVEL=3

# Override from env or args
while [[ $# -gt 0 ]]; do
    case "$1" in
        setup|run) ACTION="$1"; shift;;
        --m)          M="$2"; shift 2;;
        --c_target)   C_TARGET="$2"; shift 2;;
        --max_level)  MAX_LEVEL="$2"; shift 2;;
        *)            shift;;
    esac
done
ACTION="${ACTION:-run}"

# Detect GPUs
NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NGPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

setup() {
    echo "=== SPOT RUNNER SETUP ==="

    # Ensure CUDA is on PATH
    for cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$cuda_dir/nvcc" ]; then
            export PATH="$cuda_dir:$PATH"
            break
        fi
    done

    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found. Need CUDA toolkit."
        exit 1
    fi

    mkdir -p "$DATA_DIR"

    # Install deps
    echo "Installing dependencies..."
    pip install -q numpy numba 2>&1 | tail -3

    # Build CUDA kernel
    echo "Building CUDA kernel..."
    cd "${PROJECT_ROOT}/gpu"
    chmod +x *.sh
    ./build.sh release

    echo ""
    echo "Setup complete. Run with: bash gpu/spot_runner.sh run"
    echo "Data directory: $DATA_DIR"
    echo "GPUs: $NGPUS"
}

run() {
    echo "=== SPOT RUNNER — m=$M c=$C_TARGET ngpus=$NGPUS ===" | tee -a "$LOG_FILE"
    echo "Data: $DATA_DIR" | tee -a "$LOG_FILE"
    echo "$(date): Starting/resuming proof" | tee -a "$LOG_FILE"

    # Ensure CUDA is on PATH
    for cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$cuda_dir/nvcc" ]; then
            export PATH="$cuda_dir:$PATH"
            break
        fi
    done

    # Verify binary exists
    PROVER="${PROJECT_ROOT}/gpu/cascade_prover"
    if [ ! -x "$PROVER" ]; then
        echo "Binary not found, building..."
        cd "${PROJECT_ROOT}/gpu"
        chmod +x *.sh
        ./build.sh release
    fi

    # Run the proof — DATA_DIR env var tells the script where to store everything
    cd "${PROJECT_ROOT}/gpu"
    chmod +x run_full_proof_multi.sh run_multi_gpu.sh
    export DATA_DIR
    bash run_full_proof_multi.sh \
        --m "$M" \
        --c_target "$C_TARGET" \
        --max_level "$MAX_LEVEL" \
        --ngpus "$NGPUS" \
        --data_dir "$DATA_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    echo ""
    echo "$(date): Proof run finished" | tee -a "$LOG_FILE"

    # Show results
    echo "" | tee -a "$LOG_FILE"
    echo "=== RESULTS ===" | tee -a "$LOG_FILE"
    for level in 0 1 2 3; do
        f="${DATA_DIR}/checkpoint_L${level}_survivors.npy"
        if [ -f "$f" ]; then
            python3 -c "import numpy as np; d=np.load('$f'); print(f'  L$level: {len(d)} survivors (d={d.shape[1]})')" | tee -a "$LOG_FILE"
        fi
    done
}

case "$ACTION" in
    setup) setup;;
    run)   run;;
    *)     echo "Usage: $0 {setup|run}"; exit 1;;
esac

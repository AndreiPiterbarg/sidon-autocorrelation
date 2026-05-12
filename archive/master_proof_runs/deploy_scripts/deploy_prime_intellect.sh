#!/bin/bash
# deploy_prime_intellect.sh — Deploy and run on Prime Intellect GPU cluster.
#
# Run this INSIDE the Prime Intellect SSH session.
# Sets up everything from scratch and runs the full proof.
#
# Usage:
#   bash deploy_prime_intellect.sh
#   bash deploy_prime_intellect.sh --ngpus 4 --m 35 --c_target 1.33
#
# Prerequisites:
#   - SSH into the Prime Intellect machine
#   - NVIDIA drivers + CUDA toolkit installed (nvidia-smi works)
#   - git, python3, pip available

set -euo pipefail

# Defaults
M=35
C_TARGET=1.33
MAX_LEVEL=3
NGPUS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)          M="$2"; shift 2;;
        --c_target)   C_TARGET="$2"; shift 2;;
        --max_level)  MAX_LEVEL="$2"; shift 2;;
        --ngpus)      NGPUS="$2"; shift 2;;
        *)            shift;;
    esac
done

WORKDIR="/workspace/compact_sidon"

echo "══════════════════════════════════════════════════════════════"
echo "  Prime Intellect GPU Deploy"
echo "  m=$M  c_target=$C_TARGET  max_level=$MAX_LEVEL  gpus=$NGPUS"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: System info ──
echo "=== SYSTEM INFO ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "  Total GPUs: $GPU_COUNT"
if [ "$NGPUS" -gt "$GPU_COUNT" ]; then
    NGPUS=$GPU_COUNT
    echo "  Adjusted NGPUS to $NGPUS"
fi
echo ""

# ── Step 2: Clone / update repo ──
echo "=== CODE SETUP ==="
mkdir -p /workspace
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo exists, pulling updates..."
    cd "$WORKDIR" && git pull && cd /workspace
else
    echo "  Cloning repository..."
    git clone https://github.com/AndreiPiterbarg/compact_sidon.git "$WORKDIR"
fi
cd "$WORKDIR"
echo "  Code at: $WORKDIR"
echo ""

# ── Step 3: Install dependencies ──
echo "=== DEPENDENCIES ==="
pip install -q numpy numba 2>&1 | tail -3
echo "  numpy + numba: OK"
echo ""

# ── Step 4: Build CUDA kernel ──
echo "=== BUILD CUDA KERNEL ==="
cd gpu
chmod +x build.sh run_multi_gpu.sh run_full_proof_multi.sh
./build.sh release
cd "$WORKDIR"
echo ""

# ── Step 5: Run full proof ──
echo "=== RUNNING FULL PROOF ==="
cd gpu
./run_full_proof_multi.sh \
    --m "$M" \
    --c_target "$C_TARGET" \
    --max_level "$MAX_LEVEL" \
    --ngpus "$NGPUS"

echo ""
echo "=== DEPLOY COMPLETE ==="
echo "Results in: $WORKDIR/data/"
ls -lh "$WORKDIR/data/"checkpoint_*.npy 2>/dev/null || echo "  (no checkpoint files)"

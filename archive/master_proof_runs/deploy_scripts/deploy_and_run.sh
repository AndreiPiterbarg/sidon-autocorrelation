#!/bin/bash
# deploy_and_run.sh — Run this INSIDE the RunPod GPU pod SSH session.
#
# Sets up the environment, builds the CUDA kernel, and runs the full
# cascade proof for c_target=1.33, m=35.
#
# Usage (from SSH into RunPod pod):
#   bash deploy_and_run.sh
#   bash deploy_and_run.sh --c_target 1.35
#   bash deploy_and_run.sh --m 35 --c_target 1.33 --max_level 3

set -euo pipefail

# Default proof parameters
M="${M:-35}"
C_TARGET="${C_TARGET:-1.33}"
MAX_LEVEL="${MAX_LEVEL:-3}"

# Parse arguments (override defaults)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)        M="$2"; shift 2;;
        --c_target) C_TARGET="$2"; shift 2;;
        --max_level) MAX_LEVEL="$2"; shift 2;;
        *)          shift;;  # ignore unknown args
    esac
done

WORKDIR="/workspace/sidon-autocorrelation"

echo "══════════════════════════════════════════════════════════════"
echo "  Sidon GPU Proof — Deploy & Run"
echo "  m=$M  c_target=$C_TARGET  max_level=$MAX_LEVEL"
echo "═══════════════════���══════════════════════════════════════════"
echo ""

# ── Step 1: Verify GPU ──
echo "=== GPU CHECK ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""

# ── Step 2: Verify code is synced ──
if [ ! -d "$WORKDIR/gpu" ]; then
    echo "Error: $WORKDIR/gpu not found."
    echo "Code must be synced first (gpupod start does this automatically)."
    echo "Or run: gpupod sync"
    exit 1
fi
echo "Code directory: $WORKDIR"

# ── Step 3: Install Python deps (for L0 generation) ──
echo ""
echo "=== INSTALLING DEPENDENCIES ==="
pip install -q numpy numba 2>&1 | tail -3
echo "  numpy + numba: OK"

# ── Step 4: Build CUDA kernel ──
echo ""
echo "=== BUILDING CUDA KERNEL ==="
cd "$WORKDIR/gpu"
chmod +x build.sh run_full_proof.sh
./build.sh release

# ── Step 5: Run the full proof ��─
echo ""
echo "=== RUNNING FULL CASCADE PROOF ==="
./run_full_proof.sh --m "$M" --c_target "$C_TARGET" --max_level "$MAX_LEVEL"

echo ""
echo "=== DEPLOY & RUN COMPLETE ==="

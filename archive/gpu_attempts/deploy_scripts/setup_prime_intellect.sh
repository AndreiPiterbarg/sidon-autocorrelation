#!/bin/bash
# setup_prime_intellect.sh — One-time setup for Prime Intellect B200 spot instance.
#
# This is a SPOT instance — can be preempted at any time.
# All data is saved to DATA_DIR after every chunk (default: ./data).
# On preemption: re-provision, re-run setup, then re-run the proof script.
# It will auto-resume from the last completed chunk.
#
# Usage:
#   # On the instance (after cloning/uploading the repo):
#   cd compact_sidon
#   bash gpu/setup_prime_intellect.sh
#   bash gpu/run_proof_1_40.sh --data_dir /persistent/sidon_data
#
# If there's a persistent volume (e.g., /persistent or /data):
#   export DATA_DIR=/persistent/sidon_data
#   bash gpu/setup_prime_intellect.sh
#   bash gpu/run_proof_1_40.sh --data_dir $DATA_DIR
#
# To monitor from another terminal:
#   watch -n 5 'ls -la $DATA_DIR/L*_run/output_chunk_*.npy 2>/dev/null | wc -l'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "══════════════════════════════════════════════════════════════"
echo "  Prime Intellect B200 Setup — Sidon C_1a >= 1.40 Proof"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Check GPU ──
echo "=== GPU Check ==="
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# ── 2. Check/Install CUDA toolkit ──
echo "=== CUDA Toolkit ==="
for cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -x "$cuda_dir/nvcc" ]; then
        export PATH="$cuda_dir:$PATH"
        break
    fi
done

if command -v nvcc &>/dev/null; then
    echo "Found: $(nvcc --version | grep release)"
else
    echo "nvcc not found. Installing CUDA toolkit..."
    # Most cloud GPU images have CUDA pre-installed.
    # If not, install minimal toolkit:
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq cuda-toolkit 2>/dev/null || {
            echo "Auto-install failed. Please install CUDA toolkit manually."
            echo "  See: https://developer.nvidia.com/cuda-downloads"
            exit 1
        }
    else
        echo "ERROR: Cannot auto-install CUDA. Please install manually."
        exit 1
    fi
fi
echo ""

# ── 3. Install Python dependencies ──
echo "=== Python Dependencies ==="
PYTHON=""
for p in python3 python python3.13 python3.11 python3.10; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python found."
    exit 1
fi
echo "Python: $($PYTHON --version)"
$PYTHON -m pip install -q numpy numba 2>&1 | tail -3
echo ""

# ── 4. Build GPU kernel ──
echo "=== Building GPU Kernel ==="
cd "$SCRIPT_DIR"
chmod +x build.sh
./build.sh release
cd "$PROJECT_ROOT"
echo ""

# ── 5. Quick verification ──
echo "=== Quick Verification ==="
# Generate L0 to verify the full pipeline works
L0_TEST="/tmp/sidon_test_l0.npy"
$PYTHON "${SCRIPT_DIR}/generate_l0.py" \
    --m 20 --c_target 1.40 --n_half 3 \
    --output "$L0_TEST"

N_L0=$($PYTHON -c "import numpy as np; print(len(np.load('$L0_TEST')))")
echo "L0 verification: $N_L0 survivors (expected ~5692)"

# Quick GPU test: run L1 on a tiny subset
TINY_INPUT="/tmp/sidon_test_tiny.npy"
TINY_OUTPUT="/tmp/sidon_test_output.npy"
$PYTHON -c "
import numpy as np
parents = np.load('$L0_TEST')
np.save('$TINY_INPUT', parents[:5])  # 5 parents
"

"${SCRIPT_DIR}/cascade_prover" \
    "$TINY_INPUT" "$TINY_OUTPUT" \
    --d_parent 6 --m 20 --c_target 1.40 --max_survivors 100000

if [ $? -eq 0 ]; then
    echo "GPU kernel test: PASSED"
    if [ -f "$TINY_OUTPUT" ]; then
        N=$($PYTHON -c "import numpy as np; print(len(np.load('$TINY_OUTPUT')))")
        echo "  $N survivors from 5 parents"
    else
        echo "  0 survivors from 5 parents"
    fi
else
    echo "GPU kernel test: FAILED"
    echo "Check build flags and GPU compatibility."
    exit 1
fi

# Cleanup
rm -f "$L0_TEST" "$TINY_INPUT" "$TINY_OUTPUT"
echo ""

# ── 6. Summary ──
echo "══════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo ""
echo "  To run the proof:"
echo "    bash gpu/run_proof_1_40.sh"
echo ""
echo "  For persistent storage (recommended on spot):"
echo "    bash gpu/run_proof_1_40.sh --data_dir /persistent/sidon_data"
echo ""
echo "  To run in background (survives SSH disconnect):"
echo "    nohup bash gpu/run_proof_1_40.sh --data_dir /persistent/sidon_data > proof.log 2>&1 &"
echo "    tail -f proof.log"
echo ""
echo "  On preemption: re-provision, clone repo, run setup, run proof."
echo "  It auto-resumes from the last completed chunk."
echo "══════════════════════════════════════════════════════════════"

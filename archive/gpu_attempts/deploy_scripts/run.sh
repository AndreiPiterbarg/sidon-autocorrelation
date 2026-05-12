#!/bin/bash
# run.sh — Run the Sidon cascade GPU prover.
#
# Usage:
#   ./run.sh <level>                    # e.g., ./run.sh 3  (runs L2→L3)
#   ./run.sh <level> [max_survivors]    # override survivor cap
#   ./run.sh custom <parents.npy> <output.npy> --d_parent D --m M --c_target C
#
# Environment variables (override defaults):
#   M=35 C_TARGET=1.33 ./run.sh 3
#
# Levels:
#   1: L0→L1  (d_parent=4,  d_child=8)
#   2: L1→L2  (d_parent=8,  d_child=16)
#   3: L2→L3  (d_parent=16, d_child=32)
#   4: L3→L4  (d_parent=32, d_child=64)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

# Ensure binary exists
PROVER="${SCRIPT_DIR}/cascade_prover"
if [ ! -x "$PROVER" ]; then
    echo "Error: $PROVER not found. Run ./build.sh first."
    exit 1
fi

# Parameters — override via environment variables
M="${M:-20}"
C_TARGET="${C_TARGET:-1.4}"

if [ "${1:-}" = "custom" ]; then
    shift
    exec "$PROVER" "$@"
fi

LEVEL="${1:?Usage: $0 <level> [max_survivors]}"

case "$LEVEL" in
    1)
        D_PARENT=4
        PARENTS="${DATA_DIR}/checkpoint_L0_survivors.npy"
        OUTPUT="${DATA_DIR}/checkpoint_L1_survivors.npy"
        MAX_SURV="${2:-500000}"
        ;;
    2)
        D_PARENT=8
        PARENTS="${DATA_DIR}/checkpoint_L1_survivors.npy"
        OUTPUT="${DATA_DIR}/checkpoint_L2_survivors.npy"
        MAX_SURV="${2:-10000000}"
        ;;
    3)
        D_PARENT=16
        PARENTS="${DATA_DIR}/checkpoint_L2_survivors.npy"
        OUTPUT="${DATA_DIR}/checkpoint_L3_survivors.npy"
        MAX_SURV="${2:-200000000}"
        ;;
    4)
        D_PARENT=32
        PARENTS="${DATA_DIR}/checkpoint_L3_survivors.npy"
        OUTPUT="${DATA_DIR}/checkpoint_L4_survivors.npy"
        MAX_SURV="${2:-500000}"
        ;;
    *)
        echo "Unknown level: $LEVEL (use 1-4 or 'custom')"
        exit 1
        ;;
esac

if [ ! -f "$PARENTS" ]; then
    echo "Error: Parent checkpoint not found: $PARENTS"
    echo "Run the previous level first."
    exit 1
fi

echo "═══════════════════════════════════════════════"
echo "  Sidon Cascade GPU Prover — Level $LEVEL"
echo "  Parents:       $PARENTS"
echo "  Output:        $OUTPUT"
echo "  d_parent=$D_PARENT  m=$M  c_target=$C_TARGET"
echo "  max_survivors: $MAX_SURV"
echo "═══════════════════════════════════════════════"
echo ""

exec "$PROVER" "$PARENTS" "$OUTPUT" \
    --d_parent "$D_PARENT" \
    --m "$M" \
    --c_target "$C_TARGET" \
    --max_survivors "$MAX_SURV"

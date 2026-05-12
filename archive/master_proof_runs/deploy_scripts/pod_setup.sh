#!/usr/bin/env bash
# pod_setup.sh -- one-shot environment setup for the Sidon dual-LP
# benchmark on a fresh Linux GPU pod (Ubuntu 22.04 + CUDA 12.x).
#
# Usage (run as root, on the pod):
#     bash pod_setup.sh
#
# Prerequisites:
#   - NVIDIA driver loaded (verify: nvidia-smi)
#   - Python 3.10+ available as python3
#   - Internet access for pip
#
# Side effects:
#   - apt installs python3-pip, python3-venv
#   - creates ~/venv with all Python deps
#   - creates ~/sidon directory (you scp the project here)

set -euo pipefail

echo "=== pod_setup.sh ==="
echo "$(date -u +%FT%TZ)  starting"

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found; is the NVIDIA driver loaded?"
    exit 1
fi

# 1. apt deps
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv

# 2. venv
if [ ! -d "$HOME/venv" ]; then
    python3 -m venv "$HOME/venv"
fi
# shellcheck source=/dev/null
source "$HOME/venv/bin/activate"
pip install -q -U pip wheel setuptools

# 3. core deps (CPU)
pip install -q numpy scipy mpmath highspy ortools mosek

# 4. cuOpt -- pick CUDA 12 vs 13 wheel automatically
CUDA_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  driver: ${CUDA_VER}"
# Default to cu12 (matches CUDA 12.x toolchain installed on most pods)
CUOPT_PKG="cuopt-cu12"
# If the runtime CUDA is 13.x, switch.
if [ -f "/usr/local/cuda/version.json" ]; then
    if grep -q '"version" : "13' /usr/local/cuda/version.json; then
        CUOPT_PKG="cuopt-cu13"
    fi
fi
echo "  installing ${CUOPT_PKG} from pypi.nvidia.com"
pip install -q --extra-index-url https://pypi.nvidia.com "${CUOPT_PKG}"

# 5. project dir
mkdir -p "$HOME/sidon"

# 6. MOSEK license. If a mosek.lic was scp'd to /root/mosek/, leave it.
# Otherwise the validate / bench scripts will skip MOSEK gracefully.
mkdir -p /root/mosek

# 7. verify
echo "=== verifying installs ==="
python - <<'EOF'
import sys
mods = [
    ("numpy",  "import numpy; print('numpy', numpy.__version__)"),
    ("scipy",  "import scipy; print('scipy', scipy.__version__)"),
    ("mpmath", "import mpmath; print('mpmath', mpmath.__version__)"),
    ("highspy","import highspy; print('highspy ok')"),
    ("ortools","import ortools; print('ortools', ortools.__version__)"),
    ("mosek",  "import mosek; print('mosek', mosek.Env.getversion())"),
    ("cuopt",  "import cuopt; print('cuopt', cuopt.__version__)"),
]
ok, fail = [], []
for name, code in mods:
    try:
        exec(code)
        ok.append(name)
    except Exception as e:
        fail.append((name, str(e)))
        print(f"  FAIL {name}: {e}")
print()
print("ok :", ", ".join(ok))
print("fail:", ", ".join(n for n,_ in fail) if fail else "(none)")
EOF

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv

echo
echo "=== DONE ==="
echo "Activate env:    source ~/venv/bin/activate"
echo "Project dir:     ~/sidon"
echo "Validate:        python ~/sidon/_tier_dual_pod_validate.py"
echo "Run bench:       python ~/sidon/_tier_dual_pod_bench.py --quick"
echo "Full grid:       python ~/sidon/_tier_dual_pod_bench.py"

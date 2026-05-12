#!/usr/bin/env bash
# MATLAB + SDPNAL+ install on a fresh Ubuntu CPU pod.
#
# STAGE 1 (you, interactive): MATLAB install via mpm + activation via
#   license file OR online sign-in. ~30 min.
# STAGE 2 (this script, automated after you confirm STAGE 1 is done):
#   downloads SDPNAL+, compiles MEX, verifies the install, confirms the
#   Python->MATLAB bridge works.
#
# USAGE, on the pod:
#   bash /workspace/sidon-autocorrelation/cpupod/sdpnal_matlab_install.sh
#
# The script is idempotent: skips steps that are already done.

set -euo pipefail
MATLAB_ROOT="/opt/matlab/R2026a"
SDPNAL_ROOT="/workspace/SDPNALplus"
REPO_ROOT="/workspace/sidon-autocorrelation"

echo "======================================================================"
echo "STAGE 1 — MATLAB install (interactive, ~30 min)"
echo "======================================================================"

# ---------- 1a. Install prerequisites ----------
if ! dpkg -s unzip build-essential xorg xauth xvfb libxt6 > /dev/null 2>&1; then
    echo "[STAGE 1a] installing apt prerequisites..."
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -yqq \
        unzip wget curl ca-certificates \
        build-essential \
        xorg xauth xvfb \
        libxt6 libxtst6 libxrender1 libxrandr2 libxcomposite1 libxcursor1 \
        libxdamage1 libxi6 libxinerama1 libxtst6 libxcb1 \
        libfontconfig1 libasound2 \
        > /tmp/apt.log 2>&1
else
    echo "[STAGE 1a] apt prerequisites already installed."
fi

# ---------- 1b. Download mpm (MathWorks Package Manager) ----------
if [ ! -x /usr/local/bin/mpm ]; then
    echo "[STAGE 1b] downloading mpm..."
    wget -qO /usr/local/bin/mpm \
        https://www.mathworks.com/mpm/glnxa64/mpm
    chmod +x /usr/local/bin/mpm
else
    echo "[STAGE 1b] mpm already present."
fi

# ---------- 1c. Install MATLAB base (no toolboxes — SDPNAL+ needs none) ----------
if [ ! -x "${MATLAB_ROOT}/bin/matlab" ]; then
    echo "[STAGE 1c] installing MATLAB base via mpm (~4 GB download, ~10 min)..."
    mkdir -p "${MATLAB_ROOT}"
    mpm install \
        --release=R2026a \
        --destination="${MATLAB_ROOT}" \
        --products=MATLAB
    ln -sf "${MATLAB_ROOT}/bin/matlab" /usr/local/bin/matlab
else
    echo "[STAGE 1c] MATLAB already installed at ${MATLAB_ROOT}."
fi

# ---------- 1d. Activation — NON-AUTOMATABLE ----------
LIC_DIR="${MATLAB_ROOT}/licenses"
mkdir -p "${LIC_DIR}"

if ls "${LIC_DIR}"/*.lic > /dev/null 2>&1; then
    echo "[STAGE 1d] license file detected at ${LIC_DIR}. Skipping activation."
else
    cat <<'EOF'

======================================================================
STAGE 1d — MATLAB ACTIVATION (you must do this, ~10 min)
======================================================================

Pick ONE of the two paths below:

Path A (license file — recommended, fastest):
  1. On your laptop, open https://www.mathworks.com/licensecenter/licenses
     signed in with your Penn MathWorks account.
  2. Click your Individual license (under Penn campus-wide).
  3. "Install and Activate" -> "Activate a Computer"
     - Operating System: Linux 64-bit
     - Host ID: the MAC address of eth0 on the pod — run on the pod:
         cat /sys/class/net/eth0/address
     - Computer login name: root
     - Activation label: "sidon-runpod" or similar
  4. Download the license.lic file MathWorks generates.
  5. scp it to the pod:
         scp -P <PORT> license.lic root@<IP>:/opt/matlab/R2026a/licenses/
  6. Re-run this script — it will detect the license and continue.

Path B (online activation — interactive over X forwarding):
  1. SSH in with X forwarding:
         ssh -X -p <PORT> root@<IP>
  2. Run: /opt/matlab/R2026a/bin/activate_matlab.sh
  3. Sign in with your Penn MathWorks account when prompted.
  4. Re-run this script after activation completes.

Exiting so you can do the activation step now.

EOF
    exit 0
fi

# ---------- 1e. Smoke-test MATLAB ----------
echo "[STAGE 1e] smoke-testing MATLAB (batch mode, should print 'ok')..."
"${MATLAB_ROOT}/bin/matlab" -batch "disp('ok')" 2>&1 | tail -5

echo ""
echo "======================================================================"
echo "STAGE 2 — SDPNAL+ install (automated, ~2 min)"
echo "======================================================================"

# ---------- 2a. Download SDPNAL+ ----------
if [ ! -d "${SDPNAL_ROOT}/SDPNAL+v1.0" ]; then
    echo "[STAGE 2a] downloading SDPNAL+..."
    mkdir -p "${SDPNAL_ROOT}"
    cd /tmp
    wget -q --no-check-certificate \
        https://blog.nus.edu.sg/mattohkc/files/2024/07/SDPNALv1.0.zip \
        -O SDPNALv1.0.zip || {
        echo "Download failed. Try the GitHub mirror or manually upload the zip."
        exit 1
    }
    unzip -q SDPNALv1.0.zip -d "${SDPNAL_ROOT}"
    cd "${SDPNAL_ROOT}"
    # The zip extracts to SDPNALv1.0/SDPNAL+v1.0 nested
    if [ -d "${SDPNAL_ROOT}/SDPNALv1.0/SDPNAL+v1.0" ]; then
        mv "${SDPNAL_ROOT}/SDPNALv1.0/SDPNAL+v1.0" "${SDPNAL_ROOT}/SDPNAL+v1.0"
        rm -rf "${SDPNAL_ROOT}/SDPNALv1.0"
    fi
else
    echo "[STAGE 2a] SDPNAL+ already extracted."
fi

# ---------- 2b. Compile MEX files inside MATLAB ----------
echo "[STAGE 2b] compiling SDPNAL+ MEX files (~1 min)..."
"${MATLAB_ROOT}/bin/matlab" -batch \
    "addpath(genpath('${SDPNAL_ROOT}/SDPNAL+v1.0')); Installmex(1); savepath;"

# ---------- 2c. Verify SDPNAL+ runs ----------
echo "[STAGE 2c] running SDPNAL+ 2x2 verification test..."
"${MATLAB_ROOT}/bin/matlab" -batch \
    "addpath(genpath('${SDPNAL_ROOT}/SDPNAL+v1.0')); \
     addpath('${REPO_ROOT}/tests'); \
     verify_sdpnal_install"

# ---------- 2d. Persist path for future MATLAB sessions ----------
STARTUP="${MATLAB_ROOT}/toolbox/local/startup.m"
if ! grep -q "${SDPNAL_ROOT}" "${STARTUP}" 2>/dev/null; then
    echo "[STAGE 2d] adding SDPNAL+ to MATLAB startup.m..."
    cat >> "${STARTUP}" <<EOF
% Added by sdpnal_matlab_install.sh
addpath(genpath('${SDPNAL_ROOT}/SDPNAL+v1.0'));
addpath('${REPO_ROOT}/tests');
EOF
fi

echo ""
echo "======================================================================"
echo "SDPNAL+ READY on this pod"
echo "======================================================================"
echo "Next step: launch d=16 L3 production run."
echo "From your laptop:"
echo "  python -m cpupod launch tests/lasserre_sdpnalplus.py \\"
echo "    --d 16 --order 3 --bw 15 --mode feasibility --target 1.2802 \\"
echo "    --no-engine --matlab-bin /opt/matlab/R2026a/bin/matlab \\"
echo "    --sdpnal-maxiter 20000"

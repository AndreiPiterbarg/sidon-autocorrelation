#!/usr/bin/env bash
# deploy_sublevel_d16.sh
#
# After `python -m cpupod start` succeeds, this script:
#   1. Reads the pod SSH host/port from cpupod/.session.json
#   2. Installs mosek + scipy on the pod with python3.13
#   3. Uploads the MOSEK license from ~/mosek/mosek.lic
#   4. Verifies mosek imports successfully
#   5. Launches tests/lasserre_mosek_sublevel.py at d=16 L3 with target 1.2802
#      via `python -m cpupod launch` (detached tmux session).
#
# Usage:
#   ./deploy_sublevel_d16.sh [time_budget_s]
# Default time budget: 18000s (5h).
set -euo pipefail
cd "$(dirname "$0")"

TIME_BUDGET="${1:-18000}"

SESSION_JSON="cpupod/.session.json"
if [[ ! -f "$SESSION_JSON" ]]; then
    echo "ERROR: $SESSION_JSON missing; run 'python -m cpupod start' first."
    exit 1
fi

POD_ID=$(python -c "import json; print(json.load(open('$SESSION_JSON'))['pod_id'])")
SSH_HOST=$(python -c "import json; print(json.load(open('$SESSION_JSON'))['ssh_host'])")
SSH_PORT=$(python -c "import json; print(json.load(open('$SESSION_JSON'))['ssh_port'])")

echo "=== Pod: $POD_ID ($SSH_HOST:$SSH_PORT) ==="

SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $SSH_KEY"
SCP_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $SSH_KEY"

REMOTE_WORKDIR=/workspace/sidon-autocorrelation

echo "[1/4] Installing MOSEK + scipy on pod..."
ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" \
    "python3.13 -m pip install -q mosek scipy 2>&1 | tail -5"

echo "[2/4] Uploading MOSEK license..."
ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" "mkdir -p /root/mosek"
scp -P "$SSH_PORT" $SCP_OPTS "$HOME/mosek/mosek.lic" \
    root@"$SSH_HOST":/root/mosek/mosek.lic

echo "[3/4] Verifying mosek imports..."
ssh -p "$SSH_PORT" $SSH_OPTS root@"$SSH_HOST" \
    "python3.13 -c 'import mosek; print(\"mosek\", mosek.Env.getversion())'"

echo "[4/4] Launching d=16 L3 sublevel via cpupod..."
python -m cpupod launch tests/lasserre_mosek_sublevel.py \
    --d 16 --order 3 --sublevel \
    --target-lb 1.2802 \
    --add-per-step 50 \
    --bisect-per-step 3 \
    --time-budget-s "$TIME_BUDGET" \
    --watcher-interval 60 \
    --progress data/sublevel_d16_progress.json \
    --json data/sublevel_d16_final.json \
    --proof-dir data/mosek_d16_sublevel_proof

echo "=== LAUNCHED ==="
echo "pod_id = $POD_ID"
echo "Use 'python -m cpupod logs -f' to follow, 'status' for job state,"
echo "'fetch' to pull progress snapshots, 'teardown' when done."

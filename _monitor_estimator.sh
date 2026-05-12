#!/bin/bash
# Watchdog for cascade_estimate runs.  Polls the log every 20s.
# Kills the python process if any single config has been running > MAX_PER_CONFIG seconds.
# Restarts the script with remaining configs (the next one).
set -u
LOG=$1                            # log file path
TOTAL_BUDGET=${2:-1800}           # total seconds (default 30 min)
MAX_PER_CONFIG=${3:-360}          # max wall per config (default 6 min)
START=$(date +%s)
LAST_CONFIG_START=$START
LAST_CONFIG_TAG=""

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if [ "$ELAPSED" -gt "$TOTAL_BUDGET" ]; then
    echo "[monitor] total budget $TOTAL_BUDGET exceeded; killing python."
    pkill -9 -f _cascade_estimate.py
    break
  fi

  # Detect current config from log (last 'CONFIG: ...' line).
  CURRENT_TAG=$(grep '^CONFIG: ' "$LOG" 2>/dev/null | tail -1)
  if [ -z "$CURRENT_TAG" ]; then
    sleep 20
    continue
  fi
  if [ "$CURRENT_TAG" != "$LAST_CONFIG_TAG" ]; then
    LAST_CONFIG_TAG="$CURRENT_TAG"
    LAST_CONFIG_START=$NOW
    echo "[monitor] new config detected: $CURRENT_TAG"
  fi
  CONFIG_ELAPSED=$((NOW - LAST_CONFIG_START))
  if [ "$CONFIG_ELAPSED" -gt "$MAX_PER_CONFIG" ]; then
    echo "[monitor] config '$CURRENT_TAG' has run ${CONFIG_ELAPSED}s > ${MAX_PER_CONFIG}s.  Killing python."
    pkill -9 -f _cascade_estimate.py
    sleep 3
    break
  fi

  # Also exit if the python is no longer running
  if ! pgrep -f _cascade_estimate.py > /dev/null; then
    echo "[monitor] python no longer running; exiting."
    break
  fi

  sleep 20
done
echo "[monitor] done at $(date)."

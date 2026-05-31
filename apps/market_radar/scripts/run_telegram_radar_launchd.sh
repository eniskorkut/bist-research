#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/eniskorkut/Documents/New project 2/bist-research"
cd "$REPO_ROOT"

mkdir -p logs

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PYTHONPATH="$REPO_ROOT/apps/market_radar/src:${PYTHONPATH:-}"

# launchd has a minimal environment, so load local config explicitly.
if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
fi

# Skip weekends. launchd still schedules the job; this keeps the calendar simple.
dow="$(date +%u)"
if [ "$dow" -gt 5 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') INFO telegram_radar skipped weekend"
  exit 0
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR .venv/bin/python missing"
  exit 1
fi

# Prevent sleep while the radar refresh + KAP fetch + Telegram send is running.
exec /usr/bin/caffeinate -s -t 1800 \
  ".venv/bin/python" "apps/market_radar/scripts/run_daily_radar_and_send_telegram.py" \
  --send-telegram

#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/eniskorkut/Documents/New project 2/bist-research"
LABEL="com.eniskorkut.bist-radar.telegram"
SRC_PLIST="$REPO_ROOT/apps/market_radar/scripts/$LABEL.plist"
DST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

cd "$REPO_ROOT"
mkdir -p "$HOME/Library/LaunchAgents" logs

missing=0
if [ ! -f ".env" ]; then
  echo "ERROR .env missing. Create it from .env.example and add Telegram credentials."
  missing=1
else
  for key in TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID MARKET_RADAR_TELEGRAM_ENABLED; do
    if ! grep -q "^$key=" ".env"; then
      echo "ERROR .env missing $key"
      missing=1
    fi
  done
fi

if [ "$missing" -ne 0 ]; then
  echo "Not installing launchd job because Telegram config is incomplete."
  exit 1
fi

cp "$SRC_PLIST" "$DST_PLIST"
chmod 644 "$DST_PLIST"

launchctl bootout "gui/$(id -u)" "$DST_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$DST_PLIST"
launchctl enable "gui/$(id -u)/$LABEL"

echo "Installed $LABEL"
echo "Status: launchctl print gui/$(id -u)/$LABEL"
echo "Logs: tail -f '$REPO_ROOT/logs/telegram_radar.launchd.log' '$REPO_ROOT/logs/telegram_radar.launchd.err.log'"

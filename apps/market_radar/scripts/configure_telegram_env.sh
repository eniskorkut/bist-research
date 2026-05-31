#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/eniskorkut/Documents/New project 2/bist-research"
ENV_FILE="$REPO_ROOT/.env"

cd "$REPO_ROOT"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

upsert_env() {
  local key="$1"
  local value="$2"
  local escaped
  escaped="$(printf '%s' "$value" | sed 's/[\/&]/\\&/g')"
  if grep -q "^$key=" "$ENV_FILE"; then
    sed -i.bak "s/^$key=.*/$key=$escaped/" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

printf "Telegram bot token: "
read -rs TELEGRAM_BOT_TOKEN_INPUT
printf "\n"

printf "Telegram chat id [1350520951]: "
read -r TELEGRAM_CHAT_ID_INPUT
TELEGRAM_CHAT_ID_INPUT="${TELEGRAM_CHAT_ID_INPUT:-1350520951}"

upsert_env "TELEGRAM_BOT_TOKEN" "$TELEGRAM_BOT_TOKEN_INPUT"
upsert_env "TELEGRAM_CHAT_ID" "$TELEGRAM_CHAT_ID_INPUT"
upsert_env "MARKET_RADAR_TELEGRAM_ENABLED" "true"
upsert_env "MARKET_RADAR_TIMEZONE" "Europe/Istanbul"
upsert_env "MARKET_RADAR_ALERT_TIMES" "09:00,12:00,15:00,18:00"
upsert_env "MARKET_RADAR_STRATEGY" "adaptive_v1_cash_no_buy"
upsert_env "MARKET_RADAR_PRIORITY_FILTER" "special_strict_top10"
upsert_env "MARKET_RADAR_TOP_N" "30"
upsert_env "MARKET_RADAR_KAP_SOURCE" "mcp"
upsert_env "MARKET_RADAR_INTRADAY_MODE" "new_only"
upsert_env "MARKET_RADAR_DEDUPE_LOOKBACK_TRADING_DAYS" "3"
upsert_env "MARKET_RADAR_ALERT_MAX_SYMBOLS" "10"
upsert_env "MARKET_RADAR_INCLUDE_FULL_TOP30" "false"
upsert_env "MARKET_RADAR_KAP_SUMMARY_MAX_CHARS" "80"
upsert_env "MARKET_RADAR_REPEAT_IF_RANK_IMPROVES_BY" "5"
upsert_env "MARKET_RADAR_REPEAT_IF_QUALITY_IMPROVES_BY" "5"

rm -f "$ENV_FILE.bak"
echo ".env updated. Token hidden in terminal input."

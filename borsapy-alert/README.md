# borsapy-alert

Live BIST volume alert service. Uses `borsapy` `TradingViewStream` for quotes and daily history for 20-day average volume.

## Commands

```shell
python scripts/run_volume_alert.py --symbols THYAO ASELS GARAN --dry-run --duration 120
python scripts/run_volume_alert.py --index XU030 --dry-run --duration 300 --max-symbols 5
python scripts/run_volume_alert.py --index XU100 --dry-run --ratio-threshold 1.5 --velocity-threshold 0.05
```

## Alert Rules

- Current intraday volume / 20-day average volume >= `--ratio-threshold`
- Last 5-minute volume delta / 20-day average volume >= `--velocity-threshold`
- Same symbol cooldown: `--cooldown-seconds`

`--dry-run` marks alert rows as dry-run. CSV append remains enabled.

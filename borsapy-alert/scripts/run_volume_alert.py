from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bist_volume_alert.volume_alert import VolumeAlertRunner, build_config_from_args, env_default


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BIST live volume alert runner.")
    p.add_argument("--symbols", nargs="+", help="BIST stock symbols, e.g. THYAO ASELS GARAN.")
    p.add_argument("--index", help="BIST index components to watch, e.g. XU030, XU100, XBANK.")
    p.add_argument("--max-symbols", type=int, help="Limit index components for tests.")
    p.add_argument("--history-period", default=env_default("BORSAPY_HISTORY_PERIOD", "1ay"))
    p.add_argument("--lookback-days", type=int, default=int(env_default("BORSAPY_LOOKBACK_DAYS", "20")))
    p.add_argument("--ratio-threshold", type=float, default=float(env_default("BORSAPY_RATIO_THRESHOLD", "1.5")))
    p.add_argument("--velocity-threshold", type=float, default=float(env_default("BORSAPY_VELOCITY_THRESHOLD", "0.05")))
    p.add_argument("--cooldown-seconds", type=int, default=int(env_default("BORSAPY_COOLDOWN_SECONDS", "900")))
    p.add_argument("--check-interval", type=float, default=float(env_default("BORSAPY_CHECK_INTERVAL", "5")))
    p.add_argument("--duration", type=float, help="Stop after N seconds. Omit for continuous mode.")
    p.add_argument("--csv-path", default=os.environ.get("BORSAPY_ALERT_CSV", str(ROOT / "alerts.csv")))
    p.add_argument("--dry-run", action="store_true", help="Mark alerts as dry-run. CSV append remains enabled.")
    return p


def main() -> int:
    args = parser().parse_args()
    config = build_config_from_args(args)
    print(
        "runner ready "
        f"symbols={','.join(config.symbols)} "
        f"ratio_threshold={config.ratio_threshold:g} "
        f"velocity_threshold={config.velocity_threshold:g} "
        f"cooldown_seconds={config.cooldown_seconds} "
        f"csv_path={config.csv_path}",
        flush=True,
    )
    VolumeAlertRunner(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

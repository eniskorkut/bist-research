from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CSV_FIELDS = [
    "timestamp_utc",
    "symbol",
    "last_price",
    "current_volume",
    "avg_20d_volume",
    "day_ratio",
    "volume_5m_delta",
    "velocity_ratio",
    "reason",
    "dry_run",
]


@dataclass(frozen=True)
class VolumeAlert:
    symbol: str
    last_price: float | None
    current_volume: float
    avg_20d_volume: float
    day_ratio: float
    volume_5m_delta: float
    velocity_ratio: float
    reason: str
    dry_run: bool
    timestamp_utc: str = ""

    def row(self) -> dict[str, str | float | bool | None]:
        data = asdict(self)
        data["timestamp_utc"] = self.timestamp_utc or datetime.now(timezone.utc).isoformat()
        return data


class CsvAlertStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_header()

    def ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

    def append(self, alert: VolumeAlert) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow(alert.row())

    def append_many(self, alerts: Iterable[VolumeAlert]) -> None:
        for alert in alerts:
            self.append(alert)

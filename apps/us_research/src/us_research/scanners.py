from __future__ import annotations

import pandas as pd

from us_research.market_data import scan_us_volume_above_average


def run_us_volume_scan(
    symbols: list[str],
    lookback: int = 20,
    min_ratio: float = 1.5,
    start_date: str | None = None,
) -> pd.DataFrame:
    return scan_us_volume_above_average(
        symbols=symbols,
        lookback=lookback,
        min_ratio=min_ratio,
        start_date=start_date,
    )


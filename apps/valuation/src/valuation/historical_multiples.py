from __future__ import annotations


def get_historical_pe_median(symbol: str, years: int = 3) -> float | None:
    # TODO: calculate true historical P/E median from borsapy price history and
    # period-aligned historical net income snapshots.
    # For now return None and let UI/reporting show "hesaplanamadi".
    _ = (symbol, years)
    return None

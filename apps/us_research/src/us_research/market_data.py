from __future__ import annotations

import pandas as pd
from openbb import obb


def get_us_history(
    symbol: str,
    start_date: str | None = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    params: dict[str, str] = {"symbol": symbol, "provider": provider}
    if start_date:
        params["start_date"] = start_date
    return obb.equity.price.historical(**params).to_dataframe()


def get_us_latest_volume(symbol: str) -> float:
    df = get_us_history(symbol)
    if df.empty or "volume" not in df.columns:
        raise ValueError(f"Missing volume for symbol={symbol}")
    return float(df["volume"].iloc[-1])


def get_us_average_volume(symbol: str, lookback: int = 20) -> float:
    df = get_us_history(symbol)
    if df.empty or "volume" not in df.columns:
        raise ValueError(f"Missing volume for symbol={symbol}")
    volume = df["volume"].tail(lookback)
    if volume.empty:
        raise ValueError(f"Insufficient volume history for symbol={symbol}")
    return float(volume.mean())


def scan_us_volume_above_average(
    symbols: list[str],
    lookback: int = 20,
    min_ratio: float = 1.5,
    start_date: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for symbol in symbols:
        df = get_us_history(symbol=symbol, start_date=start_date)
        if df.empty or "volume" not in df.columns or "close" not in df.columns:
            continue
        latest_volume = float(df["volume"].iloc[-1])
        avg_volume = float(df["volume"].tail(lookback).mean())
        if avg_volume <= 0:
            continue
        ratio = latest_volume / avg_volume
        if ratio < min_ratio:
            continue
        last_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
        rows.append(
            {
                "market": "US",
                "symbol": symbol,
                "currency": "USD",
                "source": "openbb:yfinance",
                "last_date": last_date,
                "last_close": float(df["close"].iloc[-1]),
                "latest_volume": latest_volume,
                "avg_volume_20d": avg_volume,
                "volume_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


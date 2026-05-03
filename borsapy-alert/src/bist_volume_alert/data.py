from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import borsapy as bp
import pandas as pd


@dataclass(frozen=True)
class AverageVolume:
    symbol: str
    avg_20d_volume: float
    rows_used: int


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def index_symbols(index: str, max_symbols: int | None = None) -> list[str]:
    normalized = normalize_symbol(index)
    idx = bp.Index(normalized)
    symbols = normalize_symbols(getattr(idx, "component_symbols", []) or [])
    if not symbols:
        raise RuntimeError(f"{normalized} index components empty")
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    return symbols


def volume_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).lower() == "volume":
            return str(column)
    raise RuntimeError(f"Volume column not found. columns={list(df.columns)}")


def average_20d_volume(symbol: str, period: str = "1ay", lookback_days: int = 20) -> AverageVolume:
    normalized = normalize_symbol(symbol)
    df = bp.Ticker(normalized).history(period=period)
    if df is None or df.empty:
        raise RuntimeError(f"{normalized} history empty")

    column = volume_column(df)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    series = series[series > 0].tail(lookback_days)
    if series.empty:
        raise RuntimeError(f"{normalized} positive volume history empty")

    avg = float(series.mean())
    if not math.isfinite(avg) or avg <= 0:
        raise RuntimeError(f"{normalized} invalid average volume: {avg}")
    return AverageVolume(symbol=normalized, avg_20d_volume=avg, rows_used=int(series.shape[0]))


def build_average_volume_map(
    symbols: Iterable[str],
    period: str = "1ay",
    lookback_days: int = 20,
) -> tuple[dict[str, float], dict[str, str]]:
    averages: dict[str, float] = {}
    errors: dict[str, str] = {}
    for symbol in normalize_symbols(symbols):
        try:
            average = average_20d_volume(symbol, period=period, lookback_days=lookback_days)
            averages[average.symbol] = average.avg_20d_volume
        except Exception as exc:  # noqa: BLE001 - keep runner alive per symbol.
            errors[symbol] = str(exc)
    return averages, errors

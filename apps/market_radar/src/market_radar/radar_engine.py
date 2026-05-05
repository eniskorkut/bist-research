from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

import pandas as pd

from market_radar.data_access import BorsapyMarketDataClient, save_radar_result
from market_radar.symbols import normalize_bist_symbol


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


@dataclass
class RadarConfig:
    lookback_days: int = 260
    min_avg_turnover_try_active: bool = True
    min_avg_turnover_try: float = 10_000_000.0
    min_volume_ratio_active: bool = True
    min_volume_ratio: float = 1.5
    min_turnover_ratio_active: bool = True
    min_turnover_ratio: float = 1.5
    min_daily_return_active: bool = True
    min_daily_return: float = 0.0
    min_close_position_active: bool = True
    min_close_position: float = 0.65
    breakout_mode: str = "off"
    require_ma20_active: bool = True
    require_ma50_active: bool = False
    min_xu100_relative_active: bool = True
    min_xu100_relative: float = 0.0
    min_interest_score_active: bool = True
    min_interest_score: float = 50.0
    include_negative_moves: bool = False
    force_refresh: bool = False
    db_path: str = "/data/market_radar_cache.sqlite"


@dataclass
class RadarResult:
    symbol: str
    date: str | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    prev_close: float | None
    volume: float | None
    last_close: float | None
    daily_return_pct: float | None
    day_range_pct: float | None
    close_position: float | None
    avg_volume_20d: float | None
    volume_ratio_20d: float | None
    turnover_try: float | None
    avg_turnover_20d: float | None
    turnover_ratio_20d: float | None
    ma20: float | None
    ma50: float | None
    above_ma20: bool
    above_ma50: bool
    high_20d: float | None
    breakout_20d: bool
    high_52w: float | None
    near_52w_high: bool
    xu100_daily_return_pct: float | None
    xu100_relative_return_pct: float | None
    sector_relative_return_pct: float | None
    interest_score: float
    signals: list[str] = field(default_factory=list)
    passed_filters: list[str] = field(default_factory=list)
    failed_filters: list[str] = field(default_factory=list)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Interest Score": self.interest_score,
            "Last Close": self.last_close,
            "Daily Return %": self.daily_return_pct,
            "Volume Ratio 20D": self.volume_ratio_20d,
            "Turnover Ratio 20D": self.turnover_ratio_20d,
            "XU100 Relative %": self.xu100_relative_return_pct,
            "Close Position": self.close_position,
            "Breakout 20D": self.breakout_20d,
            "Above MA20": self.above_ma20,
            "Signals": ", ".join(self.signals),
        }


def _tail_average(df: pd.DataFrame, column: str, window: int, *, exclude_latest: bool = False) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    frame = df.iloc[:-1] if exclude_latest and len(df) > 1 else df
    series = frame[column].tail(window)
    if series.empty:
        return None
    return _safe_float(series.mean())


def _latest_common_row(stock: pd.DataFrame, benchmark: pd.DataFrame | None) -> tuple[pd.Series | None, pd.Series | None]:
    if stock is None or stock.empty:
        return None, None
    stock = stock.sort_index()
    if benchmark is None or benchmark.empty:
        return stock.iloc[-1], None
    benchmark = benchmark.sort_index()
    common = stock.index.intersection(benchmark.index)
    if common.empty:
        return stock.iloc[-1], benchmark.iloc[-1]
    idx = common[-1]
    return stock.loc[idx], benchmark.loc[idx]


def _daily_return_from_frame(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "close" not in df.columns:
        return None
    closes = df["close"].dropna()
    if len(closes) < 2:
        return None
    prev_close = closes.iloc[-2]
    last_close = closes.iloc[-1]
    if prev_close in (None, 0) or pd.isna(prev_close):
        return None
    return ((last_close / prev_close) - 1) * 100


def calculate_interest_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    volume_ratio = metrics.get("volume_ratio_20d")
    if isinstance(volume_ratio, (int, float)):
        if volume_ratio >= 3.0:
            score += 25
        elif volume_ratio >= 2.0:
            score += 20
        elif volume_ratio >= 1.5:
            score += 12

    turnover_ratio = metrics.get("turnover_ratio_20d")
    if isinstance(turnover_ratio, (int, float)):
        if turnover_ratio >= 3.0:
            score += 20
        elif turnover_ratio >= 2.0:
            score += 15
        elif turnover_ratio >= 1.5:
            score += 8

    daily_return = metrics.get("daily_return_pct")
    if isinstance(daily_return, (int, float)):
        if daily_return >= 5:
            score += 15
        elif daily_return >= 2:
            score += 10
        elif daily_return > 0:
            score += 5

    xu100_relative = metrics.get("xu100_relative_return_pct")
    if isinstance(xu100_relative, (int, float)):
        if xu100_relative >= 2:
            score += 10
        elif xu100_relative > 0:
            score += 5

    sector_relative = metrics.get("sector_relative_return_pct")
    if isinstance(sector_relative, (int, float)):
        if sector_relative >= 2:
            score += 10
        elif sector_relative > 0:
            score += 5

    if metrics.get("breakout_20d"):
        score += 10
    if metrics.get("near_52w_high"):
        score += 5

    close_position = metrics.get("close_position")
    if isinstance(close_position, (int, float)):
        if close_position >= 0.80:
            score += 10
        elif close_position >= 0.65:
            score += 5

    if metrics.get("above_ma20"):
        score += 5
    if metrics.get("above_ma50"):
        score += 5

    return min(score, 100.0)


def build_signals(metrics: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    volume_ratio = metrics.get("volume_ratio_20d")
    turnover_ratio = metrics.get("turnover_ratio_20d")
    daily_return = metrics.get("daily_return_pct")
    xu100_relative = metrics.get("xu100_relative_return_pct")
    sector_relative = metrics.get("sector_relative_return_pct")
    close_position = metrics.get("close_position")

    if isinstance(volume_ratio, (int, float)) and volume_ratio >= 1.5:
        signals.append("volume_spike")
    if isinstance(volume_ratio, (int, float)) and volume_ratio >= 3.0:
        signals.append("strong_volume_spike")
    if isinstance(turnover_ratio, (int, float)) and turnover_ratio >= 1.5:
        signals.append("turnover_spike")
    if isinstance(daily_return, (int, float)) and daily_return > 0:
        signals.append("positive_price_move")
    if isinstance(daily_return, (int, float)) and daily_return >= 5:
        signals.append("strong_price_move")
    if isinstance(xu100_relative, (int, float)) and xu100_relative > 0:
        signals.append("relative_strength_xu100")
    if isinstance(sector_relative, (int, float)) and sector_relative > 0:
        signals.append("relative_strength_sector")
    if metrics.get("breakout_20d"):
        signals.append("breakout_20d")
    if metrics.get("near_52w_high"):
        signals.append("near_52w_high")
    if isinstance(close_position, (int, float)) and close_position >= 0.80:
        signals.append("strong_close")
    if metrics.get("above_ma20"):
        signals.append("above_ma20")
    if metrics.get("above_ma50"):
        signals.append("above_ma50")
    return signals


def _filter_reason(name: str, passing: bool) -> tuple[str, bool]:
    return name, passing


def evaluate_filters(metrics: dict[str, Any], config: RadarConfig) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []

    if config.min_avg_turnover_try_active:
        value = metrics.get("avg_turnover_20d")
        if isinstance(value, (int, float)) and value >= config.min_avg_turnover_try:
            passed.append("min_avg_turnover_try")
        else:
            failed.append("min_avg_turnover_try")

    if config.min_volume_ratio_active:
        value = metrics.get("volume_ratio_20d")
        if isinstance(value, (int, float)) and value >= config.min_volume_ratio:
            passed.append("min_volume_ratio")
        else:
            failed.append("min_volume_ratio")

    if config.min_turnover_ratio_active:
        value = metrics.get("turnover_ratio_20d")
        if isinstance(value, (int, float)) and value >= config.min_turnover_ratio:
            passed.append("min_turnover_ratio")
        else:
            failed.append("min_turnover_ratio")

    if config.min_daily_return_active:
        value = metrics.get("daily_return_pct")
        if isinstance(value, (int, float)) and value >= config.min_daily_return:
            passed.append("min_daily_return")
        else:
            failed.append("min_daily_return")

    if config.min_close_position_active:
        value = metrics.get("close_position")
        if isinstance(value, (int, float)) and value >= config.min_close_position:
            passed.append("min_close_position")
        else:
            failed.append("min_close_position")

    if config.breakout_mode == "breakout_20d":
        if metrics.get("breakout_20d"):
            passed.append("breakout_20d")
        else:
            failed.append("breakout_20d")
    elif config.breakout_mode == "near_20d_high_2pct":
        value = metrics.get("near_52w_high")
        # named requirement: near the 20d high, approximated from the 20d breakout window
        if bool(value) or bool(metrics.get("close")) and bool(metrics.get("high_20d")) and float(metrics["close"]) >= float(metrics["high_20d"]) * 0.98:
            passed.append("near_20d_high_2pct")
        else:
            failed.append("near_20d_high_2pct")

    if config.require_ma20_active:
        if metrics.get("above_ma20"):
            passed.append("above_ma20")
        else:
            failed.append("above_ma20")

    if config.require_ma50_active:
        if metrics.get("above_ma50"):
            passed.append("above_ma50")
        else:
            failed.append("above_ma50")

    if config.min_xu100_relative_active:
        value = metrics.get("xu100_relative_return_pct")
        if isinstance(value, (int, float)) and value >= config.min_xu100_relative:
            passed.append("xu100_relative")
        else:
            failed.append("xu100_relative")

    if config.min_interest_score_active:
        if metrics.get("interest_score", 0) >= config.min_interest_score:
            passed.append("min_interest_score")
        else:
            failed.append("min_interest_score")

    if not config.include_negative_moves:
        daily_return = metrics.get("daily_return_pct")
        if isinstance(daily_return, (int, float)) and daily_return < 0:
            failed.append("negative_price_move")

    return passed, failed


def evaluate_symbol(
    symbol: str,
    history: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    config: RadarConfig | None = None,
) -> RadarResult:
    config = config or RadarConfig()
    df = history.copy()
    if df is None or df.empty:
        raise ValueError(f"No OHLCV history for symbol={symbol}")
    df = df.sort_index()
    df.columns = [str(col).strip().lower() for col in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    if any(col not in df.columns for col in required):
        raise ValueError(f"Missing OHLCV columns for symbol={symbol}")

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    open_ = _safe_float(latest.get("open"))
    high = _safe_float(latest.get("high"))
    low = _safe_float(latest.get("low"))
    close = _safe_float(latest.get("close"))
    volume = _safe_float(latest.get("volume"))
    prev_close = _safe_float(prev.get("close")) if prev is not None else None

    daily_return_pct = None
    if close is not None and prev_close not in (None, 0):
        daily_return_pct = ((close / prev_close) - 1) * 100
    day_range_pct = None
    if close is not None and high is not None and low is not None and close != 0:
        day_range_pct = ((high - low) / close) * 100
    close_position = None
    if high is not None and low is not None and close is not None and high != low:
        close_position = (close - low) / (high - low)

    avg_volume_20d = _tail_average(df, "volume", 20, exclude_latest=True)
    volume_ratio_20d = (volume / avg_volume_20d) if volume is not None and avg_volume_20d not in (None, 0) else None
    turnover_try = (close * volume) if close is not None and volume is not None else None
    turnover_series = df["close"] * df["volume"]
    avg_turnover_20d = _safe_float(turnover_series.iloc[:-1].tail(20).mean()) if len(turnover_series) > 1 else None
    turnover_ratio_20d = (turnover_try / avg_turnover_20d) if turnover_try is not None and avg_turnover_20d not in (None, 0) else None

    ma20 = _tail_average(df, "close", 20, exclude_latest=False)
    ma50 = _tail_average(df, "close", 50, exclude_latest=False)
    above_ma20 = bool(close is not None and ma20 is not None and close > ma20)
    above_ma50 = bool(close is not None and ma50 is not None and close > ma50)
    high_20d = _safe_float(df["high"].tail(20).max())
    breakout_20d = bool(close is not None and high_20d is not None and close >= high_20d)
    high_52w = _safe_float(df["high"].tail(252).max())
    near_52w_high = bool(close is not None and high_52w is not None and close >= high_52w * 0.95)

    xu100_daily_return_pct = _daily_return_from_frame(benchmark)
    xu100_relative_return_pct = (
        daily_return_pct - xu100_daily_return_pct
        if daily_return_pct is not None and xu100_daily_return_pct is not None
        else None
    )
    sector_relative_return_pct = None

    metrics: dict[str, Any] = {
        "symbol": normalize_bist_symbol(symbol),
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "prev_close": prev_close,
        "volume": volume,
        "last_close": close,
        "daily_return_pct": daily_return_pct,
        "day_range_pct": day_range_pct,
        "close_position": close_position,
        "avg_volume_20d": avg_volume_20d,
        "volume_ratio_20d": volume_ratio_20d,
        "turnover_try": turnover_try,
        "avg_turnover_20d": avg_turnover_20d,
        "turnover_ratio_20d": turnover_ratio_20d,
        "ma20": ma20,
        "ma50": ma50,
        "above_ma20": above_ma20,
        "above_ma50": above_ma50,
        "high_20d": high_20d,
        "breakout_20d": breakout_20d,
        "high_52w": high_52w,
        "near_52w_high": near_52w_high,
        "xu100_daily_return_pct": xu100_daily_return_pct,
        "xu100_relative_return_pct": xu100_relative_return_pct,
        "sector_relative_return_pct": sector_relative_return_pct,
    }
    interest_score = calculate_interest_score(metrics)
    metrics["interest_score"] = interest_score
    signals = build_signals(metrics)
    passed_filters, failed_filters = evaluate_filters(metrics, config)

    result = RadarResult(
        symbol=normalize_bist_symbol(symbol),
        date=metrics["date"],
        open=open_,
        high=high,
        low=low,
        close=close,
        prev_close=prev_close,
        volume=volume,
        last_close=close,
        daily_return_pct=daily_return_pct,
        day_range_pct=day_range_pct,
        close_position=close_position,
        avg_volume_20d=avg_volume_20d,
        volume_ratio_20d=volume_ratio_20d,
        turnover_try=turnover_try,
        avg_turnover_20d=avg_turnover_20d,
        turnover_ratio_20d=turnover_ratio_20d,
        ma20=ma20,
        ma50=ma50,
        above_ma20=above_ma20,
        above_ma50=above_ma50,
        high_20d=high_20d,
        breakout_20d=breakout_20d,
        high_52w=high_52w,
        near_52w_high=near_52w_high,
        xu100_daily_return_pct=xu100_daily_return_pct,
        xu100_relative_return_pct=xu100_relative_return_pct,
        sector_relative_return_pct=sector_relative_return_pct,
        interest_score=interest_score,
        signals=signals,
        passed_filters=passed_filters,
        failed_filters=failed_filters,
        raw_metrics=metrics,
    )
    return result


@dataclass
class ScanResult:
    """Aggregated result of scanning multiple symbols."""
    results: list[RadarResult]
    raw_results: list[RadarResult]
    failed_symbols: list[dict[str, str]]
    scan_summary: dict[str, Any]


def scan_symbols(
    symbols: list[str],
    *,
    config: RadarConfig | None = None,
    client: BorsapyMarketDataClient | None = None,
    progress_callback: Any | None = None,
) -> ScanResult:
    config = config or RadarConfig()
    client = client or BorsapyMarketDataClient()
    benchmark = client.load_history("XU100", lookback_days=config.lookback_days, db_path=config.db_path, force=config.force_refresh)
    passed: list[RadarResult] = []
    all_results: list[RadarResult] = []
    failed_symbols: list[dict[str, str]] = []
    total = len(symbols)

    for idx, raw_symbol in enumerate(symbols):
        symbol = normalize_bist_symbol(raw_symbol)
        if not symbol:
            continue
        try:
            history = client.load_history(
                symbol,
                lookback_days=config.lookback_days,
                db_path=config.db_path,
                force=config.force_refresh,
            )
            if history.empty:
                continue
            result = evaluate_symbol(symbol, history, benchmark, config=config)
            all_results.append(result)
            save_radar_result(
                config.db_path,
                {
                    "symbol": result.symbol,
                    "scanned_at": result.raw_metrics.get("date"),
                    "source": "borsapy",
                    "interest_score": result.interest_score,
                    "metrics": asdict(result),
                    "signals": result.signals,
                    "passed_filters": result.passed_filters,
                    "failed_filters": result.failed_filters,
                },
            )
            if not config.include_negative_moves and (result.daily_return_pct or 0) < 0:
                pass  # don't add to passed
            elif config.min_interest_score_active and result.interest_score < config.min_interest_score:
                pass  # don't add to passed
            elif all(name not in result.failed_filters for name in ["min_avg_turnover_try", "min_volume_ratio", "min_turnover_ratio", "min_daily_return", "min_close_position", "above_ma20", "above_ma50", "xu100_relative", "min_interest_score", "negative_price_move"]):
                passed.append(result)
            elif not result.failed_filters:
                passed.append(result)
        except Exception as exc:  # noqa: BLE001
            failed_symbols.append({"symbol": symbol, "error": str(exc)})

        if progress_callback is not None:
            try:
                progress_callback(idx + 1, total, symbol)
            except Exception:  # noqa: BLE001
                pass

    passed = sorted(passed, key=lambda item: item.interest_score, reverse=True)
    all_results = sorted(all_results, key=lambda item: item.interest_score, reverse=True)

    scan_summary = {
        "universe_symbol_count": total,
        "scanned_symbols": len(all_results) + len(failed_symbols),
        "successful_symbols": len(all_results),
        "failed_symbols": len(failed_symbols),
        "result_count": len(passed),
    }

    return ScanResult(
        results=passed,
        raw_results=all_results,
        failed_symbols=failed_symbols,
        scan_summary=scan_summary,
    )


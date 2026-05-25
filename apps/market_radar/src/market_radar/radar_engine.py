from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import json
import time
from typing import Any

import pandas as pd

from market_radar.data_access import (
    BorsapyMarketDataClient,
    HistoryLoadResult,
    get_cached_scan_result,
    get_default_ohlcv_cache_ttl_minutes,
    save_radar_results_bulk,
    upsert_cached_scan_result,
)
from market_radar.symbols import normalize_bist_symbol
from market_radar.regime import MarketRegimeDetector


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
    scan_mode: str = "positive_money_flow"
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
    min_cmf_20_active: bool = False
    min_cmf_20: float = 0.0
    require_obv_slope_5d_positive: bool = False
    require_obv_slope_20d_positive: bool = False
    min_mfi_14_active: bool = False
    min_mfi_14: float = 50.0
    max_mfi_14_active: bool = False
    max_mfi_14: float = 85.0
    max_daily_return_active: bool = False
    max_daily_return_pct: float = 2.0
    max_price_range_active: bool = False
    max_price_range_pct: float = 5.0
    min_accumulation_score_active: bool = False
    min_accumulation_score: float = 50.0
    include_negative_moves: bool = False
    force_refresh: bool = False
    db_path: str = "/data/market_radar_cache.sqlite"
    max_workers: int = 8
    ohlcv_cache_ttl_minutes: int | None = None
    use_scan_cache: bool = True
    scan_cache_ttl_minutes: int = 15
    index_symbol: str = "XUTUM"
    active_volume_spike_quality_active: bool = False
    min_last_turnover_try_active: bool = True
    min_last_turnover_try: float = 10_000_000.0
    min_avg_turnover_20d_try_active: bool = True
    min_avg_turnover_20d_try: float = 10_000_000.0
    max_rsi_14_active: bool = True
    max_rsi_14: float = 78.0
    max_return_5d_active: bool = True
    max_return_5d_pct: float = 35.0
    max_return_10d_active: bool = True
    max_return_10d_pct: float = 60.0
    require_strong_close: bool = True
    min_above_ma20_ratio: float = 1.0


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
    cmf_20: float | None
    adl_slope_5d: float | None
    adl_slope_20d: float | None
    obv_slope_5d: float | None
    obv_slope_20d: float | None
    mfi_14: float | None
    price_range_pct: float | None
    volume_price_confirmation: bool
    accumulation_score: float
    interest_score: float
    accumulation_signals: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    passed_filters: list[str] = field(default_factory=list)
    failed_filters: list[str] = field(default_factory=list)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    strategy: str = "volume_spike_strict"
    rsi_14: float | None = None
    return_5d_pct: float | None = None
    return_10d_pct: float | None = None
    filter_passed: bool = False
    filter_reasons: list[str] = field(default_factory=list)
    failed_reasons: list[str] = field(default_factory=list)
    data_latest_date: str | None = None
    data_lag_days: int | None = None
    history_rows: int = 0
    ohlcv_cache_fetched_at: str | None = None
    ohlcv_cache_age_minutes: float | None = None
    ohlcv_cache_status: str = "missing"
    selected_config: str | None = None
    freshness_warning: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Strategy": self.strategy,
            "Interest Score": self.interest_score,
            "Last Close": self.last_close,
            "Volume": self.volume,
            "Turnover": self.turnover_try,
            "Avg Turnover 20D": self.avg_turnover_20d,
            "MA20": self.ma20,
            "RSI 14": self.rsi_14,
            "Return 5D %": self.return_5d_pct,
            "Return 10D %": self.return_10d_pct,
            "Daily Return %": self.daily_return_pct,
            "Volume Ratio 20D": self.volume_ratio_20d,
            "Turnover Ratio 20D": self.turnover_ratio_20d,
            "CMF 20": self.cmf_20,
            "OBV Slope 5D": self.obv_slope_5d,
            "OBV Slope 20D": self.obv_slope_20d,
            "ADL Slope 5D": self.adl_slope_5d,
            "ADL Slope 20D": self.adl_slope_20d,
            "MFI 14": self.mfi_14,
            "Price Range %": self.price_range_pct,
            "Accumulation Score": self.accumulation_score,
            "Accumulation Signals": ", ".join(self.accumulation_signals),
            "XU100 Relative %": self.xu100_relative_return_pct,
            "Close Position": self.close_position,
            "Breakout 20D": self.breakout_20d,
            "Above MA20": self.above_ma20,
            "Data Date": self.data_latest_date,
            "Data Lag Days": self.data_lag_days,
            "OHLCV Cache Status": self.ohlcv_cache_status,
            "OHLCV Fetched At": self.ohlcv_cache_fetched_at,
            "Freshness Warning": self.freshness_warning,
            "Filter Passed": self.filter_passed,
            "Filter Reasons": ", ".join(self.filter_reasons),
            "Failed Reasons": ", ".join(self.failed_reasons),
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


def _money_flow_multiplier(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    spread = (high - low).where((high - low) != 0)
    mfm = (((close - low) - (high - close)) / spread).fillna(0.0)
    return mfm


def _cmf_20(df: pd.DataFrame) -> float | None:
    if len(df) < 20:
        return None
    mfm = _money_flow_multiplier(df)
    mfv = mfm * df["volume"].astype(float)
    mfv20 = mfv.tail(20).sum()
    vol20 = df["volume"].astype(float).tail(20).sum()
    if vol20 == 0:
        return None
    return _safe_float(mfv20 / vol20)


def _adl_series(df: pd.DataFrame) -> pd.Series:
    mfm = _money_flow_multiplier(df)
    mfv = mfm * df["volume"].astype(float)
    return mfv.cumsum()


def _obv_series(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    delta = close.diff()
    direction = delta.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).fillna(0.0)
    return (direction * volume).cumsum()


def _series_slope(series: pd.Series, days: int) -> float | None:
    needed = days + 1
    clean = series.dropna()
    if len(clean) < needed:
        return None
    return _safe_float(clean.iloc[-1] - clean.iloc[-needed])


def _mfi_14(df: pd.DataFrame) -> float | None:
    if len(df) < 15:
        return None
    tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    rmf = tp * df["volume"].astype(float)
    tp_prev = tp.shift(1)
    pos = rmf.where(tp > tp_prev, 0.0)
    neg = rmf.where(tp < tp_prev, 0.0)
    pos14 = _safe_float(pos.tail(14).sum())
    neg14 = _safe_float(neg.tail(14).sum())
    if pos14 is None or neg14 is None:
        return None
    if neg14 == 0:
        if pos14 > 0:
            return 100.0
        return None
    ratio = pos14 / neg14
    value = 100 - (100 / (1 + ratio))
    return _safe_float(value)


def _rsi_14(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "close" not in df.columns or len(df) < 15:
        return None
    close = df["close"].astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.tail(14).mean()
    avg_loss = losses.tail(14).mean()
    gain_val = _safe_float(avg_gain)
    loss_val = _safe_float(avg_loss)
    if gain_val is None or loss_val is None:
        return None
    if loss_val == 0:
        if gain_val > 0:
            return 100.0
        return None
    rs = gain_val / loss_val
    return _safe_float(100 - (100 / (1 + rs)))


def _return_pct_from_close(df: pd.DataFrame, lookback_days: int) -> float | None:
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = df["close"].astype(float).dropna()
    needed = lookback_days + 1
    if len(close) < needed:
        return None
    latest = _safe_float(close.iloc[-1])
    base = _safe_float(close.iloc[-needed])
    if latest is None or base in (None, 0):
        return None
    return ((latest / base) - 1) * 100


def calculate_accumulation_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    v = metrics.get("volume_ratio_20d")
    t = metrics.get("turnover_ratio_20d")
    d = metrics.get("daily_return_pct")
    c = metrics.get("close_position")
    cmf = metrics.get("cmf_20")
    obv5 = metrics.get("obv_slope_5d")
    obv20 = metrics.get("obv_slope_20d")
    adl5 = metrics.get("adl_slope_5d")
    mfi = metrics.get("mfi_14")
    if isinstance(v, (int, float)) and v >= 1.5:
        score += 10
    if isinstance(v, (int, float)) and v >= 2.0:
        score += 10
    if isinstance(t, (int, float)) and t >= 1.2:
        score += 10
    if isinstance(t, (int, float)) and t >= 1.5:
        score += 5
    if isinstance(d, (int, float)) and d >= 0:
        score += 10
    if isinstance(c, (int, float)) and c >= 0.60:
        score += 10
    if isinstance(c, (int, float)) and c >= 0.70:
        score += 5
    if isinstance(cmf, (int, float)) and cmf > 0:
        score += 15
    if isinstance(cmf, (int, float)) and cmf > 0.05:
        score += 10
    if isinstance(obv5, (int, float)) and obv5 > 0:
        score += 10
    if isinstance(obv20, (int, float)) and obv20 > 0:
        score += 5
    if isinstance(adl5, (int, float)) and adl5 > 0:
        score += 5
    if isinstance(mfi, (int, float)) and 50 <= mfi <= 85:
        score += 10
    return min(score, 100.0)


def build_accumulation_signals(metrics: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    if isinstance(metrics.get("volume_ratio_20d"), (int, float)) and metrics["volume_ratio_20d"] >= 1.5:
        signals.append("rvol_high")
    if isinstance(metrics.get("turnover_ratio_20d"), (int, float)) and metrics["turnover_ratio_20d"] >= 1.2:
        signals.append("turnover_high")
    if isinstance(metrics.get("cmf_20"), (int, float)) and metrics["cmf_20"] > 0:
        signals.append("cmf_positive")
    if isinstance(metrics.get("obv_slope_5d"), (int, float)) and metrics["obv_slope_5d"] > 0:
        signals.append("obv_rising")
    if isinstance(metrics.get("adl_slope_5d"), (int, float)) and metrics["adl_slope_5d"] > 0:
        signals.append("adl_rising")
    if isinstance(metrics.get("close_position"), (int, float)) and metrics["close_position"] >= 0.60:
        signals.append("strong_close")
    if isinstance(metrics.get("mfi_14"), (int, float)) and 50 <= metrics["mfi_14"] <= 85:
        signals.append("mfi_confirmed")
    return signals


def apply_scan_mode_presets(config: RadarConfig) -> RadarConfig:
    mode = config.scan_mode
    if mode == "volume_spike":
        return replace(
            config,
            min_volume_ratio_active=True,
            min_volume_ratio=max(config.min_volume_ratio, 1.5),
            min_turnover_ratio_active=True,
            min_turnover_ratio=max(config.min_turnover_ratio, 1.2),
            min_cmf_20_active=False,
            require_obv_slope_5d_positive=False,
            require_obv_slope_20d_positive=False,
            min_mfi_14_active=False,
            max_mfi_14_active=False,
            min_accumulation_score_active=False,
        )
    if mode == "positive_money_flow":
        return replace(
            config,
            min_volume_ratio_active=True,
            min_volume_ratio=max(config.min_volume_ratio, 1.5),
            min_turnover_ratio_active=True,
            min_turnover_ratio=max(config.min_turnover_ratio, 1.2),
            min_daily_return_active=True,
            min_daily_return=max(config.min_daily_return, 0.0),
            min_close_position_active=True,
            min_close_position=max(config.min_close_position, 0.60),
            min_cmf_20_active=True,
            min_cmf_20=max(config.min_cmf_20, 0.0),
            require_obv_slope_5d_positive=True,
            min_mfi_14_active=True,
            min_mfi_14=max(config.min_mfi_14, 50.0),
            max_mfi_14_active=True,
            max_mfi_14=min(config.max_mfi_14, 85.0),
            min_accumulation_score_active=True,
            min_accumulation_score=max(config.min_accumulation_score, 50.0),
        )
    if mode == "silent_accumulation":
        return replace(
            config,
            min_volume_ratio_active=True,
            min_volume_ratio=max(config.min_volume_ratio, 1.5),
            min_turnover_ratio_active=True,
            min_turnover_ratio=max(config.min_turnover_ratio, 1.2),
            min_daily_return_active=True,
            min_daily_return=max(config.min_daily_return, -1.0),
            max_daily_return_active=True,
            max_daily_return_pct=min(config.max_daily_return_pct, 2.0),
            min_close_position_active=True,
            min_close_position=max(config.min_close_position, 0.50),
            min_cmf_20_active=True,
            min_cmf_20=max(config.min_cmf_20, 0.0),
            require_obv_slope_5d_positive=True,
            require_obv_slope_20d_positive=True,
            max_price_range_active=True,
            max_price_range_pct=min(config.max_price_range_pct, 5.0),
            min_accumulation_score_active=True,
            min_accumulation_score=max(config.min_accumulation_score, 50.0),
        )
    if mode == "strong_momentum":
        return replace(
            config,
            min_avg_turnover_try_active=True,
            min_avg_turnover_try=max(config.min_avg_turnover_try, 30_000_000.0),
            min_volume_ratio_active=True,
            min_volume_ratio=max(config.min_volume_ratio, 2.0),
            min_turnover_ratio_active=True,
            min_turnover_ratio=max(config.min_turnover_ratio, 1.5),
            min_daily_return_active=True,
            min_daily_return=max(config.min_daily_return, 1.0),
            min_close_position_active=True,
            min_close_position=max(config.min_close_position, 0.70),
            min_cmf_20_active=True,
            min_cmf_20=max(config.min_cmf_20, 0.05),
            require_obv_slope_5d_positive=True,
            min_mfi_14_active=True,
            min_mfi_14=max(config.min_mfi_14, 55.0),
            require_ma20_active=True,
            min_xu100_relative_active=True,
            min_xu100_relative=max(config.min_xu100_relative, 0.0),
            min_accumulation_score_active=True,
            min_accumulation_score=max(config.min_accumulation_score, 60.0),
        )
    return config


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
    if config.max_daily_return_active:
        value = metrics.get("daily_return_pct")
        if isinstance(value, (int, float)) and value <= config.max_daily_return_pct:
            passed.append("max_daily_return_pct")
        else:
            failed.append("max_daily_return_pct")

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
        if bool(metrics.get("close")) and bool(metrics.get("high_20d")) and float(metrics["close"]) >= float(metrics["high_20d"]) * 0.98:
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
    if config.min_cmf_20_active:
        value = metrics.get("cmf_20")
        if isinstance(value, (int, float)) and value > config.min_cmf_20:
            passed.append("min_cmf_20")
        else:
            failed.append("min_cmf_20")
    if config.require_obv_slope_5d_positive:
        value = metrics.get("obv_slope_5d")
        if isinstance(value, (int, float)) and value > 0:
            passed.append("obv_slope_5d_positive")
        else:
            failed.append("obv_slope_5d_positive")
    if config.require_obv_slope_20d_positive:
        value = metrics.get("obv_slope_20d")
        if isinstance(value, (int, float)) and value > 0:
            passed.append("obv_slope_20d_positive")
        else:
            failed.append("obv_slope_20d_positive")
    if config.min_mfi_14_active:
        value = metrics.get("mfi_14")
        if isinstance(value, (int, float)) and value >= config.min_mfi_14:
            passed.append("min_mfi_14")
        else:
            failed.append("min_mfi_14")
    if config.max_mfi_14_active:
        value = metrics.get("mfi_14")
        if isinstance(value, (int, float)) and value <= config.max_mfi_14:
            passed.append("max_mfi_14")
        else:
            failed.append("max_mfi_14")
    if config.max_price_range_active:
        value = metrics.get("price_range_pct")
        if isinstance(value, (int, float)) and value <= config.max_price_range_pct:
            passed.append("max_price_range_pct")
        else:
            failed.append("max_price_range_pct")
    if config.min_accumulation_score_active:
        value = metrics.get("accumulation_score")
        if isinstance(value, (int, float)) and value >= config.min_accumulation_score:
            passed.append("min_accumulation_score")
        else:
            failed.append("min_accumulation_score")

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


def apply_active_volume_spike_quality_filters(
    metrics: dict[str, Any],
    *,
    config: RadarConfig,
) -> tuple[bool, list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []

    volume_ratio = metrics.get("volume_ratio_20d")
    turnover_ratio = metrics.get("turnover_ratio_20d")
    if isinstance(volume_ratio, (int, float)) and volume_ratio >= 1.5 and isinstance(turnover_ratio, (int, float)) and turnover_ratio >= 1.2:
        passed.append("volume_spike_strict")
    else:
        failed.append("volume_spike_strict")

    if config.min_last_turnover_try_active:
        value = metrics.get("turnover_try")
        if isinstance(value, (int, float)) and value >= config.min_last_turnover_try:
            passed.append("min_last_turnover_try")
        else:
            failed.append("min_last_turnover_try")

    if config.min_avg_turnover_20d_try_active:
        value = metrics.get("avg_turnover_20d")
        if isinstance(value, (int, float)) and value >= config.min_avg_turnover_20d_try:
            passed.append("min_avg_turnover_20d_try")
        else:
            failed.append("min_avg_turnover_20d_try")

    close_value = metrics.get("close")
    ma20_value = metrics.get("ma20")
    ma20_passed = bool(metrics.get("above_ma20"))
    if not ma20_passed and float(config.min_above_ma20_ratio) < 1.0:
        ma20_passed = (
            isinstance(close_value, (int, float))
            and isinstance(ma20_value, (int, float))
            and close_value >= ma20_value * float(config.min_above_ma20_ratio)
        )
    if ma20_passed:
        passed.append("above_ma20")
    else:
        failed.append("above_ma20")

    if config.max_rsi_14_active:
        value = metrics.get("rsi_14")
        if isinstance(value, (int, float)) and value <= config.max_rsi_14 and value <= 80:
            passed.append("max_rsi_14")
        else:
            failed.append("max_rsi_14")

    if config.max_return_5d_active:
        value = metrics.get("return_5d_pct")
        if isinstance(value, (int, float)) and value <= config.max_return_5d_pct:
            passed.append("max_return_5d_pct")
        else:
            failed.append("max_return_5d_pct")

    if config.max_return_10d_active:
        value = metrics.get("return_10d_pct")
        if isinstance(value, (int, float)) and value <= config.max_return_10d_pct:
            passed.append("max_return_10d_pct")
        else:
            failed.append("max_return_10d_pct")

    if config.require_strong_close:
        value = metrics.get("close_position")
        if isinstance(value, (int, float)) and value >= config.min_close_position:
            passed.append("strong_close")
        else:
            failed.append("strong_close")

    return len(failed) == 0, passed, failed


def evaluate_symbol(
    symbol: str,
    history: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    config: RadarConfig | None = None,
    history_meta: HistoryLoadResult | None = None,
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
    price_range_pct = day_range_pct

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
    cmf20 = _cmf_20(df)
    adl = _adl_series(df)
    adl_slope_5d = _series_slope(adl, 5)
    adl_slope_20d = _series_slope(adl, 20)
    obv = _obv_series(df)
    obv_slope_5d = _series_slope(obv, 5)
    obv_slope_20d = _series_slope(obv, 20)
    mfi14 = _mfi_14(df)
    rsi14 = _rsi_14(df)
    return_5d_pct = _return_pct_from_close(df, 5)
    return_10d_pct = _return_pct_from_close(df, 10)
    volume_price_confirmation = bool(
        isinstance(volume_ratio_20d, (int, float))
        and volume_ratio_20d >= 1.5
        and isinstance(daily_return_pct, (int, float))
        and daily_return_pct >= 0
        and isinstance(close_position, (int, float))
        and close_position >= 0.60
    )

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
        "cmf_20": cmf20,
        "adl_slope_5d": adl_slope_5d,
        "adl_slope_20d": adl_slope_20d,
        "obv_slope_5d": obv_slope_5d,
        "obv_slope_20d": obv_slope_20d,
        "mfi_14": mfi14,
        "rsi_14": rsi14,
        "return_5d_pct": return_5d_pct,
        "return_10d_pct": return_10d_pct,
        "price_range_pct": price_range_pct,
        "volume_price_confirmation": volume_price_confirmation,
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
        "data_latest_date": history_meta.data_latest_date if history_meta else None,
        "data_lag_days": history_meta.data_lag_days if history_meta else None,
        "history_rows": history_meta.history_rows if history_meta else len(df),
        "ohlcv_cache_fetched_at": history_meta.ohlcv_cache_fetched_at if history_meta else None,
        "ohlcv_cache_age_minutes": history_meta.ohlcv_cache_age_minutes if history_meta else None,
        "ohlcv_cache_status": history_meta.ohlcv_cache_status if history_meta else "missing",
        "freshness_warning": bool((history_meta.data_lag_days if history_meta else None) is not None and (history_meta.data_lag_days if history_meta else 0) > 3),
    }
    interest_score = calculate_interest_score(metrics)
    accumulation_score = calculate_accumulation_score(metrics)
    metrics["interest_score"] = interest_score
    metrics["accumulation_score"] = accumulation_score
    signals = build_signals(metrics)
    accumulation_signals = build_accumulation_signals(metrics)
    passed_filters, failed_filters = evaluate_filters(metrics, config)
    filter_passed, filter_reasons, failed_reasons = apply_active_volume_spike_quality_filters(metrics, config=config)
    if config.active_volume_spike_quality_active and not filter_passed:
        failed_filters = sorted(set(failed_filters + ["active_volume_spike_quality"]))

    return RadarResult(
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
        cmf_20=cmf20,
        adl_slope_5d=adl_slope_5d,
        adl_slope_20d=adl_slope_20d,
        obv_slope_5d=obv_slope_5d,
        obv_slope_20d=obv_slope_20d,
        mfi_14=mfi14,
        rsi_14=rsi14,
        return_5d_pct=return_5d_pct,
        return_10d_pct=return_10d_pct,
        price_range_pct=price_range_pct,
        volume_price_confirmation=volume_price_confirmation,
        accumulation_score=accumulation_score,
        accumulation_signals=accumulation_signals,
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
        filter_passed=filter_passed,
        filter_reasons=filter_reasons,
        failed_reasons=failed_reasons,
        data_latest_date=metrics["data_latest_date"],
        data_lag_days=metrics["data_lag_days"],
        history_rows=metrics["history_rows"],
        ohlcv_cache_fetched_at=metrics["ohlcv_cache_fetched_at"],
        ohlcv_cache_age_minutes=metrics["ohlcv_cache_age_minutes"],
        ohlcv_cache_status=metrics["ohlcv_cache_status"],
        freshness_warning=metrics["freshness_warning"],
    )


@dataclass
class ScanResult:
    results: list[RadarResult]
    raw_results: list[RadarResult]
    failed_symbols: list[dict[str, str]]
    scan_summary: dict[str, Any]


def _result_to_cache_row(result: RadarResult) -> dict[str, Any]:
    return asdict(result)


def _result_from_cache_row(row: dict[str, Any]) -> RadarResult:
    row.setdefault("cmf_20", None)
    row.setdefault("adl_slope_5d", None)
    row.setdefault("adl_slope_20d", None)
    row.setdefault("obv_slope_5d", None)
    row.setdefault("obv_slope_20d", None)
    row.setdefault("mfi_14", None)
    row.setdefault("price_range_pct", row.get("day_range_pct"))
    row.setdefault("volume_price_confirmation", False)
    row.setdefault("accumulation_score", 0.0)
    row.setdefault("accumulation_signals", [])
    row.setdefault("strategy", "volume_spike_strict")
    row.setdefault("rsi_14", None)
    row.setdefault("return_5d_pct", None)
    row.setdefault("return_10d_pct", None)
    row.setdefault("filter_passed", False)
    row.setdefault("filter_reasons", [])
    row.setdefault("failed_reasons", [])
    row.setdefault("selected_config", None)
    return RadarResult(**row)


def _build_scan_cache_payload(scan: ScanResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    return (
        [_result_to_cache_row(item) for item in scan.results],
        [_result_to_cache_row(item) for item in scan.raw_results],
        scan.failed_symbols,
        scan.scan_summary,
    )


def build_scan_cache_key(symbols: list[str], config: RadarConfig) -> str:
    normalized_sorted = sorted(normalize_bist_symbol(item) for item in symbols if normalize_bist_symbol(item))
    key_payload = {
        "cache_schema_version": 3,
        "index_symbol": normalize_bist_symbol(config.index_symbol),
        "symbols_hash": hashlib.sha256(",".join(normalized_sorted).encode("utf-8")).hexdigest(),
        "lookback_days": config.lookback_days,
        "filters": {
            "min_avg_turnover_try_active": config.min_avg_turnover_try_active,
            "min_avg_turnover_try": config.min_avg_turnover_try,
            "min_volume_ratio_active": config.min_volume_ratio_active,
            "min_volume_ratio": config.min_volume_ratio,
            "min_turnover_ratio_active": config.min_turnover_ratio_active,
            "min_turnover_ratio": config.min_turnover_ratio,
            "min_daily_return_active": config.min_daily_return_active,
            "min_daily_return": config.min_daily_return,
            "min_close_position_active": config.min_close_position_active,
            "min_close_position": config.min_close_position,
            "breakout_mode": config.breakout_mode,
            "require_ma20_active": config.require_ma20_active,
            "require_ma50_active": config.require_ma50_active,
            "min_xu100_relative_active": config.min_xu100_relative_active,
            "min_xu100_relative": config.min_xu100_relative,
            "min_interest_score_active": config.min_interest_score_active,
            "min_interest_score": config.min_interest_score,
            "scan_mode": config.scan_mode,
            "min_cmf_20_active": config.min_cmf_20_active,
            "min_cmf_20": config.min_cmf_20,
            "require_obv_slope_5d_positive": config.require_obv_slope_5d_positive,
            "require_obv_slope_20d_positive": config.require_obv_slope_20d_positive,
            "min_mfi_14_active": config.min_mfi_14_active,
            "min_mfi_14": config.min_mfi_14,
            "max_mfi_14_active": config.max_mfi_14_active,
            "max_mfi_14": config.max_mfi_14,
            "max_daily_return_active": config.max_daily_return_active,
            "max_daily_return_pct": config.max_daily_return_pct,
            "max_price_range_active": config.max_price_range_active,
            "max_price_range_pct": config.max_price_range_pct,
            "min_accumulation_score_active": config.min_accumulation_score_active,
            "min_accumulation_score": config.min_accumulation_score,
            "include_negative_moves": config.include_negative_moves,
            "ohlcv_cache_ttl_minutes": config.ohlcv_cache_ttl_minutes,
            "active_volume_spike_quality_active": config.active_volume_spike_quality_active,
            "min_last_turnover_try_active": config.min_last_turnover_try_active,
            "min_last_turnover_try": config.min_last_turnover_try,
            "min_avg_turnover_20d_try_active": config.min_avg_turnover_20d_try_active,
            "min_avg_turnover_20d_try": config.min_avg_turnover_20d_try,
            "max_rsi_14_active": config.max_rsi_14_active,
            "max_rsi_14": config.max_rsi_14,
            "max_return_5d_active": config.max_return_5d_active,
            "max_return_5d_pct": config.max_return_5d_pct,
            "max_return_10d_active": config.max_return_10d_active,
            "max_return_10d_pct": config.max_return_10d_pct,
            "require_strong_close": config.require_strong_close,
            "min_above_ma20_ratio": config.min_above_ma20_ratio,
        },
    }
    payload_text = json.dumps(key_payload, sort_keys=True)
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def _passes_result(result: RadarResult) -> bool:
    return len(result.failed_filters) == 0


def scan_symbols(
    symbols: list[str],
    *,
    config: RadarConfig | None = None,
    client: BorsapyMarketDataClient | None = None,
    progress_callback: Any | None = None,
    universe_source: str = "borsapy",
) -> ScanResult:
    config = config or RadarConfig()
    client = client or BorsapyMarketDataClient()
    normalized_symbols = [normalize_bist_symbol(item) for item in symbols if normalize_bist_symbol(item)]
    total = len(normalized_symbols)
    started_at = time.perf_counter()
    ohlcv_ttl = config.ohlcv_cache_ttl_minutes if config.ohlcv_cache_ttl_minutes is not None else get_default_ohlcv_cache_ttl_minutes()
    scan_cache_source = "live_scan"

    if config.use_scan_cache and not config.force_refresh:
        cache_key = build_scan_cache_key(normalized_symbols, config)
        cached = get_cached_scan_result(config.db_path, cache_key, max_age_minutes=config.scan_cache_ttl_minutes)
        if cached is not None:
            results = [_result_from_cache_row(row) for row in cached["results"]]
            raw_results = [_result_from_cache_row(row) for row in cached["raw_results"]]
            summary = dict(cached["scan_summary"])
            summary["scan_cache_source"] = "scan_cache"
            summary["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
            return ScanResult(
                results=results,
                raw_results=raw_results,
                failed_symbols=list(cached["failed_symbols"]),
                scan_summary=summary,
            )

    benchmark = client.load_history(
        "XU100",
        lookback_days=config.lookback_days,
        db_path=config.db_path,
        force=config.force_refresh,
        cache_ttl_minutes=ohlcv_ttl,
    )

    # Regime-aware config selection
    regime_detector = MarketRegimeDetector(db_path=config.db_path)
    regime_detector._xu100_df = regime_detector.load_xu100_data(client=client)
    latest_date_str = str(benchmark.index.max())[:10] if not benchmark.empty else datetime.now().date().isoformat()
    regime_info = regime_detector.detect_regime(latest_date_str, client=client)

    if regime_info["return_20d_pct"] > 1.5:
        selected_config_name = "current_config"
        min_close_pos = 0.60
    else:
        selected_config_name = "relaxed_strong_close"
        min_close_pos = 0.50

    config = replace(config, min_close_position=min_close_pos)

    all_results: list[RadarResult] = []
    failed_symbols: list[dict[str, str]] = []
    saved_rows: list[dict[str, Any]] = []

    def _worker(symbol: str) -> tuple[str, RadarResult | None, str | None]:
        try:
            if hasattr(client, "load_history_with_meta"):
                history_meta = client.load_history_with_meta(
                    symbol,
                    lookback_days=config.lookback_days,
                    db_path=config.db_path,
                    force=config.force_refresh,
                    cache_ttl_minutes=ohlcv_ttl,
                )
                history = history_meta.frame
            else:
                history = client.load_history(
                    symbol,
                    lookback_days=config.lookback_days,
                    db_path=config.db_path,
                    force=config.force_refresh,
                    cache_ttl_minutes=ohlcv_ttl,
                )
                fallback_latest_date = None
                fallback_lag_days = None
                if history is not None and not history.empty:
                    idx = history.index.max()
                    if pd.notna(idx):
                        fallback_latest_date = pd.Timestamp(idx).date().isoformat()
                        fallback_lag_days = (datetime.now().date() - pd.Timestamp(idx).date()).days
                history_meta = HistoryLoadResult(
                    frame=history,
                    symbol=normalize_bist_symbol(symbol),
                    data_latest_date=fallback_latest_date,
                    data_lag_days=fallback_lag_days,
                    history_rows=len(history.index) if history is not None else 0,
                    ohlcv_cache_fetched_at=None,
                    ohlcv_cache_age_minutes=None,
                    ohlcv_cache_status="missing",
                    source="legacy_client",
                )
            if history.empty:
                return symbol, None, "empty_history"
            result = evaluate_symbol(symbol, history, benchmark, config=config, history_meta=history_meta)
            if result is not None:
                result.selected_config = selected_config_name
            return symbol, result, None
        except Exception as exc:  # noqa: BLE001
            return symbol, None, str(exc)

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(config.max_workers))) as executor:
        future_map = {executor.submit(_worker, symbol): symbol for symbol in normalized_symbols}
        for future in as_completed(future_map):
            symbol, result, error = future.result()
            completed += 1

            if error is not None:
                failed_symbols.append({"symbol": symbol, "error": error})
            elif result is not None:
                all_results.append(result)
                saved_rows.append(
                    {
                        "symbol": result.symbol,
                        "scanned_at": datetime.now(UTC).isoformat(),
                        "source": "borsapy",
                        "interest_score": result.interest_score,
                        "metrics": asdict(result),
                        "signals": result.signals,
                        "passed_filters": result.passed_filters,
                        "failed_filters": result.failed_filters,
                    }
                )

            if progress_callback is not None:
                try:
                    progress_callback(completed, total, symbol)
                except Exception:  # noqa: BLE001
                    pass

    save_radar_results_bulk(config.db_path, saved_rows)
    passed = sorted((item for item in all_results if _passes_result(item)), key=lambda item: item.interest_score, reverse=True)
    all_results = sorted(all_results, key=lambda item: item.interest_score, reverse=True)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    scan_summary = {
        "index": normalize_bist_symbol(config.index_symbol),
        "universe_symbol_count": total,
        "scanned_symbols": len(all_results) + len(failed_symbols),
        "successful_symbols": len(all_results),
        "failed_symbols": len(failed_symbols),
        "result_count": len(passed),
        "universe_cache_source": universe_source,
        "scan_cache_source": scan_cache_source,
        "max_workers": int(config.max_workers),
        "ohlcv_cache_ttl_minutes": int(ohlcv_ttl),
        "elapsed_seconds": elapsed_seconds,
        "newest_data_date": None,
        "oldest_data_date": None,
        "max_data_lag_days": None,
        "scanned_stale_data_count": 0,
        "scanned_fresh_data_count": 0,
        "result_stale_data_count": 0,
        "result_fresh_data_count": 0,
        "regime_label": regime_info["regime_label"],
        "xu100_return_20d_pct": regime_info["return_20d_pct"],
        "xu100_close": regime_info["close"],
        "xu100_ma50": regime_info["ma50"],
        "xu100_ma200": regime_info["ma200"],
        "selected_config": selected_config_name,
    }

    dates = [item.data_latest_date for item in all_results if item.data_latest_date]
    lags = [item.data_lag_days for item in all_results if item.data_lag_days is not None]
    scanned_stale_count = sum(1 for item in all_results if item.freshness_warning)
    scanned_fresh_count = sum(1 for item in all_results if not item.freshness_warning)
    result_stale_count = sum(1 for item in passed if item.freshness_warning)
    result_fresh_count = sum(1 for item in passed if not item.freshness_warning)
    if dates:
        scan_summary["newest_data_date"] = max(dates)
        scan_summary["oldest_data_date"] = min(dates)
    if lags:
        scan_summary["max_data_lag_days"] = max(lags)
    scan_summary["scanned_stale_data_count"] = scanned_stale_count
    scan_summary["scanned_fresh_data_count"] = scanned_fresh_count
    scan_summary["result_stale_data_count"] = result_stale_count
    scan_summary["result_fresh_data_count"] = result_fresh_count
    # Backward compatibility keys.
    scan_summary["stale_data_count"] = result_stale_count
    scan_summary["fresh_data_count"] = result_fresh_count

    scan = ScanResult(
        results=passed,
        raw_results=all_results,
        failed_symbols=failed_symbols,
        scan_summary=scan_summary,
    )

    if config.use_scan_cache:
        cache_key = build_scan_cache_key(normalized_symbols, config)
        results_payload, raw_payload, failed_payload, summary_payload = _build_scan_cache_payload(scan)
        upsert_cached_scan_result(
            config.db_path,
            cache_key,
            universe_source=universe_source,
            universe_symbol_count=total,
            results=results_payload,
            raw_results=raw_payload,
            failed_symbols=failed_payload,
            scan_summary=summary_payload,
        )

    return scan

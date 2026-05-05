from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from market_radar.radar_engine import RadarConfig, calculate_interest_score, evaluate_filters, evaluate_symbol
from market_radar.symbols import normalize_bist_symbol


def _history_frame() -> pd.DataFrame:
    dates = pd.date_range(datetime.now(timezone.utc) - timedelta(days=40), periods=40, freq="D")
    opens = [100 + i for i in range(40)]
    closes = [101 + i for i in range(40)]
    highs = [c + 2 for c in closes]
    lows = [o - 2 for o in opens]
    volumes = [1_000 + i * 10 for i in range(40)]
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_volume_ratio_calculation() -> None:
    df = _history_frame()
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert result.volume_ratio_20d is not None
    assert result.avg_volume_20d is not None


def test_turnover_ratio_calculation() -> None:
    df = _history_frame()
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert result.turnover_ratio_20d is not None


def test_daily_return_pct() -> None:
    df = _history_frame()
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert result.daily_return_pct is not None


def test_close_position() -> None:
    df = _history_frame()
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert result.close_position is not None


def test_breakout_and_ma20() -> None:
    df = _history_frame()
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert isinstance(result.breakout_20d, bool)
    assert result.above_ma20 is True


def test_interest_score_reasonable_range() -> None:
    metrics = {
        "volume_ratio_20d": 2.0,
        "turnover_ratio_20d": 2.0,
        "daily_return_pct": 3.0,
        "close_position": 0.8,
        "breakout_20d": True,
        "above_ma20": True,
        "above_ma50": False,
        "near_52w_high": False,
        "xu100_relative_return_pct": 1.0,
        "sector_relative_return_pct": None,
    }
    score = calculate_interest_score(metrics)
    assert 65 <= score <= 80


def test_inactive_filter_not_failed() -> None:
    df = _history_frame()
    config = RadarConfig(require_ma50_active=False, include_negative_moves=True)
    result = evaluate_symbol("THYAO", df, None, config=config)
    assert "above_ma50" not in result.failed_filters


def test_include_negative_moves_false_excludes_negative() -> None:
    df = _history_frame().copy()
    df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-2]["close"] - 5
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=False))
    assert "negative_price_move" in result.failed_filters


def test_include_negative_moves_true_keeps_negative() -> None:
    df = _history_frame().copy()
    df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-2]["close"] - 5
    result = evaluate_symbol("THYAO", df, None, config=RadarConfig(include_negative_moves=True))
    assert "negative_price_move" not in result.failed_filters
    assert "positive_price_move" not in result.signals


def test_normalize_symbols() -> None:
    assert normalize_bist_symbol("odine") == "ODINE"
    assert normalize_bist_symbol("ODİNE") == "ODINE"


def test_filter_evaluation_min_score() -> None:
    metrics = {
        "avg_turnover_20d": 20_000_000.0,
        "volume_ratio_20d": 2.0,
        "turnover_ratio_20d": 2.0,
        "daily_return_pct": 3.0,
        "close_position": 0.8,
        "breakout_20d": True,
        "above_ma20": True,
        "above_ma50": False,
        "xu100_relative_return_pct": 2.0,
        "interest_score": 72.0,
    }
    passed, failed = evaluate_filters(metrics, RadarConfig())
    assert "min_interest_score" in passed
    assert "min_avg_turnover_try" in passed
    assert not failed or "above_ma50" in failed


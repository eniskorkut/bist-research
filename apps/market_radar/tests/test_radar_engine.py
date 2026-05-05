from __future__ import annotations

from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import pandas as pd

from market_radar.data_access import (
    BorsapyMarketDataClient,
    get_cached_universe,
    init_db,
    is_stale,
    load_bist_universe,
    upsert_cached_universe,
)
import market_radar.data_access as data_access
from market_radar.radar_engine import RadarConfig, ScanResult, calculate_interest_score, evaluate_filters, evaluate_symbol, scan_symbols
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


def test_borsapy_fetch_uses_naive_start_datetime(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            captured["symbol"] = symbol

        def history(self, *, start, interval: str):
            captured["start"] = start
            captured["interval"] = interval
            return pd.DataFrame(
                {"open": [1], "high": [2], "low": [1], "close": [2], "volume": [100]},
                index=pd.date_range("2026-01-01", periods=1, freq="D"),
            )

    monkeypatch.setattr(data_access.bp, "Ticker", FakeTicker)

    frame = BorsapyMarketDataClient()._fetch_history("thyao", 60)

    assert not frame.empty
    assert captured["symbol"] == "THYAO"
    assert captured["interval"] == "1d"
    assert captured["start"].tzinfo is None


def test_load_bist_universe_uses_xutum_components(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeIndex:
        def __init__(self, symbol: str) -> None:
            captured["index"] = symbol
            self.component_symbols = ["thyao", "ASELS", "ODİNE", "INVALIDLONG"]

    monkeypatch.setattr(data_access.bp, "Index", FakeIndex)
    db = str(tmp_path / "test_radar.sqlite")

    symbols, source = load_bist_universe("xutum", db_path=db, force=True)
    assert symbols == ["ASELS", "ODINE", "THYAO"]
    assert captured["index"] == "XUTUM"
    assert source == "borsapy"


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


# ========= Universe cache tests =========

def test_universe_cache_fresh_skips_borsapy(monkeypatch, tmp_path: Path) -> None:
    """When universe cache is fresh, borsapy should NOT be called."""
    db = str(tmp_path / "radar.sqlite")
    init_db(db)
    upsert_cached_universe(db, "XUTUM", ["THYAO", "ASELS", "GARAN"])

    borsapy_called = {"called": False}

    class FakeIndex:
        def __init__(self, symbol: str) -> None:
            borsapy_called["called"] = True
            self.component_symbols = ["SHOULD_NOT_APPEAR"]

    monkeypatch.setattr(data_access.bp, "Index", FakeIndex)

    symbols, source = load_bist_universe("XUTUM", db_path=db, force=False)
    assert source == "fresh_cache"
    assert symbols == ["THYAO", "ASELS", "GARAN"]
    assert not borsapy_called["called"]


def test_universe_cache_stale_fallback(monkeypatch, tmp_path: Path) -> None:
    """When borsapy fails and stale cache exists, stale cache is used as fallback."""
    db = str(tmp_path / "radar.sqlite")
    init_db(db)
    # Insert a cache entry with an old timestamp
    import sqlite3, json
    old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO universe_cache (index_symbol, fetched_at, symbols_json, source) VALUES (?, ?, ?, ?)",
            ("XUTUM", old_ts, json.dumps(["THYAO", "ASELS"]), "borsapy"),
        )

    class FakeIndexBroken:
        def __init__(self, symbol: str) -> None:
            raise RuntimeError("borsapy unavailable")

    monkeypatch.setattr(data_access.bp, "Index", FakeIndexBroken)

    symbols, source = load_bist_universe("XUTUM", db_path=db, force=False)
    assert source == "stale_cache"
    assert symbols == ["THYAO", "ASELS"]


# ========= Per-symbol error tolerance tests =========

def test_scan_continues_on_symbol_error(tmp_path: Path) -> None:
    """A single symbol error should not abort the entire scan."""
    db = str(tmp_path / "radar.sqlite")
    init_db(db)

    call_count = {"count": 0}

    class FakeClient:
        def load_history(self, symbol: str, lookback_days: int = 260, *, db_path: str = "", force: bool = False) -> pd.DataFrame:
            call_count["count"] += 1
            if symbol == "FAIL":
                raise RuntimeError("data unavailable")
            return _history_frame()

    config = RadarConfig(
        include_negative_moves=True,
        min_interest_score_active=False,
        db_path=db,
    )
    scan = scan_symbols(["THYAO", "FAIL", "ASELS"], config=config, client=FakeClient())

    assert isinstance(scan, ScanResult)
    assert len(scan.failed_symbols) == 1
    assert scan.failed_symbols[0]["symbol"] == "FAIL"
    assert "data unavailable" in scan.failed_symbols[0]["error"]
    # Successful symbols should still produce results
    successful_symbols = {r.symbol for r in scan.raw_results}
    assert "THYAO" in successful_symbols
    assert "ASELS" in successful_symbols
    assert scan.scan_summary["failed_symbols"] == 1
    assert scan.scan_summary["successful_symbols"] >= 2


def test_scan_empty_failed_symbols_on_success(tmp_path: Path) -> None:
    """When all symbols succeed, failed_symbols should be empty."""
    db = str(tmp_path / "radar.sqlite")
    init_db(db)

    class FakeClient:
        def load_history(self, symbol: str, lookback_days: int = 260, *, db_path: str = "", force: bool = False) -> pd.DataFrame:
            return _history_frame()

    config = RadarConfig(include_negative_moves=True, min_interest_score_active=False, db_path=db)
    scan = scan_symbols(["THYAO", "ASELS"], config=config, client=FakeClient())

    assert scan.failed_symbols == []
    assert scan.scan_summary["failed_symbols"] == 0
    assert scan.scan_summary["successful_symbols"] == 2

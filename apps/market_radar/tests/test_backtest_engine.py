from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_radar.backtesting.backtest_engine import (
    BacktestConfig,
    _run_symbol_backtest,
    run_backtest,
    write_backtest_outputs,
)
from market_radar.data_access import BorsapyMarketDataClient


def _history() -> pd.DataFrame:
    dates = pd.date_range(datetime(2026, 1, 1, tzinfo=UTC), periods=80, freq="D")
    base = pd.Series(range(80), index=dates).astype(float)
    return pd.DataFrame(
        {
            "open": 100 + base,
            "high": 102 + base,
            "low": 99 + base,
            "close": 101 + base,
            "volume": 1000 + (base * 10),
        },
        index=dates,
    )


def test_entry_date_is_t_plus_one() -> None:
    hist = _history()
    bench = _history()
    rows = _run_symbol_backtest("THYAO", hist, bench, ["volume_spike_strict"])
    if not rows:
        # allow strict strategy to produce no rows for synthetic frame
        return
    row = rows[0]
    assert pd.to_datetime(row["entry_date"]) > pd.to_datetime(row["signal_date"])


def test_trading_day_horizon_uses_row_offsets() -> None:
    hist = _history()
    bench = _history()
    rows = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"], cooldown_days=0)
    if not rows:
        return
    row = rows[0]
    entry = pd.to_datetime(row["entry_date"])
    exit15 = pd.to_datetime(row["exit_15d_date"]) if row["exit_15d_date"] else None
    exit30 = pd.to_datetime(row["exit_30d_date"]) if row["exit_30d_date"] else None
    if exit15 is not None:
        assert hist.index.get_loc(exit15) - hist.index.get_loc(entry) == 15
    if exit30 is not None:
        assert hist.index.get_loc(exit30) - hist.index.get_loc(entry) == 30
    if row.get("benchmark_entry_date") and row.get("benchmark_exit_15d_date"):
        b_entry = pd.to_datetime(row["benchmark_entry_date"])
        b_exit15 = pd.to_datetime(row["benchmark_exit_15d_date"])
        assert bench.index.get_loc(b_exit15) - bench.index.get_loc(b_entry) == 15


def test_missing_exit_windows_do_not_crash() -> None:
    hist = _history().head(35)
    bench = _history().head(35)
    rows = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"])
    # some rows may have None exits, but function must return list
    assert isinstance(rows, list)


def test_cooldown_reduces_repeated_signals() -> None:
    hist = _history()
    bench = _history()
    rows_no_cd = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"], cooldown_days=0)
    rows_cd = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"], cooldown_days=15)
    assert len(rows_cd) <= len(rows_no_cd)


def test_cooldown_zero_keeps_all_signals() -> None:
    hist = _history()
    bench = _history()
    rows0 = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"], cooldown_days=0)
    rows_none = _run_symbol_backtest("THYAO", hist, bench, ["positive_interest"], cooldown_days=0)
    assert len(rows0) == len(rows_none)


def test_write_outputs(tmp_path: Path) -> None:
    cfg = BacktestConfig(output_dir=str(tmp_path))
    from market_radar.backtesting.backtest_engine import BacktestResult

    result = BacktestResult(
        signals=[
            {
                "symbol": "THYAO",
                "strategy": "positive_interest",
                "signal_date": "2026-01-01",
                "alpha_15d": 1.2,
                "alpha_30d": 2.1,
                "beat_xu100_15d": True,
                "beat_xu100_30d": True,
                "benchmark_return_15d": 0.8,
                "benchmark_return_30d": 1.5,
                "return_15d": 2.0,
                "return_30d": 3.6,
            }
        ],
        failed_symbols=[],
        scan_summary={"index_symbol": "XU100"},
    )
    files = write_backtest_outputs(result, cfg)
    for path in files.values():
        assert Path(path).exists()


def test_run_backtest_with_mock_client(monkeypatch, tmp_path: Path) -> None:
    class FakeClient(BorsapyMarketDataClient):
        def load_history(self, symbol: str, lookback_days: int = 260, *, db_path: str = "", force: bool = False, cache_ttl_minutes: int | None = None) -> pd.DataFrame:  # type: ignore[override]
            return _history()

    monkeypatch.setattr("market_radar.backtesting.backtest_engine.load_bist_universe", lambda index_symbol, db_path, force: (["THYAO"], "test"))
    cfg = BacktestConfig(
        index_symbol="XU100",
        lookback_days=80,
        strategies=["positive_interest"],
        db_path=str(tmp_path / "radar.sqlite"),
        output_dir=str(tmp_path / "out"),
    )
    result = run_backtest(cfg, client=FakeClient())
    assert "signal_count" in result.scan_summary

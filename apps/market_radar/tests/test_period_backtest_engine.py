from datetime import date
from pathlib import Path

import pandas as pd

from market_radar.backtesting.backtest_engine import BacktestResult
from market_radar.backtesting.period_backtest_engine import (
    PeriodBacktestConfig,
    build_period_windows,
    run_period_backtest,
    write_period_outputs,
)


def _hist() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=900, freq="B")
    base = pd.Series(range(len(idx)), index=idx).astype(float)
    return pd.DataFrame(
        {
            "open": 100 + base,
            "high": 102 + base,
            "low": 99 + base,
            "close": 101 + base,
            "volume": 1000 + (base * 5),
        },
        index=idx,
    )


def test_build_period_windows_auto_end() -> None:
    windows = build_period_windows(["2026-01-01", "2026-02-01", "2026-03-01"])
    assert len(windows) == 3
    assert windows[0].period_start == date(2026, 1, 1)
    assert windows[0].period_end == date(2026, 2, 1)
    assert windows[1].period_end == date(2026, 3, 1)
    assert windows[2].period_end == date(2026, 4, 1)


def _signals() -> list[dict]:
    return [
        {
            "symbol": "AAA",
            "strategy": "positive_interest",
            "signal_date": "2026-01-02",
            "entry_date": "2026-01-03",
            "entry_close": 10.0,
            "exit_15d_date": "2026-01-18",
            "exit_15d_close": 11.0,
            "return_15d": 10.0,
            "benchmark_return_15d": 2.0,
            "alpha_15d": 8.0,
            "beat_xu100_15d": True,
            "exit_30d_date": "2026-02-02",
            "exit_30d_close": 12.0,
            "return_30d": 20.0,
            "benchmark_return_30d": 4.0,
            "alpha_30d": 16.0,
            "beat_xu100_30d": True,
            "interest_score": 80.0,
            "volume_ratio_20d": 2.0,
            "turnover_ratio_20d": 1.8,
            "close_position": 0.7,
            "daily_return_pct": 2.0,
            "near_20d_high_pct": 1.0,
            "above_ma20": True,
            "above_ma50": True,
        },
        {
            "symbol": "AAA",
            "strategy": "positive_interest",
            "signal_date": "2026-01-10",
            "entry_date": "2026-01-11",
            "entry_close": 11.0,
            "exit_15d_date": "2026-01-26",
            "exit_15d_close": 10.5,
            "return_15d": -4.5,
            "benchmark_return_15d": 1.0,
            "alpha_15d": -5.5,
            "beat_xu100_15d": False,
            "exit_30d_date": "2026-02-10",
            "exit_30d_close": 11.5,
            "return_30d": 4.5,
            "benchmark_return_30d": 3.0,
            "alpha_30d": 1.5,
            "beat_xu100_30d": True,
            "interest_score": 60.0,
            "volume_ratio_20d": 1.6,
            "turnover_ratio_20d": 1.3,
            "close_position": 0.55,
            "daily_return_pct": 0.5,
            "near_20d_high_pct": 2.0,
            "above_ma20": True,
            "above_ma50": False,
        },
        {
            "symbol": "BBB",
            "strategy": "positive_interest",
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-06",
            "entry_close": 20.0,
            "exit_15d_date": "2026-01-21",
            "exit_15d_close": 22.0,
            "return_15d": 10.0,
            "benchmark_return_15d": 2.0,
            "alpha_15d": 8.0,
            "beat_xu100_15d": True,
            "exit_30d_date": None,
            "exit_30d_close": None,
            "return_30d": None,
            "benchmark_return_30d": None,
            "alpha_30d": None,
            "beat_xu100_30d": None,
            "interest_score": 75.0,
            "volume_ratio_20d": 2.2,
            "turnover_ratio_20d": 1.7,
            "close_position": 0.8,
            "daily_return_pct": 3.0,
            "near_20d_high_pct": 0.8,
            "above_ma20": True,
            "above_ma50": True,
        },
    ]


def test_first_signal_per_symbol_and_metrics(monkeypatch) -> None:
    fake = BacktestResult(signals=_signals(), failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)
    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        lambda self, symbol, lookback_days=260, db_path="", force=False: _hist(),
    )

    cfg = PeriodBacktestConfig(
        period_starts=["2026-01-01"],
        strategies=["positive_interest"],
        basket_mode="first_signal_per_symbol",
    )
    result = run_period_backtest(cfg)
    holdings = result.holdings
    assert len(holdings) == 2
    assert sorted(holdings["symbol"].unique().tolist()) == ["AAA", "BBB"]
    summary = result.period_strategy_summary.iloc[0]
    assert summary["signal_count"] == 2
    assert summary["unique_symbol_count"] == 2
    assert summary["basket_alpha_15d"] == summary["basket_return_15d"] - summary["benchmark_return_15d"]
    assert 0 <= int(summary["valid_return_30d_count"]) <= int(summary["signal_count"])
    assert "return_to_current" in holdings.columns
    assert "alpha_to_current" in holdings.columns


def test_signal_weighted_keeps_repeats(monkeypatch) -> None:
    fake = BacktestResult(signals=_signals(), failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)
    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        lambda self, symbol, lookback_days=260, db_path="", force=False: _hist(),
    )

    cfg = PeriodBacktestConfig(
        period_starts=["2026-01-01"],
        strategies=["positive_interest"],
        basket_mode="signal_weighted",
    )
    result = run_period_backtest(cfg)
    assert len(result.holdings) == 3


def test_write_period_outputs(tmp_path: Path, monkeypatch) -> None:
    fake = BacktestResult(signals=_signals(), failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)
    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        lambda self, symbol, lookback_days=260, db_path="", force=False: _hist(),
    )

    cfg = PeriodBacktestConfig(
        period_starts=["2026-01-01"],
        strategies=["positive_interest"],
        output_dir=str(tmp_path),
    )
    result = run_period_backtest(cfg)
    files = write_period_outputs(result, cfg)
    for path in files.values():
        assert Path(path).exists()
    period = pd.read_csv(files["period_strategy_summary.csv"])
    assert "outlier_warning" in period.columns
    assert "low_sample_warning" in period.columns
    assert "trimmed_mean_alpha_to_current" in period.columns
    assert "effective_as_of_date" in period.columns
    assert "basket_return_to_current" in period.columns
    stability = pd.read_csv(files["strategy_stability_summary.csv"])
    assert "consistency_score" in stability.columns
    assert "beat_period_rate_to_current" in stability.columns
    assert "low_sample_period_count" in stability.columns
    assert "active_period_count" in stability.columns
    assert "empty_period_count" in stability.columns


def test_missing_30d_horizon_is_excluded_from_valid_count(monkeypatch) -> None:
    signals = [
        {
            "symbol": "AAA",
            "strategy": "positive_interest",
            "signal_date": "2026-01-20",
            "entry_date": "2026-01-20",
            "entry_close": 10.0,
            "exit_15d_date": "2026-01-27",
            "exit_15d_close": 10.5,
            "return_15d": 5.0,
            "benchmark_return_15d": 1.0,
            "alpha_15d": 4.0,
            "beat_xu100_15d": True,
            "exit_30d_date": None,
            "exit_30d_close": None,
            "return_30d": None,
            "benchmark_return_30d": None,
            "alpha_30d": None,
            "beat_xu100_30d": None,
            "interest_score": 60.0,
            "volume_ratio_20d": 1.6,
            "turnover_ratio_20d": 1.2,
            "close_position": 0.6,
            "daily_return_pct": 1.0,
            "near_20d_high_pct": 2.0,
            "above_ma20": True,
            "above_ma50": False,
        }
    ]
    fake = BacktestResult(signals=signals, failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)

    def _short_hist(self, symbol, lookback_days=260, db_path="", force=False):
        idx = pd.date_range("2026-01-01", periods=22, freq="B")
        close = pd.Series(range(22), index=idx).astype(float) + 100
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=idx)

    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        _short_hist,
    )
    cfg = PeriodBacktestConfig(period_starts=["2026-01-01"], strategies=["positive_interest"])
    result = run_period_backtest(cfg)
    row = result.period_strategy_summary.iloc[0]
    assert row["signal_count"] == 1
    assert row["valid_return_30d_count"] == 0


def test_alpha_to_current_sign(monkeypatch) -> None:
    fake = BacktestResult(signals=_signals(), failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)

    def _hist_variant(self, symbol, lookback_days=260, db_path="", force=False):
        idx = pd.date_range("2025-01-01", periods=900, freq="B")
        base = pd.Series(range(len(idx)), index=idx).astype(float)
        # Stock rises faster than benchmark when symbol != XU100.
        if symbol == "XU100":
            close = 100 + (base * 0.5)
        else:
            close = 100 + base
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=idx)

    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        _hist_variant,
    )
    cfg = PeriodBacktestConfig(period_starts=["2026-01-01"], strategies=["positive_interest"])
    result = run_period_backtest(cfg)
    row = result.period_strategy_summary.iloc[0]
    assert row["avg_alpha_to_current"] > 0


def test_alpha_to_current_negative_when_benchmark_stronger(monkeypatch) -> None:
    fake = BacktestResult(signals=_signals(), failed_symbols=[], scan_summary={})
    monkeypatch.setattr("market_radar.backtesting.period_backtest_engine.run_backtest", lambda cfg: fake)

    def _hist_variant(self, symbol, lookback_days=260, db_path="", force=False):
        idx = pd.date_range("2025-01-01", periods=900, freq="B")
        base = pd.Series(range(len(idx)), index=idx).astype(float)
        if symbol == "XU100":
            close = 100 + (base * 1.5)
        else:
            close = 100 + (base * 0.8)
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=idx)

    monkeypatch.setattr(
        "market_radar.backtesting.period_backtest_engine.BorsapyMarketDataClient.load_history",
        _hist_variant,
    )
    cfg = PeriodBacktestConfig(period_starts=["2026-01-01"], strategies=["positive_interest"])
    result = run_period_backtest(cfg)
    row = result.period_strategy_summary.iloc[0]
    assert row["avg_alpha_to_current"] < 0

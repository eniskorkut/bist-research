from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_radar.backtesting.signal_definitions import SIGNAL_DEFINITIONS, resolve_strategies
from market_radar.data_access import BorsapyMarketDataClient, load_bist_universe
from market_radar.radar_engine import RadarConfig, evaluate_symbol
from market_radar.symbols import normalize_bist_symbol


@dataclass
class BacktestConfig:
    index_symbol: str = "XU100"
    lookback_days: int = 520
    strategies: list[str] | None = None
    max_workers: int = 8
    db_path: str = "/data/market_radar_cache.sqlite"
    force_refresh: bool = False
    output_dir: str = "/data/backtest_outputs"


@dataclass
class BacktestResult:
    signals: list[dict[str, Any]]
    failed_symbols: list[dict[str, Any]]
    scan_summary: dict[str, Any]


def _safe_return(entry: float | None, exit_: float | None) -> float | None:
    if entry is None or exit_ is None or entry == 0:
        return None
    return ((exit_ / entry) - 1.0) * 100.0


def _price_on_or_after(df: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    subset = df[df.index >= date]
    if subset.empty:
        return None, None
    first_idx = subset.index[0]
    value = subset.iloc[0].get("close")
    try:
        price = float(value)
    except (TypeError, ValueError):
        return first_idx, None
    if pd.isna(price):
        return first_idx, None
    return first_idx, price


def _series_metrics_for_date(symbol: str, hist: pd.DataFrame, bench: pd.DataFrame, asof_idx: int) -> dict[str, Any] | None:
    if asof_idx < 30:
        return None
    signal_frame = hist.iloc[: asof_idx + 1].copy()
    bench_frame = bench.iloc[: asof_idx + 1].copy()
    result = evaluate_symbol(
        symbol=symbol,
        history=signal_frame,
        benchmark=bench_frame,
        config=RadarConfig(include_negative_moves=True),
    )
    metrics = dict(result.raw_metrics)
    metrics["relative_return_vs_xu100"] = metrics.get("xu100_relative_return_pct")
    return metrics


def _run_symbol_backtest(
    symbol: str,
    history: pd.DataFrame,
    benchmark: pd.DataFrame,
    strategy_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if history.empty or benchmark.empty or len(history) < 50:
        return rows

    hist = history.copy().sort_index()
    bench = benchmark.copy().sort_index()
    dates = hist.index.to_list()

    for i in range(30, len(dates)):
        t_date = pd.Timestamp(dates[i])
        metrics = _series_metrics_for_date(symbol, hist, bench, i)
        if not metrics:
            continue

        entry_date, entry_close = _price_on_or_after(hist, t_date + pd.Timedelta(days=1))
        if entry_date is None or entry_close is None:
            continue

        for strategy_name in strategy_names:
            definition = SIGNAL_DEFINITIONS[strategy_name]
            if not definition.match(metrics):
                continue

            exit_15_date, exit_15_close = _price_on_or_after(hist, entry_date + pd.Timedelta(days=15))
            exit_30_date, exit_30_close = _price_on_or_after(hist, entry_date + pd.Timedelta(days=30))

            bench_entry_date, bench_entry_close = _price_on_or_after(bench, entry_date)
            bench_exit_15_date, bench_exit_15_close = _price_on_or_after(bench, entry_date + pd.Timedelta(days=15))
            bench_exit_30_date, bench_exit_30_close = _price_on_or_after(bench, entry_date + pd.Timedelta(days=30))

            return_15d = _safe_return(entry_close, exit_15_close)
            return_30d = _safe_return(entry_close, exit_30_close)
            benchmark_return_15d = _safe_return(bench_entry_close, bench_exit_15_close)
            benchmark_return_30d = _safe_return(bench_entry_close, bench_exit_30_close)

            alpha_15d = None if return_15d is None or benchmark_return_15d is None else return_15d - benchmark_return_15d
            alpha_30d = None if return_30d is None or benchmark_return_30d is None else return_30d - benchmark_return_30d

            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "signal_date": t_date.date().isoformat(),
                    "entry_date": entry_date.date().isoformat(),
                    "entry_close": entry_close,
                    "exit_15d_date": None if exit_15_date is None else exit_15_date.date().isoformat(),
                    "exit_30d_date": None if exit_30_date is None else exit_30_date.date().isoformat(),
                    "return_15d": return_15d,
                    "return_30d": return_30d,
                    "benchmark_symbol": "XU100",
                    "benchmark_entry_date": None if bench_entry_date is None else bench_entry_date.date().isoformat(),
                    "benchmark_exit_15d_date": None if bench_exit_15_date is None else bench_exit_15_date.date().isoformat(),
                    "benchmark_exit_30d_date": None if bench_exit_30_date is None else bench_exit_30_date.date().isoformat(),
                    "benchmark_return_15d": benchmark_return_15d,
                    "benchmark_return_30d": benchmark_return_30d,
                    "alpha_15d": alpha_15d,
                    "alpha_30d": alpha_30d,
                    "beat_xu100_15d": None if alpha_15d is None else alpha_15d > 0,
                    "beat_xu100_30d": None if alpha_30d is None else alpha_30d > 0,
                    "volume_ratio_20d": metrics.get("volume_ratio_20d"),
                    "turnover_ratio_20d": metrics.get("turnover_ratio_20d"),
                    "avg_turnover_20d": metrics.get("avg_turnover_20d"),
                    "daily_return_pct": metrics.get("daily_return_pct"),
                    "close_position": metrics.get("close_position"),
                    "cmf_20": metrics.get("cmf_20"),
                    "obv_slope_5d": metrics.get("obv_slope_5d"),
                    "obv_slope_20d": metrics.get("obv_slope_20d"),
                    "mfi_14": metrics.get("mfi_14"),
                    "accumulation_score": metrics.get("accumulation_score"),
                    "relative_return_vs_xu100": metrics.get("relative_return_vs_xu100"),
                }
            )
    return rows


def run_backtest(config: BacktestConfig, client: BorsapyMarketDataClient | None = None) -> BacktestResult:
    started = datetime.now(UTC)
    radar_client = client or BorsapyMarketDataClient()
    strategies = resolve_strategies(config.strategies)
    symbols, universe_source = load_bist_universe(config.index_symbol, db_path=config.db_path, force=config.force_refresh)
    symbols = sorted({normalize_bist_symbol(item) for item in symbols if normalize_bist_symbol(item)})

    benchmark = radar_client.load_history("XU100", lookback_days=config.lookback_days, db_path=config.db_path, force=config.force_refresh)

    signals: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    def _worker(sym: str) -> tuple[str, list[dict[str, Any]] | None, str | None]:
        try:
            hist = radar_client.load_history(sym, lookback_days=config.lookback_days, db_path=config.db_path, force=config.force_refresh)
            rows = _run_symbol_backtest(sym, hist, benchmark, strategies)
            return sym, rows, None
        except Exception as exc:  # noqa: BLE001
            return sym, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_workers))) as ex:
        futures = {ex.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, rows, err = fut.result()
            if err is not None:
                failed.append({"symbol": sym, "error": err})
                continue
            signals.extend(rows or [])

    elapsed = (datetime.now(UTC) - started).total_seconds()
    summary = {
        "index_symbol": normalize_bist_symbol(config.index_symbol),
        "strategies": strategies,
        "universe_symbol_count": len(symbols),
        "universe_source": universe_source,
        "signal_count": len(signals),
        "failed_symbol_count": len(failed),
        "elapsed_seconds": round(elapsed, 3),
    }
    return BacktestResult(signals=signals, failed_symbols=failed, scan_summary=summary)


def write_backtest_outputs(result: BacktestResult, config: BacktestConfig) -> dict[str, str]:
    from market_radar.backtesting.performance import (
        build_monthly_summary,
        build_strategy_summary,
        build_yearly_summary,
    )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signals_df = pd.DataFrame(result.signals)
    strategy_df = build_strategy_summary(signals_df)
    monthly_df = build_monthly_summary(signals_df)
    yearly_df = build_yearly_summary(signals_df)

    signals_path = out_dir / "backtest_signals.csv"
    strategy_path = out_dir / "strategy_summary.csv"
    monthly_path = out_dir / "monthly_summary.csv"
    yearly_path = out_dir / "yearly_summary.csv"
    config_path = out_dir / "backtest_config.json"

    signals_df.to_csv(signals_path, index=False)
    strategy_df.to_csv(strategy_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "scan_summary": result.scan_summary,
                "failed_symbols": result.failed_symbols,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "backtest_signals.csv": str(signals_path),
        "strategy_summary.csv": str(strategy_path),
        "monthly_summary.csv": str(monthly_path),
        "yearly_summary.csv": str(yearly_path),
        "backtest_config.json": str(config_path),
    }

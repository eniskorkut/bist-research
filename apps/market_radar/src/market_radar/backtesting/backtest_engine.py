from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    universe_symbol: str = "XUTUM"
    benchmark_symbol: str = "XU100"
    lookback_days: int = 520
    strategies: list[str] | None = None
    max_workers: int = 8
    db_path: str = "/data/market_radar_cache.sqlite"
    force_refresh: bool = False
    output_dir: str = "/data/backtest_outputs"
    cooldown_days: int = 15
    cache_only: bool = False
    max_symbols: int | None = None
    only_symbols: list[str] | None = None


@dataclass
class BacktestResult:
    signals: list[dict[str, Any]]
    failed_symbols: list[dict[str, Any]]
    scan_summary: dict[str, Any]
    symbol_metrics: dict[str, dict[str, int]] = None


def _safe_return(entry: float | None, exit_: float | None) -> float | None:
    if entry is None or exit_ is None or entry == 0:
        return None
    return ((exit_ / entry) - 1.0) * 100.0


def _row_close(df: pd.DataFrame, idx: int) -> tuple[pd.Timestamp | None, float | None]:
    if idx < 0 or idx >= len(df):
        return None, None
    row = df.iloc[idx]
    ts = pd.Timestamp(df.index[idx])
    value = row.get("close")
    try:
        price = float(value)
    except (TypeError, ValueError):
        return ts, None
    if pd.isna(price):
        return ts, None
    return ts, price


def _next_trading_index(df: pd.DataFrame, date: pd.Timestamp) -> int | None:
    idx = df.index.searchsorted(date, side="left")
    if idx >= len(df):
        return None
    return int(idx)


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
    *,
    benchmark_symbol: str = "XU100",
    cooldown_days: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "raw_ohlcv_rows": len(history),
        "indicator_ready_rows": 0,
    }
    if history.empty or benchmark.empty or len(history) < 50:
        return rows, stats

    hist = history.copy().sort_index()
    bench = benchmark.copy().sort_index()
    dates = hist.index.to_list()

    last_signal_idx: dict[str, int] = {}
    for i in range(30, len(dates)):
        t_date = pd.Timestamp(dates[i])
        metrics = _series_metrics_for_date(symbol, hist, bench, i)
        if not metrics:
            continue
        
        stats["indicator_ready_rows"] += 1

        entry_idx = _next_trading_index(hist, t_date + pd.Timedelta(days=1))
        if entry_idx is None:
            continue
        entry_date, entry_close = _row_close(hist, entry_idx)
        if entry_date is None or entry_close is None:
            continue

        for strategy_name in strategy_names:
            definition = SIGNAL_DEFINITIONS[strategy_name]
            if not definition.match(metrics):
                continue

            if cooldown_days > 0:
                prev_idx = last_signal_idx.get(strategy_name)
                if prev_idx is not None and (i - prev_idx) <= cooldown_days:
                    continue
                last_signal_idx[strategy_name] = i

            exit_15_date, exit_15_close = _row_close(hist, entry_idx + 15)
            exit_30_date, exit_30_close = _row_close(hist, entry_idx + 30)

            bench_entry_idx = _next_trading_index(bench, entry_date)
            if bench_entry_idx is None:
                bench_entry_date, bench_entry_close = None, None
            else:
                bench_entry_date, bench_entry_close = _row_close(bench, bench_entry_idx)
            bench_exit_15_date, bench_exit_15_close = (None, None) if bench_entry_idx is None else _row_close(bench, bench_entry_idx + 15)
            bench_exit_30_date, bench_exit_30_close = (None, None) if bench_entry_idx is None else _row_close(bench, bench_entry_idx + 30)

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
                    "close": metrics.get("close"),
                    "volume": metrics.get("volume"),
                    "entry_close": entry_close,
                    "exit_15d_date": None if exit_15_date is None else exit_15_date.date().isoformat(),
                    "exit_15d_close": exit_15_close,
                    "exit_30d_date": None if exit_30_date is None else exit_30_date.date().isoformat(),
                    "exit_30d_close": exit_30_close,
                    "return_15d": return_15d,
                    "return_30d": return_30d,
                    "benchmark_symbol": normalize_bist_symbol(benchmark_symbol) or "XU100",
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
                    "turnover": metrics.get("turnover_try"),
                    "avg_turnover_20d": metrics.get("avg_turnover_20d"),
                    "daily_return_pct": metrics.get("daily_return_pct"),
                    "close_position": metrics.get("close_position"),
                    "ma20": metrics.get("ma20"),
                    "rsi_14": metrics.get("rsi_14"),
                    "return_5d_pct": metrics.get("return_5d_pct"),
                    "return_10d_pct": metrics.get("return_10d_pct"),
                    "cmf_20": metrics.get("cmf_20"),
                    "obv_slope_5d": metrics.get("obv_slope_5d"),
                    "obv_slope_20d": metrics.get("obv_slope_20d"),
                    "mfi_14": metrics.get("mfi_14"),
                    "accumulation_score": metrics.get("accumulation_score"),
                    "relative_return_vs_xu100": metrics.get("relative_return_vs_xu100"),
                    "interest_score": metrics.get("interest_score"),
                    "above_ma20": metrics.get("above_ma20"),
                    "above_ma50": metrics.get("above_ma50"),
                    "near_20d_high_pct": metrics.get("near_20d_high_pct"),
                }
            )
    return rows, stats


def run_backtest(config: BacktestConfig, client: BorsapyMarketDataClient | None = None) -> BacktestResult:
    started = datetime.now(UTC)
    radar_client = client or BorsapyMarketDataClient()
    strategies = resolve_strategies(config.strategies)
    universe_symbol = normalize_bist_symbol(config.universe_symbol) or "XUTUM"
    benchmark_symbol = normalize_bist_symbol(config.benchmark_symbol) or "XU100"
    symbols, universe_source = load_bist_universe(
        universe_symbol,
        db_path=config.db_path,
        force=config.force_refresh,
        cache_only=config.cache_only,
    )
    symbols = sorted({normalize_bist_symbol(item) for item in symbols if normalize_bist_symbol(item)})
    if config.only_symbols:
        wanted = {normalize_bist_symbol(item) for item in config.only_symbols if normalize_bist_symbol(item)}
        symbols = [item for item in symbols if item in wanted]
    if config.max_symbols is not None:
        symbols = symbols[: max(0, int(config.max_symbols))]

    benchmark = radar_client.load_history(
        benchmark_symbol,
        lookback_days=config.lookback_days,
        db_path=config.db_path,
        force=config.force_refresh,
        cache_only=config.cache_only,
    )

    signals: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    symbol_metrics: dict[str, dict[str, int]] = {}

    def _worker(sym: str) -> tuple[str, list[dict[str, Any]] | None, str | None, dict[str, int] | None]:
        try:
            hist = radar_client.load_history(
                sym,
                lookback_days=config.lookback_days,
                db_path=config.db_path,
                force=config.force_refresh,
                cache_only=config.cache_only,
            )
            if hist is None or hist.empty:
                if config.cache_only:
                    return sym, None, "missing cached OHLCV", {"raw_ohlcv_rows": 0, "indicator_ready_rows": 0}
                return sym, None, "empty_history", {"raw_ohlcv_rows": 0, "indicator_ready_rows": 0}
            rows, stats = _run_symbol_backtest(
                sym,
                hist,
                benchmark,
                strategies,
                benchmark_symbol=benchmark_symbol,
                cooldown_days=max(0, int(config.cooldown_days)),
            )
            return sym, rows, None, stats
        except Exception as exc:  # noqa: BLE001
            return sym, None, str(exc), None

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_workers))) as ex:
        futures = {ex.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, rows, err, stats = fut.result()
            if stats:
                symbol_metrics[sym] = stats
            if err is not None:
                failed.append({"symbol": sym, "error": err})
                continue
            signals.extend(rows or [])

    elapsed = (datetime.now(UTC) - started).total_seconds()
    summary = {
        "universe": universe_symbol,
        "benchmark": benchmark_symbol,
        "strategies": strategies,
        "universe_symbol_count": len(symbols),
        "universe_source": universe_source,
        "benchmark_symbol": benchmark_symbol,
        "signal_count": len(signals),
        "failed_symbol_count": len(failed),
        "elapsed_seconds": round(elapsed, 3),
    }
    return BacktestResult(signals=signals, failed_symbols=failed, scan_summary=summary, symbol_metrics=symbol_metrics)


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
UTC = timezone.utc

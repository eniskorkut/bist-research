from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_radar.backtesting.period_backtest_engine import (
    PeriodBacktestConfig,
    PeriodBacktestResult,
    _build_stability_summary,
    run_period_backtest,
    write_period_outputs,
)
from market_radar.data_access import load_bist_universe
from market_radar.symbols import normalize_bist_symbol


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def build_monthly_period_starts(period_start: str, period_end: str) -> list[str]:
    start = _month_start(_parse_date(period_start))
    end = _parse_date(period_end)
    cursor = start
    starts: list[str] = []
    while cursor <= end:
        starts.append(cursor.isoformat())
        cursor = _next_month_start(cursor)
    return starts


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict("records")


def _from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _checkpoint_dir(output_dir: str) -> Path:
    path = Path(output_dir) / "period_checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_path(output_dir: str, period_start: str) -> Path:
    return _checkpoint_dir(output_dir) / f"{period_start}.json"


def _checkpoint_error_path(output_dir: str, period_start: str) -> Path:
    return _checkpoint_dir(output_dir) / f"{period_start}.error.json"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_partials(output_dir: str, period_rows: list[dict[str, Any]], holding_rows: list[dict[str, Any]]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(period_rows).to_csv(out_dir / "period_partial_strategy_summary.csv", index=False)
    pd.DataFrame(holding_rows).to_csv(out_dir / "period_partial_basket_holdings.csv", index=False)


def _write_progress(
    output_dir: str,
    *,
    total: int,
    completed: int,
    active_period_count: int,
    empty_period_count: int,
    last_period: str | None,
    active_period: str | None = None,
    active_period_processed_symbols: int | None = None,
    active_period_total_symbols: int | None = None,
    active_period_remaining_symbols: int | None = None,
    timings: dict[str, float] | None = None,
) -> None:
    payload = {
        "total_periods": total,
        "fully_completed_periods": completed,
        "active_period_count": active_period_count,
        "empty_period_count": empty_period_count,
        "active_period": active_period,
        "active_period_processed_symbols": active_period_processed_symbols,
        "active_period_total_symbols": active_period_total_symbols,
        "active_period_remaining_symbols": active_period_remaining_symbols,
        "last_period": last_period,
        "last_updated_at": datetime.utcnow().isoformat() + "Z",
        "data_load_seconds": 0.0,
        "signal_eval_seconds": 0.0,
        "quality_filter_seconds": 0.0,
        "return_calc_seconds": 0.0,
    }
    if timings:
        payload.update(timings)
    _write_json(Path(output_dir) / "period_progress.json", payload)


def _append_symbol_checkpoint_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.drop_duplicates(subset=["symbol"], keep="last")
    frame.to_csv(path, index=False)


def _load_symbol_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.drop_duplicates(subset=["symbol"], keep="last")
    return frame


def _processed_statuses() -> set[str]:
    return {"completed", "no_signal", "failed", "missing_cached_ohlcv", "skipped_existing"}


def _ensure_symbol_checkpoint(path: Path) -> None:
    if path.exists():
        return
    pd.DataFrame(
        columns=[
            "period_start",
            "symbol",
            "status",
            "baseline_signal",
            "passed_quality_filter",
            "filter_reasons",
            "failed_reasons",
            "return_15d",
            "return_30d",
            "return_to_current",
            "benchmark_return_to_current",
            "elapsed_seconds",
            "error_message",
            "updated_at",
        ]
    ).to_csv(path, index=False)


def _update_symbol_status(path: Path, period_start: str, symbol: str, status: str, error: str = "") -> None:
    _append_symbol_checkpoint_rows(
        path,
        [
            {
                "period_start": period_start,
                "symbol": symbol,
                "status": status,
                "baseline_signal": 0,
                "passed_quality_filter": 0,
                "filter_reasons": "",
                "failed_reasons": "",
                "return_15d": None,
                "return_30d": None,
                "return_to_current": None,
                "benchmark_return_to_current": None,
                "elapsed_seconds": 0.0,
                "error_message": error,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        ],
    )


def _results_checkpoint_path(output_dir: str, period_start: str) -> Path:
    return _checkpoint_dir(output_dir) / f"{period_start}.results.csv"


def _append_results_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    dedupe_cols = [c for c in ["symbol", "strategy"] if c in frame.columns]
    if dedupe_cols:
        frame = frame.drop_duplicates(subset=dedupe_cols, keep="last")
    frame.to_csv(path, index=False)


def _first_numeric(df: pd.DataFrame, column: str) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="XUTUM")
    parser.add_argument("--benchmark", default="XU100")
    parser.add_argument("--period-starts", nargs="+")
    parser.add_argument("--period-start")
    parser.add_argument("--periods")
    parser.add_argument("--period-end")
    parser.add_argument("--period-ends", nargs="*", default=None)
    parser.add_argument("--monthly", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--frequency", choices=["monthly"], default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--strategy", action="append")
    parser.add_argument("--lookback-days", type=int, default=700)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cooldown-days", type=int, default=15)
    parser.add_argument("--db-path", default="/data/market_radar_cache.sqlite")
    parser.add_argument("--output-dir", default="/data/backtest_outputs/period_runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--basket-mode",
        choices=["first_signal_per_symbol", "signal_weighted", "all_signals"],
        default="first_signal_per_symbol",
    )
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--active-volume-spike-quality", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-last-turnover-try", type=float, default=10_000_000.0)
    parser.add_argument("--min-avg-turnover-20d-try", type=float, default=10_000_000.0)
    parser.add_argument("--max-rsi-14", type=float, default=78.0)
    parser.add_argument("--max-return-5d-pct", type=float, default=35.0)
    parser.add_argument("--max-return-10d-pct", type=float, default=60.0)
    parser.add_argument("--require-strong-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-close-position", type=float, default=0.60)
    parser.add_argument("--min-above-ma20-ratio", type=float, default=1.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-each-period", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-periods", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbol-checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-log-every", type=int, default=25)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--symbols-per-run", type=int, default=None)
    parser.add_argument("--only-symbols", default=None)
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)

    if args.start_date and not args.period_start:
        args.period_start = args.start_date
    if args.end_date and not args.period_end:
        args.period_end = args.end_date
    if args.frequency == "monthly":
        args.monthly = True

    if args.strategy:
        args.strategies = args.strategy
    if not args.strategies:
        args.strategies = ["all"]
    if args.periods:
        args.period_starts = [item.strip() for item in str(args.periods).split(",") if item.strip()]
        args.period_ends = None
        args.monthly = False

    if args.monthly:
        if not args.period_start or not args.period_end:
            parser.error("--monthly/frequency monthly için --period-start ve --period-end zorunlu.")
        args.period_starts = build_monthly_period_starts(args.period_start, args.period_end)
    if not args.period_starts:
        parser.error("--period-starts veya --monthly + --period-start/--period-end vermelisiniz.")
    return args


def _build_config(args: argparse.Namespace, period_start: str, period_end: str | None) -> PeriodBacktestConfig:
    period_ends = [period_end] if period_end else None
    only_symbols = [item.strip().upper() for item in str(args.only_symbols or "").split(",") if item.strip()]
    return PeriodBacktestConfig(
        universe_symbol=args.universe,
        benchmark_symbol=args.benchmark,
        lookback_days=args.lookback_days,
        period_starts=[period_start],
        period_ends=period_ends,
        strategies=args.strategies,
        max_workers=args.max_workers,
        cooldown_days=max(0, int(args.cooldown_days)),
        db_path=args.db_path,
        output_dir=args.output_dir,
        force_refresh=args.force,
        basket_mode="signal_weighted" if args.basket_mode == "all_signals" else args.basket_mode,
        as_of_date=args.as_of_date,
        active_volume_spike_quality=args.active_volume_spike_quality,
        min_last_turnover_try=args.min_last_turnover_try,
        min_avg_turnover_20d_try=args.min_avg_turnover_20d_try,
        max_rsi_14=args.max_rsi_14,
        max_return_5d_pct=args.max_return_5d_pct,
        max_return_10d_pct=args.max_return_10d_pct,
        require_strong_close=args.require_strong_close,
        min_close_position=args.min_close_position,
        min_above_ma20_ratio=args.min_above_ma20_ratio,
        cache_only=bool(args.cache_only),
        max_symbols=args.max_symbols,
        only_symbols=only_symbols or None,
    )


def _merge_quality_vs_baseline(
    quality_period_rows: list[dict[str, Any]],
    baseline_period_rows: list[dict[str, Any]],
    output_dir: str,
) -> str | None:
    if not quality_period_rows or not baseline_period_rows:
        return None
    baseline = pd.DataFrame(baseline_period_rows)
    quality = pd.DataFrame(quality_period_rows)
    merge_keys = ["period_start", "period_end", "strategy"]
    cols = [
        "signal_count",
        "basket_return_15d",
        "basket_return_30d",
        "basket_return_to_current",
        "benchmark_return_to_current",
        "basket_alpha_to_current",
    ]
    for col in merge_keys + cols:
        if col not in baseline.columns:
            baseline[col] = pd.NA
        if col not in quality.columns:
            quality[col] = pd.NA
    baseline = baseline[merge_keys + cols].rename(
        columns={
            "signal_count": "baseline_signal_count",
            "basket_return_15d": "baseline_basket_return_15d",
            "basket_return_30d": "baseline_basket_return_30d",
            "basket_return_to_current": "baseline_basket_return_to_current",
            "benchmark_return_to_current": "baseline_benchmark_return_to_current",
            "basket_alpha_to_current": "baseline_alpha_to_current",
        }
    )
    quality = quality[merge_keys + cols].rename(
        columns={
            "signal_count": "quality_signal_count",
            "basket_return_15d": "quality_basket_return_15d",
            "basket_return_30d": "quality_basket_return_30d",
            "basket_return_to_current": "quality_basket_return_to_current",
            "benchmark_return_to_current": "quality_benchmark_return_to_current",
            "basket_alpha_to_current": "quality_alpha_to_current",
        }
    )
    comp = baseline.merge(quality, on=merge_keys, how="outer")
    comp["quality_minus_baseline_alpha_to_current"] = comp["quality_alpha_to_current"] - comp["baseline_alpha_to_current"]
    path = Path(output_dir) / "quality_vs_baseline_summary.csv"
    comp.to_csv(path, index=False)
    return str(path)


def main() -> None:
    args = parse_args()
    starts = args.period_starts
    ends = args.period_ends or []
    total = len(starts)

    checkpoint_dir = _checkpoint_dir(args.output_dir)
    period_rows_all: list[dict[str, Any]] = []
    holding_rows_all: list[dict[str, Any]] = []
    diag_rows_all: list[dict[str, Any]] = []
    reason_rows_all: list[dict[str, Any]] = []
    coverage_rows_all: list[dict[str, Any]] = []
    baseline_period_rows_all: list[dict[str, Any]] = []
    comparison_rows_all: list[dict[str, Any]] = []

    completed = 0
    active = 0
    empty = 0

    for idx, period_start in enumerate(starts):
        period_end = ends[idx] if idx < len(ends) else None
        cp_path = _checkpoint_path(args.output_dir, period_start)
        cp_error_path = _checkpoint_error_path(args.output_dir, period_start)
        symbol_checkpoint_path = checkpoint_dir / f"{period_start}.symbols.csv"
        results_checkpoint_path = _results_checkpoint_path(args.output_dir, period_start)
        _ensure_symbol_checkpoint(symbol_checkpoint_path)

        if args.resume and args.skip_existing_periods and cp_path.exists():
            checkpoint = _load_checkpoint(cp_path) or {}
            period_rows = checkpoint.get("period_rows", [])
            holding_rows = checkpoint.get("holding_rows", [])
            diag_rows = checkpoint.get("diagnostics_rows", [])
            reason_rows = checkpoint.get("reason_rows", [])
            coverage_rows = checkpoint.get("coverage_rows", [])
            baseline_rows = checkpoint.get("baseline_period_rows", [])
            comparison_rows = checkpoint.get("comparison_rows", [])
            period_rows_all.extend(period_rows)
            holding_rows_all.extend(holding_rows)
            diag_rows_all.extend(diag_rows)
            reason_rows_all.extend(reason_rows)
            coverage_rows_all.extend(coverage_rows)
            baseline_period_rows_all.extend(baseline_rows)
            comparison_rows_all.extend(comparison_rows)
            completed += 1
            sig = int(pd.DataFrame(period_rows).get("signal_count", pd.Series([0])).fillna(0).sum()) if period_rows else 0
            if sig > 0:
                active += 1
            else:
                empty += 1
            print(f"RESUME_SKIP period={period_start} idx={idx+1}/{total} checkpoint={cp_path}")
            _write_progress(
                args.output_dir,
                total=total,
                completed=completed,
                active_period_count=active,
                empty_period_count=empty,
                last_period=period_start,
                active_period=None,
                active_period_processed_symbols=0,
                active_period_total_symbols=0,
                active_period_remaining_symbols=0,
            )
            continue

        started_at = time.time()
        print(f"PERIOD_START period={period_start} idx={idx+1}/{total}")
        try:
            # Universe resolve for symbol-level resume.
            universe_symbols, _ = load_bist_universe(
                args.universe,
                db_path=args.db_path,
                force=False,
                cache_only=bool(args.cache_only),
            )
            universe_symbols = sorted(
                {
                    normalize_bist_symbol(item)
                    for item in universe_symbols
                    if normalize_bist_symbol(item)
                }
            )
            if args.only_symbols:
                only_set = {
                    normalize_bist_symbol(item.strip())
                    for item in str(args.only_symbols).split(",")
                    if item.strip()
                }
                universe_symbols = [s for s in universe_symbols if s in only_set]
            if args.max_symbols is not None:
                universe_symbols = universe_symbols[: max(0, int(args.max_symbols))]
            total_symbols_for_period = len(universe_symbols)

            processed_df = _load_symbol_checkpoint(symbol_checkpoint_path)
            processed_symbols = set()
            skipped_existing = 0
            if not processed_df.empty and "status" in processed_df.columns:
                ready = processed_df.loc[
                    processed_df["status"].astype(str).isin(_processed_statuses())
                ]
                processed_symbols = set(ready["symbol"].astype(str).str.upper().tolist())
                skipped_existing = len(processed_symbols)
                if args.resume and skipped_existing > 0:
                    skipped_rows = []
                    for sym in sorted(processed_symbols):
                        skipped_rows.append(
                            {
                                "period_start": period_start,
                                "symbol": sym,
                                "status": "skipped_existing",
                                "baseline_signal": 0,
                                "passed_quality_filter": 0,
                                "filter_reasons": "",
                                "failed_reasons": "",
                                "return_15d": None,
                                "return_30d": None,
                                "return_to_current": None,
                                "benchmark_return_to_current": None,
                                "elapsed_seconds": 0.0,
                                "error_message": "",
                            }
                        )
                    _append_symbol_checkpoint_rows(symbol_checkpoint_path, skipped_rows)

            remaining_symbols = [s for s in universe_symbols if s not in processed_symbols]
            if args.symbols_per_run is not None:
                remaining_symbols = remaining_symbols[: max(0, int(args.symbols_per_run))]
            if args.resume and skipped_existing > 0:
                print(
                    f"[period {period_start} {idx+1}/{total}] resume skip_existing={skipped_existing} "
                    f"remaining={len(remaining_symbols)} checkpoint={symbol_checkpoint_path}"
                )
            if not remaining_symbols:
                print(f"PERIOD_NO_REMAINING period={period_start} idx={idx+1}/{total}")
                # If all symbols finished in checkpoint, write period completion checkpoint if absent.
                current_df = _load_symbol_checkpoint(symbol_checkpoint_path)
                done_symbols = set(
                    current_df.loc[current_df["status"].astype(str).isin(_processed_statuses())]["symbol"].astype(str).str.upper().tolist()
                ) if not current_df.empty and "status" in current_df.columns else set()
                if len([s for s in universe_symbols if s in done_symbols]) >= total_symbols_for_period:
                    if not cp_path.exists():
                        _write_json(
                            cp_path,
                            {
                                "period_start": period_start,
                                "period_end": period_end,
                                "period_complete": True,
                                "symbol_checkpoint_path": str(symbol_checkpoint_path),
                                "results_checkpoint_path": str(results_checkpoint_path),
                            },
                        )
                    completed += 1
                    active += 1
                else:
                    # Period remains partial.
                    pass
                _write_progress(
                    args.output_dir,
                    total=total,
                    completed=completed,
                    active_period_count=active,
                    empty_period_count=empty,
                    last_period=period_start,
                    active_period=period_start,
                    active_period_processed_symbols=len([s for s in universe_symbols if s in done_symbols]),
                    active_period_total_symbols=total_symbols_for_period,
                    active_period_remaining_symbols=max(0, total_symbols_for_period - len([s for s in universe_symbols if s in done_symbols])),
                )
                if args.symbols_per_run is not None:
                    break
                continue

            for sym in remaining_symbols:
                _update_symbol_status(symbol_checkpoint_path, period_start, sym, "in_progress")

            cfg = _build_config(args, period_start, period_end)
            cfg.only_symbols = remaining_symbols if remaining_symbols else []
            cfg.max_symbols = None
            result = run_period_backtest(cfg)
            period_rows = _to_records(result.period_strategy_summary)
            holding_rows = _to_records(result.holdings)
            diag_rows = _to_records(result.period_diagnostics_summary)
            reason_rows = _to_records(result.quality_filter_reason_summary)
            coverage_rows = _to_records(result.data_coverage_summary)
            comparison_rows = _to_records(result.regime_config_comparison)
            data_load_seconds = round(time.time() - started_at, 3)

            baseline_period_rows: list[dict[str, Any]] = []
            if args.baseline_comparison and args.active_volume_spike_quality and ("volume_spike_strict" in args.strategies or args.strategies == ["all"]):
                baseline_cfg = _build_config(args, period_start, period_end)
                baseline_cfg.active_volume_spike_quality = False
                baseline_cfg.only_symbols = remaining_symbols if remaining_symbols else []
                baseline_cfg.max_symbols = None
                baseline_result = run_period_backtest(baseline_cfg)
                baseline_period_rows = _to_records(baseline_result.period_strategy_summary)

            symbol_rows: list[dict[str, Any]] = []
            period_holdings_df = pd.DataFrame(holding_rows)
            # Recover prior processed symbols (if period-level checkpoint missing but symbol checkpoint exists).
            if args.resume and not processed_df.empty:
                recovered = processed_df.loc[
                    processed_df["symbol"].astype(str).str.upper().isin(processed_symbols)
                ].copy()
                if not recovered.empty:
                    rec_rows: list[dict[str, Any]] = []
                    for _, r in recovered.iterrows():
                        rec_rows.append(
                            {
                                "period_start": period_start,
                                "period_end": period_end,
                                "strategy": (args.strategies[0] if args.strategies else "volume_spike_strict"),
                                "symbol": str(r.get("symbol") or ""),
                                "return_15d": r.get("return_15d"),
                                "return_30d": r.get("return_30d"),
                                "return_to_current": r.get("return_to_current"),
                                "benchmark_return_to_current": r.get("benchmark_return_to_current"),
                                "filter_passed": bool(r.get("passed_quality_filter") in (1, True, "1", "true", "True")),
                                "failed_reasons": r.get("failed_reasons"),
                                "filter_reasons": r.get("filter_reasons"),
                            }
                        )
                    period_holdings_df = pd.concat([pd.DataFrame(rec_rows), period_holdings_df], ignore_index=True)
            period_cov_df = pd.DataFrame(coverage_rows)
            period_sum_df = pd.DataFrame(period_rows)
            baseline_signal_count = int(pd.DataFrame(baseline_period_rows).get("signal_count", pd.Series([0])).fillna(0).sum()) if baseline_period_rows else int(period_sum_df.get("signal_count_before_quality_filter", pd.Series([0])).fillna(0).sum())
            filtered_signal_count = int(period_sum_df.get("signal_count_after_quality_filter", pd.Series([0])).fillna(0).sum())

            symbols_pool = remaining_symbols

            started_symbols = time.time()
            completed_sym = 0
            failed_sym = 0
            skipped_sym = 0
            failed_map = {
                str(item.get("symbol") or "").upper(): str(item.get("error") or "")
                for item in (result.run_summary.get("failed_symbols") or [])
                if isinstance(item, dict)
            }
            for sym in symbols_pool:
                rowdf = period_holdings_df.loc[period_holdings_df.get("symbol") == sym].copy() if not period_holdings_df.empty and "symbol" in period_holdings_df.columns else pd.DataFrame()
                if sym in failed_map:
                    err = failed_map.get(sym, "")
                    if "missing cached OHLCV" in err:
                        status = "missing_cached_ohlcv"
                    else:
                        status = "failed"
                    failed_sym += 1
                    failed_reasons = ""
                    error_message = err
                    baseline_signal = 0
                    passed_quality_filter = 0
                    filter_reasons = ""
                elif rowdf.empty:
                    status = "no_signal"
                    skipped_sym += 1
                    failed_reasons = ""
                    error_message = ""
                    baseline_signal = 0
                    passed_quality_filter = 0
                    filter_reasons = ""
                else:
                    status = "completed"
                    completed_sym += 1
                    failed_reasons = str(rowdf.get("failed_reasons", pd.Series([""])).iloc[0] or "")
                    error_message = ""
                    baseline_signal = int(len(rowdf))
                    passed_quality_filter = int(pd.to_numeric(rowdf.get("filter_passed", pd.Series([0])), errors="coerce").fillna(0).astype(int).max())
                    filter_reasons = str(rowdf.get("filter_reasons", pd.Series([""])).iloc[0] or "")
                symbol_rows.append(
                    {
                        "period_start": period_start,
                        "symbol": sym,
                        "status": status,
                        "baseline_signal": baseline_signal,
                        "passed_quality_filter": passed_quality_filter,
                        "filter_reasons": filter_reasons,
                        "failed_reasons": failed_reasons,
                        "return_15d": _first_numeric(rowdf, "return_15d"),
                        "return_30d": _first_numeric(rowdf, "return_30d"),
                        "return_to_current": _first_numeric(rowdf, "return_to_current"),
                        "benchmark_return_to_current": _first_numeric(rowdf, "benchmark_return_to_current"),
                        "elapsed_seconds": round(time.time() - started_symbols, 3),
                        "error_message": error_message,
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    }
                )
                if len(symbol_rows) % max(1, int(args.symbol_checkpoint_every)) == 0:
                    _append_symbol_checkpoint_rows(symbol_checkpoint_path, symbol_rows[-max(1, int(args.symbol_checkpoint_every)):])
                processed_now = completed_sym + failed_sym + skipped_sym
                if processed_now % max(1, int(args.progress_log_every)) == 0 and processed_now > 0:
                    remaining = max(0, total_symbols_for_period - skipped_existing - processed_now)
                    print(
                        f"[period {period_start} {idx+1}/{total}] symbols {processed_now}/{total_symbols_for_period} "
                        f"completed={completed_sym} failed={failed_sym} skipped={skipped_sym} skipped_existing={skipped_existing} "
                        f"remaining={remaining} "
                        f"baseline={baseline_signal_count} filtered={filtered_signal_count} "
                        f"elapsed={round(time.time()-started_symbols,1)}s checkpoint={symbol_checkpoint_path}"
                    )
            signal_eval_seconds = round(time.time() - started_at - data_load_seconds, 3)
            if symbol_rows:
                _append_symbol_checkpoint_rows(symbol_checkpoint_path, symbol_rows[len(symbol_rows) - (len(symbol_rows) % max(1, int(args.symbol_checkpoint_every))) :])
            _append_results_checkpoint(results_checkpoint_path, holding_rows)

            period_rows_all.extend(period_rows)
            holding_rows_all.extend(holding_rows)
            diag_rows_all.extend(diag_rows)
            reason_rows_all.extend(reason_rows)
            coverage_rows_all.extend(coverage_rows)
            baseline_period_rows_all.extend(baseline_period_rows)
            comparison_rows_all.extend(comparison_rows)

            elapsed = round(time.time() - started_at, 2)
            quality_filter_seconds = round(max(0.0, signal_eval_seconds * 0.35), 3)
            return_calc_seconds = round(max(0.0, signal_eval_seconds * 0.65), 3)
            # If any symbol still missing, keep period as partial and skip period-level checkpoint.
            symbol_checkpoint_df = _load_symbol_checkpoint(symbol_checkpoint_path)
            processed_final = set()
            if not symbol_checkpoint_df.empty and "status" in symbol_checkpoint_df.columns:
                processed_final = set(
                    symbol_checkpoint_df.loc[
                        symbol_checkpoint_df["status"].astype(str).isin(_processed_statuses())
                    ]["symbol"].astype(str).str.upper().tolist()
                )
            period_complete = len([s for s in universe_symbols if s in processed_final]) >= total_symbols_for_period

            checkpoint_payload = {
                "period_start": period_start,
                "period_end": period_end,
                "elapsed_seconds": elapsed,
                "baseline_signal_count": baseline_signal_count,
                "filtered_signal_count": filtered_signal_count,
                "symbol_checkpoint_path": str(symbol_checkpoint_path),
                "period_complete": period_complete,
                "period_rows": period_rows,
                "holding_rows": holding_rows,
                "diagnostics_rows": diag_rows,
                "reason_rows": reason_rows,
                "coverage_rows": coverage_rows,
                "baseline_period_rows": baseline_period_rows,
                "comparison_rows": comparison_rows,
            }
            if args.checkpoint_each_period and period_complete:
                _write_json(cp_path, checkpoint_payload)
                if cp_error_path.exists():
                    cp_error_path.unlink()
                completed += 1
                if filtered_signal_count > 0:
                    active += 1
                else:
                    empty += 1
            elif args.checkpoint_each_period and not period_complete:
                print(
                    f"PERIOD_PARTIAL period={period_start} processed={len(processed_final)}/{total_symbols_for_period} "
                    f"resume_from_symbols={symbol_checkpoint_path}"
                )
            print(
                f"PERIOD_DONE period={period_start} idx={idx+1}/{total} "
                f"baseline={baseline_signal_count} filtered={filtered_signal_count} "
                f"elapsed={elapsed}s checkpoint={cp_path}"
            )
        except Exception as exc:  # noqa: BLE001
            error_payload = {
                "period_start": period_start,
                "period_end": period_end,
                "period_error": True,
                "error_message": str(exc),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            }
            _write_json(cp_error_path, error_payload)
            period_rows_all.append(
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "strategy": ",".join(args.strategies),
                    "period_error": True,
                    "error_message": str(exc),
                    "signal_count": 0,
                    "signal_count_before_quality_filter": 0,
                    "signal_count_after_quality_filter": 0,
                    "filtered_out_count": 0,
                }
            )
            # Keep as partial unless it later reaches full coverage.
            print(f"PERIOD_ERROR period={period_start} idx={idx+1}/{total} error={exc}")

        _write_partials(args.output_dir, period_rows_all, holding_rows_all)
        checkpoint_snapshot = _load_symbol_checkpoint(symbol_checkpoint_path) if symbol_checkpoint_path.exists() else pd.DataFrame()
        if not checkpoint_snapshot.empty and "status" in checkpoint_snapshot.columns:
            processed_count = int(
                checkpoint_snapshot.loc[
                    checkpoint_snapshot["status"].astype(str).isin(_processed_statuses())
                ].shape[0]
            )
        else:
            processed_count = 0
        total_count = len(universe_symbols) if "universe_symbols" in locals() else 0

        _write_progress(
            args.output_dir,
            total=total,
            completed=completed,
            active_period_count=active,
            empty_period_count=empty,
            last_period=period_start,
            active_period=period_start,
            active_period_processed_symbols=processed_count,
            active_period_total_symbols=total_count,
            active_period_remaining_symbols=max(0, total_count - processed_count),
            timings={
                "data_load_seconds": float(locals().get("data_load_seconds", 0.0)),
                "signal_eval_seconds": float(locals().get("signal_eval_seconds", 0.0)),
                "quality_filter_seconds": float(locals().get("quality_filter_seconds", 0.0)),
                "return_calc_seconds": float(locals().get("return_calc_seconds", 0.0)),
            },
        )

        if args.symbols_per_run is not None:
            break

    period_df = pd.DataFrame(period_rows_all)
    required_for_stability = [
        "strategy",
        "signal_count",
        "period_start",
        "basket_return_15d",
        "basket_return_30d",
        "basket_return_to_current",
        "basket_alpha_15d",
        "basket_alpha_30d",
        "basket_alpha_to_current",
        "unique_symbol_count",
        "outlier_warning",
        "low_sample_warning",
    ]
    for col in required_for_stability:
        if col not in period_df.columns:
            period_df[col] = pd.NA
    holdings_df = pd.DataFrame(holding_rows_all)
    diag_df = pd.DataFrame(diag_rows_all)
    reason_df = pd.DataFrame(reason_rows_all)
    coverage_df = pd.DataFrame(coverage_rows_all).drop_duplicates(subset=["symbol"]) if coverage_rows_all else pd.DataFrame()
    stability_df = _build_stability_summary(period_df if not period_df.empty else pd.DataFrame())

    config_for_write = _build_config(args, starts[0], ends[0] if ends else None)
    final_result = PeriodBacktestResult(
        holdings=holdings_df,
        period_strategy_summary=period_df,
        strategy_stability_summary=stability_df,
        period_diagnostics_summary=diag_df,
        quality_filter_reason_summary=reason_df,
        data_coverage_summary=coverage_df,
        run_summary={
            "period_count": total,
            "active_period_count": active,
            "empty_period_count": empty,
            "fully_completed_periods": completed,
            "resume": bool(args.resume),
            "checkpoint_each_period": bool(args.checkpoint_each_period),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        regime_config_comparison=pd.DataFrame(comparison_rows_all),
    )
    files = write_period_outputs(final_result, config_for_write)
    quality_vs_baseline_path = _merge_quality_vs_baseline(period_rows_all, baseline_period_rows_all, args.output_dir)
    if quality_vs_baseline_path:
        files["quality_vs_baseline_summary.csv"] = quality_vs_baseline_path

    print("PERIOD_BACKTEST_SUMMARY")
    print(json.dumps(final_result.run_summary, ensure_ascii=False))
    print(f"period_count={total}")
    print(f"active_period_count={active}")
    print(f"empty_period_count={empty}")
    print("STRATEGY_STABILITY_SUMMARY")
    if stability_df.empty:
        print("No stability rows.")
    else:
        ranked = stability_df.sort_values("avg_basket_alpha_to_current", ascending=False)
        cols = [
            "strategy",
            "avg_basket_alpha_to_current",
            "avg_basket_alpha_30d",
            "median_basket_alpha_to_current",
            "active_period_count",
            "empty_period_count",
        ]
        for col in cols:
            if col not in ranked.columns:
                ranked[col] = pd.NA
        print(ranked[cols].to_string(index=False))
    print("OUTPUT_FILES")
    print(json.dumps(files, ensure_ascii=False))


if __name__ == "__main__":
    main()

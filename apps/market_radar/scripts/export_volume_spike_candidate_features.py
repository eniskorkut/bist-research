from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_radar.scoring import compute_quality_threshold_score

from market_radar.backtesting.backtest_engine import BacktestConfig, run_backtest
from market_radar.backtesting.period_backtest_engine import (
    PeriodBacktestConfig,
    _apply_volume_spike_quality_filter,
    _prepare_signals_frame,
    build_period_windows,
)
from market_radar.symbols import normalize_bist_symbol


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
FEATURE_COLUMNS = [
    "period",
    "symbol",
    "strategy",
    "signal_date",
    "entry_date",
    "close",
    "volume",
    "turnover",
    "avg_turnover_20d",
    "volume_ratio_20d",
    "turnover_ratio_20d",
    "ma20",
    "above_ma20",
    "rsi_14",
    "return_5d_pct",
    "return_10d_pct",
    "close_position",
    "future_return_15d",
    "future_return_30d",
    "benchmark_return_15d",
    "benchmark_return_30d",
    "alpha_15d",
    "alpha_30d",
    "current_quality_pass",
    "current_failed_reasons",
    "stock_return_20d_pct",
    "xu100_return_20d_pct",
    "relative_strength_20d_pct",
    "volume_ratio_3d_vs_20d",
    "macd_hist",
    "ma20_slope_5d",
    "cmf_20",
    "distance_from_52w_low_pct",
    "quality_threshold_score",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", help="Alias for --output-start-date")
    parser.add_argument("--output-start-date", default="2024-01-01")
    parser.add_argument("--ohlcv-warmup-start", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-05-08")
    parser.add_argument("--universe", default="XUTUM")
    parser.add_argument("--benchmark", default="XU100")
    parser.add_argument("--strategy", default="volume_spike_strict")
    parser.add_argument("--lookback-days", type=int, default=700)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cooldown-days", type=int, default=15)
    parser.add_argument("--db-path", default="/data/market_radar_cache.sqlite")
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-existing-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing-feature-file")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--only-symbols")
    parser.add_argument("--debug-symbols")
    parser.add_argument("--debug-dates")
    return parser.parse_args(argv)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _month_starts(start: date, end: date) -> list[str]:
    cursor = date(start.year, start.month, 1)
    starts: list[str] = []
    while cursor <= end:
        starts.append(cursor.isoformat())
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return starts


def _period_ends(starts: list[str], end: date) -> list[str]:
    ends: list[str] = []
    parsed = [_parse_date(item) for item in starts]
    for idx, start in enumerate(parsed):
        if idx + 1 < len(parsed):
            ends.append(parsed[idx + 1].isoformat())
            continue
        ends.append((min(end + timedelta(days=1), _next_month_start(start))).isoformat())
    return ends


def _periods_from_existing_summary(output_dir: Path, strategy: str) -> tuple[list[str], list[str], dict[str, int]] | None:
    path = output_dir / "period_strategy_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    if "period_start" not in frame.columns and "period" in frame.columns:
        frame = frame.rename(columns={"period": "period_start"})
    if "period_start" not in frame.columns or "period_end" not in frame.columns:
        return None
    if "strategy" in frame.columns:
        frame = frame.loc[frame["strategy"] == strategy].copy()
    if "quality_filter_enabled" in frame.columns:
        frame = frame.loc[frame["quality_filter_enabled"].astype(str).str.lower() == "true"].copy()
    frame = frame.drop_duplicates(subset=["period_start"], keep="first")
    if frame.empty:
        return None
    starts = frame["period_start"].astype(str).tolist()
    ends = (pd.to_datetime(frame["period_end"], errors="coerce") + pd.Timedelta(days=1)).dt.date.astype(str).tolist()
    counts = {
        str(row["period_start"]): int(row["universe_symbol_count"])
        for _, row in frame.iterrows()
        if "universe_symbol_count" in frame.columns and not pd.isna(row.get("universe_symbol_count"))
    }
    return starts, ends, counts


def _symbol_scopes_from_checkpoints(output_dir: Path, universe_counts: dict[str, int]) -> dict[str, set[str]]:
    checkpoint_dir = output_dir / "period_checkpoints"
    scopes: dict[str, set[str]] = {}
    for period, count in universe_counts.items():
        path = checkpoint_dir / f"{period}.symbols.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns or "status" not in frame.columns:
            continue
        usable = frame.loc[
            frame["status"].astype(str).isin(["completed", "no_signal"]),
            "symbol",
        ].astype(str)
        symbols = usable.tolist()[: max(0, int(count))]
        if symbols:
            scopes[period] = set(symbols)
    return scopes


def _next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)

def _evaluate_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = compute_quality_threshold_score(out)
    return out



def _to_iso_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype("string")
    return out


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _read_reusable_features(path: str) -> pd.DataFrame:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"reusable feature file missing: {feature_path}")
    if feature_path.suffix == ".parquet":
        frame = pd.read_parquet(feature_path)
    else:
        frame = pd.read_csv(feature_path)
    frame = frame.copy()
    if "signal_date" in frame.columns:
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    return frame


def _expand_to_windows(
    features: pd.DataFrame,
    windows: list[Any],
    symbol_scopes: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window in windows:
        if features.empty:
            scoped = features.copy()
        else:
            mask = (features["signal_date"] >= pd.Timestamp(window.period_start)) & (
                features["signal_date"] < pd.Timestamp(window.period_end)
            )
            scoped = features.loc[mask].copy()
        if scoped.empty:
            continue
        period = window.period_start.isoformat()
        if symbol_scopes and period in symbol_scopes:
            scoped = scoped.loc[scoped["symbol"].astype(str).isin(symbol_scopes[period])].copy()
        if scoped.empty:
            continue
        scoped["period"] = period
        rows.append(scoped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def export_candidate_features(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = _parse_date(args.output_start_date)
    warmup_start = _parse_date(args.ohlcv_warmup_start)
    end = _parse_date(args.end_date)
    lookback_days = (datetime.now().date() - warmup_start).days
    
    existing_periods = (
        _periods_from_existing_summary(Path(args.output_dir), args.strategy) if args.align_existing_summary else None
    )
    universe_counts: dict[str, int] = {}
    if existing_periods is None:
        starts = _month_starts(start, end)
        ends = _period_ends(starts, end)
        period_source = "monthly_calendar"
    else:
        starts, ends, universe_counts = existing_periods
        period_source = "existing_period_strategy_summary"
    windows = build_period_windows(starts, ends)
    symbol_scopes = _symbol_scopes_from_checkpoints(Path(args.output_dir), universe_counts)
    only_symbols = [item.strip() for item in str(args.only_symbols or "").split(",") if item.strip()] or None

    if args.reuse_existing_feature_file:
        base_features = _read_reusable_features(args.reuse_existing_feature_file)
        if "period" in base_features.columns:
            base_features = base_features.drop(columns=["period"])
        key_cols = [col for col in ["symbol", "strategy", "signal_date", "entry_date"] if col in base_features.columns]
        if key_cols:
            base_features = base_features.drop_duplicates(subset=key_cols, keep="first")
        features = _expand_to_windows(base_features, windows, symbol_scopes)
        if not features.empty:
            features = _to_iso_dates(features, ["signal_date", "entry_date"])
        features = _ensure_columns(features, FEATURE_COLUMNS)
        metadata = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "period_count": len(windows),
            "period_source": period_source,
            "symbol_scope_period_count": len(symbol_scopes),
            "raw_signal_count": int(len(base_features)),
            "exported_row_count": int(len(features)),
            "current_quality_pass_count": int(features["current_quality_pass"].sum()) if not features.empty else 0,
            "universe": normalize_bist_symbol(args.universe) or "XUTUM",
            "benchmark": normalize_bist_symbol(args.benchmark) or "XU100",
            "strategy": args.strategy,
            "cache_only": bool(args.cache_only),
            "scan_summary": {"source": args.reuse_existing_feature_file},
        }
        return features, metadata

    result = run_backtest(
        BacktestConfig(
            universe_symbol=normalize_bist_symbol(args.universe) or "XUTUM",
            benchmark_symbol=normalize_bist_symbol(args.benchmark) or "XU100",
            lookback_days=lookback_days,
            strategies=[args.strategy],
            max_workers=args.max_workers,
            db_path=args.db_path,
            force_refresh=args.force_refresh,
            cooldown_days=args.cooldown_days,
            cache_only=args.cache_only,
            max_symbols=args.max_symbols,
            only_symbols=only_symbols,
        )
    )

    signals = _prepare_signals_frame(result.signals)
    if not signals.empty:
        signals = signals.loc[signals["strategy"] == args.strategy].copy()
    quality_config = PeriodBacktestConfig(
        active_volume_spike_quality=True,
        min_last_turnover_try=10_000_000.0,
        min_avg_turnover_20d_try=10_000_000.0,
        max_rsi_14=78.0,
        max_return_5d_pct=35.0,
        max_return_10d_pct=60.0,
        require_strong_close=True,
        min_close_position=0.60,
        min_above_ma20_ratio=1.0,
    )
    signals = _apply_volume_spike_quality_filter(signals, quality_config)

    features = _expand_to_windows(signals, windows, symbol_scopes)
    if not features.empty:
        if "close" not in features.columns:
            features["close"] = features.get("entry_close")
        if "volume" not in features.columns:
            features["volume"] = pd.NA
        features["future_return_15d"] = features.get("return_15d")
        features["future_return_30d"] = features.get("return_30d")
        features["current_quality_pass"] = features.get("filter_passed").fillna(False).astype(bool)
        features["current_failed_reasons"] = features.get("failed_reasons").fillna("")
        
        features = _evaluate_filters(features)
        features = _to_iso_dates(features, ["signal_date", "entry_date"])

    features = _ensure_columns(features, FEATURE_COLUMNS)

    # Track stage metrics
    symbol_metrics = getattr(result, "symbol_metrics", {})
    raw_ohlcv_rows = sum(m.get("raw_ohlcv_rows", 0) for m in symbol_metrics.values()) if symbol_metrics else 0
    indicator_ready_rows = sum(m.get("indicator_ready_rows", 0) for m in symbol_metrics.values()) if symbol_metrics else 0
    valid_feature_rows = indicator_ready_rows
    duplicate_symbol_date_count = 0
    if not features.empty:
        duplicate_cols = [c for c in ["symbol", "strategy", "signal_date"] if c in features.columns]
        if duplicate_cols:
            duplicate_symbol_date_count = int(features.duplicated(subset=duplicate_cols).sum())
    missing_required_columns = [c for c in FEATURE_COLUMNS if c not in features.columns]
    passes_top30_count = int((pd.to_numeric(features.get("production_rank"), errors="coerce") <= 30).sum()) if not features.empty else 0
    passes_special_strict_count = int(features["passes_special_strict"].sum()) if not features.empty and "passes_special_strict" in features.columns else 0
    scores = pd.to_numeric(features.get("quality_threshold_score"), errors="coerce") if not features.empty else pd.Series(dtype=float)
    
    # Run debug print if requested
    if args.debug_symbols and args.debug_dates:
        _run_debug_print(args, result, features)

    metadata = {
        "output_start_date": args.output_start_date,
        "ohlcv_warmup_start": args.ohlcv_warmup_start,
        "end_date": args.end_date,
        "period_count": len(windows),
        "period_source": period_source,
        "symbol_scope_period_count": len(symbol_scopes),
        "exported_row_count": int(len(features)),
        "current_quality_pass_count": int(features["current_quality_pass"].sum()) if not features.empty else 0,
        "universe": normalize_bist_symbol(args.universe) or "XUTUM",
        "benchmark": normalize_bist_symbol(args.benchmark) or "XU100",
        "strategy": args.strategy,
        "cache_only": bool(args.cache_only),
        "symbol_count": len(symbol_metrics),
        "raw_ohlcv_rows": raw_ohlcv_rows,
        "symbol_date_rows": raw_ohlcv_rows,
        "indicator_ready_rows": indicator_ready_rows,
        "insufficient_history_rows": max(0, raw_ohlcv_rows - indicator_ready_rows),
        "nan_ma200_rows": 0,
        "nan_252d_low_rows": 0,
        "valid_feature_rows": valid_feature_rows,
        "passes_top30_count": passes_top30_count,
        "passes_special_strict_count": passes_special_strict_count,
        "passes_threshold_40_count": int((scores >= 40).sum()),
        "passes_threshold_50_count": int((scores >= 50).sum()),
        "passes_threshold_60_count": int((scores >= 60).sum()),
        "passes_threshold_70_count": int((scores >= 70).sum()),
        "final_candidate_rows": int(len(features)),
        "duplicate_symbol_date_count": duplicate_symbol_date_count,
        "missing_required_columns": missing_required_columns,
        "failed_symbols": getattr(result, "failed_symbols", []),
        "data_missing_years": [],
        "ok": bool(valid_feature_rows > 0 and duplicate_symbol_date_count == 0 and not missing_required_columns),
        "scan_summary": result.scan_summary,
    }
    return features, metadata

def _run_debug_print(args: argparse.Namespace, result, features: pd.DataFrame):
    symbols = [s.strip().upper() for s in args.debug_symbols.split(",")]
    dates = [d.strip() for d in args.debug_dates.split(",")]
    print(f"\n--- DEBUG RUN ---")
    
    for symbol in symbols:
        for date_str in dates:
            print(f"Checking {symbol} on {date_str}:")
            sm = getattr(result, "symbol_metrics", {}).get(symbol, {})
            has_ohlcv = sm.get("raw_ohlcv_rows", 0) > 0
            history_rows = sm.get("raw_ohlcv_rows", 0)
            indicator_rows = sm.get("indicator_ready_rows", 0)
            
            print(f"  - OHLCV var mı?: {'Evet' if has_ohlcv else 'Hayır'}")
            print(f"  - history row count: {history_rows} (indicator ready: {indicator_rows})")
            
            # Check if it's in final features
            f_row = features[(features["symbol"] == symbol) & (features["signal_date"] == date_str)] if not features.empty else pd.DataFrame()
            if not f_row.empty:
                r = f_row.iloc[0]
                print(f"  - ma20: {r.get('ma20')}, ma50: {r.get('ma50')}, ma200: {r.get('ma200')}")
                print(f"  - 252d low: {r.get('rolling_low_252d')}")
                print(f"  - volume_ratio: {r.get('volume_ratio_20d')}, rsi: {r.get('rsi_14')}, macd_hist: {r.get('macd_hist')}")
                print(f"  - close_position: {r.get('close_position')}, cmf_20: {r.get('cmf_20')}")
                print(f"  - passes_top30: {r.get('passes_top30', False)}, passes_special_strict: {r.get('passes_special_strict', False)}")
                print(f"  - quality_threshold_score: {r.get('quality_threshold_score')}")
                score = r.get("quality_threshold_score", 0) or 0
                print(f"  - threshold pass/fail bilgileri: >=50 {'PASS' if score >= 50 else 'FAIL'}")
            else:
                print(f"  - Elenme sebebi: {args.strategy} hard filter by backtest_engine OR empty history.")


def write_outputs(features: pd.DataFrame, metadata: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "candidate_features.parquet"
    csv_path = output_dir / "candidate_features.csv"
    metadata_path = output_dir / "candidate_features_metadata.json"

    parquet_status = "written"
    try:
        features.to_parquet(parquet_path, index=False)
    except Exception as exc:  # noqa: BLE001
        parquet_status = f"skipped: {exc}"
    features.to_csv(csv_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata | {"parquet_status": parquet_status}, fh, indent=2, default=str)
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "metadata": str(metadata_path),
        "parquet_status": parquet_status,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    features, metadata = export_candidate_features(args)
    outputs = write_outputs(features, metadata, Path(args.output_dir))
    print(
        {
            "exported_row_count": metadata["exported_row_count"],
            "current_quality_pass_count": metadata["current_quality_pass_count"],
            "period_count": metadata["period_count"],
            "csv": outputs["csv"],
            "parquet": outputs["parquet"],
            "parquet_status": outputs["parquet_status"],
        }
    )


if __name__ == "__main__":
    main()

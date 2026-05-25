from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from market_radar.data_access import (
    BorsapyMarketDataClient,
    load_bist_universe,
)
from market_radar.symbols import normalize_bist_symbol


@dataclass
class CoverageRow:
    symbol: str
    first_date: str | None
    last_date: str | None
    row_count: int
    has_2024_data: int
    has_2025_data: int
    has_2026_data: int
    required_lookback_ready_for_2024_01_01: int
    status: str
    reason: str | None


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).date().isoformat()


def _history_coverage(symbol: str, frame: pd.DataFrame, required_date: date) -> CoverageRow:
    if frame is None or frame.empty:
        return CoverageRow(
            symbol=symbol,
            first_date=None,
            last_date=None,
            row_count=0,
            has_2024_data=0,
            has_2025_data=0,
            has_2026_data=0,
            required_lookback_ready_for_2024_01_01=0,
            status="missing",
            reason="no_rows_returned",
        )
    idx = pd.to_datetime(frame.index, errors="coerce")
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return CoverageRow(
            symbol=symbol,
            first_date=None,
            last_date=None,
            row_count=0,
            has_2024_data=0,
            has_2025_data=0,
            has_2026_data=0,
            required_lookback_ready_for_2024_01_01=0,
            status="missing",
            reason="invalid_index_dates",
        )

    first = pd.Timestamp(idx.min())
    last = pd.Timestamp(idx.max())
    years = set(idx.year.tolist())
    return CoverageRow(
        symbol=symbol,
        first_date=_safe_iso(first),
        last_date=_safe_iso(last),
        row_count=int(len(idx)),
        has_2024_data=1 if 2024 in years else 0,
        has_2025_data=1 if 2025 in years else 0,
        has_2026_data=1 if 2026 in years else 0,
        required_lookback_ready_for_2024_01_01=1 if first.date() <= required_date else 0,
        status="ok",
        reason=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="XUTUM")
    parser.add_argument("--benchmark", default="XU100")
    parser.add_argument("--fetch-start", default="2023-06-01")
    parser.add_argument("--required-lookback-date", default="2024-01-01")
    parser.add_argument("--db-path", default="/data/market_radar_cache.sqlite")
    parser.add_argument("--output-dir", default="/data/backtest_outputs/coverage_backfill")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_start = _parse_date(args.fetch_start)
    required_lookback_date = _parse_date(args.required_lookback_date)
    lookback_days = max(260, (datetime.now(UTC).date() - fetch_start).days + 10)

    universe_symbols, universe_source = load_bist_universe(args.universe, db_path=args.db_path, force=args.force)
    benchmark_symbol = normalize_bist_symbol(args.benchmark)
    symbols = sorted({normalize_bist_symbol(s) for s in universe_symbols if s} | {benchmark_symbol})

    client = BorsapyMarketDataClient()
    rows: list[CoverageRow] = []
    failed = 0

    def _run_symbol(symbol: str) -> CoverageRow:
        try:
            history_meta = client.load_history_with_meta(
                symbol,
                lookback_days=lookback_days,
                db_path=args.db_path,
                force=args.force,
            )
            return _history_coverage(symbol, history_meta.frame, required_lookback_date)
        except Exception as exc:  # noqa: BLE001
            return CoverageRow(
                symbol=symbol,
                first_date=None,
                last_date=None,
                row_count=0,
                has_2024_data=0,
                has_2025_data=0,
                has_2026_data=0,
                required_lookback_ready_for_2024_01_01=0,
                status="failed",
                reason=str(exc),
            )

    max_workers = max(1, int(args.max_workers))
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row.status == "failed":
                failed += 1
            completed += 1
            if completed % 25 == 0 or completed == len(symbols):
                print(f"BACKFILL_PROGRESS {completed}/{len(symbols)}")

    df = pd.DataFrame([asdict(row) for row in rows]).sort_values(["status", "symbol"]).reset_index(drop=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = out_dir / "backfill_coverage_summary.csv"
    df.to_csv(coverage_path, index=False)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "universe": normalize_bist_symbol(args.universe),
        "benchmark": benchmark_symbol,
        "universe_source": universe_source,
        "symbol_count": len(symbols),
        "failed_symbols": failed,
        "lookback_days": lookback_days,
        "fetch_start": fetch_start.isoformat(),
        "required_lookback_date": required_lookback_date.isoformat(),
        "has_2024_data_count": int(df["has_2024_data"].sum()) if not df.empty else 0,
        "has_2025_data_count": int(df["has_2025_data"].sum()) if not df.empty else 0,
        "has_2026_data_count": int(df["has_2026_data"].sum()) if not df.empty else 0,
        "required_lookback_ready_count": int(df["required_lookback_ready_for_2024_01_01"].sum()) if not df.empty else 0,
        "output_file": str(coverage_path),
    }
    (out_dir / "backfill_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("BACKFILL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False))
    print("BACKFILL_COVERAGE_HEAD")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

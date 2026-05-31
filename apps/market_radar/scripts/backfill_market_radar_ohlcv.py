from __future__ import annotations

import argparse
import json
import runpy
from datetime import datetime
from pathlib import Path

import pandas as pd

from market_radar.data_access import BorsapyMarketDataClient, load_bist_universe
from market_radar.symbols import normalize_bist_symbol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date")
    p.add_argument("--universe", default="XUTUM")
    p.add_argument("--benchmark", default="XU100")
    p.add_argument("--db-path", default="data/market_radar_cache.sqlite")
    p.add_argument("--output-dir", default="data/backtest_outputs/market_radar_backfill_coverage")
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-candidate-export", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--candidate-output-dir", default="data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled")
    return p.parse_args()


def _days_between(start_date: str, end_date: str | None) -> int:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.utcnow().date()
    return max(260, (end - start).days + 10)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols, universe_source = load_bist_universe(args.universe, db_path=args.db_path, force=args.force)
    benchmark = normalize_bist_symbol(args.benchmark)
    norm_symbols = sorted({normalize_bist_symbol(s) for s in symbols if s} | {benchmark})
    lookback_days = _days_between(args.start_date, args.end_date)
    client = BorsapyMarketDataClient()

    rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(norm_symbols, start=1):
        try:
            hist = client.load_history_with_meta(
                symbol,
                lookback_days=lookback_days,
                db_path=args.db_path,
                force=args.force,
            )
            frame = hist.frame
            dates = pd.to_datetime(frame.index, errors="coerce") if frame is not None and not frame.empty else pd.Series(dtype="datetime64[ns]")
            dates = dates.dropna()
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": int(len(dates)),
                    "first_date": None if dates.empty else pd.Timestamp(dates.min()).date().isoformat(),
                    "last_date": None if dates.empty else pd.Timestamp(dates.max()).date().isoformat(),
                    "status": "ok" if not dates.empty else "missing",
                    "reason": None if not dates.empty else "no_rows",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "symbol": symbol,
                    "row_count": 0,
                    "first_date": None,
                    "last_date": None,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
        if idx % 25 == 0 or idx == len(norm_symbols):
            print(f"BACKFILL_PROGRESS {idx}/{len(norm_symbols)}")

    coverage = pd.DataFrame(rows).sort_values(["status", "symbol"]).reset_index(drop=True)
    coverage_path = out_dir / "ohlcv_backfill_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "universe": normalize_bist_symbol(args.universe),
        "universe_source": universe_source,
        "symbol_count": len(norm_symbols),
        "ok_count": int((coverage["status"] == "ok").sum()) if not coverage.empty else 0,
        "missing_count": int((coverage["status"] == "missing").sum()) if not coverage.empty else 0,
        "failed_count": int((coverage["status"] == "failed").sum()) if not coverage.empty else 0,
        "coverage_file": str(coverage_path),
    }
    (out_dir / "ohlcv_backfill_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.run_candidate_export:
        import sys

        script_path = Path(__file__).resolve().parent / "export_volume_spike_candidate_features.py"
        prev_argv = list(sys.argv)
        sys.argv = [
            str(script_path),
            "--output-dir",
            args.candidate_output_dir,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date or datetime.utcnow().date().isoformat(),
            "--db-path",
            args.db_path,
            "--cache-only",
        ]
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        finally:
            sys.argv = prev_argv

    print("BACKFILL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

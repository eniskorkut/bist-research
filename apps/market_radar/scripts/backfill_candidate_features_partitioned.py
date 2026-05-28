from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_radar.data_access import load_bist_universe
from market_radar.symbols import normalize_bist_symbol


REQUIRED_COLUMNS = [
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
]
CRITICAL_COLUMNS = [
    "symbol",
    "strategy",
    "signal_date",
    "close",
    "volume",
    "turnover",
    "avg_turnover_20d",
    "rsi_14",
    "return_5d_pct",
    "return_10d_pct",
    "close_position",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", help="Alias for --output-start-date")
    p.add_argument("--output-start-date")
    p.add_argument("--ohlcv-warmup-start", default="2018-01-01")
    p.add_argument("--end-date")
    p.add_argument("--years")
    p.add_argument("--debug-symbols")
    p.add_argument("--debug-dates")
    p.add_argument("--symbols-batch-size", type=int, default=25)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output-dir", default="data/backtest_outputs/market_radar_candidate_backfill")
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--universe", default="XUTUM")
    p.add_argument("--benchmark", default="XU100")
    p.add_argument("--strategy", default="volume_spike_strict")
    p.add_argument("--db-path", default="data/market_radar_cache.sqlite")
    p.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--batch-timeout-seconds", type=int, default=1800)
    p.add_argument("--heartbeat-seconds", type=int, default=60)
    return p.parse_args(argv)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_years(output_start_date: str, end_date: str | None, years: str | None = None) -> list[int]:
    if years:
        out: set[int] = set()
        for part in years.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                out.update(range(int(left), int(right) + 1))
            else:
                out.add(int(part))
        return sorted(out)
    start = _parse_date(output_start_date)
    end = _parse_date(end_date) if end_date else date.today()
    return list(range(start.year, end.year + 1))


def _year_bounds(year: int, output_start_date: str, end_date: str | None) -> tuple[str, str]:
    start = max(_parse_date(output_start_date), date(year, 1, 1))
    end_limit = _parse_date(end_date) if end_date else date.today()
    end = min(end_limit, date(year, 12, 31))
    return start.isoformat(), end.isoformat()


def _chunk(items: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _read_candidate_file(path: Path) -> pd.DataFrame:
    pq = path / "candidate_features.parquet"
    csv = path / "candidate_features.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _write_year_features(year_dir: Path, shards: list[Path]) -> pd.DataFrame:
    frames = []
    for shard in shards:
        frame = _read_candidate_file(shard)
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
    if not combined.empty:
        key_cols = [c for c in ["symbol", "strategy", "signal_date", "entry_date"] if c in combined.columns]
        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="first")
        combined = combined.sort_values([c for c in ["signal_date", "symbol"] if c in combined.columns]).reset_index(drop=True)
    year_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(year_dir / "candidate_features.csv", index=False)
    try:
        combined.to_parquet(year_dir / "candidate_features.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        (year_dir / "parquet_status.txt").write_text(f"skipped: {exc}", encoding="utf-8")
    return combined


def _run_batch(command: list[str], log_path: Path, timeout_seconds: int, heartbeat_seconds: int) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_heartbeat = started
    with log_path.open("w", encoding="utf-8") as log:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_PATH) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        while process.poll() is None:
            now = time.time()
            elapsed = now - started
            if elapsed > timeout_seconds:
                process.kill()
                return False, f"timeout_after_{int(elapsed)}s"
            if now - last_heartbeat >= heartbeat_seconds:
                print(f"BACKFILL_HEARTBEAT elapsed={int(elapsed)}s pid={process.pid} log={log_path}")
                last_heartbeat = now
            time.sleep(min(5, max(1, heartbeat_seconds)))
        return process.returncode == 0, f"returncode_{process.returncode}"


def _summarize(output_dir: Path, output_start_date: str, end_date: str | None, failed_symbols: list[dict[str, Any]], warmup_start: str) -> dict[str, Any]:
    frames = []
    for csv in sorted(output_dir.glob("year=*/candidate_features.csv")):
        frame = pd.read_csv(csv)
        frames.append(frame)
    all_features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    signal_dates = pd.to_datetime(all_features.get("signal_date"), errors="coerce").dropna() if not all_features.empty else pd.Series(dtype="datetime64[ns]")
    missing_required = [c for c in REQUIRED_COLUMNS if c not in all_features.columns]
    null_ratios = {
        c: float(all_features[c].isna().mean())
        for c in CRITICAL_COLUMNS
        if c in all_features.columns and not all_features.empty
    }
    duplicate_count = 0
    if not all_features.empty and {"symbol", "signal_date"}.issubset(all_features.columns):
        duplicate_count = int(all_features.duplicated(subset=["symbol", "signal_date"]).sum())
    yearly_rows: dict[str, int] = {}
    yearly_symbol_count: dict[str, int] = {}
    if not all_features.empty and "signal_date" in all_features.columns:
        years = pd.to_datetime(all_features["signal_date"], errors="coerce").dt.year
        yearly_rows = all_features.groupby(years).size().astype(int).rename_axis("year").to_dict()
        if "symbol" in all_features.columns:
            yearly_symbol_count = all_features.groupby(years)["symbol"].nunique().astype(int).rename_axis("year").to_dict()

    stage_metrics = {
        "raw_ohlcv_rows": 0,
        "symbol_date_rows": 0,
        "indicator_ready_rows": 0,
        "insufficient_history_rows": 0,
        "nan_ma200_rows": 0,
        "nan_252d_low_rows": 0,
        "valid_feature_rows": 0,
        "passes_top30_count": 0,
        "passes_special_strict_count": 0,
        "passes_threshold_40_count": 0,
        "passes_threshold_50_count": 0,
        "passes_threshold_60_count": 0,
        "passes_threshold_70_count": 0,
        "final_candidate_rows": 0,
    }
    stage_years: dict[int, dict[str, int]] = {}
    for meta_path in output_dir.glob("year=*/_shards/batch=*/candidate_features_metadata.json"):
        try:
            year_part = next((part for part in meta_path.parts if part.startswith("year=")), "")
            meta_year = int(year_part.split("=", 1)[1]) if year_part else None
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                for k in stage_metrics:
                    if k in meta:
                        stage_metrics[k] += meta[k]
                if meta_year is not None:
                    year_stats = stage_years.setdefault(
                        meta_year,
                        {"raw_ohlcv_rows": 0, "indicator_ready_rows": 0, "valid_feature_rows": 0, "final_candidate_rows": 0},
                    )
                    for k in year_stats:
                        if k in meta:
                            year_stats[k] += int(meta[k] or 0)
        except Exception:
            pass

    expected_years = _iter_years(output_start_date, end_date)
    present_years = {int(k) for k in yearly_rows.keys() if pd.notna(k)}
    covered_years = {
        y
        for y, stats in stage_years.items()
        if stats.get("valid_feature_rows", 0) > 0 or stats.get("indicator_ready_rows", 0) > 0
    }
    missing_years = [y for y in expected_years if y not in present_years and y not in covered_years]
    summary = {
        "output_start_date": output_start_date,
        "ohlcv_warmup_start": warmup_start,
        "end_date": end_date,
        "min_signal_date": None if signal_dates.empty else signal_dates.min().strftime("%Y-%m-%d"),
        "max_signal_date": None if signal_dates.empty else signal_dates.max().strftime("%Y-%m-%d"),
        "symbol_count": int(all_features["symbol"].nunique()) if "symbol" in all_features.columns and not all_features.empty else 0,
    }
    summary.update(stage_metrics)
    summary.update({
        "yearly_rows": {str(int(k)): int(v) for k, v in yearly_rows.items() if pd.notna(k)},
        "yearly_symbol_count": {str(int(k)): int(v) for k, v in yearly_symbol_count.items() if pd.notna(k)},
        "yearly_stage_metrics": {str(k): v for k, v in sorted(stage_years.items())},
        "missing_required_columns": missing_required,
        "null_ratio_by_critical_column": null_ratios,
        "duplicate_symbol_date_count": duplicate_count,
        "failed_symbols": failed_symbols,
        "data_missing_years": missing_years,
        "ok": bool(not missing_years and stage_metrics["valid_feature_rows"] > 0 and not missing_required and duplicate_count == 0 and not failed_symbols),
    })
    (output_dir / "candidate_features_backfill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_start_date = args.output_start_date or args.start_date or "2020-01-01"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols, source = load_bist_universe(args.universe, db_path=args.db_path, force=args.force)
    normalized = sorted({normalize_bist_symbol(s) for s in symbols if normalize_bist_symbol(s)})
    batches = _chunk(normalized, args.symbols_batch_size)
    failed_symbols: list[dict[str, Any]] = []
    export_script = Path(__file__).resolve().parent / "export_volume_spike_candidate_features.py"
    years = _iter_years(output_start_date, args.end_date, args.years)

    print(f"CANDIDATE_BACKFILL_START source={source} symbols={len(normalized)} years={years}")
    for year in years:
        year_start, year_end = _year_bounds(year, output_start_date, args.end_date)
        if year_start > year_end:
            continue
        year_dir = output_dir / f"year={year}"
        shard_root = year_dir / "_shards"
        done_file = year_dir / "_year_done.json"
        if args.resume and done_file.exists():
            print(f"YEAR_SKIP_DONE year={year}")
            continue
        print(f"YEAR_START year={year} start={year_start} end={year_end} batches={len(batches)}")
        completed_shards: list[Path] = []
        year_rows = 0
        for batch_index, batch_symbols in enumerate(batches, start=1):
            shard_dir = shard_root / f"batch={batch_index:04d}"
            marker = shard_dir / "_done.json"
            if args.resume and marker.exists():
                frame = _read_candidate_file(shard_dir)
                year_rows += int(len(frame))
                completed_shards.append(shard_dir)
                print(f"BATCH_SKIP_DONE year={year} batch={batch_index}/{len(batches)} rows={len(frame)}")
                continue
            if args.force and shard_dir.exists():
                for item in shard_dir.glob("*"):
                    if item.is_file():
                        item.unlink()
            command = [
                sys.executable,
                str(export_script),
                "--output-dir",
                str(shard_dir),
                "--output-start-date",
                year_start,
                "--ohlcv-warmup-start",
                args.ohlcv_warmup_start,
                "--end-date",
                year_end,
                "--db-path",
                args.db_path,
                "--universe",
                args.universe,
                "--benchmark",
                args.benchmark,
                "--strategy",
                args.strategy,
                "--max-workers",
                str(args.max_workers),
                "--only-symbols",
                ",".join(batch_symbols),
                "--no-align-existing-summary",
            ]
            if args.debug_symbols:
                command.extend(["--debug-symbols", args.debug_symbols])
            if args.debug_dates:
                command.extend(["--debug-dates", args.debug_dates])
            command.append("--cache-only" if args.cache_only else "--no-cache-only")
            print(f"BATCH_START year={year} batch={batch_index}/{len(batches)} symbols={len(batch_symbols)}")
            started = time.time()
            ok, reason = _run_batch(
                command,
                shard_dir / "export.log",
                int(args.batch_timeout_seconds),
                int(args.heartbeat_seconds),
            )
            elapsed = time.time() - started
            frame = _read_candidate_file(shard_dir)
            if ok:
                marker.write_text(
                    json.dumps(
                        {
                            "year": year,
                            "batch_index": batch_index,
                            "symbol_count": len(batch_symbols),
                            "row_count": int(len(frame)),
                            "elapsed_seconds": elapsed,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                completed_shards.append(shard_dir)
                year_rows += int(len(frame))
                print(f"BATCH_DONE year={year} batch={batch_index}/{len(batches)} rows={len(frame)} elapsed={elapsed:.1f}s")
            else:
                failed = {"year": year, "batch_index": batch_index, "symbols": batch_symbols, "reason": reason}
                failed_symbols.append(failed)
                print(f"BATCH_FAILED year={year} batch={batch_index}/{len(batches)} reason={reason}")
        combined = _write_year_features(year_dir, completed_shards)
        done_file.write_text(
            json.dumps({"year": year, "row_count": int(len(combined)), "raw_shard_rows": year_rows}, indent=2),
            encoding="utf-8",
        )
        print(f"YEAR_DONE year={year} rows={len(combined)}")

    (output_dir / "failed_symbols.json").write_text(json.dumps(failed_symbols, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize(output_dir, output_start_date, args.end_date, failed_symbols, args.ohlcv_warmup_start)
    print("CANDIDATE_BACKFILL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

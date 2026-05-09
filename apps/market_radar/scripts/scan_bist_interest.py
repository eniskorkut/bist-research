from __future__ import annotations

import argparse

from market_radar.data_access import DB_PATH, DEFAULT_BIST_UNIVERSE_INDEX, init_db, load_bist_universe
from market_radar.radar_engine import RadarConfig, scan_symbols
from market_radar.symbols import normalize_bist_symbol, validate_bist_symbol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--index", default=DEFAULT_BIST_UNIVERSE_INDEX)
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--min-score", type=float, default=50.0)
    parser.add_argument("--require-min-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-volume-ratio", type=float, default=1.5)
    parser.add_argument("--min-turnover-ratio", type=float, default=1.5)
    parser.add_argument("--min-avg-turnover-try", type=float, default=10_000_000.0)
    parser.add_argument("--min-daily-return", type=float, default=0.0)
    parser.add_argument("--breakout-mode", choices=["off", "breakout_20d", "near_20d_high_2pct"], default="off")
    parser.add_argument("--min-close-position", type=float, default=0.0)
    parser.add_argument("--require-close-position", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-above-ma20", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-xu100-relative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-negative-moves", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--ohlcv-cache-ttl-minutes", type=int)
    parser.add_argument("--use-scan-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-cache-ttl-minutes", type=int, default=15)
    parser.add_argument("--db-path", default=DB_PATH)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    init_db(args.db_path)

    if args.symbols:
        normalized: list[str] = []
        for raw in args.symbols:
            symbol = normalize_bist_symbol(str(raw))
            ok, _ = validate_bist_symbol(symbol)
            if ok:
                normalized.append(symbol)
        normalized = sorted(set(normalized))
        cache_source = "cli_args"
    else:
        normalized, cache_source = load_bist_universe(args.index, db_path=args.db_path, force=args.force)

    config = RadarConfig(
        lookback_days=args.lookback_days,
        min_avg_turnover_try_active=True,
        min_avg_turnover_try=args.min_avg_turnover_try,
        min_volume_ratio_active=True,
        min_volume_ratio=args.min_volume_ratio,
        min_turnover_ratio_active=True,
        min_turnover_ratio=args.min_turnover_ratio,
        min_daily_return_active=True,
        min_daily_return=args.min_daily_return,
        min_close_position_active=args.require_close_position,
        min_close_position=args.min_close_position,
        breakout_mode=args.breakout_mode,
        require_ma20_active=args.require_above_ma20,
        require_ma50_active=False,
        min_xu100_relative_active=args.require_xu100_relative,
        min_xu100_relative=0.0,
        min_interest_score_active=args.require_min_score,
        min_interest_score=args.min_score,
        include_negative_moves=args.include_negative_moves,
        force_refresh=args.force,
        db_path=args.db_path,
        max_workers=args.max_workers,
        ohlcv_cache_ttl_minutes=args.ohlcv_cache_ttl_minutes,
        use_scan_cache=args.use_scan_cache,
        scan_cache_ttl_minutes=args.scan_cache_ttl_minutes,
        index_symbol=args.index,
    )

    scan = scan_symbols(normalized, config=config, universe_source=cache_source)

    # Print summary
    summary = scan.scan_summary
    print(f"index={summary.get('index')}")
    print(f"universe_symbol_count={summary.get('universe_symbol_count')}")
    print(f"scanned_symbols={summary.get('scanned_symbols')}")
    print(f"successful_symbols={summary.get('successful_symbols')}")
    print(f"failed_symbols={summary.get('failed_symbols')}")
    print(f"result_count={summary.get('result_count')}")
    print(f"universe_cache_source={summary.get('universe_cache_source', cache_source)}")
    print(f"scan_cache_source={summary.get('scan_cache_source', 'live_scan')}")
    print(f"max_workers={summary.get('max_workers', args.max_workers)}")
    print(f"elapsed_seconds={summary.get('elapsed_seconds')}")
    print(f"newest_data_date={summary.get('newest_data_date')}")
    print(f"oldest_data_date={summary.get('oldest_data_date')}")
    print(f"max_data_lag_days={summary.get('max_data_lag_days')}")
    print(f"stale_data_count={summary.get('stale_data_count')}")
    print(f"fresh_data_count={summary.get('fresh_data_count')}")

    if scan.failed_symbols:
        for fail in scan.failed_symbols:
            print(f"  failed: {fail['symbol']}: {fail['error']}")

    if not scan.results:
        print("No positive interest matches.")
        return
    try:
        import pandas as pd

        df = pd.DataFrame([result.to_row() for result in scan.results]).sort_values("Interest Score", ascending=False)
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(df.to_string(index=False))
    except Exception:
        for result in scan.results:
            print(result.to_row())


if __name__ == "__main__":
    main()

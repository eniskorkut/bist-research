from __future__ import annotations

import argparse

import borsapy as bp

from market_radar.data_access import DB_PATH, init_db
from market_radar.radar_engine import RadarConfig, scan_symbols
from market_radar.symbols import normalize_bist_symbol, validate_bist_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--index", default="XU100")
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--min-score", type=float, default=50.0)
    parser.add_argument("--min-volume-ratio", type=float, default=1.5)
    parser.add_argument("--min-turnover-ratio", type=float, default=1.5)
    parser.add_argument("--min-daily-return", type=float, default=0.0)
    parser.add_argument("--require-above-ma20", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-xu100-relative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-negative-moves", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--db-path", default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db(args.db_path)
    symbols = args.symbols or list(getattr(bp.Index(args.index), "component_symbols", []) or [])
    normalized: list[str] = []
    for raw in symbols:
        symbol = normalize_bist_symbol(str(raw))
        ok, _ = validate_bist_symbol(symbol)
        if ok:
            normalized.append(symbol)
    normalized = sorted(set(normalized))
    config = RadarConfig(
        lookback_days=args.lookback_days,
        min_avg_turnover_try_active=True,
        min_avg_turnover_try=10_000_000.0,
        min_volume_ratio_active=True,
        min_volume_ratio=args.min_volume_ratio,
        min_turnover_ratio_active=True,
        min_turnover_ratio=args.min_turnover_ratio,
        min_daily_return_active=True,
        min_daily_return=args.min_daily_return,
        min_close_position_active=True,
        min_close_position=0.65,
        breakout_mode="breakout_20d",
        require_ma20_active=args.require_above_ma20,
        require_ma50_active=False,
        min_xu100_relative_active=args.require_xu100_relative,
        min_xu100_relative=0.0,
        min_interest_score_active=True,
        min_interest_score=args.min_score,
        include_negative_moves=args.include_negative_moves,
        force_refresh=args.force,
        db_path=args.db_path,
    )
    results, _ = scan_symbols(normalized, config=config)
    if not results:
        print("No positive interest matches.")
        return
    try:
        import pandas as pd

        df = pd.DataFrame([result.to_row() for result in results]).sort_values("Interest Score", ascending=False)
        print(df.to_string(index=False))
    except Exception:
        for result in results:
            print(result.to_row())


if __name__ == "__main__":
    main()

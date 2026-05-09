from __future__ import annotations

import argparse
import json

import pandas as pd

from market_radar.backtesting.backtest_engine import (
    BacktestConfig,
    run_backtest,
    write_backtest_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="XU100")
    parser.add_argument("--lookback-days", type=int, default=520)
    parser.add_argument("--strategies", nargs="+", default=["all"])
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--db-path", default="/data/market_radar_cache.sqlite")
    parser.add_argument("--output-dir", default="/data/backtest_outputs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        index_symbol=args.index,
        lookback_days=args.lookback_days,
        strategies=args.strategies,
        max_workers=args.max_workers,
        db_path=args.db_path,
        output_dir=args.output_dir,
        force_refresh=args.force,
    )
    result = run_backtest(cfg)
    files = write_backtest_outputs(result, cfg)
    print("BACKTEST_SUMMARY")
    print(json.dumps(result.scan_summary, ensure_ascii=False))
    strategy = pd.read_csv(files["strategy_summary.csv"])
    print("STRATEGY_SUMMARY")
    if strategy.empty:
        print("No strategy rows.")
    else:
        print(strategy.to_string(index=False))
    print("OUTPUT_FILES")
    print(json.dumps(files, ensure_ascii=False))


if __name__ == "__main__":
    main()

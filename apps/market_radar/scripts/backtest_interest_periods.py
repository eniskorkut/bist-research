from __future__ import annotations

import argparse
import json
from datetime import date, datetime

import pandas as pd

from market_radar.backtesting.period_backtest_engine import (
    PeriodBacktestConfig,
    run_period_backtest,
    write_period_outputs,
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="XU100")
    parser.add_argument("--period-starts", nargs="+")
    parser.add_argument("--period-start")
    parser.add_argument("--period-end")
    parser.add_argument("--monthly", action="store_true")
    parser.add_argument("--period-ends", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="+", default=["all"])
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
    args = parser.parse_args(argv)
    if args.monthly:
        if not args.period_start or not args.period_end:
            parser.error("--monthly için --period-start ve --period-end zorunlu.")
        args.period_starts = build_monthly_period_starts(args.period_start, args.period_end)
    if not args.period_starts:
        parser.error("--period-starts veya --monthly + --period-start/--period-end vermelisiniz.")
    return args


def main() -> None:
    args = parse_args()
    cfg = PeriodBacktestConfig(
        index_symbol=args.index,
        lookback_days=args.lookback_days,
        period_starts=args.period_starts,
        period_ends=args.period_ends,
        strategies=args.strategies,
        max_workers=args.max_workers,
        cooldown_days=max(0, int(args.cooldown_days)),
        db_path=args.db_path,
        output_dir=args.output_dir,
        force_refresh=args.force,
        basket_mode="signal_weighted" if args.basket_mode == "all_signals" else args.basket_mode,
        as_of_date=args.as_of_date,
    )
    result = run_period_backtest(cfg)
    files = write_period_outputs(result, cfg)

    print("PERIOD_BACKTEST_SUMMARY")
    print(json.dumps(result.run_summary, ensure_ascii=False, default=str))
    period_count = result.run_summary.get("period_count")
    effective_as_of_date = result.run_summary.get("effective_as_of_date")
    print(f"period_count={period_count}")
    print(f"effective_as_of_date={effective_as_of_date}")
    print("PERIOD_STRATEGY_SUMMARY")
    period_df = pd.read_csv(files["period_strategy_summary.csv"])
    if period_df.empty:
        print("No period strategy rows.")
    else:
        active_period_count = int((period_df["signal_count"] > 0).sum()) if "signal_count" in period_df.columns else 0
        empty_period_count = int((period_df["signal_count"] == 0).sum()) if "signal_count" in period_df.columns else 0
        print(f"active_period_count={active_period_count}")
        print(f"empty_period_count={empty_period_count}")
        print(period_df.to_string(index=False))
    print("STRATEGY_STABILITY_SUMMARY")
    stability_df = pd.read_csv(files["strategy_stability_summary.csv"])
    if stability_df.empty:
        print("No stability rows.")
    else:
        rank_cols = [
            "strategy",
            "avg_basket_alpha_to_current",
            "avg_basket_alpha_30d",
            "median_basket_alpha_to_current",
            "active_period_count",
            "empty_period_count",
        ]
        ranked = stability_df.sort_values("avg_basket_alpha_to_current", ascending=False)
        for col in rank_cols:
            if col not in ranked.columns:
                ranked[col] = pd.NA
        print("STRATEGY_RANKING_BY_ALPHA_TO_CURRENT")
        print(ranked[rank_cols].to_string(index=False))
        print(stability_df.to_string(index=False))
    print("OUTPUT_FILES")
    print(json.dumps(files, ensure_ascii=False))


if __name__ == "__main__":
    main()

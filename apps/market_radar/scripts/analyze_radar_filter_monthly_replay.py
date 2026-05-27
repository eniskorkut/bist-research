from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_OUT_ROOT = "data/backtest_outputs/radar_filter_monthly_replay_2026"
DEFAULT_DB_PATH = "data/market_radar_cache.sqlite"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_MAX_HOLDINGS = 10
PERSISTENCE_GROUPS = ["top30", "special_loose", "special_mid", "special_strict", "special_strict_top10"]
MONTHLY_GROUPS = [
    ("top30", "top30_equal_weight"),
    ("special_mid", "special_mid_equal_weight"),
    ("special_strict", "special_strict_equal_weight"),
    ("special_strict_top10", "special_strict_top10_equal_weight"),
]


def _load_april_module():
    script_path = Path(__file__).resolve().parent / "analyze_daily_radar_april_replay.py"
    spec = importlib.util.spec_from_file_location("analyze_daily_radar_april_replay", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


APRIL = _load_april_module()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date")
    p.add_argument("--max-holdings", type=int, default=DEFAULT_MAX_HOLDINGS)
    p.add_argument("--exclude-stale", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _resolve_end_date(features: pd.DataFrame, end_date: str | None) -> str:
    if end_date:
        return end_date
    signal_dates = pd.to_datetime(features["signal_date"], errors="coerce").dropna()
    if signal_dates.empty:
        raise ValueError("No valid signal_date found in features")
    return signal_dates.max().strftime("%Y-%m-%d")


def _select_group(day_df: pd.DataFrame, group_name: str, max_holdings: int) -> pd.DataFrame:
    if day_df.empty:
        return pd.DataFrame()
    df = day_df.copy()
    if group_name == "top30":
        out = df.loc[df["production_rank"] <= 30].copy()
        out["group_rank"] = out["production_rank"]
        return out.sort_values(["group_rank", "symbol"]).reset_index(drop=True)
    if group_name == "special_loose":
        out = df.loc[df["passes_special_loose"]].copy()
        out["group_rank"] = out["production_rank"]
        return out.sort_values(["group_rank", "symbol"]).reset_index(drop=True)
    if group_name == "special_mid":
        out = df.loc[df["passes_special_mid"]].copy()
        out["group_rank"] = out["production_rank"]
        return out.sort_values(["group_rank", "symbol"]).reset_index(drop=True)
    if group_name == "special_strict":
        out = df.loc[df["passes_special_strict"]].copy()
        out["group_rank"] = out["production_rank"]
        return out.sort_values(["group_rank", "symbol"]).reset_index(drop=True)
    if group_name == "special_strict_top10":
        out = df.loc[df["passes_special_strict"]].copy()
        out = out.sort_values(["special_score", "symbol"], ascending=[False, True]).head(int(max_holdings)).reset_index(drop=True)
        out["group_rank"] = out.index + 1
        return out
    raise ValueError(f"Unknown group_name: {group_name}")


def _build_group_map(daily_all: pd.DataFrame, summary_df: pd.DataFrame, max_holdings: int) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    expected_days = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    for as_of_date in expected_days:
        day_df = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy()
        for group_name in PERSISTENCE_GROUPS:
            out[(as_of_date, group_name)] = _select_group(day_df, group_name, max_holdings)
    return out


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def build_daily_persistence_summary(
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    group_map: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_summary = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool)].copy()
    expected_dates = expected_summary["as_of_date"].astype(str).tolist()
    lifecycle_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for group_name in PERSISTENCE_GROUPS:
        seen_days: dict[str, int] = {}
        last_seen_pos: dict[str, int] = {}
        current_streak: dict[str, int] = {}
        max_streak: dict[str, int] = {}
        first_seen: dict[str, str] = {}
        last_seen_date: dict[str, str] = {}
        ranks: dict[str, list[float]] = {}
        scores: dict[str, list[float]] = {}
        reentered_count: dict[str, int] = {}
        prev_symbols: set[str] = set()

        for day_idx, as_of_date in enumerate(expected_dates):
            grp = group_map[(as_of_date, group_name)].copy()
            symbols = grp["symbol"].astype(str).tolist() if not grp.empty else []
            curr_symbols = set(symbols)
            repeated = len(prev_symbols & curr_symbols)
            new = len(curr_symbols - prev_symbols)
            dropped = len(prev_symbols - curr_symbols)
            overlap = (repeated / len(prev_symbols)) if prev_symbols else None
            jaccard = _jaccard(prev_symbols, curr_symbols)

            for _, row in grp.iterrows():
                symbol = str(row["symbol"])
                if symbol not in first_seen:
                    first_seen[symbol] = as_of_date
                    reentered_count[symbol] = 0
                if symbol in last_seen_pos and last_seen_pos[symbol] != day_idx - 1:
                    reentered_count[symbol] += 1
                    current_streak[symbol] = 0
                seen_days[symbol] = seen_days.get(symbol, 0) + 1
                current_streak[symbol] = current_streak.get(symbol, 0) + 1
                max_streak[symbol] = max(max_streak.get(symbol, 0), current_streak[symbol])
                last_seen_pos[symbol] = day_idx
                last_seen_date[symbol] = as_of_date
                ranks.setdefault(symbol, []).append(float(row.get("group_rank", row.get("production_rank", 0))))
                scores.setdefault(symbol, []).append(float(row.get("special_score", 0.0)))

            day_symbols_seen = len(seen_days)
            avg_days_seen = (sum(seen_days.values()) / day_symbols_seen) if day_symbols_seen else None
            avg_consecutive = (sum(max_streak.values()) / len(max_streak)) if max_streak else None
            max_consecutive = max(max_streak.values()) if max_streak else 0
            one_day_only_count = sum(1 for v in seen_days.values() if v == 1)
            row_meta = expected_summary.loc[expected_summary["as_of_date"].astype(str) == as_of_date].iloc[0]
            summary_rows.append(
                {
                    "as_of_date": as_of_date,
                    "group_name": group_name,
                    "count": int(len(grp)),
                    "repeated_vs_prev_trading_day": int(repeated),
                    "new_vs_prev_trading_day": int(new),
                    "dropped_vs_prev_trading_day": int(dropped),
                    "overlap_rate_vs_prev_trading_day": overlap,
                    "jaccard_vs_prev_trading_day": jaccard,
                    "one_day_only_count_to_date": int(one_day_only_count),
                    "avg_days_seen_to_date": avg_days_seen,
                    "avg_consecutive_days_to_date": avg_consecutive,
                    "max_consecutive_days_to_date": int(max_consecutive),
                    "stale_data_warning": bool(row_meta["stale_data_warning"]),
                    "stale_data_reason": row_meta["stale_data_reason"],
                }
            )
            prev_symbols = curr_symbols

        for symbol, first_date in first_seen.items():
            symbol_ranks = ranks.get(symbol, [])
            symbol_scores = scores.get(symbol, [])
            lifecycle_records.append(
                {
                    "group_name": group_name,
                    "symbol": symbol,
                    "first_seen_date": first_date,
                    "last_seen_date": last_seen_date[symbol],
                    "days_seen": int(seen_days[symbol]),
                    "max_consecutive_days": int(max_streak.get(symbol, 0)),
                    "avg_rank": float(sum(symbol_ranks) / len(symbol_ranks)) if symbol_ranks else None,
                    "best_rank": float(min(symbol_ranks)) if symbol_ranks else None,
                    "worst_rank": float(max(symbol_ranks)) if symbol_ranks else None,
                    "avg_special_score": float(sum(symbol_scores) / len(symbol_scores)) if symbol_scores else None,
                    "best_special_score": float(max(symbol_scores)) if symbol_scores else None,
                    "one_day_only": bool(seen_days[symbol] == 1),
                    "reentered_count": int(reentered_count.get(symbol, 0)),
                }
            )

    summary_out = pd.DataFrame(summary_rows).sort_values(["as_of_date", "group_name"]).reset_index(drop=True)
    lifecycle_out = pd.DataFrame(lifecycle_records).sort_values(["group_name", "days_seen", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    return summary_out, lifecycle_out


def _load_price_cache_for_symbols(symbols: list[str], db_path: str) -> dict[str, pd.DataFrame]:
    cache, _warning = APRIL._load_price_cache(db_path, symbols)
    return cache


def _close_on_or_before(frame: pd.DataFrame | None, as_of_date: str) -> float | None:
    if frame is None or frame.empty:
        return None
    asof = pd.Timestamp(as_of_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    current = frame.loc[frame["date"] <= asof]
    if current.empty:
        return None
    val = current.iloc[-1]["close"]
    return float(val) if pd.notna(val) else None


def _monthly_boundaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), ["as_of_date", "stale_data_warning", "stale_data_reason"]].copy()
    expected["month"] = expected["as_of_date"].astype(str).str.slice(0, 7)
    grouped = expected.groupby("month", sort=True)
    rows = []
    for month, grp in grouped:
        grp = grp.sort_values("as_of_date")
        rows.append(
            {
                "month": month,
                "entry_date": str(grp.iloc[0]["as_of_date"]),
                "exit_date": str(grp.iloc[-1]["as_of_date"]),
                "stale_entry_warning": bool(grp.iloc[0]["stale_data_warning"]),
                "stale_entry_reason": grp.iloc[0]["stale_data_reason"],
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_monthly_replay(
    summary_df: pd.DataFrame,
    group_map: dict[tuple[str, str], pd.DataFrame],
    db_path: str,
    max_holdings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = _monthly_boundaries(summary_df)
    all_symbols = set()
    for (_, _group_name), grp in group_map.items():
        all_symbols.update(grp["symbol"].astype(str).tolist())
    price_cache = _load_price_cache_for_symbols(sorted(all_symbols), db_path)

    month_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []

    for _, month_row in boundaries.iterrows():
        month = str(month_row["month"])
        entry_date = str(month_row["entry_date"])
        exit_date = str(month_row["exit_date"])
        stale_entry_warning = bool(month_row["stale_entry_warning"])
        stale_entry_reason = month_row["stale_entry_reason"]

        for source_group, output_group in MONTHLY_GROUPS:
            grp = group_map.get((entry_date, source_group), pd.DataFrame()).copy()
            selected_symbol_count = int(len(grp))
            included_returns: list[float] = []
            best_symbol = None
            best_return = None
            worst_symbol = None
            worst_return = None
            included_count = 0
            excluded_count = 0

            for _, row in grp.iterrows():
                symbol = str(row["symbol"])
                frame = price_cache.get(symbol)
                entry_price = _close_on_or_before(frame, entry_date)
                exit_price = _close_on_or_before(frame, exit_date)
                included = True
                exclusion_reason = ""
                stock_return = None
                if entry_price is None or exit_price is None:
                    included = False
                    excluded_count += 1
                    exclusion_reason = "excluded_missing_price"
                else:
                    stock_return = (exit_price / entry_price - 1.0) * 100.0
                    included_returns.append(stock_return)
                    included_count += 1
                    if best_return is None or stock_return > best_return:
                        best_symbol, best_return = symbol, stock_return
                    if worst_return is None or stock_return < worst_return:
                        worst_symbol, worst_return = symbol, stock_return
                position_rows.append(
                    {
                        "month": month,
                        "group_name": output_group,
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stock_return_pct": stock_return,
                        "production_rank": row.get("production_rank"),
                        "special_tier": row.get("special_tier"),
                        "special_score": row.get("special_score"),
                        "liquidity_safe_score": row.get("liquidity_safe_score"),
                        "balanced_score": row.get("balanced_score"),
                        "momentum_quality_score": row.get("momentum_quality_score"),
                        "included_in_portfolio": bool(included),
                        "exclusion_reason": exclusion_reason,
                    }
                )

            ret_series = pd.Series(included_returns, dtype=float)
            month_rows.append(
                {
                    "month": month,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "group_name": output_group,
                    "selected_symbol_count": selected_symbol_count,
                    "included_symbol_count": included_count,
                    "excluded_symbol_count": excluded_count,
                    "portfolio_return_pct": float(ret_series.mean()) if not ret_series.empty else None,
                    "median_stock_return_pct": float(ret_series.median()) if not ret_series.empty else None,
                    "win_rate_pct": float((ret_series > 0).mean() * 100.0) if not ret_series.empty else None,
                    "best_symbol": best_symbol,
                    "best_symbol_return_pct": best_return,
                    "worst_symbol": worst_symbol,
                    "worst_symbol_return_pct": worst_return,
                    "avg_entry_special_score": float(pd.to_numeric(grp["special_score"], errors="coerce").mean()) if not grp.empty else None,
                    "avg_entry_liquidity_safe_score": float(pd.to_numeric(grp["liquidity_safe_score"], errors="coerce").mean()) if not grp.empty else None,
                    "avg_entry_balanced_score": float(pd.to_numeric(grp["balanced_score"], errors="coerce").mean()) if not grp.empty else None,
                    "avg_entry_momentum_quality_score": float(pd.to_numeric(grp["momentum_quality_score"], errors="coerce").mean()) if not grp.empty else None,
                    "stale_entry_warning": stale_entry_warning,
                    "stale_entry_reason": stale_entry_reason,
                }
            )

    monthly_df = pd.DataFrame(month_rows).sort_values(["month", "group_name"]).reset_index(drop=True)
    positions_df = pd.DataFrame(position_rows).sort_values(["month", "group_name", "symbol"]).reset_index(drop=True)
    return monthly_df, positions_df


def _max_drawdown(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return None
    capital = 100.0
    capitals = []
    for ret in clean.tolist():
        capital *= 1.0 + ret / 100.0
        capitals.append(capital)
    series = pd.Series(capitals, dtype=float)
    running_max = series.cummax()
    dd = (series / running_max - 1.0) * 100.0
    return float(dd.min()) if not dd.empty else None


def summarize_monthly_performance(monthly_df: pd.DataFrame, exclude_stale: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, grp_all in monthly_df.groupby("group_name", sort=True):
        grp = grp_all.loc[~grp_all["stale_entry_warning"].astype(bool)].copy() if exclude_stale else grp_all.copy()
        returns = pd.to_numeric(grp["portfolio_return_pct"], errors="coerce")
        valid = grp.loc[returns.notna()].copy()
        valid_returns = pd.to_numeric(valid["portfolio_return_pct"], errors="coerce")

        capital = 100.0
        best_month = None
        best_ret = None
        worst_month = None
        worst_ret = None
        for _, row in valid.iterrows():
            ret = float(row["portfolio_return_pct"])
            capital *= 1.0 + ret / 100.0
            if best_ret is None or ret > best_ret:
                best_month = row["month"]
                best_ret = ret
            if worst_ret is None or ret < worst_ret:
                worst_month = row["month"]
                worst_ret = ret
        rows.append(
            {
                "group_name": group_name,
                "months_tested": int(len(valid)),
                "avg_monthly_return_pct": float(valid_returns.mean()) if not valid_returns.empty else None,
                "median_monthly_return_pct": float(valid_returns.median()) if not valid_returns.empty else None,
                "cumulative_return_pct": (capital - 100.0) if not valid_returns.empty else None,
                "positive_month_rate_pct": float((valid_returns > 0).mean() * 100.0) if not valid_returns.empty else None,
                "best_month": best_month,
                "best_month_return_pct": best_ret,
                "worst_month": worst_month,
                "worst_month_return_pct": worst_ret,
                "return_std_pct": float(valid_returns.std(ddof=0)) if not valid_returns.empty else None,
                "max_drawdown_pct": _max_drawdown(valid_returns),
                "avg_selected_symbol_count": float(pd.to_numeric(valid["selected_symbol_count"], errors="coerce").mean()) if not valid.empty else None,
                "avg_included_symbol_count": float(pd.to_numeric(valid["included_symbol_count"], errors="coerce").mean()) if not valid.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values("group_name").reset_index(drop=True)


def _console_daily_summary(persistence_df: pd.DataFrame, lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    merged = persistence_df.groupby("group_name", as_index=False).agg(
        avg_count=("count", "mean"),
        avg_overlap=("overlap_rate_vs_prev_trading_day", "mean"),
        avg_jaccard=("jaccard_vs_prev_trading_day", "mean"),
        avg_consecutive_days=("avg_consecutive_days_to_date", "mean"),
    )
    one_day = lifecycle_df.groupby("group_name", as_index=False)["one_day_only"].sum().rename(columns={"one_day_only": "one_day_only_count"})
    return merged.merge(one_day, on="group_name", how="left").sort_values("group_name").reset_index(drop=True)


def _console_monthly_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        [
            "group_name",
            "months_tested",
            "avg_monthly_return_pct",
            "cumulative_return_pct",
            "positive_month_rate_pct",
            "worst_month_return_pct",
            "max_drawdown_pct",
            "avg_selected_symbol_count",
        ]
    ].copy()


def _month_return_pivot(monthly_df: pd.DataFrame) -> pd.DataFrame:
    wanted = {
        "top30_equal_weight": "top30_return",
        "special_mid_equal_weight": "special_mid_return",
        "special_strict_equal_weight": "special_strict_return",
        "special_strict_top10_equal_weight": "special_strict_top10_return",
    }
    pivot = monthly_df.pivot(index="month", columns="group_name", values="portfolio_return_pct").reset_index()
    pivot = pivot.rename(columns=wanted)
    columns = ["month", "top30_return", "special_mid_return", "special_strict_return", "special_strict_top10_return"]
    for col in columns:
        if col not in pivot.columns:
            pivot[col] = None
    return pivot[columns].sort_values("month").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    features = APRIL._prepare_base_features(Path(args.output_dir))
    end_date = _resolve_end_date(features, args.end_date)
    daily_all, summary_df, _lifecycle_april, _perf_april, _audit_april = APRIL.build_april_replay(
        features,
        args.start_date,
        end_date,
        db_path=args.db_path,
        exclude_stale_performance=args.exclude_stale,
    )
    group_map = _build_group_map(daily_all, summary_df, args.max_holdings)
    persistence_df, lifecycle_df = build_daily_persistence_summary(daily_all, summary_df, group_map)
    monthly_df, positions_df = build_monthly_replay(summary_df, group_map, args.db_path, args.max_holdings)
    monthly_summary_df = summarize_monthly_performance(monthly_df, exclude_stale=args.exclude_stale)

    persistence_csv = out_root / "daily_filter_persistence_summary.csv"
    lifecycle_csv = out_root / "filter_symbol_lifecycle_2026.csv"
    monthly_csv = out_root / "monthly_filter_replay_2026.csv"
    positions_csv = out_root / "monthly_filter_replay_positions_2026.csv"
    monthly_summary_csv = out_root / "monthly_filter_replay_summary_2026.csv"

    persistence_df.to_csv(persistence_csv, index=False)
    lifecycle_df.to_csv(lifecycle_csv, index=False)
    monthly_df.to_csv(monthly_csv, index=False)
    positions_df.to_csv(positions_csv, index=False)
    monthly_summary_df.to_csv(monthly_summary_csv, index=False)

    console_daily = _console_daily_summary(persistence_df, lifecycle_df)
    console_monthly = _console_monthly_summary(monthly_summary_df)
    month_pivot = _month_return_pivot(monthly_df)

    print("DAILY_PERSISTENCE")
    print(console_daily.to_string(index=False))
    print("MONTHLY_PERFORMANCE")
    print(console_monthly.to_string(index=False))
    print("MONTHLY_RETURNS")
    print(month_pivot.to_string(index=False))

    strict_row = console_daily.loc[console_daily["group_name"] == "special_strict"].iloc[0] if not console_daily.empty and "special_strict" in console_daily["group_name"].tolist() else None
    top10_row = console_daily.loc[console_daily["group_name"] == "special_strict_top10"].iloc[0] if not console_daily.empty and "special_strict_top10" in console_daily["group_name"].tolist() else None
    top30_perf = monthly_summary_df.loc[monthly_summary_df["group_name"] == "top30_equal_weight"].iloc[0] if "top30_equal_weight" in monthly_summary_df["group_name"].tolist() else None
    mid_perf = monthly_summary_df.loc[monthly_summary_df["group_name"] == "special_mid_equal_weight"].iloc[0] if "special_mid_equal_weight" in monthly_summary_df["group_name"].tolist() else None
    strict_perf = monthly_summary_df.loc[monthly_summary_df["group_name"] == "special_strict_equal_weight"].iloc[0] if "special_strict_equal_weight" in monthly_summary_df["group_name"].tolist() else None

    print("SHORT_ANSWERS")
    print(
        {
            "special_strict_repeats_next_day": None if strict_row is None else float(strict_row["avg_overlap"]),
            "special_strict_top10_stability": None if top10_row is None else float(top10_row["avg_overlap"]),
            "filtering_vs_top30": None
            if any(x is None for x in [top30_perf, mid_perf, strict_perf])
            else {
                "top30": float(top30_perf["cumulative_return_pct"]) if pd.notna(top30_perf["cumulative_return_pct"]) else None,
                "special_mid": float(mid_perf["cumulative_return_pct"]) if pd.notna(mid_perf["cumulative_return_pct"]) else None,
                "special_strict": float(strict_perf["cumulative_return_pct"]) if pd.notna(strict_perf["cumulative_return_pct"]) else None,
            },
            "live_usage_preference": "special_strict_equal_weight",
        }
    )


if __name__ == "__main__":
    main()

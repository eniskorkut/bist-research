from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_OUT_ROOT = "data/backtest_outputs/radar_rolling_hold20_portfolio_2026"
DEFAULT_DB_PATH = "data/market_radar_cache.sqlite"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_MAX_HOLDINGS = 10
DEFAULT_HOLDING_DAYS = 20
DEFAULT_MIN_POSITION_VALUE = 100.0
BENCHMARKS = [("XU100", "xu100"), ("XUTUM", "xutum")]
STRATEGY_NAMES = [
    "top30_fresh_only",
    "special_mid_fresh_only",
    "special_strict_fresh_only",
    "special_strict_top10_fresh_only",
    "special_strict_pending_ttl3_raw",
    "special_strict_top10_pending_ttl3_raw",
    "special_strict_pending_ttl3_revalidate",
    "special_strict_top10_pending_ttl3_revalidate",
    "special_strict_pending_ttl5_revalidate",
    "special_strict_top10_pending_ttl5_revalidate",
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
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--max-holdings", type=int, default=DEFAULT_MAX_HOLDINGS)
    p.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    p.add_argument("--entry-mode", choices=["next_open", "same_close"], default="next_open")
    p.add_argument("--exit-mode", choices=["hold20_close"], default="hold20_close")
    p.add_argument("--exclude-stale-new-entries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-same-day-reentry", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-position-value", type=float, default=DEFAULT_MIN_POSITION_VALUE)
    return p.parse_args()


def _resolve_end_date(features: pd.DataFrame, end_date: str | None) -> str:
    if end_date:
        return end_date
    signal_dates = pd.to_datetime(features["signal_date"], errors="coerce").dropna()
    if signal_dates.empty:
        raise ValueError("No valid signal_date found")
    return signal_dates.max().strftime("%Y-%m-%d")


def _parse_strategy_name(strategy_name: str, max_holdings: int) -> dict[str, Any]:
    if strategy_name.startswith("top30"):
        return {"base_group": "top30", "pending_mode": False, "ttl": None, "revalidate": False, "limit": 30}
    if strategy_name.startswith("special_mid"):
        return {"base_group": "special_mid", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    pending_mode = "pending" in strategy_name
    revalidate = "revalidate" in strategy_name
    ttl = 3 if "ttl3" in strategy_name else (5 if "ttl5" in strategy_name else None)
    if "top10" in strategy_name:
        return {"base_group": "special_strict_top10", "pending_mode": pending_mode, "ttl": ttl, "revalidate": revalidate, "limit": max_holdings}
    return {"base_group": "special_strict", "pending_mode": pending_mode, "ttl": ttl, "revalidate": revalidate, "limit": max_holdings}


def _select_group(day_df: pd.DataFrame, base_group: str, max_holdings: int) -> pd.DataFrame:
    if day_df.empty:
        return pd.DataFrame()
    if base_group == "top30":
        out = day_df.loc[day_df["production_rank"] <= 30].copy()
        out = out.sort_values(["production_rank", "symbol"]).reset_index(drop=True)
        out["selection_rank"] = out["production_rank"]
        return out
    if base_group == "special_mid":
        out = day_df.loc[day_df["passes_special_mid"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict":
        out = day_df.loc[day_df["passes_special_strict"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_top10":
        out = day_df.loc[day_df["passes_special_strict"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).head(int(max_holdings)).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    raise ValueError(f"Unknown base_group {base_group}")


def _build_daily_group_map(daily_all: pd.DataFrame, summary_df: pd.DataFrame, strategy_name: str, max_holdings: int) -> dict[str, pd.DataFrame]:
    cfg = _parse_strategy_name(strategy_name, max_holdings)
    expected_dates = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    out: dict[str, pd.DataFrame] = {}
    for as_of_date in expected_dates:
        grp = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy()
        out[as_of_date] = _select_group(grp, cfg["base_group"], max_holdings)
    return out


def _build_daily_symbol_map(daily_all: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    expected_dates = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for as_of_date in expected_dates:
        grp = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy()
        if grp.empty:
            out[as_of_date] = {}
        else:
            out[as_of_date] = grp.set_index("symbol").to_dict("index")
    return out


def _load_price_cache_full(db_path: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(db_path) as conn:
        for symbol in sorted(set(symbols)):
            row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol = ?", (symbol,)).fetchone()
            if not row:
                continue
            payload = json.loads(row[0])
            frame = pd.DataFrame.from_records(payload.get("records") or [])
            if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
                continue
            for col in ["open", "close", "low"]:
                if col not in frame.columns:
                    frame[col] = frame["close"]
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
            frame = frame.dropna(subset=["date", "open", "close", "low"]).sort_values("date").reset_index(drop=True)
            if frame.empty:
                continue
            frame["date_str"] = frame["date"].dt.strftime("%Y-%m-%d")
            cache[symbol] = frame[["date", "date_str", "open", "close", "low"]].copy()
    return cache


def _price_on_date(frame: pd.DataFrame | None, date_str: str, column: str) -> float | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    row = frame.loc[frame["date_str"] == date_str]
    if row.empty:
        return None
    value = row.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _prepare_inputs(
    output_dir: str,
    db_path: str,
    start_date: str,
    end_date: str | None,
    exclude_stale: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], str]:
    features = APRIL._prepare_base_features(Path(output_dir))
    resolved_end = _resolve_end_date(features, end_date)
    daily_all, summary_df, _life, _perf, _audit = APRIL.build_april_replay(
        features,
        start_date,
        resolved_end,
        db_path=db_path,
        exclude_stale_performance=exclude_stale,
    )
    symbols = sorted(set(daily_all["symbol"].astype(str).tolist()) | {"XU100", "XUTUM"})
    price_cache = _load_price_cache_full(db_path, symbols)
    return daily_all, summary_df, price_cache, resolved_end


def _revalidate_pending(
    row: pd.Series | None,
    stale_day: bool,
    min_liquidity_safe: float,
) -> tuple[bool, str]:
    if row is None:
        return False, "revalidate_missing_row"
    if stale_day:
        return False, "revalidate_stale"
    for field in ["turnover", "volume", "avg_turnover_20d", "liquidity_safe_score", "rsi_14"]:
        if pd.isna(row.get(field)):
            return False, f"revalidate_missing_{field}"
    if float(row["rsi_14"]) > 78.0:
        return False, "revalidate_rsi_too_high"
    if float(row["liquidity_safe_score"]) < float(min_liquidity_safe):
        return False, "revalidate_liquidity_safe_below_min"
    return True, ""


def _compute_exit_date(entry_trade_date: str, trading_dates: list[str], idx_map: dict[str, int], holding_days: int) -> str | None:
    idx = idx_map.get(entry_trade_date)
    if idx is None:
        return None
    exit_idx = idx + int(holding_days) - 1
    if exit_idx >= len(trading_dates):
        return None
    return trading_dates[exit_idx]


def _event(strategy_name: str, as_of_date: str, order: dict[str, Any], event_type: str, reason: str, age: int | None) -> dict[str, Any]:
    return {
        "as_of_date": as_of_date,
        "strategy_name": strategy_name,
        "symbol": order["symbol"],
        "original_signal_date": order["original_signal_date"],
        "event_type": event_type,
        "reason": reason,
        "pending_age_trading_days": age,
        "special_score": order.get("special_score"),
        "liquidity_safe_score": order.get("liquidity_safe_score"),
        "production_rank": order.get("production_rank"),
    }


def _pending_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["original_signal_date"],
        float(item.get("special_score") or float("-inf")),
        float(item.get("liquidity_safe_score") or float("-inf")),
        -float(item.get("production_rank") or 10**9),
        item["symbol"],
    )


def _enqueue_pending(queue: list[dict[str, Any]], order: dict[str, Any], ttl: int) -> None:
    queue[:] = [item for item in queue if item["symbol"] != order["symbol"]]
    queue.append({**order, "ttl": ttl})


def _current_equity(cash: float, positions: list[dict[str, Any]], price_cache: dict[str, pd.DataFrame], as_of_date: str) -> tuple[float, float, float]:
    open_value = 0.0
    open_cost_basis = 0.0
    for pos in positions:
        close_price = _price_on_date(price_cache.get(pos["symbol"]), as_of_date, "close")
        if close_price is None:
            close_price = pos["entry_price"]
        open_value += pos["shares"] * close_price
        open_cost_basis += pos["entry_value"]
    total_equity = float(cash) + float(open_value)
    return total_equity, open_value, open_cost_basis


def _base_group_revalidate_threshold(strategy_name: str) -> float:
    if "special_mid" in strategy_name:
        return 70.0
    return 74.0


def _buy_position(
    *,
    order: dict[str, Any],
    trade_date: str,
    entry_price: float,
    cash: float,
    positions: list[dict[str, Any]],
    price_cache: dict[str, pd.DataFrame],
    max_holdings: int,
    min_position_value: float,
    trading_dates: list[str],
    idx_map: dict[str, int],
    holding_days: int,
) -> tuple[float, bool]:
    total_equity, _open_value, _open_cost = _current_equity(cash, positions, price_cache, trade_date)
    position_budget = total_equity / float(max_holdings)
    position_value = min(position_budget, cash)
    if position_value < float(min_position_value) or entry_price <= 0:
        return cash, False
    shares = position_value / entry_price
    exit_trade_date = _compute_exit_date(trade_date, trading_dates, idx_map, holding_days)
    positions.append(
        {
            **order,
            "entry_trade_date": trade_date,
            "entry_price": float(entry_price),
            "shares": float(shares),
            "entry_value": float(position_value),
            "exit_trade_date": exit_trade_date,
        }
    )
    return cash - float(position_value), True


def simulate_strategy(
    strategy_name: str,
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    *,
    initial_capital: float,
    max_holdings: int,
    holding_days: int,
    entry_mode: str,
    exit_mode: str,
    exclude_stale_new_entries: bool,
    allow_same_day_reentry: bool,
    min_position_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exit_mode != "hold20_close":
        raise ValueError(f"Unsupported exit_mode: {exit_mode}")
    cfg = _parse_strategy_name(strategy_name, max_holdings)
    daily_map = _build_daily_group_map(daily_all, summary_df, strategy_name, max_holdings)
    symbol_map = _build_daily_symbol_map(daily_all, summary_df)
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool)].copy().sort_values("as_of_date").reset_index(drop=True)
    trading_dates = expected["as_of_date"].astype(str).tolist()
    idx_map = {d: i for i, d in enumerate(trading_dates)}

    cash = float(initial_capital)
    positions: list[dict[str, Any]] = []
    pending_queue: list[dict[str, Any]] = []
    scheduled_entries: dict[str, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_equity = float(initial_capital)

    for as_of_date in trading_dates:
        meta = expected.loc[expected["as_of_date"].astype(str) == as_of_date].iloc[0]
        stale_day = bool(meta["stale_data_warning"])
        stale_reason = meta["stale_data_reason"]
        group_df = daily_map.get(as_of_date, pd.DataFrame()).copy()
        day_symbols = symbol_map.get(as_of_date, {})
        sold_today: set[str] = set()
        opened_today: set[str] = set()
        new_entries_count = 0
        exits_count = 0
        skipped_duplicate_symbol_count = 0
        skipped_no_slot_count = 0
        pending_expired_count = 0

        if entry_mode == "next_open":
            todays_scheduled = scheduled_entries.pop(as_of_date, [])
            for order in todays_scheduled:
                symbol = order["symbol"]
                if any(pos["symbol"] == symbol for pos in positions) or symbol in sold_today or symbol in opened_today:
                    skipped_duplicate_symbol_count += 1
                    pending_events.append(_event(strategy_name, as_of_date, order, "skipped_duplicate", "symbol_already_open_or_sold_today", order.get("pending_age_trading_days")))
                    continue
                entry_price = _price_on_date(price_cache.get(symbol), as_of_date, "open")
                if entry_price is None:
                    pending_events.append(_event(strategy_name, as_of_date, order, "revalidate_failed" if order["source_type"].startswith("pending") else "skipped_no_slot", "missing_entry_open", order.get("pending_age_trading_days")))
                    continue
                cash, bought = _buy_position(
                    order=order,
                    trade_date=as_of_date,
                    entry_price=entry_price,
                    cash=cash,
                    positions=positions,
                    price_cache=price_cache,
                    max_holdings=max_holdings,
                    min_position_value=min_position_value,
                    trading_dates=trading_dates,
                    idx_map=idx_map,
                    holding_days=holding_days,
                )
                if bought:
                    opened_today.add(symbol)
                    new_entries_count += 1

        remaining_positions: list[dict[str, Any]] = []
        for pos in positions:
            if pos.get("exit_trade_date") != as_of_date:
                remaining_positions.append(pos)
                continue
            exit_price = _price_on_date(price_cache.get(pos["symbol"]), as_of_date, "close")
            if exit_price is None:
                remaining_positions.append(pos)
                continue
            exit_value = pos["shares"] * exit_price
            pnl = exit_value - pos["entry_value"]
            return_pct = (exit_value / pos["entry_value"] - 1.0) * 100.0 if pos["entry_value"] else None
            cash += exit_value
            sold_today.add(pos["symbol"])
            exits_count += 1
            trades.append(
                {
                    "strategy_name": strategy_name,
                    "symbol": pos["symbol"],
                    "original_signal_date": pos["original_signal_date"],
                    "entry_trade_date": pos["entry_trade_date"],
                    "exit_trade_date": as_of_date,
                    "holding_trading_days": holding_days,
                    "entry_price": pos["entry_price"],
                    "exit_price": float(exit_price),
                    "shares": pos["shares"],
                    "entry_value": pos["entry_value"],
                    "exit_value": float(exit_value),
                    "pnl": float(pnl),
                    "return_pct": float(return_pct) if return_pct is not None else None,
                    "exit_reason": "hold20_close",
                    "production_rank": pos.get("production_rank"),
                    "special_tier": pos.get("special_tier"),
                    "special_score": pos.get("special_score"),
                    "liquidity_safe_score": pos.get("liquidity_safe_score"),
                    "balanced_score": pos.get("balanced_score"),
                    "momentum_quality_score": pos.get("momentum_quality_score"),
                    "source_type": pos.get("source_type"),
                    "pending_age_trading_days": pos.get("pending_age_trading_days"),
                    "revalidated_in_same_group": pos.get("revalidated_in_same_group"),
                    "stale_entry_warning": pos.get("stale_entry_warning"),
                }
            )
        positions = remaining_positions

        refreshed_queue: list[dict[str, Any]] = []
        for item in pending_queue:
            age = idx_map[as_of_date] - idx_map[item["original_signal_date"]]
            if item["ttl"] is not None and age >= int(item["ttl"]):
                pending_expired_count += 1
                pending_events.append(_event(strategy_name, as_of_date, item, "expired", "ttl_expired", age))
            else:
                item["pending_age_trading_days"] = age
                refreshed_queue.append(item)
        pending_queue = refreshed_queue

        def can_take_new(symbol: str) -> tuple[bool, str]:
            if symbol in {pos["symbol"] for pos in positions} or symbol in opened_today:
                return False, "symbol_already_open_or_sold_today"
            if (not allow_same_day_reentry) and symbol in sold_today:
                return False, "symbol_already_open_or_sold_today"
            if len(positions) + sum(len(v) for v in scheduled_entries.values()) >= max_holdings and entry_mode == "next_open":
                return False, "no_available_slot"
            if len(positions) >= max_holdings and entry_mode == "same_close":
                return False, "no_available_slot"
            return True, ""

        def process_order(order: dict[str, Any], source_kind: str) -> tuple[bool, str]:
            nonlocal cash, new_entries_count, skipped_duplicate_symbol_count, skipped_no_slot_count
            symbol = order["symbol"]
            allowed, reason = can_take_new(symbol)
            if not allowed:
                if reason == "symbol_already_open_or_sold_today":
                    skipped_duplicate_symbol_count += 1
                else:
                    skipped_no_slot_count += 1
                if cfg["pending_mode"] and source_kind == "fresh" and reason == "no_available_slot":
                    _enqueue_pending(pending_queue, order, int(cfg["ttl"]))
                    pending_events.append(_event(strategy_name, as_of_date, order, "queued", "no_available_slot", 0))
                else:
                    pending_events.append(_event(strategy_name, as_of_date, order, "skipped_duplicate" if reason == "symbol_already_open_or_sold_today" else "skipped_no_slot", reason, order.get("pending_age_trading_days")))
                return False, reason
            if entry_mode == "same_close":
                entry_price = _price_on_date(price_cache.get(symbol), as_of_date, "close")
                if entry_price is None:
                    pending_events.append(_event(strategy_name, as_of_date, order, "revalidate_failed" if order["source_type"].startswith("pending") else "skipped_no_slot", "missing_entry_close", order.get("pending_age_trading_days")))
                    return False, "missing_entry_close"
                cash, bought = _buy_position(
                    order=order,
                    trade_date=as_of_date,
                    entry_price=entry_price,
                    cash=cash,
                    positions=positions,
                    price_cache=price_cache,
                    max_holdings=max_holdings,
                    min_position_value=min_position_value,
                    trading_dates=trading_dates,
                    idx_map=idx_map,
                    holding_days=holding_days,
                )
                if bought:
                    opened_today.add(symbol)
                    new_entries_count += 1
                    return True, "bought_same_close"
                return False, "position_value_below_min"
            idx = idx_map[as_of_date]
            if idx + 1 >= len(trading_dates):
                skipped_no_slot_count += 1
                pending_events.append(_event(strategy_name, as_of_date, order, "skipped_no_slot", "no_next_trading_day", order.get("pending_age_trading_days")))
                return False, "no_next_trading_day"
            scheduled_entries.setdefault(trading_dates[idx + 1], []).append(order)
            return True, "scheduled_for_next_open"

        if not (exclude_stale_new_entries and stale_day):
            fresh_orders = []
            for _, row in group_df.iterrows():
                fresh_orders.append(
                    {
                        "symbol": str(row["symbol"]),
                        "original_signal_date": as_of_date,
                        "production_rank": row.get("production_rank"),
                        "special_tier": row.get("special_tier"),
                        "special_score": row.get("special_score"),
                        "liquidity_safe_score": row.get("liquidity_safe_score"),
                        "balanced_score": row.get("balanced_score"),
                        "momentum_quality_score": row.get("momentum_quality_score"),
                        "source_type": "fresh",
                        "pending_age_trading_days": 0,
                        "revalidated_in_same_group": None,
                        "stale_entry_warning": stale_day,
                    }
                )
            for order in fresh_orders:
                process_order(order, "fresh")

            if cfg["pending_mode"]:
                ordered_queue = sorted(pending_queue, key=_pending_sort_key, reverse=True)
                kept_queue: list[dict[str, Any]] = []
                selected_today_symbols = set(group_df["symbol"].astype(str).tolist()) if not group_df.empty else set()
                min_liq = _base_group_revalidate_threshold(strategy_name)
                for item in ordered_queue:
                    if len(positions) + sum(len(v) for v in scheduled_entries.values()) >= max_holdings and entry_mode == "next_open":
                        kept_queue.append(item)
                        continue
                    if len(positions) >= max_holdings and entry_mode == "same_close":
                        kept_queue.append(item)
                        continue
                    symbol = item["symbol"]
                    if symbol in {pos["symbol"] for pos in positions} or symbol in sold_today or symbol in opened_today:
                        kept_queue.append(item)
                        continue
                    age = idx_map[as_of_date] - idx_map[item["original_signal_date"]]
                    order = {
                        **item,
                        "source_type": "pending_raw",
                        "pending_age_trading_days": age,
                        "revalidated_in_same_group": None,
                        "stale_entry_warning": stale_day,
                    }
                    if cfg["revalidate"]:
                        row_dict = day_symbols.get(symbol)
                        row_series = pd.Series(row_dict) if row_dict is not None else None
                        ok, reason = _revalidate_pending(row_series, stale_day, min_liq)
                        if not ok:
                            kept_queue.append(item)
                            pending_events.append(_event(strategy_name, as_of_date, order, "revalidate_failed", reason, age))
                            continue
                        order["source_type"] = "pending_revalidated"
                        order["revalidated_in_same_group"] = bool(symbol in selected_today_symbols)
                    bought, reason = process_order(order, "pending")
                    if bought:
                        pending_events.append(_event(strategy_name, as_of_date, order, "bought_from_queue", reason, age))
                    else:
                        kept_queue.append(item)
                pending_queue = kept_queue

        total_equity, open_position_value, open_cost_basis = _current_equity(cash, positions, price_cache, as_of_date)
        daily_return_pct = ((total_equity / prev_equity) - 1.0) * 100.0 if prev_equity else None
        cumulative_return_pct = ((total_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else None
        daily_rows.append(
            {
                "as_of_date": as_of_date,
                "strategy_name": strategy_name,
                "cash": float(cash),
                "open_position_value": float(open_position_value),
                "open_cost_basis": float(open_cost_basis),
                "total_equity": float(total_equity),
                "daily_return_pct": daily_return_pct,
                "cumulative_return_pct": cumulative_return_pct,
                "open_position_count": int(len(positions)),
                "available_slots": int(max(0, max_holdings - len(positions))),
                "new_entries_count": int(new_entries_count),
                "exits_count": int(exits_count),
                "skipped_duplicate_symbol_count": int(skipped_duplicate_symbol_count),
                "skipped_no_slot_count": int(skipped_no_slot_count),
                "pending_queue_size": int(len(pending_queue)),
                "pending_expired_count": int(pending_expired_count),
                "stale_data_warning": stale_day,
                "stale_data_reason": stale_reason,
            }
        )
        prev_equity = total_equity

    for pos in positions:
        trades.append(
            {
                "strategy_name": strategy_name,
                "symbol": pos["symbol"],
                "original_signal_date": pos["original_signal_date"],
                "entry_trade_date": pos["entry_trade_date"],
                "exit_trade_date": None,
                "holding_trading_days": holding_days,
                "entry_price": pos["entry_price"],
                "exit_price": None,
                "shares": pos["shares"],
                "entry_value": pos["entry_value"],
                "exit_value": None,
                "pnl": None,
                "return_pct": None,
                "exit_reason": "open",
                "production_rank": pos.get("production_rank"),
                "special_tier": pos.get("special_tier"),
                "special_score": pos.get("special_score"),
                "liquidity_safe_score": pos.get("liquidity_safe_score"),
                "balanced_score": pos.get("balanced_score"),
                "momentum_quality_score": pos.get("momentum_quality_score"),
                "source_type": pos.get("source_type"),
                "pending_age_trading_days": pos.get("pending_age_trading_days"),
                "revalidated_in_same_group": pos.get("revalidated_in_same_group"),
                "stale_entry_warning": pos.get("stale_entry_warning"),
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(trades), pd.DataFrame(pending_events)


def _monthly_benchmark(summary_df: pd.DataFrame, price_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), ["as_of_date"]].copy()
    expected["month"] = expected["as_of_date"].astype(str).str.slice(0, 7)
    rows: list[dict[str, Any]] = []
    for month, grp in expected.groupby("month", sort=True):
        grp = grp.sort_values("as_of_date")
        entry_date = str(grp.iloc[0]["as_of_date"])
        exit_date = str(grp.iloc[-1]["as_of_date"])
        row: dict[str, Any] = {"month": month}
        for symbol, prefix in BENCHMARKS:
            entry = _price_on_date(price_cache.get(symbol), entry_date, "close")
            exitp = _price_on_date(price_cache.get(symbol), exit_date, "close")
            row[f"{prefix}_return_pct"] = ((exitp / entry - 1.0) * 100.0) if entry not in (None, 0) and exitp is not None else None
            row[f"{prefix}_missing"] = bool(entry in (None, 0) or exitp is None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _monthly_strategy_metrics(
    strategy_name: str,
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    daily = daily_df.copy()
    daily["month"] = daily["as_of_date"].astype(str).str.slice(0, 7)
    closed = trades_df.loc[trades_df["exit_trade_date"].notna()].copy()
    if not closed.empty:
        closed["month"] = closed["exit_trade_date"].astype(str).str.slice(0, 7)
    for month, grp in daily.groupby("month", sort=True):
        grp = grp.sort_values("as_of_date")
        month_start_equity = float(grp.iloc[0]["total_equity"])
        month_end_equity = float(grp.iloc[-1]["total_equity"])
        month_return = (month_end_equity / month_start_equity - 1.0) * 100.0 if month_start_equity else None
        closed_grp = closed.loc[closed["month"] == month].copy() if not closed.empty else pd.DataFrame()
        bench = benchmark_df.loc[benchmark_df["month"] == month]
        xu100 = bench["xu100_return_pct"].iloc[0] if not bench.empty else None
        xutum = bench["xutum_return_pct"].iloc[0] if not bench.empty else None
        eq_series = pd.to_numeric(grp["total_equity"], errors="coerce")
        running_max = eq_series.cummax()
        dd = ((eq_series / running_max) - 1.0) * 100.0
        rows.append(
            {
                "month": month,
                "strategy_name": strategy_name,
                "month_start_equity": month_start_equity,
                "month_end_equity": month_end_equity,
                "month_return_pct": month_return,
                "realized_pnl": float(pd.to_numeric(closed_grp["pnl"], errors="coerce").sum()) if not closed_grp.empty else 0.0,
                "unrealized_pnl": float(grp.iloc[-1]["open_position_value"] - grp.iloc[-1]["open_cost_basis"]) if not grp.empty else 0.0,
                "closed_trades_count": int(len(closed_grp)),
                "opened_trades_count": int((trades_df["entry_trade_date"].astype(str).str.slice(0, 7) == month).sum()) if not trades_df.empty else 0,
                "win_rate_closed_trades_pct": float((pd.to_numeric(closed_grp["return_pct"], errors="coerce") > 0).mean() * 100.0) if not closed_grp.empty else None,
                "avg_closed_trade_return_pct": float(pd.to_numeric(closed_grp["return_pct"], errors="coerce").mean()) if not closed_grp.empty else None,
                "open_positions_month_end": int(grp.iloc[-1]["open_position_count"]),
                "avg_open_position_count": float(pd.to_numeric(grp["open_position_count"], errors="coerce").mean()),
                "avg_cash_ratio_pct": float((pd.to_numeric(grp["cash"], errors="coerce") / pd.to_numeric(grp["total_equity"], errors="coerce")).mean() * 100.0),
                "max_drawdown_in_month_pct": float(dd.min()) if not dd.empty else None,
                "xu100_return_pct": xu100,
                "xutum_return_pct": xutum,
                "excess_vs_xu100_pct": (month_return - xu100) if month_return is not None and xu100 is not None else None,
                "excess_vs_xutum_pct": (month_return - xutum) if month_return is not None and xutum is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _benchmark_total(summary_df: pd.DataFrame, price_cache: dict[str, pd.DataFrame]) -> dict[str, float | None]:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    if not expected:
        return {"xu100_total_return_pct": None, "xutum_total_return_pct": None}
    start = expected[0]
    end = expected[-1]
    out: dict[str, float | None] = {}
    for symbol, prefix in BENCHMARKS:
        start_close = _price_on_date(price_cache.get(symbol), start, "close")
        end_close = _price_on_date(price_cache.get(symbol), end, "close")
        out[f"{prefix}_total_return_pct"] = ((end_close / start_close - 1.0) * 100.0) if start_close not in (None, 0) and end_close is not None else None
    return out


def _summarize_strategy(
    strategy_name: str,
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    benchmark_total: dict[str, float | None],
    pending_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict[str, Any]:
    final_equity = float(daily_df.iloc[-1]["total_equity"]) if not daily_df.empty else float(initial_capital)
    total_return = ((final_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else None
    eq_series = pd.to_numeric(daily_df["total_equity"], errors="coerce")
    running_max = eq_series.cummax()
    dd = ((eq_series / running_max) - 1.0) * 100.0 if not eq_series.empty else pd.Series(dtype=float)
    closed = trades_df.loc[trades_df["exit_trade_date"].notna()].copy()
    trade_returns = pd.to_numeric(closed["return_pct"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)
    month_returns = pd.to_numeric(monthly_df["month_return_pct"], errors="coerce").dropna() if not monthly_df.empty else pd.Series(dtype=float)
    best_month_idx = month_returns.idxmax() if not month_returns.empty else None
    worst_month_idx = month_returns.idxmin() if not month_returns.empty else None
    return {
        "strategy_name": strategy_name,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "xu100_total_return_pct": benchmark_total.get("xu100_total_return_pct"),
        "xutum_total_return_pct": benchmark_total.get("xutum_total_return_pct"),
        "excess_vs_xu100_pct": (total_return - benchmark_total["xu100_total_return_pct"]) if total_return is not None and benchmark_total.get("xu100_total_return_pct") is not None else None,
        "excess_vs_xutum_pct": (total_return - benchmark_total["xutum_total_return_pct"]) if total_return is not None and benchmark_total.get("xutum_total_return_pct") is not None else None,
        "max_drawdown_pct": float(dd.min()) if not dd.empty else None,
        "months_tested": int(len(month_returns)),
        "positive_month_rate_pct": float((month_returns > 0).mean() * 100.0) if not month_returns.empty else None,
        "avg_monthly_return_pct": float(month_returns.mean()) if not month_returns.empty else None,
        "median_monthly_return_pct": float(month_returns.median()) if not month_returns.empty else None,
        "best_month": None if best_month_idx is None else str(monthly_df.loc[best_month_idx, "month"]),
        "best_month_return_pct": None if best_month_idx is None else float(monthly_df.loc[best_month_idx, "month_return_pct"]),
        "worst_month": None if worst_month_idx is None else str(monthly_df.loc[worst_month_idx, "month"]),
        "worst_month_return_pct": None if worst_month_idx is None else float(monthly_df.loc[worst_month_idx, "month_return_pct"]),
        "trade_count": int(len(trades_df)),
        "closed_trade_count": int(len(closed)),
        "open_trade_count": int(len(trades_df) - len(closed)),
        "win_rate_pct": float((trade_returns > 0).mean() * 100.0) if not trade_returns.empty else None,
        "avg_trade_return_pct": float(trade_returns.mean()) if not trade_returns.empty else None,
        "median_trade_return_pct": float(trade_returns.median()) if not trade_returns.empty else None,
        "best_trade_return_pct": float(trade_returns.max()) if not trade_returns.empty else None,
        "worst_trade_return_pct": float(trade_returns.min()) if not trade_returns.empty else None,
        "avg_open_position_count": float(pd.to_numeric(daily_df["open_position_count"], errors="coerce").mean()) if not daily_df.empty else None,
        "avg_cash_ratio_pct": float((pd.to_numeric(daily_df["cash"], errors="coerce") / pd.to_numeric(daily_df["total_equity"], errors="coerce")).mean() * 100.0) if not daily_df.empty else None,
        "skipped_duplicate_symbol_count": int(pd.to_numeric(daily_df["skipped_duplicate_symbol_count"], errors="coerce").sum()) if not daily_df.empty else 0,
        "skipped_no_slot_count": int(pd.to_numeric(daily_df["skipped_no_slot_count"], errors="coerce").sum()) if not daily_df.empty else 0,
        "pending_queued_count": int((pending_df["event_type"] == "queued").sum()) if not pending_df.empty else 0,
        "pending_bought_count": int((pending_df["event_type"] == "bought_from_queue").sum()) if not pending_df.empty else 0,
        "pending_expired_count": int((pending_df["event_type"] == "expired").sum()) if not pending_df.empty else 0,
        "pending_revalidate_failed_count": int((pending_df["event_type"] == "revalidate_failed").sum()) if not pending_df.empty else 0,
    }


def run_simulation(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    daily_all, summary_df, price_cache, resolved_end = _prepare_inputs(
        args.output_dir,
        args.db_path,
        args.start_date,
        args.end_date,
        args.exclude_stale_new_entries,
    )

    all_daily: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_pending: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    benchmark_monthly = _monthly_benchmark(summary_df, price_cache)
    benchmark_total = _benchmark_total(summary_df, price_cache)

    for strategy_name in STRATEGY_NAMES:
        daily_df, trades_df, pending_df = simulate_strategy(
            strategy_name,
            daily_all,
            summary_df,
            price_cache,
            initial_capital=float(args.initial_capital),
            max_holdings=int(args.max_holdings),
            holding_days=int(args.holding_days),
            entry_mode=args.entry_mode,
            exit_mode=args.exit_mode,
            exclude_stale_new_entries=bool(args.exclude_stale_new_entries),
            allow_same_day_reentry=bool(args.allow_same_day_reentry),
            min_position_value=float(args.min_position_value),
        )
        monthly_df = _monthly_strategy_metrics(strategy_name, daily_df, trades_df, benchmark_monthly)
        all_daily.append(daily_df)
        all_trades.append(trades_df)
        all_pending.append(pending_df)
        all_monthly.append(monthly_df)
        summary_rows.append(
            _summarize_strategy(
                strategy_name,
                daily_df,
                trades_df,
                monthly_df,
                benchmark_total,
                pending_df,
                args.start_date,
                resolved_end,
                float(args.initial_capital),
            )
        )

    return {
        "daily": pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame(),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "pending": pd.concat(all_pending, ignore_index=True) if all_pending else pd.DataFrame(),
        "monthly": pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame(),
        "summary": pd.DataFrame(summary_rows).sort_values("strategy_name").reset_index(drop=True),
        "benchmark_monthly": benchmark_monthly,
    }


def _console_general(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy_name",
        "total_return_pct",
        "final_equity",
        "max_drawdown_pct",
        "win_rate_pct",
        "trade_count",
        "avg_open_position_count",
        "avg_cash_ratio_pct",
        "excess_vs_xu100_pct",
        "excess_vs_xutum_pct",
    ]
    return summary_df[cols].copy()


def _console_monthly(monthly_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    pivot = monthly_df.pivot(index="month", columns="strategy_name", values="month_return_pct").reset_index()
    for col in [
        "top30_fresh_only",
        "special_strict_fresh_only",
        "special_strict_top10_fresh_only",
        "special_strict_top10_pending_ttl3_revalidate",
    ]:
        if col not in pivot.columns:
            pivot[col] = None
    out = pivot[
        [
            "month",
            "top30_fresh_only",
            "special_strict_fresh_only",
            "special_strict_top10_fresh_only",
            "special_strict_top10_pending_ttl3_revalidate",
        ]
    ].copy()
    out = out.merge(benchmark_df[["month", "xu100_return_pct", "xutum_return_pct"]], on="month", how="left")
    return out.rename(columns={"xu100_return_pct": "xu100", "xutum_return_pct": "xutum"}).sort_values("month").reset_index(drop=True)


def _console_pending(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        [
            "strategy_name",
            "pending_queued_count",
            "pending_bought_count",
            "pending_expired_count",
            "pending_revalidate_failed_count",
            "total_return_pct",
            "max_drawdown_pct",
            "avg_cash_ratio_pct",
        ]
    ].copy()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = run_simulation(args)
    results["daily"].to_csv(out_root / "rolling_portfolio_daily_2026.csv", index=False)
    results["trades"].to_csv(out_root / "rolling_portfolio_trades_2026.csv", index=False)
    results["pending"].to_csv(out_root / "rolling_portfolio_pending_events_2026.csv", index=False)
    results["monthly"].to_csv(out_root / "rolling_portfolio_monthly_2026.csv", index=False)
    results["summary"].to_csv(out_root / "rolling_portfolio_summary_2026.csv", index=False)
    results["benchmark_monthly"].to_csv(out_root / "rolling_portfolio_benchmark_monthly_2026.csv", index=False)

    print("GENERAL_PERFORMANCE")
    print(_console_general(results["summary"]).to_string(index=False))
    print("MONTHLY_PERFORMANCE")
    print(_console_monthly(results["monthly"], results["benchmark_monthly"]).to_string(index=False))
    print("PENDING_IMPACT")
    print(_console_pending(results["summary"]).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_radar_rolling_hold20_portfolio.py"
    spec = importlib.util.spec_from_file_location("analyze_radar_rolling_hold20_portfolio", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(dates: list[str], stale_dates: set[str] | None = None) -> pd.DataFrame:
    stale_dates = stale_dates or set()
    rows = []
    for d in dates:
        rows.append(
            {
                "as_of_date": d,
                "is_expected_trading_day": True,
                "stale_data_warning": d in stale_dates,
                "stale_data_reason": "stale" if d in stale_dates else "",
            }
        )
    return pd.DataFrame(rows)


def _row(
    as_of_date: str,
    symbol: str,
    *,
    production_rank: int,
    special_score: float = 80.0,
    passes_mid: bool = True,
    passes_strict: bool = True,
    liquidity_safe_score: float = 75.0,
    balanced_score: float = 65.0,
    momentum_quality_score: float = 63.0,
    turnover: float = 60_000_000.0,
    volume: float = 1_000_000.0,
    avg_turnover_20d: float = 70_000_000.0,
    rsi_14: float = 60.0,
) -> dict[str, object]:
    return {
        "as_of_date": as_of_date,
        "symbol": symbol,
        "production_rank": production_rank,
        "special_tier": "strict" if passes_strict else ("mid" if passes_mid else ""),
        "special_score": special_score,
        "liquidity_safe_score": liquidity_safe_score,
        "balanced_score": balanced_score,
        "momentum_quality_score": momentum_quality_score,
        "passes_special_mid": passes_mid,
        "passes_special_strict": passes_strict,
        "turnover": turnover,
        "volume": volume,
        "avg_turnover_20d": avg_turnover_20d,
        "rsi_14": rsi_14,
    }


def _price_frame(points: list[tuple[str, float, float | None]]) -> pd.DataFrame:
    rows = []
    for date_str, close, open_price in points:
        rows.append(
            {
                "date": pd.Timestamp(date_str),
                "date_str": date_str,
                "open": float(close if open_price is None else open_price),
                "close": float(close),
                "low": float(close),
            }
        )
    return pd.DataFrame(rows)


def test_same_symbol_not_rebought_same_day_and_next_day_reentry_allowed() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame([_row(d, "AAA", production_rank=1) for d in dates])
    price_cache = {"AAA": _price_frame([(d, 10.0 + i, None) for i, d in enumerate(dates)])}

    daily_df, trades_df, pending_df = m.simulate_strategy(
        "top30_fresh_only",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=2,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )

    closed = trades_df.loc[trades_df["exit_trade_date"].notna()].sort_values("entry_trade_date").reset_index(drop=True)
    assert closed["entry_trade_date"].tolist() == ["2026-01-01", "2026-01-03"]
    assert closed["exit_trade_date"].tolist() == ["2026-01-02", "2026-01-04"]
    assert "skipped_duplicate" in pending_df.loc[pending_df["as_of_date"] == "2026-01-02", "event_type"].tolist()
    assert int(daily_df.loc[daily_df["as_of_date"] == "2026-01-02", "new_entries_count"].iloc[0]) == 0


def test_max_open_positions_and_queue_when_no_slot() -> None:
    m = _load_module()
    summary_df = _summary(["2026-01-01"])
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, special_score=90.0),
            _row("2026-01-01", "BBB", production_rank=2, special_score=80.0),
        ]
    )
    price_cache = {
        "AAA": _price_frame([("2026-01-01", 10.0, None)]),
        "BBB": _price_frame([("2026-01-01", 10.0, None)]),
    }

    daily_df, _trades_df, pending_df = m.simulate_strategy(
        "special_strict_pending_ttl3_raw",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=5,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )

    assert int(daily_df.iloc[0]["open_position_count"]) == 1
    assert int(daily_df.iloc[0]["pending_queue_size"]) == 1
    assert pending_df.loc[pending_df["event_type"] == "queued", "symbol"].tolist() == ["BBB"]


def test_fresh_signal_beats_pending_and_pending_ttl3_expires() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, special_score=95.0),
            _row("2026-01-01", "BBB", production_rank=2, special_score=80.0),
            _row("2026-01-02", "AAA", production_rank=1, special_score=95.0),
            _row("2026-01-02", "CCC", production_rank=2, special_score=70.0),
            _row("2026-01-03", "CCC", production_rank=1, special_score=70.0),
        ]
    )
    price_cache = {
        "AAA": _price_frame([(d, 10.0, None) for d in dates]),
        "BBB": _price_frame([(d, 10.0, None) for d in dates]),
        "CCC": _price_frame([(d, 10.0, None) for d in dates]),
    }

    _daily_df, trades_df, pending_df = m.simulate_strategy(
        "special_strict_pending_ttl3_raw",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
            max_holdings=1,
            holding_days=3,
            entry_mode="same_close",
            exit_mode="hold20_close",
            exclude_stale_new_entries=True,
            allow_same_day_reentry=False,
            min_position_value=10.0,
    )

    closed = trades_df.loc[trades_df["exit_trade_date"].notna()].sort_values("entry_trade_date").reset_index(drop=True)
    assert closed["symbol"].tolist()[:2] == ["AAA", "CCC"]
    expired = pending_df.loc[pending_df["event_type"] == "expired", "symbol"].tolist()
    assert "BBB" in expired


def test_pending_queue_order_uses_latest_signal_then_score_then_liquidity() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, special_score=99.0),
            _row("2026-01-01", "BBB", production_rank=2, special_score=80.0, liquidity_safe_score=70.0),
            _row("2026-01-01", "CCC", production_rank=3, special_score=85.0, liquidity_safe_score=72.0),
            _row("2026-01-02", "AAA", production_rank=1, special_score=99.0),
        ]
    )
    price_cache = {
        "AAA": _price_frame([(d, 10.0, None) for d in dates]),
        "BBB": _price_frame([(d, 10.0, None) for d in dates]),
        "CCC": _price_frame([(d, 10.0, None) for d in dates]),
    }

    _daily_df, trades_df, _pending_df = m.simulate_strategy(
        "special_strict_pending_ttl3_raw",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=2,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )

    closed = trades_df.loc[trades_df["exit_trade_date"].notna()].sort_values("entry_trade_date").reset_index(drop=True)
    assert closed["symbol"].tolist()[:2] == ["AAA", "CCC"]


def test_revalidate_failure_blocks_pending_buy() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, special_score=90.0),
            _row("2026-01-01", "BBB", production_rank=2, special_score=80.0),
            _row("2026-01-02", "AAA", production_rank=1, special_score=90.0),
            _row("2026-01-02", "BBB", production_rank=2, special_score=80.0, liquidity_safe_score=60.0, passes_strict=False),
        ]
    )
    price_cache = {
        "AAA": _price_frame([(d, 10.0, None) for d in dates]),
        "BBB": _price_frame([(d, 10.0, None) for d in dates]),
    }

    _daily_df, trades_df, pending_df = m.simulate_strategy(
        "special_strict_pending_ttl3_revalidate",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=2,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )

    assert "BBB" not in trades_df["symbol"].tolist()
    assert "revalidate_failed" in pending_df["event_type"].tolist()


def test_stale_day_blocks_new_entries_but_allows_exit() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    summary_df = _summary(dates, {"2026-01-02"})
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1),
            _row("2026-01-02", "BBB", production_rank=1),
            _row("2026-01-03", "CCC", production_rank=1),
        ]
    )
    price_cache = {
        "AAA": _price_frame([(d, 10.0, None) for d in dates]),
        "BBB": _price_frame([(d, 10.0, None) for d in dates]),
        "CCC": _price_frame([(d, 10.0, None) for d in dates]),
    }

    daily_df, trades_df, _pending_df = m.simulate_strategy(
        "top30_fresh_only",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=2,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )

    assert int(daily_df.loc[daily_df["as_of_date"] == "2026-01-02", "exits_count"].iloc[0]) == 1
    assert int(daily_df.loc[daily_df["as_of_date"] == "2026-01-02", "new_entries_count"].iloc[0]) == 0
    assert trades_df.loc[trades_df["exit_trade_date"].notna(), "symbol"].tolist() == ["AAA"]


def test_monthly_return_and_benchmark_return() -> None:
    m = _load_module()
    summary_df = _summary(["2026-01-02", "2026-01-30"])
    daily_df = pd.DataFrame(
        [
            {"as_of_date": "2026-01-02", "strategy_name": "s", "cash": 1000.0, "open_position_value": 0.0, "open_cost_basis": 0.0, "total_equity": 1000.0, "open_position_count": 0},
            {"as_of_date": "2026-01-30", "strategy_name": "s", "cash": 1100.0, "open_position_value": 0.0, "open_cost_basis": 0.0, "total_equity": 1100.0, "open_position_count": 0},
        ]
    )
    trades_df = pd.DataFrame(
        [
            {"entry_trade_date": "2026-01-02", "exit_trade_date": "2026-01-30", "pnl": 100.0, "return_pct": 10.0},
        ]
    )
    price_cache = {
        "XU100": _price_frame([("2026-01-02", 100.0, None), ("2026-01-30", 110.0, None)]),
        "XUTUM": _price_frame([("2026-01-02", 200.0, None), ("2026-01-30", 210.0, None)]),
    }

    bench = m._monthly_benchmark(summary_df, price_cache)
    monthly = m._monthly_strategy_metrics("s", daily_df, trades_df, bench)

    assert round(float(monthly.iloc[0]["month_return_pct"]), 6) == 10.0
    assert round(float(monthly.iloc[0]["xu100_return_pct"]), 6) == 10.0
    assert round(float(monthly.iloc[0]["xutum_return_pct"]), 6) == 5.0


def test_top30_logic_unchanged() -> None:
    m = _load_module()
    day_df = pd.DataFrame([_row("2026-01-01", f"S{i:02d}", production_rank=i + 1, passes_mid=False, passes_strict=False) for i in range(35)])
    out = m._select_group(day_df, "top30", 10)
    assert out["symbol"].tolist() == [f"S{i:02d}" for i in range(30)]

from __future__ import annotations

import importlib.util
import argparse
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
    passes_loose: bool = True,
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
        "passes_special_loose": passes_loose,
        "passes_special_strict": passes_strict,
        "turnover": turnover,
        "volume": volume,
        "avg_turnover_20d": avg_turnover_20d,
        "rsi_14": rsi_14,
        "passes_tv_volume_momentum_trend": True,
        "tv_momentum_score": 80.0,
        "volume_ratio_20d": 2.0,
        "turnover_ratio_20d": 2.0,
        "avg_turnover_30d": 70_000_000.0,
        "turnover_today": 30_000_000.0,
        "macd_hist": 1.0,
        "adx_14": 25.0,
        "adr_pct": 5.0,
        "ema8": 11.0,
        "ema21": 10.0,
        "ema20": 10.0,
        "ema50": 9.0,
        "ema60": 8.0,
        "perf_3m_pct": 5.0,
        "perf_6m_pct": 10.0,
        "daily_change_pct": 0.5,
        "price_above_52w_low_pct": 20.0,
        "close_gt_ema20": True,
        "ema20_gt_ema50": True,
        "ema8_gte_ema21": True,
        "close_gt_ema60": True,
        "quality_threshold_score": 55.0,
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

    daily_df, trades_df, _pending_df, _signals_df = m.simulate_strategy(
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

    daily_df, _trades_df, pending_df, _signals_df = m.simulate_strategy(
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

    _daily_df, trades_df, pending_df, _signals_df = m.simulate_strategy(
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

    _daily_df, trades_df, _pending_df, _signals_df = m.simulate_strategy(
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

    _daily_df, trades_df, pending_df, _signals_df = m.simulate_strategy(
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

    daily_df, trades_df, _pending_df, _signals_df = m.simulate_strategy(
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


def test_live_pilot_kap_no_news_fallback_sorts_by_quality_score() -> None:
    m = _load_module()
    daily_all = pd.DataFrame(
        [
            {**_row("2026-01-02", "BBB", production_rank=1, passes_strict=True), "quality_threshold_score": 40.0},
            {**_row("2026-01-02", "AAA", production_rank=2, passes_strict=True), "quality_threshold_score": 70.0},
        ]
    )
    regime = {"2026-01-02": {"weak_score": 1, "adaptive_selected_mode": "special_strict"}}

    out = m._build_live_pilot_daily_radar(daily_all, regime, as_of_date="2026-01-02", events_by_symbol={})

    assert out["symbol"].tolist() == ["AAA", "BBB"]
    assert out["kap_summary_short"].tolist() == ["recent_kap_none", "recent_kap_none"]
    assert out["kap_sentiment_label"].tolist() == ["unknown", "unknown"]


def test_live_pilot_kap_multiple_events_are_summarized() -> None:
    m = _load_module()
    daily_all = pd.DataFrame([_row("2026-01-08", "AAA", production_rank=1, passes_strict=True)])
    regime = {"2026-01-08": {"weak_score": 2, "adaptive_selected_mode": "threshold_50"}}
    events_by_symbol = {
        "AAA": [
            {"date": "2026-01-07", "event_type": "Ozel Durum", "title": "Yeni sozlesme imzalandi"},
            {"date": "2026-01-05", "event_type": "Temettu", "title": "Temettu karari"},
            {"date": "2025-12-01", "event_type": "Eski", "title": "Eski haber"},
        ]
    }

    out = m._build_live_pilot_daily_radar(
        daily_all,
        regime,
        as_of_date="2026-01-08",
        kap_lookback_days=7,
        events_by_symbol=events_by_symbol,
    )

    row = out.iloc[0]
    assert int(row["kap_event_count_7d"]) == 2
    assert row["kap_latest_date"] == "2026-01-07"
    assert "Ozel Durum" in row["kap_event_types"]
    assert "Temettu" in row["kap_event_types"]
    assert row["kap_sentiment_label"] == "positive"
    assert row["manual_review_note"] == "manual_kap_review_required"


def test_live_pilot_kap_breaker_events_are_filtered_out() -> None:
    m = _load_module()
    daily_all = pd.DataFrame([_row("2026-01-08", "AAA", production_rank=1, passes_strict=True)])
    regime = {"2026-01-08": {"weak_score": 2, "adaptive_selected_mode": "threshold_50"}}
    events_by_symbol = {
        "AAA": [
            {
                "date": "2026-01-07",
                "event_type": "KAP",
                "title": "BORSA ISTANBUL BISTECH DEVRE KESICI UYGULAMASI",
                "summary": "Pay bazinda devre kesici bildirimi",
            }
        ]
    }

    out = m._build_live_pilot_daily_radar(
        daily_all,
        regime,
        as_of_date="2026-01-08",
        kap_lookback_days=7,
        events_by_symbol=events_by_symbol,
    )

    row = out.iloc[0]
    assert int(row["kap_event_count_7d"]) == 0
    assert row["kap_summary_short"] == "recent_kap_none"
    assert row["kap_sentiment_label"] == "unknown"


def test_live_pilot_kap_layer_does_not_change_actions_or_quality_order() -> None:
    m = _load_module()
    daily_all = pd.DataFrame(
        [
            {**_row("2026-01-08", "CCC", production_rank=1, passes_strict=True), "quality_threshold_score": 35.0},
            {**_row("2026-01-08", "AAA", production_rank=2, passes_strict=True), "quality_threshold_score": 65.0},
        ]
    )
    regime = {"2026-01-08": {"weak_score": 3, "adaptive_selected_mode": "threshold_60"}}

    base = m._build_live_pilot_daily_radar(daily_all, regime, as_of_date="2026-01-08", events_by_symbol={})
    with_kap = m._build_live_pilot_daily_radar(
        daily_all,
        regime,
        as_of_date="2026-01-08",
        events_by_symbol={"AAA": [{"date": "2026-01-07", "event_type": "Ceza", "title": "Idari para cezasi"}]},
    )

    assert with_kap["symbol"].tolist() == ["AAA", "CCC"]
    assert with_kap["action"].tolist() == base["action"].tolist()
    assert with_kap.loc[with_kap["symbol"] == "AAA", "kap_sentiment_label"].iloc[0] == "caution"


def test_tv_filter_and_diagnostics_not_hard_filtered() -> None:
    m = _load_module()
    day_df = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, special_score=80.0),
            _row("2026-01-01", "BBB", production_rank=2, special_score=79.0, passes_strict=False),
        ]
    )
    day_df.loc[:, "daily_change_pct"] = [0.1, -1.0]
    day_df.loc[:, "price_above_52w_low_pct"] = [10.0, 20.0]
    day_df.loc[:, "daily_change_gt_2"] = day_df["daily_change_pct"] > 2.0
    day_df.loc[:, "price_above_52w_low_gte_70"] = day_df["price_above_52w_low_pct"] >= 70.0
    out = m._select_group(day_df, "tv_volume_momentum_trend", 10)
    assert set(out["symbol"].tolist()) == {"AAA", "BBB"}


def test_new_repeated_dropped_and_new_symbol_file_logic() -> None:
    m = _load_module()
    summary_df = _summary(["2026-01-01", "2026-01-02"])
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, passes_strict=True),
            _row("2026-01-01", "BBB", production_rank=2, passes_strict=True),
            _row("2026-01-02", "BBB", production_rank=1, passes_strict=True),
            _row("2026-01-02", "CCC", production_rank=2, passes_strict=True),
        ]
    )
    summary, new_symbols, _agg = m._daily_new_signal_tables(daily_all, summary_df, 10)
    row = summary.loc[(summary["as_of_date"] == "2026-01-02") & (summary["filter_name"] == "special_strict")].iloc[0]
    assert int(row["repeated_vs_prev_trading_day_count"]) == 1
    assert int(row["new_vs_prev_trading_day_count"]) == 1
    assert int(row["dropped_vs_prev_trading_day_count"]) == 1
    ns = new_symbols.loc[(new_symbols["as_of_date"] == "2026-01-02") & (new_symbols["filter_name"] == "special_strict")]
    assert ns["symbol"].tolist() == ["CCC"]


def test_forward20_and_bought_vs_skipped_flags() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, passes_strict=True, special_score=90.0),
            _row("2026-01-01", "BBB", production_rank=2, passes_strict=True, special_score=80.0),
            _row("2026-01-02", "AAA", production_rank=1, passes_strict=True, special_score=90.0),
        ]
    )
    price_cache = {
        "AAA": _price_frame([("2026-01-01", 10.0, None), ("2026-01-02", 11.0, None), ("2026-01-03", 12.0, None)]),
        "BBB": _price_frame([("2026-01-01", 10.0, None), ("2026-01-02", 9.0, None), ("2026-01-03", 8.0, None)]),
    }
    _d, _t, _p, sig = m.simulate_strategy(
        "special_strict_fresh_only",
        daily_all,
        summary_df,
        price_cache,
        initial_capital=1000.0,
        max_holdings=1,
        holding_days=20,
        entry_mode="same_close",
        exit_mode="hold20_close",
        exclude_stale_new_entries=True,
        allow_same_day_reentry=False,
        min_position_value=10.0,
    )
    fwd, _sum = m._forward20_signals(daily_all, summary_df, price_cache, sig, 10)
    first = fwd.loc[(fwd["signal_date"] == "2026-01-01") & (fwd["filter_name"] == "special_strict")].sort_values("symbol").reset_index(drop=True)
    assert bool(first.iloc[0]["was_bought_by_portfolio"]) is True
    assert bool(first.iloc[1]["was_skipped_due_to_full_slots"]) is True


def test_tv_strategy_name_included() -> None:
    m = _load_module()
    assert "tv_volume_momentum_trend_fresh_only" in m.STRATEGY_NAMES
    assert "special_strict_quality_v2_score_fresh_only" in m.STRATEGY_NAMES
    assert "special_strict_score_threshold_fresh_only" in m.STRATEGY_NAMES


def test_default_trade_simulation_has_zero_commission() -> None:
    m = _load_module()
    dates = ["2026-01-01", "2026-01-02"]
    summary_df = _summary(dates)
    daily_all = pd.DataFrame([_row("2026-01-01", "AAA", production_rank=1)])
    price_cache = {"AAA": _price_frame([("2026-01-01", 10.0, None), ("2026-01-02", 12.0, None)])}

    _daily_df, trades_df, _pending_df, _signals_df = m.simulate_strategy(
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
        min_position_value=1.0,
    )

    closed = trades_df.loc[trades_df["exit_reason"] == "hold20_close"].iloc[0]
    assert round(float(closed["entry_value"]), 6) == 1000.0
    assert round(float(closed["return_pct"]), 6) == 20.0


def test_shared_quality_score_does_not_mutate_special_strict_flag() -> None:
    m = _load_module()
    df = pd.DataFrame(
        [
            {**_row("2026-01-01", "AAA", production_rank=1, passes_strict=True), "relative_strength_20d_pct": 12.0},
            {**_row("2026-01-01", "BBB", production_rank=2, passes_strict=False), "relative_strength_20d_pct": 12.0},
        ]
    )
    before = df["passes_special_strict"].tolist()

    out = m.compute_quality_threshold_score(df.copy())

    assert out["passes_special_strict"].tolist() == before
    assert out["quality_threshold_score"].between(0, 100).all()


def test_quality_v2_score_sorting_and_nonhard_diagnostics() -> None:
    m = _load_module()
    day_df = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, passes_strict=True, special_score=90.0),
            _row("2026-01-01", "BBB", production_rank=2, passes_strict=True, special_score=85.0),
        ]
    )
    day_df.loc[:, "passes_special_strict_quality_v2"] = True
    day_df.loc[:, "quality_v2_score"] = [70.0, 90.0]
    day_df.loc[:, "liquidity_safe_score"] = [95.0, 80.0]
    day_df.loc[:, "daily_change_gt_2"] = [False, False]
    day_df.loc[:, "distance_from_52w_low_pct"] = [10.0, 20.0]
    out_liq = m._select_group(day_df, "special_strict_quality_v2", 10)
    out_score = m._select_group(day_df, "special_strict_quality_v2_score", 10)
    assert out_liq["symbol"].tolist()[0] == "AAA"
    assert out_score["symbol"].tolist()[0] == "BBB"


def test_quality_threshold_score_range_and_threshold_selection() -> None:
    m = _load_module()
    day_df = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, passes_strict=True),
            _row("2026-01-01", "BBB", production_rank=2, passes_strict=True),
            _row("2026-01-01", "CCC", production_rank=3, passes_strict=False),
        ]
    )
    day_df.loc[:, "quality_threshold_score"] = [49.9, 50.0, 90.0]
    day_df.loc[:, "passes_special_strict"] = [True, True, False]
    day_df.attrs["quality_score_threshold"] = 50.0
    out = m._select_group(day_df, "special_strict_score_threshold", 10)
    assert out["symbol"].tolist() == ["BBB"]
    assert out["quality_threshold_score"].between(0, 100).all()


def test_threshold_filter_not_top10_limited() -> None:
    m = _load_module()
    day_df = pd.DataFrame([_row("2026-01-01", f"S{i:02d}", production_rank=i + 1, passes_strict=True) for i in range(15)])
    day_df.loc[:, "quality_threshold_score"] = [70.0] * 15
    day_df.loc[:, "passes_special_strict"] = [True] * 15
    day_df.attrs["quality_score_threshold"] = 50.0
    out = m._select_group(day_df, "special_strict_score_threshold", 10)
    assert len(out) == 15


def test_parse_thresholds_cli_list() -> None:
    m = _load_module()
    args = argparse.Namespace(quality_score_threshold=50.0, quality_score_thresholds="40,50,60,70")
    assert m._parse_thresholds(args) == [40, 50, 60, 70]


def test_market_breadth_computation() -> None:
    m = _load_module()
    df = pd.DataFrame(
        [
            {"symbol": "A", "close": 11.0, "ma20": 10.0},
            {"symbol": "B", "close": 9.0, "ma20": 10.0},
            {"symbol": "C", "close": 12.0, "ma20": 10.0},
            {"symbol": "D", "close": None, "ma20": 10.0},
        ]
    )
    out = m._compute_market_breadth(df)
    assert round(float(out), 4) == round((2 / 3) * 100.0, 4)


def test_weak_score_computation() -> None:
    m = _load_module()
    weak = m._compute_weak_score(
        xu100_close=90.0,
        xu100_ma50=100.0,
        xu100_return_20d_pct=-5.0,
        xu100_ma50_slope_10d=-1.0,
        market_breadth_pct_above_ma20=40.0,
    )
    assert weak == 4


def test_adaptive_threshold_switching_logic() -> None:
    m = _load_module()
    day_df = pd.DataFrame(
        [
            _row("2026-01-01", "AAA", production_rank=1, passes_strict=True, special_score=90.0),
            _row("2026-01-01", "BBB", production_rank=2, passes_strict=True, special_score=80.0),
        ]
    )
    day_df.loc[:, "quality_threshold_score"] = [55.0, 65.0]

    day_df.attrs["weak_score"] = 1
    out_weak1 = m._select_group(day_df, "adaptive_regime_v1", 10)
    assert set(out_weak1["symbol"].tolist()) == {"AAA", "BBB"}
    assert out_weak1["adaptive_selected_mode"].iloc[0] == "special_strict"

    day_df.attrs["weak_score"] = 2
    out_weak2 = m._select_group(day_df, "adaptive_regime_v1", 10)
    assert set(out_weak2["symbol"].tolist()) == {"AAA", "BBB"}
    assert out_weak2["adaptive_selected_mode"].iloc[0] == "threshold_50"

    day_df.attrs["weak_score"] = 3
    out_weak3 = m._select_group(day_df, "adaptive_regime_v1", 10)
    assert out_weak3["symbol"].tolist() == ["BBB"]
    assert out_weak3["adaptive_selected_mode"].iloc[0] == "threshold_60"

    day_df.attrs["weak_score"] = 4
    out_weak4 = m._select_group(day_df, "adaptive_regime_v1", 10)
    assert out_weak4.empty


def test_adaptive_strategy_name_included() -> None:
    m = _load_module()
    assert "adaptive_regime_v1_fresh_only" in m.STRATEGY_NAMES


def test_prepare_inputs_builds_data_missing_coverage_summary(monkeypatch) -> None:
    m = _load_module()
    features = pd.DataFrame(
        [
            {"symbol": "AAA", "strategy": "volume_spike_strict", "signal_date": "2024-01-02"},
            {"symbol": "AAA", "strategy": "volume_spike_strict", "signal_date": "2024-01-03"},
        ]
    )
    daily_all = pd.DataFrame([_row("2024-01-02", "AAA", production_rank=1)])
    summary_df = pd.DataFrame(
        [
            {"as_of_date": "2024-01-02", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
        ]
    )
    monkeypatch.setattr(m.APRIL, "_prepare_base_features", lambda _p: features)
    monkeypatch.setattr(m.APRIL, "build_april_replay", lambda *args, **kwargs: (daily_all, summary_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(m, "_load_price_cache_full", lambda *_args, **_kwargs: {"AAA": _price_frame([("2024-01-02", 10.0, None)])})
    monkeypatch.setattr(m, "_build_indicator_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(m, "_enrich_daily_all", lambda frame, *_args, **_kwargs: frame)
    _da, _su, _pc, _end, cov = m._prepare_inputs(
        "dummy",
        None,
        "dummy.sqlite",
        "2020-01-01",
        "2024-01-31",
        True,
    )
    assert cov["available_min_signal_date"] == "2024-01-02"
    assert cov["actual_start_date"] == "2024-01-02"
    assert 2020 in cov["missing_years"]


def test_candidate_features_path_directory_is_loaded(tmp_path) -> None:
    m = _load_module()
    year_dir = tmp_path / "year=2020"
    year_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "period": "2020-01-01",
                "symbol": "AAA",
                "strategy": "volume_spike_strict",
                "signal_date": "2020-01-02",
                "entry_date": "2020-01-03",
                "close": 10.0,
                "volume": 1000,
                "turnover": 10_000_000.0,
                "avg_turnover_20d": 10_000_000.0,
                "volume_ratio_20d": 2.0,
                "turnover_ratio_20d": 2.0,
                "ma20": 9.0,
                "above_ma20": True,
                "rsi_14": 55.0,
                "return_5d_pct": 5.0,
                "return_10d_pct": 8.0,
                "close_position": 0.7,
            }
        ]
    ).to_csv(year_dir / "candidate_features.csv", index=False)
    out = m._prepare_features("unused", str(tmp_path))
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "AAA"
    assert str(out.iloc[0]["signal_date"].date()) == "2020-01-02"

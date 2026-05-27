from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_radar_filter_monthly_replay.py"
    spec = importlib.util.spec_from_file_location("analyze_radar_filter_monthly_replay", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_row(
    symbol: str,
    signal_date: str,
    *,
    close: float,
    turnover: float,
    avg_turnover_20d: float,
    volume_ratio_20d: float,
    turnover_ratio_20d: float,
    rsi_14: float,
    return_5d_pct: float,
    return_10d_pct: float,
    close_position: float,
    ma20: float = 10.0,
    above_ma20: bool = True,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": "volume_spike_strict",
        "signal_date": signal_date,
        "close": close,
        "volume": 1_000_000.0,
        "turnover": turnover,
        "avg_turnover_20d": avg_turnover_20d,
        "volume_ratio_20d": volume_ratio_20d,
        "turnover_ratio_20d": turnover_ratio_20d,
        "ma20": ma20,
        "above_ma20": above_ma20,
        "rsi_14": rsi_14,
        "return_5d_pct": return_5d_pct,
        "return_10d_pct": return_10d_pct,
        "close_position": close_position,
    }


def _write_price_cache(db_path: Path, symbol: str, points: list[tuple[str, float]]) -> None:
    payload = json.dumps({"records": [{"date": f"{d}T09:30:00+03:00", "close": c, "low": c} for d, c in points]})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_ohlcv_cache (symbol TEXT PRIMARY KEY, fetched_at TEXT, payload_json TEXT, source TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv_cache (symbol, fetched_at, payload_json, source) VALUES (?, datetime('now'), ?, 'test')",
            (symbol, payload),
        )


def test_daily_overlap_new_and_dropped() -> None:
    m = _load_module()
    df1 = pd.DataFrame(
        [
            {"symbol": "AAA", "production_rank": 1, "special_score": 90.0, "passes_special_strict": True},
            {"symbol": "BBB", "production_rank": 2, "special_score": 80.0, "passes_special_strict": True},
        ]
    )
    df2 = pd.DataFrame(
        [
            {"symbol": "BBB", "production_rank": 1, "special_score": 88.0, "passes_special_strict": True},
            {"symbol": "CCC", "production_rank": 2, "special_score": 85.0, "passes_special_strict": True},
        ]
    )
    group_map = {
        ("2026-01-02", "special_strict"): df1.assign(as_of_date="2026-01-02", group_rank=[1, 2]),
        ("2026-01-03", "special_strict"): df2.assign(as_of_date="2026-01-03", group_rank=[1, 2]),
    }
    for g in ["top30", "special_loose", "special_mid", "special_strict_top10"]:
        group_map[("2026-01-02", g)] = pd.DataFrame(columns=df1.columns.tolist() + ["as_of_date", "group_rank"])
        group_map[("2026-01-03", g)] = pd.DataFrame(columns=df1.columns.tolist() + ["as_of_date", "group_rank"])
    summary_df = pd.DataFrame(
        [
            {"as_of_date": "2026-01-02", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
            {"as_of_date": "2026-01-03", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
        ]
    )
    persistence, _life = m.build_daily_persistence_summary(pd.DataFrame(), summary_df, group_map)
    row = persistence.loc[(persistence["as_of_date"] == "2026-01-03") & (persistence["group_name"] == "special_strict")].iloc[0]
    assert int(row["repeated_vs_prev_trading_day"]) == 1
    assert int(row["new_vs_prev_trading_day"]) == 1
    assert int(row["dropped_vs_prev_trading_day"]) == 1


def test_special_strict_top10_is_subset_and_sorted() -> None:
    m = _load_module()
    day_df = pd.DataFrame(
        [
            {"symbol": "AAA", "passes_special_strict": True, "special_score": 80.0, "production_rank": 5},
            {"symbol": "BBB", "passes_special_strict": True, "special_score": 90.0, "production_rank": 2},
            {"symbol": "CCC", "passes_special_strict": False, "special_score": 95.0, "production_rank": 1},
            {"symbol": "DDD", "passes_special_strict": True, "special_score": 85.0, "production_rank": 3},
        ]
    )
    out = m._select_group(day_df, "special_strict_top10", 2)
    assert out["symbol"].tolist() == ["BBB", "DDD"]
    assert out["group_rank"].tolist() == [1, 2]


def test_month_boundaries_pick_first_and_last_expected_trading_day() -> None:
    m = _load_module()
    summary_df = pd.DataFrame(
        [
            {"as_of_date": "2026-01-01", "is_expected_trading_day": False, "stale_data_warning": False, "stale_data_reason": ""},
            {"as_of_date": "2026-01-02", "is_expected_trading_day": True, "stale_data_warning": True, "stale_data_reason": "x"},
            {"as_of_date": "2026-01-30", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
            {"as_of_date": "2026-02-02", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
            {"as_of_date": "2026-02-27", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
        ]
    )
    out = m._monthly_boundaries(summary_df)
    assert out.loc[out["month"] == "2026-01", "entry_date"].iloc[0] == "2026-01-02"
    assert out.loc[out["month"] == "2026-01", "exit_date"].iloc[0] == "2026-01-30"
    assert bool(out.loc[out["month"] == "2026-01", "stale_entry_warning"].iloc[0]) is True


def test_monthly_equal_weight_return_and_missing_price_exclusion(tmp_path: Path) -> None:
    m = _load_module()
    db_path = tmp_path / "prices.sqlite"
    _write_price_cache(db_path, "AAA", [("2026-01-02", 10.0), ("2026-01-30", 12.0)])
    _write_price_cache(db_path, "BBB", [("2026-01-02", 20.0), ("2026-01-30", 18.0)])
    summary_df = pd.DataFrame(
        [{"as_of_date": "2026-01-02", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""},
         {"as_of_date": "2026-01-30", "is_expected_trading_day": True, "stale_data_warning": False, "stale_data_reason": ""}]
    )
    grp = pd.DataFrame(
        [
            {"symbol": "AAA", "production_rank": 1, "special_tier": "strict", "special_score": 80.0, "liquidity_safe_score": 75.0, "balanced_score": 65.0, "momentum_quality_score": 63.0, "group_rank": 1},
            {"symbol": "BBB", "production_rank": 2, "special_tier": "strict", "special_score": 70.0, "liquidity_safe_score": 72.0, "balanced_score": 61.0, "momentum_quality_score": 60.0, "group_rank": 2},
            {"symbol": "CCC", "production_rank": 3, "special_tier": "", "special_score": 60.0, "liquidity_safe_score": 68.0, "balanced_score": 58.0, "momentum_quality_score": 55.0, "group_rank": 3},
        ]
    )
    empty = pd.DataFrame(columns=grp.columns)
    group_map = {}
    for key in ["top30", "special_loose", "special_mid", "special_strict", "special_strict_top10"]:
        group_map[("2026-01-02", key)] = grp.copy() if key == "top30" else (grp.iloc[:2].copy() if key == "special_strict" else empty.copy())
        group_map[("2026-01-30", key)] = empty.copy()
    group_map[("2026-01-02", "special_strict_top10")] = grp.iloc[:2].copy()
    monthly_df, positions_df = m.build_monthly_replay(summary_df, group_map, str(db_path), 10)
    top30 = monthly_df.loc[monthly_df["group_name"] == "top30_equal_weight"].iloc[0]
    assert round(float(top30["portfolio_return_pct"]), 6) == 5.0
    assert int(top30["excluded_symbol_count"]) == 1
    assert "excluded_missing_price" in positions_df.loc[positions_df["symbol"] == "CCC", "exclusion_reason"].iloc[0]


def test_cumulative_return_and_top30_unchanged() -> None:
    m = _load_module()
    monthly_df = pd.DataFrame(
        [
            {"group_name": "top30_equal_weight", "month": "2026-01", "portfolio_return_pct": 10.0, "selected_symbol_count": 10, "included_symbol_count": 10, "stale_entry_warning": False},
            {"group_name": "top30_equal_weight", "month": "2026-02", "portfolio_return_pct": -10.0, "selected_symbol_count": 10, "included_symbol_count": 10, "stale_entry_warning": False},
        ]
    )
    summary = m.summarize_monthly_performance(monthly_df, exclude_stale=False)
    row = summary.iloc[0]
    assert round(float(row["cumulative_return_pct"]), 6) == -1.0

    day_df = pd.DataFrame(
        [{"symbol": f"S{i:02d}", "production_rank": i + 1, "passes_special_loose": False, "passes_special_mid": False, "passes_special_strict": False, "special_score": 0.0} for i in range(35)]
    )
    out = m._select_group(day_df, "top30", 10)
    assert out["symbol"].tolist() == [f"S{i:02d}" for i in range(30)]

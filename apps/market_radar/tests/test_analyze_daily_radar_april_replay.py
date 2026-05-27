from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_daily_radar_april_replay.py"
    spec = importlib.util.spec_from_file_location("analyze_daily_radar_april_replay", script_path)
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


def _write_price_cache(db_path: Path, symbol: str, closes: list[float], lows: list[float] | None = None) -> None:
    lows = lows or closes
    records = []
    for idx, (close, low) in enumerate(zip(closes, lows, strict=False), start=1):
        records.append({"date": f"2026-04-{idx:02d}T09:30:00+03:00", "close": close, "low": low})
    payload = json.dumps({"records": records})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_ohlcv_cache (symbol TEXT PRIMARY KEY, fetched_at TEXT, payload_json TEXT, source TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv_cache (symbol, fetched_at, payload_json, source) VALUES (?, datetime('now'), ?, 'test')",
            (symbol, payload),
        )


def test_top30_generation_unchanged_and_special_only_marks_top30() -> None:
    m = _load_module()
    rows = []
    for idx in range(32):
        rows.append(
            _make_row(
                f"S{idx:02d}",
                "2026-04-01",
                close=12.0 + idx * 0.01,
                turnover=25_000_000.0 - idx * 100_000.0,
                avg_turnover_20d=35_000_000.0 - idx * 100_000.0,
                volume_ratio_20d=3.0 - idx * 0.03,
                turnover_ratio_20d=2.2 - idx * 0.02,
                rsi_14=60.0,
                return_5d_pct=10.0,
                return_10d_pct=15.0,
                close_position=0.55 if idx == 1 else 0.80,
            )
        )
    rows[0]["turnover"] = 80_000_000.0
    rows[0]["avg_turnover_20d"] = 90_000_000.0
    rows[0]["volume_ratio_20d"] = 4.0
    rows[0]["turnover_ratio_20d"] = 3.0
    rows[0]["return_5d_pct"] = 14.0
    rows[0]["return_10d_pct"] = 20.0

    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    day_df = m.build_daily_radar(features, "2026-04-01")

    assert int((day_df["production_rank"] <= 30).sum()) == 30
    assert bool(day_df.iloc[0]["is_top30"]) is True
    assert day_df.iloc[0]["special_tier"] == "strict"
    assert day_df["special_score"].notna().all()

    low_close_position = day_df.loc[day_df["symbol"] == "S01"].iloc[0]
    assert low_close_position["close_position"] == 0.55
    assert bool(low_close_position["is_top30"]) is True

    non_top30 = day_df.loc[day_df["production_rank"] > 30].iloc[0]
    assert bool(non_top30["passes_special_loose"]) is False
    assert "not_top30" in str(non_top30["special_failed_reasons"])


def test_weekend_and_holiday_do_not_raise_stale_warning() -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-22",
            close=11.0,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        )
    ]
    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    _daily_all, summary_df, _lifecycle_df, _perf_df, _audit_df = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-25",
        db_path="missing.sqlite",
    )

    holiday = summary_df.loc[summary_df["as_of_date"] == "2026-04-23"].iloc[0]
    weekend = summary_df.loc[summary_df["as_of_date"] == "2026-04-25"].iloc[0]
    assert bool(holiday["is_known_market_holiday"]) is True
    assert bool(holiday["is_expected_trading_day"]) is False
    assert bool(holiday["stale_data_warning"]) is False
    assert bool(weekend["is_weekend"]) is False or pd.Timestamp("2026-04-25").weekday() >= 5
    assert bool(weekend["is_expected_trading_day"]) is False
    assert bool(weekend["stale_data_warning"]) is False


def test_expected_trading_day_stale_reasons_and_fingerprint_vs_prev_trading_day() -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-22",
            close=11.0,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        )
    ]
    db_path = Path("stale_test.sqlite")
    if db_path.exists():
        db_path.unlink()
    _write_price_cache(db_path, "AAA", [11.0], [10.8])
    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    _daily_all, summary_df, _lifecycle_df, _perf_df, audit_df = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-24",
        db_path=str(db_path),
    )

    day24 = summary_df.loc[summary_df["as_of_date"] == "2026-04-24"].iloc[0]
    assert bool(day24["is_expected_trading_day"]) is True
    assert day24["max_signal_date"] == "2026-04-22"
    assert bool(day24["same_data_fingerprint_as_prev_trading_day"]) is True
    assert bool(day24["stale_data_warning"]) is True
    reason = str(day24["stale_data_reason"])
    assert "max_signal_date_before_as_of" in reason
    assert "same_fingerprint_as_previous_expected_trading_day" in reason
    assert "max_data_latest_date_before_as_of" in reason
    assert not audit_df.empty
    assert "2026-04-24" in audit_df["as_of_date"].astype(str).tolist()
    db_path.unlink()


def test_signal_date_change_only_keeps_value_fingerprint_same() -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-22",
            close=11.0,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        ),
        _make_row(
            "AAA",
            "2026-04-24",
            close=11.0,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        ),
    ]
    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    daily_all, summary_df, _lifecycle_df, _perf_df, _audit_df = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-24",
        db_path="missing.sqlite",
    )

    day22 = daily_all.loc[daily_all["as_of_date"] == "2026-04-22"].iloc[0]
    day24 = daily_all.loc[daily_all["as_of_date"] == "2026-04-24"].iloc[0]
    assert day22["data_fingerprint"] != day24["data_fingerprint"]
    assert day22["value_fingerprint"] == day24["value_fingerprint"]

    sum24 = summary_df.loc[summary_df["as_of_date"] == "2026-04-24"].iloc[0]
    assert bool(sum24["same_value_fingerprint_as_prev_trading_day"]) is True
    assert bool(sum24["same_data_fingerprint_as_prev_trading_day"]) is False
    assert bool(sum24["same_data_but_signal_date_changed_warning"]) is True
    assert "value_fingerprint_same_but_data_fingerprint_changed" in str(sum24["same_data_but_signal_date_changed_reason"])


def test_value_change_updates_value_fingerprint_and_rank_change_does_not() -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-22",
            close=11.0,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        ),
        _make_row(
            "BBB",
            "2026-04-22",
            close=12.0,
            turnover=58_000_000.0,
            avg_turnover_20d=75_000_000.0,
            volume_ratio_20d=2.9,
            turnover_ratio_20d=2.1,
            rsi_14=61.0,
            return_5d_pct=10.0,
            return_10d_pct=18.0,
            close_position=0.83,
        ),
        _make_row(
            "AAA",
            "2026-04-24",
            close=11.5,
            turnover=55_000_000.0,
            avg_turnover_20d=70_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.0,
            rsi_14=60.0,
            return_5d_pct=11.0,
            return_10d_pct=19.0,
            close_position=0.82,
        ),
        _make_row(
            "BBB",
            "2026-04-24",
            close=12.0,
            turnover=58_000_000.0,
            avg_turnover_20d=75_000_000.0,
            volume_ratio_20d=2.9,
            turnover_ratio_20d=2.1,
            rsi_14=61.0,
            return_5d_pct=10.0,
            return_10d_pct=18.0,
            close_position=0.83,
        ),
    ]
    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    daily_all, summary_df, _lifecycle_df, _perf_df, _audit_df = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-24",
        db_path="missing.sqlite",
    )
    sum24 = summary_df.loc[summary_df["as_of_date"] == "2026-04-24"].iloc[0]
    assert bool(sum24["same_value_fingerprint_as_prev_trading_day"]) is False
    assert int(sum24["changed_value_symbol_count_vs_prev_trading_day"]) >= 1
    assert float(sum24["changed_value_symbol_ratio_vs_prev_trading_day"]) > 0


def test_exclude_stale_performance_drops_expected_trading_day_stale_rows(tmp_path: Path) -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-22",
            close=10.0,
            turnover=60_000_000.0,
            avg_turnover_20d=80_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.1,
            rsi_14=60.0,
            return_5d_pct=10.0,
            return_10d_pct=18.0,
            close_position=0.85,
        )
    ]
    db_path = tmp_path / "radar.sqlite"
    _write_price_cache(db_path, "AAA", [10, 11, 12, 13, 14, 15], [10, 9.8, 11.8, 12.6, 13.8, 14.5])
    features = m._prepare_base_features_from_df(pd.DataFrame(rows))

    _daily_all_1, _summary_df_1, _lifecycle_df_1, perf_include, _audit_1 = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-24",
        db_path=str(db_path),
        exclude_stale_performance=False,
    )
    _daily_all_2, _summary_df_2, _lifecycle_df_2, perf_exclude, _audit_2 = m.build_april_replay(
        features,
        "2026-04-22",
        "2026-04-24",
        db_path=str(db_path),
        exclude_stale_performance=True,
    )

    top30_include = perf_include.loc[perf_include["group_name"] == "top30"].iloc[0]
    top30_exclude = perf_exclude.loc[perf_exclude["group_name"] == "top30"].iloc[0]
    assert int(top30_include["signal_count"]) >= int(top30_exclude["included_signal_count"])
    assert int(top30_exclude["excluded_signal_count"]) > 0
    assert int(top30_exclude["excluded_stale_days_count"]) > 0


def test_forward_performance_and_lifecycle(tmp_path: Path) -> None:
    m = _load_module()
    rows = [
        _make_row(
            "AAA",
            "2026-04-01",
            close=10.0,
            turnover=60_000_000.0,
            avg_turnover_20d=80_000_000.0,
            volume_ratio_20d=2.8,
            turnover_ratio_20d=2.1,
            rsi_14=60.0,
            return_5d_pct=10.0,
            return_10d_pct=18.0,
            close_position=0.85,
        ),
        _make_row(
            "BBB",
            "2026-04-01",
            close=9.0,
            turnover=32_000_000.0,
            avg_turnover_20d=45_000_000.0,
            volume_ratio_20d=2.0,
            turnover_ratio_20d=1.7,
            rsi_14=62.0,
            return_5d_pct=11.0,
            return_10d_pct=17.0,
            close_position=0.82,
        ),
    ]
    db_path = tmp_path / "radar.sqlite"
    _write_price_cache(db_path, "AAA", [10, 11, 12, 13, 14, 15], [10, 9.8, 11.8, 12.6, 13.8, 14.5])
    _write_price_cache(db_path, "BBB", [9, 8.5, 8.3, 8.1, 8.4, 8.7], [9, 8.2, 8.0, 7.9, 8.1, 8.5])

    features = m._prepare_base_features_from_df(pd.DataFrame(rows))
    daily_all, _summary_df, lifecycle_df, perf, _audit_df = m.build_april_replay(
        features,
        "2026-04-01",
        "2026-04-01",
        db_path=str(db_path),
    )

    assert "forward_return_5d_pct" in daily_all.columns
    assert daily_all["forward_return_5d_pct"].notna().any()
    assert set(perf["group_name"]) == {"top30", "special_loose", "special_mid", "special_strict", "special_any"}
    top30 = perf.loc[perf["group_name"] == "top30"].iloc[0]
    assert top30["avg_forward_return_5d_pct"] is not None
    assert top30["avg_max_adverse_5d_pct"] is not None
    assert lifecycle_df["days_seen_in_special_any"].max() >= 1
    assert lifecycle_df["best_special_score"].notna().all()

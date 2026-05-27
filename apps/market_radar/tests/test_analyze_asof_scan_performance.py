from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd

from market_radar.data_access import init_db


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_asof_scan_performance.py"
    spec = importlib.util.spec_from_file_location("analyze_asof_scan_performance", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asof_scan_performance_outputs_returns(tmp_path: Path) -> None:
    m = _load_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "period": "2026-04-01",
            "symbol": "AAA",
            "strategy": "volume_spike_strict",
            "signal_date": "2026-04-30",
            "entry_date": "2026-04-30",
            "close": 10.0,
            "turnover": 20_000_000.0,
            "avg_turnover_20d": 15_000_000.0,
            "volume_ratio_20d": 2.0,
            "turnover_ratio_20d": 1.4,
            "ma20": 9.5,
            "above_ma20": True,
            "rsi_14": 62.0,
            "return_5d_pct": 8.0,
            "return_10d_pct": 14.0,
            "close_position": 0.72,
        },
        {
            "period": "2026-04-01",
            "symbol": "BBB",
            "strategy": "volume_spike_strict",
            "signal_date": "2026-04-30",
            "entry_date": "2026-04-30",
            "close": 8.0,
            "turnover": 16_000_000.0,
            "avg_turnover_20d": 12_000_000.0,
            "volume_ratio_20d": 1.7,
            "turnover_ratio_20d": 1.25,
            "ma20": 7.9,
            "above_ma20": True,
            "rsi_14": 67.0,
            "return_5d_pct": 10.0,
            "return_10d_pct": 17.0,
            "close_position": 0.6,
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "candidate_features.csv", index=False)

    db = str(tmp_path / "radar.sqlite")
    init_db(db)

    def _payload(values: list[tuple[str, float]]) -> str:
        return json.dumps({"records": [{"date": d, "close": c} for d, c in values]})

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv_cache (symbol, fetched_at, payload_json, source) VALUES (?, datetime('now'), ?, 'test')",
            ("AAA", _payload([("2026-04-30", 10.0), ("2026-05-05", 12.0)])),
        )
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv_cache (symbol, fetched_at, payload_json, source) VALUES (?, datetime('now'), ?, 'test')",
            ("BBB", _payload([("2026-04-30", 8.0), ("2026-05-05", 7.2)])),
        )
        xu_vals = [("2026-04-01", 100.0), ("2026-04-30", 110.0), ("2026-05-05", 115.0)]
        for i in range(2, 22):
            xu_vals.insert(0, (f"2026-03-{i:02d}", 90.0 + i * 0.2))
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv_cache (symbol, fetched_at, payload_json, source) VALUES (?, datetime('now'), ?, 'test')",
            ("XU100", _payload(xu_vals)),
        )

    detail, summary = m.run_analysis(out_dir, db, "2026-04-30", "liquidity_safe_score")
    assert not detail.empty
    assert "production_rank" in detail.columns
    assert "score_bucket" in detail.columns
    assert "alpha_vs_xu100_pct" in detail.columns
    assert summary["as_of_date"] == "2026-04-30"
    assert summary["total_signal_count"] == 2


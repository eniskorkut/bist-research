from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    script_path = root / "scripts" / "backfill_candidate_features_partitioned.py"
    spec = importlib.util.spec_from_file_location("backfill_candidate_features_partitioned", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_iter_years_accepts_ranges_and_lists() -> None:
    m = _load_module()
    assert m._iter_years("2020-01-01", "2026-05-28", "2020,2022-2023") == [2020, 2022, 2023]


def test_backfill_summary_detects_required_columns_and_duplicates(tmp_path) -> None:
    m = _load_module()
    year_dir = tmp_path / "year=2020"
    year_dir.mkdir()
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
                "volume_ratio_20d": 1.5,
                "turnover_ratio_20d": 1.5,
                "ma20": 9.0,
                "above_ma20": True,
                "rsi_14": 55.0,
                "return_5d_pct": 5.0,
                "return_10d_pct": 8.0,
                "close_position": 0.7,
            },
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
                "volume_ratio_20d": 1.5,
                "turnover_ratio_20d": 1.5,
                "ma20": 9.0,
                "above_ma20": True,
                "rsi_14": 55.0,
                "return_5d_pct": 5.0,
                "return_10d_pct": 8.0,
                "close_position": 0.7,
            },
        ]
    ).to_csv(year_dir / "candidate_features.csv", index=False)

    summary = m._summarize(tmp_path, "2020-01-01", "2021-12-31", [], "2018-01-01")
    assert summary["min_signal_date"] == "2020-01-02"
    assert summary["duplicate_symbol_date_count"] == 1
    assert summary["data_missing_years"] == [2021]
    assert summary["missing_required_columns"] == []

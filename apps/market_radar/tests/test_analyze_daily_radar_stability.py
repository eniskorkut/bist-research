from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_daily_radar_stability.py"
    spec = importlib.util.spec_from_file_location("analyze_daily_radar_stability", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_overlap_status_and_rates() -> None:
    m = _load_module()
    d1 = pd.DataFrame(
        [
            {"symbol": "AAA", "production_rank": 1, "production_score": 90.0, "score_bucket": "top20"},
            {"symbol": "BBB", "production_rank": 2, "production_score": 80.0, "score_bucket": "top20"},
            {"symbol": "CCC", "production_rank": 3, "production_score": 70.0, "score_bucket": "top20"},
        ]
    )
    d2 = pd.DataFrame(
        [
            {"symbol": "BBB", "production_rank": 1, "production_score": 88.0, "score_bucket": "top20"},
            {"symbol": "CCC", "production_rank": 2, "production_score": 79.0, "score_bucket": "top20"},
            {"symbol": "DDD", "production_rank": 3, "production_score": 60.0, "score_bucket": "top20"},
        ]
    )

    ov, summary = m.compute_overlap(d1, d2, "2026-05-21", "2026-05-22")
    statuses = dict(zip(ov["symbol"], ov["status"], strict=False))
    assert statuses["BBB"] == "repeated"
    assert statuses["CCC"] == "repeated"
    assert statuses["DDD"] == "new"
    assert statuses["AAA"] == "dropped"
    assert summary["repeated_count"] == 2
    assert summary["new_count"] == 1
    assert summary["dropped_count"] == 1
    assert summary["overlap_rate"] == 2 / 3
    assert summary["new_rate"] == 1 / 3
    assert summary["dropped_rate"] == 1 / 3


def test_topn_overlap_and_rank_direction() -> None:
    m = _load_module()
    d1 = pd.DataFrame(
        [
            {"symbol": "AAA", "production_rank": 1, "production_score": 90.0, "score_bucket": "top20"},
            {"symbol": "BBB", "production_rank": 2, "production_score": 85.0, "score_bucket": "top20"},
            {"symbol": "CCC", "production_rank": 3, "production_score": 80.0, "score_bucket": "top20"},
            {"symbol": "DDD", "production_rank": 4, "production_score": 75.0, "score_bucket": "top20"},
        ]
    )
    d2 = pd.DataFrame(
        [
            {"symbol": "BBB", "production_rank": 1, "production_score": 92.0, "score_bucket": "top20"},
            {"symbol": "AAA", "production_rank": 2, "production_score": 88.0, "score_bucket": "top20"},
            {"symbol": "EEE", "production_rank": 3, "production_score": 84.0, "score_bucket": "top20"},
            {"symbol": "CCC", "production_rank": 4, "production_score": 77.0, "score_bucket": "top20"},
        ]
    )

    _ov, summary = m.compute_overlap(d1, d2, "2026-05-21", "2026-05-22")
    assert summary["top20_overlap_count"] == 3
    assert summary["top30_overlap_count"] == 3
    assert summary["top50_overlap_count"] == 3
    assert summary["top30_new_symbols"] == ["EEE"]
    assert summary["top30_dropped_symbols"] == ["DDD"]
    assert summary["top30_rank_improved_symbols"] == ["BBB"]
    assert summary["top30_rank_worsened_symbols"] == ["AAA", "CCC"]

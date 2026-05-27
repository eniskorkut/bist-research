from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_regime_adaptive_filters.py"
    spec = importlib.util.spec_from_file_location("analyze_regime_adaptive_filters", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_winner_among_four_prefers_max_alpha() -> None:
    m = _load_module()
    out = m._winner_among_four(
        current_alpha=1.1,
        relaxed_alpha=2.2,
        regime_alpha=1.9,
        inverse_alpha=0.5,
    )
    assert out == "relaxed_strong_close"


def test_production_recommendation_rule_for_regime_adaptive() -> None:
    m = _load_module()
    out = m._production_recommendation(
        current_avg=2.0,
        relaxed_avg=2.5,
        regime_avg=3.0,
        inverse_avg=2.7,
        current_median=1.8,
        relaxed_median=2.0,
        regime_median=2.1,
        inverse_median=1.9,
        positive_period_count=18,
    )
    assert out == "regime_adaptive"


def test_build_threshold_robustness_outputs_expected_rows() -> None:
    m = _load_module()
    df = pd.DataFrame(
        [
            {
                "period": "2024-01-01",
                "xu100_return_20d_pct": 2.0,
                "current_alpha_30d": 3.0,
                "relaxed_alpha_30d": 2.0,
                "current_signal_count": 10,
                "relaxed_signal_count": 12,
            },
            {
                "period": "2024-02-01",
                "xu100_return_20d_pct": 0.5,
                "current_alpha_30d": -1.0,
                "relaxed_alpha_30d": 1.0,
                "current_signal_count": 9,
                "relaxed_signal_count": 11,
            },
        ]
    )
    out, summary = m.build_threshold_robustness(df, [0.0, 1.5])
    assert list(out["threshold"]) == [0.0, 1.5]
    assert "avg_alpha_30d" in out.columns
    assert "selected_current_month_count" in out.columns
    assert int(out.loc[out["threshold"] == 1.5, "selected_current_month_count"].iloc[0]) == 1
    assert "best_by_avg_alpha_30d" in summary


def test_build_oos_validation_selects_best_train_threshold_and_summary() -> None:
    m = _load_module()
    df = pd.DataFrame(
        [
            {
                "period": "2024-01-01",
                "xu100_return_20d_pct": 2.5,
                "current_alpha_30d": 4.0,
                "relaxed_alpha_30d": 1.0,
                "current_signal_count": 10,
                "relaxed_signal_count": 12,
            },
            {
                "period": "2024-02-01",
                "xu100_return_20d_pct": 0.5,
                "current_alpha_30d": -1.0,
                "relaxed_alpha_30d": 2.0,
                "current_signal_count": 9,
                "relaxed_signal_count": 11,
            },
            {
                "period": "2025-07-01",
                "xu100_return_20d_pct": 3.0,
                "current_alpha_30d": 1.0,
                "relaxed_alpha_30d": 0.5,
                "current_signal_count": 8,
                "relaxed_signal_count": 9,
            },
            {
                "period": "2025-08-01",
                "xu100_return_20d_pct": 0.0,
                "current_alpha_30d": -0.2,
                "relaxed_alpha_30d": 0.3,
                "current_signal_count": 7,
                "relaxed_signal_count": 10,
            },
        ]
    )
    out, summary = m.build_oos_validation(df, [0.0, 1.0, 2.0])
    assert len(out) == 2
    assert summary["selected_threshold"] == 1.0
    assert "test_regime_avg_alpha_30d" in summary
    assert "recommendation" in summary

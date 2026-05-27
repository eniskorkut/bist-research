from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "analyze_production_score_topn.py"
    spec = importlib.util.spec_from_file_location("analyze_production_score_topn", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_topn_analysis_outputs_expected_columns(tmp_path: Path) -> None:
    m = _load_module()
    rows = [
        {
            "period": "2025-07-01",
            "symbol": "AAA",
            "strategy": "volume_spike_strict",
            "signal_date": "2025-07-03",
            "entry_date": "2025-07-04",
            "volume_ratio_20d": 2.0,
            "turnover_ratio_20d": 1.7,
            "turnover": 30_000_000.0,
            "avg_turnover_20d": 20_000_000.0,
            "close_position": 0.70,
            "close": 12.0,
            "ma20": 11.5,
            "above_ma20": True,
            "rsi_14": 63.0,
            "return_5d_pct": 10.0,
            "return_10d_pct": 18.0,
            "future_return_15d": 4.0,
            "future_return_30d": 8.0,
            "benchmark_return_15d": 2.0,
            "benchmark_return_30d": 5.0,
            "alpha_30d": 3.0,
        },
        {
            "period": "2025-07-01",
            "symbol": "BBB",
            "strategy": "volume_spike_strict",
            "signal_date": "2025-07-05",
            "entry_date": "2025-07-06",
            "volume_ratio_20d": 1.6,
            "turnover_ratio_20d": 1.3,
            "turnover": 15_000_000.0,
            "avg_turnover_20d": 11_000_000.0,
            "close_position": 0.55,
            "close": 9.0,
            "ma20": 8.8,
            "above_ma20": True,
            "rsi_14": 70.0,
            "return_5d_pct": 14.0,
            "return_10d_pct": 22.0,
            "future_return_15d": 2.0,
            "future_return_30d": 4.0,
            "benchmark_return_15d": 2.0,
            "benchmark_return_30d": 5.0,
            "alpha_30d": -1.0,
        },
        {
            "period": "2025-08-01",
            "symbol": "CCC",
            "strategy": "volume_spike_strict",
            "signal_date": "2025-08-02",
            "entry_date": "2025-08-05",
            "volume_ratio_20d": 2.1,
            "turnover_ratio_20d": 1.8,
            "turnover": 40_000_000.0,
            "avg_turnover_20d": 25_000_000.0,
            "close_position": 0.75,
            "close": 20.0,
            "ma20": 19.0,
            "above_ma20": True,
            "rsi_14": 61.0,
            "return_5d_pct": 8.0,
            "return_10d_pct": 14.0,
            "future_return_15d": 3.0,
            "future_return_30d": 6.0,
            "benchmark_return_15d": 1.5,
            "benchmark_return_30d": 4.5,
            "alpha_30d": 1.5,
        },
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "candidate_features.csv", index=False)
    pd.DataFrame(
        [
            {"period": "2025-07-01", "xu100_return_20d_pct": 2.0},
            {"period": "2025-08-01", "xu100_return_20d_pct": 0.5},
        ]
    ).to_csv(out_dir / "regime_config_comparison.csv", index=False)

    details, summary = m.build_topn_analysis(output_dir=out_dir)
    assert not details.empty
    required_cols = {
        "period",
        "score_name",
        "regime_bonus_enabled",
        "top_n",
        "selected_count",
        "basket_return_15d",
        "basket_return_30d",
        "basket_alpha_30d",
        "baseline_relaxed_alpha_30d",
        "score_minus_baseline",
        "positive_period",
        "beat_baseline",
    }
    assert required_cols.issubset(set(details.columns))
    assert "variants" in summary
    assert len(summary["variants"]) == 30
    sample = summary["variants"][0]
    assert "avg_alpha_30d" in sample
    assert "production_candidate_passed" in sample


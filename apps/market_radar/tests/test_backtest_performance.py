import pandas as pd

from market_radar.backtesting.performance import (
    build_monthly_summary,
    build_strategy_summary,
    build_yearly_summary,
)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "strategy": "positive_interest", "signal_date": "2026-01-10", "alpha_15d": 2.0, "alpha_30d": 3.0, "return_15d": 5.0, "return_30d": 7.0, "beat_xu100_15d": True, "beat_xu100_30d": True},
            {"symbol": "BBB", "strategy": "positive_interest", "signal_date": "2026-01-15", "alpha_15d": -1.0, "alpha_30d": 1.0, "return_15d": 1.0, "return_30d": 4.0, "beat_xu100_15d": False, "beat_xu100_30d": True},
        ]
    )


def test_strategy_summary() -> None:
    out = build_strategy_summary(_signals())
    assert not out.empty
    assert out.iloc[0]["strategy"] == "positive_interest"


def test_monthly_summary() -> None:
    out = build_monthly_summary(_signals())
    assert not out.empty
    assert "month" in out.columns


def test_yearly_summary() -> None:
    out = build_yearly_summary(_signals())
    assert not out.empty
    assert "year" in out.columns

from __future__ import annotations

from valuation.data_access import BistSnapshot
from valuation.profit_estimator import (
    estimate_net_income_auto,
    estimate_net_income_revenue_margin,
    estimate_net_income_seasonal,
    estimate_net_income_ttm,
    get_latest_reported_net_income,
)


class FakeClient:
    def __init__(self, snapshot: BistSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, symbol: str) -> BistSnapshot:
        return self.snapshot


def test_individual_estimators() -> None:
    snapshot = BistSnapshot(
        symbol="THYAO",
        price=100.0,
        market_cap=100_000.0,
        pe_ratio=8.0,
        pb_ratio=1.8,
        shares_outstanding=1_000.0,
        paid_in_capital=1_000.0,
        equity=50_000.0,
        net_income_latest_period=6_000.0,
        net_income_ttm=10_500.0,
        revenue_latest_period=45_000.0,
        previous_year_same_period_net_income=5_000.0,
        previous_year_full_net_income=9_000.0,
        previous_year_same_period_revenue=40_000.0,
        previous_year_full_revenue=80_000.0,
        average_margin_3y=0.11,
        period_type="interim",
        period_label="2025/06",
        missing_fields=[],
    )
    client = FakeClient(snapshot)
    assert get_latest_reported_net_income("THYAO", client=client) == 6_000.0
    assert estimate_net_income_ttm("THYAO", client=client) == 10_500.0
    assert estimate_net_income_seasonal("THYAO", client=client) == 10_800.0
    assert estimate_net_income_revenue_margin("THYAO", client=client) == 9_900.0


def test_auto_estimator_annual_uses_reported() -> None:
    snapshot = BistSnapshot(
        symbol="ASELS",
        price=90.0,
        market_cap=90_000.0,
        pe_ratio=10.0,
        pb_ratio=2.0,
        shares_outstanding=1_000.0,
        paid_in_capital=1_000.0,
        equity=45_000.0,
        net_income_latest_period=8_500.0,
        net_income_ttm=8_400.0,
        revenue_latest_period=50_000.0,
        previous_year_same_period_net_income=4_500.0,
        previous_year_full_net_income=8_000.0,
        previous_year_same_period_revenue=25_000.0,
        previous_year_full_revenue=48_000.0,
        average_margin_3y=0.13,
        period_type="annual",
        period_label="2024/12",
        missing_fields=[],
    )
    result = estimate_net_income_auto("ASELS", client=FakeClient(snapshot))
    assert result.selected_method == "reported_annual"
    assert result.estimated_net_income == 8_500.0


def test_auto_estimator_interim_uses_median() -> None:
    snapshot = BistSnapshot(
        symbol="TUPRS",
        price=120.0,
        market_cap=120_000.0,
        pe_ratio=7.0,
        pb_ratio=1.2,
        shares_outstanding=1_000.0,
        paid_in_capital=1_000.0,
        equity=60_000.0,
        net_income_latest_period=5_000.0,
        net_income_ttm=11_000.0,
        revenue_latest_period=42_000.0,
        previous_year_same_period_net_income=4_000.0,
        previous_year_full_net_income=9_000.0,
        previous_year_same_period_revenue=35_000.0,
        previous_year_full_revenue=78_000.0,
        average_margin_3y=0.12,
        period_type="interim",
        period_label="2025/06",
        missing_fields=[],
    )
    result = estimate_net_income_auto("TUPRS", client=FakeClient(snapshot))
    # ttm=11000, seasonal=11250, revenue_margin=11232 -> median=11232
    assert result.selected_method == "median_of_available_methods"
    assert result.estimated_net_income == 11232.0

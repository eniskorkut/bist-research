from __future__ import annotations

from valuation.data_access import BistSnapshot
from valuation.valuation_engine import run_valuation


class FakeClient:
    def __init__(self, snapshot: BistSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, symbol: str) -> BistSnapshot:
        return self.snapshot


def test_valuation_engine_formulas() -> None:
    snapshot = BistSnapshot(
        symbol="THYAO",
        price=100.0,
        market_cap=100_000.0,
        pe_ratio=10.0,
        pb_ratio=2.0,
        shares_outstanding=1_000.0,
        paid_in_capital=1_000.0,
        equity=50_000.0,
        net_income_latest_period=6_000.0,
        net_income_ttm=10_000.0,
        revenue_latest_period=40_000.0,
        previous_year_same_period_net_income=5_000.0,
        previous_year_full_net_income=8_000.0,
        previous_year_same_period_revenue=30_000.0,
        previous_year_full_revenue=75_000.0,
        average_margin_3y=0.12,
        period_type="interim",
        period_label="2025/06",
        source="borsapy",
        missing_fields=[],
    )
    sector_metrics = {
        "pe_median": 12.0,
        "pe_aggregate": 11.0,
        "pb_median": 1.8,
        "pb_aggregate": 1.6,
        "roe_aggregate": 0.15,
    }
    result = run_valuation("THYAO", client=FakeClient(snapshot), sector_metrics=sector_metrics)
    # auto estimate median(ttm=10000, seasonal=9600, rev_margin=12000) => 10000
    assert result.estimated_net_income == 10000.0
    assert result.target_prices["cari_fk"] == 100.0
    assert result.target_prices["pd_dd"] == 100.0
    assert result.target_prices["odenmis_sermaye"] == 1000.0
    assert result.target_prices["potansiyel_piyasa_degeri"] == 100.0
    assert result.target_prices["ozsermaye_karliligi"] == 20.0
    assert result.average_target_price == 264.0
    assert result.upside_potential_pct == 164.0
    assert result.sector_target_prices["sektor_fk_hedef"] == 120.0
    assert result.sector_target_prices["sektor_pd_dd_hedef"] == 90.0

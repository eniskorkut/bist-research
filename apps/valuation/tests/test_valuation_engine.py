from __future__ import annotations

from valuation.data_access import BistSnapshot
from valuation.valuation_engine import run_valuation, run_valuation_from_snapshot


class FakeClient:
    def __init__(self, snapshot: BistSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, symbol: str) -> BistSnapshot:
        return self.snapshot


def _make_snapshot() -> BistSnapshot:
    return BistSnapshot(
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


def test_valuation_engine_formulas() -> None:
    snapshot = _make_snapshot()
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


def test_run_valuation_from_snapshot_no_borsapy() -> None:
    """run_valuation_from_snapshot must produce correct results without
    touching borsapy at all."""

    cached_snapshot = {
        "symbol": "THYAO",
        "price": 100.0,
        "market_cap": 100_000.0,
        "pe_ratio": 10.0,
        "pb_ratio": 2.0,
        "shares_outstanding": 1_000.0,
        "paid_in_capital": 1_000.0,
        "equity": 50_000.0,
        "estimated_net_income": 10_000.0,
        "period_type": "interim",
        "financial_period": "2025/06",
        "missing_fields_json": [],
    }
    sector_metrics = {
        "pe_median": 12.0,
        "pe_aggregate": 11.0,
        "pb_median": 1.8,
        "pb_aggregate": 1.6,
        "roe_aggregate": 0.15,
    }

    result = run_valuation_from_snapshot(cached_snapshot, sector_metrics=sector_metrics)

    assert result.source == "cache"
    assert result.estimation.selected_method == "cache"
    assert result.estimated_net_income == 10_000.0
    assert result.target_prices["cari_fk"] == 100.0
    assert result.target_prices["pd_dd"] == 100.0
    assert result.target_prices["odenmis_sermaye"] == 1000.0
    assert result.target_prices["potansiyel_piyasa_degeri"] == 100.0
    assert result.target_prices["ozsermaye_karliligi"] == 20.0
    assert result.average_target_price == 264.0
    assert result.upside_potential_pct == 164.0
    assert result.sector_target_prices["sektor_fk_hedef"] == 120.0
    assert result.sector_target_prices["sektor_pd_dd_hedef"] == 90.0


def test_run_valuation_from_snapshot_missing_fields_string() -> None:
    """missing_fields_json stored as a JSON string should be parsed."""
    cached_snapshot = {
        "symbol": "TEST",
        "price": 50.0,
        "market_cap": 50_000.0,
        "pe_ratio": 5.0,
        "pb_ratio": 1.0,
        "shares_outstanding": 1_000.0,
        "paid_in_capital": 1_000.0,
        "equity": 50_000.0,
        "estimated_net_income": None,
        "period_type": "interim",
        "financial_period": "2025/06",
        "missing_fields_json": '["equity", "net_income_latest_period"]',
    }
    result = run_valuation_from_snapshot(cached_snapshot)
    assert "equity" in result.missing_fields
    assert "estimated_net_income" in result.missing_fields


def test_run_valuation_from_refresh_snapshot() -> None:
    """Refresh sonrası cache'e yazılan snapshot ile valuation çalışmalı."""
    cached_snapshot = {
        "symbol": "ASELS",
        "price": 120.0,
        "market_cap": 240_000.0,
        "pe_ratio": 12.0,
        "pb_ratio": 2.4,
        "shares_outstanding": 2_000.0,
        "paid_in_capital": 2_000.0,
        "equity": 100_000.0,
        "estimated_net_income": 18_000.0,
        "period_type": "interim",
        "financial_period": "2025/06",
        "source": "borsapy",
        "missing_fields_json": [],
    }
    sector_metrics = {"pe_median": 10.0, "pb_median": 2.0, "roe_aggregate": 0.16}
    result = run_valuation_from_snapshot(cached_snapshot, sector_metrics=sector_metrics)
    assert result.symbol == "ASELS"
    assert result.source == "cache"
    assert result.target_prices["cari_fk"] == 108.0

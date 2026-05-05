from __future__ import annotations

from valuation.data_access import BistSnapshot
import valuation.valuation_engine as engine
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
        net_income_source="financial_statement",
        equity_source="financial_statement",
        revenue_source="financial_statement",
        missing_fields=[],
    )


def test_scenario_structure_exists() -> None:
    result = run_valuation("THYAO", client=FakeClient(_make_snapshot()))
    assert "ttm" in result.valuation_scenarios
    assert "year_end" in result.valuation_scenarios


def test_paid_capital_course_formula() -> None:
    cached = {
        "symbol": "THYAO",
        "price": 308.25,
        "market_cap": 425_385_000_000.0,
        "shares_outstanding": 1_380_000_000.0,
        "paid_in_capital": 1_380_000_000.0,
        "pe_ratio": 3.0,
        "pb_ratio": 0.4,
        "equity": 1_063_462_500_000.0,
        "estimated_net_income": 142_763_197_242.16,
        "net_income_ttm": 142_763_197_242.16,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "implied_from_pe",
        "equity_source": "implied_from_pb",
    }
    result = run_valuation_from_snapshot(cached)
    scenario = result.valuation_scenarios["year_end"]
    eps = scenario.paid_capital_details["eps"]
    assert round(float(eps), 2) == 103.45
    assert round(float(scenario.paid_capital_details["x10"]), 2) == 1034.52
    assert scenario.target_prices["odenmis_sermaye_final"] != 31888.0


def test_historical_pe_none_uses_sector_pe_fallback() -> None:
    cached = {
        "symbol": "ASELS",
        "price": 120.0,
        "market_cap": 240_000.0,
        "shares_outstanding": 2_000.0,
        "paid_in_capital": 2_000.0,
        "pe_ratio": 12.0,
        "pb_ratio": 2.4,
        "equity": 100_000.0,
        "estimated_net_income": 18_000.0,
        "net_income_ttm": 18_000.0,
        "period_type": "interim",
        "financial_period": "2025/06",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(
        cached,
        sector_metrics={"pe_median": 3.0},
    )
    scenario = result.valuation_scenarios["year_end"]
    assert scenario.paid_capital_details["historical_pe_median"] is None
    assert scenario.paid_capital_details["final_method"] == "average_x10_sector_pe"
    assert scenario.paid_capital_details["final"] is not None
    assert scenario.paid_capital_details["included_in_fair_value"] is True


def test_historical_and_sector_none_uses_current_pe_only() -> None:
    cached = {
        "symbol": "ASELS",
        "price": 120.0,
        "market_cap": 240_000.0,
        "shares_outstanding": 2_000.0,
        "paid_in_capital": 2_000.0,
        "pe_ratio": 3.0,
        "pb_ratio": 2.4,
        "equity": 100_000.0,
        "estimated_net_income": 18_000.0,
        "net_income_ttm": 18_000.0,
        "period_type": "interim",
        "financial_period": "2025/06",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached, sector_metrics={"pe_median": None})
    scenario = result.valuation_scenarios["year_end"]
    assert scenario.paid_capital_details["final_method"] == "current_pe_only"
    assert scenario.paid_capital_details["final"] == scenario.paid_capital_details["current_pe_value"]


def test_only_x10_available_is_info_only(monkeypatch) -> None:
    monkeypatch.setattr(engine, "get_historical_pe_median", lambda symbol: None)
    cached = {
        "symbol": "ASELS",
        "price": 120.0,
        "market_cap": 240_000.0,
        "shares_outstanding": 2_000.0,
        "paid_in_capital": 2_000.0,
        "pe_ratio": None,
        "pb_ratio": 2.4,
        "equity": 100_000.0,
        "estimated_net_income": 18_000.0,
        "net_income_ttm": 18_000.0,
        "period_type": "interim",
        "financial_period": "2025/06",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached, sector_metrics={"pe_median": None})
    scenario = result.valuation_scenarios["year_end"]
    assert scenario.paid_capital_details["x10"] is not None
    assert scenario.paid_capital_details["final"] is not None
    assert scenario.paid_capital_details["final_method"] == "current_pe_only"


def test_negative_income_excludes_pe_eps_methods() -> None:
    cached = {
        "symbol": "ODINE",
        "price": 1139.0,
        "market_cap": 125_859_500_000.0,
        "shares_outstanding": 110_500_000.0,
        "paid_in_capital": 110_500_000.0,
        "pe_ratio": None,
        "pb_ratio": None,
        "equity": 2_206_218_485.0,
        "estimated_net_income": -21_986_528.0,
        "net_income_ttm": -21_986_528.0,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    scenario = result.valuation_scenarios["year_end"]
    assert scenario.target_prices["cari_fk"] is None
    assert scenario.target_prices["odenmis_sermaye_x10"] is None
    assert scenario.target_prices["odenmis_sermaye_final"] is None
    assert scenario.valuation_status == "negative_net_income"
    assert scenario.fair_value_median is None
    assert result.ratio_sources["pe_ratio_source"] == "not_applicable_negative_income"


def test_negative_ttm_with_positive_year_end_runs_only_year_end() -> None:
    cached = {
        "symbol": "ODINE",
        "price": 100.0,
        "market_cap": 1_000_000.0,
        "shares_outstanding": 10_000.0,
        "paid_in_capital": 10_000.0,
        "pe_ratio": 4.0,
        "pb_ratio": 1.2,
        "equity": 800_000.0,
        "estimated_net_income": 120_000.0,
        "net_income_ttm": -10_000.0,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "auto_estimate",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    ttm = result.valuation_scenarios["ttm"]
    year_end = result.valuation_scenarios["year_end"]
    assert ttm.valuation_status == "negative_net_income"
    assert ttm.fair_value_median is None
    assert year_end.valuation_status in {"full", "partial", "insufficient_independent_methods"}


def test_ttm_positive_year_end_negative_ttm_remains_open_year_end_disabled() -> None:
    cached = {
        "symbol": "TEST",
        "price": 100.0,
        "market_cap": 1_000_000.0,
        "shares_outstanding": 10_000.0,
        "paid_in_capital": 10_000.0,
        "pe_ratio": None,
        "pb_ratio": 1.5,
        "equity": 600_000.0,
        "estimated_net_income": -20_000.0,
        "net_income_ttm": 80_000.0,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    assert result.valuation_scenarios["ttm"].valuation_status in {"full", "partial", "insufficient_independent_methods"}
    assert result.valuation_scenarios["year_end"].valuation_status == "negative_net_income"


def test_ttm_missing_uses_missing_ttm_status() -> None:
    cached = {
        "symbol": "TEST",
        "price": 100.0,
        "market_cap": 1_000_000.0,
        "shares_outstanding": 10_000.0,
        "paid_in_capital": 10_000.0,
        "pe_ratio": 5.0,
        "pb_ratio": 1.5,
        "equity": 600_000.0,
        "estimated_net_income": 50_000.0,
        "net_income_ttm": None,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    assert result.valuation_scenarios["ttm"].valuation_status == "missing_ttm_net_income"


def test_derived_pb_not_independent_target() -> None:
    cached = {
        "symbol": "TEST",
        "price": 100.0,
        "market_cap": 1_000_000.0,
        "shares_outstanding": 10_000.0,
        "paid_in_capital": 10_000.0,
        "pe_ratio": None,
        "pb_ratio": None,
        "equity": 500_000.0,
        "estimated_net_income": 200_000.0,
        "net_income_ttm": 200_000.0,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    scenario = result.valuation_scenarios["year_end"]
    assert result.ratio_sources["pb_ratio_source"] == "derived"
    assert scenario.method_types["pd_dd"] == "current_implied"
    assert "pd_dd" not in scenario.included_methods


def test_derived_pb_ratio_used_when_missing() -> None:
    cached = {
        "symbol": "TEST",
        "price": 100.0,
        "market_cap": 1_000_000.0,
        "shares_outstanding": 10_000.0,
        "paid_in_capital": 10_000.0,
        "pe_ratio": 5.0,
        "pb_ratio": None,
        "equity": 500_000.0,
        "estimated_net_income": 200_000.0,
        "net_income_ttm": 200_000.0,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "financial_statement",
        "equity_source": "financial_statement",
    }
    result = run_valuation_from_snapshot(cached)
    assert result.pb_ratio == 2.0
    assert result.ratio_sources["pb_ratio_source"] == "derived"


def test_thyao_full_valuation_not_broken() -> None:
    cached = {
        "symbol": "THYAO",
        "price": 308.25,
        "market_cap": 425_385_000_000.0,
        "shares_outstanding": 1_380_000_000.0,
        "paid_in_capital": 1_380_000_000.0,
        "pe_ratio": 3.0,
        "pb_ratio": 0.4,
        "equity": 1_063_462_500_000.0,
        "estimated_net_income": 142_763_197_242.16,
        "net_income_ttm": 142_763_197_242.16,
        "period_type": "interim",
        "financial_period": "2025",
        "data_quality_status": "usable",
        "missing_fields_json": [],
        "net_income_source": "implied_from_pe",
        "equity_source": "implied_from_pb",
    }
    result = run_valuation_from_snapshot(cached)
    assert result.valuation_scenarios["ttm"].fair_value_median is not None
    assert result.valuation_scenarios["year_end"].fair_value_median is not None

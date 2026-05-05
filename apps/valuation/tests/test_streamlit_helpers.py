from __future__ import annotations

from valuation.streamlit_app import (
    build_target_price_dataframe,
    fmt_money,
    to_plain_dict,
)
from valuation.valuation_engine import ScenarioValuation


def _scenario() -> ScenarioValuation:
    return ScenarioValuation(
        scenario_key="year_end",
        period_label="2025",
        net_income=100.0,
        net_income_source="cache",
        historical_pe_median=None,
        target_prices={
            "cari_fk": 120.0,
            "odenmis_sermaye_x10": 500.0,
            "odenmis_sermaye_final": 300.0,
        },
        included_methods=["cari_fk", "odenmis_sermaye_final"],
        excluded_methods=["odenmis_sermaye_x10"],
        fair_value_median=200.0,
        fair_value_mean_filtered=210.0,
        upside_potential_pct=10.0,
        paid_capital_details={
            "eps": 10.0,
            "x10": 100.0,
            "historical_pe_median": None,
            "historical_pe_value": None,
            "sector_pe_median": None,
            "sector_pe_value": None,
            "current_pe_value": 40.0,
            "final": 40.0,
            "final_method": "current_pe_only",
            "included_in_fair_value": True,
        },
        method_notes={},
        method_types={"cari_fk": "independent_target"},
        valuation_status="full",
    )


def test_fmt_money_none() -> None:
    assert fmt_money(None) == "N/A"


def test_fmt_money_value() -> None:
    assert fmt_money(1234.5) == "1,234.50"


def test_build_target_price_dataframe_included_excluded() -> None:
    df = build_target_price_dataframe(_scenario())
    included = df[df["Yöntem"] == "Cari F/K"].iloc[0]
    excluded = df[df["Yöntem"] == "Ödenmiş Sermaye: EPS × 10"].iloc[0]
    assert included["Adil Değere Dahil"] == "Evet"
    assert excluded["Adil Değere Dahil"] == "Hayır"


def test_to_plain_dict_dataclass() -> None:
    plain = to_plain_dict(_scenario())
    assert isinstance(plain, dict)
    assert plain["scenario_key"] == "year_end"

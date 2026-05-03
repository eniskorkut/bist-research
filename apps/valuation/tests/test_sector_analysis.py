from __future__ import annotations

from valuation.sector_analysis import calculate_sector_metrics, compare_company_to_sector


def test_calculate_sector_metrics() -> None:
    snapshots = [
        {"pe_ratio": 10.0, "pb_ratio": 1.5, "market_cap": 100.0, "estimated_net_income": 10.0, "equity": 50.0},
        {"pe_ratio": 8.0, "pb_ratio": 1.2, "market_cap": 80.0, "estimated_net_income": 8.0, "equity": 40.0},
        {"pe_ratio": -3.0, "pb_ratio": 0.0, "market_cap": 20.0, "estimated_net_income": -2.0, "equity": 10.0},
    ]
    metrics = calculate_sector_metrics(snapshots)
    assert metrics["pe_median"] == 9.0
    assert metrics["pb_median"] == 1.35
    assert round(metrics["pe_aggregate"], 4) == 11.1111
    assert metrics["pb_aggregate"] == 2.0
    assert metrics["roe_aggregate"] == 0.18


def test_compare_company_to_sector() -> None:
    company = {"pe_ratio": 7.0, "pb_ratio": 1.0, "roe": 0.25, "estimated_net_income": 100, "equity": 400}
    sector = {"pe_median": 10.0, "pe_aggregate": 11.0, "pb_median": 1.4, "pb_aggregate": 1.5, "roe_aggregate": 0.2}
    comparison = compare_company_to_sector(company, sector)
    assert "fk_sektore_gore_iskontolu" in comparison["interpretation_flags"]
    assert "pd_dd_sektore_gore_iskontolu" in comparison["interpretation_flags"]
    assert "roe_sektor_ustu" in comparison["interpretation_flags"]

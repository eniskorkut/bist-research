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
    # pe_aggregate: only companies 1&2 (net_income>0 AND market_cap>0)
    # (100+80)/(10+8) = 10.0
    assert metrics["pe_aggregate"] == 10.0
    # pb_aggregate: all three have equity>0 AND market_cap>0
    # (100+80+20)/(50+40+10) = 2.0
    assert metrics["pb_aggregate"] == 2.0
    # roe_aggregate: all three have equity>0
    # (10+8+(-2))/(50+40+10) = 0.16
    assert metrics["roe_aggregate"] == 0.16
    assert metrics["negative_income_count"] == 1
    assert metrics["pe_valid_count"] == 2


def test_calculate_sector_metrics_same_valid_set() -> None:
    """Verify numerator and denominator use the same valid company set."""
    snapshots = [
        {"pe_ratio": 5.0, "pb_ratio": 1.0, "market_cap": 200.0, "estimated_net_income": 20.0, "equity": 100.0},
        {"pe_ratio": 3.0, "pb_ratio": 0.5, "market_cap": 0.0, "estimated_net_income": 10.0, "equity": 50.0},
    ]
    metrics = calculate_sector_metrics(snapshots)
    # pe_aggregate: company 2 has market_cap=0 → excluded
    assert metrics["pe_aggregate"] == 200.0 / 20.0
    # pb_aggregate: company 2 has market_cap=0 → excluded
    assert metrics["pb_aggregate"] == 200.0 / 100.0
    # roe_aggregate: both have equity>0
    assert metrics["roe_aggregate"] == (20.0 + 10.0) / (100.0 + 50.0)


def test_compare_company_to_sector() -> None:
    company = {"pe_ratio": 7.0, "pb_ratio": 1.0, "roe": 0.25, "estimated_net_income": 100, "equity": 400}
    sector = {"pe_median": 10.0, "pe_aggregate": 11.0, "pb_median": 1.4, "pb_aggregate": 1.5, "roe_aggregate": 0.2}
    comparison = compare_company_to_sector(company, sector)
    assert "fk_sektore_gore_iskontolu" in comparison["interpretation_flags"]
    assert "pd_dd_sektore_gore_iskontolu" in comparison["interpretation_flags"]
    assert "roe_sektor_ustu" in comparison["interpretation_flags"]

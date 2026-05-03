from __future__ import annotations

from pathlib import Path

from valuation.cache import (
    get_company_snapshot,
    get_sector_metrics,
    init_db,
    is_stale,
    save_valuation_result,
    upsert_company_snapshot,
    upsert_sector_metrics,
)


def test_cache_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    init_db(str(db))
    upsert_company_snapshot(
        str(db),
        {
            "symbol": "THYAO",
            "market": "BIST",
            "sector_index": "XULAS",
            "sector_name": "Ulastirma",
            "price": 300.0,
            "market_cap": 420_000_000_000.0,
            "shares_outstanding": 1_380_000_000.0,
            "paid_in_capital": 1_380_000_000.0,
            "pe_ratio": 4.2,
            "pb_ratio": 1.1,
            "roe": 0.24,
            "equity": 150_000_000_000.0,
            "net_income_latest_period": 20_000_000_000.0,
            "net_income_ttm": 32_000_000_000.0,
            "estimated_net_income": 30_000_000_000.0,
            "financial_period": "2025/06",
            "period_type": "interim",
            "source": "borsapy",
            "missing_fields_json": [],
        },
    )
    row = get_company_snapshot(str(db), "THYAO")
    assert row is not None
    assert row["symbol"] == "THYAO"
    assert row["source"] == "borsapy"

    upsert_sector_metrics(
        str(db),
        {
            "sector_index": "XULAS",
            "sector_name": "Ulastirma",
            "member_count": 10,
            "valid_member_count": 8,
            "pe_median": 7.5,
            "pe_aggregate": 8.2,
            "pb_median": 1.4,
            "pb_aggregate": 1.6,
            "roe_aggregate": 0.18,
        },
    )
    sector = get_sector_metrics(str(db), "XULAS")
    assert sector is not None
    assert sector["member_count"] == 10

    save_valuation_result(
        str(db),
        {
            "symbol": "THYAO",
            "valuation_date": "2026-05-03",
            "price": 300.0,
            "fair_value": 360.0,
            "upside_percent": 20.0,
            "target_prices_json": {"cari_fk": 340.0},
            "estimation_json": {"selected_method": "ttm"},
            "sector_comparison_json": {"flag": "ok"},
            "confidence_score": 0.9,
        },
    )
    assert is_stale(None) is True

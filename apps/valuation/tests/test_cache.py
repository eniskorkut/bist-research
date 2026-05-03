from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from valuation.cache import (
    evaluate_snapshot_quality,
    get_company_snapshot,
    get_sector_metrics,
    init_db,
    is_snapshot_usable,
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


def test_is_stale_none() -> None:
    assert is_stale(None) is True


def test_is_stale_invalid_string() -> None:
    assert is_stale("not-a-date") is True


def test_is_stale_fresh() -> None:
    recent = datetime.now(UTC).isoformat()
    assert is_stale(recent) is False


def test_is_stale_25_hours_ago() -> None:
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    assert is_stale(old) is True


def test_is_stale_23_hours_ago() -> None:
    recent = (datetime.now(UTC) - timedelta(hours=23)).isoformat()
    assert is_stale(recent) is False


def test_is_stale_custom_max_age() -> None:
    ts = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    assert is_stale(ts, max_age_hours=4) is True
    assert is_stale(ts, max_age_hours=6) is False


def test_fresh_snapshot_skipped_on_upsert(tmp_path: Path) -> None:
    """Simulate refresh_bist_cache skip logic: fresh snapshots should be
    detected via is_stale and not re-fetched."""
    db = str(tmp_path / "cache.sqlite")
    init_db(db)
    upsert_company_snapshot(
        db,
        {
            "symbol": "GARAN",
            "market": "BIST",
            "sector_index": "XBANK",
            "sector_name": "Banka",
            "price": 150.0,
            "market_cap": 200_000.0,
            "shares_outstanding": 1_000.0,
            "paid_in_capital": 1_000.0,
            "pe_ratio": 5.0,
            "pb_ratio": 1.2,
            "roe": 0.20,
            "equity": 160_000.0,
            "net_income_latest_period": 10_000.0,
            "net_income_ttm": 12_000.0,
            "estimated_net_income": 11_000.0,
            "financial_period": "2025/06",
            "period_type": "interim",
            "source": "borsapy",
            "missing_fields_json": [],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    existing = get_company_snapshot(db, "GARAN")
    assert existing is not None
    assert not is_stale(existing["updated_at"])  # should be fresh


def test_snapshot_quality_unusable_all_critical_missing() -> None:
    snapshot = {
        "price": None,
        "market_cap": None,
        "shares_outstanding": None,
        "paid_in_capital": None,
        "equity": None,
        "estimated_net_income": None,
    }
    status, errors = evaluate_snapshot_quality(snapshot)
    assert status == "unusable"
    assert "price_marketcap_shares_all_missing" in errors
    assert is_snapshot_usable(snapshot) is False


def test_snapshot_quality_partial_vs_usable() -> None:
    partial = {
        "price": 100.0,
        "market_cap": 1000.0,
        "shares_outstanding": 10.0,
        "paid_in_capital": 10.0,
        "equity": 500.0,
        "estimated_net_income": None,
    }
    status_partial, _ = evaluate_snapshot_quality(partial)
    assert status_partial == "partial"
    assert is_snapshot_usable(partial) is False

    usable = dict(partial)
    usable["estimated_net_income"] = 70.0
    status_usable, _ = evaluate_snapshot_quality(usable)
    assert status_usable == "usable"
    assert is_snapshot_usable(usable) is True

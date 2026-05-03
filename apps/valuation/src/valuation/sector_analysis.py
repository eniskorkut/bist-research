from __future__ import annotations

from statistics import median
from typing import Any

import borsapy as bp


def get_bist_sector_map() -> dict[str, str]:
    return {
        "XBANK": "Banka",
        "XUSIN": "Sanayi",
        "XUMAL": "Mali",
        "XHOLD": "Holding",
        "XUTEK": "Teknoloji",
        "XGIDA": "Gida",
        "XULAS": "Ulastirma",
        "XSGRT": "Sigorta",
        "XGMYO": "GYO",
        "XMADN": "Madencilik",
        "XELKT": "Elektrik",
        "XKMYA": "Kimya",
        "XTEKS": "Tekstil",
        "XTCRT": "Ticaret",
        "XTRZM": "Turizm",
    }


def get_sector_symbols(sector_index: str) -> list[str]:
    symbols = getattr(bp.Index(sector_index).component_symbols, "__iter__", None)
    if symbols is None:
        return []
    return sorted({str(s).strip().upper() for s in bp.Index(sector_index).component_symbols if s})


def get_sector_index_for_symbol(symbol: str) -> str | None:
    target = symbol.strip().upper()
    for sector_index in get_bist_sector_map():
        members = get_sector_symbols(sector_index)
        if target in members:
            return sector_index
    return None


def calculate_sector_metrics(company_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    pe_values = [s.get("pe_ratio") for s in company_snapshots if isinstance(s.get("pe_ratio"), (int, float)) and s.get("pe_ratio") > 0]
    pb_values = [s.get("pb_ratio") for s in company_snapshots if isinstance(s.get("pb_ratio"), (int, float)) and s.get("pb_ratio") > 0]

    # pe_aggregate: only companies with estimated_net_income > 0 AND market_cap > 0
    pe_valid = [s for s in company_snapshots if (s.get("estimated_net_income") or 0) > 0 and (s.get("market_cap") or 0) > 0]
    pe_market_cap_sum = sum(float(s["market_cap"]) for s in pe_valid)
    pe_income_sum = sum(float(s["estimated_net_income"]) for s in pe_valid)

    # pb_aggregate: only companies with equity > 0 AND market_cap > 0
    pb_valid = [s for s in company_snapshots if (s.get("equity") or 0) > 0 and (s.get("market_cap") or 0) > 0]
    pb_market_cap_sum = sum(float(s["market_cap"]) for s in pb_valid)
    pb_equity_sum = sum(float(s["equity"]) for s in pb_valid)

    # roe_aggregate: only companies with equity > 0
    roe_valid = [s for s in company_snapshots if (s.get("equity") or 0) > 0]
    roe_income_sum = sum(float(s.get("estimated_net_income") or 0) for s in roe_valid)
    roe_equity_sum = sum(float(s["equity"]) for s in roe_valid)

    return {
        "member_count": len(company_snapshots),
        "valid_member_count": len([s for s in company_snapshots if not s.get("missing_fields_json")]),
        "pe_median": median(pe_values) if pe_values else None,
        "pe_aggregate": (pe_market_cap_sum / pe_income_sum) if pe_income_sum > 0 else None,
        "pb_median": median(pb_values) if pb_values else None,
        "pb_aggregate": (pb_market_cap_sum / pb_equity_sum) if pb_equity_sum > 0 else None,
        "roe_aggregate": (roe_income_sum / roe_equity_sum) if roe_equity_sum > 0 else None,
    }


def compare_company_to_sector(company_snapshot: dict[str, Any], sector_metrics: dict[str, Any]) -> dict[str, Any]:
    company_pe = company_snapshot.get("pe_ratio")
    sector_pe_median = sector_metrics.get("pe_median")
    sector_pe_aggregate = sector_metrics.get("pe_aggregate")
    company_pb = company_snapshot.get("pb_ratio")
    sector_pb_median = sector_metrics.get("pb_median")
    sector_pb_aggregate = sector_metrics.get("pb_aggregate")
    company_roe = company_snapshot.get("roe")
    sector_roe_aggregate = sector_metrics.get("roe_aggregate")

    flags: list[str] = []
    if company_snapshot.get("estimated_net_income") is not None and company_snapshot.get("estimated_net_income") <= 0:
        flags.append("negatif_kar")
    if company_snapshot.get("equity") is not None and company_snapshot.get("equity") <= 0:
        flags.append("negatif_ozkaynak")

    pe_discount = None
    if isinstance(company_pe, (int, float)) and isinstance(sector_pe_median, (int, float)) and sector_pe_median > 0:
        pe_discount = ((sector_pe_median - company_pe) / sector_pe_median) * 100
        if pe_discount > 0:
            flags.append("fk_sektore_gore_iskontolu")

    pb_discount = None
    if isinstance(company_pb, (int, float)) and isinstance(sector_pb_median, (int, float)) and sector_pb_median > 0:
        pb_discount = ((sector_pb_median - company_pb) / sector_pb_median) * 100
        if pb_discount > 0:
            flags.append("pd_dd_sektore_gore_iskontolu")

    roe_vs_sector = None
    if isinstance(company_roe, (int, float)) and isinstance(sector_roe_aggregate, (int, float)) and sector_roe_aggregate != 0:
        roe_vs_sector = ((company_roe - sector_roe_aggregate) / abs(sector_roe_aggregate)) * 100
        if roe_vs_sector > 0:
            flags.append("roe_sektor_ustu")

    if any(company_snapshot.get(k) is None for k in ["pe_ratio", "pb_ratio", "roe"]):
        flags.append("veri_eksik")

    return {
        "company_pe": company_pe,
        "sector_pe_median": sector_pe_median,
        "sector_pe_aggregate": sector_pe_aggregate,
        "pe_discount_to_sector_median_pct": pe_discount,
        "company_pb": company_pb,
        "sector_pb_median": sector_pb_median,
        "sector_pb_aggregate": sector_pb_aggregate,
        "pb_discount_to_sector_median_pct": pb_discount,
        "company_roe": company_roe,
        "sector_roe_aggregate": sector_roe_aggregate,
        "roe_vs_sector_pct": roe_vs_sector,
        "interpretation_flags": sorted(set(flags)),
    }

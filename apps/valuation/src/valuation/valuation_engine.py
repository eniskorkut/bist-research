from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC, datetime

from valuation.cache import evaluate_snapshot_quality, save_valuation_result
from valuation.data_access import BorsapyFinancialClient
from valuation.profit_estimator import ProfitEstimationResult, estimate_net_income_auto
from valuation.sector_analysis import compare_company_to_sector


@dataclass
class ValuationResult:
    symbol: str
    price: float | None
    shares_outstanding: float | None
    paid_in_capital: float | None
    estimated_net_income: float | None
    equity: float | None
    market_cap: float | None
    pe_ratio: float | None
    pb_ratio: float | None
    target_prices: dict[str, float | None]
    sector_target_prices: dict[str, float | None]
    average_target_price: float | None
    average_with_sector_price: float | None
    upside_potential_pct: float | None
    estimation: ProfitEstimationResult
    sector_comparison: dict[str, float | str | list[str] | None]
    source: str
    missing_fields: list[str]


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def run_valuation(
    symbol: str,
    client: BorsapyFinancialClient | None = None,
    sector_metrics: dict | None = None,
    db_path: str | None = None,
) -> ValuationResult:
    financial_client = client or BorsapyFinancialClient()
    snapshot = financial_client.load_snapshot(symbol)
    estimation = estimate_net_income_auto(symbol=symbol, client=financial_client)

    estimated_net_income = estimation.estimated_net_income
    shares = snapshot.shares_outstanding
    price = snapshot.price
    market_cap = snapshot.market_cap
    equity = snapshot.equity
    pe = snapshot.pe_ratio
    pb = snapshot.pb_ratio
    paid_in_capital = snapshot.paid_in_capital

    price_by_pe = _safe_div(
        (estimated_net_income * pe) if estimated_net_income is not None and estimated_net_income > 0 and pe is not None and pe > 0 else None,
        shares,
    )
    price_by_pb = _safe_div(
        (equity * pb) if equity is not None and equity > 0 and pb is not None and pb > 0 else None,
        shares,
    )

    # Paid-in capital method: estimate fair value from market-cap to paid-cap ratio carried to estimated profitability.
    capital_ratio = _safe_div(market_cap, paid_in_capital)
    price_by_paid_in_capital = _safe_div(
        (estimated_net_income * capital_ratio)
        if estimated_net_income is not None and estimated_net_income > 0 and capital_ratio is not None and capital_ratio > 0
        else None,
        shares,
    )

    # Potential market value: use average of valid implied market caps from PE/PB methods.
    implied_caps = []
    if estimated_net_income is not None and estimated_net_income > 0 and pe is not None and pe > 0:
        implied_caps.append(estimated_net_income * pe)
    if equity is not None and equity > 0 and pb is not None and pb > 0:
        implied_caps.append(equity * pb)
    potential_market_cap = sum(implied_caps) / len(implied_caps) if implied_caps else None
    price_by_potential_market_value = _safe_div(potential_market_cap, shares)

    # ROE method: fair price = (estimated ROE * equity * current P/B) / shares.
    estimated_roe = _safe_div(estimated_net_income, equity) if (estimated_net_income or 0) > 0 and (equity or 0) > 0 else None
    price_by_roe = _safe_div(
        (estimated_roe * equity * pb) if estimated_roe is not None and equity is not None and pb is not None else None,
        shares,
    )

    target_prices = {
        "cari_fk": price_by_pe,
        "pd_dd": price_by_pb,
        "odenmis_sermaye": price_by_paid_in_capital,
        "potansiyel_piyasa_degeri": price_by_potential_market_value,
        "ozsermaye_karliligi": price_by_roe,
    }
    sector_target_prices = {"sektor_fk_hedef": None, "sektor_pd_dd_hedef": None}
    sector_comparison: dict[str, float | str | list[str] | None] = {}
    if sector_metrics:
        sector_pe = sector_metrics.get("pe_median") or sector_metrics.get("pe_aggregate")
        sector_pb = sector_metrics.get("pb_median") or sector_metrics.get("pb_aggregate")
        estimated_eps = _safe_div(estimated_net_income, shares)
        book_value_per_share = _safe_div(equity, shares)
        if estimated_eps is not None and sector_pe and sector_pe > 0 and estimated_eps > 0:
            sector_target_prices["sektor_fk_hedef"] = estimated_eps * float(sector_pe)
        if book_value_per_share is not None and sector_pb and sector_pb > 0 and book_value_per_share > 0:
            sector_target_prices["sektor_pd_dd_hedef"] = book_value_per_share * float(sector_pb)

        company_snapshot = {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "roe": estimated_roe,
            "estimated_net_income": estimated_net_income,
            "equity": equity,
        }
        sector_comparison = compare_company_to_sector(company_snapshot, sector_metrics)

    valid_targets = [value for value in target_prices.values() if value is not None]
    average_target_price = sum(valid_targets) / len(valid_targets) if valid_targets else None
    all_targets = valid_targets + [value for value in sector_target_prices.values() if value is not None]
    average_with_sector_price = sum(all_targets) / len(all_targets) if all_targets else None
    upside_potential_pct = (
        ((average_target_price - price) / price) * 100
        if average_target_price is not None and price not in (None, 0)
        else None
    )

    missing = list(snapshot.missing_fields)
    if estimated_net_income is None:
        missing.append("estimated_net_income")
    result = ValuationResult(
        symbol=snapshot.symbol,
        price=price,
        shares_outstanding=shares,
        paid_in_capital=paid_in_capital,
        estimated_net_income=estimated_net_income,
        equity=equity,
        market_cap=market_cap,
        pe_ratio=pe,
        pb_ratio=pb,
        target_prices=target_prices,
        sector_target_prices=sector_target_prices,
        average_target_price=average_target_price,
        average_with_sector_price=average_with_sector_price,
        upside_potential_pct=upside_potential_pct,
        estimation=estimation,
        sector_comparison=sector_comparison,
        source=snapshot.source,
        missing_fields=missing,
    )
    if db_path and quality_status != "unusable":
        save_valuation_result(
            db_path,
            {
                "symbol": result.symbol,
                "valuation_date": datetime.now(UTC).date().isoformat(),
                "price": result.price,
                "fair_value": result.average_target_price,
                "upside_percent": result.upside_potential_pct,
                "target_prices_json": {**result.target_prices, **result.sector_target_prices},
                "estimation_json": asdict(result.estimation),
                "sector_comparison_json": result.sector_comparison,
                "confidence_score": max(0.0, 1.0 - (len(set(result.missing_fields)) / 12)),
            },
        )
    return result


def run_valuation_from_snapshot(
    snapshot: dict,
    sector_metrics: dict | None = None,
    db_path: str | None = None,
) -> ValuationResult:
    """Run valuation using a cached company_snapshot dict.

    This function never calls borsapy.  All numeric inputs come from the
    *snapshot* dictionary (as stored in the ``company_snapshot`` SQLite table).
    """
    symbol = snapshot["symbol"]
    quality_status = snapshot.get("data_quality_status")
    quality_errors = snapshot.get("data_quality_errors_json") or []
    if not quality_status:
        quality_status, quality_errors = evaluate_snapshot_quality(snapshot)
    estimated_net_income = snapshot.get("estimated_net_income")
    shares = snapshot.get("shares_outstanding")
    price = snapshot.get("price")
    market_cap = snapshot.get("market_cap")
    equity = snapshot.get("equity")
    pe = snapshot.get("pe_ratio")
    pb = snapshot.get("pb_ratio")
    paid_in_capital = snapshot.get("paid_in_capital")

    # Build a lightweight ProfitEstimationResult from the cached values.
    missing_fields_raw = snapshot.get("missing_fields_json") or []
    if isinstance(missing_fields_raw, str):
        import json
        missing_fields_raw = json.loads(missing_fields_raw)
    if isinstance(quality_errors, str):
        import json
        quality_errors = json.loads(quality_errors)

    estimation = ProfitEstimationResult(
        symbol=symbol,
        selected_method="cache",
        estimated_net_income=estimated_net_income,
        method_values={},
        period_type=snapshot.get("period_type") or "unknown",
        period_label=snapshot.get("financial_period") or "unknown",
        missing_fields=list(missing_fields_raw),
    )

    # ---------- target prices (same formulas as run_valuation) ----------
    price_by_pe = _safe_div(
        (estimated_net_income * pe) if estimated_net_income is not None and estimated_net_income > 0 and pe is not None and pe > 0 else None,
        shares,
    )
    price_by_pb = _safe_div(
        (equity * pb) if equity is not None and equity > 0 and pb is not None and pb > 0 else None,
        shares,
    )

    capital_ratio = _safe_div(market_cap, paid_in_capital)
    price_by_paid_in_capital = _safe_div(
        (estimated_net_income * capital_ratio)
        if estimated_net_income is not None and estimated_net_income > 0 and capital_ratio is not None and capital_ratio > 0
        else None,
        shares,
    )

    implied_caps = []
    if estimated_net_income is not None and estimated_net_income > 0 and pe is not None and pe > 0:
        implied_caps.append(estimated_net_income * pe)
    if equity is not None and equity > 0 and pb is not None and pb > 0:
        implied_caps.append(equity * pb)
    potential_market_cap = sum(implied_caps) / len(implied_caps) if implied_caps else None
    price_by_potential_market_value = _safe_div(potential_market_cap, shares)

    estimated_roe = _safe_div(estimated_net_income, equity) if (estimated_net_income or 0) > 0 and (equity or 0) > 0 else None
    price_by_roe = _safe_div(
        (estimated_roe * equity * pb) if estimated_roe is not None and equity is not None and pb is not None else None,
        shares,
    )

    target_prices = {
        "cari_fk": price_by_pe,
        "pd_dd": price_by_pb,
        "odenmis_sermaye": price_by_paid_in_capital,
        "potansiyel_piyasa_degeri": price_by_potential_market_value,
        "ozsermaye_karliligi": price_by_roe,
    }

    # ---------- sector target prices ----------
    sector_target_prices = {"sektor_fk_hedef": None, "sektor_pd_dd_hedef": None}
    sector_comparison: dict[str, float | str | list[str] | None] = {}
    if sector_metrics:
        sector_pe = sector_metrics.get("pe_median") or sector_metrics.get("pe_aggregate")
        sector_pb = sector_metrics.get("pb_median") or sector_metrics.get("pb_aggregate")
        estimated_eps = _safe_div(estimated_net_income, shares)
        book_value_per_share = _safe_div(equity, shares)
        if estimated_eps is not None and sector_pe and sector_pe > 0 and estimated_eps > 0:
            sector_target_prices["sektor_fk_hedef"] = estimated_eps * float(sector_pe)
        if book_value_per_share is not None and sector_pb and sector_pb > 0 and book_value_per_share > 0:
            sector_target_prices["sektor_pd_dd_hedef"] = book_value_per_share * float(sector_pb)

        company_snapshot_for_compare = {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "roe": estimated_roe,
            "estimated_net_income": estimated_net_income,
            "equity": equity,
        }
        sector_comparison = compare_company_to_sector(company_snapshot_for_compare, sector_metrics)

    # ---------- averages & upside ----------
    valid_targets = [value for value in target_prices.values() if value is not None]
    average_target_price = sum(valid_targets) / len(valid_targets) if valid_targets else None
    all_targets = valid_targets + [value for value in sector_target_prices.values() if value is not None]
    average_with_sector_price = sum(all_targets) / len(all_targets) if all_targets else None
    upside_potential_pct = (
        ((average_target_price - price) / price) * 100
        if average_target_price is not None and price not in (None, 0)
        else None
    )

    missing = list(missing_fields_raw)
    if estimated_net_income is None and "estimated_net_income" not in missing:
        missing.append("estimated_net_income")

    if quality_status == "unusable":
        missing = sorted(set(list(missing_fields_raw) + list(quality_errors) + ["snapshot_unusable"]))
    result = ValuationResult(
        symbol=symbol,
        price=price,
        shares_outstanding=shares,
        paid_in_capital=paid_in_capital,
        estimated_net_income=estimated_net_income,
        equity=equity,
        market_cap=market_cap,
        pe_ratio=pe,
        pb_ratio=pb,
        target_prices=target_prices,
        sector_target_prices=sector_target_prices,
        average_target_price=average_target_price,
        average_with_sector_price=average_with_sector_price,
        upside_potential_pct=upside_potential_pct,
        estimation=estimation,
        sector_comparison=sector_comparison,
        source="cache",
        missing_fields=missing,
    )
    if db_path:
        save_valuation_result(
            db_path,
            {
                "symbol": result.symbol,
                "valuation_date": datetime.now(UTC).date().isoformat(),
                "price": result.price,
                "fair_value": result.average_target_price,
                "upside_percent": result.upside_potential_pct,
                "target_prices_json": {**result.target_prices, **result.sector_target_prices},
                "estimation_json": asdict(result.estimation),
                "sector_comparison_json": result.sector_comparison,
                "confidence_score": max(0.0, 1.0 - (len(set(result.missing_fields)) / 12)),
            },
        )
    return result

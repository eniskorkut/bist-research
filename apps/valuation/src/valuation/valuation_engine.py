from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from statistics import median

from valuation.cache import evaluate_snapshot_quality, save_valuation_result
from valuation.data_access import BorsapyFinancialClient
from valuation.historical_multiples import get_historical_pe_median
from valuation.profit_estimator import ProfitEstimationResult, estimate_net_income_auto
from valuation.sector_analysis import compare_company_to_sector


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _filtered_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 4:
        return _mean(values)
    s = sorted(values)
    q1 = s[len(s) // 4]
    q3 = s[(len(s) * 3) // 4]
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [x for x in s if low <= x <= high]
    return _mean(filtered) if filtered else _mean(values)


@dataclass
class ScenarioValuation:
    scenario_key: str
    period_label: str
    net_income: float | None
    net_income_source: str
    historical_pe_median: float | None
    target_prices: dict[str, float | None]
    included_methods: list[str]
    excluded_methods: list[str]
    fair_value_median: float | None
    fair_value_mean_filtered: float | None
    upside_potential_pct: float | None
    paid_capital_details: dict[str, float | None | str]


@dataclass
class ValuationResult:
    symbol: str
    price: float | None
    shares_outstanding: float | None
    paid_in_capital: float | None
    equity: float | None
    market_cap: float | None
    pe_ratio: float | None
    pb_ratio: float | None
    valuation_scenarios: dict[str, ScenarioValuation]
    estimation: ProfitEstimationResult
    sector_comparison: dict[str, float | str | list[str] | None]
    source: str
    net_income_source: str
    equity_source: str
    data_quality_status: str
    missing_fields: list[str]


def _build_scenario(
    *,
    scenario_key: str,
    period_label: str,
    net_income: float | None,
    net_income_source: str,
    price: float | None,
    shares: float | None,
    paid_in_capital: float | None,
    equity: float | None,
    pe: float | None,
    pb: float | None,
    historical_pe_median: float | None,
) -> ScenarioValuation:
    eps = _safe_div(net_income, shares)
    paid_cap_eps = _safe_div(net_income, paid_in_capital)

    cari_fk = _safe_div((net_income * pe) if net_income is not None and net_income > 0 and (pe or 0) > 0 else None, shares)
    pd_dd = _safe_div((equity * pb) if (equity or 0) > 0 and (pb or 0) > 0 else None, shares)

    odenmis_sermaye_x10 = (paid_cap_eps * 10) if paid_cap_eps is not None and paid_cap_eps > 0 else None
    odenmis_sermaye_historical_pe = (
        paid_cap_eps * historical_pe_median
        if paid_cap_eps is not None and paid_cap_eps > 0 and (historical_pe_median or 0) > 0
        else None
    )
    paid_cap_values = [x for x in [odenmis_sermaye_x10, odenmis_sermaye_historical_pe] if x is not None]
    odenmis_sermaye_final = _mean(paid_cap_values) if paid_cap_values else None

    implied_caps = []
    if net_income is not None and net_income > 0 and (pe or 0) > 0:
        implied_caps.append(net_income * pe)
    if (equity or 0) > 0 and (pb or 0) > 0:
        implied_caps.append(equity * pb)
    potansiyel_piyasa_degeri = _safe_div(_mean(implied_caps), shares)

    estimated_roe = _safe_div(net_income, equity) if (net_income or 0) > 0 and (equity or 0) > 0 else None
    ozsermaye_karliligi = _safe_div(
        (estimated_roe * equity * pb) if estimated_roe is not None and (pb or 0) > 0 and (equity or 0) > 0 else None,
        shares,
    )

    target_prices = {
        "cari_fk": cari_fk,
        "pd_dd": pd_dd,
        "odenmis_sermaye_x10": odenmis_sermaye_x10,
        "odenmis_sermaye_historical_pe": odenmis_sermaye_historical_pe,
        "odenmis_sermaye_final": odenmis_sermaye_final,
        "potansiyel_piyasa_degeri": potansiyel_piyasa_degeri,
        "ozsermaye_karliligi": ozsermaye_karliligi,
    }
    included_methods = [k for k, v in target_prices.items() if v is not None and k != "odenmis_sermaye_x10"]
    excluded_methods = [k for k, v in target_prices.items() if v is None or k == "odenmis_sermaye_x10"]
    fair_values = [target_prices[k] for k in included_methods if target_prices[k] is not None]
    fair_value_median = median(fair_values) if fair_values else None
    fair_value_mean_filtered = _filtered_mean([float(v) for v in fair_values if v is not None])
    upside_potential_pct = (
        ((fair_value_median - price) / price) * 100 if fair_value_median is not None and price not in (None, 0) else None
    )

    return ScenarioValuation(
        scenario_key=scenario_key,
        period_label=period_label,
        net_income=net_income,
        net_income_source=net_income_source,
        historical_pe_median=historical_pe_median,
        target_prices=target_prices,
        included_methods=included_methods,
        excluded_methods=excluded_methods,
        fair_value_median=fair_value_median,
        fair_value_mean_filtered=fair_value_mean_filtered,
        upside_potential_pct=upside_potential_pct,
        paid_capital_details={
            "eps": paid_cap_eps,
            "x10": odenmis_sermaye_x10,
            "historical_pe_median": historical_pe_median,
            "historical_pe_value": odenmis_sermaye_historical_pe,
            "final": odenmis_sermaye_final,
            "final_method": "x10_only" if odenmis_sermaye_historical_pe is None and odenmis_sermaye_x10 is not None else "average_x10_historical_pe",
        },
    )


def run_valuation(
    symbol: str,
    client: BorsapyFinancialClient | None = None,
    sector_metrics: dict | None = None,
    db_path: str | None = None,
) -> ValuationResult:
    financial_client = client or BorsapyFinancialClient()
    snapshot = financial_client.load_snapshot(symbol)
    estimation = estimate_net_income_auto(symbol=symbol, client=financial_client)
    historical_pe_median = get_historical_pe_median(symbol)

    ttm_scenario = _build_scenario(
        scenario_key="ttm",
        period_label="Son 12 Ay / TTM",
        net_income=snapshot.net_income_ttm,
        net_income_source=snapshot.net_income_source,
        price=snapshot.price,
        shares=snapshot.shares_outstanding,
        paid_in_capital=snapshot.paid_in_capital,
        equity=snapshot.equity,
        pe=snapshot.pe_ratio,
        pb=snapshot.pb_ratio,
        historical_pe_median=historical_pe_median,
    )
    year_end_scenario = _build_scenario(
        scenario_key="year_end",
        period_label=snapshot.period_label or "Yil Sonu Tahmini",
        net_income=estimation.estimated_net_income,
        net_income_source=estimation.selected_method,
        price=snapshot.price,
        shares=snapshot.shares_outstanding,
        paid_in_capital=snapshot.paid_in_capital,
        equity=snapshot.equity,
        pe=snapshot.pe_ratio,
        pb=snapshot.pb_ratio,
        historical_pe_median=historical_pe_median,
    )

    estimated_roe = _safe_div(estimation.estimated_net_income, snapshot.equity) if (estimation.estimated_net_income or 0) > 0 and (snapshot.equity or 0) > 0 else None
    sector_comparison = compare_company_to_sector(
        {
            "pe_ratio": snapshot.pe_ratio,
            "pb_ratio": snapshot.pb_ratio,
            "roe": estimated_roe,
            "estimated_net_income": estimation.estimated_net_income,
            "equity": snapshot.equity,
        },
        sector_metrics or {},
    ) if sector_metrics else {}

    missing = list(snapshot.missing_fields)
    quality_status, _ = evaluate_snapshot_quality(
        {
            "price": snapshot.price,
            "market_cap": snapshot.market_cap,
            "shares_outstanding": snapshot.shares_outstanding,
            "paid_in_capital": snapshot.paid_in_capital,
            "equity": snapshot.equity,
            "estimated_net_income": estimation.estimated_net_income,
        }
    )

    result = ValuationResult(
        symbol=snapshot.symbol,
        price=snapshot.price,
        shares_outstanding=snapshot.shares_outstanding,
        paid_in_capital=snapshot.paid_in_capital,
        equity=snapshot.equity,
        market_cap=snapshot.market_cap,
        pe_ratio=snapshot.pe_ratio,
        pb_ratio=snapshot.pb_ratio,
        valuation_scenarios={"ttm": ttm_scenario, "year_end": year_end_scenario},
        estimation=estimation,
        sector_comparison=sector_comparison,
        source=snapshot.source,
        net_income_source=snapshot.net_income_source,
        equity_source=snapshot.equity_source,
        data_quality_status=quality_status,
        missing_fields=missing,
    )
    if db_path:
        save_valuation_result(
            db_path,
            {
                "symbol": result.symbol,
                "valuation_date": datetime.now(UTC).date().isoformat(),
                "price": result.price,
                "fair_value": result.valuation_scenarios["year_end"].fair_value_median,
                "upside_percent": result.valuation_scenarios["year_end"].upside_potential_pct,
                "target_prices_json": {
                    "ttm": result.valuation_scenarios["ttm"].target_prices,
                    "year_end": result.valuation_scenarios["year_end"].target_prices,
                },
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
    symbol = snapshot["symbol"]
    quality_status, quality_errors = evaluate_snapshot_quality(snapshot)
    historical_pe_median = get_historical_pe_median(symbol)

    estimation = ProfitEstimationResult(
        symbol=symbol,
        selected_method="cache",
        estimated_net_income=snapshot.get("estimated_net_income"),
        method_values={},
        period_type=snapshot.get("period_type") or "unknown",
        period_label=snapshot.get("financial_period") or "unknown",
        missing_fields=list(snapshot.get("missing_fields_json") or []),
    )

    ttm_scenario = _build_scenario(
        scenario_key="ttm",
        period_label="Son 12 Ay / TTM",
        net_income=snapshot.get("net_income_ttm"),
        net_income_source=snapshot.get("net_income_source", "unknown"),
        price=snapshot.get("price"),
        shares=snapshot.get("shares_outstanding"),
        paid_in_capital=snapshot.get("paid_in_capital"),
        equity=snapshot.get("equity"),
        pe=snapshot.get("pe_ratio"),
        pb=snapshot.get("pb_ratio"),
        historical_pe_median=historical_pe_median,
    )
    year_end_scenario = _build_scenario(
        scenario_key="year_end",
        period_label=snapshot.get("financial_period") or "Yil Sonu Tahmini",
        net_income=snapshot.get("estimated_net_income"),
        net_income_source=snapshot.get("net_income_source", "unknown"),
        price=snapshot.get("price"),
        shares=snapshot.get("shares_outstanding"),
        paid_in_capital=snapshot.get("paid_in_capital"),
        equity=snapshot.get("equity"),
        pe=snapshot.get("pe_ratio"),
        pb=snapshot.get("pb_ratio"),
        historical_pe_median=historical_pe_median,
    )

    estimated_roe = _safe_div(snapshot.get("estimated_net_income"), snapshot.get("equity")) if (snapshot.get("estimated_net_income") or 0) > 0 and (snapshot.get("equity") or 0) > 0 else None
    sector_comparison = compare_company_to_sector(
        {
            "pe_ratio": snapshot.get("pe_ratio"),
            "pb_ratio": snapshot.get("pb_ratio"),
            "roe": estimated_roe,
            "estimated_net_income": snapshot.get("estimated_net_income"),
            "equity": snapshot.get("equity"),
        },
        sector_metrics or {},
    ) if sector_metrics else {}

    missing = list(snapshot.get("missing_fields_json") or [])
    if quality_status == "unusable":
        missing = sorted(set(missing + list(quality_errors) + ["snapshot_unusable"]))

    result = ValuationResult(
        symbol=symbol,
        price=snapshot.get("price"),
        shares_outstanding=snapshot.get("shares_outstanding"),
        paid_in_capital=snapshot.get("paid_in_capital"),
        equity=snapshot.get("equity"),
        market_cap=snapshot.get("market_cap"),
        pe_ratio=snapshot.get("pe_ratio"),
        pb_ratio=snapshot.get("pb_ratio"),
        valuation_scenarios={"ttm": ttm_scenario, "year_end": year_end_scenario},
        estimation=estimation,
        sector_comparison=sector_comparison,
        source="cache",
        net_income_source=snapshot.get("net_income_source", "unknown"),
        equity_source=snapshot.get("equity_source", "unknown"),
        data_quality_status=quality_status,
        missing_fields=missing,
    )
    if db_path and quality_status != "unusable":
        save_valuation_result(
            db_path,
            {
                "symbol": result.symbol,
                "valuation_date": datetime.now(UTC).date().isoformat(),
                "price": result.price,
                "fair_value": result.valuation_scenarios["year_end"].fair_value_median,
                "upside_percent": result.valuation_scenarios["year_end"].upside_potential_pct,
                "target_prices_json": {
                    "ttm": result.valuation_scenarios["ttm"].target_prices,
                    "year_end": result.valuation_scenarios["year_end"].target_prices,
                },
                "estimation_json": asdict(result.estimation),
                "sector_comparison_json": result.sector_comparison,
                "confidence_score": max(0.0, 1.0 - (len(set(result.missing_fields)) / 12)),
            },
        )
    return result

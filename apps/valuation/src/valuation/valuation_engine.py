from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from statistics import median

from valuation.cache import evaluate_snapshot_quality, save_valuation_result
from valuation.data_access import BorsapyFinancialClient
from valuation.historical_multiples import get_historical_pe_median
from valuation.profit_estimator import (
    ProfitEstimationResult,
    estimate_net_income_from_snapshot,
)
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
    method_notes: dict[str, str]
    method_types: dict[str, str]
    valuation_status: str


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
    ratio_sources: dict[str, str]


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
    pe_source: str,
    pb_source: str,
    historical_pe_median: float | None,
    sector_pe_median: float | None,
    disable_reason: str | None = None,
) -> ScenarioValuation:
    all_methods = [
        "cari_fk",
        "pd_dd",
        "odenmis_sermaye_x10",
        "odenmis_sermaye_historical_pe",
        "odenmis_sermaye_sector_pe",
        "odenmis_sermaye_current_pe",
        "odenmis_sermaye_final",
        "potansiyel_piyasa_degeri",
        "ozsermaye_karliligi",
    ]
    if disable_reason is not None:
        target_prices = {method: None for method in all_methods}
        return ScenarioValuation(
            scenario_key=scenario_key,
            period_label=period_label,
            net_income=net_income,
            net_income_source=net_income_source,
            historical_pe_median=historical_pe_median,
            target_prices=target_prices,
            included_methods=[],
            excluded_methods=list(all_methods),
            fair_value_median=None,
            fair_value_mean_filtered=None,
            upside_potential_pct=None,
            paid_capital_details={
                "eps": None,
                "x10": None,
                "historical_pe_median": historical_pe_median,
                "historical_pe_value": None,
                "sector_pe_median": sector_pe_median,
                "sector_pe_value": None,
                "current_pe_value": None,
                "final": None,
                "final_method": "skipped",
                "included_in_fair_value": False,
            },
            method_notes={method: disable_reason for method in all_methods},
            method_types={method: "independent_target" for method in all_methods},
            valuation_status=disable_reason,
        )

    eps = _safe_div(net_income, shares)
    paid_cap_eps = _safe_div(net_income, paid_in_capital)

    method_notes: dict[str, str] = {}
    method_types: dict[str, str] = {}
    if net_income is not None and net_income <= 0:
        method_notes["cari_fk"] = "negative_net_income"
    cari_fk = _safe_div((net_income * pe) if net_income is not None and net_income > 0 and (pe or 0) > 0 else None, shares)
    if pe in (None, 0):
        method_notes["cari_fk"] = "missing_multiplier"
    pd_dd = _safe_div((equity * pb) if (equity or 0) > 0 and (pb or 0) > 0 else None, shares)
    if (equity or 0) <= 0:
        method_notes["pd_dd"] = "missing_equity"
    elif pb in (None, 0):
        method_notes["pd_dd"] = "missing_multiplier"
    method_types["pd_dd"] = "current_implied" if pb_source == "derived" else "independent_target"

    odenmis_sermaye_x10 = (paid_cap_eps * 10) if paid_cap_eps is not None and paid_cap_eps > 0 else None
    odenmis_sermaye_historical_pe = (
        paid_cap_eps * historical_pe_median
        if paid_cap_eps is not None and paid_cap_eps > 0 and (historical_pe_median or 0) > 0
        else None
    )
    odenmis_sermaye_sector_pe = (
        paid_cap_eps * sector_pe_median
        if paid_cap_eps is not None and paid_cap_eps > 0 and (sector_pe_median or 0) > 0
        else None
    )
    odenmis_sermaye_current_pe = (
        paid_cap_eps * pe if paid_cap_eps is not None and paid_cap_eps > 0 and (pe or 0) > 0 else None
    )
    odenmis_sermaye_final = None
    paid_cap_method = "info_only_x10_no_multiplier"
    if odenmis_sermaye_historical_pe is not None and odenmis_sermaye_x10 is not None:
        odenmis_sermaye_final = _mean([odenmis_sermaye_x10, odenmis_sermaye_historical_pe])
        paid_cap_method = "average_x10_historical_pe"
    elif odenmis_sermaye_sector_pe is not None and odenmis_sermaye_x10 is not None:
        odenmis_sermaye_final = _mean([odenmis_sermaye_x10, odenmis_sermaye_sector_pe])
        paid_cap_method = "average_x10_sector_pe"
    elif odenmis_sermaye_current_pe is not None:
        odenmis_sermaye_final = odenmis_sermaye_current_pe
        paid_cap_method = "current_pe_only"

    if net_income is not None and net_income <= 0:
        for m in [
            "odenmis_sermaye_x10",
            "odenmis_sermaye_historical_pe",
            "odenmis_sermaye_sector_pe",
            "odenmis_sermaye_current_pe",
            "odenmis_sermaye_final",
            "ozsermaye_karliligi",
        ]:
            method_notes[m] = "negative_net_income"
    implied_caps: list[float] = []
    pe_cap_added = False
    pb_cap_added = False
    if net_income is not None and net_income > 0 and (pe or 0) > 0:
        implied_caps.append(net_income * pe)
        pe_cap_added = True
    if (equity or 0) > 0 and (pb or 0) > 0:
        implied_caps.append(equity * pb)
        pb_cap_added = True
    potansiyel_piyasa_degeri = _safe_div(_mean(implied_caps), shares)
    if not implied_caps:
        method_notes["potansiyel_piyasa_degeri"] = "missing_multiplier"
    if pe_cap_added or pb_cap_added:
        pe_is_independent = pe_cap_added and pe_source == "borsapy"
        pb_is_independent = pb_cap_added and pb_source == "borsapy"
        method_types["potansiyel_piyasa_degeri"] = (
            "independent_target" if (pe_is_independent or pb_is_independent) else "current_implied"
        )

    estimated_roe = _safe_div(net_income, equity) if (net_income or 0) > 0 and (equity or 0) > 0 else None
    ozsermaye_karliligi = _safe_div(
        (estimated_roe * equity * pb) if estimated_roe is not None and (pb or 0) > 0 and (equity or 0) > 0 else None,
        shares,
    )
    if (equity or 0) <= 0:
        method_notes["ozsermaye_karliligi"] = "missing_equity"
    elif pb in (None, 0):
        method_notes["ozsermaye_karliligi"] = "missing_multiplier"

    target_prices = {
        "cari_fk": cari_fk,
        "pd_dd": pd_dd,
        "odenmis_sermaye_x10": odenmis_sermaye_x10,
        "odenmis_sermaye_historical_pe": odenmis_sermaye_historical_pe,
        "odenmis_sermaye_sector_pe": odenmis_sermaye_sector_pe,
        "odenmis_sermaye_current_pe": odenmis_sermaye_current_pe,
        "odenmis_sermaye_final": odenmis_sermaye_final,
        "potansiyel_piyasa_degeri": potansiyel_piyasa_degeri,
        "ozsermaye_karliligi": ozsermaye_karliligi,
    }
    fair_values: list[float] = []
    included_methods: list[str] = []
    excluded_methods: list[str] = []
    for method, value in target_prices.items():
        if value is None:
            excluded_methods.append(method)
            continue
        if method_types.get(method) == "current_implied":
            excluded_methods.append(method)
            method_notes.setdefault(method, "info_only")
            continue
        if method in {"odenmis_sermaye_x10", "odenmis_sermaye_historical_pe", "odenmis_sermaye_sector_pe", "odenmis_sermaye_current_pe"}:
            excluded_methods.append(method)
            method_notes.setdefault(method, "info_only")
            continue
        if value <= 0:
            excluded_methods.append(method)
            continue
        if price not in (None, 0):
            if value > price * 5 or value < price * 0.2:
                excluded_methods.append(method)
                continue
        included_methods.append(method)
        fair_values.append(float(value))
        method_types.setdefault(method, "independent_target")
        method_notes.setdefault(method, "ok")

    independent_included = [m for m in included_methods if method_types.get(m) == "independent_target"]
    valuation_status = "full"
    if len(independent_included) < 1 or len(included_methods) < 2:
        fair_values = []
        fair_value_median = None
        fair_value_mean_filtered = None
        upside_potential_pct = None
        valuation_status = "insufficient_independent_methods"
    else:
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
            "sector_pe_median": sector_pe_median,
            "sector_pe_value": odenmis_sermaye_sector_pe,
            "current_pe_value": odenmis_sermaye_current_pe,
            "final": odenmis_sermaye_final,
            "final_method": paid_cap_method,
            "included_in_fair_value": odenmis_sermaye_final is not None,
        },
        method_notes=method_notes,
        method_types=method_types,
        valuation_status=valuation_status,
    )


def _resolve_pb_ratio(market_cap: float | None, equity: float | None, pb: float | None) -> tuple[float | None, str]:
    pb_ratio = pb
    pb_source = "borsapy" if pb_ratio is not None else "missing"
    if pb_ratio is None and (market_cap or 0) > 0 and (equity or 0) > 0:
        pb_ratio = market_cap / equity
        pb_source = "derived"
    return pb_ratio, pb_source


def _resolve_pe_ratio(market_cap: float | None, income: float | None, pe: float | None) -> tuple[float | None, str]:
    if pe is not None:
        return pe, "borsapy"
    if income is None:
        return None, "missing"
    if income <= 0:
        return None, "not_applicable_negative_income"
    if (market_cap or 0) > 0:
        return market_cap / income, "derived"
    return None, "missing"


def run_valuation(
    symbol: str,
    client: BorsapyFinancialClient | None = None,
    sector_metrics: dict | None = None,
    db_path: str | None = None,
) -> ValuationResult:
    financial_client = client or BorsapyFinancialClient()
    snapshot = financial_client.load_snapshot(symbol)
    estimation = estimate_net_income_from_snapshot(snapshot)
    historical_pe_median = get_historical_pe_median(symbol)
    sector_pe_median = (sector_metrics or {}).get("pe_median")
    resolved_pb, pb_source = _resolve_pb_ratio(snapshot.market_cap, snapshot.equity, snapshot.pb_ratio)
    ttm_pe, ttm_pe_source = _resolve_pe_ratio(snapshot.market_cap, snapshot.net_income_ttm, snapshot.pe_ratio)
    year_end_pe, year_end_pe_source = _resolve_pe_ratio(snapshot.market_cap, estimation.estimated_net_income, snapshot.pe_ratio)
    ratio_sources = {
        "pe_ratio_source": year_end_pe_source,
        "ttm_pe_ratio_source": ttm_pe_source,
        "year_end_pe_ratio_source": year_end_pe_source,
        "pb_ratio_source": pb_source,
    }
    ttm_income = snapshot.net_income_ttm
    year_end_income = estimation.estimated_net_income
    ttm_disable_reason = None
    if ttm_income is None:
        ttm_disable_reason = "missing_ttm_net_income"
    elif ttm_income <= 0:
        ttm_disable_reason = "negative_net_income"
    year_end_disable_reason = None
    if year_end_income is None:
        year_end_disable_reason = "missing_estimated_net_income"
    elif year_end_income <= 0:
        year_end_disable_reason = "negative_net_income"

    ttm_scenario = _build_scenario(
        scenario_key="ttm",
        period_label="Son 12 Ay / TTM",
        net_income=snapshot.net_income_ttm,
        net_income_source=snapshot.net_income_source,
        price=snapshot.price,
        shares=snapshot.shares_outstanding,
        paid_in_capital=snapshot.paid_in_capital,
        equity=snapshot.equity,
        pe=ttm_pe,
        pb=resolved_pb,
        pe_source=ttm_pe_source,
        pb_source=pb_source,
        historical_pe_median=historical_pe_median,
        sector_pe_median=sector_pe_median,
        disable_reason=ttm_disable_reason,
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
        pe=year_end_pe,
        pb=resolved_pb,
        pe_source=year_end_pe_source,
        pb_source=pb_source,
        historical_pe_median=historical_pe_median,
        sector_pe_median=sector_pe_median,
        disable_reason=year_end_disable_reason,
    )

    estimated_roe = _safe_div(estimation.estimated_net_income, snapshot.equity) if (estimation.estimated_net_income or 0) > 0 and (snapshot.equity or 0) > 0 else None
    sector_comparison = compare_company_to_sector(
        {
            "pe_ratio": year_end_pe,
            "pb_ratio": resolved_pb,
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
        pe_ratio=year_end_pe,
        pb_ratio=resolved_pb,
        valuation_scenarios={"ttm": ttm_scenario, "year_end": year_end_scenario},
        estimation=estimation,
        sector_comparison=sector_comparison,
        source=snapshot.source,
        net_income_source=snapshot.net_income_source,
        equity_source=snapshot.equity_source,
        data_quality_status=quality_status,
        missing_fields=missing,
        ratio_sources=ratio_sources,
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
    sector_pe_median = (sector_metrics or {}).get("pe_median")
    resolved_pb, pb_source = _resolve_pb_ratio(snapshot.get("market_cap"), snapshot.get("equity"), snapshot.get("pb_ratio"))
    ttm_pe, ttm_pe_source = _resolve_pe_ratio(snapshot.get("market_cap"), snapshot.get("net_income_ttm"), snapshot.get("pe_ratio"))
    year_end_pe, year_end_pe_source = _resolve_pe_ratio(snapshot.get("market_cap"), snapshot.get("estimated_net_income"), snapshot.get("pe_ratio"))
    ratio_sources = {
        "pe_ratio_source": year_end_pe_source,
        "ttm_pe_ratio_source": ttm_pe_source,
        "year_end_pe_ratio_source": year_end_pe_source,
        "pb_ratio_source": pb_source,
    }
    ttm_income = snapshot.get("net_income_ttm")
    year_end_income = snapshot.get("estimated_net_income")
    ttm_disable_reason = None
    if ttm_income is None:
        ttm_disable_reason = "missing_ttm_net_income"
    elif ttm_income <= 0:
        ttm_disable_reason = "negative_net_income"
    year_end_disable_reason = None
    if year_end_income is None:
        year_end_disable_reason = "missing_estimated_net_income"
    elif year_end_income <= 0:
        year_end_disable_reason = "negative_net_income"

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
        pe=ttm_pe,
        pb=resolved_pb,
        pe_source=ttm_pe_source,
        pb_source=pb_source,
        historical_pe_median=historical_pe_median,
        sector_pe_median=sector_pe_median,
        disable_reason=ttm_disable_reason,
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
        pe=year_end_pe,
        pb=resolved_pb,
        pe_source=year_end_pe_source,
        pb_source=pb_source,
        historical_pe_median=historical_pe_median,
        sector_pe_median=sector_pe_median,
        disable_reason=year_end_disable_reason,
    )

    estimated_roe = _safe_div(snapshot.get("estimated_net_income"), snapshot.get("equity")) if (snapshot.get("estimated_net_income") or 0) > 0 and (snapshot.get("equity") or 0) > 0 else None
    sector_comparison = compare_company_to_sector(
        {
            "pe_ratio": year_end_pe,
            "pb_ratio": resolved_pb,
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
        pe_ratio=year_end_pe,
        pb_ratio=resolved_pb,
        valuation_scenarios={"ttm": ttm_scenario, "year_end": year_end_scenario},
        estimation=estimation,
        sector_comparison=sector_comparison,
        source="cache",
        net_income_source=snapshot.get("net_income_source", "unknown"),
        equity_source=snapshot.get("equity_source", "unknown"),
        data_quality_status=quality_status,
        missing_fields=missing,
        ratio_sources=ratio_sources,
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

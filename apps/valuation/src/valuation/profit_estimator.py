from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from valuation.data_access import BistSnapshot, BorsapyFinancialClient


@dataclass
class ProfitEstimationResult:
    symbol: str
    selected_method: str
    estimated_net_income: float | None
    method_values: dict[str, float | None]
    period_type: str
    period_label: str
    missing_fields: list[str]


def get_latest_reported_net_income(symbol: str, client: BorsapyFinancialClient | None = None) -> float | None:
    snapshot = (client or BorsapyFinancialClient()).load_snapshot(symbol)
    return snapshot.net_income_latest_period


def estimate_net_income_ttm(symbol: str, client: BorsapyFinancialClient | None = None) -> float | None:
    snapshot = (client or BorsapyFinancialClient()).load_snapshot(symbol)
    return snapshot.net_income_ttm


def estimate_net_income_seasonal(symbol: str, client: BorsapyFinancialClient | None = None) -> float | None:
    snapshot = (client or BorsapyFinancialClient()).load_snapshot(symbol)
    current = snapshot.net_income_latest_period
    prev_same = snapshot.previous_year_same_period_net_income
    prev_full = snapshot.previous_year_full_net_income
    if current is None or prev_same in (None, 0) or prev_full is None:
        return None
    return (current / prev_same) * prev_full


def estimate_net_income_revenue_margin(symbol: str, client: BorsapyFinancialClient | None = None) -> float | None:
    snapshot = (client or BorsapyFinancialClient()).load_snapshot(symbol)
    current_rev = snapshot.revenue_latest_period
    prev_same_rev = snapshot.previous_year_same_period_revenue
    prev_full_rev = snapshot.previous_year_full_revenue
    avg_margin = snapshot.average_margin_3y
    if current_rev is None or prev_same_rev in (None, 0) or prev_full_rev is None or avg_margin is None:
        return None
    estimated_revenue = (current_rev / prev_same_rev) * prev_full_rev
    return estimated_revenue * avg_margin


def _auto_from_snapshot(snapshot: BistSnapshot) -> ProfitEstimationResult:
    method_values: dict[str, float | None] = {
        "reported_annual": snapshot.net_income_latest_period if snapshot.period_type == "annual" else None,
        "ttm": snapshot.net_income_ttm,
        "seasonal": None,
        "revenue_margin": None,
    }
    if (
        snapshot.net_income_latest_period is not None
        and snapshot.previous_year_same_period_net_income not in (None, 0)
        and snapshot.previous_year_full_net_income is not None
    ):
        method_values["seasonal"] = (
            snapshot.net_income_latest_period / snapshot.previous_year_same_period_net_income
        ) * snapshot.previous_year_full_net_income

    if (
        snapshot.revenue_latest_period is not None
        and snapshot.previous_year_same_period_revenue not in (None, 0)
        and snapshot.previous_year_full_revenue is not None
        and snapshot.average_margin_3y is not None
    ):
        estimated_revenue = (
            snapshot.revenue_latest_period / snapshot.previous_year_same_period_revenue
        ) * snapshot.previous_year_full_revenue
        method_values["revenue_margin"] = estimated_revenue * snapshot.average_margin_3y

    if snapshot.period_type == "annual" and method_values["reported_annual"] is not None:
        return ProfitEstimationResult(
            symbol=snapshot.symbol,
            selected_method="reported_annual",
            estimated_net_income=method_values["reported_annual"],
            method_values=method_values,
            period_type=snapshot.period_type,
            period_label=snapshot.period_label,
            missing_fields=snapshot.missing_fields,
        )

    valid_estimates = [
        value for key, value in method_values.items() if key in {"ttm", "seasonal", "revenue_margin"} and value is not None
    ]
    selected_estimate = median(valid_estimates) if valid_estimates else None
    return ProfitEstimationResult(
        symbol=snapshot.symbol,
        selected_method="median_of_available_methods" if selected_estimate is not None else "insufficient_data",
        estimated_net_income=selected_estimate,
        method_values=method_values,
        period_type=snapshot.period_type,
        period_label=snapshot.period_label,
        missing_fields=snapshot.missing_fields,
    )


def estimate_net_income_auto(symbol: str, client: BorsapyFinancialClient | None = None) -> ProfitEstimationResult:
    snapshot = (client or BorsapyFinancialClient()).load_snapshot(symbol)
    return _auto_from_snapshot(snapshot)


def estimate_net_income_from_snapshot(snapshot: BistSnapshot) -> ProfitEstimationResult:
    return _auto_from_snapshot(snapshot)

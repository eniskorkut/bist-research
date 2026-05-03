from __future__ import annotations

from dataclasses import dataclass

from valuation.data_access import BorsapyFinancialClient
from valuation.profit_estimator import ProfitEstimationResult, estimate_net_income_auto


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
    average_target_price: float | None
    upside_potential_pct: float | None
    estimation: ProfitEstimationResult
    missing_fields: list[str]


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def run_valuation(symbol: str, client: BorsapyFinancialClient | None = None) -> ValuationResult:
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

    price_by_pe = _safe_div((estimated_net_income * pe) if estimated_net_income is not None and pe is not None else None, shares)
    price_by_pb = _safe_div((equity * pb) if equity is not None and pb is not None else None, shares)

    # Paid-in capital method: estimate fair value from market-cap to paid-cap ratio carried to estimated profitability.
    capital_ratio = _safe_div(market_cap, paid_in_capital)
    price_by_paid_in_capital = _safe_div(
        (estimated_net_income * capital_ratio) if estimated_net_income is not None and capital_ratio is not None else None,
        shares,
    )

    # Potential market value: use average of valid implied market caps from PE/PB methods.
    implied_caps = []
    if estimated_net_income is not None and pe is not None:
        implied_caps.append(estimated_net_income * pe)
    if equity is not None and pb is not None:
        implied_caps.append(equity * pb)
    potential_market_cap = sum(implied_caps) / len(implied_caps) if implied_caps else None
    price_by_potential_market_value = _safe_div(potential_market_cap, shares)

    # ROE method: fair price = (estimated ROE * equity * current P/B) / shares.
    estimated_roe = _safe_div(estimated_net_income, equity)
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
    valid_targets = [value for value in target_prices.values() if value is not None]
    average_target_price = sum(valid_targets) / len(valid_targets) if valid_targets else None
    upside_potential_pct = (
        ((average_target_price - price) / price) * 100
        if average_target_price is not None and price not in (None, 0)
        else None
    )

    missing = list(snapshot.missing_fields)
    if estimated_net_income is None:
        missing.append("estimated_net_income")

    return ValuationResult(
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
        average_target_price=average_target_price,
        upside_potential_pct=upside_potential_pct,
        estimation=estimation,
        missing_fields=missing,
    )

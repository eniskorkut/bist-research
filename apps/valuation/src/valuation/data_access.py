from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import borsapy as bp
import pandas as pd


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _to_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return val


def _extract_first_number(mapping: dict[str, Any], aliases: list[str]) -> float | None:
    lowered = {str(k).strip().lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias in lowered:
            parsed = _to_float(lowered[alias])
            if parsed is not None:
                return parsed
    return None


def _safe_mapping(obj: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if obj is None:
        return result
    if isinstance(obj, dict):
        iterator = obj.keys()
        for key in iterator:
            try:
                result[str(key)] = obj[key]
            except Exception:  # noqa: BLE001
                continue
        return result
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:  # noqa: BLE001
            continue
        if callable(value):
            continue
        result[str(key)] = value
    return result


def _extract_series_value(df: pd.DataFrame, aliases: list[str]) -> float | None:
    if df is None or df.empty:
        return None
    normalized_aliases = {item.lower() for item in aliases}
    index_lookup = {str(idx).strip().lower(): idx for idx in df.index}
    for alias in normalized_aliases:
        key = index_lookup.get(alias)
        if key is None:
            continue
        row = df.loc[key]
        if isinstance(row, pd.Series):
            for value in row.tolist():
                parsed = _to_float(value)
                if parsed is not None:
                    return parsed
        else:
            parsed = _to_float(row)
            if parsed is not None:
                return parsed
    return None


@dataclass
class BistSnapshot:
    symbol: str
    price: float | None
    market_cap: float | None
    pe_ratio: float | None
    pb_ratio: float | None
    shares_outstanding: float | None
    paid_in_capital: float | None
    equity: float | None
    net_income_latest_period: float | None
    net_income_ttm: float | None
    revenue_latest_period: float | None
    previous_year_same_period_net_income: float | None
    previous_year_full_net_income: float | None
    previous_year_same_period_revenue: float | None
    previous_year_full_revenue: float | None
    average_margin_3y: float | None
    period_type: str
    period_label: str
    source: str
    missing_fields: list[str]


class BorsapyFinancialClient:
    def load_snapshot(self, symbol: str) -> BistSnapshot:
        normalized_symbol = _normalize_symbol(symbol)
        ticker = bp.Ticker(normalized_symbol)
        info = _safe_mapping(getattr(ticker, "info", None))
        fast_info = _safe_mapping(getattr(ticker, "fast_info", None))
        combined = {**info, **fast_info}

        income = ticker.get_income_stmt()
        balance = ticker.get_balance_sheet()
        ttm_income = getattr(ticker, "get_ttm_income_stmt", lambda: None)()

        price = _extract_first_number(
            combined,
            ["last_price", "regularmarketprice", "price", "close"],
        )
        market_cap = _extract_first_number(combined, ["marketcap", "market_cap", "market_value"])
        pe_ratio = _extract_first_number(combined, ["trailingpe", "pe", "fk", "f/k"])
        pb_ratio = _extract_first_number(combined, ["pricetobook", "pb", "pddd", "pd/dd"])
        shares_outstanding = _extract_first_number(
            combined,
            ["sharesoutstanding", "shares", "numberofshares", "hisse_sayisi"],
        )

        paid_in_capital = _extract_series_value(
            balance,
            ["paid in capital", "odenmis sermaye", "ödenmiş sermaye", "share capital"],
        )
        equity = _extract_series_value(
            balance,
            ["total stockholder equity", "equity", "ozkaynak", "özkaynak", "total equity"],
        )

        net_income_latest = _extract_series_value(
            income,
            ["net income", "netincome", "net period profit", "donem net kari", "dönem net karı"],
        )
        revenue_latest = _extract_series_value(
            income,
            ["total revenue", "revenue", "sales", "hasilat", "satislar", "satışlar"],
        )

        net_income_ttm = _extract_series_value(
            ttm_income if isinstance(ttm_income, pd.DataFrame) else pd.DataFrame(),
            ["net income", "netincome", "net period profit", "donem net kari", "dönem net karı"],
        )

        if isinstance(income, pd.DataFrame) and income.shape[1] >= 2:
            latest_col = income.columns[0]
            prev_year_col = income.columns[min(1, income.shape[1] - 1)]
            prev3_col = income.columns[min(2, income.shape[1] - 1)]
            prev4_col = income.columns[min(3, income.shape[1] - 1)] if income.shape[1] >= 4 else None
            previous_year_same_period_net_income = _extract_series_value(
                income[[prev_year_col]],
                ["net income", "netincome", "net period profit", "donem net kari", "dönem net karı"],
            )
            previous_year_same_period_revenue = _extract_series_value(
                income[[prev_year_col]],
                ["total revenue", "revenue", "sales", "hasilat", "satislar", "satışlar"],
            )
            previous_year_full_net_income = _extract_series_value(
                income[[prev3_col]],
                ["net income", "netincome", "net period profit", "donem net kari", "dönem net karı"],
            )
            previous_year_full_revenue = _extract_series_value(
                income[[prev3_col]],
                ["total revenue", "revenue", "sales", "hasilat", "satislar", "satışlar"],
            )
            margin_samples: list[float] = []
            cols_for_margin = [latest_col, prev_year_col, prev3_col]
            if prev4_col is not None:
                cols_for_margin.append(prev4_col)
            for col in cols_for_margin:
                n = _extract_series_value(
                    income[[col]],
                    ["net income", "netincome", "net period profit", "donem net kari", "dönem net karı"],
                )
                r = _extract_series_value(
                    income[[col]],
                    ["total revenue", "revenue", "sales", "hasilat", "satislar", "satışlar"],
                )
                if n is not None and r and r != 0:
                    margin_samples.append(n / r)
            average_margin_3y = sum(margin_samples) / len(margin_samples) if margin_samples else None
        else:
            previous_year_same_period_net_income = None
            previous_year_same_period_revenue = None
            previous_year_full_net_income = None
            previous_year_full_revenue = None
            average_margin_3y = None

        period_label = str(income.columns[0]) if isinstance(income, pd.DataFrame) and not income.empty else "unknown"
        lower_label = period_label.lower()
        period_type = "annual" if any(token in lower_label for token in ["12", "year", "annual", "yillik", "yıllık"]) else "interim"

        missing: list[str] = []
        required = {
            "price": price,
            "market_cap": market_cap,
            "shares_outstanding": shares_outstanding,
            "equity": equity,
            "net_income_latest_period": net_income_latest,
            "revenue_latest_period": revenue_latest,
        }
        for key, value in required.items():
            if value is None:
                missing.append(key)

        return BistSnapshot(
            symbol=normalized_symbol,
            price=price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            shares_outstanding=shares_outstanding,
            paid_in_capital=paid_in_capital,
            equity=equity,
            net_income_latest_period=net_income_latest,
            net_income_ttm=net_income_ttm,
            revenue_latest_period=revenue_latest,
            previous_year_same_period_net_income=previous_year_same_period_net_income,
            previous_year_full_net_income=previous_year_full_net_income,
            previous_year_same_period_revenue=previous_year_same_period_revenue,
            previous_year_full_revenue=previous_year_full_revenue,
            average_margin_3y=average_margin_3y,
            period_type=period_type,
            period_label=period_label,
            source="borsapy",
            missing_fields=missing,
        )

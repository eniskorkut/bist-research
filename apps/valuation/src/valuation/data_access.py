from __future__ import annotations

from dataclasses import dataclass
import re
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


def _normalize_label(label: str) -> str:
    text = str(label).lower().strip()
    tr_map = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})
    text = text.translate(tr_map)
    text = re.sub(r"[\/\-\_\.\(\)\[\]\:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_first_number(mapping: dict[str, Any], aliases: list[str]) -> float | None:
    lowered = {_normalize_label(str(k)): v for k, v in mapping.items()}
    for alias in aliases:
        normalized_alias = _normalize_label(alias)
        if normalized_alias in lowered:
            parsed = _to_float(lowered[normalized_alias])
            if parsed is not None:
                return parsed
    return None


def _safe_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "todict"):
        try:
            data = obj.todict()
            if isinstance(data, dict):
                return dict(data)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())
        except Exception:  # noqa: BLE001
            pass
    result: dict[str, Any] = {}
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


def _pick_latest_column(df: pd.DataFrame) -> Any:
    if df is None or df.empty:
        return None
    def _score(col: Any) -> tuple[int, str]:
        text = _normalize_label(str(col))
        nums = re.findall(r"\d{4}", text)
        year = int(nums[0]) if nums else -1
        return (year, text)
    cols = list(df.columns)
    return sorted(cols, key=_score, reverse=True)[0] if cols else None


def _extract_from_series(series: pd.Series) -> float | None:
    if series is None:
        return None
    for value in series.tolist():
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _extract_series_value(df: pd.DataFrame, aliases: list[str]) -> tuple[float | None, str | None, str | None]:
    if df is None or df.empty:
        return None, None, None
    alias_norm = [_normalize_label(item) for item in aliases]

    def _search(frame: pd.DataFrame, source: str) -> tuple[float | None, str | None, str | None]:
        index_map = {_normalize_label(str(idx)): idx for idx in frame.index}
        # exact
        for alias in alias_norm:
            key = index_map.get(alias)
            if key is not None:
                col = _pick_latest_column(frame)
                row = frame.loc[key]
                value = _extract_from_series(row if isinstance(row, pd.Series) else pd.Series([row]))
                if value is not None:
                    return value, str(key), source
        # contains / token overlap
        for idx in frame.index:
            idx_norm = _normalize_label(str(idx))
            for alias in alias_norm:
                if alias in idx_norm or idx_norm in alias:
                    row = frame.loc[idx]
                    value = _extract_from_series(row if isinstance(row, pd.Series) else pd.Series([row]))
                    if value is not None:
                        return value, str(idx), source
                alias_tokens = set(alias.split())
                idx_tokens = set(idx_norm.split())
                if alias_tokens and len(alias_tokens.intersection(idx_tokens)) >= max(1, len(alias_tokens) - 1):
                    row = frame.loc[idx]
                    value = _extract_from_series(row if isinstance(row, pd.Series) else pd.Series([row]))
                    if value is not None:
                        return value, str(idx), source
        return None, None, None

    value, label, src = _search(df, "financial_statement")
    if value is not None:
        return value, label, src
    return _search(df.T, "transposed_financial_statement")


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
    net_income_source: str
    equity_source: str
    revenue_source: str
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
            ["last", "regularmarketprice", "currentprice", "last_price", "price", "close"],
        )
        market_cap = _extract_first_number(combined, ["marketcap", "market_cap", "market_value"])
        pe_ratio = _extract_first_number(combined, ["trailingpe", "pe_ratio", "pe", "fk", "f/k"])
        pb_ratio = _extract_first_number(combined, ["pricetobook", "pb_ratio", "pb", "pddd", "pd/dd"])
        shares_outstanding = _extract_first_number(
            combined,
            ["sharesoutstanding", "shares_outstanding", "shares", "numberofshares", "hisse_sayisi"],
        )

        paid_in_capital, _, _ = _extract_series_value(
            balance,
            ["paid in capital", "odenmis sermaye", "ödenmiş sermaye", "share capital"],
        )
        equity, _, equity_source = _extract_series_value(
            balance,
            [
                "total stockholder equity",
                "equity",
                "total equity",
                "ozkaynak",
                "ozkaynaklar",
                "ana ortakliga ait ozkaynaklar",
                "toplam ozkaynaklar",
            ],
        )

        net_income_latest, _, net_income_source = _extract_series_value(
            income,
            [
                "net income",
                "netincome",
                "net period profit",
                "donem net kari",
                "net donem kari",
                "donem kari zarari",
                "net donem kari zarari",
                "ana ortaklik paylari",
                "ana ortakliga ait net donem kari",
                "surdurulen faaliyetler donem kari",
            ],
        )
        revenue_latest, _, revenue_source = _extract_series_value(
            income,
            [
                "total revenue",
                "revenue",
                "sales",
                "hasilat",
                "satislar",
                "net satislar",
                "satis gelirleri",
            ],
        )

        net_income_ttm, _, _ = _extract_series_value(
            ttm_income if isinstance(ttm_income, pd.DataFrame) else pd.DataFrame(),
            [
                "net income",
                "netincome",
                "net period profit",
                "donem net kari",
                "net donem kari",
                "donem kari zarari",
                "net donem kari zarari",
                "ana ortaklik paylari",
            ],
        )

        if isinstance(income, pd.DataFrame) and income.shape[1] >= 2:
            latest_col = income.columns[0]
            prev_year_col = income.columns[min(1, income.shape[1] - 1)]
            prev3_col = income.columns[min(2, income.shape[1] - 1)]
            prev4_col = income.columns[min(3, income.shape[1] - 1)] if income.shape[1] >= 4 else None
            previous_year_same_period_net_income, _, _ = _extract_series_value(
                income[[prev_year_col]],
                ["net income", "netincome", "net period profit", "donem net kari", "ana ortaklik paylari"],
            )
            previous_year_same_period_revenue, _, _ = _extract_series_value(
                income[[prev_year_col]],
                ["total revenue", "revenue", "sales", "hasilat", "satislar", "satis gelirleri"],
            )
            previous_year_full_net_income, _, _ = _extract_series_value(
                income[[prev3_col]],
                ["net income", "netincome", "net period profit", "donem net kari", "ana ortaklik paylari"],
            )
            previous_year_full_revenue, _, _ = _extract_series_value(
                income[[prev3_col]],
                ["total revenue", "revenue", "sales", "hasilat", "satislar", "satis gelirleri"],
            )
            margin_samples: list[float] = []
            cols_for_margin = [latest_col, prev_year_col, prev3_col]
            if prev4_col is not None:
                cols_for_margin.append(prev4_col)
            for col in cols_for_margin:
                n, _, _ = _extract_series_value(
                    income[[col]],
                    ["net income", "netincome", "net period profit", "donem net kari", "ana ortaklik paylari"],
                )
                r, _, _ = _extract_series_value(
                    income[[col]],
                    ["total revenue", "revenue", "sales", "hasilat", "satislar", "satis gelirleri"],
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

        if (net_income_ttm is None or net_income_latest is None) and (market_cap or 0) > 0 and (pe_ratio or 0) > 0:
            implied_net_income = market_cap / pe_ratio
            net_income_ttm = net_income_ttm or implied_net_income
            net_income_latest = net_income_latest or implied_net_income
            net_income_source = "implied_from_pe"

        if equity is None and (market_cap or 0) > 0 and (pb_ratio or 0) > 0:
            equity = market_cap / pb_ratio
            equity_source = "implied_from_pb"

        if revenue_source is None:
            revenue_source = "unknown"
        if net_income_source is None:
            net_income_source = "financial_statement" if net_income_latest is not None else "unknown"
        if equity_source is None:
            equity_source = "financial_statement" if equity is not None else "unknown"

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
            net_income_source=net_income_source,
            equity_source=equity_source,
            revenue_source=revenue_source,
            missing_fields=missing,
        )

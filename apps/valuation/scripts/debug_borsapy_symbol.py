from __future__ import annotations

import argparse
from pprint import pprint

import borsapy as bp

from valuation.data_access import BorsapyFinancialClient


def _safe_mapping(obj):
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
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.strip().upper()
    ticker = bp.Ticker(symbol)

    info = _safe_mapping(getattr(ticker, "info", None))
    fast_info = _safe_mapping(getattr(ticker, "fast_info", None))
    income = ticker.get_income_stmt()
    balance = ticker.get_balance_sheet()
    ttm_income = getattr(ticker, "get_ttm_income_stmt", lambda: None)()

    print("== info keys ==")
    print(sorted(info.keys())[:200])
    print("== info selected ==")
    for key in ["last", "regularMarketPrice", "currentPrice", "last_price", "close", "marketCap", "sharesOutstanding", "trailingPE", "priceToBook"]:
        print(f"{key}={info.get(key)}")

    print("== fast_info keys ==")
    print(sorted(fast_info.keys())[:200])
    print("== fast_info selected ==")
    for key in ["last", "regularMarketPrice", "currentPrice", "last_price", "close", "marketCap", "sharesOutstanding", "trailingPE", "priceToBook"]:
        print(f"{key}={fast_info.get(key)}")

    print("== income stmt ==")
    print(f"shape={getattr(income, 'shape', None)}")
    print(f"columns={list(getattr(income, 'columns', [])[:40])}")
    print(f"rows={list(getattr(income, 'index', [])[:40])}")

    print("== balance sheet ==")
    print(f"shape={getattr(balance, 'shape', None)}")
    print(f"columns={list(getattr(balance, 'columns', [])[:40])}")
    print(f"rows={list(getattr(balance, 'index', [])[:40])}")

    print("== ttm income ==")
    print(f"shape={getattr(ttm_income, 'shape', None)}")
    print(f"columns={list(getattr(ttm_income, 'columns', [])[:40])}")
    print(f"rows={list(getattr(ttm_income, 'index', [])[:40])}")

    snapshot = BorsapyFinancialClient().load_snapshot(symbol)
    print("== extracted snapshot ==")
    pprint(snapshot.__dict__)
    print("== missing_fields ==")
    print(snapshot.missing_fields)
    print("== extracted fields ==")
    print(f"revenue_latest_period={snapshot.revenue_latest_period}")
    print(f"net_income_latest_period={snapshot.net_income_latest_period}")
    print(f"net_income_ttm={snapshot.net_income_ttm}")
    print(f"equity={snapshot.equity}")
    print(f"paid_in_capital={snapshot.paid_in_capital}")
    print("== extraction source ==")
    print(f"net_income_source={snapshot.net_income_source}")
    print(f"equity_source={snapshot.equity_source}")
    print(f"revenue_source={snapshot.revenue_source}")


if __name__ == "__main__":
    main()

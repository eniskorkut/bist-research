from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from edgar import Company, set_identity


def configure_edgar_identity() -> str:
    load_dotenv()
    identity = os.getenv("EDGAR_IDENTITY")
    if not identity:
        raise RuntimeError("EDGAR_IDENTITY is required for SEC EDGAR access.")
    set_identity(identity)
    return identity


def get_company_financials(symbol: str) -> Any:
    configure_edgar_identity()
    company = Company(symbol)
    return company.get_financials()


def get_income_statement(symbol: str) -> Any:
    return get_company_financials(symbol).income_statement()


def get_balance_sheet(symbol: str) -> Any:
    return get_company_financials(symbol).balance_sheet()


def get_cashflow_statement(symbol: str) -> Any:
    return get_company_financials(symbol).cashflow_statement()


def get_recent_filings(symbol: str, form: str = "10-K") -> Any:
    configure_edgar_identity()
    company = Company(symbol)
    return company.get_filings(form=form)


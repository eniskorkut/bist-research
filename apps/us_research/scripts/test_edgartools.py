from __future__ import annotations

from edgar import Company

from us_research.sec_financials import configure_edgar_identity, get_company_financials


def main() -> None:
    identity = configure_edgar_identity()
    print(f"edgar identity configured: {identity}")

    company = Company("AAPL")
    financials = get_company_financials("AAPL")
    income = financials.income_statement()
    balance = financials.balance_sheet()
    cashflow = financials.cashflow_statement()

    print(f"company loaded: {company}")
    print(f"income rows: {len(income)}")
    print(f"balance rows: {len(balance)}")
    print(f"cashflow rows: {len(cashflow)}")
    print("edgartools test ok")


if __name__ == "__main__":
    main()


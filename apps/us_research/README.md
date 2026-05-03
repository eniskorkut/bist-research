# us_research

US market research module in this repository. Uses OpenBB (`openbb`, `openbb-yfinance`) for US OHLCV/volume and `edgartools` for SEC filings and financial statements.

## Environment

- `EDGAR_IDENTITY` is required for SEC EDGAR access.
- API key is not required for `openbb-yfinance`.

## Commands

```shell
python scripts/test_openbb.py
python scripts/test_edgartools.py
python scripts/scan_us_volume.py --symbols AAPL NVDA MSFT TSLA AMZN --lookback 20 --min-ratio 1.5 --start-date 2024-01-01
```

Outputs include:
- `market=US`
- `currency=USD`
- `source=openbb:yfinance`

This project is for research and analysis only. Not investment advice.


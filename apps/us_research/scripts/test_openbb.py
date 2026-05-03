from __future__ import annotations

from us_research.market_data import get_us_history


def main() -> None:
    df = get_us_history(symbol="AAPL", start_date="2024-01-01", provider="yfinance")
    if "volume" not in df.columns:
        raise RuntimeError("volume column missing from OpenBB output")
    print(df.tail(5).to_string())
    print("openbb test ok: volume column present")


if __name__ == "__main__":
    main()


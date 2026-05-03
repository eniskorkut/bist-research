from __future__ import annotations

from valuation.valuation_engine import run_valuation


def main() -> None:
    for symbol in ["THYAO", "ASELS", "TUPRS"]:
        print(f"\n=== {symbol} ===")
        try:
            result = run_valuation(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
            continue
        print(f"price={result.price}")
        print(f"estimated_net_income={result.estimated_net_income}")
        print(f"target_prices={result.target_prices}")
        print(f"average_target_price={result.average_target_price}")
        print(f"upside_potential_pct={result.upside_potential_pct}")
        if result.missing_fields:
            print(f"missing_fields={result.missing_fields}")


if __name__ == "__main__":
    main()


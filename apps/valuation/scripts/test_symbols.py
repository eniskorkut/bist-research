from __future__ import annotations

from valuation.valuation_engine import run_valuation
from valuation.symbols import normalize_bist_symbol


def main() -> None:
    for raw_symbol in ["thyao", "asels", "tuprs"]:
        symbol = normalize_bist_symbol(raw_symbol)
        print(f"\n=== {symbol} ===")
        try:
            result = run_valuation(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
            continue
        print(f"price={result.price}")
        print(f"ttm_net_income={result.valuation_scenarios['ttm'].net_income}")
        print(f"year_end_net_income={result.valuation_scenarios['year_end'].net_income}")
        print(f"ttm_target_prices={result.valuation_scenarios['ttm'].target_prices}")
        print(f"year_end_target_prices={result.valuation_scenarios['year_end'].target_prices}")
        print(f"year_end_fair_value_median={result.valuation_scenarios['year_end'].fair_value_median}")
        print(f"year_end_upside_potential_pct={result.valuation_scenarios['year_end'].upside_potential_pct}")
        if result.missing_fields:
            print(f"missing_fields={result.missing_fields}")


if __name__ == "__main__":
    main()

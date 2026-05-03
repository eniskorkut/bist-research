from __future__ import annotations

import argparse

import borsapy as bp

from valuation.cache import (
    evaluate_snapshot_quality,
    get_company_snapshot,
    init_db,
    is_stale,
    upsert_company_snapshot,
    upsert_sector_metrics,
)
from valuation.data_access import BorsapyFinancialClient
from valuation.profit_estimator import estimate_net_income_auto
from valuation.sector_analysis import (
    calculate_sector_metrics,
    get_bist_sector_map,
    get_sector_index_for_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="XU100")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--db-path", default="/data/valuation_cache.sqlite")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db(args.db_path)
    symbols = args.symbols or list(getattr(bp.Index(args.index), "component_symbols", []) or [])
    symbols = sorted({str(s).strip().upper() for s in symbols if s})
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    client = BorsapyFinancialClient()
    errors: list[str] = []
    by_sector: dict[str, list[dict]] = {}
    sector_name_map = get_bist_sector_map()
    refreshed_symbols: list[str] = []
    skipped_fresh_symbols: list[str] = []
    usable_symbols: list[str] = []
    partial_symbols: list[str] = []
    unusable_symbols: list[str] = []

    for symbol in symbols:
        try:
            # Check cache freshness before calling borsapy
            existing = get_company_snapshot(args.db_path, symbol)
            existing_status = (existing or {}).get("data_quality_status")
            if (
                existing is not None
                and not is_stale(existing.get("updated_at"))
                and existing_status in {"usable", "partial"}
                and not args.force
            ):
                skipped_fresh_symbols.append(symbol)
                sector_index = existing.get("sector_index")
                if sector_index:
                    by_sector.setdefault(sector_index, []).append(existing)
                continue

            snapshot = client.load_snapshot(symbol)
            estimation = estimate_net_income_auto(symbol, client=client)
            estimated_net_income = estimation.estimated_net_income
            roe = (
                (estimated_net_income / snapshot.equity)
                if estimated_net_income is not None and snapshot.equity not in (None, 0)
                else None
            )
            sector_index = get_sector_index_for_symbol(symbol)
            sector_name = sector_name_map.get(sector_index or "", "Bilinmiyor")
            payload = {
                "symbol": snapshot.symbol,
                "market": "BIST",
                "sector_index": sector_index,
                "sector_name": sector_name,
                "price": snapshot.price,
                "market_cap": snapshot.market_cap,
                "shares_outstanding": snapshot.shares_outstanding,
                "paid_in_capital": snapshot.paid_in_capital,
                "pe_ratio": snapshot.pe_ratio,
                "pb_ratio": snapshot.pb_ratio,
                "roe": roe,
                "equity": snapshot.equity,
                "net_income_latest_period": snapshot.net_income_latest_period,
                "net_income_ttm": snapshot.net_income_ttm,
                "estimated_net_income": estimated_net_income,
                "financial_period": snapshot.period_label,
                "period_type": snapshot.period_type,
                "source": "borsapy",
                "missing_fields_json": sorted(set(snapshot.missing_fields + estimation.missing_fields)),
            }
            quality_status, quality_errors = evaluate_snapshot_quality(payload)
            payload["data_quality_status"] = quality_status
            payload["data_quality_errors_json"] = quality_errors
            upsert_company_snapshot(args.db_path, payload)
            refreshed_symbols.append(symbol)
            if quality_status == "usable":
                usable_symbols.append(symbol)
            elif quality_status == "partial":
                partial_symbols.append(symbol)
            else:
                unusable_symbols.append(symbol)
            if sector_index and quality_status != "unusable":
                by_sector.setdefault(sector_index, []).append(payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")

    sector_count = 0
    for sector_index, rows in by_sector.items():
        metrics = calculate_sector_metrics(rows)
        metrics["sector_index"] = sector_index
        metrics["sector_name"] = sector_name_map.get(sector_index, "Bilinmiyor")
        upsert_sector_metrics(args.db_path, metrics)
        sector_count += 1

    print(f"processed_symbols={len(symbols)}")
    print(f"refreshed_symbols={len(refreshed_symbols)}")
    print(f"skipped_fresh_symbols={len(skipped_fresh_symbols)}")
    print(f"usable_symbols={len(usable_symbols)}")
    print(f"partial_symbols={len(partial_symbols)}")
    print(f"unusable_symbols={len(unusable_symbols)}")
    print(f"errors={len(errors)}")
    if errors:
        for item in errors:
            print(f"warning={item}")
    if skipped_fresh_symbols:
        print(f"skipped_list={','.join(skipped_fresh_symbols)}")
    if unusable_symbols:
        print(f"unusable_list={','.join(unusable_symbols)}")
    print(f"sector_metrics_computed={sector_count}")
    print(f"db_path={args.db_path}")


if __name__ == "__main__":
    main()

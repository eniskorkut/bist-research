from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from valuation.cache import (
    evaluate_snapshot_quality,
    is_snapshot_usable,
    get_company_snapshot,
    get_sector_metrics,
    init_db,
    is_stale,
    upsert_company_snapshot,
    upsert_sector_metrics,
)
from valuation.data_access import BorsapyFinancialClient
from valuation.profit_estimator import estimate_net_income_auto
from valuation.sector_analysis import (
    calculate_sector_metrics,
    compare_company_to_sector,
    get_bist_sector_map,
    get_sector_index_for_symbol,
    get_sector_symbols,
)
from valuation.valuation_engine import run_valuation_from_snapshot

DB_PATH = "/data/valuation_cache.sqlite"


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def _render_scenario_block(title: str, scenario) -> None:
    st.header(title)
    st.write(f"Kullanilan donem: `{scenario.period_label}`")
    st.write(f"Kullanilan net kar: `{_fmt(scenario.net_income)}`")
    st.write(f"Net kar kaynagi: `{scenario.net_income_source}`")

    st.subheader("Hedef Fiyat Tablosu")
    st.table(
        {
            "Yontem": list(scenario.target_prices.keys()),
            "Hedef Fiyat": [_fmt(v) for v in scenario.target_prices.values()],
        }
    )
    st.write(f"Dahil edilen yontemler: `{', '.join(scenario.included_methods) if scenario.included_methods else 'yok'}`")
    st.write(f"Haric tutulan yontemler: `{', '.join(scenario.excluded_methods) if scenario.excluded_methods else 'yok'}`")
    st.write(f"Medyan adil deger: `{_fmt(scenario.fair_value_median)}`")
    st.write(f"Filtreli ortalama adil deger: `{_fmt(scenario.fair_value_mean_filtered)}`")
    st.write(f"Prim potansiyeli (%): `{_fmt(scenario.upside_potential_pct)}`")

    st.subheader("Odenmis Sermaye Hesabi")
    st.write(f"EPS (net_income / paid_in_capital): `{_fmt(scenario.paid_capital_details.get('eps'))}`")
    st.write(f"Kurs formulu (EPS x 10): `{_fmt(scenario.paid_capital_details.get('x10'))}`")
    st.write(f"Gecmis F/K medyan: `{_fmt(scenario.paid_capital_details.get('historical_pe_median'))}`")
    st.write(f"EPS x gecmis F/K: `{_fmt(scenario.paid_capital_details.get('historical_pe_value'))}`")
    st.write(f"Odenmis sermaye final: `{_fmt(scenario.paid_capital_details.get('final'))}`")
    if scenario.paid_capital_details.get("historical_pe_median") is None:
        st.warning("Gecmis F/K verisi hesaplanamadi. Final deger EPS x 10 olarak kullanildi.")


def _refresh_symbol_and_sector(
    symbol: str,
    db_path: str,
    *,
    force: bool = False,
) -> tuple[dict | None, dict | None, str | None]:
    """Refresh the given *symbol* and its sector peers into the cache.

    For each symbol in the sector:
    - If the cached snapshot is fresh **and** *force* is ``False``, it is
      re-used from the cache (borsapy is NOT called).
    - Otherwise the data is fetched from borsapy and upserted.

    Sector metrics are always re-computed from the (possibly mixed
    cache / fresh) snapshots.
    """
    init_db(db_path)
    client = BorsapyFinancialClient()
    sector_names = get_bist_sector_map()
    target_sector = get_sector_index_for_symbol(symbol)
    process_symbols = [symbol]
    if target_sector:
        process_symbols = sorted(set(process_symbols + get_sector_symbols(target_sector)))

    snapshots: list[dict] = []
    for sym in process_symbols:
        # Check existing cache first
        existing = get_company_snapshot(db_path, sym)
        if existing is not None and not is_stale(existing.get("updated_at")) and not force:
            # Fresh & no force → skip borsapy call, reuse cached payload
            if (existing.get("sector_index") or "") == target_sector:
                snapshots.append(existing)
            continue

        # Stale / missing / forced → fetch from borsapy
        snapshot = client.load_snapshot(sym)
        estimation = estimate_net_income_auto(sym, client=client)
        estimated_net_income = estimation.estimated_net_income
        roe = (estimated_net_income / snapshot.equity) if estimated_net_income is not None and snapshot.equity not in (None, 0) else None
        sector_index = get_sector_index_for_symbol(sym)
        payload = {
            "symbol": snapshot.symbol,
            "market": "BIST",
            "sector_index": sector_index,
            "sector_name": sector_names.get(sector_index or "", "Bilinmiyor"),
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
            "net_income_source": snapshot.net_income_source,
            "equity_source": snapshot.equity_source,
            "revenue_source": snapshot.revenue_source,
            "missing_fields_json": sorted(set(snapshot.missing_fields + estimation.missing_fields)),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        quality_status, quality_errors = evaluate_snapshot_quality(payload)
        payload["data_quality_status"] = quality_status
        payload["data_quality_errors_json"] = quality_errors
        upsert_company_snapshot(db_path, payload)
        if sector_index == target_sector:
            snapshots.append(payload)

    sector_metrics = None
    if target_sector and snapshots:
        sector_metrics = calculate_sector_metrics(snapshots)
        sector_metrics.update(
            {
                "sector_index": target_sector,
                "sector_name": sector_names.get(target_sector, "Bilinmiyor"),
                "calculated_at": datetime.now(UTC).isoformat(),
            }
        )
        upsert_sector_metrics(db_path, sector_metrics)
    cached_company = get_company_snapshot(db_path, symbol)
    cached_sector = sector_metrics or (get_sector_metrics(db_path, target_sector) if target_sector else None)
    return cached_company, cached_sector, target_sector


st.set_page_config(page_title="BIST Otomatik Degerleme", layout="wide")
st.title("BIST Otomatik Degerleme")
st.caption("Yatirim tavsiyesi degildir.")

init_db(DB_PATH)
symbol = st.text_input("Hisse kodu", value="THYAO").strip().upper()
col_a, col_b = st.columns(2)
analyze = col_a.button("Analiz")
refresh = col_b.button("Cache'i yenile")

if refresh and symbol:
    try:
        _refresh_symbol_and_sector(symbol, DB_PATH, force=True)
        st.success("Cache yenilendi.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cache yenileme hatasi: {exc}")

if analyze:
    if not symbol:
        st.error("Hisse kodu girin.")
    else:
        # ---- determine freshness ----
        cached = get_company_snapshot(DB_PATH, symbol)
        company_usable = cached is not None and is_snapshot_usable(cached)
        company_fresh = cached is not None and not is_stale(cached.get("updated_at")) and company_usable

        sector_index = cached.get("sector_index") if cached else None
        if not sector_index:
            sector_index = get_sector_index_for_symbol(symbol)

        sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
        sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))

        # ---- refresh as needed ----
        valuation_source = "cache"

        if not company_fresh:
            # Company stale/missing → refresh both company and sector
            try:
                _refresh_symbol_and_sector(symbol, DB_PATH)
                cached = get_company_snapshot(DB_PATH, symbol)
                sector_index = cached.get("sector_index") if cached else sector_index
                sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
                company_fresh = cached is not None and not is_stale(cached.get("updated_at"))
                sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))
                valuation_source = "borsapy_refresh"
            except Exception as exc:  # noqa: BLE001
                st.error(f"Veri cekme hatasi: {exc}")
                st.stop()
        elif not sector_fresh:
            # Company fresh, sector stale/missing → refresh sector only
            # (the helper will skip the fresh company snapshot automatically)
            try:
                _refresh_symbol_and_sector(symbol, DB_PATH)
                cached = get_company_snapshot(DB_PATH, symbol)
                sector_index = cached.get("sector_index") if cached else sector_index
                sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
                sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))
                valuation_source = "cache"  # company came from cache
            except Exception as exc:  # noqa: BLE001
                st.error(f"Sektor yenileme hatasi: {exc}")
                # sector_metrics may remain None, continue anyway

        if cached is None:
            st.error("Sirket verisi bulunamadi.")
            st.stop()
        if not is_snapshot_usable(cached):
            st.error(
                "Cache fresh ama veri kalitesi yetersiz. borsapy verisi bos dondu veya parser alanlari cikarilamadi."
            )
            st.error(
                f"{symbol} icin gerekli temel veriler alinamadi. Debug icin: "
                "`python scripts/debug_borsapy_symbol.py {symbol}`"
            )
            st.json(
                {
                    "data_quality_status": cached.get("data_quality_status"),
                    "data_quality_errors": cached.get("data_quality_errors_json"),
                    "missing_fields": cached.get("missing_fields_json"),
                    "updated_at": cached.get("updated_at"),
                }
            )
            st.stop()

        # ---- run valuation (always from cache snapshot) ----
        result = run_valuation_from_snapshot(cached, sector_metrics=sector_metrics, db_path=DB_PATH)

        sector_name = get_bist_sector_map().get(sector_index or "", "Bilinmiyor")
        sector_comparison = compare_company_to_sector(
            {
                "pe_ratio": result.pe_ratio,
                "pb_ratio": result.pb_ratio,
                "roe": (result.valuation_scenarios["year_end"].net_income / result.equity) if result.valuation_scenarios["year_end"].net_income is not None and result.equity not in (None, 0) else None,
                "estimated_net_income": result.valuation_scenarios["year_end"].net_income,
                "equity": result.equity,
            },
            sector_metrics or {},
        ) if sector_metrics else {}

        st.header("Temel Veriler")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hisse Fiyati", _fmt(result.price))
        col2.metric("Piyasa Degeri", _fmt(result.market_cap))
        col3.metric("Hisse Sayisi", _fmt(result.shares_outstanding))
        col4.metric("Tahmini Net Kar (Yil Sonu)", _fmt(result.valuation_scenarios["year_end"].net_income))
        col5, col6, col7 = st.columns(3)
        col5.metric("F/K", _fmt(result.pe_ratio))
        col6.metric("PD/DD", _fmt(result.pb_ratio))
        col7.metric("Ozkaynak", _fmt(result.equity))
        st.write(f"Net kar kaynagi: `{cached.get('net_income_source', 'unknown')}`")
        st.write(f"Ozkaynak kaynagi: `{cached.get('equity_source', 'unknown')}`")
        st.write(f"Veri kalite durumu: `{cached.get('data_quality_status', 'unknown')}`")

        st.header("Tahmini Net Kar")
        st.write(f"Secilen yontem: `{result.estimation.selected_method}`")
        st.write(f"TTM tahmin: `{_fmt(result.estimation.method_values.get('ttm'))}`")
        st.write(f"Sezonsallik tahmin: `{_fmt(result.estimation.method_values.get('seasonal'))}`")
        st.write(f"Ciro x marj tahmin: `{_fmt(result.estimation.method_values.get('revenue_margin'))}`")
        st.write(f"Nihai tahmini net kar: `{_fmt(result.estimation.estimated_net_income)}`")
        if cached.get("net_income_source") == "implied_from_pe":
            st.warning("Tahmini net kar F/K ve piyasa degeri uzerinden turetildi.")
        if cached.get("equity_source") == "implied_from_pb":
            st.warning("Ozkaynak PD/DD ve piyasa degeri uzerinden turetildi.")
        if result.missing_fields:
            st.warning("Eksik veri alanlari: " + ", ".join(sorted(set(result.missing_fields))))

        _render_scenario_block("Son 12 Ay / TTM Degerleme", result.valuation_scenarios["ttm"])
        _render_scenario_block("Yil Sonu Tahmini Degerleme", result.valuation_scenarios["year_end"])

        st.header("Sektor Analizi")
        st.write(f"Sektor endeksi: `{sector_index or 'N/A'}`")
        st.write(f"Sektor adi: `{sector_name}`")
        if sector_metrics:
            st.write(f"Sektor F/K medyan: `{_fmt(sector_metrics.get('pe_median'))}`")
            st.write(f"Sektor F/K aggregate: `{_fmt(sector_metrics.get('pe_aggregate'))}`")
            st.write(f"Sektor PD/DD medyan: `{_fmt(sector_metrics.get('pb_median'))}`")
            st.write(f"Sektor PD/DD aggregate: `{_fmt(sector_metrics.get('pb_aggregate'))}`")
            st.write(f"Sektor ROE aggregate: `{_fmt(sector_metrics.get('roe_aggregate'))}`")
            st.json(sector_comparison)
        else:
            st.warning("Sektor metrikleri yok.")

        st.header("Cache Durumu")
        updated_at = cached.get("updated_at")
        company_cache_status = "fresh" if company_fresh else ("stale" if cached else "missing")
        sector_cache_status = "fresh" if sector_fresh else ("stale" if sector_metrics else "missing")
        st.write(f"Kaynak: `{valuation_source}`")
        st.write(f"company_cache_status: `{company_cache_status}`")
        st.write(f"sector_cache_status: `{sector_cache_status}`")
        st.write(f"valuation_source: `{valuation_source}`")
        st.write(f"updated_at: `{updated_at}`")
        if valuation_source == "borsapy_refresh":
            st.warning("Cache stale veya yoktu; borsapy'den yenilendi.")

        st.info("Yatirim tavsiyesi degildir.")

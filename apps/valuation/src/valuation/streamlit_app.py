from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from valuation.cache import (
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
from valuation.valuation_engine import run_valuation

DB_PATH = "/data/valuation_cache.sqlite"


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def _refresh_symbol_and_sector(symbol: str, db_path: str) -> tuple[dict | None, dict | None, str | None]:
    init_db(db_path)
    client = BorsapyFinancialClient()
    sector_names = get_bist_sector_map()
    target_sector = get_sector_index_for_symbol(symbol)
    process_symbols = [symbol]
    if target_sector:
        process_symbols = sorted(set(process_symbols + get_sector_symbols(target_sector)))

    snapshots: list[dict] = []
    for sym in process_symbols:
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
            "missing_fields_json": sorted(set(snapshot.missing_fields + estimation.missing_fields)),
            "updated_at": datetime.now(UTC).isoformat(),
        }
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
        _refresh_symbol_and_sector(symbol, DB_PATH)
        st.success("Cache yenilendi.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cache yenileme hatasi: {exc}")

if analyze:
    if not symbol:
        st.error("Hisse kodu girin.")
    else:
        cached = get_company_snapshot(DB_PATH, symbol)
        company_from_cache = cached is not None and not is_stale(cached.get("updated_at"))
        sector_metrics = None
        sector_index = cached.get("sector_index") if cached else None
        if sector_index:
            sector_metrics = get_sector_metrics(DB_PATH, sector_index)

        if not company_from_cache:
            try:
                cached, sector_metrics, sector_index = _refresh_symbol_and_sector(symbol, DB_PATH)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Veri cekme hatasi: {exc}")
                st.stop()

        if cached is None:
            st.error("Sirket verisi bulunamadi.")
            st.stop()

        result = run_valuation(symbol, sector_metrics=sector_metrics, db_path=DB_PATH)
        sector_name = get_bist_sector_map().get(sector_index or "", "Bilinmiyor")
        sector_comparison = compare_company_to_sector(
            {
                "pe_ratio": result.pe_ratio,
                "pb_ratio": result.pb_ratio,
                "roe": (result.estimated_net_income / result.equity) if result.estimated_net_income is not None and result.equity not in (None, 0) else None,
                "estimated_net_income": result.estimated_net_income,
                "equity": result.equity,
            },
            sector_metrics or {},
        ) if sector_metrics else {}

        st.header("Temel Veriler")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hisse Fiyati", _fmt(result.price))
        col2.metric("Piyasa Degeri", _fmt(result.market_cap))
        col3.metric("Hisse Sayisi", _fmt(result.shares_outstanding))
        col4.metric("Tahmini Net Kar", _fmt(result.estimated_net_income))
        col5, col6, col7 = st.columns(3)
        col5.metric("F/K", _fmt(result.pe_ratio))
        col6.metric("PD/DD", _fmt(result.pb_ratio))
        col7.metric("Ozkaynak", _fmt(result.equity))

        st.header("Tahmini Net Kar")
        st.write(f"Secilen yontem: `{result.estimation.selected_method}`")
        st.write(f"TTM tahmin: `{_fmt(result.estimation.method_values.get('ttm'))}`")
        st.write(f"Sezonsallik tahmin: `{_fmt(result.estimation.method_values.get('seasonal'))}`")
        st.write(f"Ciro x marj tahmin: `{_fmt(result.estimation.method_values.get('revenue_margin'))}`")
        st.write(f"Nihai tahmini net kar: `{_fmt(result.estimated_net_income)}`")
        if result.missing_fields:
            st.warning("Eksik veri alanlari: " + ", ".join(sorted(set(result.missing_fields))))

        st.header("5 Yontemli Degerleme Tablosu")
        st.table({"Yontem": list(result.target_prices.keys()), "Hedef Fiyat": [_fmt(v) for v in result.target_prices.values()]})
        st.write(f"5 yontem ortalamasi: `{_fmt(result.average_target_price)}`")
        st.write(f"Prim potansiyeli (%): `{_fmt(result.upside_potential_pct)}`")

        st.header("Sektor Bazli Ek Degerleme")
        st.table({"Yontem": list(result.sector_target_prices.keys()), "Hedef Fiyat": [_fmt(v) for v in result.sector_target_prices.values()]})
        st.write(f"Sektor dahil ortalama: `{_fmt(result.average_with_sector_price)}`")

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
        st.write(f"Kaynak: `borsapy`")
        st.write(f"Cache kullanildi mi: `{'Evet' if company_from_cache else 'Hayir (anlik yenilendi)'}`")
        st.write(f"updated_at: `{updated_at}`")
        st.write(f"stale/fresh: `{'stale' if is_stale(updated_at) else 'fresh'}`")
        if not company_from_cache:
            st.warning("Cache stale veya yoktu; secili hisse ve sektoru yenilendi.")

        st.info("Yatirim tavsiyesi degildir.")

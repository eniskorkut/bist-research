from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from valuation.cache import (
    evaluate_snapshot_quality,
    get_company_snapshot,
    get_sector_metrics,
    init_db,
    is_snapshot_usable,
    is_snapshot_valuation_ready,
    is_stale,
    should_write_snapshot,
    upsert_company_snapshot,
    upsert_sector_metrics,
)
from valuation.data_access import BorsapyFinancialClient
from valuation.profit_estimator import estimate_net_income_from_snapshot
from valuation.sector_analysis import (
    calculate_sector_metrics,
    compare_company_to_sector,
    get_bist_sector_map,
    get_sector_index_for_symbol,
    get_sector_symbols,
)
from valuation.valuation_engine import run_valuation_from_snapshot

DB_PATH = "/data/valuation_cache.sqlite"


def fmt_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}{suffix}"


def fmt_money(value: float | int | None) -> str:
    return fmt_number(value)


def fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}%"


def fmt_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return "N/A"
    return str(value)


def render_status_badge(label: str, status: str) -> str:
    return f"{label}: {fmt_text(status)}"


def to_plain_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_plain_dict(x) for x in obj]
    return obj


def build_target_price_dataframe(scenario: Any) -> pd.DataFrame:
    method_map = {
        "cari_fk": ("Cari F/K", "Net Kâr × F/K / Hisse Sayısı"),
        "pd_dd": ("PD/DD", "Özkaynak × PD/DD / Hisse Sayısı"),
        "odenmis_sermaye_x10": ("Ödenmiş Sermaye: EPS × 10", "EPS × 10"),
        "odenmis_sermaye_historical_pe": ("Ödenmiş Sermaye: EPS × Geçmiş F/K", "EPS × Geçmiş F/K"),
        "odenmis_sermaye_sector_pe": ("Ödenmiş Sermaye: EPS × Sektör F/K", "EPS × Sektör F/K"),
        "odenmis_sermaye_current_pe": ("Ödenmiş Sermaye: EPS × Güncel F/K", "EPS × Güncel F/K"),
        "odenmis_sermaye_final": ("Ödenmiş Sermaye Final", "Fallback kuralına göre final"),
        "potansiyel_piyasa_degeri": ("Potansiyel Piyasa Değeri", "Çarpanlardan türetilen potansiyel değer"),
        "ozsermaye_karliligi": ("Özsermaye Kârlılığı", "ROE temelli yaklaşım"),
        "sektor_fk_hedef": ("Sektör F/K Hedefi", "EPS × Sektör F/K"),
        "sektor_pd_dd_hedef": ("Sektör PD/DD Hedefi", "Defter Değeri × Sektör PD/DD"),
    }
    rows: list[dict[str, str]] = []
    included = set(scenario.included_methods or [])
    for method, value in scenario.target_prices.items():
        label, formula = method_map.get(method, (method, "-"))
        in_fair_value = "Evet" if method in included else "Hayır"
        if value is None:
            status = "Eksik veri"
        elif method in included:
            status = "Kullanıldı"
        else:
            status = "Hariç tutuldu / bilgi amaçlı"
        rows.append(
            {
                "Yöntem": label,
                "Formül": formula,
                "Hedef Fiyat": fmt_money(value),
                "Adil Değere Dahil": in_fair_value,
                "Durum": status,
            }
        )
    return pd.DataFrame(rows)


def build_summary_dataframe(result: Any, cached: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Piyasa Değeri", fmt_money(result.market_cap), "borsapy"),
        ("Hisse Sayısı", fmt_number(result.shares_outstanding), "borsapy"),
        ("Ödenmiş Sermaye", fmt_number(result.paid_in_capital), "borsapy"),
        ("Özkaynak", fmt_money(result.equity), cached.get("equity_source")),
        ("Net Kâr TTM", fmt_money(cached.get("net_income_ttm")), cached.get("net_income_source")),
        ("Tahmini Yıl Sonu Net Kâr", fmt_money(result.valuation_scenarios["year_end"].net_income), result.valuation_scenarios["year_end"].net_income_source),
        ("Net Kâr Kaynağı", fmt_text(cached.get("net_income_source")), "system"),
        ("Özkaynak Kaynağı", fmt_text(cached.get("equity_source")), "system"),
        ("Veri Kalite Durumu", fmt_text(cached.get("data_quality_status")), "cache"),
    ]
    return pd.DataFrame(rows, columns=["Alan", "Değer", "Kaynak"])


def build_sector_dataframe(
    result: Any,
    sector_metrics: dict[str, Any] | None,
    sector_comparison: dict[str, Any] | None,
    sector_index: str | None,
    sector_name: str,
) -> pd.DataFrame:
    comp = sector_comparison or {}
    metrics = sector_metrics or {}
    rows = [
        ("Sektör Endeksi", fmt_text(sector_index)),
        ("Sektör Adı", fmt_text(sector_name)),
        ("Şirket F/K", fmt_number(result.pe_ratio)),
        ("Sektör F/K Medyan", fmt_number(metrics.get("pe_median"))),
        ("Sektör F/K Aggregate", fmt_number(metrics.get("pe_aggregate"))),
        ("Şirket PD/DD", fmt_number(result.pb_ratio)),
        ("Sektör PD/DD Medyan", fmt_number(metrics.get("pb_median"))),
        ("Sektör PD/DD Aggregate", fmt_number(metrics.get("pb_aggregate"))),
        ("Şirket ROE", fmt_pct(comp.get("company_roe"))),
        ("Sektör ROE Aggregate", fmt_pct(metrics.get("roe_aggregate"))),
    ]
    return pd.DataFrame(rows, columns=["Gösterge", "Değer"])


def _refresh_symbol_and_sector(symbol: str, db_path: str, *, force: bool = False) -> tuple[dict | None, dict | None, str | None, dict]:
    init_db(db_path)
    client = BorsapyFinancialClient()
    sector_names = get_bist_sector_map()
    target_sector = get_sector_index_for_symbol(symbol)
    process_symbols = [symbol]
    if target_sector:
        process_symbols = sorted(set(process_symbols + get_sector_symbols(target_sector)))

    snapshots: list[dict] = []
    refresh_meta = {
        "wrote_to_cache": 0,
        "preserved_existing_cache": 0,
        "rejected_unusable_refresh": 0,
        "messages": [],
    }

    for sym in process_symbols:
        existing = get_company_snapshot(db_path, sym)
        if existing is not None and not is_stale(existing.get("updated_at")) and not force:
            if (existing.get("sector_index") or "") == target_sector:
                snapshots.append(existing)
            continue

        snapshot = client.load_snapshot(sym)
        estimation = estimate_net_income_from_snapshot(snapshot)
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
        write_ok, reason = should_write_snapshot(payload, existing)
        if write_ok:
            upsert_company_snapshot(db_path, payload)
            refresh_meta["wrote_to_cache"] += 1
            if sector_index == target_sector:
                snapshots.append(payload)
        else:
            refresh_meta["preserved_existing_cache"] += 1
            if quality_status == "unusable":
                refresh_meta["rejected_unusable_refresh"] += 1
            refresh_meta["messages"].append(f"{sym}: {reason}")
            if existing is not None and (existing.get("sector_index") or "") == target_sector:
                snapshots.append(existing)

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
    return cached_company, cached_sector, target_sector, refresh_meta


def _render_insufficient_data_panel(symbol: str, cached: dict[str, Any] | None, ready_errors: list[str]) -> None:
    st.error("Değerleme için yeterli veri yok")
    data_quality_errors = (cached or {}).get("data_quality_errors_json") if cached else []
    missing_fields = (cached or {}).get("missing_fields_json") if cached else []
    panel = pd.DataFrame(
        [
            ("Eksik kritik alanlar", ", ".join(ready_errors) if ready_errors else "N/A"),
            ("Son cache zamanı", fmt_text((cached or {}).get("updated_at"))),
            ("data_quality_status", fmt_text((cached or {}).get("data_quality_status"))),
            ("data_quality_errors_json", fmt_text(data_quality_errors)),
            ("missing_fields_json", fmt_text(missing_fields)),
        ],
        columns=["Alan", "Değer"],
    )
    st.dataframe(panel, use_container_width=True, hide_index=True)
    st.code(f"docker compose run --rm valuation-app python scripts/debug_borsapy_symbol.py {symbol}")


def _render_scenario_tab(scenario: Any, title: str) -> None:
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("Kullanılan Dönem", fmt_text(scenario.period_label))
    c2.metric("Kullanılan Net Kâr", fmt_money(scenario.net_income))
    c3.metric("Net Kâr Kaynağı", fmt_text(scenario.net_income_source))
    c4, c5 = st.columns(2)
    c4.metric("Medyan Adil Değer", fmt_money(scenario.fair_value_median))
    c5.metric("Prim Potansiyeli", fmt_pct(scenario.upside_potential_pct))
    st.dataframe(build_target_price_dataframe(scenario), use_container_width=True, hide_index=True)

    st.markdown("#### Ödenmiş Sermaye Detayı")
    details = scenario.paid_capital_details
    info = pd.DataFrame(
        [
            ("EPS", fmt_money(details.get("eps"))),
            ("Kurs formülü: EPS × 10", fmt_money(details.get("x10"))),
            ("Geçmiş F/K medyan", fmt_number(details.get("historical_pe_median"))),
            ("EPS × Geçmiş F/K", fmt_money(details.get("historical_pe_value"))),
            ("Sektör F/K medyan", fmt_number(details.get("sector_pe_median"))),
            ("EPS × Sektör F/K", fmt_money(details.get("sector_pe_value"))),
            ("EPS × Güncel F/K", fmt_money(details.get("current_pe_value"))),
            ("Final değer", fmt_money(details.get("final"))),
            ("Final yöntemi", fmt_text(details.get("final_method"))),
        ],
        columns=["Alan", "Değer"],
    )
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.info(
        "x10 tek başına ana adil değeri belirlemez. Geçmiş F/K yoksa sektör/güncel F/K fallback kullanılır. "
        "Final boşsa x10 sadece bilgi amaçlıdır."
    )


def main() -> None:
    st.set_page_config(page_title="BIST Otomatik Değerleme", layout="wide")
    st.markdown(
        """
        <style>
        .small-note { color: #6b7280; font-size: 0.9rem; }
        div[data-testid="stMetric"] {
            background: #131a24;
            border: 1px solid #2a3445;
            padding: 10px;
            border-radius: 8px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #e6edf3;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #9fb0c7;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #9fd3ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("BIST Otomatik Değerleme")
    st.caption("Bu çalışma yatırım tavsiyesi değildir. Veriler otomatik kaynaklardan alınır; eksik veya hatalı veri olabilir.")

    init_db(DB_PATH)
    if "analyze_requested" not in st.session_state:
        st.session_state["analyze_requested"] = False
    if "last_refresh_meta" not in st.session_state:
        st.session_state["last_refresh_meta"] = None

    with st.sidebar:
        st.header("Girdi")
        symbol = st.text_input("Hisse kodu", value="THYAO").strip().upper()
        analyze = st.button("Analiz")
        refresh = st.button("Cache’i yenile")
        st.markdown("<div class='small-note'>Cache taze ise borsapy yeniden çağrılmaz.</div>", unsafe_allow_html=True)

    if analyze:
        st.session_state["analyze_requested"] = True

    if refresh and symbol:
        try:
            with st.spinner("Cache yenileniyor..."):
                _, _, _, refresh_meta = _refresh_symbol_and_sector(symbol, DB_PATH, force=True)
            st.session_state["last_refresh_meta"] = refresh_meta
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cache yenileme hatası: {exc}")
            return

    if not st.session_state["analyze_requested"]:
        st.info("Analiz başlatmak için hisse kodu girip Analiz düğmesine basın.")
        return
    if not symbol:
        st.error("Hisse kodu girin.")
        return

    cached = get_company_snapshot(DB_PATH, symbol)
    company_usable = cached is not None and is_snapshot_usable(cached)
    company_fresh = cached is not None and not is_stale(cached.get("updated_at")) and company_usable
    sector_index = (cached or {}).get("sector_index") or get_sector_index_for_symbol(symbol)
    sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
    sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))
    valuation_source = "cache"

    if not company_fresh:
        try:
            with st.spinner("Şirket ve sektör cache güncelleniyor..."):
                _, _, _, refresh_meta = _refresh_symbol_and_sector(symbol, DB_PATH)
            st.session_state["last_refresh_meta"] = refresh_meta
            cached = get_company_snapshot(DB_PATH, symbol)
            sector_index = (cached or {}).get("sector_index") or sector_index
            sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
            company_fresh = cached is not None and not is_stale(cached.get("updated_at"))
            sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))
            valuation_source = "borsapy_refresh"
        except Exception as exc:  # noqa: BLE001
            st.error(f"Veri çekme hatası: {exc}")
            return
    elif not sector_fresh:
        try:
            with st.spinner("Sektör cache güncelleniyor..."):
                _, _, _, refresh_meta = _refresh_symbol_and_sector(symbol, DB_PATH)
            st.session_state["last_refresh_meta"] = refresh_meta
            cached = get_company_snapshot(DB_PATH, symbol)
            sector_index = (cached or {}).get("sector_index") or sector_index
            sector_metrics = get_sector_metrics(DB_PATH, sector_index) if sector_index else None
            sector_fresh = sector_metrics is not None and not is_stale(sector_metrics.get("calculated_at"))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Sektör yenileme hatası: {exc}")

    if cached is None:
        _render_insufficient_data_panel(symbol, None, ["cache_kaydi_bulunamadi"])
        return
    if not is_snapshot_usable(cached):
        _render_insufficient_data_panel(symbol, cached, ["snapshot_usable_degil"])
        return
    valuation_ready, valuation_ready_errors = is_snapshot_valuation_ready(cached)
    if not valuation_ready:
        _render_insufficient_data_panel(symbol, cached, valuation_ready_errors)
        return

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

    with st.sidebar:
        st.header("Cache Durumu")
        company_cache_status = "fresh" if company_fresh else ("stale" if cached else "missing")
        sector_cache_status = "fresh" if sector_fresh else ("stale" if sector_metrics else "missing")
        sidebar_rows = [
            render_status_badge("company_cache_status", company_cache_status),
            render_status_badge("sector_cache_status", sector_cache_status),
            render_status_badge("data_quality_status", cached.get("data_quality_status")),
            render_status_badge("valuation_source", valuation_source),
            render_status_badge("updated_at", cached.get("updated_at")),
        ]
        for row in sidebar_rows:
            st.caption(row)

    year_end = result.valuation_scenarios["year_end"]
    ttm = result.valuation_scenarios["ttm"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Güncel Fiyat", fmt_money(result.price))
    col2.metric("Yıl Sonu Adil Değer", fmt_money(year_end.fair_value_median))
    col3.metric("Yıl Sonu Prim Potansiyeli", fmt_pct(year_end.upside_potential_pct))
    col4.metric("TTM Adil Değer", fmt_money(ttm.fair_value_median))
    col5.metric("F/K", fmt_number(result.pe_ratio))
    col6.metric("PD/DD", fmt_number(result.pb_ratio))

    tabs = st.tabs(
        [
            "Özet",
            "Yıl Sonu Tahmini",
            "Son 12 Ay / TTM",
            "Sektör Karşılaştırması",
            "Veri Kalitesi",
            "Debug",
        ]
    )

    with tabs[0]:
        if year_end.fair_value_median is None:
            st.warning("Adil değer hesaplanamadı.")
        elif (year_end.upside_potential_pct or 0) > 20:
            st.info("Model pozitif prim potansiyeli hesaplıyor.")
        elif -10 <= (year_end.upside_potential_pct or 0) <= 20:
            st.info("Adil değer güncel fiyata yakın.")
        else:
            st.warning("Model güncel fiyatın üzerinde değerleme riski işaret ediyor.")
        st.dataframe(build_summary_dataframe(result, cached), use_container_width=True, hide_index=True)
        scenario_compare = pd.DataFrame(
            [
                (
                    "Yıl Sonu Tahmini",
                    fmt_money(year_end.net_income),
                    fmt_money(year_end.fair_value_median),
                    fmt_money(year_end.fair_value_mean_filtered),
                    fmt_pct(year_end.upside_potential_pct),
                    len(year_end.included_methods),
                    len(year_end.excluded_methods),
                ),
                (
                    "Son 12 Ay / TTM",
                    fmt_money(ttm.net_income),
                    fmt_money(ttm.fair_value_median),
                    fmt_money(ttm.fair_value_mean_filtered),
                    fmt_pct(ttm.upside_potential_pct),
                    len(ttm.included_methods),
                    len(ttm.excluded_methods),
                ),
            ],
            columns=[
                "Senaryo",
                "Kullanılan Net Kâr",
                "Medyan Adil Değer",
                "Filtreli Ortalama",
                "Prim Potansiyeli",
                "Dahil Edilen Yöntem Sayısı",
                "Hariç Tutulan Yöntem Sayısı",
            ],
        )
        st.dataframe(scenario_compare, use_container_width=True, hide_index=True)

    with tabs[1]:
        _render_scenario_tab(year_end, "Yıl Sonu Tahmini Değerleme")

    with tabs[2]:
        _render_scenario_tab(ttm, "Son 12 Ay / TTM Değerleme")

    with tabs[3]:
        st.dataframe(
            build_sector_dataframe(result, sector_metrics, sector_comparison, sector_index, sector_name),
            use_container_width=True,
            hide_index=True,
        )
        company_pe = sector_comparison.get("company_pe")
        sector_pe = sector_comparison.get("sector_pe_median")
        company_pb = sector_comparison.get("company_pb")
        sector_pb = sector_comparison.get("sector_pb_median")
        company_roe = sector_comparison.get("company_roe")
        sector_roe = sector_metrics.get("roe_aggregate") if sector_metrics else None
        notes: list[str] = []
        if isinstance(company_pe, (int, float)) and isinstance(sector_pe, (int, float)):
            notes.append(
                "Şirket F/K bazında sektöre göre iskontolu görünüyor."
                if company_pe < sector_pe
                else "Şirket F/K bazında sektörün üzerinde fiyatlanıyor."
            )
        if isinstance(company_pb, (int, float)) and isinstance(sector_pb, (int, float)):
            notes.append(
                "Şirket PD/DD bazında sektöre göre iskontolu görünüyor."
                if company_pb < sector_pb
                else "Şirket PD/DD bazında sektörün üzerinde fiyatlanıyor."
            )
        if not isinstance(company_roe, (int, float)) or not isinstance(sector_roe, (int, float)):
            notes.append("ROE karşılaştırması için yeterli veri yok.")
        st.info(" ".join(notes) if notes else "Sektör karşılaştırması hazır.")

    with tabs[4]:
        refresh_meta = st.session_state.get("last_refresh_meta") or {}
        quality_rows = [
            ("company_cache_status", "fresh" if company_fresh else ("stale" if cached else "missing")),
            ("sector_cache_status", "fresh" if sector_fresh else ("stale" if sector_metrics else "missing")),
            ("valuation_source", valuation_source),
            ("updated_at", cached.get("updated_at")),
            ("data_quality_status", cached.get("data_quality_status")),
            ("data_quality_errors_json", cached.get("data_quality_errors_json")),
            ("missing_fields_json", cached.get("missing_fields_json")),
            ("net_income_source", cached.get("net_income_source")),
            ("equity_source", cached.get("equity_source")),
            ("revenue_source", cached.get("revenue_source")),
            ("refresh.wrote_to_cache", refresh_meta.get("wrote_to_cache")),
            ("refresh.preserved_existing_cache", refresh_meta.get("preserved_existing_cache")),
            ("refresh.rejected_unusable_refresh", refresh_meta.get("rejected_unusable_refresh")),
            ("refresh.messages", refresh_meta.get("messages")),
        ]
        st.dataframe(pd.DataFrame(quality_rows, columns=["Alan", "Değer"]), use_container_width=True, hide_index=True)

    with tabs[5]:
        with st.expander("Cached company snapshot"):
            st.json(to_plain_dict(cached))
        with st.expander("Sector metrics"):
            st.json(to_plain_dict(sector_metrics))
        with st.expander("Sector comparison raw"):
            st.json(to_plain_dict(sector_comparison))
        with st.expander("Valuation result raw"):
            st.json(to_plain_dict(result))

    st.info("Yatırım tavsiyesi değildir.")


if __name__ == "__main__":
    main()

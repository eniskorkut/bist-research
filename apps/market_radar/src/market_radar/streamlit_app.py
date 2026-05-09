from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from market_radar.data_access import DB_PATH, DEFAULT_BIST_UNIVERSE_INDEX, get_default_ohlcv_cache_ttl_minutes, init_db, load_bist_universe
from market_radar.radar_engine import RadarConfig, RadarResult, ScanResult, scan_symbols


def _threshold_select(label: str, options: list[float], default: float) -> float:
    idx = options.index(default) if default in options else 0
    return float(st.selectbox(label, options, index=idx, format_func=lambda v: f"{v:,.1f}" if v >= 10 else f"{v:.1f}"))


def _render_summary_cards(scan: ScanResult, universe_count: int) -> None:
    results = scan.results
    positive_count = len(results)
    highest_score = max((r.interest_score for r in results), default=0)
    volume_spike_count = sum(1 for r in results if (r.volume_ratio_20d or 0) >= 1.5)
    breakout_count = sum(1 for r in results if r.breakout_20d)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Taranan Hisse", scan.scan_summary.get("successful_symbols", universe_count))
    c2.metric("Pozitif Sinyal", positive_count)
    c3.metric("En Yüksek Skor", f"{highest_score:.0f}")
    c4.metric("Hacim Patlaması", volume_spike_count)
    c5.metric("Breakout", breakout_count)


def _render_scan_info(scan: ScanResult, cache_source: str) -> None:
    summary = scan.scan_summary
    st.markdown("#### Tarama Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taranan sembol", summary.get("scanned_symbols", 0))
    c2.metric("Başarılı", summary.get("successful_symbols", 0))
    c3.metric("Hatalı", summary.get("failed_symbols", 0))
    c4.metric("Sonuç üreten", summary.get("result_count", 0))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Evren", summary.get("index", DEFAULT_BIST_UNIVERSE_INDEX))
    p2.metric("Paralel işçi", summary.get("max_workers", "N/A"))
    p3.metric("OHLCV TTL (dk)", summary.get("ohlcv_cache_ttl_minutes", "N/A"))
    p4.metric("Süre (sn)", summary.get("elapsed_seconds", "N/A"))
    st.caption(
        "Evren kaynağı: "
        f"`{summary.get('universe_cache_source', cache_source)}` | "
        f"Scan cache kaynağı: `{summary.get('scan_cache_source', 'live_scan')}` | "
        f"Evren sembol sayısı: `{summary.get('universe_symbol_count', 0)}`"
    )


def _render_failed_symbols(scan: ScanResult) -> None:
    if not scan.failed_symbols:
        return
    with st.expander(f"Hata veren semboller ({len(scan.failed_symbols)})"):
        df = pd.DataFrame(scan.failed_symbols)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_filters_panel(config: RadarConfig) -> None:
    st.markdown("#### Filtre Açıklamaları")
    notes = [
        ("Minimum Ortalama TL Hacim", "Son 20 günün ortalama TL hacmi belirlenen eşik üstünde olmalı."),
        ("RVOL 20D / Hacim Oranı", "Bugünkü hacmin son 20 günlük ortalama hacme oranı."),
        ("TL Hacim Oranı", "Bugünkü TL hacmin son 20 günlük ortalama TL hacme oranı."),
        ("Günlük Fiyat Değişimi", "Günlük kapanış değişimi belirlenen eşik üstünde olmalı."),
        ("Kapanış Gücü", "Kapanışın gün içi aralıktaki konumu güçlü olmalı."),
        ("20 Günlük Zirve", "Kapanış 20 günlük zirveyi kırmalı veya ona yaklaşmalı."),
        ("MA20 Üstü", "Kapanış 20 günlük hareketli ortalamanın üstünde olmalı."),
        ("MA50 Üstü", "Kapanış 50 günlük hareketli ortalamanın üstünde olmalı."),
        ("XU100'e Göre Güçlü", "Hissenin günlük getirisi XU100'ü geçmeli."),
        ("Minimum İlgi Skoru", "Toplam ilgi skoru belirlenen eşik üstünde olmalı."),
        ("Negatif fiyat hareketlerini dahil et", "Kapalıysa negatif günlük performans sonuçtan çıkarılır."),
    ]
    active = []
    if config.min_avg_turnover_try_active:
        active.append(f"Minimum Ortalama TL Hacim: {config.min_avg_turnover_try:,.0f}")
    if config.min_volume_ratio_active:
        active.append(f"RVOL 20D: {config.min_volume_ratio:.1f}x")
    if config.min_turnover_ratio_active:
        active.append(f"TL Hacim Oranı: {config.min_turnover_ratio:.1f}x")
    if config.min_daily_return_active:
        active.append(f"Günlük Fiyat Değişimi: {config.min_daily_return:.1f}%")
    if config.min_close_position_active:
        active.append(f"Kapanış Gücü: {config.min_close_position:.2f}")
    if config.breakout_mode != "off":
        active.append(f"20 Günlük Zirve: {config.breakout_mode}")
    if config.require_ma20_active:
        active.append("MA20 Üstü")
    if config.require_ma50_active:
        active.append("MA50 Üstü")
    if config.min_xu100_relative_active:
        active.append(f"XU100'e Göre Güçlü: {config.min_xu100_relative:.1f}%")
    if config.min_interest_score_active:
        active.append(f"Minimum İlgi Skoru: {config.min_interest_score:.0f}")
    if not config.include_negative_moves:
        active.append("Negatif fiyat hareketleri dışarıda")
    st.caption("Aktif filtreler: " + " | ".join(active))
    for title, desc in notes:
        st.caption(f"{title}: {desc}")


def _render_details(results: list[RadarResult]) -> None:
    for result in results:
        with st.expander(f"{result.symbol} - Skor {result.interest_score:.0f}"):
            detail = pd.DataFrame(
                [
                    ("Date", result.date),
                    ("Signals", ", ".join(result.signals)),
                    ("Passed Filters", ", ".join(result.passed_filters)),
                    ("Failed Filters", ", ".join(result.failed_filters)),
                    ("Open", result.open),
                    ("High", result.high),
                    ("Low", result.low),
                    ("Close", result.close),
                    ("Prev Close", result.prev_close),
                    ("Volume", result.volume),
                    ("Avg Volume 20D", result.avg_volume_20d),
                    ("Volume Ratio 20D", result.volume_ratio_20d),
                    ("Turnover TRY", result.turnover_try),
                    ("Avg Turnover 20D", result.avg_turnover_20d),
                    ("Turnover Ratio 20D", result.turnover_ratio_20d),
                    ("Daily Return %", result.daily_return_pct),
                    ("Day Range %", result.day_range_pct),
                    ("Close Position", result.close_position),
                    ("MA20", result.ma20),
                    ("MA50", result.ma50),
                    ("Breakout 20D", result.breakout_20d),
                    ("High 52W", result.high_52w),
                    ("Near 52W High", result.near_52w_high),
                    ("XU100 Daily Return %", result.xu100_daily_return_pct),
                    ("XU100 Relative %", result.xu100_relative_return_pct),
                    ("Sector Relative %", result.sector_relative_return_pct),
                ],
                columns=["Alan", "Değer"],
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)
            st.json(asdict(result))


def render_positive_interest_radar_page(*, embedded: bool = False) -> None:
    if not embedded:
        st.set_page_config(page_title="BIST Pozitif İlgi Radarı", layout="wide")
    st.title("BIST Pozitif İlgi Radarı")
    st.caption("Bu ekran yatırım tavsiyesi değildir. Hacim, fiyat ve teknik hareketlenme metrikleriyle araştırma listesi üretir.")
    init_db(DB_PATH)
    if "radar_scan" not in st.session_state:
        st.session_state["radar_scan"] = None
    if "radar_cache_source" not in st.session_state:
        st.session_state["radar_cache_source"] = None

    with st.sidebar:
        st.header("Girdi")
        universe_index = st.selectbox("Tarama evreni", ["XU030", "XU100", "XUTUM"], index=2)
        st.caption("Manuel sembol girişi yok; semboller borsapy endeks bileşenlerinden otomatik alınır.")
        lookback_days = st.number_input("Lookback days", min_value=60, value=260, step=10)
        max_workers = int(st.selectbox("Paralel işçi sayısı", [4, 8, 12, 16], index=1))
        ttl_label = st.selectbox("OHLCV Cache TTL", ["Otomatik", "15 dakika", "60 dakika", "6 saat", "24 saat"], index=0)
        ttl_map = {
            "Otomatik": None,
            "15 dakika": 15,
            "60 dakika": 60,
            "6 saat": 360,
            "24 saat": 1440,
        }
        ohlcv_cache_ttl_minutes = ttl_map[ttl_label]
        use_scan_cache = st.checkbox("Scan result cache kullan", value=True)
        scan_cache_ttl_minutes = int(st.selectbox("Scan cache TTL", [5, 15, 30, 60], index=1))
        force_refresh = st.checkbox("Cache'i yenile", value=False)
        st.markdown("#### Filtreler")
        min_avg_turnover_try_active = st.checkbox("Minimum Ortalama TL Hacim", value=True)
        min_avg_turnover_try = _threshold_select("Ortalama TL Hacim Eşiği", [5_000_000.0, 10_000_000.0, 25_000_000.0, 50_000_000.0, 100_000_000.0], 10_000_000.0)
        min_volume_ratio_active = st.checkbox("RVOL 20D / Hacim Oranı", value=True)
        min_volume_ratio = _threshold_select("RVOL Eşiği", [1.2, 1.5, 2.0, 3.0], 1.5)
        min_turnover_ratio_active = st.checkbox("TL Hacim Oranı", value=True)
        min_turnover_ratio = _threshold_select("TL Hacim Oranı Eşiği", [1.2, 1.5, 2.0, 3.0], 1.5)
        min_daily_return_active = st.checkbox("Günlük Fiyat Değişimi", value=True)
        min_daily_return = _threshold_select("Günlük Getiri Eşiği", [0.0, 1.0, 2.0, 3.0, 5.0], 0.0)
        min_close_position_active = st.checkbox("Kapanış Gücü", value=True)
        min_close_position = _threshold_select("Kapanış Gücü Eşiği", [0.60, 0.65, 0.75, 0.85], 0.65)
        breakout_mode = st.selectbox("20 Günlük Zirve Kırılımı", ["off", "breakout_20d", "near_20d_high_2pct"], index=0, format_func=lambda v: {"off": "Kapalı", "breakout_20d": "20 günlük zirveyi kıranlar", "near_20d_high_2pct": "20 günlük zirveye %2 yakın olanlar"}[v])
        require_ma20_active = st.checkbox("MA20 Üstü", value=True)
        require_ma50_active = st.checkbox("MA50 Üstü", value=False)
        min_xu100_relative_active = st.checkbox("XU100'e Göre Güçlü", value=True)
        min_xu100_relative = _threshold_select("XU100 Relative Eşiği", [0.0, 1.0, 2.0, 3.0], 0.0)
        min_interest_score_active = st.checkbox("Minimum İlgi Skoru", value=True)
        min_interest_score = _threshold_select("İlgi Skoru Eşiği", [40.0, 50.0, 60.0, 70.0, 80.0], 50.0)
        include_negative_moves = st.checkbox("Negatif fiyat hareketlerini dahil et", value=False)
        start_scan = st.button("Taramayı Başlat")
        _render_filters_panel(
            RadarConfig(
                lookback_days=int(lookback_days),
                min_avg_turnover_try_active=min_avg_turnover_try_active,
                min_avg_turnover_try=float(min_avg_turnover_try),
                min_volume_ratio_active=min_volume_ratio_active,
                min_volume_ratio=float(min_volume_ratio),
                min_turnover_ratio_active=min_turnover_ratio_active,
                min_turnover_ratio=float(min_turnover_ratio),
                min_daily_return_active=min_daily_return_active,
                min_daily_return=float(min_daily_return),
                min_close_position_active=min_close_position_active,
                min_close_position=float(min_close_position),
                breakout_mode=breakout_mode,
                require_ma20_active=require_ma20_active,
                require_ma50_active=require_ma50_active,
                min_xu100_relative_active=min_xu100_relative_active,
                min_xu100_relative=float(min_xu100_relative),
                min_interest_score_active=min_interest_score_active,
                min_interest_score=float(min_interest_score),
                include_negative_moves=include_negative_moves,
                force_refresh=force_refresh,
                db_path=DB_PATH,
                max_workers=max_workers,
                ohlcv_cache_ttl_minutes=ohlcv_cache_ttl_minutes,
                use_scan_cache=use_scan_cache,
                scan_cache_ttl_minutes=scan_cache_ttl_minutes,
                index_symbol=universe_index,
            )
        )

    # Load universe
    try:
        raw_symbols, cache_source = load_bist_universe(universe_index, db_path=DB_PATH, force=force_refresh)
    except Exception as exc:
        st.error("BIST hisse evreni alınamadı. borsapy XUTUM bileşenlerini döndürmedi.")
        st.caption(str(exc))
        return
    if not raw_symbols:
        st.error("BIST hisse evreni boş döndü. Tarama başlatılamadı.")
        return
    st.caption(f"Tarama evreni: {len(raw_symbols)} BIST hissesi | Evren kaynağı: `{cache_source}`")

    config = RadarConfig(
        lookback_days=int(lookback_days),
        min_avg_turnover_try_active=min_avg_turnover_try_active,
        min_avg_turnover_try=float(min_avg_turnover_try),
        min_volume_ratio_active=min_volume_ratio_active,
        min_volume_ratio=float(min_volume_ratio),
        min_turnover_ratio_active=min_turnover_ratio_active,
        min_turnover_ratio=float(min_turnover_ratio),
        min_daily_return_active=min_daily_return_active,
        min_daily_return=float(min_daily_return),
        min_close_position_active=min_close_position_active,
        min_close_position=float(min_close_position),
        breakout_mode=breakout_mode,
        require_ma20_active=require_ma20_active,
        require_ma50_active=require_ma50_active,
        min_xu100_relative_active=min_xu100_relative_active,
        min_xu100_relative=float(min_xu100_relative),
        min_interest_score_active=min_interest_score_active,
        min_interest_score=float(min_interest_score),
        include_negative_moves=include_negative_moves,
        force_refresh=force_refresh,
        db_path=DB_PATH,
        max_workers=max_workers,
        ohlcv_cache_ttl_minutes=ohlcv_cache_ttl_minutes,
        use_scan_cache=use_scan_cache,
        scan_cache_ttl_minutes=scan_cache_ttl_minutes,
        index_symbol=universe_index,
    )

    if start_scan:
        progress_bar = st.progress(0, text="Tarama başlatılıyor...")
        status_text = st.empty()

        def _progress_callback(current: int, total: int, symbol: str) -> None:
            pct = current / total if total > 0 else 0
            progress_bar.progress(pct, text=f"Taranıyor: {current} / {total}")
            status_text.caption(f"Son taranan: `{symbol}`")

        try:
            scan = scan_symbols(
                raw_symbols,
                config=config,
                progress_callback=_progress_callback,
                universe_source=cache_source,
            )
            st.session_state["radar_scan"] = scan
            st.session_state["radar_cache_source"] = cache_source
            if scan.scan_summary.get("scan_cache_source") == "scan_cache":
                progress_bar.empty()
                status_text.empty()
                st.info("Sonuç cache'ten getirildi.")
            else:
                progress_bar.progress(1.0, text="Tarama tamamlandı.")
                status_text.empty()
        except Exception as exc:
            st.session_state["radar_scan"] = None
            st.session_state["radar_cache_source"] = None
            progress_bar.empty()
            status_text.empty()
            st.error("Radar verisi alınamadı. borsapy geçici olarak veri döndürmemiş olabilir.")
            st.caption(str(exc))
    elif st.session_state.get("radar_scan") is None:
        st.info("Filtreleri ayarla ve tüm Borsa İstanbul evrenini taramak için Taramayı Başlat'a bas.")

    scan: ScanResult | None = st.session_state.get("radar_scan")
    stored_cache_source = st.session_state.get("radar_cache_source") or cache_source

    if scan is not None:
        if scan.scan_summary.get("scan_cache_source") == "scan_cache":
            st.caption("Tarama bu koşullar için cache'ten hızlı getirildi.")
        if config.ohlcv_cache_ttl_minutes is None:
            st.caption(f"OHLCV TTL (otomatik): {get_default_ohlcv_cache_ttl_minutes()} dakika")
        _render_scan_info(scan, stored_cache_source)
        _render_summary_cards(scan, len(raw_symbols))
        _render_failed_symbols(scan)

        results = scan.results
        if results:
            df = pd.DataFrame([result.to_row() for result in results]).sort_values("Interest Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Filtreleri gevşet veya daha geniş bir sembol listesi dene.")
        _render_details(results)
    else:
        # No scan yet - show empty state
        pass


def main() -> None:
    render_positive_interest_radar_page(embedded=False)


if __name__ == "__main__":
    main()

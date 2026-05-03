from __future__ import annotations

import streamlit as st

from valuation.valuation_engine import run_valuation


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


st.set_page_config(page_title="BIST Otomatik Degerleme", layout="wide")
st.title("BIST Otomatik Degerleme")
st.caption("Yatirim tavsiyesi degildir.")

symbol = st.text_input("Hisse kodu", value="THYAO").strip().upper()
run_button = st.button("Analiz")

if run_button:
    if not symbol:
        st.error("Hisse kodu girin.")
    else:
        try:
            result = run_valuation(symbol)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analiz hatasi: {exc}")
        else:
            st.subheader("Temel Veriler")
            col1, col2, col3 = st.columns(3)
            col1.metric("Hisse Fiyati", _fmt(result.price))
            col2.metric("Piyasa Degeri", _fmt(result.market_cap))
            col3.metric("Hisse Sayisi", _fmt(result.shares_outstanding))

            col4, col5, col6 = st.columns(3)
            col4.metric("Hisse F/K", _fmt(result.pe_ratio))
            col5.metric("Hisse PD/DD", _fmt(result.pb_ratio))
            col6.metric("Ozkaynak", _fmt(result.equity))

            st.subheader("Tahmini Net Kar")
            st.write(f"Donem: `{result.estimation.period_label}` ({result.estimation.period_type})")
            st.write(f"Secilen yontem: `{result.estimation.selected_method}`")
            st.write(f"Tahmini yillik net kar: `{_fmt(result.estimated_net_income)}`")
            st.json(result.estimation.method_values)

            st.subheader("5 Yontem Hedef Fiyat")
            st.table(
                {
                    "Yontem": list(result.target_prices.keys()),
                    "Hedef Fiyat": [_fmt(value) for value in result.target_prices.values()],
                }
            )
            st.write(f"5 yontem ortalamasi: `{_fmt(result.average_target_price)}`")
            st.write(f"Prim potansiyeli (%): `{_fmt(result.upside_potential_pct)}`")

            if result.missing_fields:
                st.warning("Eksik veri alanlari: " + ", ".join(sorted(set(result.missing_fields))))

            st.info("Bu ciktidaki tum finansal veriler borsapy kaynagindan cekilir. Yatirim tavsiyesi degildir.")

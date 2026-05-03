from __future__ import annotations

import pandas as pd

from valuation.data_access import _extract_first_number, _extract_series_value, _normalize_label, _safe_mapping


class _WithToDict:
    def todict(self):
        return {"last": 123.45, "marketCap": 999}


def test_safe_mapping_prefers_todict() -> None:
    obj = _WithToDict()
    data = _safe_mapping(obj)
    assert data["last"] == 123.45
    assert data["marketCap"] == 999


def test_extract_aliases_new_keys() -> None:
    mapping = {
        "last": 10.0,
        "marketCap": 1000.0,
        "sharesOutstanding": 100.0,
        "trailingPE": 5.0,
        "priceToBook": 1.2,
    }
    assert _extract_first_number(mapping, ["last", "close"]) == 10.0
    assert _extract_first_number(mapping, ["marketcap"]) == 1000.0
    assert _extract_first_number(mapping, ["sharesoutstanding"]) == 100.0
    assert _extract_first_number(mapping, ["trailingpe"]) == 5.0
    assert _extract_first_number(mapping, ["pricetobook"]) == 1.2


def test_normalize_label_turkish_and_punctuation() -> None:
    assert _normalize_label("Özkaynaklar (Ana-Ortaklık)/Toplam") == "ozkaynaklar ana ortaklik toplam"


def test_extract_series_value_contains_and_transpose() -> None:
    df = pd.DataFrame(
        {"2025": [1000.0], "2024": [900.0]},
        index=["Ana Ortaklığa Ait Özkaynaklar"],
    )
    val, _, src = _extract_series_value(df, ["ana ortakliga ait ozkaynaklar"])
    assert val == 1000.0
    assert src == "financial_statement"

    df_t = df.T
    val_t, _, src_t = _extract_series_value(df_t, ["ana ortakliga ait ozkaynaklar"])
    assert val_t == 1000.0
    assert src_t in {"financial_statement", "transposed_financial_statement"}

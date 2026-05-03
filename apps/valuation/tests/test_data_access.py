from __future__ import annotations

from valuation.data_access import _extract_first_number, _safe_mapping


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

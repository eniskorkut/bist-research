from __future__ import annotations

from valuation.symbols import normalize_bist_symbol, validate_bist_symbol


def test_normalize_lower() -> None:
    assert normalize_bist_symbol("asels") == "ASELS"


def test_normalize_upper() -> None:
    assert normalize_bist_symbol("ASELS") == "ASELS"


def test_normalize_odine_lower() -> None:
    assert normalize_bist_symbol("odine") == "ODINE"


def test_normalize_odine_upper() -> None:
    assert normalize_bist_symbol("ODINE") == "ODINE"


def test_normalize_turkish_i() -> None:
    assert normalize_bist_symbol("ODİNE") == "ODINE"


def test_normalize_is_suffix() -> None:
    assert normalize_bist_symbol("thyao.is") == "THYAO"


def test_normalize_whitespace() -> None:
    assert normalize_bist_symbol(" thy ao ") == "THYAO"


def test_validate_valid_symbol() -> None:
    ok, err = validate_bist_symbol("ASELS")
    assert ok is True
    assert err is None


def test_validate_empty_symbol() -> None:
    ok, err = validate_bist_symbol("")
    assert ok is False
    assert err == "Hisse kodu boş olamaz"


def test_validate_too_long_symbol() -> None:
    ok, err = validate_bist_symbol("ASELSAN")
    assert ok is False
    assert err == "Geçerli BIST kodu gibi görünmüyor"


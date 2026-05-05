from __future__ import annotations

import re


_TR_MAP = str.maketrans(
    {
        "İ": "I",
        "I": "I",
        "ı": "I",
        "i": "I",
        "Ş": "S",
        "ş": "S",
        "Ğ": "G",
        "ğ": "G",
        "Ü": "U",
        "ü": "U",
        "Ö": "O",
        "ö": "O",
        "Ç": "C",
        "ç": "C",
    }
)


def normalize_bist_symbol(raw: str | None) -> str:
    if raw is None:
        return ""
    symbol = raw.strip()
    if not symbol:
        return ""
    symbol = symbol.translate(_TR_MAP)
    symbol = symbol.upper()
    if symbol.endswith(".IS"):
        symbol = symbol[:-3]
    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
    return symbol


def validate_bist_symbol(symbol: str) -> tuple[bool, str | None]:
    if not symbol:
        return False, "Hisse kodu boş olamaz"
    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        return False, "Geçerli BIST kodu gibi görünmüyor"
    if not (3 <= len(symbol) <= 6):
        return False, "Geçerli BIST kodu gibi görünmüyor"
    return True, None


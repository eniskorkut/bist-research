from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import borsapy as bp
import pandas as pd

from market_radar.symbols import normalize_bist_symbol

DB_PATH = "/data/market_radar_cache.sqlite"
DEFAULT_BIST_UNIVERSE_INDEX = "XUTUM"


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _normalize_history_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    keep = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep]
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index.name = "date"
    for column in df.columns:
        df[column] = df[column].apply(_to_float)
    return df.dropna(how="all")


def _frame_to_payload(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"index_name": "date", "records": []}
    records: list[dict[str, Any]] = []
    for idx, row in df.reset_index().iterrows():
        payload = row.to_dict()
        date_value = payload.get("date")
        if hasattr(date_value, "isoformat"):
            payload["date"] = date_value.isoformat()
        records.append(payload)
    return {"index_name": "date", "records": records}


def _payload_to_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    records = payload.get("records") or []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    return _normalize_history_frame(df)


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_ohlcv_cache (
                symbol TEXT PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS radar_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                source TEXT NOT NULL,
                interest_score REAL,
                metrics_json TEXT NOT NULL,
                signals_json TEXT NOT NULL,
                passed_filters_json TEXT NOT NULL,
                failed_filters_json TEXT NOT NULL
            )
            """
        )


def is_stale(fetched_at: str | None, max_age_minutes: int = 15) -> bool:
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at)
    except ValueError:
        return True
    age = datetime.now(UTC) - ts.astimezone(UTC)
    return age > timedelta(minutes=max_age_minutes)


def get_cached_history(db_path: str, symbol: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at, source, payload_json FROM daily_ohlcv_cache WHERE symbol = ?",
            (normalize_bist_symbol(symbol),),
        ).fetchone()
    if row is None:
        return pd.DataFrame(), None
    fetched_at, source, payload_json = row
    payload = json.loads(payload_json)
    frame = _payload_to_frame(payload)
    meta = {"fetched_at": fetched_at, "source": source, "payload": payload}
    return frame, meta


def upsert_cached_history(db_path: str, symbol: str, df: pd.DataFrame, source: str = "borsapy") -> None:
    init_db(db_path)
    normalized = normalize_bist_symbol(symbol)
    payload = _frame_to_payload(_normalize_history_frame(df))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO daily_ohlcv_cache (symbol, fetched_at, source, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                source=excluded.source,
                payload_json=excluded.payload_json
            """,
            (normalized, datetime.now(UTC).isoformat(), source, json.dumps(payload)),
        )


def save_radar_result(db_path: str, result: dict[str, Any]) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO radar_results (
                symbol, scanned_at, source, interest_score,
                metrics_json, signals_json, passed_filters_json, failed_filters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("symbol"),
                result.get("scanned_at") or datetime.now(UTC).isoformat(),
                result.get("source") or "borsapy",
                result.get("interest_score"),
                json.dumps(result.get("metrics", {})),
                json.dumps(result.get("signals", [])),
                json.dumps(result.get("passed_filters", [])),
                json.dumps(result.get("failed_filters", [])),
            ),
        )


def load_bist_universe(index_symbol: str = DEFAULT_BIST_UNIVERSE_INDEX) -> list[str]:
    index = bp.Index(normalize_bist_symbol(index_symbol))
    components = getattr(index, "component_symbols", []) or []
    if callable(components):
        components = components()
    symbols = {normalize_bist_symbol(str(symbol)) for symbol in components}
    return sorted(symbol for symbol in symbols if 3 <= len(symbol) <= 6 and symbol.isalnum())


class BorsapyMarketDataClient:
    def _fetch_history(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        ticker = bp.Ticker(normalize_bist_symbol(symbol))
        # borsapy subtracts this value from datetime.now() internally, so keep
        # it offset-naive to avoid mixed timezone arithmetic.
        start = datetime.now().replace(tzinfo=None) - timedelta(days=max(int(lookback_days * 1.8), lookback_days + 30))
        frame = ticker.history(start=start, interval="1d")
        return _normalize_history_frame(frame)

    def load_history(
        self,
        symbol: str,
        lookback_days: int = 260,
        *,
        db_path: str = DB_PATH,
        force: bool = False,
    ) -> pd.DataFrame:
        normalized = normalize_bist_symbol(symbol)
        cached_frame, meta = get_cached_history(db_path, normalized)
        if not force and meta is not None and not is_stale(meta.get("fetched_at"), max_age_minutes=15):
            return cached_frame
        try:
            frame = self._fetch_history(normalized, lookback_days)
            if not frame.empty:
                upsert_cached_history(db_path, normalized, frame)
                return frame
        except Exception:
            if not cached_frame.empty:
                return cached_frame
            raise
        return cached_frame

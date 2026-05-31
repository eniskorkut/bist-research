from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import borsapy as bp
import pandas as pd

from market_radar.symbols import normalize_bist_symbol

DB_PATH = "/data/market_radar_cache.sqlite"
DEFAULT_BIST_UNIVERSE_INDEX = "XUTUM"
ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
UTC = timezone.utc


@dataclass
class HistoryLoadResult:
    frame: pd.DataFrame
    symbol: str
    data_latest_date: str | None
    data_lag_days: int | None
    history_rows: int
    ohlcv_cache_fetched_at: str | None
    ohlcv_cache_age_minutes: float | None
    ohlcv_cache_status: str
    source: str


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


def _merge_history_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return _normalize_history_frame(incoming)
    if incoming is None or incoming.empty:
        return _normalize_history_frame(existing)
    merged = pd.concat([existing, incoming], axis=0)
    merged = _normalize_history_frame(merged)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


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
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_cache (
                index_symbol TEXT PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS radar_scan_cache (
                cache_key TEXT PRIMARY KEY,
                scanned_at TEXT NOT NULL,
                universe_source TEXT,
                universe_symbol_count INTEGER,
                results_json TEXT NOT NULL,
                raw_results_json TEXT NOT NULL,
                failed_symbols_json TEXT NOT NULL,
                scan_summary_json TEXT NOT NULL
            )
            """
        )


def get_default_ohlcv_cache_ttl_minutes(now: datetime | None = None) -> int:
    current = now.astimezone(ISTANBUL_TZ) if now is not None else datetime.now(ISTANBUL_TZ)
    weekday = current.weekday()
    if weekday >= 5:
        return 24 * 60
    hhmm = current.hour * 60 + current.minute
    market_open = 10 * 60
    market_close = 18 * 60 + 30
    if market_open <= hhmm <= market_close:
        return 15
    return 360


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


def save_radar_results_bulk(db_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    init_db(db_path)
    payload_rows = [
        (
            row.get("symbol"),
            row.get("scanned_at") or datetime.now(UTC).isoformat(),
            row.get("source") or "borsapy",
            row.get("interest_score"),
            json.dumps(row.get("metrics", {})),
            json.dumps(row.get("signals", [])),
            json.dumps(row.get("passed_filters", [])),
            json.dumps(row.get("failed_filters", [])),
        )
        for row in rows
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO radar_results (
                symbol, scanned_at, source, interest_score,
                metrics_json, signals_json, passed_filters_json, failed_filters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload_rows,
        )


def get_cached_scan_result(db_path: str, cache_key: str, max_age_minutes: int) -> dict[str, Any] | None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT scanned_at, universe_source, universe_symbol_count,
                   results_json, raw_results_json, failed_symbols_json, scan_summary_json
            FROM radar_scan_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    scanned_at = row[0]
    if is_stale(scanned_at, max_age_minutes=max_age_minutes):
        return None
    return {
        "scanned_at": scanned_at,
        "universe_source": row[1],
        "universe_symbol_count": row[2],
        "results": json.loads(row[3]),
        "raw_results": json.loads(row[4]),
        "failed_symbols": json.loads(row[5]),
        "scan_summary": json.loads(row[6]),
    }


def upsert_cached_scan_result(
    db_path: str,
    cache_key: str,
    *,
    universe_source: str,
    universe_symbol_count: int,
    results: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    failed_symbols: list[dict[str, Any]],
    scan_summary: dict[str, Any],
) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO radar_scan_cache (
                cache_key, scanned_at, universe_source, universe_symbol_count,
                results_json, raw_results_json, failed_symbols_json, scan_summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                scanned_at=excluded.scanned_at,
                universe_source=excluded.universe_source,
                universe_symbol_count=excluded.universe_symbol_count,
                results_json=excluded.results_json,
                raw_results_json=excluded.raw_results_json,
                failed_symbols_json=excluded.failed_symbols_json,
                scan_summary_json=excluded.scan_summary_json
            """,
            (
                cache_key,
                datetime.now(UTC).isoformat(),
                universe_source,
                universe_symbol_count,
                json.dumps(results),
                json.dumps(raw_results),
                json.dumps(failed_symbols),
                json.dumps(scan_summary),
            ),
        )


def get_cached_universe(db_path: str, index_symbol: str) -> dict[str, Any] | None:
    """Return cached universe row as dict, or None."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at, symbols_json, source FROM universe_cache WHERE index_symbol = ?",
            (normalize_bist_symbol(index_symbol),),
        ).fetchone()
    if row is None:
        return None
    return {"fetched_at": row[0], "symbols_json": json.loads(row[1]), "source": row[2]}


def upsert_cached_universe(db_path: str, index_symbol: str, symbols: list[str], source: str = "borsapy") -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO universe_cache (index_symbol, fetched_at, symbols_json, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(index_symbol) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                symbols_json=excluded.symbols_json,
                source=excluded.source
            """,
            (normalize_bist_symbol(index_symbol), datetime.now(UTC).isoformat(), json.dumps(symbols), source),
        )


def _fetch_bist_universe_from_borsapy(index_symbol: str) -> list[str]:
    """Fetch universe symbols directly from borsapy (no cache)."""
    index = bp.Index(normalize_bist_symbol(index_symbol))
    components = getattr(index, "component_symbols", []) or []
    if callable(components):
        components = components()
    symbols = {normalize_bist_symbol(str(symbol)) for symbol in components}
    return sorted(symbol for symbol in symbols if 3 <= len(symbol) <= 6 and symbol.isalnum())


def load_bist_universe(
    index_symbol: str = DEFAULT_BIST_UNIVERSE_INDEX,
    *,
    db_path: str = DB_PATH,
    force: bool = False,
    cache_only: bool = False,
) -> tuple[list[str], str]:
    """Load BIST universe symbols with 24h cache.

    Returns ``(symbols, cache_source)`` where *cache_source* is one of
    ``"fresh_cache"``, ``"stale_cache"``, or ``"borsapy"``.
    """
    init_db(db_path)
    cached = get_cached_universe(db_path, index_symbol)

    # If cache is fresh and not forced, return cached
    if cached is not None and not force and not is_stale(cached["fetched_at"], max_age_minutes=24 * 60):
        return cached["symbols_json"], "fresh_cache"

    if cache_only:
        if cached is not None and cached["symbols_json"]:
            return cached["symbols_json"], "stale_cache"
        raise RuntimeError(f"BIST universe ({index_symbol}) cache not found (cache_only mode).")

    # Try to fetch from borsapy
    try:
        symbols = _fetch_bist_universe_from_borsapy(index_symbol)
        if symbols:
            upsert_cached_universe(db_path, index_symbol, symbols)
            return symbols, "borsapy"
    except Exception:  # noqa: BLE001
        pass

    # Fallback to stale cache
    if cached is not None and cached["symbols_json"]:
        return cached["symbols_json"], "stale_cache"

    # No cache, re-raise
    raise RuntimeError(f"BIST universe ({index_symbol}) could not be fetched and no cache available.")


class BorsapyMarketDataClient:
    def _fetch_history(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        ticker = bp.Ticker(normalize_bist_symbol(symbol))
        # borsapy subtracts this value from datetime.now() internally, so keep
        # it offset-naive to avoid mixed timezone arithmetic.
        start = datetime.now().replace(tzinfo=None) - timedelta(days=max(int(lookback_days * 1.8), lookback_days + 30))
        frame = ticker.history(start=start, interval="1d")
        return _normalize_history_frame(frame)

    def load_history_with_meta(
        self,
        symbol: str,
        lookback_days: int = 260,
        *,
        db_path: str = DB_PATH,
        force: bool = False,
        cache_ttl_minutes: int | None = None,
        cache_only: bool = False,
    ) -> HistoryLoadResult:
        normalized = normalize_bist_symbol(symbol)
        cached_frame, meta = get_cached_history(db_path, normalized)
        ttl_minutes = cache_ttl_minutes if cache_ttl_minutes is not None else get_default_ohlcv_cache_ttl_minutes()
        now_utc = datetime.now(UTC)

        def _build_result(frame: pd.DataFrame, status: str, source: str, fetched_at: str | None) -> HistoryLoadResult:
            latest_date = None
            lag_days = None
            rows = len(frame.index) if frame is not None else 0
            if frame is not None and not frame.empty:
                idx = frame.index.max()
                if pd.notna(idx):
                    latest_date = pd.Timestamp(idx).date().isoformat()
                    lag_days = (datetime.now(ISTANBUL_TZ).date() - pd.Timestamp(idx).date()).days
            cache_age = None
            if fetched_at:
                try:
                    cache_age = (now_utc - datetime.fromisoformat(fetched_at).astimezone(UTC)).total_seconds() / 60.0
                except Exception:  # noqa: BLE001
                    cache_age = None
            return HistoryLoadResult(
                frame=frame,
                symbol=normalized,
                data_latest_date=latest_date,
                data_lag_days=lag_days,
                history_rows=rows,
                ohlcv_cache_fetched_at=fetched_at,
                ohlcv_cache_age_minutes=cache_age,
                ohlcv_cache_status=status,
                source=source,
            )

        if not force and meta is not None and not is_stale(meta.get("fetched_at"), max_age_minutes=ttl_minutes):
            return _build_result(cached_frame, "fresh_cache", str(meta.get("source") or "cache"), meta.get("fetched_at"))

        if cache_only:
            if not cached_frame.empty:
                fetched_at = meta.get("fetched_at") if meta else None
                status = "stale_cache" if meta is not None else "missing"
                return _build_result(cached_frame, status, str((meta or {}).get("source") or "cache"), fetched_at)
            return _build_result(pd.DataFrame(), "missing", "missing", None)
        retries = [2, 5]
        for attempt in range(len(retries) + 1):
            try:
                frame = self._fetch_history(normalized, lookback_days)
                if not frame.empty:
                    merged = _merge_history_frames(cached_frame, frame)
                    upsert_cached_history(db_path, normalized, merged)
                    refreshed_meta = get_cached_history(db_path, normalized)[1]
                    fetched_at = refreshed_meta.get("fetched_at") if refreshed_meta else None
                    return _build_result(merged, "live_fetch", "borsapy", fetched_at)
                break
            except Exception as exc:
                message = str(exc)
                is_rate_limited = ("429" in message) or ("Too Many Requests" in message)
                if is_rate_limited and attempt < len(retries):
                    time.sleep(retries[attempt])
                    continue
                if not cached_frame.empty:
                    fetched_at = meta.get("fetched_at") if meta else None
                    return _build_result(cached_frame, "fallback_cache", str((meta or {}).get("source") or "cache"), fetched_at)
                raise
        if not cached_frame.empty:
            fetched_at = meta.get("fetched_at") if meta else None
            status = "stale_cache" if meta is not None else "missing"
            return _build_result(cached_frame, status, str((meta or {}).get("source") or "cache"), fetched_at)
        return _build_result(pd.DataFrame(), "missing", "missing", None)

    def load_history(
        self,
        symbol: str,
        lookback_days: int = 260,
        *,
        db_path: str = DB_PATH,
        force: bool = False,
        cache_ttl_minutes: int | None = None,
        cache_only: bool = False,
    ) -> pd.DataFrame:
        return self.load_history_with_meta(
            symbol,
            lookback_days=lookback_days,
            db_path=db_path,
            force=force,
            cache_ttl_minutes=cache_ttl_minutes,
            cache_only=cache_only,
        ).frame

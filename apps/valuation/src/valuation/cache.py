from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS company_snapshot (
                symbol TEXT PRIMARY KEY,
                market TEXT,
                sector_index TEXT,
                sector_name TEXT,
                price REAL,
                market_cap REAL,
                shares_outstanding REAL,
                paid_in_capital REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                roe REAL,
                equity REAL,
                net_income_latest_period REAL,
                net_income_ttm REAL,
                estimated_net_income REAL,
                financial_period TEXT,
                period_type TEXT,
                source TEXT,
                missing_fields_json TEXT,
                data_quality_status TEXT,
                data_quality_errors_json TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS sector_metrics (
                sector_index TEXT PRIMARY KEY,
                sector_name TEXT,
                member_count INTEGER,
                valid_member_count INTEGER,
                pe_median REAL,
                pe_aggregate REAL,
                pb_median REAL,
                pb_aggregate REAL,
                roe_aggregate REAL,
                calculated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS valuation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                valuation_date TEXT,
                price REAL,
                fair_value REAL,
                upside_percent REAL,
                target_prices_json TEXT,
                estimation_json TEXT,
                sector_comparison_json TEXT,
                confidence_score REAL,
                created_at TEXT
            );
            """
        )
        # best-effort backward-compatible migrations
        for sql in [
            "ALTER TABLE company_snapshot ADD COLUMN data_quality_status TEXT",
            "ALTER TABLE company_snapshot ADD COLUMN data_quality_errors_json TEXT",
        ]:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass


def upsert_company_snapshot(db_path: str, snapshot: dict[str, Any]) -> None:
    payload = dict(snapshot)
    payload.setdefault("updated_at", datetime.now(UTC).isoformat())
    payload["missing_fields_json"] = json.dumps(payload.get("missing_fields_json", []))
    payload["data_quality_errors_json"] = json.dumps(payload.get("data_quality_errors_json", []))
    payload.setdefault("data_quality_status", "partial")
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO company_snapshot (
                symbol, market, sector_index, sector_name, price, market_cap, shares_outstanding,
                paid_in_capital, pe_ratio, pb_ratio, roe, equity, net_income_latest_period,
                net_income_ttm, estimated_net_income, financial_period, period_type, source,
                missing_fields_json, data_quality_status, data_quality_errors_json, updated_at
            ) VALUES (
                :symbol, :market, :sector_index, :sector_name, :price, :market_cap, :shares_outstanding,
                :paid_in_capital, :pe_ratio, :pb_ratio, :roe, :equity, :net_income_latest_period,
                :net_income_ttm, :estimated_net_income, :financial_period, :period_type, :source,
                :missing_fields_json, :data_quality_status, :data_quality_errors_json, :updated_at
            )
            ON CONFLICT(symbol) DO UPDATE SET
                market=excluded.market,
                sector_index=excluded.sector_index,
                sector_name=excluded.sector_name,
                price=excluded.price,
                market_cap=excluded.market_cap,
                shares_outstanding=excluded.shares_outstanding,
                paid_in_capital=excluded.paid_in_capital,
                pe_ratio=excluded.pe_ratio,
                pb_ratio=excluded.pb_ratio,
                roe=excluded.roe,
                equity=excluded.equity,
                net_income_latest_period=excluded.net_income_latest_period,
                net_income_ttm=excluded.net_income_ttm,
                estimated_net_income=excluded.estimated_net_income,
                financial_period=excluded.financial_period,
                period_type=excluded.period_type,
                source=excluded.source,
                missing_fields_json=excluded.missing_fields_json,
                data_quality_status=excluded.data_quality_status,
                data_quality_errors_json=excluded.data_quality_errors_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )


def get_company_snapshot(db_path: str, symbol: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM company_snapshot WHERE symbol = ?", (symbol.upper(),)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["missing_fields_json"] = json.loads(out.get("missing_fields_json") or "[]")
    out["data_quality_errors_json"] = json.loads(out.get("data_quality_errors_json") or "[]")
    return out


def upsert_sector_metrics(db_path: str, metrics: dict[str, Any]) -> None:
    payload = dict(metrics)
    payload.setdefault("calculated_at", datetime.now(UTC).isoformat())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sector_metrics (
                sector_index, sector_name, member_count, valid_member_count,
                pe_median, pe_aggregate, pb_median, pb_aggregate, roe_aggregate, calculated_at
            ) VALUES (
                :sector_index, :sector_name, :member_count, :valid_member_count,
                :pe_median, :pe_aggregate, :pb_median, :pb_aggregate, :roe_aggregate, :calculated_at
            )
            ON CONFLICT(sector_index) DO UPDATE SET
                sector_name=excluded.sector_name,
                member_count=excluded.member_count,
                valid_member_count=excluded.valid_member_count,
                pe_median=excluded.pe_median,
                pe_aggregate=excluded.pe_aggregate,
                pb_median=excluded.pb_median,
                pb_aggregate=excluded.pb_aggregate,
                roe_aggregate=excluded.roe_aggregate,
                calculated_at=excluded.calculated_at
            """,
            payload,
        )


def get_sector_metrics(db_path: str, sector_index: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sector_metrics WHERE sector_index = ?", (sector_index,)).fetchone()
    return dict(row) if row else None


def save_valuation_result(db_path: str, result: dict[str, Any]) -> None:
    payload = dict(result)
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    payload["target_prices_json"] = json.dumps(payload.get("target_prices_json", {}))
    payload["estimation_json"] = json.dumps(payload.get("estimation_json", {}))
    payload["sector_comparison_json"] = json.dumps(payload.get("sector_comparison_json", {}))
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO valuation_results (
                symbol, valuation_date, price, fair_value, upside_percent,
                target_prices_json, estimation_json, sector_comparison_json,
                confidence_score, created_at
            ) VALUES (
                :symbol, :valuation_date, :price, :fair_value, :upside_percent,
                :target_prices_json, :estimation_json, :sector_comparison_json,
                :confidence_score, :created_at
            )
            """,
            payload,
        )


def is_stale(updated_at: str | None, max_age_hours: int = 24) -> bool:
    if not updated_at:
        return True
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(UTC) - dt > timedelta(hours=max_age_hours)


def evaluate_snapshot_quality(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    price = snapshot.get("price")
    market_cap = snapshot.get("market_cap")
    shares_outstanding = snapshot.get("shares_outstanding")
    paid_in_capital = snapshot.get("paid_in_capital")
    equity = snapshot.get("equity")
    estimated_net_income = snapshot.get("estimated_net_income")

    if price is None and market_cap is None and shares_outstanding is None:
        errors.append("price_marketcap_shares_all_missing")
    if shares_outstanding is None and paid_in_capital is None:
        errors.append("shares_or_paid_in_capital_missing")
    if equity is None:
        errors.append("equity_missing")
    if estimated_net_income is None:
        errors.append("estimated_net_income_missing")

    if len(errors) == 0:
        return "usable", []
    if "price_marketcap_shares_all_missing" in errors or "shares_or_paid_in_capital_missing" in errors:
        return "unusable", errors
    return "partial", errors


def is_snapshot_usable(snapshot: dict[str, Any]) -> bool:
    status, _ = evaluate_snapshot_quality(snapshot)
    return status == "usable"

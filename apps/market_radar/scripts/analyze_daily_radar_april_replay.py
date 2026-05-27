from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_OUT_ROOT = "data/backtest_outputs/daily_radar_april_replay_2026"
DEFAULT_DB_PATH = "data/market_radar_cache.sqlite"
DEFAULT_START_DATE = "2026-04-01"
DEFAULT_END_DATE = "2026-04-30"
KNOWN_MARKET_HOLIDAYS = {"2026-04-23"}
FINGERPRINT_FIELDS = [
    "symbol",
    "close",
    "volume",
    "turnover",
    "avg_turnover_20d",
    "rsi_14",
    "return_5d_pct",
    "return_10d_pct",
    "close_position",
    "ma20",
    "signal_date",
]
VALUE_FINGERPRINT_FIELDS = [
    "symbol",
    "close",
    "volume",
    "turnover",
    "avg_turnover_20d",
    "volume_ratio_20d",
    "turnover_ratio_20d",
    "rsi_14",
    "return_5d_pct",
    "return_10d_pct",
    "close_position",
    "ma20",
    "above_ma20",
    "balanced_score",
    "momentum_quality_score",
    "liquidity_safe_score",
    "special_score",
]
PERFORMANCE_WINDOWS = [5, 10, 20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--exclude-stale-performance", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype("string").str.lower().isin(["true", "1", "yes"])


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _clip_0_100(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(max(0.0, min(100.0, value)))


def compute_production_scores(metrics: dict[str, Any], regime_context: dict[str, Any] | None = None) -> dict[str, float]:
    volume_ratio = _safe_float(metrics.get("volume_ratio_20d")) or 0.0
    turnover_ratio = _safe_float(metrics.get("turnover_ratio_20d")) or 0.0
    avg_turnover = _safe_float(metrics.get("avg_turnover_20d")) or 0.0
    turnover_try = _safe_float(metrics.get("turnover_try")) or 0.0
    close_position = _safe_float(metrics.get("close_position")) or 0.0
    rsi_14 = _safe_float(metrics.get("rsi_14"))
    return_5d = _safe_float(metrics.get("return_5d_pct")) or 0.0
    return_10d = _safe_float(metrics.get("return_10d_pct")) or 0.0
    close = _safe_float(metrics.get("close"))
    ma20 = _safe_float(metrics.get("ma20"))

    volume_spike_score = _clip_0_100((volume_ratio / 3.0) * 100.0 * 0.5 + (turnover_ratio / 2.5) * 100.0 * 0.5)
    liquidity_score = _clip_0_100(
        min(avg_turnover / 50_000_000.0, 1.0) * 70.0 + min(turnover_try / 50_000_000.0, 1.0) * 30.0
    )
    close_position_score = _clip_0_100(close_position * 100.0)
    trend_quality_score = 0.0
    if close is not None and ma20 not in (None, 0):
        trend_quality_score = _clip_0_100((((close / ma20) - 0.95) / 0.10) * 100.0)
    rsi_safety_score = 60.0 if rsi_14 is None else _clip_0_100(100.0 - abs(rsi_14 - 62.0) * 3.0)
    overextension_penalty = _clip_0_100(max(0.0, return_5d - 12.0) * 2.0 + max(0.0, return_10d - 20.0) * 1.3)
    regime_bonus = 3.0 if regime_context and bool(regime_context.get("market_supportive", False)) else 0.0

    balanced = _clip_0_100(
        0.24 * volume_spike_score
        + 0.20 * liquidity_score
        + 0.18 * close_position_score
        + 0.20 * trend_quality_score
        + 0.18 * rsi_safety_score
        - 0.12 * overextension_penalty
        + regime_bonus
    )
    momentum_quality = _clip_0_100(
        0.32 * volume_spike_score
        + 0.12 * liquidity_score
        + 0.22 * close_position_score
        + 0.24 * trend_quality_score
        + 0.10 * rsi_safety_score
        - 0.18 * overextension_penalty
        + regime_bonus
    )
    liquidity_safe = _clip_0_100(
        0.12 * volume_spike_score
        + 0.34 * liquidity_score
        + 0.12 * close_position_score
        + 0.12 * trend_quality_score
        + 0.30 * rsi_safety_score
        - 0.18 * overextension_penalty
        + regime_bonus
    )
    return {
        "balanced_score": balanced,
        "momentum_quality_score": momentum_quality,
        "liquidity_safe_score": liquidity_safe,
        "production_score": liquidity_safe,
    }


def _score_bucket(rank: int) -> str:
    if rank <= 20:
        return "top20"
    if rank <= 30:
        return "top30"
    if rank <= 50:
        return "top50"
    if rank <= 75:
        return "top75"
    return "watchlist"


def _read_features(output_dir: Path) -> pd.DataFrame:
    pq = output_dir / "candidate_features.parquet"
    csv = output_dir / "candidate_features.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            if not csv.exists():
                raise
    if not csv.exists():
        raise FileNotFoundError(f"candidate features missing: {csv}")
    return pd.read_csv(csv)


def _relaxed_mask(df: pd.DataFrame) -> pd.Series:
    ma_ok = df["above_ma20"] | df["close"].ge(df["ma20"])
    return (
        df["turnover"].ge(10_000_000.0)
        & df["avg_turnover_20d"].ge(10_000_000.0)
        & ma_ok.fillna(False)
        & df["rsi_14"].le(78.0)
        & df["return_5d_pct"].le(35.0)
        & df["return_10d_pct"].le(60.0)
        & df["close_position"].ge(0.50)
    ).fillna(False)


def _prepare_base_features(output_dir: Path) -> pd.DataFrame:
    raw = _read_features(output_dir)
    return _prepare_base_features_from_df(raw)


def _prepare_base_features_from_df(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy()
    f = f.loc[f["strategy"].astype(str) == "volume_spike_strict"].copy()
    f["signal_date"] = pd.to_datetime(f["signal_date"], errors="coerce").dt.tz_localize(None)
    f = _to_numeric(
        f,
        [
            "close",
            "volume",
            "turnover",
            "avg_turnover_20d",
            "volume_ratio_20d",
            "turnover_ratio_20d",
            "ma20",
            "rsi_14",
            "return_5d_pct",
            "return_10d_pct",
            "close_position",
        ],
    )
    f["above_ma20"] = _to_bool(f.get("above_ma20", pd.Series(False, index=f.index)))
    return f


def _normalize_fingerprint_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def _row_fingerprint(row: pd.Series) -> str:
    payload = "|".join(_normalize_fingerprint_value(row.get(field)) for field in FINGERPRINT_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_value_fingerprint(row: pd.Series) -> str:
    payload = "|".join(_normalize_fingerprint_value(row.get(field)) for field in VALUE_FINGERPRINT_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _top_symbols(df: pd.DataFrame, n: int) -> list[str]:
    return df.loc[df["production_rank"] <= n, "symbol"].astype(str).tolist()


def _daily_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    payload = "|".join(df.sort_values("symbol")["data_fingerprint"].astype(str).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _daily_value_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    payload = "|".join(df.sort_values("symbol")["value_fingerprint"].astype(str).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_known_market_holiday(as_of_date: str) -> bool:
    return as_of_date in KNOWN_MARKET_HOLIDAYS


def _is_expected_trading_day(as_of_date: str) -> bool:
    ts = pd.Timestamp(as_of_date)
    if ts.weekday() >= 5:
        return False
    if _is_known_market_holiday(as_of_date):
        return False
    return True


def _special_pass_details(row: pd.Series, top30_only: bool = True) -> dict[str, Any]:
    checks = {
        "loose": [
            ("avg_turnover_20d>=30m", _safe_float(row.get("avg_turnover_20d")) is not None and float(row.get("avg_turnover_20d")) >= 30_000_000.0),
            ("turnover>=20m", _safe_float(row.get("turnover")) is not None and float(row.get("turnover")) >= 20_000_000.0),
            ("50<=rsi_14<=72", _safe_float(row.get("rsi_14")) is not None and 50.0 <= float(row.get("rsi_14")) <= 72.0),
            ("return_5d_pct<=22", _safe_float(row.get("return_5d_pct")) is not None and float(row.get("return_5d_pct")) <= 22.0),
            ("return_10d_pct<=40", _safe_float(row.get("return_10d_pct")) is not None and float(row.get("return_10d_pct")) <= 40.0),
            ("liquidity_safe_score>=68", _safe_float(row.get("liquidity_safe_score")) is not None and float(row.get("liquidity_safe_score")) >= 68.0),
            ("balanced_score>=56", _safe_float(row.get("balanced_score")) is not None and float(row.get("balanced_score")) >= 56.0),
            ("momentum_quality_score>=54", _safe_float(row.get("momentum_quality_score")) is not None and float(row.get("momentum_quality_score")) >= 54.0),
        ],
        "mid": [
            ("avg_turnover_20d>=40m", _safe_float(row.get("avg_turnover_20d")) is not None and float(row.get("avg_turnover_20d")) >= 40_000_000.0),
            ("turnover>=30m", _safe_float(row.get("turnover")) is not None and float(row.get("turnover")) >= 30_000_000.0),
            ("52<=rsi_14<=70", _safe_float(row.get("rsi_14")) is not None and 52.0 <= float(row.get("rsi_14")) <= 70.0),
            ("return_5d_pct<=18", _safe_float(row.get("return_5d_pct")) is not None and float(row.get("return_5d_pct")) <= 18.0),
            ("return_10d_pct<=32", _safe_float(row.get("return_10d_pct")) is not None and float(row.get("return_10d_pct")) <= 32.0),
            ("liquidity_safe_score>=70", _safe_float(row.get("liquidity_safe_score")) is not None and float(row.get("liquidity_safe_score")) >= 70.0),
            ("balanced_score>=60", _safe_float(row.get("balanced_score")) is not None and float(row.get("balanced_score")) >= 60.0),
            ("momentum_quality_score>=58", _safe_float(row.get("momentum_quality_score")) is not None and float(row.get("momentum_quality_score")) >= 58.0),
        ],
        "strict": [
            ("avg_turnover_20d>=60m", _safe_float(row.get("avg_turnover_20d")) is not None and float(row.get("avg_turnover_20d")) >= 60_000_000.0),
            ("turnover>=50m", _safe_float(row.get("turnover")) is not None and float(row.get("turnover")) >= 50_000_000.0),
            ("52<=rsi_14<=68", _safe_float(row.get("rsi_14")) is not None and 52.0 <= float(row.get("rsi_14")) <= 68.0),
            ("return_5d_pct<=15", _safe_float(row.get("return_5d_pct")) is not None and float(row.get("return_5d_pct")) <= 15.0),
            ("return_10d_pct<=25", _safe_float(row.get("return_10d_pct")) is not None and float(row.get("return_10d_pct")) <= 25.0),
            ("liquidity_safe_score>=74", _safe_float(row.get("liquidity_safe_score")) is not None and float(row.get("liquidity_safe_score")) >= 74.0),
            ("balanced_score>=64", _safe_float(row.get("balanced_score")) is not None and float(row.get("balanced_score")) >= 64.0),
            ("momentum_quality_score>=62", _safe_float(row.get("momentum_quality_score")) is not None and float(row.get("momentum_quality_score")) >= 62.0),
        ],
    }

    out: dict[str, Any] = {}
    is_top30 = bool(row.get("is_top30"))
    passed_tiers: list[str] = []
    failed_reasons: list[str] = []
    passed_reasons: list[str] = []
    if top30_only and not is_top30:
        failed_reasons.append("not_top30")

    for tier, items in checks.items():
        passed = is_top30 if top30_only else True
        tier_failed: list[str] = []
        tier_passed: list[str] = []
        for label, ok in items:
            if ok:
                tier_passed.append(label)
            else:
                tier_failed.append(label)
        if tier_failed:
            passed = False
        out[f"passes_special_{tier}"] = bool(passed)
        if passed:
            passed_tiers.append(tier)
            passed_reasons.extend(f"{tier}:{label}" for label in tier_passed)
        else:
            failed_reasons.extend(f"{tier}:{label}" for label in tier_failed)

    special_tier = ""
    if out["passes_special_strict"]:
        special_tier = "strict"
    elif out["passes_special_mid"]:
        special_tier = "mid"
    elif out["passes_special_loose"]:
        special_tier = "loose"

    out["special_tier"] = special_tier
    out["special_filter_reasons"] = ",".join(sorted(set(passed_reasons)))
    out["special_failed_reasons"] = ",".join(sorted(set(failed_reasons)))
    return out


def _liquidity_depth_score(row: pd.Series) -> float:
    avg_turnover = _safe_float(row.get("avg_turnover_20d")) or 0.0
    turnover = _safe_float(row.get("turnover")) or 0.0
    avg_score = min(avg_turnover / 100_000_000.0, 1.0) * 60.0
    current_score = min(turnover / 100_000_000.0, 1.0) * 40.0
    return _clip_0_100(avg_score + current_score)


def _overheat_penalty(row: pd.Series) -> float:
    rsi = _safe_float(row.get("rsi_14")) or 0.0
    ret5 = _safe_float(row.get("return_5d_pct")) or 0.0
    ret10 = _safe_float(row.get("return_10d_pct")) or 0.0
    penalty = max(0.0, rsi - 70.0) * 1.5
    penalty += max(0.0, ret5 - 18.0) * 1.2
    penalty += max(0.0, ret10 - 32.0) * 0.8
    return penalty


def _special_score(row: pd.Series) -> float:
    liquidity_depth = _liquidity_depth_score(row)
    score = (
        0.40 * (_safe_float(row.get("liquidity_safe_score")) or 0.0)
        + 0.30 * (_safe_float(row.get("balanced_score")) or 0.0)
        + 0.20 * (_safe_float(row.get("momentum_quality_score")) or 0.0)
        + 0.10 * liquidity_depth
        - _overheat_penalty(row)
    )
    return float(score)


def build_daily_radar(features: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    asof = pd.Timestamp(as_of_date)
    f = features.loc[features["signal_date"] <= asof].copy()
    if f.empty:
        return pd.DataFrame()

    f = f.sort_values(["symbol", "signal_date"]).drop_duplicates(subset=["symbol"], keep="last")
    f = f.loc[_relaxed_mask(f)].copy()
    if f.empty:
        return pd.DataFrame()

    score_rows: list[dict[str, Any]] = []
    for _, row in f.iterrows():
        metrics = {
            "volume_ratio_20d": row.get("volume_ratio_20d"),
            "turnover_ratio_20d": row.get("turnover_ratio_20d"),
            "avg_turnover_20d": row.get("avg_turnover_20d"),
            "turnover_try": row.get("turnover"),
            "close_position": row.get("close_position"),
            "rsi_14": row.get("rsi_14"),
            "return_5d_pct": row.get("return_5d_pct"),
            "return_10d_pct": row.get("return_10d_pct"),
            "close": row.get("close"),
            "ma20": row.get("ma20"),
        }
        score_rows.append(compute_production_scores(metrics, {"market_supportive": False}))
    f = pd.concat([f, pd.DataFrame(score_rows, index=f.index)], axis=1)
    f["production_score"] = f["liquidity_safe_score"]
    f["data_fingerprint"] = f.apply(_row_fingerprint, axis=1)
    f = f.sort_values(["production_score", "signal_date", "symbol"], ascending=[False, True, True]).reset_index(drop=True)
    f["production_rank"] = f.index + 1
    f["score_bucket"] = f["production_rank"].apply(_score_bucket)
    f["is_top30"] = f["production_rank"] <= 30
    f["special_score"] = f.apply(_special_score, axis=1)

    details = pd.DataFrame([_special_pass_details(row) for _, row in f.iterrows()], index=f.index)
    f = pd.concat([f, details], axis=1)
    f["value_fingerprint"] = f.apply(_row_value_fingerprint, axis=1)
    f["as_of_date"] = as_of_date
    f["signal_date"] = f["signal_date"].dt.strftime("%Y-%m-%d")
    return f


def _overlap_counts(prev_symbols: list[str], curr_symbols: list[str], prefix: str) -> dict[str, Any]:
    prev_set = set(prev_symbols)
    curr_set = set(curr_symbols)
    repeated = len(prev_set & curr_set)
    new = len(curr_set - prev_set)
    dropped = len(prev_set - curr_set)
    return {
        f"{prefix}_new_vs_prev": new,
        f"{prefix}_overlap_rate_vs_prev": (repeated / len(prev_set)) if prev_set else None,
        f"{prefix}_repeated_vs_prev": repeated,
        f"{prefix}_dropped_vs_prev": dropped,
    }


def _summarize_special_counts(day_df: pd.DataFrame) -> dict[str, Any]:
    special_any = day_df["special_tier"].astype(str).ne("")
    return {
        "special_loose_count": int(day_df["passes_special_loose"].sum()),
        "special_mid_count": int(day_df["passes_special_mid"].sum()),
        "special_strict_count": int(day_df["passes_special_strict"].sum()),
        "special_any_count": int(special_any.sum()),
    }


def _load_price_cache(db_path: str, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], str]:
    cache: dict[str, pd.DataFrame] = {}
    missing = 0
    try:
        with sqlite3.connect(db_path) as conn:
            for symbol in sorted(set(symbols)):
                row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol = ?", (symbol,)).fetchone()
                if not row:
                    missing += 1
                    continue
                payload = json.loads(row[0])
                records = payload.get("records") or []
                if not records:
                    missing += 1
                    continue
                frame = pd.DataFrame.from_records(records)
                needed = [c for c in ["date", "close", "low"] if c in frame.columns]
                if "date" not in needed or "close" not in needed:
                    missing += 1
                    continue
                frame = frame[needed].copy()
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
                frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
                if "low" in frame.columns:
                    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
                else:
                    frame["low"] = frame["close"]
                frame = frame.dropna(subset=["date", "close", "low"]).sort_values("date").reset_index(drop=True)
                if frame.empty:
                    missing += 1
                    continue
                cache[symbol] = frame
    except sqlite3.OperationalError:
        return {}, "forward_data_missing"
    warning = ""
    if missing > 0:
        warning = "forward_data_partial_missing"
    if not cache:
        warning = "forward_data_missing"
    return cache, warning


def _latest_data_date_map(price_cache: dict[str, pd.DataFrame]) -> dict[str, str]:
    out: dict[str, str] = {}
    for symbol, frame in price_cache.items():
        if frame.empty:
            continue
        out[symbol] = frame["date"].max().strftime("%Y-%m-%d")
    return out


def _forward_metrics_for_row(row: pd.Series, price_cache: dict[str, pd.DataFrame]) -> dict[str, Any]:
    symbol = str(row.get("symbol"))
    frame = price_cache.get(symbol)
    blank = {}
    for window in PERFORMANCE_WINDOWS:
        blank[f"forward_return_{window}d_pct"] = None
        blank[f"max_adverse_{window}d_pct"] = None
    if frame is None or frame.empty:
        return blank

    asof = pd.Timestamp(row.get("as_of_date")) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    current = frame.loc[frame["date"] <= asof]
    if current.empty:
        return blank
    anchor_idx = current.index[-1]
    anchor_close = float(frame.loc[anchor_idx, "close"])
    if anchor_close == 0:
        return blank

    out: dict[str, Any] = {}
    for window in PERFORMANCE_WINDOWS:
        end_pos = min(anchor_idx + window, len(frame) - 1)
        if end_pos <= anchor_idx:
            out[f"forward_return_{window}d_pct"] = None
            out[f"max_adverse_{window}d_pct"] = None
            continue
        window_frame = frame.iloc[anchor_idx + 1 : end_pos + 1]
        if window_frame.empty:
            out[f"forward_return_{window}d_pct"] = None
            out[f"max_adverse_{window}d_pct"] = None
            continue
        end_close = float(window_frame.iloc[-1]["close"])
        min_low = float(window_frame["low"].min())
        out[f"forward_return_{window}d_pct"] = (end_close / anchor_close - 1.0) * 100.0
        out[f"max_adverse_{window}d_pct"] = (min_low / anchor_close - 1.0) * 100.0
    return out


def attach_forward_metrics(daily_all: pd.DataFrame, db_path: str) -> tuple[pd.DataFrame, str, dict[str, str]]:
    if daily_all.empty:
        return daily_all.copy(), "forward_data_missing", {}
    price_cache, warning = _load_price_cache(db_path, daily_all["symbol"].astype(str).unique().tolist())
    metrics = pd.DataFrame([_forward_metrics_for_row(row, price_cache) for _, row in daily_all.iterrows()], index=daily_all.index)
    out = pd.concat([daily_all.copy(), metrics], axis=1)
    return out, warning, _latest_data_date_map(price_cache)


def build_performance_summary(daily_all: pd.DataFrame, forward_warning: str, exclude_stale_performance: bool = False) -> pd.DataFrame:
    groups = {
        "top30": daily_all["is_top30"],
        "special_loose": daily_all["passes_special_loose"],
        "special_mid": daily_all["passes_special_mid"],
        "special_strict": daily_all["passes_special_strict"],
        "special_any": daily_all["special_tier"].astype(str).ne(""),
    }
    stale_mask = daily_all["stale_data_warning"].astype(bool) & daily_all["is_expected_trading_day"].astype(bool)
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        grp_all = daily_all.loc[mask].copy()
        grp = grp_all.loc[~stale_mask.loc[grp_all.index]].copy() if exclude_stale_performance else grp_all.copy()
        row: dict[str, Any] = {
            "group_name": group_name,
            "signal_count": int(len(grp_all)),
            "included_signal_count": int(len(grp)),
            "excluded_signal_count": int(len(grp_all) - len(grp)),
            "excluded_stale_days_count": int(daily_all.loc[mask & stale_mask, "as_of_date"].nunique()),
            "data_warning": forward_warning,
        }
        for window in PERFORMANCE_WINDOWS:
            ret = pd.to_numeric(grp.get(f"forward_return_{window}d_pct"), errors="coerce")
            adverse = pd.to_numeric(grp.get(f"max_adverse_{window}d_pct"), errors="coerce")
            row[f"avg_forward_return_{window}d_pct"] = float(ret.mean()) if not ret.dropna().empty else None
            row[f"median_forward_return_{window}d_pct"] = float(ret.median()) if not ret.dropna().empty else None
            row[f"win_rate_{window}d"] = float((ret > 0).mean()) if not ret.dropna().empty else None
            row[f"avg_max_adverse_{window}d_pct"] = float(adverse.mean()) if not adverse.dropna().empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_april_replay(
    features: pd.DataFrame,
    start_date: str,
    end_date: str,
    db_path: str | None = None,
    exclude_stale_performance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_days = pd.date_range(start_date, end_date, freq="D")
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    prev_day_df: pd.DataFrame | None = None
    prev_calendar_fingerprint: str | None = None
    prev_trading_fingerprint: str | None = None
    prev_calendar_value_fingerprint: str | None = None
    prev_trading_value_fingerprint: str | None = None
    prev_trading_value_map: dict[str, str] = {}
    prev_top30: list[str] = []
    prev_loose: list[str] = []
    prev_mid: list[str] = []
    prev_strict: list[str] = []
    prev_any: list[str] = []
    seen_month_top30: set[str] = set()

    for day in replay_days:
        as_of_date = day.strftime("%Y-%m-%d")
        is_weekend = pd.Timestamp(as_of_date).weekday() >= 5
        is_known_holiday = _is_known_market_holiday(as_of_date)
        is_expected_trading_day = _is_expected_trading_day(as_of_date)
        day_df = build_daily_radar(features, as_of_date)
        if not day_df.empty:
            day_df = day_df.copy()
            day_df["daily_data_fingerprint"] = _daily_fingerprint(day_df)
            day_df["daily_value_fingerprint"] = _daily_value_fingerprint(day_df)
            daily_frames.append(day_df)

        top30 = _top_symbols(day_df, 30) if not day_df.empty else []
        top30_scores = pd.to_numeric(day_df.loc[day_df["is_top30"], "liquidity_safe_score"], errors="coerce")
        special_any_mask = day_df["special_tier"].astype(str).ne("") if not day_df.empty else pd.Series(dtype=bool)
        special_any_scores = pd.to_numeric(day_df.loc[special_any_mask, "liquidity_safe_score"], errors="coerce") if not day_df.empty else pd.Series(dtype=float)
        special_any_special_scores = pd.to_numeric(day_df.loc[special_any_mask, "special_score"], errors="coerce") if not day_df.empty else pd.Series(dtype=float)
        top30_special_scores = pd.to_numeric(day_df.loc[day_df["is_top30"], "special_score"], errors="coerce") if not day_df.empty else pd.Series(dtype=float)

        loose_syms = day_df.loc[day_df["passes_special_loose"], "symbol"].astype(str).tolist() if not day_df.empty else []
        mid_syms = day_df.loc[day_df["passes_special_mid"], "symbol"].astype(str).tolist() if not day_df.empty else []
        strict_syms = day_df.loc[day_df["passes_special_strict"], "symbol"].astype(str).tolist() if not day_df.empty else []
        any_syms = day_df.loc[special_any_mask, "symbol"].astype(str).tolist() if not day_df.empty else []

        min_signal_date = None
        max_signal_date = None
        unique_signal_date_count = 0
        if not day_df.empty:
            signal_dates = day_df["signal_date"].dropna().astype(str)
            if not signal_dates.empty:
                min_signal_date = signal_dates.min()
                max_signal_date = signal_dates.max()
                unique_signal_date_count = int(signal_dates.nunique())

        day_fingerprint = _daily_fingerprint(day_df)
        day_value_fingerprint = _daily_value_fingerprint(day_df)
        same_count = 0
        same_daily_fingerprint = False
        same_prev_calendar = False
        same_prev_trading = False
        same_value_prev_calendar = False
        same_value_prev_trading = False
        data_warning = ""
        if prev_day_df is not None and not day_df.empty and not prev_day_df.empty:
            prev_fp = prev_day_df.set_index("symbol")["data_fingerprint"].to_dict()
            curr_fp = day_df.set_index("symbol")["data_fingerprint"].to_dict()
            same_count = sum(1 for symbol, fp in curr_fp.items() if prev_fp.get(symbol) == fp)
        if prev_calendar_fingerprint is not None:
            same_prev_calendar = prev_calendar_fingerprint == day_fingerprint
        if prev_trading_fingerprint is not None:
            same_prev_trading = prev_trading_fingerprint == day_fingerprint
        if prev_calendar_value_fingerprint is not None:
            same_value_prev_calendar = prev_calendar_value_fingerprint == day_value_fingerprint
        if prev_trading_value_fingerprint is not None:
            same_value_prev_trading = prev_trading_value_fingerprint == day_value_fingerprint
        same_daily_fingerprint = same_prev_calendar

        special_counts = _summarize_special_counts(day_df) if not day_df.empty else {
            "special_loose_count": 0,
            "special_mid_count": 0,
            "special_strict_count": 0,
            "special_any_count": 0,
        }

        stale_reasons: list[str] = []
        same_data_but_signal_date_changed_warning = False
        same_data_but_signal_date_changed_reason = ""
        same_value_symbol_count = 0
        changed_value_symbol_count = 0
        changed_value_symbol_ratio = None
        if not day_df.empty and prev_trading_value_map:
            current_value_map = day_df.set_index("symbol")["value_fingerprint"].astype(str).to_dict()
            common_symbols = sorted(set(current_value_map) & set(prev_trading_value_map))
            if common_symbols:
                same_value_symbol_count = sum(1 for s in common_symbols if current_value_map[s] == prev_trading_value_map[s])
                changed_value_symbol_count = len(common_symbols) - same_value_symbol_count
                changed_value_symbol_ratio = changed_value_symbol_count / len(common_symbols)
        if is_expected_trading_day:
            if max_signal_date is not None and max_signal_date < as_of_date:
                stale_reasons.append("max_signal_date_before_as_of")
            if same_prev_trading:
                stale_reasons.append("same_fingerprint_as_previous_expected_trading_day")
            if (not same_prev_trading) and same_value_prev_trading:
                same_data_but_signal_date_changed_warning = True
                same_data_but_signal_date_changed_reason = "value_fingerprint_same_but_data_fingerprint_changed"

        summary_rows.append(
            {
                "as_of_date": as_of_date,
                "is_weekend": bool(is_weekend),
                "is_known_market_holiday": bool(is_known_holiday),
                "is_expected_trading_day": bool(is_expected_trading_day),
                "daily_signal_count": int(len(day_df)),
                "top20_count": int((day_df["production_rank"] <= 20).sum()) if not day_df.empty else 0,
                "top30_count": int((day_df["production_rank"] <= 30).sum()) if not day_df.empty else 0,
                "top50_count": int((day_df["production_rank"] <= 50).sum()) if not day_df.empty else 0,
                "min_signal_date": min_signal_date,
                "max_signal_date": max_signal_date,
                "unique_signal_date_count": unique_signal_date_count,
                "top30_repeated_vs_prev": _overlap_counts(prev_top30, top30, "top30")["top30_repeated_vs_prev"],
                "top30_new_vs_prev": _overlap_counts(prev_top30, top30, "top30")["top30_new_vs_prev"],
                "top30_dropped_vs_prev": _overlap_counts(prev_top30, top30, "top30")["top30_dropped_vs_prev"],
                "top30_overlap_rate_vs_prev": _overlap_counts(prev_top30, top30, "top30")["top30_overlap_rate_vs_prev"],
                "top30_new_vs_month": int(len(set(top30) - seen_month_top30)),
                "top30_seen_before_count": int(len(set(top30) & seen_month_top30)),
                "avg_liquidity_safe_score_top30": float(top30_scores.mean()) if not top30_scores.dropna().empty else None,
                "min_liquidity_safe_score_top30": float(top30_scores.min()) if not top30_scores.dropna().empty else None,
                "max_liquidity_safe_score_top30": float(top30_scores.max()) if not top30_scores.dropna().empty else None,
                "same_data_fingerprint_as_prev": bool(same_daily_fingerprint),
                "same_data_fingerprint_as_prev_calendar_day": bool(same_prev_calendar),
                "same_data_fingerprint_as_prev_trading_day": bool(same_prev_trading),
                "daily_value_fingerprint": day_value_fingerprint,
                "same_value_fingerprint_as_prev_calendar_day": bool(same_value_prev_calendar),
                "same_value_fingerprint_as_prev_trading_day": bool(same_value_prev_trading),
                "same_data_but_signal_date_changed_warning": bool(same_data_but_signal_date_changed_warning),
                "same_data_but_signal_date_changed_reason": same_data_but_signal_date_changed_reason,
                "same_value_symbol_count_vs_prev_trading_day": int(same_value_symbol_count),
                "changed_value_symbol_count_vs_prev_trading_day": int(changed_value_symbol_count),
                "changed_value_symbol_ratio_vs_prev_trading_day": changed_value_symbol_ratio,
                "same_data_symbol_count": int(same_count),
                "data_warning": data_warning,
                **special_counts,
                **{k: v for k, v in _overlap_counts(prev_loose, loose_syms, "special_loose").items() if not k.endswith("_repeated_vs_prev") and not k.endswith("_dropped_vs_prev")},
                **{k: v for k, v in _overlap_counts(prev_mid, mid_syms, "special_mid").items() if not k.endswith("_repeated_vs_prev") and not k.endswith("_dropped_vs_prev")},
                **{k: v for k, v in _overlap_counts(prev_strict, strict_syms, "special_strict").items() if not k.endswith("_repeated_vs_prev") and not k.endswith("_dropped_vs_prev")},
                **{k: v for k, v in _overlap_counts(prev_any, any_syms, "special_any").items() if not k.endswith("_repeated_vs_prev") and not k.endswith("_dropped_vs_prev")},
                "avg_special_score_top30": float(top30_special_scores.mean()) if not top30_special_scores.dropna().empty else None,
                "avg_special_score_special_any": float(special_any_special_scores.mean()) if not special_any_special_scores.dropna().empty else None,
                "avg_liquidity_safe_score_special_any": float(special_any_scores.mean()) if not special_any_scores.dropna().empty else None,
                "daily_data_fingerprint": day_fingerprint,
                "daily_fingerprint": day_fingerprint,
                "stale_data_warning": bool(is_expected_trading_day and len(stale_reasons) > 0),
                "stale_data_reason": ",".join(stale_reasons),
            }
        )

        seen_month_top30.update(top30)
        prev_day_df = day_df
        prev_calendar_fingerprint = day_fingerprint
        prev_calendar_value_fingerprint = day_value_fingerprint
        if is_expected_trading_day:
            prev_trading_fingerprint = day_fingerprint
            prev_trading_value_fingerprint = day_value_fingerprint
            prev_trading_value_map = day_df.set_index("symbol")["value_fingerprint"].astype(str).to_dict() if not day_df.empty else {}
        prev_top30 = top30
        prev_loose = loose_syms
        prev_mid = mid_syms
        prev_strict = strict_syms
        prev_any = any_syms

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    daily_all, forward_warning, latest_date_map = attach_forward_metrics(daily_all, db_path or DEFAULT_DB_PATH)
    summary_df = pd.DataFrame(summary_rows)
    if not daily_all.empty:
        daily_all["data_latest_date"] = daily_all["symbol"].astype(str).map(latest_date_map)
    if not summary_df.empty:
        max_data_latest_dates: list[str | None] = []
        unique_data_latest_date_counts: list[int] = []
        reasons: list[str] = []
        for _, row in summary_df.iterrows():
            as_of_date = str(row["as_of_date"])
            day_rows = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy() if not daily_all.empty else pd.DataFrame()
            latest_dates = day_rows["data_latest_date"].dropna().astype(str) if not day_rows.empty and "data_latest_date" in day_rows.columns else pd.Series(dtype=str)
            max_latest = latest_dates.max() if not latest_dates.empty else None
            unique_count = int(latest_dates.nunique()) if not latest_dates.empty else 0
            current_reasons = [part for part in str(row["stale_data_reason"]).split(",") if part and part != "nan"]
            if bool(row["is_expected_trading_day"]) and max_latest is not None and max_latest < as_of_date:
                current_reasons.append("max_data_latest_date_before_as_of")
            max_data_latest_dates.append(max_latest)
            unique_data_latest_date_counts.append(unique_count)
            reasons.append(",".join(sorted(set(current_reasons))))
        summary_df["max_data_latest_date"] = max_data_latest_dates
        summary_df["unique_data_latest_date_count"] = unique_data_latest_date_counts
        summary_df["stale_data_reason"] = reasons
        summary_df["stale_data_warning"] = (
            summary_df["is_expected_trading_day"].astype(bool)
            & summary_df["stale_data_reason"].astype(str).ne("")
            & summary_df["stale_data_reason"].astype(str).ne("nan")
        )
        if not daily_all.empty:
            merge_cols = [
                "as_of_date",
                "is_expected_trading_day",
                "stale_data_warning",
                "stale_data_reason",
                "same_data_fingerprint_as_prev_trading_day",
                "same_data_fingerprint_as_prev_calendar_day",
                "same_value_fingerprint_as_prev_trading_day",
                "same_value_fingerprint_as_prev_calendar_day",
                "same_data_but_signal_date_changed_warning",
                "same_data_but_signal_date_changed_reason",
                "same_value_symbol_count_vs_prev_trading_day",
                "changed_value_symbol_count_vs_prev_trading_day",
                "changed_value_symbol_ratio_vs_prev_trading_day",
                "max_signal_date",
                "max_data_latest_date",
            ]
            daily_all = daily_all.merge(summary_df[merge_cols], on="as_of_date", how="left")
    if not summary_df.empty and forward_warning:
        mask = summary_df["data_warning"].astype(str).eq("")
        summary_df.loc[mask, "data_warning"] = forward_warning
    lifecycle_df = build_symbol_lifecycle(daily_all)
    performance_df = build_performance_summary(daily_all, forward_warning, exclude_stale_performance=exclude_stale_performance)
    audit_df = summary_df.loc[
        (summary_df["as_of_date"].astype(str) >= "2026-04-22") & (summary_df["as_of_date"].astype(str) <= "2026-04-30"),
        [
            "as_of_date",
            "is_expected_trading_day",
            "top30_count",
            "special_mid_count",
            "special_strict_count",
            "avg_special_score_special_any",
            "min_signal_date",
            "max_signal_date",
            "unique_signal_date_count",
            "max_data_latest_date",
            "unique_data_latest_date_count",
            "daily_fingerprint",
            "daily_value_fingerprint",
            "same_data_fingerprint_as_prev_trading_day",
            "same_value_fingerprint_as_prev_trading_day",
            "same_data_but_signal_date_changed_warning",
            "same_data_but_signal_date_changed_reason",
            "same_value_symbol_count_vs_prev_trading_day",
            "changed_value_symbol_count_vs_prev_trading_day",
            "changed_value_symbol_ratio_vs_prev_trading_day",
            "stale_data_warning",
            "stale_data_reason",
        ],
    ].copy()
    return daily_all, summary_df, lifecycle_df, performance_df, audit_df


def build_symbol_lifecycle(daily_all: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "first_seen_date",
        "last_seen_date",
        "days_seen_in_radar",
        "days_seen_in_top30",
        "max_consecutive_days_top30",
        "best_rank",
        "worst_rank",
        "avg_rank",
        "best_liquidity_safe_score",
        "avg_liquidity_safe_score",
        "days_seen_in_special_loose",
        "days_seen_in_special_mid",
        "days_seen_in_special_strict",
        "days_seen_in_special_any",
        "first_special_loose_date",
        "first_special_mid_date",
        "first_special_strict_date",
        "first_special_any_date",
        "max_consecutive_special_any_days",
        "best_special_score",
        "avg_special_score",
    ]
    if daily_all.empty:
        return pd.DataFrame(columns=columns)

    work = daily_all.copy()
    work["as_of_date"] = pd.to_datetime(work["as_of_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for symbol, grp in work.groupby("symbol", sort=True):
        grp = grp.sort_values("as_of_date").reset_index(drop=True)
        top30_dates = grp.loc[grp["is_top30"], "as_of_date"].dropna().tolist()
        special_any_dates = grp.loc[grp["special_tier"].astype(str).ne(""), "as_of_date"].dropna().tolist()

        def _max_consecutive(dates: list[pd.Timestamp]) -> int:
            max_consecutive = 0
            streak = 0
            prev_dt: pd.Timestamp | None = None
            for dt in dates:
                if prev_dt is not None and (dt - prev_dt).days == 1:
                    streak += 1
                else:
                    streak = 1
                max_consecutive = max(max_consecutive, streak)
                prev_dt = dt
            return max_consecutive

        rows.append(
            {
                "symbol": str(symbol),
                "first_seen_date": grp["as_of_date"].min().strftime("%Y-%m-%d"),
                "last_seen_date": grp["as_of_date"].max().strftime("%Y-%m-%d"),
                "days_seen_in_radar": int(grp["as_of_date"].nunique()),
                "days_seen_in_top30": int(grp["is_top30"].sum()),
                "max_consecutive_days_top30": int(_max_consecutive(top30_dates)),
                "best_rank": int(pd.to_numeric(grp["production_rank"], errors="coerce").min()),
                "worst_rank": int(pd.to_numeric(grp["production_rank"], errors="coerce").max()),
                "avg_rank": float(pd.to_numeric(grp["production_rank"], errors="coerce").mean()),
                "best_liquidity_safe_score": float(pd.to_numeric(grp["liquidity_safe_score"], errors="coerce").max()),
                "avg_liquidity_safe_score": float(pd.to_numeric(grp["liquidity_safe_score"], errors="coerce").mean()),
                "days_seen_in_special_loose": int(grp["passes_special_loose"].sum()),
                "days_seen_in_special_mid": int(grp["passes_special_mid"].sum()),
                "days_seen_in_special_strict": int(grp["passes_special_strict"].sum()),
                "days_seen_in_special_any": int(grp["special_tier"].astype(str).ne("").sum()),
                "first_special_loose_date": _first_flag_date(grp, "passes_special_loose"),
                "first_special_mid_date": _first_flag_date(grp, "passes_special_mid"),
                "first_special_strict_date": _first_flag_date(grp, "passes_special_strict"),
                "first_special_any_date": _first_special_any_date(grp),
                "max_consecutive_special_any_days": int(_max_consecutive(special_any_dates)),
                "best_special_score": float(pd.to_numeric(grp["special_score"], errors="coerce").max()),
                "avg_special_score": float(pd.to_numeric(grp["special_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["days_seen_in_special_any", "days_seen_in_top30", "best_rank", "symbol"],
        ascending=[False, False, True, True],
    )


def _first_flag_date(grp: pd.DataFrame, column: str) -> str | None:
    subset = grp.loc[grp[column].astype(bool), "as_of_date"].dropna()
    if subset.empty:
        return None
    return subset.min().strftime("%Y-%m-%d")


def _first_special_any_date(grp: pd.DataFrame) -> str | None:
    subset = grp.loc[grp["special_tier"].astype(str).ne(""), "as_of_date"].dropna()
    if subset.empty:
        return None
    return subset.min().strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    daily_csv = out_root / "daily_radar_april_2026.csv"
    summary_csv = out_root / "daily_radar_april_summary.csv"
    lifecycle_csv = out_root / "symbol_lifecycle_april_2026.csv"
    performance_csv = out_root / "special_filter_performance_april_2026.csv"
    audit_csv = out_root / "daily_radar_stale_data_audit_april_2026.csv"

    base = _prepare_base_features(output_dir)
    daily_all, summary_df, lifecycle_df, performance_df, audit_df = build_april_replay(
        base,
        args.start_date,
        args.end_date,
        db_path=args.db_path,
        exclude_stale_performance=args.exclude_stale_performance,
    )

    daily_columns = [
        "as_of_date",
        "symbol",
        "signal_date",
        "close",
        "volume",
        "turnover",
        "avg_turnover_20d",
        "volume_ratio_20d",
        "turnover_ratio_20d",
        "ma20",
        "above_ma20",
        "rsi_14",
        "return_5d_pct",
        "return_10d_pct",
        "close_position",
        "balanced_score",
        "momentum_quality_score",
        "liquidity_safe_score",
        "production_score",
        "production_rank",
        "score_bucket",
        "is_top30",
        "special_score",
        "special_tier",
        "passes_special_loose",
        "passes_special_mid",
        "passes_special_strict",
        "special_filter_reasons",
        "special_failed_reasons",
        "data_fingerprint",
        "value_fingerprint",
        "daily_data_fingerprint",
        "daily_value_fingerprint",
        "data_latest_date",
        "is_expected_trading_day",
        "stale_data_warning",
        "stale_data_reason",
        "same_data_fingerprint_as_prev_trading_day",
        "same_data_fingerprint_as_prev_calendar_day",
        "same_value_fingerprint_as_prev_trading_day",
        "same_value_fingerprint_as_prev_calendar_day",
        "same_data_but_signal_date_changed_warning",
        "same_data_but_signal_date_changed_reason",
        "same_value_symbol_count_vs_prev_trading_day",
        "changed_value_symbol_count_vs_prev_trading_day",
        "changed_value_symbol_ratio_vs_prev_trading_day",
        "forward_return_5d_pct",
        "forward_return_10d_pct",
        "forward_return_20d_pct",
        "max_adverse_5d_pct",
        "max_adverse_10d_pct",
        "max_adverse_20d_pct",
    ]
    if daily_all.empty:
        pd.DataFrame(columns=daily_columns).to_csv(daily_csv, index=False)
    else:
        daily_all[daily_columns].to_csv(daily_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    lifecycle_df.to_csv(lifecycle_csv, index=False)
    performance_df.to_csv(performance_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)

    print(
        {
            "daily_csv": str(daily_csv),
            "summary_csv": str(summary_csv),
            "lifecycle_csv": str(lifecycle_csv),
            "performance_csv": str(performance_csv),
            "audit_csv": str(audit_csv),
            "day_count": int(len(summary_df)),
            "row_count": int(len(daily_all)),
        }
    )


if __name__ == "__main__":
    main()

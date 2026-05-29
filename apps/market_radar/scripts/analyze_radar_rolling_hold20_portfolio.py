from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_radar.scoring import compute_quality_threshold_score


DEFAULT_OUTPUT_DIR = "data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_OUT_ROOT = "data/backtest_outputs/radar_rolling_hold20_portfolio_2026"
DEFAULT_DB_PATH = "data/market_radar_cache.sqlite"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_MAX_HOLDINGS = 10
DEFAULT_HOLDING_DAYS = 20
DEFAULT_MIN_POSITION_VALUE = 100.0
BENCHMARKS = [("XU100", "xu100"), ("XUTUM", "xutum")]
FILTER_NAMES = [
    "top30",
    "special_loose",
    "special_mid",
    "special_strict",
    "special_strict_top10",
    "tv_volume_momentum_trend",
    "special_strict_quality_v2",
    "adaptive_v1_cash_no_buy",
    "special_strict_score_threshold_50",
    "special_strict_score_threshold_60",
]
STRATEGY_NAMES = [
    "top30_fresh_only",
    "special_mid_fresh_only",
    "special_strict_fresh_only",
    "special_strict_top10_fresh_only",
    "special_strict_score_threshold_fresh_only",
    "special_strict_score_threshold_50_fresh_only",
    "special_strict_score_threshold_60_fresh_only",
    "adaptive_v1_cash_no_buy_fresh_only",
    "adaptive_regime_v1_fresh_only",
    "special_strict_quality_v2_fresh_only",
    "special_strict_quality_v2_score_fresh_only",
    "special_strict_pending_ttl3_raw",
    "special_strict_top10_pending_ttl3_raw",
    "special_strict_pending_ttl3_revalidate",
    "special_strict_top10_pending_ttl3_revalidate",
    "special_strict_pending_ttl5_revalidate",
    "special_strict_top10_pending_ttl5_revalidate",
    "tv_volume_momentum_trend_fresh_only",
]


def _load_april_module():
    script_path = Path(__file__).resolve().parent / "analyze_daily_radar_april_replay.py"
    spec = importlib.util.spec_from_file_location("analyze_daily_radar_april_replay", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


APRIL = _load_april_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--candidate-features-path")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date")
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--max-holdings", type=int, default=DEFAULT_MAX_HOLDINGS)
    p.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    p.add_argument("--entry-mode", choices=["next_open", "same_close"], default="next_open")
    p.add_argument("--exit-mode", choices=["hold20_close"], default="hold20_close")
    p.add_argument("--exclude-stale-new-entries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-same-day-reentry", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-position-value", type=float, default=DEFAULT_MIN_POSITION_VALUE)
    p.add_argument("--quality-score-thresholds", default="40,50,60,70")
    return p.parse_args()


def _parse_thresholds(args: argparse.Namespace) -> list[int]:
    raw = getattr(args, "quality_score_thresholds", None)
    if raw:
        return sorted({int(float(x.strip())) for x in str(raw).split(",") if x.strip()})
    return [int(float(getattr(args, "quality_score_threshold", 50.0)))]


def _resolve_end_date(features: pd.DataFrame, end_date: str | None) -> str:
    if end_date:
        return end_date
    signal_dates = pd.to_datetime(features["signal_date"], errors="coerce").dropna()
    if signal_dates.empty:
        raise ValueError("No valid signal_date found")
    return signal_dates.max().strftime("%Y-%m-%d")


def _compute_market_breadth(day_df: pd.DataFrame) -> float | None:
    if day_df.empty or "close" not in day_df.columns or "ma20" not in day_df.columns:
        return None
    close = pd.to_numeric(day_df["close"], errors="coerce")
    ma20 = pd.to_numeric(day_df["ma20"], errors="coerce")
    valid = close.notna() & ma20.notna()
    if not bool(valid.any()):
        return None
    return float((close.loc[valid] > ma20.loc[valid]).mean() * 100.0)


def _compute_weak_score(
    *,
    xu100_close: float | None,
    xu100_ma50: float | None,
    xu100_return_20d_pct: float | None,
    xu100_ma50_slope_10d: float | None,
    market_breadth_pct_above_ma20: float | None,
) -> int:
    score = 0
    if pd.notna(xu100_close) and pd.notna(xu100_ma50) and float(xu100_close) < float(xu100_ma50):
        score += 1
    if pd.notna(xu100_return_20d_pct) and float(xu100_return_20d_pct) < -3.0:
        score += 1
    if pd.notna(xu100_ma50_slope_10d) and float(xu100_ma50_slope_10d) < 0.0:
        score += 1
    if pd.notna(market_breadth_pct_above_ma20) and float(market_breadth_pct_above_ma20) < 45.0:
        score += 1
    return int(score)


def _adaptive_mode_for_weak_score(weak_score: int) -> str:
    if weak_score <= 1:
        return "special_strict"
    if weak_score == 2:
        return "threshold_50"
    if weak_score == 3:
        return "threshold_60"
    return "cash"


def _parse_strategy_name(strategy_name: str, max_holdings: int) -> dict[str, Any]:
    if strategy_name.startswith("top30"):
        return {"base_group": "top30", "pending_mode": False, "ttl": None, "revalidate": False, "limit": 30}
    if strategy_name.startswith("special_mid"):
        return {"base_group": "special_mid", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("tv_volume_momentum_trend"):
        return {"base_group": "tv_volume_momentum_trend", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("special_strict_quality_v2_score"):
        return {"base_group": "special_strict_quality_v2_score", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("special_strict_quality_v2"):
        return {"base_group": "special_strict_quality_v2", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("adaptive_regime_v1"):
        return {"base_group": "adaptive_regime_v1", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("adaptive_v1_cash_no_buy"):
        return {"base_group": "adaptive_v1_cash_no_buy", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("special_strict_score_threshold_50"):
        return {"base_group": "special_strict_score_threshold_50", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("special_strict_score_threshold_60"):
        return {"base_group": "special_strict_score_threshold_60", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    if strategy_name.startswith("special_strict_score_threshold"):
        return {"base_group": "special_strict_score_threshold", "pending_mode": False, "ttl": None, "revalidate": False, "limit": max_holdings}
    pending_mode = "pending" in strategy_name
    revalidate = "revalidate" in strategy_name
    ttl = 3 if "ttl3" in strategy_name else (5 if "ttl5" in strategy_name else None)
    if "top10" in strategy_name:
        return {"base_group": "special_strict_top10", "pending_mode": pending_mode, "ttl": ttl, "revalidate": revalidate, "limit": max_holdings}
    return {"base_group": "special_strict", "pending_mode": pending_mode, "ttl": ttl, "revalidate": revalidate, "limit": max_holdings}


def _safe_num(s: pd.Series, col: str, default: float = 0.0) -> pd.Series:
    if col not in s:
        return pd.Series(default, index=s.index, dtype=float)
    return pd.to_numeric(s[col], errors="coerce")


def _rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return pd.to_numeric(rsi, errors="coerce")


def _adx14(frame: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, float("nan")))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, float("nan")))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))) * 100
    dx = pd.to_numeric(dx, errors="coerce")
    return dx.ewm(alpha=1 / 14, adjust=False).mean().astype(float)


def _load_price_cache_full(db_path: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(db_path) as conn:
        for symbol in sorted(set(symbols)):
            row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol = ?", (symbol,)).fetchone()
            if not row:
                continue
            payload = json.loads(row[0])
            frame = pd.DataFrame.from_records(payload.get("records") or [])
            if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
                continue
            for col in ["open", "close", "high", "low", "volume"]:
                if col not in frame.columns:
                    frame[col] = frame["close"] if col != "volume" else pd.NA
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"], format="ISO8601", utc=True).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
            frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if frame.empty:
                continue
            frame["date_str"] = frame["date"].dt.strftime("%Y-%m-%d")
            cache[symbol] = frame[["date", "date_str", "open", "close", "high", "low", "volume"]].copy()
    return cache


def _compute_price_features_for_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    df["turnover_today"] = close * volume
    df["avg_turnover_30d"] = df["turnover_today"].rolling(30, min_periods=1).mean()
    df["ema8"] = close.ewm(span=8, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema60"] = close.ewm(span=60, adjust=False).mean()
    df["daily_change_pct"] = close.pct_change() * 100
    df["rolling_low_252d"] = close.rolling(252, min_periods=1).min()
    df["price_above_52w_low_pct"] = (close / df["rolling_low_252d"] - 1.0) * 100.0
    df["rsi_14_price"] = _rsi14(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    df["adx_14"] = _adx14(df)
    df["adr_pct"] = (((high - low) / close.replace(0, pd.NA)) * 100.0).rolling(20, min_periods=10).mean()
    df["perf_3m_pct"] = (close / close.shift(63) - 1.0) * 100.0
    df["perf_6m_pct"] = (close / close.shift(126) - 1.0) * 100.0
    df["close_vs_ema20_pct"] = (close / df["ema20"] - 1.0) * 100.0
    df["ema8_vs_ema21_pct"] = (df["ema8"] / df["ema21"] - 1.0) * 100.0
    return df


def _build_indicator_snapshot(price_cache: dict[str, pd.DataFrame]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol, frame in price_cache.items():
        f = _compute_price_features_for_symbol(frame)
        cols = [
            "date_str",
            "close",
            "volume",
            "turnover_today",
            "avg_turnover_30d",
            "rsi_14_price",
            "macd_hist",
            "adx_14",
            "adr_pct",
            "ema8",
            "ema21",
            "ema20",
            "ema50",
            "ema60",
            "perf_3m_pct",
            "perf_6m_pct",
            "daily_change_pct",
            "price_above_52w_low_pct",
            "close_vs_ema20_pct",
            "ema8_vs_ema21_pct",
        ]
        for row_dict in f[cols].to_dict('records'):
            out[(symbol, str(row_dict["date_str"]))] = row_dict
    return out


def _enrich_daily_all(daily_all: pd.DataFrame, indicator_map: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    out = daily_all.copy()
    for col in [
        "turnover_today",
        "avg_turnover_30d",
        "macd_hist",
        "adx_14",
        "adr_pct",
        "ema8",
        "ema21",
        "ema20",
        "ema50",
        "ema60",
        "perf_3m_pct",
        "perf_6m_pct",
        "daily_change_pct",
        "price_above_52w_low_pct",
        "close_vs_ema20_pct",
        "ema8_vs_ema21_pct",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    for i, row in out.iterrows():
        key = (str(row["symbol"]), str(row["as_of_date"]))
        snap = indicator_map.get(key)
        if not snap:
            continue
        if pd.isna(row.get("close")) and pd.notna(snap.get("close")):
            out.at[i, "close"] = snap.get("close")
        if pd.isna(row.get("volume")) and pd.notna(snap.get("volume")):
            out.at[i, "volume"] = snap.get("volume")
        if pd.isna(row.get("rsi_14")) and pd.notna(snap.get("rsi_14_price")):
            out.at[i, "rsi_14"] = snap.get("rsi_14_price")
        for col in [
            "turnover_today",
            "avg_turnover_30d",
            "macd_hist",
            "adx_14",
            "adr_pct",
            "ema8",
            "ema21",
            "ema20",
            "ema50",
            "ema60",
            "perf_3m_pct",
            "perf_6m_pct",
            "daily_change_pct",
            "price_above_52w_low_pct",
            "close_vs_ema20_pct",
            "ema8_vs_ema21_pct",
        ]:
            if pd.notna(snap.get(col)):
                out.at[i, col] = snap.get(col)
    out["daily_change_gt_2"] = pd.to_numeric(out["daily_change_pct"], errors="coerce") > 2.0
    out["price_above_52w_low_gte_70"] = pd.to_numeric(out["price_above_52w_low_pct"], errors="coerce") >= 70.0
    out["turnover_today"] = pd.to_numeric(out["turnover_today"], errors="coerce")
    out["avg_turnover_30d"] = pd.to_numeric(out["avg_turnover_30d"], errors="coerce")
    out["volume_ratio_20d"] = pd.to_numeric(out.get("volume_ratio_20d"), errors="coerce")
    out["avg_turnover_20d"] = pd.to_numeric(out.get("avg_turnover_20d"), errors="coerce")
    out["rsi_14"] = pd.to_numeric(out.get("rsi_14"), errors="coerce")
    out["macd_hist"] = pd.to_numeric(out.get("macd_hist"), errors="coerce")
    out["adx_14"] = pd.to_numeric(out.get("adx_14"), errors="coerce")
    out["adr_pct"] = pd.to_numeric(out.get("adr_pct"), errors="coerce")
    out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
    out["ema20"] = pd.to_numeric(out.get("ema20"), errors="coerce")
    out["ema50"] = pd.to_numeric(out.get("ema50"), errors="coerce")
    out["ema60"] = pd.to_numeric(out.get("ema60"), errors="coerce")
    out["ema8"] = pd.to_numeric(out.get("ema8"), errors="coerce")
    out["ema21"] = pd.to_numeric(out.get("ema21"), errors="coerce")
    out["perf_3m_pct"] = pd.to_numeric(out.get("perf_3m_pct"), errors="coerce")
    out["perf_6m_pct"] = pd.to_numeric(out.get("perf_6m_pct"), errors="coerce")
    out["turnover_floor_tv"] = out["avg_turnover_30d"].where(out["avg_turnover_30d"].notna(), out["avg_turnover_20d"])
    # cross-sectional rank-based score per day
    out["tv_momentum_score"] = pd.NA
    for as_of_date, grp in out.groupby("as_of_date"):
        g = grp.copy()
        comp_cols = [
            "volume_ratio_20d",
            "turnover_today",
            "turnover_floor_tv",
            "rsi_14",
            "macd_hist",
            "adx_14",
            "adr_pct",
            "perf_3m_pct",
            "perf_6m_pct",
            "close_vs_ema20_pct",
            "ema8_vs_ema21_pct",
        ]
        ranks = []
        for c in comp_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            r = vals.rank(pct=True, method="average")
            ranks.append(r.fillna(0.0))
        score = sum(ranks) / len(ranks) * 100.0
        out.loc[g.index, "tv_momentum_score"] = score
    out["tv_momentum_score"] = pd.to_numeric(out["tv_momentum_score"], errors="coerce")
    out["passes_tv_volume_momentum_trend"] = (
        (out["close"] >= 1.0)
        & (out["turnover_floor_tv"] >= 20_000_000.0)
        & (out["turnover_today"] >= 10_000_000.0)
        & (out["volume_ratio_20d"] >= 1.5)
        & (out["rsi_14"] > 50.0)
        & (out["macd_hist"] > 0.0)
        & (out["close"] > out["ema20"])
        & (out["ema20"] > out["ema50"])
        & (out["ema8"] >= out["ema21"])
        & (out["close"] > out["ema60"])
        & (out["adx_14"] > 20.0)
        & (out["adr_pct"] >= 4.5)
        & (out["perf_3m_pct"] > 0.0)
        & (out["perf_6m_pct"] > 0.0)
    ).fillna(False)
    out["close_gt_ema20"] = (out["close"] > out["ema20"]).fillna(False)
    out["ema20_gt_ema50"] = (out["ema20"] > out["ema50"]).fillna(False)
    out["ema8_gte_ema21"] = (out["ema8"] >= out["ema21"]).fillna(False)
    out["close_gt_ema60"] = (out["close"] > out["ema60"]).fillna(False)
    out = compute_quality_threshold_score(out)
    return out


def _price_on_date(frame: pd.DataFrame | None, date_str: str, column: str) -> float | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    if frame.index.name == "date_str":
        try:
            val = frame.at[date_str, column]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return None if pd.isna(val) else float(val)
        except KeyError:
            return None
    row = frame.loc[frame["date_str"] == date_str]
    if row.empty:
        return None
    value = row.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _build_market_regime_map(
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    expected_dates = (
        summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"]
        .astype(str)
        .tolist()
    )
    expected_set = set(expected_dates)
    breadth_counts: dict[str, list[int]] = {d: [0, 0] for d in expected_dates}

    for symbol, frame in price_cache.items():
        if symbol in {"XU100", "XUTUM"} or frame.empty:
            continue
        f = frame.copy().sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(f.get("close"), errors="coerce")
        f["ma20_breadth"] = close.rolling(20, min_periods=20).mean()
        sel = f.loc[f["date_str"].isin(expected_set), ["date_str", "close", "ma20_breadth"]]
        for row_dict in sel.to_dict('records'):
            close_val = row_dict["close"]
            ma20_val = row_dict["ma20_breadth"]
            if pd.isna(close_val) or pd.isna(ma20_val):
                continue
            counts = breadth_counts[str(row_dict["date_str"])]
            counts[1] += 1
            if float(close_val) > float(ma20_val):
                counts[0] += 1

    xu100 = price_cache.get("XU100")
    xu_map: dict[str, dict[str, Any]] = {}
    if xu100 is not None and not xu100.empty:
        x = xu100.copy().sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(x["close"], errors="coerce")
        x["ma50"] = close.rolling(50, min_periods=50).mean()
        x["return_20d_pct"] = (close / close.shift(20) - 1.0) * 100.0
        x["ma50_slope_10d"] = (x["ma50"] / x["ma50"].shift(10) - 1.0) * 100.0
        for row_dict in x.loc[x["date_str"].isin(expected_set)].to_dict('records'):
            xu_map[str(row_dict["date_str"])] = {
                "xu100_close": row_dict.get("close"),
                "xu100_ma50": row_dict.get("ma50"),
                "xu100_return_20d_pct": row_dict.get("return_20d_pct"),
                "xu100_ma50_slope_10d": row_dict.get("ma50_slope_10d"),
            }

    out: dict[str, dict[str, Any]] = {}
    for as_of_date in expected_dates:
        above, total = breadth_counts.get(as_of_date, [0, 0])
        breadth = (above / total * 100.0) if total else _compute_market_breadth(
            daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date]
        )
        xu = xu_map.get(as_of_date, {})
        weak_score = _compute_weak_score(
            xu100_close=xu.get("xu100_close"),
            xu100_ma50=xu.get("xu100_ma50"),
            xu100_return_20d_pct=xu.get("xu100_return_20d_pct"),
            xu100_ma50_slope_10d=xu.get("xu100_ma50_slope_10d"),
            market_breadth_pct_above_ma20=breadth,
        )
        out[as_of_date] = {
            "weak_score": weak_score,
            "adaptive_selected_mode": _adaptive_mode_for_weak_score(weak_score),
            "xu100_close": xu.get("xu100_close"),
            "xu100_ma50": xu.get("xu100_ma50"),
            "xu100_return_20d_pct": xu.get("xu100_return_20d_pct"),
            "xu100_ma50_slope_10d": xu.get("xu100_ma50_slope_10d"),
            "market_breadth_pct_above_ma20": breadth,
            "breadth_universe_count": total,
        }
    return out


def _read_candidate_features_path(path: str | Path) -> pd.DataFrame:
    feature_path = Path(path)
    if feature_path.is_file():
        if feature_path.suffix == ".parquet":
            return pd.read_parquet(feature_path)
        return pd.read_csv(feature_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"candidate features path missing: {feature_path}")
    files: list[Path] = []
    for year_dir in sorted(feature_path.glob("year=*")):
        parquet = year_dir / "candidate_features.parquet"
        csv = year_dir / "candidate_features.csv"
        if parquet.exists():
            files.append(parquet)
        elif csv.exists():
            files.append(csv)
    if not files:
        parquet = feature_path / "candidate_features.parquet"
        csv = feature_path / "candidate_features.csv"
        if parquet.exists():
            files.append(parquet)
        elif csv.exists():
            files.append(csv)
    if not files:
        raise FileNotFoundError(f"candidate features files missing under: {feature_path}")
    frames: list[pd.DataFrame] = []
    for item in files:
        try:
            frame = pd.read_parquet(item) if item.suffix == ".parquet" else pd.read_csv(item)
        except Exception:
            fallback = item.with_suffix(".csv")
            if item.suffix == ".parquet" and fallback.exists():
                frame = pd.read_csv(fallback)
            else:
                raise
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prepare_features(output_dir: str, candidate_features_path: str | None = None) -> pd.DataFrame:
    if candidate_features_path:
        raw = _read_candidate_features_path(candidate_features_path)
        return APRIL._prepare_base_features_from_df(raw)
    return APRIL._prepare_base_features(Path(output_dir))


def _prepare_inputs(
    output_dir: str,
    candidate_features_path: str | None,
    db_path: str,
    start_date: str,
    end_date: str | None,
    exclude_stale: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], str, dict]:
    features = _prepare_features(output_dir, candidate_features_path)
    resolved_end = _resolve_end_date(features, end_date)
    feature_dates = pd.to_datetime(features.get("signal_date"), errors="coerce").dropna()
    available_min = None if feature_dates.empty else feature_dates.min().strftime("%Y-%m-%d")
    available_max = None if feature_dates.empty else feature_dates.max().strftime("%Y-%m-%d")
    actual_start = start_date
    if available_min and start_date < available_min:
        actual_start = available_min
    daily_all, summary_df, _life, _perf, _audit = APRIL.build_april_replay(
        features,
        actual_start,
        resolved_end,
        db_path=db_path,
        exclude_stale_performance=exclude_stale,
    )
    symbols = sorted(set(daily_all["symbol"].astype(str).tolist()) | {"XU100", "XUTUM"})
    price_cache = _load_price_cache_full(db_path, symbols)
    indicators = _build_indicator_snapshot(price_cache)
    daily_all = _enrich_daily_all(daily_all, indicators)
    requested_start_year = int(str(start_date)[:4])
    end_year = int(str(resolved_end)[:4])
    tested_years = sorted({int(x[:4]) for x in summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()})
    coverage_summary = {
        "requested_start_date": start_date,
        "actual_start_date": actual_start,
        "available_min_signal_date": available_min,
        "available_max_signal_date": available_max,
        "missing_years": [y for y in range(requested_start_year, end_year + 1) if y not in tested_years],
        "tested_years": tested_years,
    }
    return daily_all, summary_df, price_cache, resolved_end, coverage_summary


def _select_group(day_df: pd.DataFrame, base_group: str, max_holdings: int) -> pd.DataFrame:
    if day_df.empty:
        return pd.DataFrame()
    if base_group == "top30":
        out = day_df.loc[day_df["production_rank"] <= 30].copy()
        out = out.sort_values(["production_rank", "symbol"]).reset_index(drop=True)
        out["selection_rank"] = out["production_rank"]
        return out
    if base_group == "special_loose":
        out = day_df.loc[day_df["passes_special_loose"]].copy()
        out = out.sort_values(["production_rank", "symbol"]).reset_index(drop=True)
        out["selection_rank"] = out["production_rank"]
        return out
    if base_group == "special_mid":
        out = day_df.loc[day_df["passes_special_mid"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict":
        out = day_df.loc[day_df["passes_special_strict"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_top10":
        out = day_df.loc[day_df["passes_special_strict"]].copy()
        out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).head(int(max_holdings)).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "tv_volume_momentum_trend":
        out = day_df.loc[day_df["passes_tv_volume_momentum_trend"]].copy()
        if "liquidity_safe_score" in out.columns:
            out = out.sort_values(["liquidity_safe_score", "tv_momentum_score", "volume_ratio_20d", "symbol"], ascending=[False, False, False, True]).reset_index(drop=True)
        else:
            out = out.sort_values(["tv_momentum_score", "volume_ratio_20d", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_quality_v2":
        mask = day_df.get("passes_special_strict_quality_v2", pd.Series(False, index=day_df.index)).fillna(False)
        out = day_df.loc[mask].copy()
        for col in ["liquidity_safe_score", "quality_v2_score"]:
            if col not in out.columns:
                out[col] = pd.NA
        out = out.sort_values(["liquidity_safe_score", "quality_v2_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_quality_v2_score":
        mask = day_df.get("passes_special_strict_quality_v2", pd.Series(False, index=day_df.index)).fillna(False)
        out = day_df.loc[mask].copy()
        for col in ["quality_v2_score", "liquidity_safe_score"]:
            if col not in out.columns:
                out[col] = pd.NA
        out = out.sort_values(["quality_v2_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_score_threshold":
        threshold = float(day_df.attrs.get("quality_score_threshold", 50.0))
        strict_mask = day_df.get("passes_special_strict", pd.Series(False, index=day_df.index)).fillna(False)
        score = pd.to_numeric(day_df.get("quality_threshold_score"), errors="coerce")
        out = day_df.loc[strict_mask & (score >= threshold)].copy()
        out = out.sort_values(["quality_threshold_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_score_threshold_50":
        strict_mask = day_df.get("passes_special_strict", pd.Series(False, index=day_df.index)).fillna(False)
        score = pd.to_numeric(day_df.get("quality_threshold_score"), errors="coerce")
        out = day_df.loc[strict_mask & (score >= 50.0)].copy()
        out = out.sort_values(["quality_threshold_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group == "special_strict_score_threshold_60":
        strict_mask = day_df.get("passes_special_strict", pd.Series(False, index=day_df.index)).fillna(False)
        score = pd.to_numeric(day_df.get("quality_threshold_score"), errors="coerce")
        out = day_df.loc[strict_mask & (score >= 60.0)].copy()
        out = out.sort_values(["quality_threshold_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        out["selection_rank"] = out.index + 1
        return out
    if base_group in ("adaptive_regime_v1", "adaptive_v1_cash_no_buy"):
        weak_score = int(day_df.attrs.get("weak_score", 0))
        mode = _adaptive_mode_for_weak_score(weak_score)
        if weak_score <= 1:
            out = day_df.loc[day_df["passes_special_strict"]].copy()
            out = out.sort_values(["special_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        elif weak_score == 2:
            strict_mask = day_df.get("passes_special_strict", pd.Series(False, index=day_df.index)).fillna(False)
            score = pd.to_numeric(day_df.get("quality_threshold_score"), errors="coerce")
            out = day_df.loc[strict_mask & (score >= 50.0)].copy()
            out = out.sort_values(["quality_threshold_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        elif weak_score == 3:
            strict_mask = day_df.get("passes_special_strict", pd.Series(False, index=day_df.index)).fillna(False)
            score = pd.to_numeric(day_df.get("quality_threshold_score"), errors="coerce")
            out = day_df.loc[strict_mask & (score >= 60.0)].copy()
            out = out.sort_values(["quality_threshold_score", "liquidity_safe_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        else:
            out = pd.DataFrame()
        if not out.empty:
            out["selection_rank"] = out.index + 1
            out["adaptive_selected_mode"] = mode
            out["weak_score"] = weak_score
        return out
    raise ValueError(f"Unknown base_group {base_group}")


def _build_daily_group_map(
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    base_group: str,
    max_holdings: int,
    market_regime_map: dict | None = None,
) -> dict[str, pd.DataFrame]:
    expected_dates = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    out: dict[str, pd.DataFrame] = {}
    for as_of_date in expected_dates:
        grp = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy()
        if base_group in ("adaptive_regime_v1", "adaptive_v1_cash_no_buy") and market_regime_map is not None:
            regime = market_regime_map.get(as_of_date, {})
            grp.attrs["weak_score"] = regime.get("weak_score", 0)
        out[as_of_date] = _select_group(grp, base_group, max_holdings)
    return out


def _build_daily_symbol_map(daily_all: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    expected_dates = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for as_of_date in expected_dates:
        grp = daily_all.loc[daily_all["as_of_date"].astype(str) == as_of_date].copy()
        out[as_of_date] = {} if grp.empty else grp.set_index("symbol").to_dict("index")
    return out


def _compute_exit_date(entry_trade_date: str, trading_dates: list[str], idx_map: dict[str, int], holding_days: int) -> str | None:
    idx = idx_map.get(entry_trade_date)
    if idx is None:
        return None
    exit_idx = idx + int(holding_days) - 1
    if exit_idx >= len(trading_dates):
        return None
    return trading_dates[exit_idx]


def _current_equity(cash: float, positions: list[dict[str, Any]], price_cache: dict[str, pd.DataFrame], as_of_date: str) -> tuple[float, float, float]:
    open_value = 0.0
    open_cost_basis = 0.0
    for pos in positions:
        close_price = _price_on_date(price_cache.get(pos["symbol"]), as_of_date, "close")
        if close_price is None:
            close_price = pos["entry_price"]
        open_value += pos["shares"] * close_price
        open_cost_basis += pos["entry_value"]
    total_equity = float(cash) + float(open_value)
    return total_equity, open_value, open_cost_basis


def _revalidate_pending(row: pd.Series | None, stale_day: bool, min_liquidity_safe: float) -> tuple[bool, str]:
    if row is None:
        return False, "revalidate_missing_row"
    if stale_day:
        return False, "revalidate_stale"
    for field in ["turnover", "volume", "avg_turnover_20d", "liquidity_safe_score", "rsi_14"]:
        if pd.isna(row.get(field)):
            return False, f"revalidate_missing_{field}"
    if float(row["rsi_14"]) > 78.0:
        return False, "revalidate_rsi_too_high"
    if float(row["liquidity_safe_score"]) < float(min_liquidity_safe):
        return False, "revalidate_liquidity_safe_below_min"
    return True, ""


def _pending_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["original_signal_date"],
        float(item.get("special_score") or float("-inf")),
        float(item.get("liquidity_safe_score") or float("-inf")),
        -float(item.get("production_rank") or 10**9),
        item["symbol"],
    )


def _base_group_revalidate_threshold(strategy_name: str) -> float:
    if "special_mid" in strategy_name:
        return 70.0
    return 74.0


def _buy_position(
    *,
    order: dict[str, Any],
    trade_date: str,
    entry_price: float,
    cash: float,
    positions: list[dict[str, Any]],
    price_cache: dict[str, pd.DataFrame],
    max_holdings: int,
    min_position_value: float,
    trading_dates: list[str],
    idx_map: dict[str, int],
    holding_days: int,
    cost_bps: float = 0.0,
) -> tuple[float, bool]:
    total_equity, _open_value, _open_cost = _current_equity(cash, positions, price_cache, trade_date)
    position_budget = total_equity / float(max_holdings)
    position_value = min(position_budget, cash)
    if position_value < float(min_position_value) or entry_price <= 0:
        return cash, False
    actual_entry_price = entry_price * (1.0 + cost_bps / 10000.0)
    shares = position_value / actual_entry_price
    exit_trade_date = _compute_exit_date(trade_date, trading_dates, idx_map, holding_days)
    positions.append(
        {
            **order,
            "entry_trade_date": trade_date,
            "entry_price": float(entry_price),
            "shares": float(shares),
            "entry_value": float(position_value),
            "exit_trade_date": exit_trade_date,
        }
    )
    return cash - float(position_value), True


def simulate_strategy(
    strategy_name: str,
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    *,
    initial_capital: float,
    max_holdings: int,
    holding_days: int,
    entry_mode: str,
    exit_mode: str,
    exclude_stale_new_entries: bool,
    allow_same_day_reentry: bool,
    min_position_value: float,
    market_regime_map: dict | None = None,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exit_mode != "hold20_close":
        raise ValueError(f"Unsupported exit_mode: {exit_mode}")
    cfg = _parse_strategy_name(strategy_name, max_holdings)
    daily_map = _build_daily_group_map(daily_all, summary_df, cfg["base_group"], max_holdings, market_regime_map)
    symbol_map = _build_daily_symbol_map(daily_all, summary_df)
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool)].copy().sort_values("as_of_date").reset_index(drop=True)
    trading_dates = expected["as_of_date"].astype(str).tolist()
    idx_map = {d: i for i, d in enumerate(trading_dates)}

    cash = float(initial_capital)
    positions: list[dict[str, Any]] = []
    pending_queue: list[dict[str, Any]] = []
    scheduled_entries: dict[str, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    prev_equity = float(initial_capital)

    for as_of_date in trading_dates:
        meta = expected.loc[expected["as_of_date"].astype(str) == as_of_date].iloc[0]
        stale_day = bool(meta["stale_data_warning"])
        stale_reason = meta["stale_data_reason"]
        regime = (market_regime_map or {}).get(as_of_date, {})
        group_df = daily_map.get(as_of_date, pd.DataFrame()).copy()
        day_symbols = symbol_map.get(as_of_date, {})
        sold_today: set[str] = set()
        opened_today: set[str] = set()
        new_entries_count = 0
        exits_count = 0
        skipped_duplicate_symbol_count = 0
        skipped_no_slot_count = 0
        pending_expired_count = 0

        if entry_mode == "next_open":
            todays_scheduled = scheduled_entries.pop(as_of_date, [])
            for order in todays_scheduled:
                symbol = order["symbol"]
                if any(pos["symbol"] == symbol for pos in positions) or symbol in sold_today or symbol in opened_today:
                    skipped_duplicate_symbol_count += 1
                    continue
                entry_price = _price_on_date(price_cache.get(symbol), as_of_date, "open")
                if entry_price is None:
                    continue
                cash, bought = _buy_position(
                    order=order,
                    trade_date=as_of_date,
                    entry_price=entry_price,
                    cash=cash,
                    positions=positions,
                    price_cache=price_cache,
                    max_holdings=max_holdings,
                    min_position_value=min_position_value,
                    trading_dates=trading_dates,
                    idx_map=idx_map,
                    holding_days=holding_days,
                    cost_bps=cost_bps,
                )
                if bought:
                    opened_today.add(symbol)
                    new_entries_count += 1

        remaining_positions: list[dict[str, Any]] = []
        for pos in positions:
            scheduled_exit = pos.get("exit_trade_date")
            entry_trade_date = pos.get("entry_trade_date")
            entry_idx = idx_map.get(entry_trade_date)
            current_idx = idx_map.get(as_of_date)

            exit_idx = None
            if scheduled_exit:
                exit_idx = idx_map.get(scheduled_exit)
            elif entry_idx is not None:
                exit_idx = entry_idx + holding_days - 1

            should_exit = False
            force_exit_reason = None

            if current_idx is not None and exit_idx is not None:
                if current_idx >= exit_idx:
                    should_exit = True
                    # If we are strictly past the scheduled exit day, we force-close at entry price if needed
                    if current_idx > exit_idx:
                        force_exit_reason = "hold_exceeded_fallback"

            if not should_exit:
                remaining_positions.append(pos)
                continue

            exit_price = _price_on_date(price_cache.get(pos["symbol"]), as_of_date, "close")
            if exit_price is None:
                # If still no price, try entry price as last resort after holding too long (past exit date)
                if force_exit_reason:
                    exit_price = pos["entry_price"]
                    force_exit_reason = "hold_exceeded_no_price_force"
                else:
                    remaining_positions.append(pos)
                    continue
            actual_exit_price = exit_price * (1.0 - cost_bps / 10000.0)
            exit_value = pos["shares"] * actual_exit_price
            pnl = exit_value - pos["entry_value"]
            return_pct = (exit_value / pos["entry_value"] - 1.0) * 100.0 if pos["entry_value"] else None
            cash += exit_value
            sold_today.add(pos["symbol"])
            exits_count += 1
            trades.append(
                {
                    "strategy_name": strategy_name,
                    "symbol": pos["symbol"],
                    "original_signal_date": pos["original_signal_date"],
                    "entry_trade_date": pos["entry_trade_date"],
                    "exit_trade_date": as_of_date,
                    "holding_trading_days": holding_days,
                    "entry_price": pos["entry_price"],
                    "exit_price": float(exit_price),
                    "shares": pos["shares"],
                    "entry_value": pos["entry_value"],
                    "exit_value": float(exit_value),
                    "pnl": float(pnl),
                    "return_pct": float(return_pct) if return_pct is not None else None,
                    "exit_reason": force_exit_reason or "hold20_close",
                    "production_rank": pos.get("production_rank"),
                    "special_tier": pos.get("special_tier"),
                    "special_score": pos.get("special_score"),
                    "liquidity_safe_score": pos.get("liquidity_safe_score"),
                    "balanced_score": pos.get("balanced_score"),
                    "momentum_quality_score": pos.get("momentum_quality_score"),
                    "tv_momentum_score": pos.get("tv_momentum_score"),
                    "source_type": pos.get("source_type"),
                    "pending_age_trading_days": pos.get("pending_age_trading_days"),
                    "revalidated_in_same_group": pos.get("revalidated_in_same_group"),
                    "stale_entry_warning": pos.get("stale_entry_warning"),
                }
            )
        positions = remaining_positions

        refreshed_queue: list[dict[str, Any]] = []
        for item in pending_queue:
            age = idx_map[as_of_date] - idx_map[item["original_signal_date"]]
            if item["ttl"] is not None and age >= int(item["ttl"]):
                pending_expired_count += 1
                pending_events.append(
                    {
                        "as_of_date": as_of_date,
                        "strategy_name": strategy_name,
                        "symbol": item["symbol"],
                        "original_signal_date": item["original_signal_date"],
                        "event_type": "expired",
                        "reason": "ttl_expired",
                        "pending_age_trading_days": age,
                        "special_score": item.get("special_score"),
                        "liquidity_safe_score": item.get("liquidity_safe_score"),
                        "production_rank": item.get("production_rank"),
                    }
                )
            else:
                item["pending_age_trading_days"] = age
                refreshed_queue.append(item)
        pending_queue = refreshed_queue

        def can_take_new(symbol: str) -> tuple[bool, str]:
            if symbol in {pos["symbol"] for pos in positions} or symbol in opened_today:
                return False, "symbol_already_open_or_sold_today"
            if (not allow_same_day_reentry) and symbol in sold_today:
                return False, "symbol_already_open_or_sold_today"
            if len(positions) + sum(len(v) for v in scheduled_entries.values()) >= max_holdings and entry_mode == "next_open":
                return False, "no_available_slot"
            if len(positions) >= max_holdings and entry_mode == "same_close":
                return False, "no_available_slot"
            return True, ""

        def process_order(order: dict[str, Any], source_kind: str) -> tuple[bool, str]:
            nonlocal cash, new_entries_count, skipped_duplicate_symbol_count, skipped_no_slot_count
            symbol = order["symbol"]
            allowed, reason = can_take_new(symbol)
            if not allowed:
                if reason == "symbol_already_open_or_sold_today":
                    skipped_duplicate_symbol_count += 1
                else:
                    skipped_no_slot_count += 1
                if cfg["pending_mode"] and source_kind == "fresh" and reason == "no_available_slot":
                    pending_queue[:] = [x for x in pending_queue if x["symbol"] != symbol]
                    pending_queue.append({**order, "ttl": int(cfg["ttl"])})
                    pending_events.append(
                        {
                            "as_of_date": as_of_date,
                            "strategy_name": strategy_name,
                            "symbol": symbol,
                            "original_signal_date": order["original_signal_date"],
                            "event_type": "queued",
                            "reason": "no_available_slot",
                            "pending_age_trading_days": 0,
                            "special_score": order.get("special_score"),
                            "liquidity_safe_score": order.get("liquidity_safe_score"),
                            "production_rank": order.get("production_rank"),
                        }
                    )
                return False, reason
            if entry_mode == "same_close":
                entry_price = _price_on_date(price_cache.get(symbol), as_of_date, "close")
                if entry_price is None:
                    return False, "missing_entry_close"
                cash, bought = _buy_position(
                    order=order,
                    trade_date=as_of_date,
                    entry_price=entry_price,
                    cash=cash,
                    positions=positions,
                    price_cache=price_cache,
                    max_holdings=max_holdings,
                    min_position_value=min_position_value,
                    trading_dates=trading_dates,
                    idx_map=idx_map,
                    holding_days=holding_days,
                    cost_bps=cost_bps,
                )
                if bought:
                    opened_today.add(symbol)
                    new_entries_count += 1
                    return True, "bought_same_close"
                return False, "position_value_below_min"
            idx = idx_map[as_of_date]
            if idx + 1 >= len(trading_dates):
                return False, "no_next_trading_day"
            scheduled_entries.setdefault(trading_dates[idx + 1], []).append(order)
            return True, "scheduled_for_next_open"

        if not (exclude_stale_new_entries and stale_day):
            fresh_orders = []
            for _, row in group_df.iterrows():
                order = {
                    "symbol": str(row["symbol"]),
                    "original_signal_date": as_of_date,
                    "production_rank": row.get("production_rank"),
                    "special_tier": row.get("special_tier"),
                    "special_score": row.get("special_score"),
                    "liquidity_safe_score": row.get("liquidity_safe_score"),
                    "balanced_score": row.get("balanced_score"),
                    "momentum_quality_score": row.get("momentum_quality_score"),
                    "tv_momentum_score": row.get("tv_momentum_score"),
                    "source_type": "fresh",
                    "pending_age_trading_days": 0,
                    "revalidated_in_same_group": None,
                    "stale_entry_warning": stale_day,
                }
                bought, reason = process_order(order, "fresh")
                signal_rows.append(
                    {
                        "signal_date": as_of_date,
                        "strategy_name": strategy_name,
                        "symbol": order["symbol"],
                        "was_bought_by_portfolio": bool(bought),
                        "was_skipped_due_to_full_slots": reason == "no_available_slot",
                        "was_skipped_due_to_already_holding": reason == "symbol_already_open_or_sold_today",
                        "was_skipped_due_to_missing_price": reason in ("missing_entry_open", "missing_entry_close"),
                    }
                )
                fresh_orders.append(order)

            if cfg["pending_mode"]:
                ordered_queue = sorted(pending_queue, key=_pending_sort_key, reverse=True)
                kept_queue: list[dict[str, Any]] = []
                selected_today_symbols = set(group_df["symbol"].astype(str).tolist()) if not group_df.empty else set()
                min_liq = _base_group_revalidate_threshold(strategy_name)
                for item in ordered_queue:
                    if len(positions) + sum(len(v) for v in scheduled_entries.values()) >= max_holdings and entry_mode == "next_open":
                        kept_queue.append(item)
                        continue
                    if len(positions) >= max_holdings and entry_mode == "same_close":
                        kept_queue.append(item)
                        continue
                    symbol = item["symbol"]
                    if symbol in {pos["symbol"] for pos in positions} or symbol in sold_today or symbol in opened_today:
                        kept_queue.append(item)
                        continue
                    age = idx_map[as_of_date] - idx_map[item["original_signal_date"]]
                    order = {
                        **item,
                        "source_type": "pending_raw",
                        "pending_age_trading_days": age,
                        "revalidated_in_same_group": None,
                        "stale_entry_warning": stale_day,
                    }
                    if cfg["revalidate"]:
                        row_dict = day_symbols.get(symbol)
                        row_series = pd.Series(row_dict) if row_dict is not None else None
                        ok, reason = _revalidate_pending(row_series, stale_day, min_liq)
                        if not ok:
                            kept_queue.append(item)
                            pending_events.append(
                                {
                                    "as_of_date": as_of_date,
                                    "strategy_name": strategy_name,
                                    "symbol": symbol,
                                    "original_signal_date": item["original_signal_date"],
                                    "event_type": "revalidate_failed",
                                    "reason": reason,
                                    "pending_age_trading_days": age,
                                    "special_score": item.get("special_score"),
                                    "liquidity_safe_score": item.get("liquidity_safe_score"),
                                    "production_rank": item.get("production_rank"),
                                }
                            )
                            continue
                        order["source_type"] = "pending_revalidated"
                        order["revalidated_in_same_group"] = bool(symbol in selected_today_symbols)
                    bought, reason = process_order(order, "pending")
                    if bought:
                        pending_events.append(
                            {
                                "as_of_date": as_of_date,
                                "strategy_name": strategy_name,
                                "symbol": symbol,
                                "original_signal_date": item["original_signal_date"],
                                "event_type": "bought_from_queue",
                                "reason": reason,
                                "pending_age_trading_days": age,
                                "special_score": item.get("special_score"),
                                "liquidity_safe_score": item.get("liquidity_safe_score"),
                                "production_rank": item.get("production_rank"),
                            }
                        )
                    else:
                        kept_queue.append(item)
                pending_queue = kept_queue

        total_equity, open_position_value, open_cost_basis = _current_equity(cash, positions, price_cache, as_of_date)
        daily_return_pct = ((total_equity / prev_equity) - 1.0) * 100.0 if prev_equity else None
        cumulative_return_pct = ((total_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else None
        daily_rows.append(
            {
                "as_of_date": as_of_date,
                "strategy_name": strategy_name,
                "cash": float(cash),
                "open_position_value": float(open_position_value),
                "open_cost_basis": float(open_cost_basis),
                "total_equity": float(total_equity),
                "daily_return_pct": daily_return_pct,
                "cumulative_return_pct": cumulative_return_pct,
                "open_position_count": int(len(positions)),
                "available_slots": int(max(0, max_holdings - len(positions))),
                "new_entries_count": int(new_entries_count),
                "exits_count": int(exits_count),
                "skipped_duplicate_symbol_count": int(skipped_duplicate_symbol_count),
                "skipped_no_slot_count": int(skipped_no_slot_count),
                "pending_queue_size": int(len(pending_queue)),
                "pending_expired_count": int(pending_expired_count),
                "stale_data_warning": stale_day,
                "stale_data_reason": stale_reason,
                "weak_score": regime.get("weak_score"),
                "adaptive_selected_mode": regime.get("adaptive_selected_mode"),
                "cash_no_buy_day": bool(
                    cfg["base_group"] in ("adaptive_regime_v1", "adaptive_v1_cash_no_buy")
                    and regime.get("adaptive_selected_mode") == "cash"
                ),
            }
        )
        prev_equity = total_equity

    for pos in positions:
        trades.append(
            {
                "strategy_name": strategy_name,
                "symbol": pos["symbol"],
                "original_signal_date": pos["original_signal_date"],
                "entry_trade_date": pos["entry_trade_date"],
                "exit_trade_date": None,
                "holding_trading_days": holding_days,
                "entry_price": pos["entry_price"],
                "exit_price": None,
                "shares": pos["shares"],
                "entry_value": pos["entry_value"],
                "exit_value": None,
                "pnl": None,
                "return_pct": None,
                "exit_reason": "open",
                "production_rank": pos.get("production_rank"),
                "special_tier": pos.get("special_tier"),
                "special_score": pos.get("special_score"),
                "liquidity_safe_score": pos.get("liquidity_safe_score"),
                "balanced_score": pos.get("balanced_score"),
                "momentum_quality_score": pos.get("momentum_quality_score"),
                "tv_momentum_score": pos.get("tv_momentum_score"),
                "source_type": pos.get("source_type"),
                "pending_age_trading_days": pos.get("pending_age_trading_days"),
                "revalidated_in_same_group": pos.get("revalidated_in_same_group"),
                "stale_entry_warning": pos.get("stale_entry_warning"),
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(trades), pd.DataFrame(pending_events), pd.DataFrame(signal_rows)


def _monthly_benchmark(summary_df: pd.DataFrame, price_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), ["as_of_date"]].copy()
    expected["month"] = expected["as_of_date"].astype(str).str.slice(0, 7)
    rows: list[dict[str, Any]] = []
    for month, grp in expected.groupby("month", sort=True):
        row: dict[str, Any] = {"month": month}
        for symbol, prefix in BENCHMARKS:
            df_sym = price_cache.get(symbol)
            if df_sym is not None and not df_sym.empty:
                df_month = df_sym[df_sym["date_str"].str.slice(0, 7) == month]
                if not df_month.empty:
                    df_month = df_month.sort_values("date")
                    entry = float(df_month.iloc[0]["close"])
                    exitp = float(df_month.iloc[-1]["close"])
                else:
                    entry, exitp = None, None
            else:
                entry, exitp = None, None
            row[f"{prefix}_return_pct"] = ((exitp / entry - 1.0) * 100.0) if entry not in (None, 0) and exitp is not None else None
            row[f"{prefix}_missing"] = bool(entry in (None, 0) or exitp is None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _monthly_strategy_metrics(strategy_name: str, daily_df: pd.DataFrame, trades_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    daily = daily_df.copy()
    daily["month"] = daily["as_of_date"].astype(str).str.slice(0, 7)
    if not trades_df.empty and "exit_trade_date" in trades_df.columns:
        closed = trades_df.loc[trades_df["exit_trade_date"].notna()].copy()
    else:
        closed = pd.DataFrame()
    if not closed.empty:
        closed["month"] = closed["exit_trade_date"].astype(str).str.slice(0, 7)
    for month, grp in daily.groupby("month", sort=True):
        grp = grp.sort_values("as_of_date")
        month_start_equity = float(grp.iloc[0]["total_equity"])
        month_end_equity = float(grp.iloc[-1]["total_equity"])
        month_return = (month_end_equity / month_start_equity - 1.0) * 100.0 if month_start_equity else None
        closed_grp = closed.loc[closed["month"] == month].copy() if not closed.empty else pd.DataFrame()
        bench = benchmark_df.loc[benchmark_df["month"] == month]
        xu100 = bench["xu100_return_pct"].iloc[0] if not bench.empty else None
        xutum = bench["xutum_return_pct"].iloc[0] if not bench.empty else None
        eq_series = pd.to_numeric(grp["total_equity"], errors="coerce")
        running_max = eq_series.cummax()
        dd = ((eq_series / running_max) - 1.0) * 100.0
        rows.append(
            {
                "month": month,
                "strategy_name": strategy_name,
                "month_start_equity": month_start_equity,
                "month_end_equity": month_end_equity,
                "month_return_pct": month_return,
                "realized_pnl": float(pd.to_numeric(closed_grp["pnl"], errors="coerce").sum()) if not closed_grp.empty else 0.0,
                "unrealized_pnl": float(grp.iloc[-1]["open_position_value"] - grp.iloc[-1]["open_cost_basis"]) if not grp.empty else 0.0,
                "closed_trades_count": int(len(closed_grp)),
                "opened_trades_count": int((trades_df["entry_trade_date"].astype(str).str.slice(0, 7) == month).sum()) if not trades_df.empty else 0,
                "win_rate_closed_trades_pct": float((pd.to_numeric(closed_grp["return_pct"], errors="coerce") > 0).mean() * 100.0) if not closed_grp.empty else None,
                "avg_closed_trade_return_pct": float(pd.to_numeric(closed_grp["return_pct"], errors="coerce").mean()) if not closed_grp.empty else None,
                "open_positions_month_end": int(grp.iloc[-1]["open_position_count"]),
                "avg_open_position_count": float(pd.to_numeric(grp["open_position_count"], errors="coerce").mean()),
                "avg_cash_ratio_pct": float((pd.to_numeric(grp["cash"], errors="coerce") / pd.to_numeric(grp["total_equity"], errors="coerce")).mean() * 100.0),
                "max_drawdown_in_month_pct": float(dd.min()) if not dd.empty else None,
                "xu100_return_pct": xu100,
                "xutum_return_pct": xutum,
                "excess_vs_xu100_pct": (month_return - xu100) if month_return is not None and xu100 is not None else None,
                "excess_vs_xutum_pct": (month_return - xutum) if month_return is not None and xutum is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _benchmark_total(summary_df: pd.DataFrame, price_cache: dict[str, pd.DataFrame]) -> dict[str, float | None]:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    if not expected:
        return {"xu100_total_return_pct": None, "xutum_total_return_pct": None}
    out: dict[str, float | None] = {}
    for symbol, prefix in BENCHMARKS:
        df_sym = price_cache.get(symbol)
        if df_sym is not None and not df_sym.empty:
            df_filtered = df_sym[df_sym["date_str"].isin(expected)]
            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values("date")
                start_close = float(df_filtered.iloc[0]["close"])
                end_close = float(df_filtered.iloc[-1]["close"])
            else:
                start_close, end_close = None, None
        else:
            start_close, end_close = None, None
        out[f"{prefix}_total_return_pct"] = ((end_close / start_close - 1.0) * 100.0) if start_close not in (None, 0) and end_close is not None else None
    return out


def _summarize_strategy(
    strategy_name: str,
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    benchmark_total: dict[str, float | None],
    pending_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict[str, Any]:
    final_equity = float(daily_df.iloc[-1]["total_equity"]) if not daily_df.empty else float(initial_capital)
    total_return = ((final_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else None
    eq_series = pd.to_numeric(daily_df["total_equity"], errors="coerce")
    running_max = eq_series.cummax()
    dd = ((eq_series / running_max) - 1.0) * 100.0 if not eq_series.empty else pd.Series(dtype=float)
    if not trades_df.empty and "exit_trade_date" in trades_df.columns:
        closed = trades_df.loc[trades_df["exit_trade_date"].notna()].copy()
    else:
        closed = pd.DataFrame()
    trade_returns = pd.to_numeric(closed["return_pct"], errors="coerce").dropna() if not closed.empty and "return_pct" in closed.columns else pd.Series(dtype=float)
    month_returns = pd.to_numeric(monthly_df["month_return_pct"], errors="coerce").dropna() if not monthly_df.empty else pd.Series(dtype=float)
    best_month_idx = month_returns.idxmax() if not month_returns.empty else None
    worst_month_idx = month_returns.idxmin() if not month_returns.empty else None
    return {
        "strategy_name": strategy_name,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "xu100_total_return_pct": benchmark_total.get("xu100_total_return_pct"),
        "xutum_total_return_pct": benchmark_total.get("xutum_total_return_pct"),
        "excess_vs_xu100_pct": (total_return - benchmark_total["xu100_total_return_pct"]) if total_return is not None and benchmark_total.get("xu100_total_return_pct") is not None else None,
        "excess_vs_xutum_pct": (total_return - benchmark_total["xutum_total_return_pct"]) if total_return is not None and benchmark_total.get("xutum_total_return_pct") is not None else None,
        "max_drawdown_pct": float(dd.min()) if not dd.empty else None,
        "months_tested": int(len(month_returns)),
        "positive_month_rate_pct": float((month_returns > 0).mean() * 100.0) if not month_returns.empty else None,
        "avg_monthly_return_pct": float(month_returns.mean()) if not month_returns.empty else None,
        "median_monthly_return_pct": float(month_returns.median()) if not month_returns.empty else None,
        "best_month": None if best_month_idx is None else str(monthly_df.loc[best_month_idx, "month"]),
        "best_month_return_pct": None if best_month_idx is None else float(monthly_df.loc[best_month_idx, "month_return_pct"]),
        "worst_month": None if worst_month_idx is None else str(monthly_df.loc[worst_month_idx, "month"]),
        "worst_month_return_pct": None if worst_month_idx is None else float(monthly_df.loc[worst_month_idx, "month_return_pct"]),
        "trade_count": int(len(trades_df)),
        "closed_trade_count": int(len(closed)),
        "open_trade_count": int(len(trades_df) - len(closed)),
        "win_rate_pct": float((trade_returns > 0).mean() * 100.0) if not trade_returns.empty else None,
        "avg_trade_return_pct": float(trade_returns.mean()) if not trade_returns.empty else None,
        "median_trade_return_pct": float(trade_returns.median()) if not trade_returns.empty else None,
        "best_trade_return_pct": float(trade_returns.max()) if not trade_returns.empty else None,
        "worst_trade_return_pct": float(trade_returns.min()) if not trade_returns.empty else None,
        "avg_open_position_count": float(pd.to_numeric(daily_df["open_position_count"], errors="coerce").mean()) if not daily_df.empty else None,
        "avg_cash_ratio_pct": float((pd.to_numeric(daily_df["cash"], errors="coerce") / pd.to_numeric(daily_df["total_equity"], errors="coerce")).mean() * 100.0) if not daily_df.empty else None,
        "skipped_duplicate_symbol_count": int(pd.to_numeric(daily_df["skipped_duplicate_symbol_count"], errors="coerce").sum()) if not daily_df.empty else 0,
        "skipped_no_slot_count": int(pd.to_numeric(daily_df["skipped_no_slot_count"], errors="coerce").sum()) if not daily_df.empty else 0,
        "pending_queued_count": int((pending_df["event_type"] == "queued").sum()) if not pending_df.empty else 0,
        "pending_bought_count": int((pending_df["event_type"] == "bought_from_queue").sum()) if not pending_df.empty else 0,
        "pending_expired_count": int((pending_df["event_type"] == "expired").sum()) if not pending_df.empty else 0,
        "pending_revalidate_failed_count": int((pending_df["event_type"] == "revalidate_failed").sum()) if not pending_df.empty else 0,
        "cash_no_buy_day_count": int(daily_df["cash_no_buy_day"].fillna(False).astype(bool).sum()) if not daily_df.empty and "cash_no_buy_day" in daily_df.columns else 0,
    }


def _daily_new_signal_tables(
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    max_holdings: int,
    market_regime_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), ["as_of_date", "stale_data_warning", "stale_data_reason"]].copy()
    expected_dates = expected["as_of_date"].astype(str).tolist()

    # Pre-convert dates to strings once and group/index for O(1) lookups
    daily_all_str = daily_all.copy()
    daily_all_str["as_of_date_str"] = daily_all_str["as_of_date"].astype(str).str.slice(0, 10)
    daily_all_by_date = {d: grp for d, grp in daily_all_str.groupby("as_of_date_str")}

    expected_str = expected.copy()
    expected_str["as_of_date_str"] = expected_str["as_of_date"].astype(str).str.slice(0, 10)
    expected_meta_by_date = expected_str.set_index("as_of_date_str").to_dict('index')

    summary_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    for filter_name in FILTER_NAMES:
        seen: set[str] = set()
        prev: set[str] = set()
        for as_of_date in expected_dates:
            meta = expected_meta_by_date.get(as_of_date)
            if not meta:
                continue
            day_df = daily_all_by_date.get(as_of_date)
            if day_df is None or day_df.empty:
                day_df = pd.DataFrame()
            else:
                day_df = day_df.copy()

            if filter_name in ("adaptive_regime_v1", "adaptive_v1_cash_no_buy") and market_regime_map is not None:
                regime = market_regime_map.get(as_of_date, {})
                day_df.attrs["weak_score"] = regime.get("weak_score", 0)
            sel = _select_group(day_df, filter_name, max_holdings)
            symbols = sel["symbol"].astype(str).tolist() if not sel.empty else []
            curr = set(symbols)
            repeated = curr & prev
            new = curr - prev
            dropped = prev - curr
            first_time = [s for s in new if s not in seen]
            if not sel.empty and "symbol" in sel.columns:
                for _, row in sel.loc[sel["symbol"].astype(str).isin(new)].iterrows():
                    sym = str(row["symbol"])
                    new_rows.append(
                        {
                            "as_of_date": as_of_date,
                            "filter_name": filter_name,
                            "symbol": sym,
                            "production_rank": row.get("production_rank"),
                            "special_tier": row.get("special_tier"),
                            "special_score": row.get("special_score"),
                            "liquidity_safe_score": row.get("liquidity_safe_score"),
                            "balanced_score": row.get("balanced_score"),
                            "momentum_quality_score": row.get("momentum_quality_score"),
                            "tv_momentum_score": row.get("tv_momentum_score"),
                            "volume_ratio_20d": row.get("volume_ratio_20d"),
                            "turnover_ratio_20d": row.get("turnover_ratio_20d"),
                            "avg_turnover_20d": row.get("avg_turnover_20d"),
                            "avg_turnover_30d": row.get("avg_turnover_30d"),
                            "turnover_today": row.get("turnover_today"),
                            "rsi_14": row.get("rsi_14"),
                            "macd_hist": row.get("macd_hist"),
                            "adx_14": row.get("adx_14"),
                            "adr_pct": row.get("adr_pct"),
                            "ema8": row.get("ema8"),
                            "ema21": row.get("ema21"),
                            "ema20": row.get("ema20"),
                            "ema50": row.get("ema50"),
                            "ema60": row.get("ema60"),
                            "close": row.get("close"),
                            "close_gt_ema20": row.get("close_gt_ema20"),
                            "ema20_gt_ema50": row.get("ema20_gt_ema50"),
                            "ema8_gte_ema21": row.get("ema8_gte_ema21"),
                            "close_gt_ema60": row.get("close_gt_ema60"),
                            "perf_3m_pct": row.get("perf_3m_pct"),
                            "perf_6m_pct": row.get("perf_6m_pct"),
                            "daily_change_pct": row.get("daily_change_pct"),
                            "daily_change_gt_2": row.get("daily_change_gt_2"),
                            "price_above_52w_low_pct": row.get("price_above_52w_low_pct"),
                            "price_above_52w_low_gte_70": row.get("price_above_52w_low_gte_70"),
                            "first_time_seen_in_filter": sym in first_time,
                            "stale_data_warning": bool(meta["stale_data_warning"]),
                            "stale_data_reason": meta["stale_data_reason"],
                        }
                    )
            seen.update(curr)
            total = len(curr)
            new_cnt = len(new)
            summary_rows.append(
                {
                    "as_of_date": as_of_date,
                    "filter_name": filter_name,
                    "total_signal_count": total,
                    "repeated_vs_prev_trading_day_count": len(repeated),
                    "new_vs_prev_trading_day_count": new_cnt,
                    "dropped_vs_prev_trading_day_count": len(dropped),
                    "new_signal_rate_pct": (new_cnt / total * 100.0) if total else None,
                    "repeated_rate_pct": (len(repeated) / total * 100.0) if total else None,
                    "unique_symbols_seen_to_date": len(seen),
                    "never_seen_before_count": len(first_time),
                    "stale_data_warning": bool(meta["stale_data_warning"]),
                    "stale_data_reason": meta["stale_data_reason"],
                }
            )
            prev = curr
        fsum = pd.DataFrame([r for r in summary_rows if r["filter_name"] == filter_name])
        agg_rows.append(
            {
                "filter_name": filter_name,
                "avg_daily_total_count": float(pd.to_numeric(fsum["total_signal_count"], errors="coerce").mean()) if not fsum.empty else None,
                "avg_daily_new_count": float(pd.to_numeric(fsum["new_vs_prev_trading_day_count"], errors="coerce").mean()) if not fsum.empty else None,
                "median_daily_new_count": float(pd.to_numeric(fsum["new_vs_prev_trading_day_count"], errors="coerce").median()) if not fsum.empty else None,
                "avg_new_signal_rate_pct": float(pd.to_numeric(fsum["new_signal_rate_pct"], errors="coerce").mean()) if not fsum.empty else None,
                "unique_symbols_seen": int(pd.to_numeric(fsum["unique_symbols_seen_to_date"], errors="coerce").max()) if not fsum.empty else 0,
                "never_seen_total": int(pd.to_numeric(fsum["never_seen_before_count"], errors="coerce").sum()) if not fsum.empty else 0,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(new_rows), pd.DataFrame(agg_rows)


def _forward20_signals(
    daily_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    signal_decisions: pd.DataFrame,
    max_holdings: int,
    market_regime_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_dates = summary_df.loc[summary_df["is_expected_trading_day"].astype(bool), "as_of_date"].astype(str).tolist()
    idx_map = {d: i for i, d in enumerate(expected_dates)}
    strat_map = {
        "top30": "top30_fresh_only",
        "special_loose": None,
        "special_mid": "special_mid_fresh_only",
        "special_strict": "special_strict_fresh_only",
        "special_strict_top10": "special_strict_top10_fresh_only",
        "tv_volume_momentum_trend": "tv_volume_momentum_trend_fresh_only",
        "special_strict_quality_v2": "special_strict_quality_v2_fresh_only",
        "adaptive_v1_cash_no_buy": "adaptive_v1_cash_no_buy_fresh_only",
        "special_strict_score_threshold_50": "special_strict_score_threshold_50_fresh_only",
        "special_strict_score_threshold_60": "special_strict_score_threshold_60_fresh_only",
    }

    # Pre-convert daily_all as_of_date to string once and group
    daily_all_str = daily_all.copy()
    daily_all_str["as_of_date_str"] = daily_all_str["as_of_date"].astype(str).str.slice(0, 10)
    daily_all_by_date = {d: grp for d, grp in daily_all_str.groupby("as_of_date_str")}

    # Pre-index signal_decisions by strategy_name, signal_date, and symbol for O(1) lookups
    decisions_map = {}
    if not signal_decisions.empty:
        sd = signal_decisions.copy()
        sd["signal_date_str"] = sd["signal_date"].astype(str).str.slice(0, 10)
        sd["symbol_str"] = sd["symbol"].astype(str)
        for (sname, sdate, sym), grp in sd.groupby(["strategy_name", "signal_date_str", "symbol_str"]):
            decisions_map[(sname, sdate, sym)] = grp

    rows: list[dict[str, Any]] = []
    for filter_name in FILTER_NAMES:
        for signal_date in expected_dates:
            day_df = daily_all_by_date.get(signal_date)
            if day_df is None or day_df.empty:
                continue
            day_df = day_df.copy()
            if filter_name in ("adaptive_regime_v1", "adaptive_v1_cash_no_buy") and market_regime_map is not None:
                regime = market_regime_map.get(signal_date, {})
                day_df.attrs["weak_score"] = regime.get("weak_score", 0)
            sel = _select_group(day_df, filter_name, max_holdings)
            if sel.empty:
                continue
            idx = idx_map[signal_date]
            exit_idx = idx + 20
            exit_date = expected_dates[exit_idx] if exit_idx < len(expected_dates) else None
            for _, row in sel.iterrows():
                symbol = str(row["symbol"])
                signal_close = _price_on_date(price_cache.get(symbol), signal_date, "close")
                exit_close = _price_on_date(price_cache.get(symbol), exit_date, "close") if exit_date else None
                fwd = ((exit_close / signal_close - 1.0) * 100.0) if signal_close not in (None, 0) and exit_close is not None else None
                sname = strat_map.get(filter_name)

                dec = decisions_map.get((sname, signal_date, symbol)) if sname else None
                was_bought = bool(dec["was_bought_by_portfolio"].any()) if dec is not None and not dec.empty else False
                skipped_full = bool(dec["was_skipped_due_to_full_slots"].any()) if dec is not None and not dec.empty else False
                skipped_hold = bool(dec["was_skipped_due_to_already_holding"].any()) if dec is not None and not dec.empty else False
                skipped_miss = bool(dec["was_skipped_due_to_missing_price"].any()) if dec is not None and not dec.empty else (signal_close is None or (exit_date is not None and exit_close is None))
                rows.append(
                    {
                        "signal_date": signal_date,
                        "filter_name": filter_name,
                        "symbol": symbol,
                        "signal_close": signal_close,
                        "exit_date_20td": exit_date,
                        "exit_close": exit_close,
                        "forward_return_20d_pct": fwd,
                        "was_bought_by_portfolio": was_bought,
                        "was_skipped_due_to_full_slots": skipped_full,
                        "was_skipped_due_to_already_holding": skipped_hold,
                        "was_skipped_due_to_missing_price": skipped_miss,
                        "production_rank": row.get("production_rank"),
                        "special_tier": row.get("special_tier"),
                        "special_score": row.get("special_score"),
                        "liquidity_safe_score": row.get("liquidity_safe_score"),
                        "balanced_score": row.get("balanced_score"),
                        "momentum_quality_score": row.get("momentum_quality_score"),
                        "tv_momentum_score": row.get("tv_momentum_score"),
                    }
                )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for filter_name, grp in detail.groupby("filter_name", sort=True):
        gret = pd.to_numeric(grp["forward_return_20d_pct"], errors="coerce")
        b = grp.loc[grp["was_bought_by_portfolio"]]
        s = grp.loc[~grp["was_bought_by_portfolio"]]
        s_full = grp.loc[grp["was_skipped_due_to_full_slots"]]
        best = s.loc[pd.to_numeric(s["forward_return_20d_pct"], errors="coerce").idxmax()] if not s.empty and pd.to_numeric(s["forward_return_20d_pct"], errors="coerce").notna().any() else None
        worst = s.loc[pd.to_numeric(s["forward_return_20d_pct"], errors="coerce").idxmin()] if not s.empty and pd.to_numeric(s["forward_return_20d_pct"], errors="coerce").notna().any() else None
        summary_rows.append(
            {
                "filter_name": filter_name,
                "total_signals": int(len(grp)),
                "bought_signal_count": int(grp["was_bought_by_portfolio"].sum()),
                "skipped_signal_count": int((~grp["was_bought_by_portfolio"]).sum()),
                "skipped_due_to_full_slots_count": int(grp["was_skipped_due_to_full_slots"].sum()),
                "skipped_due_to_already_holding_count": int(grp["was_skipped_due_to_already_holding"].sum()),
                "avg_forward_return_all_signals_pct": float(gret.mean()) if gret.notna().any() else None,
                "avg_forward_return_bought_pct": float(pd.to_numeric(b["forward_return_20d_pct"], errors="coerce").mean()) if not b.empty else None,
                "avg_forward_return_skipped_pct": float(pd.to_numeric(s["forward_return_20d_pct"], errors="coerce").mean()) if not s.empty else None,
                "avg_forward_return_skipped_full_slots_pct": float(pd.to_numeric(s_full["forward_return_20d_pct"], errors="coerce").mean()) if not s_full.empty else None,
                "best_skipped_symbol": None if best is None else best["symbol"],
                "best_skipped_signal_date": None if best is None else best["signal_date"],
                "best_skipped_forward_return_20d_pct": None if best is None else float(best["forward_return_20d_pct"]),
                "worst_skipped_symbol": None if worst is None else worst["symbol"],
                "worst_skipped_signal_date": None if worst is None else worst["signal_date"],
                "worst_skipped_forward_return_20d_pct": None if worst is None else float(worst["forward_return_20d_pct"]),
            }
        )
    return detail, pd.DataFrame(summary_rows).sort_values("filter_name").reset_index(drop=True)


def _run_single_strategy(task_args):
    (
        strategy_name,
        daily_all,
        summary_df,
        price_cache,
        initial_capital,
        max_holdings,
        holding_days,
        entry_mode,
        exit_mode,
        exclude_stale_new_entries,
        allow_same_day_reentry,
        min_position_value,
        market_regime_map,
    ) = task_args
    daily_df, trades_df, pending_df, signal_df = simulate_strategy(
        strategy_name,
        daily_all,
        summary_df,
        price_cache,
        initial_capital=initial_capital,
        max_holdings=max_holdings,
        holding_days=holding_days,
        entry_mode=entry_mode,
        exit_mode=exit_mode,
        exclude_stale_new_entries=exclude_stale_new_entries,
        allow_same_day_reentry=allow_same_day_reentry,
        min_position_value=min_position_value,
        market_regime_map=market_regime_map,
    )
    return strategy_name, daily_df, trades_df, pending_df, signal_df


def run_simulation(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    daily_all, summary_df, price_cache, resolved_end, coverage_summary = _prepare_inputs(
        args.output_dir,
        args.candidate_features_path,
        args.db_path,
        args.start_date,
        args.end_date,
        args.exclude_stale_new_entries,
    )

    # Pre-index price_cache by date_str for fast O(1) lookups during simulation
    for symbol in price_cache:
        price_cache[symbol] = price_cache[symbol].set_index("date_str", drop=False)

    all_daily: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_pending: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    monthly_returns_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    benchmark_monthly = _monthly_benchmark(summary_df, price_cache)
    benchmark_total = _benchmark_total(summary_df, price_cache)
    market_regime_map = _build_market_regime_map(daily_all, summary_df, price_cache)

    # Parallel strategy execution setup
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    tasks = []
    for strategy_name in STRATEGY_NAMES:
        tasks.append((
            strategy_name,
            daily_all,
            summary_df,
            price_cache,
            float(args.initial_capital),
            int(args.max_holdings),
            int(args.holding_days),
            args.entry_mode,
            args.exit_mode,
            bool(args.exclude_stale_new_entries),
            bool(args.allow_same_day_reentry),
            float(args.min_position_value),
            market_regime_map,
        ))

    try:
        multiprocessing.set_start_method('fork', force=True)
    except Exception:
        pass

    num_workers = min(multiprocessing.cpu_count(), len(STRATEGY_NAMES))
    print(f"Running simulation for {len(STRATEGY_NAMES)} strategies in parallel using {num_workers} processes...")

    strategy_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_single_strategy, t) for t in tasks]
        for fut in futures:
            strategy_results.append(fut.result())

    # Map back results in original order
    results_map = {res[0]: res[1:] for res in strategy_results}

    for strategy_name in STRATEGY_NAMES:
        daily_df, trades_df, pending_df, signal_df = results_map[strategy_name]
        monthly_df = _monthly_strategy_metrics(strategy_name, daily_df, trades_df, benchmark_monthly)
        all_daily.append(daily_df)
        all_trades.append(trades_df)
        all_pending.append(pending_df)
        all_signals.append(signal_df)
        all_monthly.append(monthly_df)
        for _, row in monthly_df.iterrows():
            m = str(row["month"])
            c = trades_df.loc[trades_df["entry_trade_date"].astype(str).str.startswith(m)] if not trades_df.empty and "entry_trade_date" in trades_df.columns else pd.DataFrame()
            cc = trades_df.loc[trades_df["exit_trade_date"].astype(str).str.startswith(m)] if not trades_df.empty and "exit_trade_date" in trades_df.columns else pd.DataFrame()
            monthly_returns_rows.append(
                {
                    "month": m,
                    "filter_name": strategy_name.replace("_fresh_only", ""),
                    "monthly_return_pct": row.get("month_return_pct"),
                    "month_start_capital": row.get("month_start_equity"),
                    "month_end_capital": row.get("month_end_equity"),
                    "trades_opened": int(len(c)),
                    "trades_closed": int(len(cc)),
                    "avg_hold_days": float(pd.to_numeric(cc["holding_trading_days"], errors="coerce").mean()) if not cc.empty else None,
                    "win_rate_pct": float((pd.to_numeric(cc["return_pct"], errors="coerce") > 0).mean() * 100.0) if not cc.empty else None,
                }
            )
        summary_rows.append(
            _summarize_strategy(
                strategy_name,
                daily_df,
                trades_df,
                monthly_df,
                benchmark_total,
                pending_df,
                args.start_date,
                resolved_end,
                float(args.initial_capital),
            )
        )

    daily_summary, new_symbols, new_signal_agg = _daily_new_signal_tables(
        daily_all,
        summary_df,
        int(args.max_holdings),
        market_regime_map=market_regime_map,
    )
    signal_decisions = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    fwd_detail, fwd_summary = _forward20_signals(
        daily_all,
        summary_df,
        price_cache,
        signal_decisions,
        int(args.max_holdings),
        market_regime_map=market_regime_map,
    )

    return {
        "daily": pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame(),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "pending": pd.concat(all_pending, ignore_index=True) if all_pending else pd.DataFrame(),
        "monthly": pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame(),
        "summary": pd.DataFrame(summary_rows).sort_values("strategy_name").reset_index(drop=True),
        "benchmark_monthly": benchmark_monthly,
        "daily_filter_new_signal_summary": daily_summary,
        "daily_filter_new_signal_symbols": new_symbols,
        "filter_signal_forward_return_20d": fwd_detail,
        "filter_missed_opportunity_summary": fwd_summary,
        "rolling_portfolio_monthly_returns": pd.DataFrame(monthly_returns_rows).sort_values(["month", "filter_name"]).reset_index(drop=True),
        "new_signal_agg": new_signal_agg,
        "adaptive_regime_summary": pd.DataFrame(
            [{"date": d, **meta} for d, meta in market_regime_map.items()]
        ),
    }


def _console_rolling_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        [
            "strategy_name",
            "final_equity",
            "total_return_pct",
            "closed_trade_count",
            "win_rate_pct",
            "avg_trade_return_pct",
            "max_drawdown_pct",
            "cash_no_buy_day_count",
        ]
    ].copy()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = run_simulation(args)
    results["daily"].to_csv(out_root / "rolling_portfolio_daily_2026.csv", index=False)
    results["trades"].to_csv(out_root / "rolling_portfolio_trades_2026.csv", index=False)
    results["pending"].to_csv(out_root / "rolling_portfolio_pending_events_2026.csv", index=False)
    results["monthly"].to_csv(out_root / "rolling_portfolio_monthly_2026.csv", index=False)
    results["summary"].to_csv(out_root / "rolling_portfolio_summary_2026.csv", index=False)
    results["benchmark_monthly"].to_csv(out_root / "rolling_portfolio_benchmark_monthly_2026.csv", index=False)
    results["daily_filter_new_signal_summary"].to_csv(out_root / "daily_filter_new_signal_summary_2026.csv", index=False)
    results["daily_filter_new_signal_symbols"].to_csv(out_root / "daily_filter_new_signal_symbols_2026.csv", index=False)
    results["filter_signal_forward_return_20d"].to_csv(out_root / "filter_signal_forward_return_20d_2026.csv", index=False)
    results["filter_missed_opportunity_summary"].to_csv(out_root / "filter_missed_opportunity_summary_2026.csv", index=False)
    results["rolling_portfolio_monthly_returns"].to_csv(out_root / "rolling_portfolio_monthly_returns_2026.csv", index=False)
    results["adaptive_regime_summary"].to_csv(out_root / "adaptive_regime_summary_2026.csv", index=False)

    print("FILTER_DAILY_NEW_SIGNAL_SUMMARY")
    print(results["new_signal_agg"].to_string(index=False))
    print("ROLLING_PORTFOLIO_SUMMARY")
    print(_console_rolling_summary(results["summary"]).to_string(index=False))
    print("MISSED_OPPORTUNITY_SUMMARY")
    miss = results["filter_missed_opportunity_summary"][
        [
            "filter_name",
            "total_signals",
            "bought_signal_count",
            "skipped_due_to_full_slots_count",
            "avg_forward_return_bought_pct",
            "avg_forward_return_skipped_full_slots_pct",
            "best_skipped_symbol",
            "best_skipped_forward_return_20d_pct",
        ]
    ].copy()
    print(miss.to_string(index=False))


if __name__ == "__main__":
    main()

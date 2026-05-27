from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_OUT_ROOT = "/data/backtest_outputs/daily_radar_stability_2026_05_21_22"
DAY1 = "2026-05-21"
DAY2 = "2026-05-22"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--day1", default=DAY1)
    p.add_argument("--day2", default=DAY2)
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
    # Mirrors market_radar.radar_engine.compute_production_scores.
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
    path = csv
    if not path.exists():
        raise FileNotFoundError(f"candidate features missing: {path}")
    return pd.read_csv(path)


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
    f = raw.copy()
    f = f.loc[f["strategy"].astype(str) == "volume_spike_strict"].copy()
    f["signal_date"] = pd.to_datetime(f["signal_date"], errors="coerce").dt.tz_localize(None)
    f = _to_numeric(
        f,
        [
            "close",
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


def build_daily_radar(features: pd.DataFrame, as_of_date: str) -> tuple[pd.DataFrame, bool]:
    asof = pd.Timestamp(as_of_date)
    f = features.loc[features["signal_date"] <= asof].copy()
    if f.empty:
        return pd.DataFrame(), True

    f = f.sort_values(["symbol", "signal_date"]).drop_duplicates(subset=["symbol"], keep="last")
    f = f.loc[_relaxed_mask(f)].copy()
    if f.empty:
        return pd.DataFrame(), True

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
    f = f.sort_values(["production_score", "signal_date", "symbol"], ascending=[False, True, True]).reset_index(drop=True)
    f["production_rank"] = f.index + 1
    f["score_bucket"] = f["production_rank"].apply(_score_bucket)
    out = f[
        [
            "symbol",
            "signal_date",
            "production_rank",
            "score_bucket",
            "balanced_score",
            "momentum_quality_score",
            "liquidity_safe_score",
            "production_score",
        ]
    ].copy()
    out["as_of_date"] = as_of_date
    out["signal_date"] = out["signal_date"].dt.strftime("%Y-%m-%d")
    return out, False


def compute_overlap(day1_df: pd.DataFrame, day2_df: pd.DataFrame, day1: str, day2: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    d1 = day1_df.copy()
    d2 = day2_df.copy()
    d1["symbol"] = d1["symbol"].astype(str)
    d2["symbol"] = d2["symbol"].astype(str)

    cols1 = {
        "production_rank": f"rank_{day1.replace('-', '_')}",
        "production_score": f"score_{day1.replace('-', '_')}",
        "score_bucket": f"score_bucket_{day1.replace('-', '_')}",
    }
    cols2 = {
        "production_rank": f"rank_{day2.replace('-', '_')}",
        "production_score": f"score_{day2.replace('-', '_')}",
        "score_bucket": f"score_bucket_{day2.replace('-', '_')}",
    }

    x = d1[["symbol", *cols1.keys()]].rename(columns=cols1)
    y = d2[["symbol", *cols2.keys()]].rename(columns=cols2)
    ov = x.merge(y, on="symbol", how="outer")
    ov[f"in_{day1.replace('-', '_')}"] = ov[cols1["production_rank"]].notna()
    ov[f"in_{day2.replace('-', '_')}"] = ov[cols2["production_rank"]].notna()

    def _status(r: pd.Series) -> str:
        if bool(r[f"in_{day1.replace('-', '_')}"]) and bool(r[f"in_{day2.replace('-', '_')}"]):
            return "repeated"
        if bool(r[f"in_{day2.replace('-', '_')}"]):
            return "new"
        return "dropped"

    ov["status"] = ov.apply(_status, axis=1)
    ov["rank_change"] = ov[cols2["production_rank"]] - ov[cols1["production_rank"]]
    ov["score_change"] = ov[cols2["production_score"]] - ov[cols1["production_score"]]
    ov = ov.sort_values(["status", cols2["production_rank"], cols1["production_rank"], "symbol"], ascending=[True, True, True, True]).reset_index(drop=True)

    day1_symbols = set(d1["symbol"].tolist())
    day2_symbols = set(d2["symbol"].tolist())
    repeated = sorted(day1_symbols & day2_symbols)
    new = sorted(day2_symbols - day1_symbols)
    dropped = sorted(day1_symbols - day2_symbols)
    day1_count = len(day1_symbols)
    day2_count = len(day2_symbols)
    repeated_count = len(repeated)
    new_count = len(new)
    dropped_count = len(dropped)

    def _top(df: pd.DataFrame, n: int) -> list[str]:
        return df.loc[df["production_rank"] <= n, "symbol"].astype(str).tolist()

    top20_a = set(_top(d1, 20))
    top20_b = set(_top(d2, 20))
    top30_a = _top(d1, 30)
    top30_b = _top(d2, 30)
    top30_a_set = set(top30_a)
    top30_b_set = set(top30_b)
    top50_a = set(_top(d1, 50))
    top50_b = set(_top(d2, 50))

    r1 = d1.set_index("symbol")["production_rank"].to_dict()
    r2 = d2.set_index("symbol")["production_rank"].to_dict()
    stable_top30 = sorted(top30_a_set & top30_b_set)
    improved = sorted([s for s in stable_top30 if int(r2[s]) < int(r1[s])])
    worsened = sorted([s for s in stable_top30 if int(r2[s]) > int(r1[s])])

    summary: dict[str, Any] = {
        "day1": day1,
        "day2": day2,
        "day1_signal_count": day1_count,
        "day2_signal_count": day2_count,
        "repeated_count": repeated_count,
        "new_count": new_count,
        "dropped_count": dropped_count,
        "overlap_rate": (repeated_count / day1_count) if day1_count else None,
        "new_rate": (new_count / day2_count) if day2_count else None,
        "dropped_rate": (dropped_count / day1_count) if day1_count else None,
        "top20_overlap_count": len(top20_a & top20_b),
        "top30_overlap_count": len(top30_a_set & top30_b_set),
        "top50_overlap_count": len(top50_a & top50_b),
        "top30_new_symbols": sorted(top30_b_set - top30_a_set),
        "top30_dropped_symbols": sorted(top30_a_set - top30_b_set),
        "top30_rank_improved_symbols": improved,
        "top30_rank_worsened_symbols": worsened,
        "interpretation": "",
    }
    return ov, summary


def _top_symbols(df: pd.DataFrame, n: int) -> list[str]:
    return df.loc[df["production_rank"] <= n, "symbol"].astype(str).tolist()


def _build_data_missing_summary(day1: str, day2: str, reason: str) -> dict[str, Any]:
    return {
        "day1": day1,
        "day2": day2,
        "data_missing": True,
        "reason": reason,
        "day1_signal_count": 0,
        "day2_signal_count": 0,
        "repeated_count": 0,
        "new_count": 0,
        "dropped_count": 0,
        "overlap_rate": None,
        "new_rate": None,
        "dropped_rate": None,
        "top20_overlap_count": 0,
        "top30_overlap_count": 0,
        "top50_overlap_count": 0,
        "top30_new_symbols": [],
        "top30_dropped_symbols": [],
        "interpretation": "data_missing",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    day1_csv = out_root / f"daily_radar_{args.day1}.csv"
    day2_csv = out_root / f"daily_radar_{args.day2}.csv"
    overlap_csv = out_root / f"daily_radar_overlap_{args.day1}_{args.day2}.csv"
    summary_json = out_root / "daily_radar_overlap_summary.json"

    try:
        base = _prepare_base_features(output_dir)
    except FileNotFoundError as exc:
        summary = _build_data_missing_summary(args.day1, args.day2, str(exc))
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print({"status": "data_missing", "summary_json": str(summary_json), "reason": str(exc)})
        return

    d1, missing1 = build_daily_radar(base, args.day1)
    d2, missing2 = build_daily_radar(base, args.day2)
    if missing1 or missing2:
        reason = f"no_signals_or_no_data_for_dates day1={args.day1} missing={missing1} day2={args.day2} missing={missing2}"
        summary = _build_data_missing_summary(args.day1, args.day2, reason)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        d1.to_csv(day1_csv, index=False)
        d2.to_csv(day2_csv, index=False)
        pd.DataFrame(columns=["symbol"]).to_csv(overlap_csv, index=False)
        print({"status": "data_missing", "summary_json": str(summary_json), "reason": reason})
        return

    d1.to_csv(day1_csv, index=False)
    d2.to_csv(day2_csv, index=False)

    overlap, summary = compute_overlap(d1, d2, args.day1, args.day2)
    overlap.to_csv(overlap_csv, index=False)

    summary["day1_total_signal_count"] = int(len(d1))
    summary["day2_total_signal_count"] = int(len(d2))
    summary["day1_unique_symbol_count"] = int(d1["symbol"].nunique())
    summary["day2_unique_symbol_count"] = int(d2["symbol"].nunique())
    summary["day1_top20_symbols"] = _top_symbols(d1, 20)
    summary["day1_top30_symbols"] = _top_symbols(d1, 30)
    summary["day1_top50_symbols"] = _top_symbols(d1, 50)
    summary["day2_top20_symbols"] = _top_symbols(d2, 20)
    summary["day2_top30_symbols"] = _top_symbols(d2, 30)
    summary["day2_top50_symbols"] = _top_symbols(d2, 50)
    summary["interpretation"] = (
        "stable" if (summary["overlap_rate"] is not None and summary["overlap_rate"] >= 0.6) else "changing"
    )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        {
            "day1_csv": str(day1_csv),
            "day2_csv": str(day2_csv),
            "overlap_csv": str(overlap_csv),
            "summary_json": str(summary_json),
            "day1_count": int(summary["day1_signal_count"]),
            "day2_count": int(summary["day2_signal_count"]),
            "repeated_count": int(summary["repeated_count"]),
            "new_count": int(summary["new_count"]),
            "dropped_count": int(summary["dropped_count"]),
        }
    )


if __name__ == "__main__":
    main()

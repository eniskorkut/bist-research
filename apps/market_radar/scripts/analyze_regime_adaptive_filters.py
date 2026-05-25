from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_DB_PATH = "/data/market_radar_cache.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-05-08")
    return parser.parse_args()


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_periods(output_dir: Path, start: str, end: str) -> list[str]:
    summary_path = output_dir / "period_strategy_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "period_start" in summary.columns and "strategy" in summary.columns:
            summary = summary.loc[summary["strategy"] == "volume_spike_strict"].copy()
            summary = summary.loc[(summary["period_start"] >= start) & (summary["period_start"] <= end)].copy()
            periods = sorted(summary["period_start"].astype(str).unique().tolist())
            if periods:
                return periods
    features_path = output_dir / "candidate_features.parquet"
    if features_path.exists():
        f = pd.read_parquet(features_path)
    else:
        f = pd.read_csv(output_dir / "candidate_features.csv")
    periods = sorted(f["period"].astype(str).unique().tolist())
    return [p for p in periods if start <= p <= end]


def _load_xu100_returns(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol='XU100'").fetchone()
    conn.close()
    if not row:
        raise ValueError("XU100 cache not found")
    payload = json.loads(row[0])
    recs = payload.get("records") or []
    xu = pd.DataFrame.from_records(recs)
    xu["date"] = pd.to_datetime(xu["date"], errors="coerce").dt.tz_localize(None)
    xu = xu.dropna(subset=["date"]).sort_values("date")
    xu["close"] = pd.to_numeric(xu["close"], errors="coerce")
    xu["close_20d_ago"] = xu["close"].shift(20)
    xu["xu100_return_20d_pct"] = (xu["close"] / xu["close_20d_ago"] - 1.0) * 100.0
    xu["date_str"] = xu["date"].dt.strftime("%Y-%m-%d")
    daily_idx = pd.date_range(xu["date"].min(), xu["date"].max(), freq="D")
    out = xu.set_index("date_str")[["xu100_return_20d_pct"]].copy()
    out = out.reindex(daily_idx.strftime("%Y-%m-%d")).ffill().reset_index()
    out = out.rename(columns={"index": "period"})
    return out


def _load_current_summary(output_dir: Path, periods: list[str]) -> pd.DataFrame:
    summary_path = output_dir / "period_strategy_summary.csv"
    if summary_path.exists():
        current = pd.read_csv(summary_path)
    else:
        current = pd.DataFrame()
    if not current.empty and {"strategy", "period_start", "signal_count", "basket_alpha_30d"}.issubset(current.columns):
        current = current.loc[current["strategy"] == "volume_spike_strict"].copy()
        current = current.loc[current["period_start"].isin(periods)].copy()
        current = _to_numeric(current, ["signal_count", "basket_alpha_30d"])
        current = current[["period_start", "signal_count", "basket_alpha_30d"]].rename(
            columns={
                "period_start": "period",
                "signal_count": "current_signal_count",
                "basket_alpha_30d": "current_alpha_30d",
            }
        )
    else:
        current = _build_current_from_candidates(output_dir, periods)
    current["current_net_alpha_30d"] = current["current_alpha_30d"] - 0.30
    return current


def _build_current_from_candidates(output_dir: Path, periods: list[str]) -> pd.DataFrame:
    features_path = output_dir / "candidate_features.parquet"
    if features_path.exists():
        f = pd.read_parquet(features_path)
    else:
        f = pd.read_csv(output_dir / "candidate_features.csv")
    f = f.loc[f["period"].astype(str).isin(periods)].copy()
    f["current_quality_pass"] = f["current_quality_pass"].astype(str).str.lower().eq("true")
    f = _to_numeric(f, ["alpha_30d"])
    cur = f.loc[f["current_quality_pass"]].copy()
    cur = cur.sort_values(["period", "signal_date", "entry_date", "symbol"])
    cur = cur.drop_duplicates(subset=["period", "symbol"], keep="first")
    grouped = (
        cur.groupby("period", as_index=False)
        .agg(current_signal_count=("symbol", "count"), current_alpha_30d=("alpha_30d", "mean"))
    )
    return grouped


def _build_relaxed_from_candidates(output_dir: Path, periods: list[str]) -> pd.DataFrame:
    features_path = output_dir / "candidate_features.parquet"
    if features_path.exists():
        f = pd.read_parquet(features_path)
    else:
        f = pd.read_csv(output_dir / "candidate_features.csv")
    f = f.loc[f["period"].astype(str).isin(periods)].copy()
    f = _to_numeric(
        f,
        [
            "turnover",
            "avg_turnover_20d",
            "close",
            "ma20",
            "rsi_14",
            "return_5d_pct",
            "return_10d_pct",
            "close_position",
            "future_return_30d",
            "benchmark_return_30d",
            "alpha_30d",
        ],
    )
    f["above_ma20"] = f["above_ma20"].astype(str).str.lower().eq("true")
    base_pass = (
        (f["turnover"] >= 10_000_000.0)
        & (f["avg_turnover_20d"] >= 10_000_000.0)
        & (f["above_ma20"] | (f["close"] >= f["ma20"]))
        & (f["rsi_14"] <= 78.0)
        & (f["return_5d_pct"] <= 35.0)
        & (f["return_10d_pct"] <= 60.0)
    )
    relaxed = f.loc[base_pass & (f["close_position"] >= 0.50)].copy()
    relaxed = relaxed.sort_values(["period", "signal_date", "entry_date", "symbol"])
    relaxed = relaxed.drop_duplicates(subset=["period", "symbol"], keep="first")
    grouped = (
        relaxed.groupby("period", as_index=False)
        .agg(
            relaxed_signal_count=("symbol", "count"),
            relaxed_alpha_30d=("alpha_30d", "mean"),
            relaxed_signature=("symbol", lambda s: "|".join(sorted(s.astype(str).tolist()))),
        )
    )
    grouped["relaxed_net_alpha_30d"] = grouped["relaxed_alpha_30d"] - 0.30
    return grouped


def _build_current_signatures(output_dir: Path, periods: list[str]) -> pd.DataFrame:
    features_path = output_dir / "candidate_features.parquet"
    if features_path.exists():
        f = pd.read_parquet(features_path)
    else:
        f = pd.read_csv(output_dir / "candidate_features.csv")
    f = f.loc[f["period"].astype(str).isin(periods)].copy()
    f["current_quality_pass"] = f["current_quality_pass"].astype(str).str.lower().eq("true")
    cur = f.loc[f["current_quality_pass"]].copy()
    cur = cur.sort_values(["period", "signal_date", "entry_date", "symbol"])
    cur = cur.drop_duplicates(subset=["period", "symbol"], keep="first")
    return (
        cur.groupby("period", as_index=False)
        .agg(
            current_sig_count_from_features=("symbol", "count"),
            current_signature=("symbol", lambda s: "|".join(sorted(s.astype(str).tolist()))),
        )
    )


def build_regime_comparison(output_dir: Path, db_path: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    periods = _load_periods(output_dir, start, end)
    current = _load_current_summary(output_dir, periods)
    relaxed = _build_relaxed_from_candidates(output_dir, periods)
    xu = _load_xu100_returns(db_path)
    xu = xu.loc[xu["period"].isin(periods)].copy()
    cur_sig = _build_current_signatures(output_dir, periods)

    merged = current.merge(relaxed, on="period", how="left").merge(xu, on="period", how="left").merge(cur_sig, on="period", how="left")
    merged["regime_label"] = merged["xu100_return_20d_pct"].apply(lambda x: "Bull" if pd.notna(x) and x > 1.5 else "WeakOrNeutral")
    merged["selected_config"] = merged["regime_label"].map(lambda x: "current_config" if x == "Bull" else "relaxed_strong_close")
    merged["regime_signal_count"] = merged.apply(
        lambda r: r["current_signal_count"] if r["selected_config"] == "current_config" else r["relaxed_signal_count"],
        axis=1,
    )
    merged["regime_alpha_30d"] = merged.apply(
        lambda r: r["current_alpha_30d"] if r["selected_config"] == "current_config" else r["relaxed_alpha_30d"],
        axis=1,
    )
    merged["regime_net_alpha_30d"] = merged["regime_alpha_30d"] - 0.30
    merged["winner_config"] = merged.apply(
        lambda r: "current_config"
        if pd.isna(r["relaxed_alpha_30d"]) or (pd.notna(r["current_alpha_30d"]) and r["current_alpha_30d"] > r["relaxed_alpha_30d"])
        else ("relaxed_strong_close" if pd.notna(r["relaxed_alpha_30d"]) and r["relaxed_alpha_30d"] > r["current_alpha_30d"] else "Same"),
        axis=1,
    )
    merged["regime_minus_current"] = merged["regime_alpha_30d"] - merged["current_alpha_30d"]
    merged["regime_minus_relaxed"] = merged["regime_alpha_30d"] - merged["relaxed_alpha_30d"]
    merged["same_alpha_allclose"] = (merged["current_alpha_30d"] - merged["relaxed_alpha_30d"]).abs() < 1e-12
    merged["same_signal_set"] = merged["current_signature"] == merged["relaxed_signature"]
    merged["same_alpha_diagnosis"] = merged.apply(
        lambda r: "same_signal_set"
        if bool(r["same_alpha_allclose"]) and bool(r["same_signal_set"])
        else ("different_signal_set_or_calc_issue" if bool(r["same_alpha_allclose"]) else "not_equal_alpha"),
        axis=1,
    )

    keep_cols = [
        "period",
        "regime_label",
        "xu100_return_20d_pct",
        "selected_config",
        "current_signal_count",
        "relaxed_signal_count",
        "regime_signal_count",
        "current_alpha_30d",
        "relaxed_alpha_30d",
        "regime_alpha_30d",
        "current_net_alpha_30d",
        "relaxed_net_alpha_30d",
        "regime_net_alpha_30d",
        "winner_config",
        "regime_minus_current",
        "regime_minus_relaxed",
        "same_alpha_allclose",
        "same_signal_set",
        "same_alpha_diagnosis",
    ]
    out = merged[keep_cols].sort_values("period").reset_index(drop=True)

    regime_net = pd.to_numeric(out["regime_net_alpha_30d"], errors="coerce").dropna()
    regime_minus_current = pd.to_numeric(out["regime_minus_current"], errors="coerce").dropna()
    regime_minus_relaxed = pd.to_numeric(out["regime_minus_relaxed"], errors="coerce").dropna()
    worst_idx = regime_net.idxmin() if not regime_net.empty else None
    all_same_alpha = bool(out["same_alpha_allclose"].all()) if not out.empty else False
    same_signal_when_equal = bool(out.loc[out["same_alpha_allclose"], "same_signal_set"].all()) if out["same_alpha_allclose"].any() else False

    summary: dict[str, Any] = {
        "period_count": int(out.shape[0]),
        "avg_net_alpha_30d": float(regime_net.mean()) if not regime_net.empty else None,
        "median_net_alpha_30d": float(regime_net.median()) if not regime_net.empty else None,
        "positive_period_count": int((regime_net > 0).sum()) if not regime_net.empty else 0,
        "beat_current_month_count": int((regime_minus_current > 0).sum()) if not regime_minus_current.empty else 0,
        "beat_relaxed_month_count": int((regime_minus_relaxed > 0).sum()) if not regime_minus_relaxed.empty else 0,
        "worst_month": (
            {
                "period": str(out.loc[worst_idx, "period"]),
                "regime_net_alpha_30d": float(out.loc[worst_idx, "regime_net_alpha_30d"]),
            }
            if worst_idx is not None
            else None
        ),
        "signal_count_total": int(pd.to_numeric(out["regime_signal_count"], errors="coerce").fillna(0).sum()),
        "all_periods_current_relaxed_same_alpha": all_same_alpha,
        "same_alpha_root_cause": (
            "same_signal_set" if all_same_alpha and same_signal_when_equal else ("possible_bug_or_calc_issue" if all_same_alpha else "not_applicable")
        ),
        "production_recommendation": (
            "regime_adaptive"
            if float(regime_minus_current.mean()) > 0 and float(regime_minus_relaxed.mean()) > 0
            else ("relaxed_strong_close" if float(regime_minus_relaxed.mean()) <= 0 and float(regime_minus_current.mean()) <= 0 else "conditional")
        ),
    }
    return out, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    comparison, summary = build_regime_comparison(output_dir, args.db_path, args.start_date, args.end_date)
    comp_path = output_dir / "regime_config_comparison.csv"
    summary_path = output_dir / "regime_config_comparison_summary.json"
    comparison.to_csv(comp_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print({"comparison_csv": str(comp_path), "summary_json": str(summary_path), "summary": summary})


if __name__ == "__main__":
    main()

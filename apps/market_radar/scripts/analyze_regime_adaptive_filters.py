from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_DB_PATH = "/data/market_radar_cache.sqlite"
DEFAULT_THRESHOLDS = [0.0, 1.0, 1.5, 2.0, 3.0, 5.0]
TRAIN_START = "2024-01-01"
TRAIN_END = "2025-06-01"
TEST_START = "2025-07-01"
TEST_END = "2026-05-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-05-08")
    parser.add_argument("--transaction-cost-pct", type=float, default=0.0)
    parser.add_argument("--thresholds", default="0.0,1.0,1.5,2.0,3.0,5.0")
    return parser.parse_args()


def _parse_thresholds(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or list(DEFAULT_THRESHOLDS)


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


def _load_current_summary(output_dir: Path, periods: list[str], transaction_cost_pct: float) -> pd.DataFrame:
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
    current["current_net_alpha_30d"] = current["current_alpha_30d"] - float(transaction_cost_pct)
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


def _build_relaxed_from_candidates(output_dir: Path, periods: list[str], transaction_cost_pct: float) -> pd.DataFrame:
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
    grouped["relaxed_net_alpha_30d"] = grouped["relaxed_alpha_30d"] - float(transaction_cost_pct)
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


def build_regime_comparison(
    output_dir: Path, db_path: str, start: str, end: str, transaction_cost_pct: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    periods = _load_periods(output_dir, start, end)
    current = _load_current_summary(output_dir, periods, transaction_cost_pct)
    relaxed = _build_relaxed_from_candidates(output_dir, periods, transaction_cost_pct)
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
    merged["regime_net_alpha_30d"] = merged["regime_alpha_30d"] - float(transaction_cost_pct)
    merged["inverse_selected_config"] = merged["regime_label"].map(
        lambda x: "relaxed_strong_close" if x == "Bull" else "current_config"
    )
    merged["inverse_signal_count"] = merged.apply(
        lambda r: r["current_signal_count"] if r["inverse_selected_config"] == "current_config" else r["relaxed_signal_count"],
        axis=1,
    )
    merged["inverse_alpha_30d"] = merged.apply(
        lambda r: r["current_alpha_30d"] if r["inverse_selected_config"] == "current_config" else r["relaxed_alpha_30d"],
        axis=1,
    )
    merged["inverse_net_alpha_30d"] = merged["inverse_alpha_30d"] - float(transaction_cost_pct)
    merged["transaction_cost_pct"] = float(transaction_cost_pct)
    merged["winner_config"] = merged.apply(
        lambda r: "current_config"
        if pd.isna(r["relaxed_alpha_30d"]) or (pd.notna(r["current_alpha_30d"]) and r["current_alpha_30d"] > r["relaxed_alpha_30d"])
        else ("relaxed_strong_close" if pd.notna(r["relaxed_alpha_30d"]) and r["relaxed_alpha_30d"] > r["current_alpha_30d"] else "Same"),
        axis=1,
    )
    merged["regime_minus_current"] = merged["regime_alpha_30d"] - merged["current_alpha_30d"]
    merged["regime_minus_relaxed"] = merged["regime_alpha_30d"] - merged["relaxed_alpha_30d"]
    merged["regime_minus_inverse"] = merged["regime_alpha_30d"] - merged["inverse_alpha_30d"]
    merged["winner_among_4"] = merged.apply(
        lambda r: _winner_among_four(
            current_alpha=r["current_alpha_30d"],
            relaxed_alpha=r["relaxed_alpha_30d"],
            regime_alpha=r["regime_alpha_30d"],
            inverse_alpha=r["inverse_alpha_30d"],
        ),
        axis=1,
    )
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
        "transaction_cost_pct",
        "selected_config",
        "current_signal_count",
        "relaxed_signal_count",
        "regime_signal_count",
        "inverse_selected_config",
        "inverse_signal_count",
        "current_alpha_30d",
        "relaxed_alpha_30d",
        "regime_alpha_30d",
        "inverse_alpha_30d",
        "current_net_alpha_30d",
        "relaxed_net_alpha_30d",
        "regime_net_alpha_30d",
        "inverse_net_alpha_30d",
        "winner_config",
        "winner_among_4",
        "regime_minus_current",
        "regime_minus_relaxed",
        "regime_minus_inverse",
        "same_alpha_allclose",
        "same_signal_set",
        "same_alpha_diagnosis",
    ]
    out = merged[keep_cols].sort_values("period").reset_index(drop=True)

    regime_net = pd.to_numeric(out["regime_net_alpha_30d"], errors="coerce").dropna()
    regime_minus_current = pd.to_numeric(out["regime_minus_current"], errors="coerce").dropna()
    regime_minus_relaxed = pd.to_numeric(out["regime_minus_relaxed"], errors="coerce").dropna()
    regime_minus_inverse = pd.to_numeric(out["regime_minus_inverse"], errors="coerce").dropna()
    current_net = pd.to_numeric(out["current_net_alpha_30d"], errors="coerce").dropna()
    relaxed_net = pd.to_numeric(out["relaxed_net_alpha_30d"], errors="coerce").dropna()
    inverse_net = pd.to_numeric(out["inverse_net_alpha_30d"], errors="coerce").dropna()
    worst_idx = regime_net.idxmin() if not regime_net.empty else None
    all_same_alpha = bool(out["same_alpha_allclose"].all()) if not out.empty else False
    same_signal_when_equal = bool(out.loc[out["same_alpha_allclose"], "same_signal_set"].all()) if out["same_alpha_allclose"].any() else False

    current_gross = pd.to_numeric(out["current_alpha_30d"], errors="coerce").dropna()
    relaxed_gross = pd.to_numeric(out["relaxed_alpha_30d"], errors="coerce").dropna()
    regime_gross = pd.to_numeric(out["regime_alpha_30d"], errors="coerce").dropna()
    inverse_gross = pd.to_numeric(out["inverse_alpha_30d"], errors="coerce").dropna()
    current_avg = float(current_net.mean()) if not current_net.empty else None
    relaxed_avg = float(relaxed_net.mean()) if not relaxed_net.empty else None
    regime_avg = float(regime_net.mean()) if not regime_net.empty else None
    inverse_avg = float(inverse_net.mean()) if not inverse_net.empty else None
    current_median = float(current_net.median()) if not current_net.empty else None
    relaxed_median = float(relaxed_net.median()) if not relaxed_net.empty else None
    regime_median = float(regime_net.median()) if not regime_net.empty else None
    inverse_median = float(inverse_net.median()) if not inverse_net.empty else None
    positive_period_count = int((regime_net > 0).sum()) if not regime_net.empty else 0
    winner_distribution = out["winner_among_4"].value_counts(dropna=False).to_dict()
    current_worst_idx = pd.to_numeric(out["current_alpha_30d"], errors="coerce").idxmin()
    relaxed_worst_idx = pd.to_numeric(out["relaxed_alpha_30d"], errors="coerce").idxmin()
    regime_worst_idx = pd.to_numeric(out["regime_alpha_30d"], errors="coerce").idxmin()
    inverse_worst_idx = pd.to_numeric(out["inverse_alpha_30d"], errors="coerce").idxmin()
    recommendation = _production_recommendation(
        current_avg=current_avg,
        relaxed_avg=relaxed_avg,
        regime_avg=regime_avg,
        inverse_avg=inverse_avg,
        current_median=current_median,
        relaxed_median=relaxed_median,
        regime_median=regime_median,
        inverse_median=inverse_median,
        positive_period_count=positive_period_count,
    )

    summary: dict[str, Any] = {
        "period_count": int(out.shape[0]),
        "transaction_cost_pct": float(transaction_cost_pct),
        "current_avg_gross_alpha_30d": float(current_gross.mean()) if not current_gross.empty else None,
        "relaxed_avg_gross_alpha_30d": float(relaxed_gross.mean()) if not relaxed_gross.empty else None,
        "regime_avg_gross_alpha_30d": float(regime_gross.mean()) if not regime_gross.empty else None,
        "inverse_avg_gross_alpha_30d": float(inverse_gross.mean()) if not inverse_gross.empty else None,
        "current_avg_alpha_30d": float(current_gross.mean()) if not current_gross.empty else None,
        "relaxed_avg_alpha_30d": float(relaxed_gross.mean()) if not relaxed_gross.empty else None,
        "regime_avg_alpha_30d": float(regime_gross.mean()) if not regime_gross.empty else None,
        "inverse_avg_alpha_30d": float(inverse_gross.mean()) if not inverse_gross.empty else None,
        "avg_net_alpha_30d": float(regime_net.mean()) if not regime_net.empty else None,
        "median_net_alpha_30d": float(regime_net.median()) if not regime_net.empty else None,
        "positive_period_count": positive_period_count,
        "beat_current_month_count": int((regime_minus_current > 0).sum()) if not regime_minus_current.empty else 0,
        "beat_relaxed_month_count": int((regime_minus_relaxed > 0).sum()) if not regime_minus_relaxed.empty else 0,
        "current_avg_net_alpha_30d": current_avg,
        "relaxed_avg_net_alpha_30d": relaxed_avg,
        "regime_avg_net_alpha_30d": regime_avg,
        "inverse_avg_net_alpha_30d": inverse_avg,
        "current_median_net_alpha_30d": current_median,
        "relaxed_median_net_alpha_30d": relaxed_median,
        "regime_median_net_alpha_30d": regime_median,
        "inverse_median_net_alpha_30d": inverse_median,
        "current_median_alpha_30d": float(current_gross.median()) if not current_gross.empty else None,
        "relaxed_median_alpha_30d": float(relaxed_gross.median()) if not relaxed_gross.empty else None,
        "regime_median_alpha_30d": float(regime_gross.median()) if not regime_gross.empty else None,
        "inverse_median_alpha_30d": float(inverse_gross.median()) if not inverse_gross.empty else None,
        "regime_beat_current_count": int((regime_minus_current > 0).sum()) if not regime_minus_current.empty else 0,
        "regime_beat_relaxed_count": int((regime_minus_relaxed > 0).sum()) if not regime_minus_relaxed.empty else 0,
        "regime_beat_inverse_count": int((regime_minus_inverse > 0).sum()) if not regime_minus_inverse.empty else 0,
        "inverse_beat_regime_count": int((regime_minus_inverse < 0).sum()) if not regime_minus_inverse.empty else 0,
        "winner_distribution": winner_distribution,
        "worst_month_current_config": (
            {"period": str(out.loc[current_worst_idx, "period"]), "alpha_30d": float(out.loc[current_worst_idx, "current_alpha_30d"])}
            if pd.notna(current_worst_idx)
            else None
        ),
        "worst_month_relaxed_strong_close": (
            {"period": str(out.loc[relaxed_worst_idx, "period"]), "alpha_30d": float(out.loc[relaxed_worst_idx, "relaxed_alpha_30d"])}
            if pd.notna(relaxed_worst_idx)
            else None
        ),
        "worst_month_regime_adaptive": (
            {"period": str(out.loc[regime_worst_idx, "period"]), "alpha_30d": float(out.loc[regime_worst_idx, "regime_alpha_30d"])}
            if pd.notna(regime_worst_idx)
            else None
        ),
        "worst_month_inverse_regime": (
            {"period": str(out.loc[inverse_worst_idx, "period"]), "alpha_30d": float(out.loc[inverse_worst_idx, "inverse_alpha_30d"])}
            if pd.notna(inverse_worst_idx)
            else None
        ),
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
        "production_recommendation": recommendation,
    }
    return out, summary


def _winner_among_four(
    current_alpha: Any, relaxed_alpha: Any, regime_alpha: Any, inverse_alpha: Any
) -> str:
    candidates = {
        "current_config": pd.to_numeric(pd.Series([current_alpha]), errors="coerce").iloc[0],
        "relaxed_strong_close": pd.to_numeric(pd.Series([relaxed_alpha]), errors="coerce").iloc[0],
        "regime_adaptive": pd.to_numeric(pd.Series([regime_alpha]), errors="coerce").iloc[0],
        "inverse_regime": pd.to_numeric(pd.Series([inverse_alpha]), errors="coerce").iloc[0],
    }
    valid = {k: v for k, v in candidates.items() if pd.notna(v)}
    if not valid:
        return "None"
    max_v = max(valid.values())
    winners = sorted([k for k, v in valid.items() if v == max_v])
    if len(winners) == 1:
        return winners[0]
    return "Tie:" + "|".join(winners)


def _production_recommendation(
    *,
    current_avg: float | None,
    relaxed_avg: float | None,
    regime_avg: float | None,
    inverse_avg: float | None,
    current_median: float | None,
    relaxed_median: float | None,
    regime_median: float | None,
    inverse_median: float | None,
    positive_period_count: int,
) -> str:
    if (
        regime_avg is not None
        and current_avg is not None
        and relaxed_avg is not None
        and inverse_avg is not None
        and regime_median is not None
        and current_median is not None
        and relaxed_median is not None
        and regime_avg > current_avg
        and regime_avg > relaxed_avg
        and regime_avg > inverse_avg
        and regime_median >= current_median
        and regime_median >= relaxed_median
        and positive_period_count >= 18
        and inverse_avg <= regime_avg
    ):
        return "regime_adaptive"
    avg_map = {
        "current_config": current_avg,
        "relaxed_strong_close": relaxed_avg,
        "regime_adaptive": regime_avg,
        "inverse_regime": inverse_avg,
    }
    valid = {k: v for k, v in avg_map.items() if v is not None}
    if not valid:
        return "needs_more_validation"
    best = sorted(valid.items(), key=lambda x: x[1], reverse=True)[0][0]
    if best == "regime_adaptive":
        if inverse_median is not None and regime_median is not None and inverse_median > regime_median:
            return "needs_more_validation"
        return "needs_more_validation"
    return best


def build_threshold_robustness(comparison: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = comparison.copy()
    base = _to_numeric(
        base,
        [
            "xu100_return_20d_pct",
            "current_alpha_30d",
            "relaxed_alpha_30d",
            "current_signal_count",
            "relaxed_signal_count",
        ],
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        chosen_current = base["xu100_return_20d_pct"] > float(threshold)
        alpha = base["current_alpha_30d"].where(chosen_current, base["relaxed_alpha_30d"])
        signals = base["current_signal_count"].where(chosen_current, base["relaxed_signal_count"])
        delta_current = alpha - base["current_alpha_30d"]
        delta_relaxed = alpha - base["relaxed_alpha_30d"]
        worst_idx = alpha.idxmin() if alpha.notna().any() else None
        rows.append(
            {
                "threshold": float(threshold),
                "avg_alpha_30d": float(alpha.mean()) if alpha.notna().any() else None,
                "median_alpha_30d": float(alpha.median()) if alpha.notna().any() else None,
                "positive_period_count": int((alpha > 0).sum()),
                "beat_current_count": int((delta_current > 0).sum()),
                "beat_relaxed_count": int((delta_relaxed > 0).sum()),
                "worst_month": str(base.loc[worst_idx, "period"]) if worst_idx is not None else None,
                "worst_month_alpha_30d": float(alpha.loc[worst_idx]) if worst_idx is not None else None,
                "selected_current_month_count": int(chosen_current.sum()),
                "selected_relaxed_month_count": int((~chosen_current).sum()),
                "signal_count_total": int(pd.to_numeric(signals, errors="coerce").fillna(0).sum()),
            }
        )
    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_idx = df["avg_alpha_30d"].idxmax() if not df.empty else None
    summary = {
        "thresholds": thresholds,
        "best_by_avg_alpha_30d": (df.loc[best_idx].to_dict() if best_idx is not None else None),
        "rows": df.to_dict("records"),
    }
    return df, summary


def _series_stats(series: pd.Series) -> tuple[float | None, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.mean()), float(s.median())


def build_oos_validation(comparison: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = comparison.copy()
    base = _to_numeric(
        base,
        [
            "xu100_return_20d_pct",
            "current_alpha_30d",
            "relaxed_alpha_30d",
            "current_signal_count",
            "relaxed_signal_count",
        ],
    )
    train = base.loc[(base["period"] >= TRAIN_START) & (base["period"] <= TRAIN_END)].copy()
    test = base.loc[(base["period"] >= TEST_START) & (base["period"] <= TEST_END)].copy()

    train_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        chosen_current = train["xu100_return_20d_pct"] > float(threshold)
        train_alpha = train["current_alpha_30d"].where(chosen_current, train["relaxed_alpha_30d"])
        train_rows.append(
            {
                "threshold": float(threshold),
                "train_avg_alpha_30d": float(train_alpha.mean()) if train_alpha.notna().any() else None,
            }
        )
    train_df = pd.DataFrame(train_rows)
    best_idx = train_df["train_avg_alpha_30d"].idxmax() if not train_df.empty else None
    selected_threshold = float(train_df.loc[best_idx, "threshold"]) if best_idx is not None else float(thresholds[0])
    train_avg_alpha_30d = float(train_df.loc[best_idx, "train_avg_alpha_30d"]) if best_idx is not None else None

    chosen_current_test = test["xu100_return_20d_pct"] > selected_threshold
    test_regime_alpha = test["current_alpha_30d"].where(chosen_current_test, test["relaxed_alpha_30d"])
    test_inverse_alpha = test["relaxed_alpha_30d"].where(chosen_current_test, test["current_alpha_30d"])
    test["selected_threshold"] = selected_threshold
    test["selected_config"] = chosen_current_test.map({True: "current_config", False: "relaxed_strong_close"})
    test["inverse_selected_config"] = chosen_current_test.map({True: "relaxed_strong_close", False: "current_config"})
    test["test_current_alpha_30d"] = test["current_alpha_30d"]
    test["test_relaxed_alpha_30d"] = test["relaxed_alpha_30d"]
    test["test_regime_alpha_30d"] = test_regime_alpha
    test["test_inverse_alpha_30d"] = test_inverse_alpha
    test["test_regime_signal_count"] = test["current_signal_count"].where(chosen_current_test, test["relaxed_signal_count"])
    test["test_inverse_signal_count"] = test["relaxed_signal_count"].where(chosen_current_test, test["current_signal_count"])

    cur_avg, cur_med = _series_stats(test["test_current_alpha_30d"])
    rel_avg, rel_med = _series_stats(test["test_relaxed_alpha_30d"])
    reg_avg, reg_med = _series_stats(test["test_regime_alpha_30d"])
    inv_avg, inv_med = _series_stats(test["test_inverse_alpha_30d"])
    regime_minus_current = test["test_regime_alpha_30d"] - test["test_current_alpha_30d"]
    regime_minus_relaxed = test["test_regime_alpha_30d"] - test["test_relaxed_alpha_30d"]
    regime_minus_inverse = test["test_regime_alpha_30d"] - test["test_inverse_alpha_30d"]
    test_period_count = int(test.shape[0])
    test_regime_beat_relaxed_count = int((regime_minus_relaxed > 0).sum())
    test_positive_period_count = int((pd.to_numeric(test["test_regime_alpha_30d"], errors="coerce") > 0).sum())

    if (
        reg_avg is not None
        and rel_avg is not None
        and reg_avg > rel_avg
        and reg_med is not None
        and rel_med is not None
        and reg_med >= rel_med
        and test_regime_beat_relaxed_count >= (test_period_count / 2.0)
    ):
        recommendation = "use_regime_adaptive"
    else:
        recommendation = "keep_relaxed_as_default_use_regime_as_diagnostic"

    summary = {
        "selected_threshold": selected_threshold,
        "train_avg_alpha_30d": train_avg_alpha_30d,
        "test_period_count": test_period_count,
        "test_current_avg_alpha_30d": cur_avg,
        "test_relaxed_avg_alpha_30d": rel_avg,
        "test_regime_avg_alpha_30d": reg_avg,
        "test_inverse_avg_alpha_30d": inv_avg,
        "test_current_median_alpha_30d": cur_med,
        "test_relaxed_median_alpha_30d": rel_med,
        "test_regime_median_alpha_30d": reg_med,
        "test_inverse_median_alpha_30d": inv_med,
        "test_regime_beat_current_count": int((regime_minus_current > 0).sum()),
        "test_regime_beat_relaxed_count": test_regime_beat_relaxed_count,
        "test_regime_beat_inverse_count": int((regime_minus_inverse > 0).sum()),
        "test_positive_period_count": test_positive_period_count,
        "recommendation": recommendation,
        "production_recommendation": recommendation,
    }
    oos_cols = [
        "period",
        "selected_threshold",
        "xu100_return_20d_pct",
        "selected_config",
        "inverse_selected_config",
        "test_current_alpha_30d",
        "test_relaxed_alpha_30d",
        "test_regime_alpha_30d",
        "test_inverse_alpha_30d",
        "current_signal_count",
        "relaxed_signal_count",
        "test_regime_signal_count",
        "test_inverse_signal_count",
    ]
    return test[oos_cols].reset_index(drop=True), summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    thresholds = _parse_thresholds(args.thresholds)
    comparison, summary = build_regime_comparison(
        output_dir, args.db_path, args.start_date, args.end_date, float(args.transaction_cost_pct)
    )
    comp_path = output_dir / "regime_config_comparison.csv"
    summary_path = output_dir / "regime_config_comparison_summary.json"
    robustness_csv = output_dir / "regime_threshold_robustness.csv"
    robustness_json = output_dir / "regime_threshold_robustness_summary.json"
    oos_csv = output_dir / "regime_oos_validation.csv"
    oos_json = output_dir / "regime_oos_validation_summary.json"
    robustness_df, robustness_summary = build_threshold_robustness(comparison, thresholds)
    oos_df, oos_summary = build_oos_validation(comparison, thresholds)
    comparison.to_csv(comp_path, index=False)
    robustness_df.to_csv(robustness_csv, index=False)
    oos_df.to_csv(oos_csv, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with robustness_json.open("w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)
    with oos_json.open("w", encoding="utf-8") as f:
        json.dump(oos_summary, f, indent=2)
    print(
        {
            "comparison_csv": str(comp_path),
            "summary_json": str(summary_path),
            "robustness_csv": str(robustness_csv),
            "robustness_summary_json": str(robustness_json),
            "oos_csv": str(oos_csv),
            "oos_summary_json": str(oos_json),
            "summary": summary,
        }
    )


if __name__ == "__main__":
    main()

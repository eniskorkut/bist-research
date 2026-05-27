from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
TOP_N_VALUES = [20, 30, 50, 75, -1]
SCORE_NAMES = ["balanced_score", "momentum_quality_score", "liquidity_safe_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-file")
    return parser.parse_args()


def _read_features(output_dir: Path, feature_file: str | None) -> pd.DataFrame:
    if feature_file:
        path = Path(feature_file)
    else:
        pq = output_dir / "candidate_features.parquet"
        csv = output_dir / "candidate_features.csv"
        path = pq if pq.exists() else csv
    if not path.exists():
        raise FileNotFoundError(f"candidate feature file missing: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period"] = out["period"].astype(str)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["entry_date"] = pd.to_datetime(out.get("entry_date", out["signal_date"]), errors="coerce")
    out["above_ma20"] = _to_bool(out.get("above_ma20", pd.Series(False, index=out.index)))
    num_cols = [
        "volume_ratio_20d",
        "turnover_ratio_20d",
        "turnover",
        "avg_turnover_20d",
        "close_position",
        "close",
        "ma20",
        "rsi_14",
        "return_5d_pct",
        "return_10d_pct",
        "future_return_15d",
        "future_return_30d",
        "benchmark_return_15d",
        "benchmark_return_30d",
        "alpha_30d",
    ]
    out = _to_numeric(out, num_cols)
    return out


def _relaxed_mask(df: pd.DataFrame) -> pd.Series:
    ma_ok = df["above_ma20"] | df["close"].ge(df["ma20"])
    mask = (
        df["turnover"].ge(10_000_000.0)
        & df["avg_turnover_20d"].ge(10_000_000.0)
        & ma_ok.fillna(False)
        & df["rsi_14"].le(78.0)
        & df["return_5d_pct"].le(35.0)
        & df["return_10d_pct"].le(60.0)
        & df["close_position"].ge(0.50)
    )
    return mask.fillna(False)


def _norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    min_v = x.min()
    max_v = x.max()
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(0.0, index=s.index)
    if float(max_v) - float(min_v) < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (x - min_v) / (max_v - min_v)


def _regime_map(output_dir: Path) -> pd.DataFrame:
    p = output_dir / "regime_config_comparison.csv"
    if not p.exists():
        return pd.DataFrame(columns=["period", "xu100_return_20d_pct"])
    df = pd.read_csv(p)
    if "period" not in df.columns or "xu100_return_20d_pct" not in df.columns:
        return pd.DataFrame(columns=["period", "xu100_return_20d_pct"])
    df["period"] = df["period"].astype(str)
    df["xu100_return_20d_pct"] = pd.to_numeric(df["xu100_return_20d_pct"], errors="coerce")
    return df[["period", "xu100_return_20d_pct"]].drop_duplicates(subset=["period"], keep="first")


def _add_scores(df: pd.DataFrame, regime_bonus_enabled: bool) -> pd.DataFrame:
    out = df.copy()
    chunks: list[pd.DataFrame] = []
    for period, g in out.groupby("period", sort=False):
        gg = g.copy()
        vol = 0.5 * _norm(gg["volume_ratio_20d"]) + 0.5 * _norm(gg["turnover_ratio_20d"])
        liq = 0.6 * _norm(gg["avg_turnover_20d"]) + 0.4 * _norm(gg["turnover"])
        close_pos = gg["close_position"].clip(lower=0.0, upper=1.0).fillna(0.0)
        trend_raw = (gg["close"] / gg["ma20"]).clip(lower=0.90, upper=1.10)
        trend = ((trend_raw - 0.90) / 0.20).fillna(0.0)
        rsi = gg["rsi_14"]
        rsi_safety = (1.0 - ((rsi - 62.0).abs() / 20.0)).clip(lower=0.0, upper=1.0).fillna(0.0)
        overext = ((gg["return_5d_pct"] - 12.0).clip(lower=0.0) / 25.0) + (
            (gg["return_10d_pct"] - 20.0).clip(lower=0.0) / 40.0
        )
        overext = overext.clip(lower=0.0, upper=1.0).fillna(0.0)

        regime_bonus = pd.Series(0.0, index=gg.index)
        if regime_bonus_enabled:
            ru = float(gg["xu100_return_20d_pct"].iloc[0]) if pd.notna(gg["xu100_return_20d_pct"].iloc[0]) else 0.0
            regime_bonus = pd.Series(0.05 if ru > 1.0 else 0.0, index=gg.index)

        gg["balanced_score"] = (
            0.26 * vol + 0.20 * liq + 0.16 * close_pos + 0.18 * trend + 0.15 * rsi_safety - 0.15 * overext + regime_bonus
        )
        gg["momentum_quality_score"] = (
            0.32 * vol + 0.12 * liq + 0.20 * close_pos + 0.24 * trend + 0.12 * rsi_safety - 0.20 * overext + regime_bonus
        )
        gg["liquidity_safe_score"] = (
            0.14 * vol + 0.34 * liq + 0.14 * close_pos + 0.15 * trend + 0.23 * rsi_safety - 0.16 * overext + regime_bonus
        )
        chunks.append(gg)
    return pd.concat(chunks, ignore_index=True) if chunks else out


def _baseline_relaxed(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, g in df.groupby("period", sort=True):
        b = g.sort_values(["signal_date", "entry_date", "symbol"]).drop_duplicates(subset=["symbol"], keep="first")
        ret15 = pd.to_numeric(b["future_return_15d"], errors="coerce").dropna()
        ret30 = pd.to_numeric(b["future_return_30d"], errors="coerce").dropna()
        br15 = pd.to_numeric(b["benchmark_return_15d"], errors="coerce").dropna()
        br30 = pd.to_numeric(b["benchmark_return_30d"], errors="coerce").dropna()
        r15 = float(ret15.mean()) if not ret15.empty else None
        r30 = float(ret30.mean()) if not ret30.empty else None
        b15 = float(br15.mean()) if not br15.empty else None
        b30 = float(br30.mean()) if not br30.empty else None
        alpha30 = None if r30 is None or b30 is None else r30 - b30
        rows.append(
            {
                "period": period,
                "baseline_relaxed_alpha_30d": alpha30,
                "baseline_relaxed_return_15d": r15,
                "baseline_relaxed_return_30d": r30,
                "baseline_selected_count": int(len(b)),
            }
        )
    return pd.DataFrame(rows)


def _topn_name(v: int) -> str:
    return "all" if v < 0 else f"top{v}"


def _evaluate(df: pd.DataFrame, baseline: pd.DataFrame, score_name: str, top_n: int, regime_bonus_enabled: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, g in df.groupby("period", sort=True):
        ordered = g.sort_values([score_name, "signal_date", "entry_date", "symbol"], ascending=[False, True, True, True])
        picked = ordered if top_n < 0 else ordered.head(top_n)
        picked = picked.drop_duplicates(subset=["symbol"], keep="first")
        ret15 = pd.to_numeric(picked["future_return_15d"], errors="coerce").dropna()
        ret30 = pd.to_numeric(picked["future_return_30d"], errors="coerce").dropna()
        br15 = pd.to_numeric(picked["benchmark_return_15d"], errors="coerce").dropna()
        br30 = pd.to_numeric(picked["benchmark_return_30d"], errors="coerce").dropna()
        b15 = float(ret15.mean()) if not ret15.empty else None
        b30 = float(ret30.mean()) if not ret30.empty else None
        m15 = float(br15.mean()) if not br15.empty else None
        m30 = float(br30.mean()) if not br30.empty else None
        alpha30 = None if b30 is None or m30 is None else b30 - m30
        rows.append(
            {
                "period": period,
                "score_name": score_name,
                "regime_bonus_enabled": bool(regime_bonus_enabled),
                "top_n": _topn_name(top_n),
                "selected_count": int(len(picked)),
                "basket_return_15d": b15,
                "basket_return_30d": b30,
                "basket_alpha_30d": alpha30,
            }
        )
    out = pd.DataFrame(rows).merge(baseline[["period", "baseline_relaxed_alpha_30d"]], on="period", how="left")
    out["score_minus_baseline"] = out["basket_alpha_30d"] - out["baseline_relaxed_alpha_30d"]
    out["positive_period"] = out["basket_alpha_30d"] > 0
    out["beat_baseline"] = out["basket_alpha_30d"] > out["baseline_relaxed_alpha_30d"]
    return out


def _summary_for_bucket(df: pd.DataFrame, baseline_stats: dict[str, float]) -> dict[str, Any]:
    alpha = pd.to_numeric(df["basket_alpha_30d"], errors="coerce").dropna()
    beat = df["beat_baseline"].fillna(False).astype(bool)
    sel = pd.to_numeric(df["selected_count"], errors="coerce").fillna(0)
    avg = float(alpha.mean()) if not alpha.empty else None
    med = float(alpha.median()) if not alpha.empty else None
    worst_idx = alpha.idxmin() if not alpha.empty else None
    worst_period = None
    worst_alpha = None
    if worst_idx is not None:
        worst_period = str(df.loc[worst_idx, "period"])
        worst_alpha = float(df.loc[worst_idx, "basket_alpha_30d"])
    beat_rate = float(beat.mean()) if len(beat) else 0.0
    top_n_label = str(df["top_n"].iloc[0])
    top_n_value = -1 if top_n_label == "all" else int(top_n_label.replace("top", ""))
    min_required = min(top_n_value, 10) if top_n_value > 0 else 10

    passed = bool(
        avg is not None
        and med is not None
        and worst_alpha is not None
        and avg >= baseline_stats["avg_alpha"] + 0.50
        and med >= baseline_stats["median_alpha"]
        and beat_rate >= 0.60
        and worst_alpha >= baseline_stats["worst_alpha"] - 1.0
        and float(sel.min()) >= float(min_required)
    )
    return {
        "avg_alpha_30d": avg,
        "median_alpha_30d": med,
        "positive_period_count": int((alpha > 0).sum()) if not alpha.empty else 0,
        "beat_baseline_count": int(beat.sum()),
        "beat_baseline_rate": beat_rate,
        "worst_month": worst_period,
        "worst_month_alpha_30d": worst_alpha,
        "avg_selected_count": float(sel.mean()) if len(sel) else 0.0,
        "min_selected_count": int(sel.min()) if len(sel) else 0,
        "avg_minus_median_alpha": None if avg is None or med is None else avg - med,
        "production_candidate_passed": passed,
    }


def build_topn_analysis(output_dir: Path, feature_file: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_features(output_dir, feature_file)
    f = _prepare(raw)
    f = f.loc[f["strategy"].astype(str) == "volume_spike_strict"].copy()
    regimes = _regime_map(output_dir)
    f = f.merge(regimes, on="period", how="left")
    relaxed = f.loc[_relaxed_mask(f)].copy()
    relaxed = relaxed.sort_values(["period", "signal_date", "entry_date", "symbol"]).drop_duplicates(
        subset=["period", "symbol", "signal_date"], keep="first"
    )
    baseline = _baseline_relaxed(relaxed)
    b_alpha = pd.to_numeric(baseline["baseline_relaxed_alpha_30d"], errors="coerce").dropna()
    baseline_stats = {
        "avg_alpha": float(b_alpha.mean()) if not b_alpha.empty else 0.0,
        "median_alpha": float(b_alpha.median()) if not b_alpha.empty else 0.0,
        "worst_alpha": float(b_alpha.min()) if not b_alpha.empty else 0.0,
    }

    chunks: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "baseline": {
            "avg_alpha_30d": baseline_stats["avg_alpha"],
            "median_alpha_30d": baseline_stats["median_alpha"],
            "worst_month_alpha_30d": baseline_stats["worst_alpha"],
        },
        "variants": [],
    }

    for regime_bonus_enabled in [False, True]:
        scored = _add_scores(relaxed, regime_bonus_enabled=regime_bonus_enabled)
        for score_name in SCORE_NAMES:
            for top_n in TOP_N_VALUES:
                out = _evaluate(scored, baseline, score_name=score_name, top_n=top_n, regime_bonus_enabled=regime_bonus_enabled)
                chunks.append(out)
                item = {
                    "score_name": score_name,
                    "regime_bonus_enabled": bool(regime_bonus_enabled),
                    "top_n": _topn_name(top_n),
                }
                item.update(_summary_for_bucket(out, baseline_stats))
                summary["variants"].append(item)

    full = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not full.empty:
        full["regime_bonus_enabled"] = full["regime_bonus_enabled"].astype(bool)
    best = None
    passed = [v for v in summary["variants"] if v["production_candidate_passed"]]
    if passed:
        best = sorted(
            passed,
            key=lambda x: (
                x["avg_alpha_30d"] if x["avg_alpha_30d"] is not None else -1e9,
                x["median_alpha_30d"] if x["median_alpha_30d"] is not None else -1e9,
                x["beat_baseline_rate"],
                x["avg_selected_count"],
            ),
            reverse=True,
        )[0]
    summary["recommendation"] = best or {"status": "no_production_candidate_passed"}
    return full, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_csv = output_dir / "production_score_topn_comparison.csv"
    out_json = output_dir / "production_score_topn_summary.json"
    details, summary = build_topn_analysis(output_dir=output_dir, feature_file=args.feature_file)
    details.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print({"comparison_csv": str(out_csv), "summary_json": str(out_json), "variant_count": len(summary.get("variants", []))})


if __name__ == "__main__":
    main()


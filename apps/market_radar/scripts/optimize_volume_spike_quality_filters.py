from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"

BASE_CONFIG: dict[str, Any] = {
    "min_last_turnover_try": 10_000_000.0,
    "min_avg_turnover_20d_try": 10_000_000.0,
    "max_rsi_14": 78.0,
    "max_return_5d_pct": 35.0,
    "max_return_10d_pct": 60.0,
    "require_strong_close": True,
    "min_close_position": 0.60,
    "min_above_ma20_ratio": 1.0,
}

CANDIDATES: list[dict[str, Any]] = [
    {"config_name": "current_config"},
    {"config_name": "relaxed_strong_close", "min_close_position": 0.50},
    {"config_name": "no_strong_close", "require_strong_close": False},
    {"config_name": "rsi_relaxed", "max_rsi_14": 82.0},
    {"config_name": "combined_relaxed", "min_close_position": 0.50, "max_rsi_14": 82.0},
    {"config_name": "ma20_relaxed", "min_above_ma20_ratio": 0.98},
    {
        "config_name": "combined_ma20_strong_relaxed",
        "min_above_ma20_ratio": 0.98,
        "min_close_position": 0.50,
        "max_rsi_14": 82.0,
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-file")
    parser.add_argument("--strategy", default="volume_spike_strict")
    return parser.parse_args(argv)


def _read_features(output_dir: Path, feature_file: str | None) -> pd.DataFrame:
    if feature_file:
        path = Path(feature_file)
    else:
        parquet_path = output_dir / "candidate_features.parquet"
        csv_path = output_dir / "candidate_features.csv"
        path = parquet_path if parquet_path.exists() else csv_path
    if not path.exists():
        raise FileNotFoundError(f"candidate feature file missing: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def _prepare_features(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = frame.copy()
    rename = {
        "return_15d": "future_return_15d",
        "return_30d": "future_return_30d",
        "filter_passed": "current_quality_pass",
        "failed_reasons": "current_failed_reasons",
    }
    for src, dst in rename.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    if "period" not in out.columns and "period_start" in out.columns:
        out["period"] = out["period_start"]
    if "close" not in out.columns and "entry_close" in out.columns:
        out["close"] = out["entry_close"]
    if "strategy" in out.columns:
        out = out.loc[out["strategy"] == strategy].copy()
    for col in ["period", "symbol", "signal_date"]:
        if col not in out.columns:
            raise KeyError(f"required feature column missing: {col}")
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    if "entry_date" in out.columns:
        out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    else:
        out["entry_date"] = out["signal_date"]
    out["above_ma20"] = _bool_series(out.get("above_ma20", pd.Series(False, index=out.index)))
    if "current_quality_pass" in out.columns:
        out["current_quality_pass"] = _bool_series(out["current_quality_pass"])
    numeric_cols = [
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
        "future_return_15d",
        "future_return_30d",
        "benchmark_return_15d",
        "benchmark_return_30d",
        "alpha_15d",
        "alpha_30d",
    ]
    return _numeric(out, numeric_cols)


def _merged_config(candidate: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(BASE_CONFIG)
    cfg.update({k: v for k, v in candidate.items() if k != "config_name"})
    return cfg


def _candidate_mask(features: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=features.index)
    if "volume_ratio_20d" in features.columns and "turnover_ratio_20d" in features.columns:
        mask &= features["volume_ratio_20d"].ge(1.5) & features["turnover_ratio_20d"].ge(1.2)
    mask &= features["turnover"].ge(float(config["min_last_turnover_try"]))
    mask &= features["avg_turnover_20d"].ge(float(config["min_avg_turnover_20d_try"]))

    ma20_passed = features["above_ma20"].fillna(False)
    if float(config["min_above_ma20_ratio"]) < 1.0:
        ma20_passed = ma20_passed | features["close"].ge(features["ma20"] * float(config["min_above_ma20_ratio"]))
    mask &= ma20_passed.fillna(False)

    mask &= features["rsi_14"].le(float(config["max_rsi_14"]))
    mask &= features["return_5d_pct"].le(float(config["max_return_5d_pct"]))
    mask &= features["return_10d_pct"].le(float(config["max_return_10d_pct"]))
    if bool(config["require_strong_close"]):
        mask &= features["close_position"].ge(float(config["min_close_position"]))
    return mask.fillna(False)


def _pick_basket(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ordered = rows.sort_values(["period", "signal_date", "entry_date", "symbol"]).copy()
    return ordered.drop_duplicates(subset=["period", "symbol"], keep="first")


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _monthly_rows(features: pd.DataFrame, config_name: str, config: dict[str, Any], periods: list[str]) -> pd.DataFrame:
    selected = features.loc[_candidate_mask(features, config)].copy()
    basket = _pick_basket(selected)
    rows: list[dict[str, Any]] = []
    for period in periods:
        scoped = basket.loc[basket["period"] == period].copy()
        ret15 = scoped["future_return_15d"].dropna()
        ret30 = scoped["future_return_30d"].dropna()
        b15 = scoped["benchmark_return_15d"].dropna()
        b30 = scoped["benchmark_return_30d"].dropna()
        basket_return_15d = _safe_float(ret15.mean()) if not ret15.empty else None
        basket_return_30d = _safe_float(ret30.mean()) if not ret30.empty else None
        benchmark_return_15d = _safe_float(b15.mean()) if not b15.empty else None
        benchmark_return_30d = _safe_float(b30.mean()) if not b30.empty else None
        basket_alpha_15d = (
            None
            if basket_return_15d is None or benchmark_return_15d is None
            else basket_return_15d - benchmark_return_15d
        )
        basket_alpha_30d = (
            None
            if basket_return_30d is None or benchmark_return_30d is None
            else basket_return_30d - benchmark_return_30d
        )
        rows.append(
            {
                "config_name": config_name,
                "period": period,
                "signal_count": int(len(scoped)),
                "basket_return_15d": basket_return_15d,
                "basket_return_30d": basket_return_30d,
                "benchmark_return_15d": benchmark_return_15d,
                "benchmark_return_30d": benchmark_return_30d,
                "basket_alpha_15d": basket_alpha_15d,
                "basket_alpha_30d": basket_alpha_30d,
                "positive_rate_30d": _safe_float((ret30 > 0).mean()) if not ret30.empty else None,
                "avg_alpha_30d": _safe_float(scoped["alpha_30d"].mean()) if "alpha_30d" in scoped else None,
                "median_alpha_30d": _safe_float(scoped["alpha_30d"].median()) if "alpha_30d" in scoped else None,
            }
        )
    return pd.DataFrame(rows)


def _warning_flags(summary: dict[str, Any]) -> str:
    flags: list[str] = []
    if (summary.get("min_signal_count") or 0) < 30:
        flags.append("low_monthly_signal_count")
    if summary.get("min_monthly_alpha_30d") is not None and summary["min_monthly_alpha_30d"] < -15:
        flags.append("large_negative_month")
    avg_alpha = summary.get("avg_alpha_30d")
    median_alpha = summary.get("median_alpha_30d")
    if avg_alpha is not None and median_alpha is not None and abs(avg_alpha - median_alpha) > 8:
        flags.append("avg_median_gap")
    return ",".join(flags)


def _summarize(config_name: str, monthly: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    active = monthly.loc[monthly["signal_count"] > 0].copy()
    alpha30 = pd.to_numeric(active["basket_alpha_30d"], errors="coerce").dropna()
    ret30 = pd.to_numeric(active["basket_return_30d"], errors="coerce").dropna()
    ret15 = pd.to_numeric(active["basket_return_15d"], errors="coerce").dropna()
    signal_count = pd.to_numeric(monthly["signal_count"], errors="coerce").fillna(0)
    summary = {
        "config_name": config_name,
        "period_count": int(monthly.shape[0]),
        "active_period_count": int(active.shape[0]),
        "total_signal_count": int(signal_count.sum()),
        "avg_signal_count": float(signal_count.mean()) if not signal_count.empty else 0.0,
        "min_signal_count": int(signal_count.min()) if not signal_count.empty else 0,
        "avg_alpha_30d": _safe_float(alpha30.mean()) if not alpha30.empty else None,
        "median_alpha_30d": _safe_float(alpha30.median()) if not alpha30.empty else None,
        "positive_rate_alpha_30d": _safe_float((alpha30 > 0).mean()) if not alpha30.empty else None,
        "avg_return_30d": _safe_float(ret30.mean()) if not ret30.empty else None,
        "median_return_30d": _safe_float(ret30.median()) if not ret30.empty else None,
        "positive_rate_return_30d": _safe_float((ret30 > 0).mean()) if not ret30.empty else None,
        "avg_return_15d": _safe_float(ret15.mean()) if not ret15.empty else None,
        "median_return_15d": _safe_float(ret15.median()) if not ret15.empty else None,
        "min_monthly_alpha_30d": _safe_float(alpha30.min()) if not alpha30.empty else None,
        "max_monthly_alpha_30d": _safe_float(alpha30.max()) if not alpha30.empty else None,
        "beat_month_count": int((alpha30 > 0).sum()) if not alpha30.empty else 0,
        "loss_month_count": int((alpha30 < 0).sum()) if not alpha30.empty else 0,
        "valid_30d_period_count": int(alpha30.shape[0]),
        "min_last_turnover_try": float(config["min_last_turnover_try"]),
        "min_avg_turnover_20d_try": float(config["min_avg_turnover_20d_try"]),
        "max_rsi_14": float(config["max_rsi_14"]),
        "max_return_5d_pct": float(config["max_return_5d_pct"]),
        "max_return_10d_pct": float(config["max_return_10d_pct"]),
        "require_strong_close": bool(config["require_strong_close"]),
        "min_close_position": float(config["min_close_position"]),
        "min_above_ma20_ratio": float(config["min_above_ma20_ratio"]),
    }
    summary["warning_flags"] = _warning_flags(summary)
    return summary


def _add_current_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    current = out.loc[out["config_name"] == "current_config"]
    if current.empty:
        out["delta_vs_current_alpha_30d"] = pd.NA
        out["delta_vs_current_signal_count"] = pd.NA
        return out
    current_alpha = pd.to_numeric(current.iloc[0].get("median_alpha_30d"), errors="coerce")
    current_signals = pd.to_numeric(current.iloc[0].get("total_signal_count"), errors="coerce")
    out["delta_vs_current_alpha_30d"] = pd.to_numeric(out["median_alpha_30d"], errors="coerce") - current_alpha
    out["delta_vs_current_signal_count"] = pd.to_numeric(out["total_signal_count"], errors="coerce") - current_signals
    return out


def _score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out = out.sort_values(
        by=[
            "median_alpha_30d",
            "positive_rate_alpha_30d",
            "avg_alpha_30d",
            "avg_signal_count",
            "avg_return_15d",
            "min_monthly_alpha_30d",
        ],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["score_rank"] = range(1, len(out) + 1)
    return out


def _match_check(output_dir: Path, monthly: pd.DataFrame) -> dict[str, Any]:
    path = output_dir / "period_strategy_summary.csv"
    if not path.exists():
        return {"status": "reference_missing"}
    reference = pd.read_csv(path)
    if "strategy" in reference.columns:
        reference = reference.loc[reference["strategy"] == "volume_spike_strict"].copy()
    if "quality_filter_enabled" in reference.columns:
        reference = reference.loc[reference["quality_filter_enabled"].astype(str).str.lower() == "true"].copy()
    offline = monthly.loc[monthly["config_name"] == "current_config"].rename(columns={"period": "period_start"})
    merged = reference.merge(offline, on="period_start", suffixes=("_reference", "_offline"))
    if merged.empty:
        return {"status": "no_overlap"}
    checks: dict[str, Any] = {"status": "ok", "matched_period_count": int(merged.shape[0])}
    for col in ["signal_count", "basket_return_30d", "basket_alpha_30d", "basket_return_15d"]:
        ref_col = f"{col}_reference"
        off_col = f"{col}_offline"
        if ref_col not in merged or off_col not in merged:
            continue
        diff = pd.to_numeric(merged[off_col], errors="coerce") - pd.to_numeric(merged[ref_col], errors="coerce")
        checks[f"max_abs_delta_{col}"] = _safe_float(diff.abs().max())
        checks[f"avg_delta_{col}"] = _safe_float(diff.mean())
    return checks


def build_outputs(features: pd.DataFrame, output_dir: Path, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared = _prepare_features(features, strategy)
    periods = sorted(prepared["period"].dropna().astype(str).unique().tolist())
    monthly_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        config_name = str(candidate["config_name"])
        config = _merged_config(candidate)
        monthly = _monthly_rows(prepared, config_name, config, periods)
        monthly_frames.append(monthly)
        summary_rows.append(_summarize(config_name, monthly, config))

    monthly_details = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    summary = _score(_add_current_deltas(pd.DataFrame(summary_rows)))
    best_config = summary.iloc[0].to_dict() if not summary.empty else {}
    metadata = {
        "objective_metric": "median_basket_alpha_30d",
        "secondary_metric": "avg_basket_alpha_30d",
        "tie_breaker_metric": "basket_return_15d",
        "risk_controls": ["min_monthly_alpha_30d", "signal_count"],
        "feature_row_count": int(prepared.shape[0]),
        "period_count": len(periods),
        "best_config": best_config,
        "current_config_match_check": _match_check(output_dir, monthly_details),
        "summary": summary.to_dict("records"),
    }
    return summary, monthly_details, metadata


def write_outputs(summary: pd.DataFrame, monthly: pd.DataFrame, metadata: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "filter_optimization_from_candidates.csv"
    json_path = output_dir / "filter_optimization_from_candidates.json"
    monthly_path = output_dir / "filter_optimization_monthly_details.csv"
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return {"summary": str(summary_path), "json": str(json_path), "monthly": str(monthly_path)}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    features = _read_features(output_dir, args.feature_file)
    summary, monthly, metadata = build_outputs(features, output_dir, args.strategy)
    outputs = write_outputs(summary, monthly, metadata, output_dir)
    best = metadata.get("best_config", {})
    print(
        {
            "best_config": best.get("config_name"),
            "best_median_alpha_30d": best.get("median_alpha_30d"),
            "best_total_signal_count": best.get("total_signal_count"),
            "summary": outputs["summary"],
            "monthly": outputs["monthly"],
            "json": outputs["json"],
            "match_check": metadata.get("current_config_match_check"),
        }
    )


if __name__ == "__main__":
    main()

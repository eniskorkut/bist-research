from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_RERUN_ROOT = "/data/backtest_outputs/volume_spike_quality_filter_reruns"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rerun-root", default=DEFAULT_RERUN_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    file_path = path / name
    if not file_path.exists():
        raise FileNotFoundError(f"required file missing: {file_path}")
    return pd.read_csv(file_path)


def _metric(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"avg": None, "median": None, "positive_rate": None, "valid_count": 0}
    return {
        "avg": float(values.mean()),
        "median": float(values.median()),
        "positive_rate": float((values > 0).mean()),
        "valid_count": int(values.shape[0]),
    }


def _summarize(config_name: str, output_dir: Path) -> dict[str, Any]:
    period = _read_csv(output_dir, "period_strategy_summary.csv")
    baseline = _read_csv(output_dir, "quality_vs_baseline_summary.csv")
    for frame in [period, baseline]:
        for col in frame.columns:
            if col not in {"period_start", "period_end", "strategy", "outlier_warning", "low_sample_warning"}:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    alpha30 = _metric(period["basket_alpha_30d"])
    ret30 = _metric(period["basket_return_30d"])
    ret15 = _metric(period["basket_return_15d"])
    signal_count = pd.to_numeric(period["signal_count"], errors="coerce").fillna(0)
    baseline_diff = pd.to_numeric(baseline["quality_minus_baseline_alpha_to_current"], errors="coerce")
    outlier_cols = [col for col in ["outlier_warning", "low_sample_warning"] if col in period.columns]
    warning_count = 0
    for col in outlier_cols:
        warning_count += int(period[col].fillna(False).astype(bool).sum())

    return {
        "config_name": config_name,
        "period_count": int(period.shape[0]),
        "active_period_count": int((signal_count > 0).sum()),
        "total_final_signal_count": int(signal_count.sum()),
        "avg_basket_alpha_30d": alpha30["avg"],
        "median_basket_alpha_30d": alpha30["median"],
        "positive_rate_basket_alpha_30d": alpha30["positive_rate"],
        "avg_basket_return_30d": ret30["avg"],
        "median_basket_return_30d": ret30["median"],
        "positive_rate_basket_return_30d": ret30["positive_rate"],
        "avg_basket_return_15d": ret15["avg"],
        "median_basket_return_15d": ret15["median"],
        "baseline_vs_quality_avg_diff": float(baseline_diff.mean()),
        "xu100_30d_beat_months": int((pd.to_numeric(period["basket_alpha_30d"], errors="coerce") > 0).sum()),
        "xu100_30d_loss_months": int((pd.to_numeric(period["basket_alpha_30d"], errors="coerce") < 0).sum()),
        "low_signal_period_count": int((signal_count < 30).sum()),
        "valid_30d_period_count": int(alpha30["valid_count"]),
        "outlier_warning": int(warning_count),
        "score_rank": 0,
        "warning_flags": "",
        "output_dir": str(output_dir),
    }


def _warning_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if int(row["valid_30d_period_count"]) < 20:
        flags.append("low_valid_30d_period_count")
    if int(row["low_signal_period_count"]) > 0:
        flags.append("low_signal_periods")
    if row["avg_basket_alpha_30d"] is not None and row["median_basket_alpha_30d"] is not None:
        if abs(float(row["avg_basket_alpha_30d"]) - float(row["median_basket_alpha_30d"])) > 5:
            flags.append("avg_median_alpha_gap")
    if int(row["outlier_warning"]) > 0:
        flags.append("outlier_or_low_sample_warning")
    return ",".join(flags)


def build_comparison(current_output_dir: Path, rerun_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    configs = {
        "current_config": current_output_dir,
        "relaxed_strong_close": rerun_root / "relaxed_strong_close",
        "no_strong_close": rerun_root / "no_strong_close",
        "combined_relaxed": rerun_root / "combined_relaxed",
    }
    rows = [_summarize(name, path) for name, path in configs.items()]
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        [
            "median_basket_alpha_30d",
            "positive_rate_basket_alpha_30d",
            "avg_basket_alpha_30d",
            "low_signal_period_count",
            "avg_basket_return_30d",
            "avg_basket_return_15d",
        ],
        ascending=[False, False, False, True, False, False],
    ).reset_index(drop=True)
    summary["score_rank"] = range(1, len(summary) + 1)
    summary["warning_flags"] = summary.apply(_warning_flags, axis=1)
    best = summary.iloc[0].to_dict()
    current = summary.loc[summary["config_name"] == "current_config"].iloc[0].to_dict()
    metadata = {
        "decision_order": [
            "median_basket_alpha_30d",
            "positive_rate_basket_alpha_30d",
            "avg_basket_alpha_30d",
            "signal_count_sufficiency",
            "basket_return_15d_tie_breaker",
        ],
        "objective_metric": "basket_alpha_30d",
        "fallback_metric": "basket_return_30d",
        "tie_breaker_metric": "basket_return_15d",
        "diagnostic_only": "basket_alpha_to_current",
        "best_config": best,
        "current_config": current,
    }
    return summary, metadata


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary, metadata = build_comparison(Path(args.current_output_dir), Path(args.rerun_root))

    csv_path = output_dir / "filter_config_comparison.csv"
    json_path = output_dir / "filter_config_comparison.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"comparison_csv={csv_path}")
    print(f"comparison_json={json_path}")
    print(f"best_config={metadata['best_config']['config_name']}")


if __name__ == "__main__":
    main()

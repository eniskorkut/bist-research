from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CURRENT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_CANDIDATE_ROOT = "/data/backtest_outputs/volume_spike_quality_filter_fast_reruns"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-output-dir", default=DEFAULT_CURRENT_OUTPUT_DIR)
    parser.add_argument("--candidate-root", default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_CURRENT_OUTPUT_DIR)
    parser.add_argument("--candidates", default="relaxed_strong_close,combined_relaxed")
    return parser.parse_args(argv)


def _read_period_summary(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "period_strategy_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"required period summary missing: {path}")
    frame = pd.read_csv(path)
    for col in [
        "signal_count",
        "basket_return_15d",
        "basket_return_30d",
        "basket_alpha_30d",
        "basket_alpha_to_current",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _summary(rows: pd.DataFrame) -> dict[str, Any]:
    alpha = pd.to_numeric(rows["candidate_basket_alpha_30d"], errors="coerce").dropna()
    ret30 = pd.to_numeric(rows["candidate_basket_return_30d"], errors="coerce").dropna()
    ret15 = pd.to_numeric(rows["candidate_basket_return_15d"], errors="coerce").dropna()
    signal_delta = pd.to_numeric(rows["signal_count_delta"], errors="coerce").fillna(0)
    alpha_delta = pd.to_numeric(rows["basket_alpha_30d_delta"], errors="coerce").fillna(0)
    return {
        "period_count": int(rows.shape[0]),
        "avg_alpha_30d": float(alpha.mean()) if not alpha.empty else None,
        "median_alpha_30d": float(alpha.median()) if not alpha.empty else None,
        "positive_rate_alpha_30d": float((alpha > 0).mean()) if not alpha.empty else None,
        "avg_return_30d": float(ret30.mean()) if not ret30.empty else None,
        "median_return_30d": float(ret30.median()) if not ret30.empty else None,
        "avg_return_15d": float(ret15.mean()) if not ret15.empty else None,
        "median_return_15d": float(ret15.median()) if not ret15.empty else None,
        "improved_period_count": int((alpha_delta > 0).sum()),
        "worse_period_count": int((alpha_delta < 0).sum()),
        "same_period_count": int((alpha_delta == 0).sum()),
        "signal_count_delta_sum": int(signal_delta.sum()),
        "signal_count_delta_avg": float(signal_delta.mean()) if not signal_delta.empty else 0.0,
    }


def build_comparison(current_output_dir: Path, candidate_root: Path, candidates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = _read_period_summary(current_output_dir)
    current_cols = [
        "period_start",
        "signal_count",
        "basket_return_15d",
        "basket_return_30d",
        "basket_alpha_30d",
        "basket_alpha_to_current",
    ]
    current = current[[col for col in current_cols if col in current.columns]].rename(
        columns={
            "signal_count": "current_signal_count",
            "basket_return_15d": "current_basket_return_15d",
            "basket_return_30d": "current_basket_return_30d",
            "basket_alpha_30d": "current_basket_alpha_30d",
            "basket_alpha_to_current": "current_basket_alpha_to_current",
        }
    )
    rows: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for candidate in candidates:
        cand = _read_period_summary(candidate_root / candidate)
        cand = cand[[col for col in current_cols if col in cand.columns]].rename(
            columns={
                "signal_count": "candidate_signal_count",
                "basket_return_15d": "candidate_basket_return_15d",
                "basket_return_30d": "candidate_basket_return_30d",
                "basket_alpha_30d": "candidate_basket_alpha_30d",
                "basket_alpha_to_current": "candidate_basket_alpha_to_current",
            }
        )
        merged = current.merge(cand, on="period_start", how="inner")
        merged.insert(0, "config_name", candidate)
        merged["signal_count_delta"] = merged["candidate_signal_count"] - merged["current_signal_count"]
        merged["basket_alpha_30d_delta"] = merged["candidate_basket_alpha_30d"] - merged["current_basket_alpha_30d"]
        merged["basket_return_30d_delta"] = merged["candidate_basket_return_30d"] - merged["current_basket_return_30d"]
        merged["basket_return_15d_delta"] = merged["candidate_basket_return_15d"] - merged["current_basket_return_15d"]
        summaries[candidate] = _summary(merged)
        rows.append(merged)
    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            item[1]["median_alpha_30d"] if item[1]["median_alpha_30d"] is not None else float("-inf"),
            item[1]["positive_rate_alpha_30d"] if item[1]["positive_rate_alpha_30d"] is not None else float("-inf"),
            item[1]["avg_alpha_30d"] if item[1]["avg_alpha_30d"] is not None else float("-inf"),
            item[1]["signal_count_delta_avg"],
            item[1]["avg_return_15d"] if item[1]["avg_return_15d"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    metadata = {
        "objective_metric": "basket_alpha_30d",
        "fallback_metric": "basket_return_30d",
        "tie_breaker_metric": "basket_return_15d",
        "diagnostic_only": "basket_alpha_to_current",
        "candidate_summaries": summaries,
        "best_candidate": ranked[0][0] if ranked else None,
        "selected_periods": sorted(comparison["period_start"].unique().tolist()) if not comparison.empty else [],
    }
    return comparison, metadata


def main() -> None:
    args = parse_args()
    candidates = [item.strip() for item in str(args.candidates).split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    comparison, metadata = build_comparison(Path(args.current_output_dir), Path(args.candidate_root), candidates)
    csv_path = output_dir / "candidate_fast_comparison.csv"
    json_path = output_dir / "candidate_fast_comparison.json"
    comparison.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"comparison_csv={csv_path}")
    print(f"comparison_json={json_path}")
    print(f"best_candidate={metadata['best_candidate']}")


if __name__ == "__main__":
    main()

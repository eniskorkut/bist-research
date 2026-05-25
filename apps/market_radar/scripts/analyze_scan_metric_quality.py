from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _read_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.exists():
        raise FileNotFoundError(f"required output file missing: {path}")
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _metric_row(frame: pd.DataFrame, metric: str, column: str) -> dict[str, Any]:
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return {
            "row_type": "metric",
            "metric": metric,
            "n": 0,
            "valid_period_count": 0,
            "avg": None,
            "median": None,
            "positive_rate": None,
            "min": None,
            "max": None,
            "std": None,
            "p10": None,
            "p90": None,
            "recommendation": "",
        }
    return {
        "row_type": "metric",
        "metric": metric,
        "n": int(series.shape[0]),
        "valid_period_count": int(series.shape[0]),
        "avg": float(series.mean()),
        "median": float(series.median()),
        "positive_rate": float((series > 0).mean()),
        "min": float(series.min()),
        "max": float(series.max()),
        "std": float(series.std(ddof=0)),
        "p10": float(series.quantile(0.10)),
        "p90": float(series.quantile(0.90)),
        "recommendation": "",
    }


def _build_monthly(periods: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    monthly_cols = [
        "period_start",
        "basket_return_15d",
        "basket_return_30d",
        "basket_alpha_15d",
        "basket_alpha_30d",
        "basket_alpha_to_current",
        "signal_count",
        "signal_count_before_quality_filter",
        "signal_count_after_quality_filter",
    ]
    monthly = periods[[col for col in monthly_cols if col in periods.columns]].copy()
    monthly["beat_xu100_to_current"] = monthly["basket_alpha_to_current"] > 0
    monthly["beat_xu100_30d"] = monthly["basket_alpha_30d"] > 0
    monthly["beat_xu100_15d"] = monthly["basket_alpha_15d"] > 0

    baseline_cols = [
        "period_start",
        "baseline_alpha_to_current",
        "quality_alpha_to_current",
        "quality_minus_baseline_alpha_to_current",
        "baseline_signal_count",
        "quality_signal_count",
    ]
    keep = [col for col in baseline_cols if col in baseline.columns]
    if keep:
        monthly = monthly.merge(baseline[keep], on="period_start", how="left")
        monthly["quality_improved_vs_baseline"] = monthly["quality_minus_baseline_alpha_to_current"] > 0
    return monthly.sort_values("period_start")


def _filter_summary(reasons: pd.DataFrame, total_before: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reason_cols = [col for col in reasons.columns if col.startswith("failed_")]
    for col in reason_cols:
        total = float(pd.to_numeric(reasons[col], errors="coerce").fillna(0).sum())
        rate = total / total_before if total_before else 0.0
        if col in {"failed_strong_close", "failed_above_ma20"}:
            stance = "consider_relaxing"
        elif col in {"failed_return_5d_pct", "failed_return_10d_pct", "failed_turnover"}:
            stance = "keep_strict"
        else:
            stance = "monitor"
        rows.append(
            {
                "row_type": "filter_reason",
                "metric": col,
                "n": int(total),
                "avg": rate,
                "median": None,
                "positive_rate": None,
                "min": None,
                "max": None,
                "std": None,
                "p10": None,
                "p90": None,
                "recommendation": stance,
            }
        )
    return sorted(rows, key=lambda item: item["n"], reverse=True)


def build_recommendation(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    periods = _numeric(
        _read_csv(output_dir, "period_strategy_summary.csv"),
        [
            "basket_return_15d",
            "basket_return_30d",
            "basket_alpha_15d",
            "basket_alpha_30d",
            "basket_alpha_to_current",
            "signal_count",
            "signal_count_before_quality_filter",
            "signal_count_after_quality_filter",
        ],
    )
    baseline = _numeric(_read_csv(output_dir, "quality_vs_baseline_summary.csv"), [])
    holdings = _read_csv(output_dir, "period_basket_holdings.csv")
    reasons = _read_csv(output_dir, "quality_filter_reason_summary.csv")

    monthly = _build_monthly(periods, baseline)
    total_before = float(periods["signal_count_before_quality_filter"].fillna(0).sum())
    total_after = float(periods["signal_count_after_quality_filter"].fillna(0).sum())
    final_signals = float(periods["signal_count"].fillna(0).sum())

    rows = [
        _metric_row(periods, "primary_candidate_30d_alpha", "basket_alpha_30d"),
        _metric_row(periods, "primary_candidate_30d_return", "basket_return_30d"),
        _metric_row(periods, "secondary_candidate_15d_return", "basket_return_15d"),
        _metric_row(periods, "diagnostic_alpha_to_current", "basket_alpha_to_current"),
    ]
    rows.extend(_filter_summary(reasons, total_before))

    recommendation = {
        "objective_metric": "basket_alpha_30d",
        "fallback_metric": "basket_return_30d",
        "tie_breaker_metric": "basket_return_15d",
        "primary_metric": "basket_alpha_30d",
        "primary_fallback_metric": "basket_return_30d",
        "secondary_tie_breaker": "basket_return_15d",
        "diagnostic_only": "basket_alpha_to_current",
        "risk_controls": {
            "min_signal_count_period_warning": 30,
            "prefer_periods_with_signal_count_at_least": 100,
            "ignore_30d_when_valid_return_30d_missing": True,
        },
        "rationale": {
            "alpha_to_current": "Do not rank production scans by alpha_to_current; long-horizon periods dominate the aggregate.",
            "metric_choice": "30d alpha/return has stronger consistency than 15d while still matching actionable scan horizon.",
            "secondary_metric": "15d return is useful as a recency tie-breaker.",
        },
        "summary": {
            "period_count": int(periods.shape[0]),
            "active_period_count": int((periods["signal_count"].fillna(0) > 0).sum()),
            "empty_period_count": int((periods["signal_count"].fillna(0) == 0).sum()),
            "xu100_beat_to_current_count": int((periods["basket_alpha_to_current"] > 0).sum()),
            "xu100_loss_to_current_count": int((periods["basket_alpha_to_current"] < 0).sum()),
            "signals_before_quality": int(total_before),
            "signals_after_quality": int(total_after),
            "final_signal_count": int(final_signals),
            "holding_rows": int(holdings.shape[0]),
            "baseline_avg_alpha_to_current": float(pd.to_numeric(baseline["baseline_alpha_to_current"], errors="coerce").mean()),
            "quality_avg_alpha_to_current": float(pd.to_numeric(baseline["quality_alpha_to_current"], errors="coerce").mean()),
            "quality_minus_baseline_avg_alpha_to_current": float(
                pd.to_numeric(baseline["quality_minus_baseline_alpha_to_current"], errors="coerce").mean()
            ),
        },
        "monthly": monthly.to_dict("records"),
    }
    return pd.DataFrame(rows), recommendation


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    recommendation_df, recommendation = build_recommendation(output_dir)

    csv_path = output_dir / "scan_metric_recommendation.csv"
    json_path = output_dir / "scan_metric_recommendation.json"
    recommendation_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"recommendation_csv={csv_path}")
    print(f"recommendation_json={json_path}")
    print(f"primary_metric={recommendation['primary_metric']}")
    print(f"secondary_tie_breaker={recommendation['secondary_tie_breaker']}")


if __name__ == "__main__":
    main()

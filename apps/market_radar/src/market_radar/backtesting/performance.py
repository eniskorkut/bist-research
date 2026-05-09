from __future__ import annotations

import pandas as pd


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    if pd.isna(value):
        return None
    return float(value)


def build_strategy_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "signal_count",
                "avg_return_15d",
                "avg_return_30d",
                "avg_alpha_15d",
                "avg_alpha_30d",
                "median_return_15d",
                "median_return_30d",
                "median_alpha_15d",
                "median_alpha_30d",
                "positive_rate_15d",
                "positive_rate_30d",
                "beat_rate_15d",
                "beat_rate_30d",
                "valid_return_15d_count",
                "valid_return_30d_count",
            ]
        )
    grouped = []
    def _series(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series(dtype=float)

    def _safe_median(series: pd.Series) -> float | None:
        if series.empty:
            return None
        value = series.median()
        if pd.isna(value):
            return None
        return float(value)

    def _positive_rate(series: pd.Series) -> float | None:
        if series.empty:
            return None
        return float((series > 0).mean())

    for strategy, frame in signals.groupby("strategy"):
        return15 = _series(frame, "return_15d").dropna()
        return30 = _series(frame, "return_30d").dropna()
        alpha15 = _series(frame, "alpha_15d").dropna()
        alpha30 = _series(frame, "alpha_30d").dropna()
        grouped.append(
            {
                "strategy": strategy,
                "signal_count": int(len(frame)),
                "avg_return_15d": _safe_mean(return15),
                "avg_return_30d": _safe_mean(return30),
                "avg_alpha_15d": _safe_mean(alpha15),
                "avg_alpha_30d": _safe_mean(alpha30),
                "median_return_15d": _safe_median(return15),
                "median_return_30d": _safe_median(return30),
                "median_alpha_15d": _safe_median(alpha15),
                "median_alpha_30d": _safe_median(alpha30),
                "positive_rate_15d": _positive_rate(return15),
                "positive_rate_30d": _positive_rate(return30),
                "beat_rate_15d": _safe_mean(_series(frame, "beat_xu100_15d").dropna().astype(float)),
                "beat_rate_30d": _safe_mean(_series(frame, "beat_xu100_30d").dropna().astype(float)),
                "valid_return_15d_count": int(len(return15)),
                "valid_return_30d_count": int(len(return30)),
            }
        )
    return pd.DataFrame(grouped).sort_values("signal_count", ascending=False)


def build_monthly_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["month", "signal_count", "avg_alpha_15d", "avg_alpha_30d"])
    frame = signals.copy()
    if "alpha_15d" not in frame.columns:
        frame["alpha_15d"] = pd.NA
    if "alpha_30d" not in frame.columns:
        frame["alpha_30d"] = pd.NA
    frame["month"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.to_period("M").astype(str)
    out = (
        frame.groupby("month")
        .agg(
            signal_count=("symbol", "count"),
            avg_alpha_15d=("alpha_15d", "mean"),
            avg_alpha_30d=("alpha_30d", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    return out


def build_yearly_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["year", "signal_count", "avg_alpha_15d", "avg_alpha_30d"])
    frame = signals.copy()
    if "alpha_15d" not in frame.columns:
        frame["alpha_15d"] = pd.NA
    if "alpha_30d" not in frame.columns:
        frame["alpha_30d"] = pd.NA
    frame["year"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.year
    out = (
        frame.groupby("year")
        .agg(
            signal_count=("symbol", "count"),
            avg_alpha_15d=("alpha_15d", "mean"),
            avg_alpha_30d=("alpha_30d", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    return out

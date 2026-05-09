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
                "beat_rate_15d",
                "beat_rate_30d",
            ]
        )
    grouped = []
    def _series(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series(dtype=float)

    for strategy, frame in signals.groupby("strategy"):
        grouped.append(
            {
                "strategy": strategy,
                "signal_count": int(len(frame)),
                "avg_return_15d": _safe_mean(_series(frame, "return_15d").dropna()),
                "avg_return_30d": _safe_mean(_series(frame, "return_30d").dropna()),
                "avg_alpha_15d": _safe_mean(_series(frame, "alpha_15d").dropna()),
                "avg_alpha_30d": _safe_mean(_series(frame, "alpha_30d").dropna()),
                "beat_rate_15d": _safe_mean(_series(frame, "beat_xu100_15d").dropna().astype(float)),
                "beat_rate_30d": _safe_mean(_series(frame, "beat_xu100_30d").dropna().astype(float)),
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

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_radar.backtesting.backtest_engine import BacktestConfig, run_backtest
from market_radar.data_access import BorsapyMarketDataClient
from market_radar.symbols import normalize_bist_symbol
from market_radar.regime import MarketRegimeDetector


@dataclass(frozen=True)
class PeriodWindow:
    period_start: date
    period_end: date


@dataclass
class PeriodBacktestConfig:
    universe_symbol: str = "XUTUM"
    benchmark_symbol: str = "XU100"
    lookback_days: int = 700
    strategies: list[str] | None = None
    max_workers: int = 8
    cooldown_days: int = 15
    force_refresh: bool = False
    db_path: str = "/data/market_radar_cache.sqlite"
    output_dir: str = "/data/backtest_outputs/period_runs"
    period_starts: list[str] | None = None
    period_ends: list[str] | None = None
    basket_mode: str = "first_signal_per_symbol"
    as_of_date: str | None = None
    active_volume_spike_quality: bool = False
    min_last_turnover_try: float = 10_000_000.0
    min_avg_turnover_20d_try: float = 10_000_000.0
    max_rsi_14: float = 78.0
    max_return_5d_pct: float = 35.0
    max_return_10d_pct: float = 60.0
    require_strong_close: bool = True
    min_close_position: float = 0.60
    min_above_ma20_ratio: float = 1.0
    transaction_cost_pct: float = 0.0
    cache_only: bool = False
    max_symbols: int | None = None
    only_symbols: list[str] | None = None


@dataclass
class PeriodBacktestResult:
    holdings: pd.DataFrame
    period_strategy_summary: pd.DataFrame
    strategy_stability_summary: pd.DataFrame
    period_diagnostics_summary: pd.DataFrame
    quality_filter_reason_summary: pd.DataFrame
    data_coverage_summary: pd.DataFrame
    run_summary: dict[str, Any]
    regime_config_comparison: pd.DataFrame | None = None


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def build_period_windows(period_starts: list[str], period_ends: list[str] | None = None) -> list[PeriodWindow]:
    starts = sorted({_parse_date(item) for item in period_starts})
    if not starts:
        return []
    end_values = [_parse_date(item) for item in (period_ends or [])]
    windows: list[PeriodWindow] = []
    for idx, start in enumerate(starts):
        if idx < len(end_values):
            end = end_values[idx]
        elif idx + 1 < len(starts):
            end = starts[idx + 1]
        else:
            end = _next_month_start(start)
        windows.append(PeriodWindow(period_start=start, period_end=end))
    return windows


def _trimmed_mean(series: pd.Series, proportion: float = 0.1) -> float | None:
    clean = series.dropna().sort_values()
    n = len(clean)
    if n == 0:
        return None
    k = int(n * proportion)
    if k == 0 or n <= (2 * k):
        value = clean.mean()
    else:
        value = clean.iloc[k : n - k].mean()
    if pd.isna(value):
        return None
    return float(value)


def _safe_mean(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    value = clean.mean()
    if pd.isna(value):
        return None
    return float(value)


def _safe_median(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    value = clean.median()
    if pd.isna(value):
        return None
    return float(value)


def _safe_rate_positive(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float((clean > 0).mean())


def _safe_bool_rate(series: pd.Series) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series.replace({True: 1, False: 0}), errors="coerce")
    clean = numeric.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _prepare_signals_frame(raw_signals: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(raw_signals).copy()
    if frame.empty:
        return frame
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_15d_date"] = pd.to_datetime(frame["exit_15d_date"], errors="coerce")
    frame["exit_30d_date"] = pd.to_datetime(frame["exit_30d_date"], errors="coerce")
    for col in [
        "return_15d",
        "return_30d",
        "benchmark_return_15d",
        "benchmark_return_30d",
        "alpha_15d",
        "alpha_30d",
        "interest_score",
        "volume_ratio_20d",
        "turnover_ratio_20d",
        "close_position",
        "daily_return_pct",
        "near_20d_high_pct",
        "turnover",
        "avg_turnover_20d",
        "ma20",
        "rsi_14",
        "return_5d_pct",
        "return_10d_pct",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "beat_xu100_15d" in frame.columns:
        frame["beat_xu100_15d"] = frame["beat_xu100_15d"].map(_to_bool)
    if "beat_xu100_30d" in frame.columns:
        frame["beat_xu100_30d"] = frame["beat_xu100_30d"].map(_to_bool)
    return frame


def _apply_volume_spike_quality_filter(frame: pd.DataFrame, config: PeriodBacktestConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["quality_filter_enabled"] = bool(config.active_volume_spike_quality)
    out["filter_reasons"] = ""
    out["failed_reasons"] = ""
    out["filter_passed"] = True
    if not config.active_volume_spike_quality:
        return out

    filter_reasons = [
        "volume_spike_strict",
        "min_last_turnover_try",
        "min_avg_turnover_20d_try",
        "above_ma20",
        "max_rsi_14",
        "max_return_5d_pct",
        "max_return_10d_pct",
    ]
    if config.require_strong_close:
        filter_reasons.append("strong_close")

    for idx, row in out.iterrows():
        failed: list[str] = []
        strategy = str(row.get("strategy") or "")
        if strategy != "volume_spike_strict":
            out.at[idx, "filter_reasons"] = ",".join(filter_reasons)
            out.at[idx, "failed_reasons"] = ""
            out.at[idx, "filter_passed"] = True
            continue

        v_ratio = pd.to_numeric(row.get("volume_ratio_20d"), errors="coerce")
        t_ratio = pd.to_numeric(row.get("turnover_ratio_20d"), errors="coerce")
        if pd.isna(v_ratio) or pd.isna(t_ratio) or float(v_ratio) < 1.5 or float(t_ratio) < 1.2:
            failed.append("volume_spike_strict")

        turnover = pd.to_numeric(row.get("turnover"), errors="coerce")
        if pd.isna(turnover) or float(turnover) < float(config.min_last_turnover_try):
            failed.append("min_last_turnover_try")

        avg_turn = pd.to_numeric(row.get("avg_turnover_20d"), errors="coerce")
        if pd.isna(avg_turn) or float(avg_turn) < float(config.min_avg_turnover_20d_try):
            failed.append("min_avg_turnover_20d_try")

        close_value = pd.to_numeric(
            row.get("entry_price", row.get("entry_close", row.get("close"))),
            errors="coerce",
        )
        ma20_value = pd.to_numeric(row.get("ma20"), errors="coerce")
        above_ma20 = row.get("above_ma20")
        ma20_passed = above_ma20 is True
        if not ma20_passed and float(config.min_above_ma20_ratio) < 1.0:
            ma20_passed = (
                not pd.isna(close_value)
                and not pd.isna(ma20_value)
                and float(close_value) >= float(ma20_value) * float(config.min_above_ma20_ratio)
            )
        if not ma20_passed:
            failed.append("above_ma20")

        rsi = pd.to_numeric(row.get("rsi_14"), errors="coerce")
        if pd.isna(rsi) or float(rsi) > float(config.max_rsi_14) or float(rsi) > 80.0:
            failed.append("max_rsi_14")

        ret5 = pd.to_numeric(row.get("return_5d_pct"), errors="coerce")
        if pd.isna(ret5) or float(ret5) > float(config.max_return_5d_pct):
            failed.append("max_return_5d_pct")

        ret10 = pd.to_numeric(row.get("return_10d_pct"), errors="coerce")
        if pd.isna(ret10) or float(ret10) > float(config.max_return_10d_pct):
            failed.append("max_return_10d_pct")

        if config.require_strong_close:
            close_pos = pd.to_numeric(row.get("close_position"), errors="coerce")
            if pd.isna(close_pos) or float(close_pos) < float(config.min_close_position):
                failed.append("strong_close")

        out.at[idx, "filter_reasons"] = ",".join(filter_reasons)
        out.at[idx, "failed_reasons"] = ",".join(failed)
        out.at[idx, "filter_passed"] = len(failed) == 0
    return out


def _reason_flags(failed_reasons: str) -> dict[str, int]:
    parts = {item.strip() for item in str(failed_reasons or "").split(",") if item.strip()}
    return {
        "failed_rsi_14": int("max_rsi_14" in parts),
        "failed_turnover": int("min_last_turnover_try" in parts),
        "failed_avg_turnover_20d": int("min_avg_turnover_20d_try" in parts),
        "failed_above_ma20": int("above_ma20" in parts),
        "failed_return_5d_pct": int("max_return_5d_pct" in parts),
        "failed_return_10d_pct": int("max_return_10d_pct" in parts),
        "failed_strong_close": int("strong_close" in parts),
        "failed_missing_metric": int(any(x in parts for x in {"max_rsi_14", "max_return_5d_pct", "max_return_10d_pct"}) and "volume_spike_strict" not in parts),
    }


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().sort_index()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    return out


def _history_latest_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or df.empty:
        return None
    return pd.Timestamp(df.index.max()).tz_localize(None)


def _close_on_or_before(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    if df is None or df.empty:
        return None, None
    target = pd.Timestamp(target).tz_localize(None)
    idx = df.index.searchsorted(target, side="right") - 1
    if idx < 0 or idx >= len(df):
        return None, None
    dt = pd.Timestamp(df.index[idx]).tz_localize(None)
    val = pd.to_numeric(df.iloc[idx].get("close"), errors="coerce")
    if pd.isna(val):
        return dt, None
    return dt, float(val)


def _safe_return(entry: float | None, exit_: float | None) -> float | None:
    if entry is None or exit_ is None or entry == 0:
        return None
    return ((exit_ / entry) - 1.0) * 100.0


def _trading_row_index(df: pd.DataFrame, target: pd.Timestamp) -> int | None:
    if df is None or df.empty:
        return None
    target_day = pd.Timestamp(target).tz_localize(None).normalize()
    index_days = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    matches = (index_days >= target_day).nonzero()[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _row_close(df: pd.DataFrame, idx: int) -> tuple[pd.Timestamp | None, float | None]:
    if idx < 0 or idx >= len(df):
        return None, None
    dt = pd.Timestamp(df.index[idx]).tz_localize(None)
    val = pd.to_numeric(df.iloc[idx].get("close"), errors="coerce")
    if pd.isna(val):
        return dt, None
    return dt, float(val)


def _pick_basket(frame: pd.DataFrame, basket_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["signal_date", "entry_date", "symbol"]).copy()
    if basket_mode == "signal_weighted":
        return ordered
    return ordered.drop_duplicates(subset=["symbol"], keep="first")


def _contribution_pct(series: pd.Series, top_n: int) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    total = clean.sum()
    if total == 0:
        return None
    top_sum = clean.sort_values(ascending=False).head(top_n).sum()
    return float((top_sum / total) * 100.0)


def _add_to_latest_metrics(
    basket: pd.DataFrame,
    symbol_histories: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame,
    effective_as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if basket.empty:
        return basket.copy()
    out = basket.copy()
    out["exit_date_to_current"] = pd.NaT
    out["exit_price_to_current"] = pd.NA
    out["return_to_current"] = pd.NA
    out["benchmark_return_to_current"] = pd.NA
    out["alpha_to_current"] = pd.NA
    out["beat_xu100_to_current"] = pd.NA

    for idx, row in out.iterrows():
        symbol = str(row.get("symbol") or "")
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        if pd.isna(signal_date):
            continue
        signal_date = pd.Timestamp(signal_date).tz_localize(None)
        sym_hist = symbol_histories.get(symbol)
        if sym_hist is None or sym_hist.empty:
            continue
        entry_idx = _trading_row_index(sym_hist, signal_date)
        if entry_idx is None:
            continue
        entry_dt, entry_price = _row_close(sym_hist, entry_idx)
        if entry_dt is None or entry_price is None:
            continue

        # Entry convention for period backtest: signal-day close
        out.at[idx, "entry_date"] = entry_dt
        out.at[idx, "entry_close"] = entry_price

        # 15/30 trading-day exits from signal day
        exit15_dt, exit15_price = _row_close(sym_hist, entry_idx + 15)
        exit30_dt, exit30_price = _row_close(sym_hist, entry_idx + 30)
        out.at[idx, "exit_15d_date"] = exit15_dt
        out.at[idx, "exit_15d_close"] = exit15_price
        out.at[idx, "exit_30d_date"] = exit30_dt
        out.at[idx, "exit_30d_close"] = exit30_price
        ret15 = _safe_return(entry_price, exit15_price)
        ret30 = _safe_return(entry_price, exit30_price)
        out.at[idx, "return_15d"] = ret15
        out.at[idx, "return_30d"] = ret30

        bench_entry_dt, bench_entry_price = _close_on_or_before(benchmark_history, entry_dt)
        bench_exit_15_dt, bench_exit_15_price = (None, None)
        bench_exit_30_dt, bench_exit_30_price = (None, None)
        if exit15_dt is not None:
            bench_exit_15_dt, bench_exit_15_price = _close_on_or_before(benchmark_history, exit15_dt)
        if exit30_dt is not None:
            bench_exit_30_dt, bench_exit_30_price = _close_on_or_before(benchmark_history, exit30_dt)

        bench_ret15 = _safe_return(bench_entry_price, bench_exit_15_price)
        bench_ret30 = _safe_return(bench_entry_price, bench_exit_30_price)
        out.at[idx, "benchmark_return_15d"] = bench_ret15
        out.at[idx, "benchmark_return_30d"] = bench_ret30
        alpha15 = None if ret15 is None or bench_ret15 is None else ret15 - bench_ret15
        alpha30 = None if ret30 is None or bench_ret30 is None else ret30 - bench_ret30
        out.at[idx, "alpha_15d"] = alpha15
        out.at[idx, "alpha_30d"] = alpha30
        out.at[idx, "beat_xu100_15d"] = None if alpha15 is None else alpha15 > 0
        out.at[idx, "beat_xu100_30d"] = None if alpha30 is None else alpha30 > 0

        exit_current_dt, exit_current_price = _close_on_or_before(sym_hist, effective_as_of_date)
        if exit_current_dt is None or exit_current_price is None:
            continue
        bench_exit_cur_dt, bench_exit_cur_price = _close_on_or_before(benchmark_history, exit_current_dt)
        ret_cur = _safe_return(entry_price, exit_current_price)
        bench_ret_cur = _safe_return(bench_entry_price, bench_exit_cur_price)
        alpha_cur = None if ret_cur is None or bench_ret_cur is None else ret_cur - bench_ret_cur

        out.at[idx, "exit_date_to_current"] = exit_current_dt
        out.at[idx, "exit_price_to_current"] = exit_current_price
        out.at[idx, "return_to_current"] = ret_cur
        out.at[idx, "benchmark_return_to_current"] = bench_ret_cur
        out.at[idx, "alpha_to_current"] = alpha_cur
        out.at[idx, "beat_xu100_to_current"] = None if alpha_cur is None else alpha_cur > 0
    return out


def _build_period_strategy_summary(
    period_start: date,
    period_end: date,
    effective_as_of_date: pd.Timestamp | None,
    universe_symbol: str,
    benchmark_symbol: str,
    universe_symbol_count: int,
    strategy: str,
    basket_mode: str,
    basket: pd.DataFrame,
    *,
    quality_filter_enabled: bool,
    quality_config: PeriodBacktestConfig,
    signal_count_before_quality_filter: int,
    signal_count_after_quality_filter: int,
    regime_label: str = "Neutral",
    xu100_return_20d_pct: float = 0.0,
    xu100_close: float | None = None,
    xu100_ma50: float | None = None,
    xu100_ma200: float | None = None,
    selected_config: str = "current_config",
    transaction_cost_pct: float = 0.0,
) -> dict[str, Any]:
    signal_count = int(len(basket))
    unique_symbol_count = int(basket["symbol"].nunique()) if not basket.empty else 0
    ret15 = basket["return_15d"] if "return_15d" in basket.columns else pd.Series(dtype=float)
    ret30 = basket["return_30d"] if "return_30d" in basket.columns else pd.Series(dtype=float)
    b15 = basket["benchmark_return_15d"] if "benchmark_return_15d" in basket.columns else pd.Series(dtype=float)
    b30 = basket["benchmark_return_30d"] if "benchmark_return_30d" in basket.columns else pd.Series(dtype=float)
    a15 = basket["alpha_15d"] if "alpha_15d" in basket.columns else pd.Series(dtype=float)
    a30 = basket["alpha_30d"] if "alpha_30d" in basket.columns else pd.Series(dtype=float)
    ret_current = basket["return_to_current"] if "return_to_current" in basket.columns else pd.Series(dtype=float)
    b_current = basket["benchmark_return_to_current"] if "benchmark_return_to_current" in basket.columns else pd.Series(dtype=float)
    a_current = basket["alpha_to_current"] if "alpha_to_current" in basket.columns else pd.Series(dtype=float)

    basket_return_15d = _safe_mean(ret15)
    basket_return_30d = _safe_mean(ret30)
    benchmark_return_15d = _safe_mean(b15)
    benchmark_return_30d = _safe_mean(b30)
    basket_alpha_15d = None if basket_return_15d is None or benchmark_return_15d is None else basket_return_15d - benchmark_return_15d
    basket_alpha_30d = None if basket_return_30d is None or benchmark_return_30d is None else basket_return_30d - benchmark_return_30d
    basket_return_to_current = _safe_mean(ret_current)
    benchmark_return_to_current = _safe_mean(b_current)
    basket_alpha_to_current = None if basket_return_to_current is None or benchmark_return_to_current is None else basket_return_to_current - benchmark_return_to_current

    valid30 = ret30.dropna()
    best_symbol_30d = None
    best_return_30d = None
    worst_symbol_30d = None
    worst_return_30d = None
    if not valid30.empty:
        best_idx = valid30.idxmax()
        worst_idx = valid30.idxmin()
        best_symbol_30d = basket.loc[best_idx, "symbol"]
        best_return_30d = float(valid30.loc[best_idx])
        worst_symbol_30d = basket.loc[worst_idx, "symbol"]
        worst_return_30d = float(valid30.loc[worst_idx])

    valid15 = ret15.dropna()
    best_symbol_15d = None
    best_return_15d = None
    worst_symbol_15d = None
    worst_return_15d = None
    if not valid15.empty:
        best_idx = valid15.idxmax()
        worst_idx = valid15.idxmin()
        best_symbol_15d = basket.loc[best_idx, "symbol"]
        best_return_15d = float(valid15.loc[best_idx])
        worst_symbol_15d = basket.loc[worst_idx, "symbol"]
        worst_return_15d = float(valid15.loc[worst_idx])

    top1 = _contribution_pct(valid30, 1)
    top3 = _contribution_pct(valid30, 3)
    top1_current = _contribution_pct(ret_current.dropna(), 1)
    top3_current = _contribution_pct(ret_current.dropna(), 3)
    mean30 = _safe_mean(ret30)
    median30 = _safe_median(ret30)
    outlier_warning = False
    if top1_current is not None and top1_current >= 50:
        outlier_warning = True
    if top3_current is not None and top3_current >= 80:
        outlier_warning = True
    if mean30 is not None and median30 is not None and (mean30 - median30) >= 10:
        outlier_warning = True
    low_sample_warning = signal_count < 10

    best_symbol_current = None
    best_return_current = None
    worst_symbol_current = None
    worst_return_current = None
    valid_current = ret_current.dropna()
    if not valid_current.empty:
        best_idx = valid_current.idxmax()
        worst_idx = valid_current.idxmin()
        best_symbol_current = basket.loc[best_idx, "symbol"]
        best_return_current = float(valid_current.loc[best_idx])
        worst_symbol_current = basket.loc[worst_idx, "symbol"]
        worst_return_current = float(valid_current.loc[worst_idx])

    net_basket_alpha_30d = None
    if basket_alpha_30d is not None:
        net_basket_alpha_30d = basket_alpha_30d - float(transaction_cost_pct)

    return {
        "period": period_start.isoformat(),
        "regime_label": regime_label,
        "xu100_return_20d_pct": xu100_return_20d_pct,
        "xu100_close": xu100_close,
        "xu100_ma50": xu100_ma50,
        "xu100_ma200": xu100_ma200,
        "selected_config": selected_config,
        "transaction_cost_pct": float(transaction_cost_pct),
        "signal_count": signal_count,
        "basket_return_15d": basket_return_15d,
        "basket_return_30d": basket_return_30d,
        "basket_alpha_30d": basket_alpha_30d,
        "net_basket_alpha_30d": net_basket_alpha_30d,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "effective_as_of_date": None if effective_as_of_date is None else effective_as_of_date.date().isoformat(),
        "universe": universe_symbol,
        "benchmark": benchmark_symbol,
        "universe_symbol_count": universe_symbol_count,
        "benchmark_symbol": benchmark_symbol,
        "strategy": strategy,
        "basket_mode": basket_mode,
        "quality_filter_enabled": quality_filter_enabled,
        "min_last_turnover_try": quality_config.min_last_turnover_try,
        "min_avg_turnover_20d_try": quality_config.min_avg_turnover_20d_try,
        "max_rsi_14": quality_config.max_rsi_14,
        "max_return_5d_pct": quality_config.max_return_5d_pct,
        "max_return_10d_pct": quality_config.max_return_10d_pct,
        "require_strong_close": quality_config.require_strong_close,
        "min_close_position": quality_config.min_close_position,
        "min_above_ma20_ratio": quality_config.min_above_ma20_ratio,
        "signal_count_before_quality_filter": signal_count_before_quality_filter,
        "signal_count_after_quality_filter": signal_count_after_quality_filter,
        "filtered_out_count": max(0, signal_count_before_quality_filter - signal_count_after_quality_filter),
        "unique_symbol_count": unique_symbol_count,
        "benchmark_return_15d": benchmark_return_15d,
        "benchmark_return_30d": benchmark_return_30d,
        "basket_alpha_15d": basket_alpha_15d,
        "basket_return_to_current": basket_return_to_current,
        "benchmark_return_to_current": benchmark_return_to_current,
        "basket_alpha_to_current": basket_alpha_to_current,
        "avg_return_15d": _safe_mean(ret15),
        "avg_return_30d": _safe_mean(ret30),
        "avg_return_to_current": _safe_mean(ret_current),
        "median_return_15d": _safe_median(ret15),
        "median_return_30d": _safe_median(ret30),
        "median_return_to_current": _safe_median(ret_current),
        "trimmed_mean_return_15d": _trimmed_mean(ret15),
        "trimmed_mean_return_30d": _trimmed_mean(ret30),
        "trimmed_mean_return_to_current": _trimmed_mean(ret_current),
        "avg_alpha_15d": _safe_mean(a15),
        "avg_alpha_30d": _safe_mean(a30),
        "avg_alpha_to_current": _safe_mean(a_current),
        "median_alpha_15d": _safe_median(a15),
        "median_alpha_30d": _safe_median(a30),
        "median_alpha_to_current": _safe_median(a_current),
        "trimmed_mean_alpha_15d": _trimmed_mean(a15),
        "trimmed_mean_alpha_30d": _trimmed_mean(a30),
        "trimmed_mean_alpha_to_current": _trimmed_mean(a_current),
        "positive_rate_15d": _safe_rate_positive(ret15),
        "positive_rate_30d": _safe_rate_positive(ret30),
        "positive_rate_to_current": _safe_rate_positive(ret_current),
        "beat_rate_15d": _safe_bool_rate(basket["beat_xu100_15d"] if "beat_xu100_15d" in basket.columns else pd.Series(dtype=float)),
        "beat_rate_30d": _safe_bool_rate(basket["beat_xu100_30d"] if "beat_xu100_30d" in basket.columns else pd.Series(dtype=float)),
        "beat_rate_to_current": _safe_bool_rate(basket["beat_xu100_to_current"] if "beat_xu100_to_current" in basket.columns else pd.Series(dtype=float)),
        "valid_return_15d_count": int(ret15.dropna().shape[0]),
        "valid_return_30d_count": int(ret30.dropna().shape[0]),
        "valid_return_to_current_count": int(ret_current.dropna().shape[0]),
        "best_symbol_15d": best_symbol_15d,
        "best_return_15d": best_return_15d,
        "worst_symbol_15d": worst_symbol_15d,
        "worst_return_15d": worst_return_15d,
        "best_symbol_30d": best_symbol_30d,
        "best_return_30d": best_return_30d,
        "worst_symbol_30d": worst_symbol_30d,
        "worst_return_30d": worst_return_30d,
        "best_symbol_to_current": best_symbol_current,
        "best_return_to_current": best_return_current,
        "worst_symbol_to_current": worst_symbol_current,
        "worst_return_to_current": worst_return_current,
        "top_1_contribution_30d_pct": top1,
        "top_3_contribution_30d_pct": top3,
        "top_1_contribution_to_current_pct": top1_current,
        "top_3_contribution_to_current_pct": top3_current,
        "outlier_warning": outlier_warning,
        "low_sample_warning": low_sample_warning,
    }


def _build_stability_summary(period_summary: pd.DataFrame) -> pd.DataFrame:
    if period_summary.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "period_count",
                "active_period_count",
                "empty_period_count",
                "total_signal_count",
                "total_unique_symbol_count",
                "avg_basket_return_15d",
                "avg_basket_return_30d",
                "avg_basket_return_to_current",
                "median_basket_return_15d",
                "median_basket_return_30d",
                "median_basket_return_to_current",
                "avg_basket_alpha_15d",
                "avg_basket_alpha_30d",
                "avg_basket_alpha_to_current",
                "median_basket_alpha_15d",
                "median_basket_alpha_30d",
                "median_basket_alpha_to_current",
                "positive_period_rate_15d",
                "positive_period_rate_30d",
                "positive_period_rate_to_current",
                "beat_period_rate_15d",
                "beat_period_rate_30d",
                "beat_period_rate_to_current",
                "best_period_to_current",
                "worst_period_to_current",
                "consistency_score",
                "outlier_period_count",
                "low_sample_period_count",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for strategy, frame in period_summary.groupby("strategy"):
        active_frame = frame.loc[frame["signal_count"] > 0].copy()
        ret15 = active_frame["basket_return_15d"] if not active_frame.empty else pd.Series(dtype=float)
        ret30 = active_frame["basket_return_30d"] if not active_frame.empty else pd.Series(dtype=float)
        ret_current = active_frame["basket_return_to_current"] if not active_frame.empty else pd.Series(dtype=float)
        a15 = active_frame["basket_alpha_15d"] if not active_frame.empty else pd.Series(dtype=float)
        a30 = active_frame["basket_alpha_30d"] if not active_frame.empty else pd.Series(dtype=float)
        a_current = active_frame["basket_alpha_to_current"] if not active_frame.empty else pd.Series(dtype=float)
        best_period_current = None
        worst_period_current = None
        if not ret_current.dropna().empty:
            best_idx = ret_current.idxmax()
            worst_idx = ret_current.idxmin()
            best_period_current = active_frame.loc[best_idx, "period_start"]
            worst_period_current = active_frame.loc[worst_idx, "period_start"]
        period_count = int(frame["period_start"].nunique())
        active_period_count = int((frame["signal_count"] > 0).sum())
        empty_period_count = period_count - active_period_count
        positive_period_rate_15d = _safe_rate_positive(ret15)
        positive_period_rate_30d = _safe_rate_positive(ret30)
        positive_period_rate_current = _safe_rate_positive(ret_current)
        beat_period_rate_15d = _safe_rate_positive(a15)
        beat_period_rate_30d = _safe_rate_positive(a30)
        beat_period_rate_current = _safe_rate_positive(a_current)
        consistency_score = 0.0
        for value in [positive_period_rate_15d, positive_period_rate_30d, positive_period_rate_current, beat_period_rate_15d, beat_period_rate_30d, beat_period_rate_current]:
            if value is not None:
                consistency_score += value * (100.0 / 6.0)
        out_rows.append(
            {
                "strategy": strategy,
                "period_count": period_count,
                "active_period_count": active_period_count,
                "empty_period_count": empty_period_count,
                "total_signal_count": int(frame["signal_count"].sum()),
                "total_unique_symbol_count": int(active_frame["unique_symbol_count"].sum()) if not active_frame.empty else 0,
                "avg_basket_return_15d": _safe_mean(ret15),
                "avg_basket_return_30d": _safe_mean(ret30),
                "avg_basket_return_to_current": _safe_mean(ret_current),
                "median_basket_return_15d": _safe_median(ret15),
                "median_basket_return_30d": _safe_median(ret30),
                "median_basket_return_to_current": _safe_median(ret_current),
                "avg_basket_alpha_15d": _safe_mean(a15),
                "avg_basket_alpha_30d": _safe_mean(a30),
                "avg_basket_alpha_to_current": _safe_mean(a_current),
                "median_basket_alpha_15d": _safe_median(a15),
                "median_basket_alpha_30d": _safe_median(a30),
                "median_basket_alpha_to_current": _safe_median(a_current),
                "positive_period_rate_15d": positive_period_rate_15d,
                "positive_period_rate_30d": positive_period_rate_30d,
                "positive_period_rate_to_current": positive_period_rate_current,
                "beat_period_rate_15d": beat_period_rate_15d,
                "beat_period_rate_30d": beat_period_rate_30d,
                "beat_period_rate_to_current": beat_period_rate_current,
                "best_period_to_current": best_period_current,
                "worst_period_to_current": worst_period_current,
                "consistency_score": consistency_score,
                "outlier_period_count": int(frame["outlier_warning"].fillna(False).astype(bool).sum()),
                "low_sample_period_count": int(frame["low_sample_warning"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("strategy")


def _compute_config_metrics(
    scoped_signals: pd.DataFrame,
    strategy: str,
    cfg: PeriodBacktestConfig,
    symbol_histories: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame,
    effective_as_of: pd.Timestamp | None,
) -> dict[str, Any]:
    strat = scoped_signals.loc[scoped_signals["strategy"] == strategy].copy()
    if strat.empty:
        return {"alpha_30d": None, "signal_count": 0, "signature": None}
    strat = _apply_volume_spike_quality_filter(strat, cfg)
    if cfg.active_volume_spike_quality and strategy == "volume_spike_strict":
        strat = strat.loc[strat["filter_passed"] == True].copy()  # noqa: E712
    basket = _pick_basket(strat, cfg.basket_mode)
    if basket.empty:
        return {"alpha_30d": None, "signal_count": 0, "signature": None}
    if effective_as_of is not None:
        basket = _add_to_latest_metrics(
            basket=basket,
            symbol_histories=symbol_histories,
            benchmark_history=benchmark_history,
            effective_as_of_date=effective_as_of,
        )
    a30 = basket["alpha_30d"] if "alpha_30d" in basket.columns else pd.Series(dtype=float)
    fingerprint_series = (
        basket.sort_values(["symbol", "signal_date"])[["symbol", "signal_date"]]
        .astype(str)
        .agg(":".join, axis=1)
    )
    fingerprint = "|".join(fingerprint_series.tolist())
    signature = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest() if fingerprint else None
    return {
        "alpha_30d": _safe_mean(a30),
        "signal_count": int(len(basket)),
        "signature": signature,
    }


def run_period_backtest(config: PeriodBacktestConfig) -> PeriodBacktestResult:
    windows = build_period_windows(config.period_starts or [], config.period_ends)
    universe_symbol = normalize_bist_symbol(config.universe_symbol) or "XUTUM"
    benchmark_symbol = normalize_bist_symbol(config.benchmark_symbol) or "XU100"
    raw = run_backtest(
        BacktestConfig(
            universe_symbol=universe_symbol,
            benchmark_symbol=benchmark_symbol,
            lookback_days=config.lookback_days,
            strategies=config.strategies,
            max_workers=config.max_workers,
            db_path=config.db_path,
            force_refresh=config.force_refresh,
            cooldown_days=config.cooldown_days,
            cache_only=config.cache_only,
            max_symbols=config.max_symbols,
            only_symbols=config.only_symbols,
        )
    )
    signals = _prepare_signals_frame(raw.signals)
    
    # Quality filter will be applied dynamically inside the loop per period
    if signals.empty:
        empty = pd.DataFrame()
        return PeriodBacktestResult(
            holdings=empty,
            period_strategy_summary=empty,
            strategy_stability_summary=empty,
            period_diagnostics_summary=empty,
            quality_filter_reason_summary=empty,
            data_coverage_summary=empty,
            run_summary={
                "period_count": len(windows),
                "basket_mode": config.basket_mode,
                "scan_summary": raw.scan_summary,
                "failed_symbols": raw.failed_symbols,
            },
        )

    strategies = sorted(set(signals["strategy"].dropna().astype(str).tolist()))
    radar_client = BorsapyMarketDataClient()
    symbol_histories: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(signals["symbol"].dropna().astype(str).tolist())):
        try:
            history = radar_client.load_history(
                symbol,
                lookback_days=config.lookback_days,
                db_path=config.db_path,
                force=config.force_refresh,
                cache_only=config.cache_only,
            )
        except TypeError:
            history = radar_client.load_history(
                symbol,
                lookback_days=config.lookback_days,
                db_path=config.db_path,
                force=config.force_refresh,
            )
        symbol_histories[symbol] = _normalize_history(history)
    coverage_rows: list[dict[str, Any]] = []
    for symbol, hist in symbol_histories.items():
        if hist.empty:
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "first_date": None,
                    "last_date": None,
                    "row_count": 0,
                    "has_2024_data": False,
                    "has_2025_data": False,
                    "has_2026_data": False,
                }
            )
            continue
        years = set(pd.DatetimeIndex(hist.index).year.tolist())
        coverage_rows.append(
            {
                "symbol": symbol,
                "first_date": pd.Timestamp(hist.index.min()).date().isoformat(),
                "last_date": pd.Timestamp(hist.index.max()).date().isoformat(),
                "row_count": int(len(hist)),
                "has_2024_data": 2024 in years,
                "has_2025_data": 2025 in years,
                "has_2026_data": 2026 in years,
            }
        )
    try:
        benchmark_loaded = radar_client.load_history(
            benchmark_symbol,
            lookback_days=config.lookback_days,
            db_path=config.db_path,
            force=config.force_refresh,
            cache_only=config.cache_only,
        )
    except TypeError:
        benchmark_loaded = radar_client.load_history(
            benchmark_symbol,
            lookback_days=config.lookback_days,
            db_path=config.db_path,
            force=config.force_refresh,
        )
    benchmark_history = _normalize_history(benchmark_loaded)
    if config.as_of_date:
        effective_as_of = pd.Timestamp(_parse_date(config.as_of_date))
    else:
        latest_candidates = [_history_latest_date(benchmark_history)]
        latest_candidates.extend(_history_latest_date(hist) for hist in symbol_histories.values())
        latest_candidates = [x for x in latest_candidates if x is not None]
        effective_as_of = min(latest_candidates) if latest_candidates else None

    # Initialize regime detector
    regime_detector = MarketRegimeDetector(db_path=config.db_path)
    regime_detector._xu100_df = regime_detector.load_xu100_data(client=radar_client)

    holdings_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        period_end_filter = window.period_end
        period_end_display = window.period_end
        if (
            idx == len(windows) - 1
            and not config.period_ends
            and effective_as_of is not None
            and period_end_filter <= effective_as_of.date()
        ):
            period_end_filter = (effective_as_of + pd.Timedelta(days=1)).date()
            period_end_display = effective_as_of.date()

        # Look-ahead-free regime detection
        regime_info = regime_detector.detect_regime(window.period_start.isoformat(), client=radar_client)
        if regime_info["return_20d_pct"] > 1.5:
            selected_config_name = "current_config"
            min_close_pos = 0.60
        else:
            selected_config_name = "relaxed_strong_close"
            min_close_pos = 0.50
            
        period_config = replace(config, min_close_position=min_close_pos)

        mask = (signals["signal_date"] >= pd.Timestamp(window.period_start)) & (
            signals["signal_date"] < pd.Timestamp(period_end_filter)
        )
        scoped = signals.loc[mask].copy()
        
        current_cfg = replace(config, min_close_position=0.60)
        relaxed_cfg = replace(config, min_close_position=0.50)
        
        current_metrics = _compute_config_metrics(
            scoped_signals=scoped,
            strategy="volume_spike_strict",
            cfg=current_cfg,
            symbol_histories=symbol_histories,
            benchmark_history=benchmark_history,
            effective_as_of=effective_as_of,
        )
        current_alpha_30d = current_metrics.get("alpha_30d")
        
        relaxed_metrics = _compute_config_metrics(
            scoped_signals=scoped,
            strategy="volume_spike_strict",
            cfg=relaxed_cfg,
            symbol_histories=symbol_histories,
            benchmark_history=benchmark_history,
            effective_as_of=effective_as_of,
        )
        relaxed_alpha_30d = relaxed_metrics.get("alpha_30d")
        
        regime_alpha_30d = (
            current_alpha_30d if selected_config_name == "current_config" else relaxed_alpha_30d
        )
        regime_signal_count = (
            int(current_metrics.get("signal_count", 0))
            if selected_config_name == "current_config"
            else int(relaxed_metrics.get("signal_count", 0))
        )
        current_signal_count = int(current_metrics.get("signal_count", 0))
        relaxed_signal_count = int(relaxed_metrics.get("signal_count", 0))
        
        current_net_alpha_30d = None
        relaxed_net_alpha_30d = None
        regime_net_alpha_30d = None
        cost = float(config.transaction_cost_pct)
        if current_alpha_30d is not None:
            current_net_alpha_30d = current_alpha_30d - cost
        if relaxed_alpha_30d is not None:
            relaxed_net_alpha_30d = relaxed_alpha_30d - cost
        if regime_alpha_30d is not None:
            regime_net_alpha_30d = regime_alpha_30d - cost

        if current_alpha_30d is None and relaxed_alpha_30d is None:
            winner = "None"
        elif current_alpha_30d is None:
            winner = "relaxed_strong_close"
        elif relaxed_alpha_30d is None:
            winner = "current_config"
        else:
            if current_alpha_30d > relaxed_alpha_30d:
                winner = "current_config"
            elif relaxed_alpha_30d > current_alpha_30d:
                winner = "relaxed_strong_close"
            else:
                winner = "Same"

        regime_minus_current = (
            (regime_alpha_30d - current_alpha_30d)
            if regime_alpha_30d is not None and current_alpha_30d is not None
            else None
        )
        regime_minus_relaxed = (
            (regime_alpha_30d - relaxed_alpha_30d)
            if regime_alpha_30d is not None and relaxed_alpha_30d is not None
            else None
        )
        same_alpha_allclose = (
            current_alpha_30d is not None
            and relaxed_alpha_30d is not None
            and abs(float(current_alpha_30d) - float(relaxed_alpha_30d)) < 1e-12
        )
        same_signal_set = (
            current_metrics.get("signature") is not None
            and current_metrics.get("signature") == relaxed_metrics.get("signature")
        )
        same_alpha_diagnosis = (
            "same_signal_set"
            if same_alpha_allclose and same_signal_set
            else ("different_signal_set_or_calc_issue" if same_alpha_allclose else "not_equal_alpha")
        )

        comparison_rows.append(
            {
                "period": window.period_start.isoformat(),
                "regime_label": regime_info["regime_label"],
                "xu100_return_20d_pct": regime_info.get("return_20d_pct"),
                "selected_config": selected_config_name,
                "current_signal_count": current_signal_count,
                "relaxed_signal_count": relaxed_signal_count,
                "regime_signal_count": regime_signal_count,
                "current_alpha_30d": current_alpha_30d,
                "relaxed_alpha_30d": relaxed_alpha_30d,
                "regime_alpha_30d": regime_alpha_30d,
                "current_net_alpha_30d": current_net_alpha_30d,
                "relaxed_net_alpha_30d": relaxed_net_alpha_30d,
                "regime_net_alpha_30d": regime_net_alpha_30d,
                "transaction_cost_pct": float(config.transaction_cost_pct),
                "winner_config": winner,
                "regime_minus_current": regime_minus_current,
                "regime_minus_relaxed": regime_minus_relaxed,
                "same_alpha_allclose": same_alpha_allclose,
                "same_signal_set": same_signal_set,
                "same_alpha_diagnosis": same_alpha_diagnosis,
            }
        )
        
        # Apply the quality filter dynamically for this period
        scoped = _apply_volume_spike_quality_filter(scoped, period_config)
        symbols_with_any_price_data = sum(1 for hist in symbol_histories.values() if not hist.empty)
        symbols_with_required_lookback = 0
        symbols_with_valid_volume = 0
        symbols_with_valid_ohlcv = 0
        for hist in symbol_histories.values():
            if hist.empty:
                continue
            hist_scope = hist.loc[hist.index < pd.Timestamp(period_end_filter)]
            if len(hist_scope) >= 30:
                symbols_with_required_lookback += 1
            if "volume" in hist_scope.columns and pd.to_numeric(hist_scope["volume"], errors="coerce").dropna().shape[0] > 0:
                symbols_with_valid_volume += 1
            required_cols = {"open", "high", "low", "close", "volume"}
            if required_cols.issubset(set(hist_scope.columns)) and len(hist_scope) > 0:
                symbols_with_valid_ohlcv += 1

        before_spike = int(scoped.loc[scoped["strategy"] == "volume_spike_strict", "symbol"].nunique())
        if config.active_volume_spike_quality:
            q_scoped = scoped.loc[(scoped["strategy"] == "volume_spike_strict") & (scoped["filter_passed"] == True)]  # noqa: E712
            after_quality_symbols = int(q_scoped["symbol"].nunique())
        else:
            after_quality_symbols = before_spike
        diagnostics_rows.append(
            {
                "period_start": window.period_start.isoformat(),
                "period_end": period_end_display.isoformat(),
                "universe_symbol_count": int(raw.scan_summary.get("universe_symbol_count", 0)),
                "symbols_with_any_price_data": symbols_with_any_price_data,
                "symbols_with_required_lookback": symbols_with_required_lookback,
                "symbols_with_valid_volume": symbols_with_valid_volume,
                "symbols_with_valid_ohlcv": symbols_with_valid_ohlcv,
                "symbols_with_volume_spike_strict_before_quality": before_spike,
                "symbols_after_quality_filter": after_quality_symbols,
                "filtered_out_count": max(0, before_spike - after_quality_symbols),
            }
        )
        if config.active_volume_spike_quality:
            failed_set = scoped.loc[(scoped["strategy"] == "volume_spike_strict") & (scoped["filter_passed"] == False)]  # noqa: E712
            totals = {
                "failed_rsi_14": 0,
                "failed_turnover": 0,
                "failed_avg_turnover_20d": 0,
                "failed_above_ma20": 0,
                "failed_return_5d_pct": 0,
                "failed_return_10d_pct": 0,
                "failed_strong_close": 0,
                "failed_missing_metric": 0,
            }
            for value in failed_set.get("failed_reasons", pd.Series(dtype=str)).tolist():
                flags = _reason_flags(str(value))
                for key, num in flags.items():
                    totals[key] += num
            reason_rows.append(
                {
                    "period_start": window.period_start.isoformat(),
                    "period_end": period_end_display.isoformat(),
                    **totals,
                }
            )
        for strategy in strategies:
            strat = scoped.loc[scoped["strategy"] == strategy].copy()
            before_quality_count = int(len(strat))
            if config.active_volume_spike_quality and strategy == "volume_spike_strict":
                strat = strat.loc[strat["filter_passed"] == True].copy()  # noqa: E712
            after_quality_count = int(len(strat))
            basket = _pick_basket(strat, config.basket_mode)
            if effective_as_of is not None:
                basket = _add_to_latest_metrics(
                    basket=basket,
                    symbol_histories=symbol_histories,
                    benchmark_history=benchmark_history,
                    effective_as_of_date=effective_as_of,
                )
            if basket.empty:
                period_rows.append(
                    _build_period_strategy_summary(
                        window.period_start,
                        period_end_display,
                        effective_as_of,
                        universe_symbol,
                        benchmark_symbol,
                        int(raw.scan_summary.get("universe_symbol_count", 0)),
                        strategy,
                        config.basket_mode,
                        basket,
                        quality_filter_enabled=bool(config.active_volume_spike_quality and strategy == "volume_spike_strict"),
                        quality_config=config,
                        signal_count_before_quality_filter=before_quality_count,
                        signal_count_after_quality_filter=after_quality_count,
                        regime_label=regime_info["regime_label"],
                        xu100_return_20d_pct=regime_info["return_20d_pct"],
                        xu100_close=regime_info["close"],
                        xu100_ma50=regime_info["ma50"],
                        xu100_ma200=regime_info["ma200"],
                        selected_config=selected_config_name,
                        transaction_cost_pct=float(config.transaction_cost_pct),
                    )
                )
                continue

            export = basket.copy()
            export["period_start"] = window.period_start.isoformat()
            export["period_end"] = period_end_display.isoformat()
            export["effective_as_of_date"] = None if effective_as_of is None else effective_as_of.date().isoformat()
            export["universe"] = universe_symbol
            export["benchmark"] = benchmark_symbol
            export["universe_symbol_count"] = int(raw.scan_summary.get("universe_symbol_count", 0))
            export["benchmark_symbol"] = benchmark_symbol
            export["quality_filter_enabled"] = bool(config.active_volume_spike_quality and strategy == "volume_spike_strict")
            export["min_last_turnover_try"] = float(config.min_last_turnover_try)
            export["min_avg_turnover_20d_try"] = float(config.min_avg_turnover_20d_try)
            export["max_rsi_14"] = float(config.max_rsi_14)
            export["max_return_5d_pct"] = float(config.max_return_5d_pct)
            export["max_return_10d_pct"] = float(config.max_return_10d_pct)
            export["require_strong_close"] = bool(config.require_strong_close)
            export["min_close_position"] = float(config.min_close_position)
            export["min_above_ma20_ratio"] = float(config.min_above_ma20_ratio)
            export["signal_count_before_quality_filter"] = before_quality_count
            export["signal_count_after_quality_filter"] = after_quality_count
            export["filtered_out_count"] = max(0, before_quality_count - after_quality_count)
            export["entry_price"] = export.get("entry_close")
            export["exit_price_15d"] = export.get("exit_15d_close")
            export["exit_price_30d"] = export.get("exit_30d_close")
            export["current_price"] = export.get("exit_price_to_current")
            export["exit_date_15d"] = export.get("exit_15d_date")
            export["exit_date_30d"] = export.get("exit_30d_date")
            export["selected_config"] = selected_config_name
            export = export.rename(columns={"volume_ratio_20d": "rvol_20d"})
            holdings_rows.extend(export.to_dict("records"))

            period_rows.append(
                _build_period_strategy_summary(
                    window.period_start,
                    period_end_display,
                    effective_as_of,
                    universe_symbol,
                    benchmark_symbol,
                    int(raw.scan_summary.get("universe_symbol_count", 0)),
                    strategy,
                    config.basket_mode,
                    basket,
                    quality_filter_enabled=bool(config.active_volume_spike_quality and strategy == "volume_spike_strict"),
                    quality_config=config,
                    signal_count_before_quality_filter=before_quality_count,
                    signal_count_after_quality_filter=after_quality_count,
                    regime_label=regime_info["regime_label"],
                    xu100_return_20d_pct=regime_info["return_20d_pct"],
                    xu100_close=regime_info["close"],
                    xu100_ma50=regime_info["ma50"],
                    xu100_ma200=regime_info["ma200"],
                    selected_config=selected_config_name,
                    transaction_cost_pct=float(config.transaction_cost_pct),
                )
            )

    holdings = pd.DataFrame(holdings_rows)
    period_summary = pd.DataFrame(period_rows)
    stability = _build_stability_summary(period_summary)
    diagnostics = pd.DataFrame(diagnostics_rows)
    reason_summary = pd.DataFrame(reason_rows)
    coverage_summary = pd.DataFrame(coverage_rows)
    coverage_warning = False
    if not diagnostics.empty:
        early = diagnostics.loc[diagnostics["period_start"] < "2026-01-01"].copy()
        if not early.empty:
            coverage_warning = bool((early["symbols_with_required_lookback"] < (0.3 * early["universe_symbol_count"])).any())
    comparison_df = pd.DataFrame(comparison_rows)
    return PeriodBacktestResult(
        holdings=holdings,
        period_strategy_summary=period_summary.sort_values(["period_start", "strategy"]) if not period_summary.empty else period_summary,
        strategy_stability_summary=stability,
        period_diagnostics_summary=diagnostics,
        quality_filter_reason_summary=reason_summary,
        data_coverage_summary=coverage_summary,
        run_summary={
            "period_count": len(windows),
            "basket_mode": config.basket_mode,
            "universe": universe_symbol,
            "benchmark": benchmark_symbol,
            "universe_symbol_count": int(raw.scan_summary.get("universe_symbol_count", 0)),
            "benchmark_symbol": benchmark_symbol,
            "effective_as_of_date": None if effective_as_of is None else effective_as_of.date().isoformat(),
            "entry_convention": "entry_price is signal-day close; 15d/30d exits are trading-day offsets from signal date",
            "quality_filter_enabled": config.active_volume_spike_quality,
            "coverage_warning_2024_2025": coverage_warning,
            "period_windows": [asdict(item) for item in windows],
            "scan_summary": raw.scan_summary,
            "failed_symbols": raw.failed_symbols,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        regime_config_comparison=comparison_df,
    )


def write_period_outputs(result: PeriodBacktestResult, config: PeriodBacktestConfig) -> dict[str, str]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdings_path = out_dir / "period_basket_holdings.csv"
    period_summary_path = out_dir / "period_strategy_summary.csv"
    stability_path = out_dir / "strategy_stability_summary.csv"
    diagnostics_path = out_dir / "period_diagnostics_summary.csv"
    quality_reason_path = out_dir / "quality_filter_reason_summary.csv"
    coverage_path = out_dir / "data_coverage_summary.csv"
    config_path = out_dir / "backtest_period_config.json"
    comparison_path = out_dir / "regime_config_comparison.csv"

    holdings_columns = [
        "period_start",
        "period_end",
        "effective_as_of_date",
        "universe",
        "benchmark",
        "universe_symbol_count",
        "benchmark_symbol",
        "strategy",
        "quality_filter_enabled",
        "min_last_turnover_try",
        "min_avg_turnover_20d_try",
        "max_rsi_14",
        "max_return_5d_pct",
        "max_return_10d_pct",
        "require_strong_close",
        "min_close_position",
        "min_above_ma20_ratio",
        "signal_count_before_quality_filter",
        "signal_count_after_quality_filter",
        "filtered_out_count",
        "symbol",
        "signal_date",
        "entry_price",
        "exit_date_15d",
        "exit_15d_close",
        "return_15d",
        "benchmark_return_15d",
        "alpha_15d",
        "beat_xu100_15d",
        "exit_date_30d",
        "exit_30d_close",
        "return_30d",
        "benchmark_return_30d",
        "alpha_30d",
        "beat_xu100_30d",
        "exit_date_to_current",
        "exit_price_to_current",
        "return_to_current",
        "benchmark_return_to_current",
        "alpha_to_current",
        "beat_xu100_to_current",
        "turnover",
        "avg_turnover_20d",
        "ma20",
        "rsi_14",
        "return_5d_pct",
        "return_10d_pct",
        "filter_passed",
        "filter_reasons",
        "failed_reasons",
        "selected_config",
        "interest_score",
        "rvol_20d",
        "turnover_ratio_20d",
        "close_position",
        "daily_return_pct",
        "near_20d_high_pct",
        "above_ma20",
        "above_ma50",
    ]

    period_summary_columns = [
        "period",
        "regime_label",
        "xu100_return_20d_pct",
        "xu100_close",
        "xu100_ma50",
        "xu100_ma200",
        "selected_config",
        "transaction_cost_pct",
        "signal_count",
        "basket_return_15d",
        "basket_return_30d",
        "basket_alpha_30d",
        "net_basket_alpha_30d",
    ]

    stability_columns = [
        "strategy",
        "period_count",
        "active_period_count",
        "empty_period_count",
        "total_signal_count",
        "total_unique_symbol_count",
        "avg_basket_return_15d",
        "avg_basket_return_30d",
        "avg_basket_return_to_current",
        "median_basket_return_15d",
        "median_basket_return_30d",
        "median_basket_return_to_current",
        "avg_basket_alpha_15d",
        "avg_basket_alpha_30d",
        "avg_basket_alpha_to_current",
        "median_basket_alpha_15d",
        "median_basket_alpha_30d",
        "median_basket_alpha_to_current",
        "positive_period_rate_15d",
        "positive_period_rate_30d",
        "positive_period_rate_to_current",
        "beat_period_rate_15d",
        "beat_period_rate_30d",
        "beat_period_rate_to_current",
        "best_period_to_current",
        "worst_period_to_current",
        "consistency_score",
        "outlier_period_count",
        "low_sample_period_count",
    ]

    holdings_df = result.holdings.copy()
    holdings_df = holdings_df.rename(
        columns={
            "entry_close": "entry_price",
            "exit_15d_date": "exit_date_15d",
            "exit_30d_date": "exit_date_30d",
        }
    )
    for col in holdings_columns:
        if col not in holdings_df.columns:
            holdings_df[col] = pd.NA
    holdings_df = holdings_df[holdings_columns]
    holdings_df.to_csv(holdings_path, index=False)

    period_df = result.period_strategy_summary.copy()
    for col in period_summary_columns:
        if col not in period_df.columns:
            period_df[col] = pd.NA
    period_df = period_df[period_summary_columns]
    period_df.to_csv(period_summary_path, index=False)

    stability_df = result.strategy_stability_summary.copy()
    for col in stability_columns:
        if col not in stability_df.columns:
            stability_df[col] = pd.NA
    stability_df = stability_df[stability_columns]
    stability_df.to_csv(stability_path, index=False)
    result.period_diagnostics_summary.to_csv(diagnostics_path, index=False)
    result.quality_filter_reason_summary.to_csv(quality_reason_path, index=False)
    result.data_coverage_summary.to_csv(coverage_path, index=False)
    if result.regime_config_comparison is not None:
        result.regime_config_comparison.to_csv(comparison_path, index=False)

    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "run_summary": result.run_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    outputs = {
        "period_strategy_summary.csv": str(period_summary_path),
        "period_basket_holdings.csv": str(holdings_path),
        "strategy_stability_summary.csv": str(stability_path),
        "period_diagnostics_summary.csv": str(diagnostics_path),
        "quality_filter_reason_summary.csv": str(quality_reason_path),
        "data_coverage_summary.csv": str(coverage_path),
        "backtest_period_config.json": str(config_path),
    }
    if result.regime_config_comparison is not None:
        outputs["regime_config_comparison.csv"] = str(comparison_path)
    return outputs

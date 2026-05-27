from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from market_radar.radar_engine import compute_production_scores


DEFAULT_OUTPUT_DIR = "/data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"
DEFAULT_DB_PATH = "/data/market_radar_cache.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", default="2026-04-30")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--top-n", default="20,30,50,75,all")
    parser.add_argument("--ranking-score", default="liquidity_safe_score")
    return parser.parse_args()


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


def _read_features(output_dir: Path) -> pd.DataFrame:
    pq = output_dir / "candidate_features.parquet"
    csv = output_dir / "candidate_features.csv"
    path = pq if pq.exists() else csv
    if not path.exists():
        raise FileNotFoundError(f"candidate features missing: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _relaxed_mask(df: pd.DataFrame) -> pd.Series:
    ma_ok = df["above_ma20"] | df["close"].ge(df["ma20"])
    return (
        df["turnover"].ge(10_000_000.0)
        & df["avg_turnover_20d"].ge(10_000_000.0)
        & ma_ok.fillna(False)
        & df["rsi_14"].le(78.0)
        & df["return_5d_pct"].le(35.0)
        & df["return_10d_pct"].le(60.0)
        & df["close_position"].ge(0.50)
    ).fillna(False)


def _load_cache_payload(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol=?", (symbol,)).fetchone()
    if not row:
        return []
    payload = json.loads(row[0])
    records = payload.get("records") or []
    return records if isinstance(records, list) else []


def _series_points(records: list[dict[str, Any]], as_of_date: str) -> dict[str, Any]:
    if not records:
        return {
            "entry_close_price": None,
            "latest_close_price": None,
            "as_of_date_used": None,
            "latest_date": None,
        }
    frame = pd.DataFrame.from_records(records)
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        return {
            "entry_close_price": None,
            "latest_close_price": None,
            "as_of_date_used": None,
            "latest_date": None,
        }
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        return {
            "entry_close_price": None,
            "latest_close_price": None,
            "as_of_date_used": None,
            "latest_date": None,
        }
    as_of_dt = pd.Timestamp(as_of_date)
    as_of_slice = frame.loc[frame["date"] <= as_of_dt]
    if as_of_slice.empty:
        return {
            "entry_close_price": None,
            "latest_close_price": None,
            "as_of_date_used": None,
            "latest_date": frame.iloc[-1]["date"].date().isoformat(),
        }
    entry = as_of_slice.iloc[-1]
    latest = frame.iloc[-1]
    return {
        "entry_close_price": float(entry["close"]),
        "latest_close_price": float(latest["close"]),
        "as_of_date_used": entry["date"].date().isoformat(),
        "latest_date": latest["date"].date().isoformat(),
    }


def _return_pct(entry: float | None, latest: float | None) -> float | None:
    if entry in (None, 0) or latest is None:
        return None
    return (latest / entry - 1.0) * 100.0


def _score_bucket(rank: int) -> str:
    if rank <= 20:
        return "top20"
    if rank <= 30:
        return "top30"
    if rank <= 50:
        return "top50"
    if rank <= 75:
        return "top75"
    return "watchlist"


def run_analysis(output_dir: Path, db_path: str, as_of_date: str, ranking_score: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_features(output_dir)
    f = raw.copy()
    f = f.loc[f["strategy"].astype(str) == "volume_spike_strict"].copy()
    f["signal_date"] = pd.to_datetime(f["signal_date"], errors="coerce").dt.tz_localize(None)
    f = _to_numeric(
        f,
        [
            "close",
            "turnover",
            "avg_turnover_20d",
            "volume_ratio_20d",
            "turnover_ratio_20d",
            "ma20",
            "rsi_14",
            "return_5d_pct",
            "return_10d_pct",
            "close_position",
        ],
    )
    f["above_ma20"] = _to_bool(f.get("above_ma20", pd.Series(False, index=f.index)))
    asof = pd.Timestamp(as_of_date)
    f = f.loc[f["signal_date"] <= asof].copy()
    f = f.sort_values(["symbol", "signal_date"]).drop_duplicates(subset=["symbol"], keep="last")
    f = f.loc[_relaxed_mask(f)].copy()

    with sqlite3.connect(db_path) as conn:
        xu_records = _load_cache_payload(conn, "XU100")
        xu_points = _series_points(xu_records, as_of_date)
        xu_ret = _return_pct(xu_points["entry_close_price"], xu_points["latest_close_price"])
        asof_ret_20d = None
        if xu_records:
            xu = pd.DataFrame.from_records(xu_records)
            xu["date"] = pd.to_datetime(xu["date"], errors="coerce").dt.tz_localize(None)
            xu["close"] = pd.to_numeric(xu["close"], errors="coerce")
            xu = xu.dropna(subset=["date", "close"]).sort_values("date")
            if not xu.empty:
                cur = xu.loc[xu["date"] <= asof]
                if not cur.empty:
                    end_idx = cur.index[-1]
                    pos = xu.index.get_loc(end_idx)
                    if isinstance(pos, int) and pos >= 20:
                        close_now = float(xu.loc[end_idx, "close"])
                        close_prev = float(xu.iloc[pos - 20]["close"])
                        if close_prev != 0:
                            asof_ret_20d = (close_now / close_prev - 1.0) * 100.0

        market_supportive = bool(asof_ret_20d is not None and asof_ret_20d > 1.0)
        regime_label = "Bull" if market_supportive else "WeakOrNeutral"

        score_rows: list[dict[str, Any]] = []
        for _, row in f.iterrows():
            metrics = {
                "volume_ratio_20d": row.get("volume_ratio_20d"),
                "turnover_ratio_20d": row.get("turnover_ratio_20d"),
                "avg_turnover_20d": row.get("avg_turnover_20d"),
                "turnover_try": row.get("turnover"),
                "close_position": row.get("close_position"),
                "rsi_14": row.get("rsi_14"),
                "return_5d_pct": row.get("return_5d_pct"),
                "return_10d_pct": row.get("return_10d_pct"),
                "close": row.get("close"),
                "ma20": row.get("ma20"),
            }
            scores = compute_production_scores(metrics, {"market_supportive": market_supportive})
            score_rows.append(scores)
        score_df = pd.DataFrame(score_rows, index=f.index)
        f = pd.concat([f, score_df], axis=1)
        f["production_score"] = f[ranking_score]
        f = f.sort_values(["production_score", "signal_date", "symbol"], ascending=[False, True, True]).reset_index(drop=True)
        f["production_rank"] = f.index + 1
        f["score_bucket"] = f["production_rank"].apply(_score_bucket)
        f["regime_label"] = regime_label
        f["xu100_return_20d_pct"] = asof_ret_20d
        f["market_supportive"] = market_supportive

        points = []
        for symbol in f["symbol"].astype(str).tolist():
            recs = _load_cache_payload(conn, symbol)
            points.append(_series_points(recs, as_of_date))
        p = pd.DataFrame(points, index=f.index)
        f = pd.concat([f, p], axis=1)

    f["return_from_entry_close_pct"] = f.apply(
        lambda r: _return_pct(r.get("entry_close_price"), r.get("latest_close_price")),
        axis=1,
    )
    f["xu100_return_same_period_pct"] = xu_ret
    f["alpha_vs_xu100_pct"] = f["return_from_entry_close_pct"] - f["xu100_return_same_period_pct"]
    latest_date = f["latest_date"].dropna().max() if "latest_date" in f.columns and not f.empty else xu_points.get("latest_date")

    out = f[
        [
            "symbol",
            "production_rank",
            "score_bucket",
            "balanced_score",
            "momentum_quality_score",
            "liquidity_safe_score",
            "production_score",
            "regime_label",
            "xu100_return_20d_pct",
            "market_supportive",
            "entry_close_price",
            "latest_close_price",
            "return_from_entry_close_pct",
            "xu100_return_same_period_pct",
            "alpha_vs_xu100_pct",
        ]
    ].copy()
    out["as_of_date"] = as_of_date
    out["latest_date"] = latest_date
    out = out[
        [
            "symbol",
            "as_of_date",
            "latest_date",
            "production_rank",
            "score_bucket",
            "balanced_score",
            "momentum_quality_score",
            "liquidity_safe_score",
            "production_score",
            "regime_label",
            "xu100_return_20d_pct",
            "market_supportive",
            "entry_close_price",
            "latest_close_price",
            "return_from_entry_close_pct",
            "xu100_return_same_period_pct",
            "alpha_vs_xu100_pct",
        ]
    ]

    def _bucket_stats(df: pd.DataFrame) -> dict[str, float | None]:
        r = pd.to_numeric(df["return_from_entry_close_pct"], errors="coerce").dropna()
        a = pd.to_numeric(df["alpha_vs_xu100_pct"], errors="coerce").dropna()
        return {
            "avg_return_pct": float(r.mean()) if not r.empty else None,
            "median_return_pct": float(r.median()) if not r.empty else None,
            "avg_alpha_vs_xu100_pct": float(a.mean()) if not a.empty else None,
        }

    s_all = _bucket_stats(out)
    s20 = _bucket_stats(out.loc[out["production_rank"] <= 20])
    s30 = _bucket_stats(out.loc[out["production_rank"] <= 30])
    s50 = _bucket_stats(out.loc[out["production_rank"] <= 50])
    s75 = _bucket_stats(out.loc[out["production_rank"] <= 75])
    ret_series = pd.to_numeric(out["return_from_entry_close_pct"], errors="coerce")
    best_idx = ret_series.idxmax() if not ret_series.dropna().empty else None
    worst_idx = ret_series.idxmin() if not ret_series.dropna().empty else None
    summary = {
        "as_of_date": as_of_date,
        "latest_date": latest_date,
        "total_signal_count": int(len(out)),
        "benchmark_xu100_return_pct": xu_ret,
        "all_avg_return_pct": s_all["avg_return_pct"],
        "all_median_return_pct": s_all["median_return_pct"],
        "all_avg_alpha_vs_xu100_pct": s_all["avg_alpha_vs_xu100_pct"],
        "top20_avg_return_pct": s20["avg_return_pct"],
        "top20_median_return_pct": s20["median_return_pct"],
        "top20_avg_alpha_vs_xu100_pct": s20["avg_alpha_vs_xu100_pct"],
        "top30_avg_return_pct": s30["avg_return_pct"],
        "top30_median_return_pct": s30["median_return_pct"],
        "top30_avg_alpha_vs_xu100_pct": s30["avg_alpha_vs_xu100_pct"],
        "top50_avg_return_pct": s50["avg_return_pct"],
        "top50_median_return_pct": s50["median_return_pct"],
        "top50_avg_alpha_vs_xu100_pct": s50["avg_alpha_vs_xu100_pct"],
        "top75_avg_return_pct": s75["avg_return_pct"],
        "top75_median_return_pct": s75["median_return_pct"],
        "top75_avg_alpha_vs_xu100_pct": s75["avg_alpha_vs_xu100_pct"],
        "best_symbol": None if best_idx is None else str(out.loc[best_idx, "symbol"]),
        "worst_symbol": None if worst_idx is None else str(out.loc[worst_idx, "symbol"]),
        "positive_symbol_count": int((ret_series > 0).sum()),
        "negative_symbol_count": int((ret_series < 0).sum()),
    }
    return out, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    detail, summary = run_analysis(
        output_dir=out_dir,
        db_path=args.db_path,
        as_of_date=args.as_of_date,
        ranking_score=args.ranking_score,
    )
    out_csv = out_dir / f"asof_scan_performance_{args.as_of_date}.csv"
    out_json = out_dir / f"asof_scan_performance_{args.as_of_date}_summary.json"
    detail.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print({"output_csv": str(out_csv), "summary_json": str(out_json), "row_count": len(detail)})


if __name__ == "__main__":
    main()


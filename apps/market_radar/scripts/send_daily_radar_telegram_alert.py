from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from market_radar.telegram_notifier import send_telegram_message

DEFAULT_LIVE_PILOT_DIR = "data/backtest_outputs/market_radar_live_pilot"
DEFAULT_STRATEGY = "adaptive_v1_cash_no_buy"
DEFAULT_PRIORITY_FILTER = "special_strict_top10"
DEFAULT_TOP_N = 30


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_radar.telegram_alert")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--live-pilot-dir", default=DEFAULT_LIVE_PILOT_DIR)
    p.add_argument("--date")
    p.add_argument("--kap-source", default=os.getenv("MARKET_RADAR_KAP_SOURCE", "mcp"))
    p.add_argument("--strategy", default=os.getenv("MARKET_RADAR_STRATEGY", DEFAULT_STRATEGY))
    p.add_argument("--priority-filter", default=os.getenv("MARKET_RADAR_PRIORITY_FILTER", DEFAULT_PRIORITY_FILTER))
    p.add_argument("--top-n", type=int, default=int(os.getenv("MARKET_RADAR_TOP_N", str(DEFAULT_TOP_N))))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--send-telegram", action="store_true")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--timezone", default=os.getenv("MARKET_RADAR_TIMEZONE", "Europe/Istanbul"))
    return p.parse_args()


def _latest_csv(live_pilot_dir: Path, date_text: str | None = None) -> Path:
    if date_text:
        p = live_pilot_dir / f"daily_radar_final_{date_text}.csv"
        if not p.exists():
            raise FileNotFoundError(f"daily radar file not found for date {date_text}: {p}")
        return p
    files = sorted(live_pilot_dir.glob("daily_radar_final_*.csv"))
    if not files:
        raise FileNotFoundError(f"no daily radar csv found under {live_pilot_dir}")
    return files[-1]


def _bool_text(v: bool) -> str:
    return "yes" if bool(v) else "no"


def _clean_text(v: object, fallback: str = "yok") -> str:
    if v is None:
        return fallback
    if pd.isna(v):
        return fallback
    text = str(v).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _format_message(df: pd.DataFrame, *, strategy: str, priority_filter: str, top_n: int, now_text: str) -> str:
    if df.empty:
        return (
            f"📊 BIST Radar | {now_text} | {strategy}\n\n"
            "Bugün radar adayı yok / adaptive no-buy modunda.\n\n"
            "Not: Bu mesaj otomatik işlem sinyali değil, manuel karar destek çıktısıdır."
        )

    weak_score = df["weak_score"].iloc[0] if "weak_score" in df.columns else ""
    regime_label = df["regime_label"].iloc[0] if "regime_label" in df.columns else ""
    selected = df["selected_strategy"].iloc[0] if "selected_strategy" in df.columns else strategy
    weak_val = pd.to_numeric(pd.Series([weak_score]), errors="coerce").iloc[0]
    market_supportive = "yes" if pd.notna(weak_val) and float(weak_val) <= 1 else "no"

    ranked = df.copy()
    for col in ["quality_threshold_score", "liquidity_safe_score", "production_rank"]:
        if col not in ranked.columns:
            ranked[col] = pd.NA
    ranked = ranked.sort_values(
        ["quality_threshold_score", "liquidity_safe_score", "production_rank", "symbol"],
        ascending=[False, False, True, True],
        na_position="last",
    ).head(top_n)

    lines: list[str] = []
    lines.append(f"📊 BIST Radar | {now_text} | {strategy} | {priority_filter}")
    lines.append("")
    lines.append(
        "Rejim: "
        f"weak_score={weak_score} | selected={selected} | market_supportive={market_supportive} | regime={regime_label}"
    )
    lines.append(f"Priority filter: {priority_filter} | TopN={top_n}")
    lines.append("")
    lines.append("Adaylar (quality_score DESC):")

    detailed_count = min(10, len(ranked))
    for idx, row in enumerate(ranked.head(detailed_count).itertuples(index=False), start=1):
        symbol = getattr(row, "symbol", "")
        q = getattr(row, "quality_threshold_score", None)
        q_txt = "-" if pd.isna(q) else f"{float(q):.1f}"
        liq = getattr(row, "liquidity_safe_score", None)
        liq_txt = "-" if pd.isna(liq) else f"{float(liq):.1f}"
        mom = getattr(row, "momentum_quality_score", None)
        mom_txt = "-" if pd.isna(mom) else f"{float(mom):.1f}"
        strict = _bool_text(bool(getattr(row, "passes_special_strict", False)))
        kap_count = getattr(row, "kap_event_count_7d", None)
        kap_count_txt = "0" if pd.isna(kap_count) else str(int(float(kap_count)))
        kap_label = _clean_text(getattr(row, "kap_sentiment_label", ""), fallback="yok")
        kap_date = _clean_text(getattr(row, "kap_latest_date", ""), fallback="yok")
        kap_summary = _clean_text(getattr(row, "kap_summary_short", ""), fallback="yok")
        lines.append(
            f"{idx}) #{getattr(row, 'production_rank', '-')}/{symbol} | q={q_txt} | lq={liq_txt} | mom={mom_txt} | strict={strict}"
        )
        lines.append(
            f"   weak={weak_score} | KAP7g={kap_count_txt} | KAP={kap_label} | son={kap_date} | {kap_summary}"
        )

    remaining = len(ranked) - detailed_count
    if remaining > 0:
        lines.append(f"... ve {remaining} aday daha (Top{top_n} içinde).")

    lines.append("")
    lines.append("Not: Bu mesaj otomatik işlem sinyali değil, manuel karar destek çıktısıdır.")
    return "\n".join(lines)


def _no_candidate_message(now_text: str, strategy: str) -> str:
    return (
        f"📊 BIST Radar | {now_text} | {strategy}\n\n"
        "Bugün radar adayı yok / adaptive no-buy modunda.\n\n"
        "Not: Bu mesaj otomatik işlem sinyali değil, manuel karar destek çıktısıdır."
    )


def main() -> None:
    args = _parse_args()
    _load_env_file(Path(args.env_file))
    logger = _setup_logger(Path("logs/telegram_radar.log"))

    tz = ZoneInfo(args.timezone)
    now = datetime.now(tz)
    now_text = now.strftime("%Y-%m-%d %H:%M")

    csv_path = _latest_csv(Path(args.live_pilot_dir), args.date)
    df = pd.read_csv(csv_path)

    # Keep deterministic report order and top-n cut
    msg = _format_message(
        df,
        strategy=args.strategy,
        priority_filter=args.priority_filter,
        top_n=int(args.top_n),
        now_text=now_text,
    ) if not df.empty else _no_candidate_message(now_text, args.strategy)

    logger.info(
        "run_time=%s strategy=%s priority_filter=%s top_n=%s csv=%s candidate_count=%s dry_run=%s send_telegram=%s kap_source=%s",
        now_text,
        args.strategy,
        args.priority_filter,
        args.top_n,
        csv_path,
        len(df),
        args.dry_run,
        args.send_telegram,
        args.kap_source,
    )

    if args.dry_run or not args.send_telegram:
        print(msg)
        logger.info("telegram_send=skipped reason=dry_run_or_flag_missing")
        return

    send_result = send_telegram_message(msg)
    logger.info("telegram_send_result=%s", send_result)
    print(send_result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_radar.telegram_notifier import send_telegram_message

REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_PILOT_DIR = REPO_ROOT / "data/backtest_outputs/market_radar_live_pilot"
STATE_PATH = LIVE_PILOT_DIR / "telegram_sent_state.json"
LOG_PATH = REPO_ROOT / "logs/telegram_radar.log"


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def _preload_env_file_from_argv() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", default=".env")
    args, _ = parser.parse_known_args(sys.argv[1:])
    _load_env_file(Path(args.env_file))


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_radar.telegram_runner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def _parse_alert_times(raw: str) -> list[str]:
    out = []
    for item in str(raw).split(","):
        t = item.strip()
        if t:
            out.append(t)
    return out or ["09:00", "12:00", "15:00", "18:00"]


def _nearest_slot(now: datetime, alert_times: list[str]) -> str:
    hhmm = now.strftime("%H:%M")
    if hhmm in alert_times:
        return hhmm
    return hhmm


def _latest_completed_signal_date(candidate_features_path: str, before_date: str) -> str | None:
    root = Path(candidate_features_path)
    if not root.exists():
        return None
    files = [root] if root.is_file() else sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.csv"))
    latest: str | None = None
    for path in files:
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=["signal_date"])
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, usecols=["signal_date"])
            else:
                continue
        except Exception:
            continue
        if frame.empty or "signal_date" not in frame.columns:
            continue
        dates = frame["signal_date"].astype(str).str.slice(0, 10)
        dates = dates[dates <= before_date]
        if dates.empty:
            continue
        candidate = str(dates.max())
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def _default_target_date(now: datetime, candidate_features_path: str) -> str:
    # Daily radar is close-based. Without an explicit --date, use only completed days.
    before_date = (now.date() - timedelta(days=1)).isoformat()
    return _latest_completed_signal_date(candidate_features_path, before_date) or before_date


def _state_key(date_text: str, slot: str, strategy: str, priority_filter: str, top_n: int) -> str:
    return f"{date_text}|{slot}|{strategy}|{priority_filter}|{top_n}"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "sent_keys": [],
            "sent_by_day": {},
            "symbol_last_sent": {},
            "symbol_history": {},
            "events_by_day": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sent_keys"), list):
            data.setdefault("sent_by_day", {})
            data.setdefault("symbol_last_sent", {})
            data.setdefault("symbol_history", {})
            data.setdefault("events_by_day", {})
            return data
    except Exception:
        pass
    return {
        "sent_keys": [],
        "sent_by_day": {},
        "symbol_last_sent": {},
        "symbol_history": {},
        "events_by_day": {},
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_token(token: str | None) -> str:
    if not token:
        return "missing"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", default=".env")
    p.add_argument("--date")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--send-telegram", action="store_true")
    p.add_argument("--force-send", action="store_true")
    p.add_argument("--state-path", default=str(STATE_PATH))
    p.add_argument("--live-pilot-dir", default=str(LIVE_PILOT_DIR))
    p.add_argument("--strategy", default=os.getenv("MARKET_RADAR_STRATEGY", "adaptive_v1_cash_no_buy"))
    p.add_argument("--priority-filter", default=os.getenv("MARKET_RADAR_PRIORITY_FILTER", "special_strict_top10"))
    p.add_argument("--top-n", type=int, default=int(os.getenv("MARKET_RADAR_TOP_N", "30")))
    p.add_argument("--max-symbols", type=int, default=int(os.getenv("MARKET_RADAR_ALERT_MAX_SYMBOLS", "10")))
    p.add_argument("--intraday-mode", default=os.getenv("MARKET_RADAR_INTRADAY_MODE", "new_only"))
    p.add_argument(
        "--dedupe-lookback-trading-days",
        type=int,
        default=int(os.getenv("MARKET_RADAR_DEDUPE_LOOKBACK_TRADING_DAYS", "3")),
    )
    p.add_argument("--include-full-top30", action=argparse.BooleanOptionalAction, default=os.getenv("MARKET_RADAR_INCLUDE_FULL_TOP30", "false").lower() == "true")
    p.add_argument("--kap-summary-max-chars", type=int, default=int(os.getenv("MARKET_RADAR_KAP_SUMMARY_MAX_CHARS", "80")))
    p.add_argument("--repeat-if-rank-improves-by", type=int, default=int(os.getenv("MARKET_RADAR_REPEAT_IF_RANK_IMPROVES_BY", "5")))
    p.add_argument("--repeat-if-quality-improves-by", type=float, default=float(os.getenv("MARKET_RADAR_REPEAT_IF_QUALITY_IMPROVES_BY", "5")))
    p.add_argument("--kap-source", default=os.getenv("MARKET_RADAR_KAP_SOURCE", "mcp"))
    p.add_argument("--timezone", default=os.getenv("MARKET_RADAR_TIMEZONE", "Europe/Istanbul"))
    p.add_argument("--candidate-features-path", default=os.getenv("MARKET_RADAR_CANDIDATE_FEATURES_PATH", "data/backtest_outputs/market_radar_candidate_backfill_final"))
    p.add_argument("--output-dir", default=os.getenv("MARKET_RADAR_OUTPUT_DIR", "data/backtest_outputs/period_runs_volume_spike_quality_2024_backfilled"))
    p.add_argument("--db-path", default=os.getenv("MARKET_RADAR_DB_PATH", "data/market_radar_cache.sqlite"))
    return p.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_text(v: Any, fallback: str = "") -> str:
    if v is None or pd.isna(v):
        return fallback
    text = str(v).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _truncate_text(text: str, max_chars: int) -> str:
    t = _safe_text(text, "")
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)] + "..."


def _kap_note_from_category(text: str) -> str:
    folded = text.casefold()
    mappings = [
        (("kredi derecelendirme",), "Kredi derecelendirme bildirimi"),
        (("pay dışı", "pay disi", "sermaye piyasası aracı", "sermaye piyasasi araci"), "Sermaye piyasası aracı işlemi"),
        (("özel durum", "ozel durum"), "Özel durum açıklaması"),
        (("finansal rapor", "finansal tablo"), "Finansal rapor bildirimi"),
        (("temettü", "temettu", "kar payı", "kar payi"), "Kar payı/temettü bildirimi"),
        (("ihale", "sözleşme", "sozlesme", "iş ilişkisi", "is iliskisi"), "İhale/sözleşme bildirimi"),
        (("genel kurul",), "Genel kurul bildirimi"),
        (("geri alım", "geri alim"), "Pay geri alım bildirimi"),
    ]
    for needles, label in mappings:
        if any(needle in folded for needle in needles):
            return label
    return ""


def _kap_category_from_text(text: str) -> str:
    category_matches = re.findall(r"\(([^()]{3,160})\)", text)
    if category_matches:
        return re.sub(r"\s+", " ", category_matches[-1]).strip(" )")
    if "(" in text:
        tail = text.rsplit("(", 1)[-1]
        tail = re.sub(r"\s+", " ", tail).strip(" )")
        if 3 <= len(tail) <= 160:
            return tail
    return ""


def _clean_raw_kap_summary(text: str) -> str:
    summary = _safe_text(text, "")
    if not summary:
        return ""
    first_event = summary.split("|", 1)[0].strip()
    first_event = re.sub(r"^\d{4}-\d{2}-\d{2}\s*:\s*", "", first_event).strip()
    first_event = re.sub(r"^KAP\s*-\s*", "", first_event, flags=re.IGNORECASE).strip()

    category = _kap_category_from_text(first_event)
    if category:
        note = _kap_note_from_category(category)
        return note or category
    note = _kap_note_from_category(first_event)
    if note:
        return note

    without_symbol_block = re.sub(r"\*{2,}.*?\*{2,}", "", first_event).strip(" -")
    without_symbol_block = re.sub(r"\s+", " ", without_symbol_block).strip()
    if without_symbol_block and without_symbol_block != first_event:
        return without_symbol_block
    return "KAP bildirimi"


def _clean_kap_note(row: pd.Series, max_chars: int) -> str:
    summary = _safe_text(row.get("kap_summary_short"), "")
    if summary.lower() in {"recent_kap_none", "kap_fetch_failed", "unknown", "yok"}:
        summary = ""
    if summary:
        summary = _clean_raw_kap_summary(summary)
    if not summary:
        summary = _safe_text(row.get("kap_event_types"), "")
        if summary.casefold() in {"kap", "unknown", "recent_kap_none", "kap_fetch_failed"}:
            summary = ""
    if summary:
        category_note = _kap_note_from_category(summary)
        if category_note:
            summary = category_note
    return _truncate_text(summary, max_chars)


def _trading_dates_from_live_pilot(live_pilot_dir: Path, upto_date: str) -> list[str]:
    dates: list[str] = []
    for p in sorted(live_pilot_dir.glob("daily_radar_final_*.csv")):
        stem = p.stem.replace("daily_radar_final_", "")
        if len(stem) == 10 and stem <= upto_date:
            dates.append(stem)
    return sorted(set(dates))


def _recent_trading_days(trading_dates: list[str], target_date: str, lookback: int) -> list[str]:
    if target_date not in trading_dates:
        return trading_dates[-lookback:]
    idx = trading_dates.index(target_date)
    start = max(0, idx - lookback)
    return trading_dates[start:idx]


def _sent_on_day(state: dict[str, Any], date_text: str) -> set[str]:
    return set(state.get("sent_by_day", {}).get(date_text, []))


def _sent_in_recent_days(state: dict[str, Any], symbol: str, days: list[str]) -> bool:
    history = state.get("symbol_history", {}).get(symbol, [])
    return any(day in history for day in days)


def _should_repeat_symbol(
    symbol: str,
    current: dict[str, Any],
    prev: dict[str, Any] | None,
    *,
    rank_improve_by: int,
    quality_improve_by: float,
) -> tuple[bool, str]:
    if prev is None:
        return True, "new_symbol"
    prev_rank = _safe_int(prev.get("rank"), 10**6)
    cur_rank = _safe_int(current.get("rank"), 10**6)
    if prev_rank > cur_rank and (prev_rank - cur_rank) >= int(rank_improve_by):
        return True, "rank_improved"
    prev_q = _safe_float(prev.get("quality"), 0.0)
    cur_q = _safe_float(current.get("quality"), 0.0)
    if (cur_q - prev_q) >= float(quality_improve_by):
        return True, "quality_improved"
    if (not bool(prev.get("strict", False))) and bool(current.get("strict", False)):
        return True, "strict_promoted"
    prev_kap = _safe_text(prev.get("kap_sentiment"), "yok").lower()
    cur_kap = _safe_text(current.get("kap_sentiment"), "yok").lower()
    if cur_kap in {"positive", "caution"} and prev_kap != cur_kap:
        return True, "kap_sentiment_changed"
    if _safe_int(prev.get("rank"), 10**6) > 10 and _safe_int(current.get("rank"), 10**6) <= 10:
        return True, "entered_top10"
    return False, "duplicate_hidden"


def _candidate_meta(row: pd.Series, slot: str, date_text: str, kap_summary_max_chars: int) -> dict[str, Any]:
    return {
        "symbol": _safe_text(row.get("symbol"), ""),
        "date": date_text,
        "time": slot,
        "rank": _safe_int(row.get("production_rank"), 10**6),
        "quality": _safe_float(row.get("quality_threshold_score"), 0.0),
        "strict": bool(row.get("passes_special_strict", False)),
        "kap_sentiment": _safe_text(row.get("kap_sentiment_label"), "yok").lower() or "yok",
        "kap_latest_date": _safe_text(row.get("kap_latest_date"), "yok") or "yok",
        "kap_note": _clean_kap_note(row, kap_summary_max_chars),
        "liquidity": _safe_float(row.get("liquidity_safe_score"), 0.0),
        "momentum": _safe_float(row.get("momentum_quality_score"), 0.0),
        "kap_count": _safe_int(row.get("kap_event_count_7d"), 0),
    }


def _select_symbols_for_alert(
    df: pd.DataFrame,
    *,
    state: dict[str, Any],
    date_text: str,
    slot: str,
    live_pilot_dir: Path,
    top_n: int,
    max_symbols: int,
    lookback_trading_days: int,
    rank_improve_by: int,
    quality_improve_by: float,
    kap_summary_max_chars: int,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    ranked = df.copy()
    for col in ["quality_threshold_score", "liquidity_safe_score", "production_rank", "symbol"]:
        if col not in ranked.columns:
            ranked[col] = pd.NA
    ranked = ranked.sort_values(
        ["quality_threshold_score", "liquidity_safe_score", "production_rank", "symbol"],
        ascending=[False, False, True, True],
        na_position="last",
    ).head(top_n)

    same_day_symbols = _sent_on_day(state, date_text)
    trading_dates = _trading_dates_from_live_pilot(live_pilot_dir, date_text)
    recent_days = _recent_trading_days(trading_dates, date_text, lookback_trading_days)
    last_map: dict[str, Any] = state.get("symbol_last_sent", {})

    selected: list[dict[str, Any]] = []
    hidden_meta: list[dict[str, Any]] = []
    hidden = 0
    for _, row in ranked.iterrows():
        meta = _candidate_meta(row, slot, date_text, kap_summary_max_chars)
        sym = meta["symbol"]
        if not sym:
            hidden += 1
            continue
        if sym in same_day_symbols:
            hidden += 1
            hidden_meta.append(meta)
            continue
        prev = last_map.get(sym)
        recently_sent = _sent_in_recent_days(state, sym, recent_days)
        if not recently_sent:
            meta["event_reason"] = "new_or_stale"
            selected.append(meta)
        else:
            allow, reason = _should_repeat_symbol(
                sym,
                meta,
                prev,
                rank_improve_by=rank_improve_by,
                quality_improve_by=quality_improve_by,
            )
            if allow:
                meta["event_reason"] = reason
                selected.append(meta)
            else:
                hidden += 1
                hidden_meta.append(meta)
    selected = selected[:max_symbols]
    hidden += max(0, len(ranked) - len(selected) - hidden)
    hidden_sorted = sorted(
        hidden_meta,
        key=lambda x: (-_safe_float(x.get("quality"), 0.0), _safe_int(x.get("rank"), 10**6), _safe_text(x.get("symbol"), "")),
    )
    hidden_symbols = [_safe_text(item.get("symbol"), "") for item in hidden_sorted if _safe_text(item.get("symbol"), "")]
    return selected, hidden, hidden_symbols


def _market_supportive_text(weak_score: Any) -> str:
    w = _safe_int(weak_score, 99)
    return "destekleyici" if w <= 1 else "zayıf"


def _build_alert_message(
    *,
    df: pd.DataFrame,
    selected: list[dict[str, Any]],
    hidden_count: int,
    hidden_symbols: list[str],
    now_text: str,
    strategy: str,
    priority_filter: str,
    slot: str,
) -> str:
    weak_score = df["weak_score"].iloc[0] if "weak_score" in df.columns and not df.empty else ""
    market_txt = _market_supportive_text(weak_score)
    lines = [
        f"📊 BIST Radar | {now_text}",
        f"Strateji: {strategy} | Öncelik: {priority_filter}",
        f"Rejim: weak_score={weak_score} | piyasa={market_txt}",
        "",
    ]
    hidden_suffix = ""
    if hidden_symbols:
        hidden_suffix = " | " + ", ".join(hidden_symbols[:8])
    if slot == "18:00":
        lines.append(f"Günün en güçlü {min(5, len(selected))} adayı:")
        for i, meta in enumerate(selected[:5], start=1):
            lines.append(
                f"{i}) {meta['symbol']} | q={meta['quality']:.1f} | rank={meta['rank']} | strict={'yes' if meta['strict'] else 'no'} | KAP={meta['kap_sentiment']}"
            )
            if meta["kap_sentiment"] not in {"unknown", "yok"} and _safe_text(meta.get("kap_note"), ""):
                lines.append(f"   Not: {meta['kap_note']}")
    elif not selected:
        lines.append("Yeni/önemli aday yok.")
        lines.append(f"Gizlenen tekrarlar: {hidden_count}{hidden_suffix}")
        lines.append("Detay: CSV/MD dosyasında.")
        lines.append("Not: Otomatik işlem sinyali değildir.")
        return "\n".join(lines)
    else:
        lines.append(f"Yeni / önemli adaylar: {len(selected)}")
        lines.append("")
        for i, meta in enumerate(selected, start=1):
            lines.append(
                f"{i}) {meta['symbol']} | q={meta['quality']:.1f} | rank={meta['rank']} | strict={'yes' if meta['strict'] else 'no'} | KAP={meta['kap_sentiment']}"
            )
            if meta["kap_sentiment"] not in {"unknown", "yok"} and _safe_text(meta.get("kap_note"), ""):
                lines.append(f"   Not: {meta['kap_note']}")
        lines.append("")
        lines.append(f"Gizlenen tekrarlar: {hidden_count}{hidden_suffix}")
    lines.append("Detay: CSV/MD dosyasında.")
    lines.append("Not: Otomatik işlem sinyali değildir.")
    return "\n".join(lines)


def _update_state_with_sent(
    state: dict[str, Any],
    *,
    date_text: str,
    key: str,
    selected: list[dict[str, Any]],
) -> None:
    sent_keys = state.setdefault("sent_keys", [])
    sent_keys.append(key)
    if len(sent_keys) > 500:
        state["sent_keys"] = sent_keys[-500:]

    sent_by_day = state.setdefault("sent_by_day", {})
    day_list = sent_by_day.setdefault(date_text, [])
    symbol_last_sent = state.setdefault("symbol_last_sent", {})
    symbol_history = state.setdefault("symbol_history", {})
    events_by_day = state.setdefault("events_by_day", {})
    day_events = events_by_day.setdefault(date_text, [])

    for meta in selected:
        sym = meta["symbol"]
        if sym not in day_list:
            day_list.append(sym)
        symbol_last_sent[sym] = {
            "date": meta["date"],
            "time": meta["time"],
            "rank": meta["rank"],
            "quality": meta["quality"],
            "strict": meta["strict"],
            "kap_sentiment": meta["kap_sentiment"],
            "kap_latest_date": meta["kap_latest_date"],
        }
        hist = symbol_history.setdefault(sym, [])
        if meta["date"] not in hist:
            hist.append(meta["date"])
            hist[:] = sorted(hist)[-20:]
        day_events.append(
            {
                "symbol": sym,
                "time": meta["time"],
                "reason": meta.get("event_reason", ""),
                "rank": meta["rank"],
                "quality": meta["quality"],
                "strict": meta["strict"],
                "kap_sentiment": meta["kap_sentiment"],
            }
        )


def _refresh_daily_radar(
    *,
    target_date: str,
    live_pilot_dir: Path,
    kap_source: str,
    candidate_features_path: str,
    output_dir: str,
    db_path: str,
) -> Path:
    analyze_path = REPO_ROOT / "apps/market_radar/scripts/analyze_radar_rolling_hold20_portfolio.py"
    mod = _load_script_module(analyze_path, "analyze_radar_rolling_hold20_portfolio")
    start_date = f"{target_date[:4]}-01-01"
    daily_all, summary_df, price_cache, _, _ = mod._prepare_inputs(  # noqa: SLF001
        output_dir,
        candidate_features_path,
        db_path,
        start_date,
        target_date,
        True,
    )
    for symbol in price_cache:
        price_cache[symbol] = price_cache[symbol].set_index("date_str", drop=False)
    regime = mod._build_market_regime_map(daily_all, summary_df, price_cache)  # noqa: SLF001
    fetcher = mod.BorsaMcpKapFetcher("docker compose run --rm -T borsa-mcp borsa-mcp") if kap_source == "mcp" else None
    try:
        frame = mod._build_live_pilot_daily_radar(  # noqa: SLF001
            daily_all,
            regime,
            as_of_date=target_date,
            kap_lookback_days=7,
            kap_fetcher=fetcher,
        )
    finally:
        if fetcher is not None:
            fetcher.close()
    live_pilot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = live_pilot_dir / f"daily_radar_final_{target_date}.csv"
    md_path = live_pilot_dir / f"daily_radar_final_{target_date}.md"
    frame.to_csv(csv_path, index=False)
    mod._write_live_pilot_markdown(frame, md_path)  # noqa: SLF001
    return csv_path


def _build_message(csv_path: Path, strategy: str, priority_filter: str, top_n: int, timezone_name: str) -> str:
    send_path = REPO_ROOT / "apps/market_radar/scripts/send_daily_radar_telegram_alert.py"
    send_mod = _load_script_module(send_path, "send_daily_radar_telegram_alert")
    df = pd.read_csv(csv_path)
    now_text = datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M")
    if df.empty:
        return send_mod._no_candidate_message(now_text, strategy)  # noqa: SLF001
    return send_mod._format_message(  # noqa: SLF001
        df,
        strategy=strategy,
        priority_filter=priority_filter,
        top_n=top_n,
        now_text=now_text,
    )


def main() -> None:
    _preload_env_file_from_argv()
    args = _parse_args()
    _load_env_file(Path(args.env_file))
    logger = _setup_logger()
    tz = ZoneInfo(args.timezone)
    now = datetime.now(tz)
    target_date = args.date or _default_target_date(now, args.candidate_features_path)
    alert_times = _parse_alert_times(os.getenv("MARKET_RADAR_ALERT_TIMES", "09:00,12:00,15:00,18:00"))
    slot = _nearest_slot(now, alert_times)
    state_path = Path(args.state_path)
    state = _load_state(state_path)
    key = _state_key(target_date, slot, args.strategy, args.priority_filter, int(args.top_n))

    token_masked = _mask_token(os.getenv("TELEGRAM_BOT_TOKEN"))
    logger.info(
        "runner_start date=%s slot=%s strategy=%s priority_filter=%s top_n=%s kap_source=%s dry_run=%s send_telegram=%s token=%s",
        target_date,
        slot,
        args.strategy,
        args.priority_filter,
        args.top_n,
        args.kap_source,
        args.dry_run,
        args.send_telegram,
        token_masked,
    )

    csv_path = _refresh_daily_radar(
        target_date=target_date,
        live_pilot_dir=Path(args.live_pilot_dir),
        kap_source=args.kap_source,
        candidate_features_path=args.candidate_features_path,
        output_dir=args.output_dir,
        db_path=args.db_path,
    )
    df = pd.read_csv(csv_path)
    logger.info("radar_refreshed csv=%s candidate_count=%s", csv_path, len(df))
    selected, hidden_count, hidden_symbols = _select_symbols_for_alert(
        df,
        state=state,
        date_text=target_date,
        slot=slot,
        live_pilot_dir=Path(args.live_pilot_dir),
        top_n=int(args.top_n),
        max_symbols=int(args.max_symbols),
        lookback_trading_days=int(args.dedupe_lookback_trading_days),
        rank_improve_by=int(args.repeat_if_rank_improves_by),
        quality_improve_by=float(args.repeat_if_quality_improves_by),
        kap_summary_max_chars=int(args.kap_summary_max_chars),
    )
    message = _build_alert_message(
        df=df,
        selected=selected,
        hidden_count=hidden_count,
        hidden_symbols=hidden_symbols,
        now_text=now.strftime("%Y-%m-%d %H:%M"),
        strategy=args.strategy,
        priority_filter=args.priority_filter,
        slot=slot,
    )

    if key in state.get("sent_keys", []) and not args.force_send:
        logger.info("telegram_send=skipped reason=duplicate key=%s", key)
        print({"ok": True, "skipped": True, "reason": "duplicate", "key": key})
        return

    if args.dry_run or not args.send_telegram:
        logger.info("telegram_send=skipped reason=dry_run_or_flag_missing key=%s", key)
        print(message)
        return

    send_result = send_telegram_message(message)
    logger.info("telegram_send_result=%s key=%s", send_result, key)
    if send_result.get("ok"):
        _update_state_with_sent(
            state,
            date_text=target_date,
            key=key,
            selected=selected,
        )
        _save_state(state_path, state)
    print(send_result)


if __name__ == "__main__":
    main()

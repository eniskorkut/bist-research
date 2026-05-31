from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_daily_radar_and_send_telegram.py"
    spec = importlib.util.spec_from_file_location("run_daily_radar_and_send_telegram", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mk_args(tmp_path: Path, *, dry_run: bool, send_telegram: bool) -> argparse.Namespace:
    return argparse.Namespace(
        env_file=str(tmp_path / ".env"),
        date="2026-05-29",
        dry_run=dry_run,
        send_telegram=send_telegram,
        force_send=False,
        state_path=str(tmp_path / "state.json"),
        live_pilot_dir=str(tmp_path),
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        top_n=30,
        max_symbols=10,
        intraday_mode="new_only",
        dedupe_lookback_trading_days=3,
        include_full_top30=False,
        kap_summary_max_chars=80,
        repeat_if_rank_improves_by=5,
        repeat_if_quality_improves_by=5.0,
        kap_source="none",
        timezone="Europe/Istanbul",
        candidate_features_path="x",
        output_dir="x",
        db_path="x",
    )


def test_same_day_duplicate_filtered() -> None:
    m = _load_module()
    df = pd.DataFrame([{"symbol": "AAA", "quality_threshold_score": 60.0, "production_rank": 12, "passes_special_strict": True}])
    state = {
        "sent_by_day": {"2026-05-29": ["AAA"]},
        "symbol_history": {"AAA": ["2026-05-29"]},
        "symbol_last_sent": {"AAA": {"rank": 20, "quality": 55.0, "strict": True, "kap_sentiment": "neutral"}},
    }
    selected, hidden, hidden_symbols = m._select_symbols_for_alert(  # noqa: SLF001
        df,
        state=state,
        date_text="2026-05-29",
        slot="12:00",
        live_pilot_dir=Path("/tmp"),
        top_n=30,
        max_symbols=10,
        lookback_trading_days=3,
        rank_improve_by=5,
        quality_improve_by=5.0,
        kap_summary_max_chars=80,
    )
    assert selected == []
    assert hidden >= 1
    assert "AAA" in hidden_symbols


def test_last3days_filtered_and_rank_improvement_allows(tmp_path: Path) -> None:
    m = _load_module()
    live_dir = tmp_path
    for d in ["2026-05-27", "2026-05-28", "2026-05-29"]:
        (live_dir / f"daily_radar_final_{d}.csv").write_text("symbol\nAAA\n", encoding="utf-8")
    state = {
        "sent_by_day": {"2026-05-28": ["AAA"], "2026-05-29": []},
        "symbol_history": {"AAA": ["2026-05-28"]},
        "symbol_last_sent": {"AAA": {"rank": 20, "quality": 50.0, "strict": False, "kap_sentiment": "neutral"}},
    }
    df = pd.DataFrame([{"symbol": "AAA", "quality_threshold_score": 52.0, "production_rank": 18, "passes_special_strict": False}])
    selected, hidden, _ = m._select_symbols_for_alert(  # noqa: SLF001
        df,
        state=state,
        date_text="2026-05-29",
        slot="12:00",
        live_pilot_dir=live_dir,
        top_n=30,
        max_symbols=10,
        lookback_trading_days=3,
        rank_improve_by=5,
        quality_improve_by=5.0,
        kap_summary_max_chars=80,
    )
    assert selected == []
    assert hidden >= 1

    df2 = pd.DataFrame([{"symbol": "AAA", "quality_threshold_score": 52.0, "production_rank": 10, "passes_special_strict": False}])
    selected2, _, _ = m._select_symbols_for_alert(  # noqa: SLF001
        df2,
        state=state,
        date_text="2026-05-29",
        slot="12:00",
        live_pilot_dir=live_dir,
        top_n=30,
        max_symbols=10,
        lookback_trading_days=3,
        rank_improve_by=5,
        quality_improve_by=5.0,
        kap_summary_max_chars=80,
    )
    assert len(selected2) == 1
    assert selected2[0]["event_reason"] == "rank_improved"


def test_repeat_on_quality_or_kap_or_strict_or_top10() -> None:
    m = _load_module()
    prev = {"rank": 15, "quality": 50.0, "strict": False, "kap_sentiment": "neutral"}
    cur = {"rank": 15, "quality": 56.0, "strict": False, "kap_sentiment": "neutral"}
    ok, reason = m._should_repeat_symbol("AAA", cur, prev, rank_improve_by=5, quality_improve_by=5.0)  # noqa: SLF001
    assert ok and reason == "quality_improved"

    cur2 = {"rank": 15, "quality": 50.0, "strict": True, "kap_sentiment": "neutral"}
    ok2, reason2 = m._should_repeat_symbol("AAA", cur2, prev, rank_improve_by=5, quality_improve_by=5.0)  # noqa: SLF001
    assert ok2 and reason2 == "strict_promoted"

    cur3 = {"rank": 15, "quality": 50.0, "strict": False, "kap_sentiment": "positive"}
    ok3, reason3 = m._should_repeat_symbol("AAA", cur3, prev, rank_improve_by=5, quality_improve_by=5.0)  # noqa: SLF001
    assert ok3 and reason3 == "kap_sentiment_changed"

    prev4 = {"rank": 25, "quality": 50.0, "strict": True, "kap_sentiment": "neutral"}
    cur4 = {"rank": 9, "quality": 50.0, "strict": True, "kap_sentiment": "neutral"}
    ok4, reason4 = m._should_repeat_symbol("AAA", cur4, prev4, rank_improve_by=5, quality_improve_by=5.0)  # noqa: SLF001
    assert ok4 and reason4 in {"rank_improved", "entered_top10"}


def test_kap_summary_truncate_and_no_long_raw() -> None:
    m = _load_module()
    long = "2026-05-25: KAP - ***ECILC ** ECZYT*** EIS ECZACIBASI ILAC SINAI VE FINANSAL YATIRIMLAR A.S. (Kredi Derecelendirme) " + ("A" * 200)
    row = pd.Series({"kap_summary_short": long})
    note = m._clean_kap_note(row, 80)  # noqa: SLF001
    assert len(note) <= 80
    assert note == "Kredi derecelendirme bildirimi"
    assert "***" not in note
    assert "ECILC" not in note


def test_kap_summary_prefers_open_parenthesis_category() -> None:
    m = _load_module()
    raw = "2026-05-22: KAP - ***YYLGD*** YAYLA AGRO GIDA SANAYI VE TICARET A.S. (Bagimsiz Denetim Kurulusunun Belirlen..."
    note = m._clean_kap_note(pd.Series({"kap_summary_short": raw}), 80)  # noqa: SLF001
    assert note == "Bagimsiz Denetim Kurulusunun Belirlen..."
    assert "YAYLA" not in note


def test_kap_summary_maps_new_business_relation() -> None:
    m = _load_module()
    raw = "2026-05-22: KAP - ***ONCSM*** ONCOSEM ONKOLOJIK SISTEMLER SANAYI VE TICARET A.S. (Yeni Is Iliskisi)"
    note = m._clean_kap_note(pd.Series({"kap_summary_short": raw}), 80)  # noqa: SLF001
    assert note == "İhale/sözleşme bildirimi"


def test_empty_new_candidates_message() -> None:
    m = _load_module()
    df = pd.DataFrame([{"weak_score": 0}])
    msg = m._build_alert_message(  # noqa: SLF001
        df=df,
        selected=[],
        hidden_count=30,
        hidden_symbols=["ECZYT", "EUPWR", "CEOEM"],
        now_text="2026-05-29 12:00",
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        slot="12:00",
    )
    assert "Yeni/önemli aday yok." in msg
    assert "Gizlenen tekrarlar: 30 | ECZYT, EUPWR, CEOEM" in msg


def test_1800_summary_mode_message() -> None:
    m = _load_module()
    df = pd.DataFrame([{"weak_score": 0}])
    selected = [
        {"symbol": "AAA", "quality": 61.2, "rank": 5, "strict": True, "kap_sentiment": "positive", "kap_note": "Yeni pozitif KAP"},
        {"symbol": "BBB", "quality": 58.1, "rank": 8, "strict": False, "kap_sentiment": "neutral", "kap_note": "Ozel durum"},
    ]
    msg = m._build_alert_message(  # noqa: SLF001
        df=df,
        selected=selected,
        hidden_count=0,
        hidden_symbols=[],
        now_text="2026-05-29 18:00",
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        slot="18:00",
    )
    assert "Günün en güçlü" in msg
    assert "AAA | q=61.2" in msg


def test_hidden_repeat_symbol_summary_order_and_max8() -> None:
    m = _load_module()
    df = pd.DataFrame([{"weak_score": 0}])
    hidden_symbols = ["ECZYT", "EUPWR", "CEOEM", "BIGTK", "ADEL", "KORDS", "AGHOL", "NIBAS", "PNSUT"]
    msg = m._build_alert_message(  # noqa: SLF001
        df=df,
        selected=[{"symbol": "AAA", "quality": 60.0, "rank": 1, "strict": True, "kap_sentiment": "positive", "kap_note": "x"}],
        hidden_count=26,
        hidden_symbols=hidden_symbols,
        now_text="2026-05-29 12:00",
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        slot="12:00",
    )
    assert "Gizlenen tekrarlar: 26 | ECZYT, EUPWR, CEOEM, BIGTK, ADEL, KORDS, AGHOL, NIBAS" in msg
    assert "PNSUT" not in msg


def test_kap_unknown_hides_note_line() -> None:
    m = _load_module()
    df = pd.DataFrame([{"weak_score": 0}])
    msg = m._build_alert_message(  # noqa: SLF001
        df=df,
        selected=[{"symbol": "AAA", "quality": 55.0, "rank": 3, "strict": True, "kap_sentiment": "unknown", "kap_note": "KAP notu yok"}],
        hidden_count=0,
        hidden_symbols=[],
        now_text="2026-05-29 12:00",
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        slot="12:00",
    )
    assert "AAA | q=55.0" in msg
    assert "Not: KAP notu yok" not in msg


def test_dry_run_skips_telegram_send(monkeypatch, tmp_path: Path) -> None:
    m = _load_module()
    state_path = tmp_path / "state.json"
    csv_path = tmp_path / "daily_radar_final_2026-05-29.csv"
    pd.DataFrame([{"symbol": "AAA", "quality_threshold_score": 60.0, "production_rank": 1}]).to_csv(csv_path, index=False)

    args = _mk_args(tmp_path, dry_run=True, send_telegram=True)
    monkeypatch.setattr(m, "_parse_args", lambda: args)
    monkeypatch.setattr(m, "_nearest_slot", lambda now, alert_times: "09:00")
    monkeypatch.setattr(m, "_refresh_daily_radar", lambda **kwargs: csv_path)
    called = {"sent": False}
    monkeypatch.setattr(m, "send_telegram_message", lambda text: called.__setitem__("sent", True) or {"ok": True})

    m.main()
    assert called["sent"] is False
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data.get("sent_keys", []) == []


def test_default_target_date_uses_completed_signal_day(tmp_path: Path) -> None:
    m = _load_module()
    path = tmp_path / "candidate_features.csv"
    pd.DataFrame(
        [
            {"signal_date": "2026-05-29", "symbol": "AAA"},
            {"signal_date": "2026-06-01", "symbol": "BBB"},
        ]
    ).to_csv(path, index=False)

    now = datetime.fromisoformat("2026-06-01T09:00:00")
    assert m._default_target_date(now, str(path)) == "2026-05-29"  # noqa: SLF001

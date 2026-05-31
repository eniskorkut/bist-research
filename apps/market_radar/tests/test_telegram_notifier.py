from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_radar.telegram_notifier import send_telegram_message, split_telegram_message


class _FakeResp:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_split_message_chunks_long_text() -> None:
    text = "A" * 8005
    parts = split_telegram_message(text, max_len=3900)
    assert len(parts) == 3
    assert sum(len(p) for p in parts) == len(text)
    assert all(len(p) <= 3900 for p in parts)


def test_send_telegram_message_missing_config_safe(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    result = send_telegram_message("hello", bot_token=None, chat_id=None, enabled=True)
    assert result["ok"] is False
    assert result["reason"] == "missing_config"


def test_send_telegram_message_success_and_chunking() -> None:
    calls = []

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        calls.append((req.full_url, req.data.decode("utf-8")))
        return _FakeResp({"ok": True, "result": {"message_id": 1}})

    with patch("market_radar.telegram_notifier.request.urlopen", side_effect=_fake_urlopen):
        result = send_telegram_message(
            "B" * 8000,
            bot_token="t",
            chat_id="1",
            enabled=True,
        )
    assert result["ok"] is True
    assert result["sent_parts"] == 3
    assert len(calls) == 3


def test_no_candidate_message_text_from_script() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "send_daily_radar_telegram_alert.py"
    spec = importlib.util.spec_from_file_location("send_daily_radar_telegram_alert", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)

    df = pd.DataFrame()
    msg = module._format_message(
        df,
        strategy="adaptive_v1_cash_no_buy",
        priority_filter="special_strict_top10",
        top_n=30,
        now_text="2026-05-29 09:00",
    )
    assert "adaptive no-buy modunda" in msg

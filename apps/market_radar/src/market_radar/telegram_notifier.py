from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any
from urllib import parse, request

logger = logging.getLogger(__name__)

MAX_TELEGRAM_TEXT_LEN = 3900


def split_telegram_message(text: str, max_len: int = MAX_TELEGRAM_TEXT_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    lines = text.splitlines()
    current = ""
    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = line
            continue
        # Single very long line fallback
        start = 0
        while start < len(line):
            parts.append(line[start : start + max_len])
            start += max_len
        current = ""
    if current:
        parts.append(current)
    return parts


def _post_send_message(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict[str, str] = {"chat_id": str(chat_id), "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    encoded = parse.urlencode(payload).encode("utf-8")
    req = request.Request(url=url, data=encoded, method="POST")
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _post_send_message_curl(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        str(int(timeout_seconds)),
        "-X",
        "POST",
        url,
        "-d",
        f"chat_id={chat_id}",
        "--data-urlencode",
        f"text={text}",
    ]
    if parse_mode:
        cmd.extend(["-d", f"parse_mode={parse_mode}"])
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def send_telegram_message(
    text: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str | None = None,
    enabled: bool | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    is_enabled = enabled if enabled is not None else os.getenv("MARKET_RADAR_TELEGRAM_ENABLED", "true").lower() == "true"

    if not is_enabled:
        logger.info("Telegram send skipped: MARKET_RADAR_TELEGRAM_ENABLED is false")
        return {"ok": False, "sent_parts": 0, "reason": "disabled"}
    if not token or not chat:
        logger.warning("Telegram send skipped: missing bot token or chat id")
        return {"ok": False, "sent_parts": 0, "reason": "missing_config"}

    parts = split_telegram_message(text)
    sent_parts = 0
    try:
        for idx, part in enumerate(parts, start=1):
            chunk_text = part if len(parts) == 1 else f"[{idx}/{len(parts)}]\n{part}"
            try:
                data = _post_send_message(
                    token,
                    chat,
                    chunk_text,
                    parse_mode=parse_mode,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("urllib send failed, trying curl fallback: %s", exc)
                data = _post_send_message_curl(
                    token,
                    chat,
                    chunk_text,
                    parse_mode=parse_mode,
                    timeout_seconds=timeout_seconds,
                )
            if not data.get("ok"):
                logger.error("Telegram send failed on part %s: %s", idx, data)
                return {"ok": False, "sent_parts": sent_parts, "reason": "api_error", "response": data}
            sent_parts += 1
        return {"ok": True, "sent_parts": sent_parts}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram send failed: %s", exc)
        return {"ok": False, "sent_parts": sent_parts, "reason": "exception", "error": str(exc)}

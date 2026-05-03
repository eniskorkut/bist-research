from __future__ import annotations

import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque

import borsapy as bp

from .alert_store import CsvAlertStore, VolumeAlert
from .data import build_average_volume_map, index_symbols, normalize_symbols


@dataclass(frozen=True)
class AlertConfig:
    symbols: list[str]
    avg_volumes: dict[str, float]
    ratio_threshold: float = 1.5
    velocity_threshold: float = 0.05
    cooldown_seconds: int = 900
    check_interval: float = 5.0
    duration: float | None = None
    csv_path: str = "alerts.csv"
    dry_run: bool = False


@dataclass
class SymbolState:
    volume_points: Deque[tuple[float, float]] = field(default_factory=deque)
    cooldown_until: float = 0.0


class VolumeAlertRunner:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self.store = CsvAlertStore(config.csv_path)
        self.states = {symbol: SymbolState() for symbol in config.symbols}
        self._stop = False

    def request_stop(self, *_args: object) -> None:
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

        stream = bp.TradingViewStream()
        stream.connect()
        print(f"stream connected symbols={','.join(self.config.symbols)} dry_run={self.config.dry_run}", flush=True)
        try:
            for symbol in self.config.symbols:
                stream.subscribe(symbol)
                print(f"subscribed symbol={symbol} avg_20d_volume={self.config.avg_volumes[symbol]:.0f}", flush=True)

            started = time.monotonic()
            while not self._stop:
                now = time.monotonic()
                if self.config.duration is not None and now - started >= self.config.duration:
                    print("duration reached, stopping", flush=True)
                    break

                for symbol in self.config.symbols:
                    quote = stream.get_quote(symbol)
                    if not quote:
                        quote = stream.wait_for_quote(symbol, timeout=1.0)
                    if quote:
                        self.process_quote(symbol, quote, now)
                time.sleep(self.config.check_interval)
        finally:
            stream.disconnect()
            print("stream disconnected", flush=True)

    def process_quote(self, symbol: str, quote: dict, now: float) -> None:
        avg_volume = self.config.avg_volumes[symbol]
        current_volume = _float_quote(quote, "volume")
        if current_volume is None or current_volume <= 0:
            return

        state = self.states[symbol]
        state.volume_points.append((now, current_volume))
        while state.volume_points and now - state.volume_points[0][0] > 300:
            state.volume_points.popleft()

        baseline_volume = state.volume_points[0][1] if state.volume_points else current_volume
        volume_5m_delta = max(0.0, current_volume - baseline_volume)
        day_ratio = current_volume / avg_volume
        velocity_ratio = volume_5m_delta / avg_volume

        reasons: list[str] = []
        if day_ratio >= self.config.ratio_threshold:
            reasons.append(f"day_ratio>={self.config.ratio_threshold:g}")
        if velocity_ratio >= self.config.velocity_threshold:
            reasons.append(f"velocity_ratio>={self.config.velocity_threshold:g}")

        if not reasons:
            return
        if now < state.cooldown_until:
            return

        state.cooldown_until = now + self.config.cooldown_seconds
        alert = VolumeAlert(
            symbol=symbol,
            last_price=_float_quote(quote, "last"),
            current_volume=current_volume,
            avg_20d_volume=avg_volume,
            day_ratio=day_ratio,
            volume_5m_delta=volume_5m_delta,
            velocity_ratio=velocity_ratio,
            reason=";".join(reasons),
            dry_run=self.config.dry_run,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        print(_format_alert(alert), flush=True)
        self.store.append(alert)


def _float_quote(quote: dict, key: str) -> float | None:
    value = quote.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_alert(alert: VolumeAlert) -> str:
    price = "n/a" if alert.last_price is None else f"{alert.last_price:g}"
    return (
        "ALERT "
        f"symbol={alert.symbol} price={price} volume={alert.current_volume:.0f} "
        f"avg20={alert.avg_20d_volume:.0f} day_ratio={alert.day_ratio:.3f} "
        f"delta5m={alert.volume_5m_delta:.0f} velocity={alert.velocity_ratio:.3f} "
        f"reason={alert.reason} dry_run={alert.dry_run}"
    )


def build_config_from_args(args) -> AlertConfig:
    symbols = args.symbols or []
    if args.index:
        symbols = index_symbols(args.index, max_symbols=args.max_symbols)
        print(f"index {args.index.upper()} symbols={','.join(symbols)}", flush=True)
    symbols = normalize_symbols(symbols)
    if not symbols:
        raise SystemExit("symbols or --index required")

    avg_volumes, errors = build_average_volume_map(
        symbols,
        period=args.history_period,
        lookback_days=args.lookback_days,
    )
    for symbol, error in errors.items():
        print(f"history skipped symbol={symbol} error={error}", flush=True)
    runnable_symbols = [symbol for symbol in symbols if symbol in avg_volumes]
    if not runnable_symbols:
        raise SystemExit("no symbols with usable average volume")

    return AlertConfig(
        symbols=runnable_symbols,
        avg_volumes=avg_volumes,
        ratio_threshold=args.ratio_threshold,
        velocity_threshold=args.velocity_threshold,
        cooldown_seconds=args.cooldown_seconds,
        check_interval=args.check_interval,
        duration=args.duration,
        csv_path=args.csv_path,
        dry_run=args.dry_run,
    )


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)

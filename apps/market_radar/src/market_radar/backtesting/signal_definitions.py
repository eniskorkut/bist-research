from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    description: str
    min_avg_turnover_20d: float | None = None
    min_volume_ratio_20d: float | None = None
    min_turnover_ratio_20d: float | None = None
    min_daily_return_pct: float | None = None
    max_daily_return_pct: float | None = None
    min_close_position: float | None = None
    min_cmf_20: float | None = None
    require_obv_slope_5d_positive: bool = False
    require_obv_slope_20d_positive: bool = False
    min_mfi_14: float | None = None
    max_mfi_14: float | None = None
    min_accumulation_score: float | None = None
    require_above_ma20: bool = False
    min_relative_return_vs_xu100: float | None = None

    def match(self, metrics: dict[str, Any]) -> bool:
        def _num(name: str) -> float | None:
            value = metrics.get(name)
            return value if isinstance(value, (int, float)) else None

        if self.min_avg_turnover_20d is not None:
            v = _num("avg_turnover_20d")
            if v is None or v < self.min_avg_turnover_20d:
                return False
        if self.min_volume_ratio_20d is not None:
            v = _num("volume_ratio_20d")
            if v is None or v < self.min_volume_ratio_20d:
                return False
        if self.min_turnover_ratio_20d is not None:
            v = _num("turnover_ratio_20d")
            if v is None or v < self.min_turnover_ratio_20d:
                return False
        if self.min_daily_return_pct is not None:
            v = _num("daily_return_pct")
            if v is None or v < self.min_daily_return_pct:
                return False
        if self.max_daily_return_pct is not None:
            v = _num("daily_return_pct")
            if v is None or v > self.max_daily_return_pct:
                return False
        if self.min_close_position is not None:
            v = _num("close_position")
            if v is None or v < self.min_close_position:
                return False
        if self.min_cmf_20 is not None:
            v = _num("cmf_20")
            if v is None or v <= self.min_cmf_20:
                return False
        if self.require_obv_slope_5d_positive:
            v = _num("obv_slope_5d")
            if v is None or v <= 0:
                return False
        if self.require_obv_slope_20d_positive:
            v = _num("obv_slope_20d")
            if v is None or v <= 0:
                return False
        if self.min_mfi_14 is not None:
            v = _num("mfi_14")
            if v is None or v < self.min_mfi_14:
                return False
        if self.max_mfi_14 is not None:
            v = _num("mfi_14")
            if v is None or v > self.max_mfi_14:
                return False
        if self.min_accumulation_score is not None:
            v = _num("accumulation_score")
            if v is None or v < self.min_accumulation_score:
                return False
        if self.require_above_ma20:
            if metrics.get("above_ma20") is not True:
                return False
        if self.min_relative_return_vs_xu100 is not None:
            v = _num("relative_return_vs_xu100")
            if v is None or v < self.min_relative_return_vs_xu100:
                return False
        return True


SIGNAL_DEFINITIONS: dict[str, SignalDefinition] = {
    "volume_spike_strict": SignalDefinition(
        name="volume_spike_strict",
        description="Hacim patlamasi odakli",
        min_volume_ratio_20d=2.0,
        min_turnover_ratio_20d=1.5,
        min_daily_return_pct=0.0,
    ),
    "positive_interest": SignalDefinition(
        name="positive_interest",
        description="Pozitif ilgi temel filtreleri",
        min_volume_ratio_20d=1.5,
        min_turnover_ratio_20d=1.2,
        min_daily_return_pct=0.0,
        min_close_position=0.6,
    ),
    "positive_money_flow": SignalDefinition(
        name="positive_money_flow",
        description="Pozitif para girisi proxysi",
        min_volume_ratio_20d=1.5,
        min_turnover_ratio_20d=1.2,
        min_daily_return_pct=0.0,
        min_close_position=0.6,
        min_cmf_20=0.0,
        require_obv_slope_5d_positive=True,
        min_mfi_14=50.0,
        max_mfi_14=85.0,
        min_accumulation_score=50.0,
    ),
    "silent_accumulation": SignalDefinition(
        name="silent_accumulation",
        description="Dar fiyat araliginda sessiz toplama",
        min_volume_ratio_20d=1.5,
        min_turnover_ratio_20d=1.2,
        min_daily_return_pct=-1.0,
        max_daily_return_pct=2.0,
        min_close_position=0.5,
        min_cmf_20=0.0,
        require_obv_slope_5d_positive=True,
        require_obv_slope_20d_positive=True,
        min_accumulation_score=50.0,
    ),
    "strong_momentum": SignalDefinition(
        name="strong_momentum",
        description="Guclu momentum + para girisi",
        min_avg_turnover_20d=30_000_000.0,
        min_volume_ratio_20d=2.0,
        min_turnover_ratio_20d=1.5,
        min_daily_return_pct=1.0,
        min_close_position=0.7,
        min_cmf_20=0.05,
        require_obv_slope_5d_positive=True,
        min_mfi_14=55.0,
        require_above_ma20=True,
        min_relative_return_vs_xu100=0.0,
        min_accumulation_score=60.0,
    ),
}


def resolve_strategies(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return sorted(SIGNAL_DEFINITIONS.keys())
    names: list[str] = []
    for item in requested:
        name = item.strip()
        if name in SIGNAL_DEFINITIONS:
            names.append(name)
    return sorted(set(names))

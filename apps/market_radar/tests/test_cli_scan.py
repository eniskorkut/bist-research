from __future__ import annotations

from pathlib import Path
import runpy

from market_radar.radar_engine import RadarConfig, evaluate_filters


def _load_parse_args():
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_bist_interest.py"
    module_globals = runpy.run_path(str(script))
    return module_globals["parse_args"]


def test_cli_allows_breakout_off() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--breakout-mode", "off"])
    assert args.breakout_mode == "off"


def test_no_require_close_position_disables_filter() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--no-require-close-position"])
    assert args.require_close_position is False

    config = RadarConfig(
        min_close_position_active=args.require_close_position,
        min_close_position=args.min_close_position,
        min_interest_score_active=False,
    )
    metrics = {
        "avg_turnover_20d": 20_000_000.0,
        "volume_ratio_20d": 1.5,
        "turnover_ratio_20d": 1.2,
        "daily_return_pct": 0.5,
        "close_position": 0.1,
        "breakout_20d": False,
        "above_ma20": True,
        "above_ma50": False,
        "xu100_relative_return_pct": 0.2,
        "interest_score": 45.0,
    }
    passed, failed = evaluate_filters(metrics, config)
    assert "min_close_position" not in failed
    assert "min_close_position" not in passed


def test_no_require_min_score_keeps_low_score_candidate() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--no-require-min-score", "--min-volume-ratio", "1.0"])
    assert args.require_min_score is False

    config = RadarConfig(
        min_volume_ratio_active=True,
        min_volume_ratio=1.0,
        min_turnover_ratio_active=False,
        min_daily_return_active=False,
        min_close_position_active=False,
        require_ma20_active=False,
        require_ma50_active=False,
        min_xu100_relative_active=False,
        min_interest_score_active=args.require_min_score,
        include_negative_moves=True,
    )
    metrics = {
        "avg_turnover_20d": 20_000_000.0,
        "volume_ratio_20d": 1.1,
        "turnover_ratio_20d": 0.0,
        "daily_return_pct": -1.0,
        "close_position": 0.2,
        "breakout_20d": False,
        "above_ma20": False,
        "above_ma50": False,
        "xu100_relative_return_pct": -1.0,
        "interest_score": 10.0,
    }
    passed, failed = evaluate_filters(metrics, config)
    assert "min_volume_ratio" in passed
    assert "min_interest_score" not in failed

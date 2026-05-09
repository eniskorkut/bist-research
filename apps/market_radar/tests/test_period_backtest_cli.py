import importlib.util
from pathlib import Path


def _load_parse_args():
    script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_interest_periods.py"
    spec = importlib.util.spec_from_file_location("backtest_interest_periods", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.parse_args


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_interest_periods.py"
    spec = importlib.util.spec_from_file_location("backtest_interest_periods", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_parse_period_args_defaults() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--period-starts", "2026-01-01", "2026-02-01"])
    assert args.period_starts == ["2026-01-01", "2026-02-01"]
    assert args.basket_mode == "first_signal_per_symbol"


def test_parse_period_args_signal_weighted() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--period-starts",
            "2026-01-01",
            "2026-02-01",
            "--period-ends",
            "2026-02-01",
            "2026-03-01",
            "--basket-mode",
            "signal_weighted",
            "--as-of-date",
            "2026-05-09",
        ]
    )
    assert args.period_ends == ["2026-02-01", "2026-03-01"]
    assert args.basket_mode == "signal_weighted"
    assert args.as_of_date == "2026-05-09"


def test_parse_monthly_period_args() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--period-start",
            "2025-01-15",
            "--period-end",
            "2025-03-10",
            "--monthly",
        ]
    )
    assert args.period_starts == ["2025-01-01", "2025-02-01", "2025-03-01"]


def test_parse_all_signals_mode() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--period-starts", "2026-01-01", "--basket-mode", "all_signals"])
    assert args.basket_mode == "all_signals"

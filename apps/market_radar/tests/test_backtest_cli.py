import importlib.util
from pathlib import Path


def _load_parse_args():
    script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_interest_signals.py"
    spec = importlib.util.spec_from_file_location("backtest_interest_signals", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.parse_args


def test_cli_parses_strategies_and_workers() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--index", "XU100", "--strategies", "all", "--max-workers", "8", "--cooldown-days", "0"])
    assert args.index == "XU100"
    assert args.strategies == ["all"]
    assert args.max_workers == 8
    assert args.cooldown_days == 0


def test_cli_parses_no_cooldown_flag() -> None:
    parse_args = _load_parse_args()
    args = parse_args(["--no-cooldown"])
    assert args.no_cooldown is True

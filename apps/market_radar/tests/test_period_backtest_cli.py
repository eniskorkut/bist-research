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
    assert args.universe == "XUTUM"
    assert args.benchmark == "XU100"


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


def test_parse_universe_and_benchmark() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        ["--period-starts", "2026-01-01", "--universe", "XUTUM", "--benchmark", "XU100"]
    )
    assert args.universe == "XUTUM"
    assert args.benchmark == "XU100"


def test_parse_quality_filter_args() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--period-starts",
            "2026-01-01",
            "--strategies",
            "volume_spike_strict",
            "--active-volume-spike-quality",
            "--min-last-turnover-try",
            "10000000",
            "--min-avg-turnover-20d-try",
            "12000000",
            "--max-rsi-14",
            "78",
            "--max-return-5d-pct",
            "35",
            "--max-return-10d-pct",
            "60",
            "--require-strong-close",
            "--min-close-position",
            "0.60",
        ]
    )
    assert args.active_volume_spike_quality is True
    assert args.min_last_turnover_try == 10_000_000.0
    assert args.min_avg_turnover_20d_try == 12_000_000.0
    assert args.max_rsi_14 == 78.0
    assert args.max_return_5d_pct == 35.0
    assert args.max_return_10d_pct == 60.0
    assert args.require_strong_close is True
    assert args.min_close_position == 0.60


def test_parse_legacy_monthly_aliases() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-03-31",
            "--frequency",
            "monthly",
            "--strategy",
            "volume_spike_strict",
        ]
    )
    assert args.monthly is True
    assert args.period_starts == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert args.strategies == ["volume_spike_strict"]


def test_parse_resume_flags() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--period-starts",
            "2026-01-01",
            "--resume",
            "--checkpoint-each-period",
            "--skip-existing-periods",
        ]
    )
    assert args.resume is True
    assert args.checkpoint_each_period is True
    assert args.skip_existing_periods is True
    assert args.cache_only is True


def test_parse_symbol_checkpoint_and_debug_args() -> None:
    parse_args = _load_parse_args()
    args = parse_args(
        [
            "--period-starts",
            "2026-01-01",
            "--symbol-checkpoint-every",
            "10",
            "--progress-log-every",
            "5",
            "--max-symbols",
            "100",
            "--symbols-per-run",
            "25",
            "--only-symbols",
            "THYAO,ASELS",
            "--no-cache-only",
        ]
    )
    assert args.symbol_checkpoint_every == 10
    assert args.progress_log_every == 5
    assert args.max_symbols == 100
    assert args.symbols_per_run == 25
    assert args.only_symbols == "THYAO,ASELS"
    assert args.cache_only is False

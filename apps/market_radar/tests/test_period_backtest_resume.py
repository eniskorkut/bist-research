from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from market_radar.backtesting.period_backtest_engine import PeriodBacktestResult


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_interest_periods.py"
    spec = importlib.util.spec_from_file_location("backtest_interest_periods", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _fake_result(period_start: str, signal_count: int = 1) -> PeriodBacktestResult:
    period_end = "2026-02-01" if period_start == "2026-01-01" else "2026-03-01"
    holdings = pd.DataFrame(
        [
            {
                "period_start": period_start,
                "period_end": period_end,
                "strategy": "volume_spike_strict",
                "symbol": "AAA",
                "signal_date": period_start,
                "entry_price": 10.0,
            }
        ]
    )
    period = pd.DataFrame(
        [
            {
                "period_start": period_start,
                "period_end": period_end,
                "strategy": "volume_spike_strict",
                "signal_count": signal_count,
                "signal_count_before_quality_filter": signal_count + 1,
                "signal_count_after_quality_filter": signal_count,
                "filtered_out_count": 1,
                "avg_basket_alpha_to_current": 1.0,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "strategy": "volume_spike_strict",
                "period_count": 1,
                "active_period_count": 1,
                "empty_period_count": 0,
                "avg_basket_alpha_to_current": 1.0,
            }
        ]
    )
    diag = pd.DataFrame([{"period_start": period_start, "symbols_with_required_lookback": 10}])
    reasons = pd.DataFrame([{"period_start": period_start, "failed_rsi_14": 1}])
    coverage = pd.DataFrame([{"symbol": "AAA", "row_count": 100}])
    return PeriodBacktestResult(
        holdings=holdings,
        period_strategy_summary=period,
        strategy_stability_summary=stability,
        period_diagnostics_summary=diag,
        quality_filter_reason_summary=reasons,
        data_coverage_summary=coverage,
        run_summary={"period_count": 1},
    )


def test_checkpoint_files_written_and_resume_skips(tmp_path, monkeypatch) -> None:
    module = _load_module()
    calls = {"count": 0}

    def _fake_run(cfg):
        calls["count"] += 1
        return _fake_result(cfg.period_starts[0], 1)

    monkeypatch.setattr(module, "run_period_backtest", _fake_run)
    monkeypatch.setattr(module, "load_bist_universe", lambda *args, **kwargs: (["AAA"], "stale_cache"))
    monkeypatch.setattr(module, "write_period_outputs", lambda result, cfg: {"period_strategy_summary.csv": str(Path(cfg.output_dir) / "period_strategy_summary.csv")})

    argv = [
        "--period-starts",
        "2026-01-01",
        "2026-02-01",
        "--output-dir",
        str(tmp_path),
        "--checkpoint-each-period",
    ]
    monkeypatch.setattr("sys.argv", ["x", *argv])
    module.main()
    assert calls["count"] == 2
    cp1 = tmp_path / "period_checkpoints" / "2026-01-01.json"
    cp2 = tmp_path / "period_checkpoints" / "2026-02-01.json"
    sc1 = tmp_path / "period_checkpoints" / "2026-01-01.symbols.csv"
    assert cp1.exists()
    assert cp2.exists()
    assert sc1.exists()
    assert "period_start,symbol,status" in sc1.read_text(encoding="utf-8")

    calls["count"] = 0
    argv2 = [
        "--period-starts",
        "2026-01-01",
        "2026-02-01",
        "--output-dir",
        str(tmp_path),
        "--checkpoint-each-period",
        "--resume",
        "--skip-existing-periods",
    ]
    monkeypatch.setattr("sys.argv", ["x", *argv2])
    module.main()
    assert calls["count"] == 0


def test_error_checkpoint_written(tmp_path, monkeypatch) -> None:
    module = _load_module()

    def _fake_run(cfg):
        if cfg.period_starts[0] == "2026-02-01":
            raise RuntimeError("boom")
        return _fake_result(cfg.period_starts[0], 1)

    monkeypatch.setattr(module, "run_period_backtest", _fake_run)
    monkeypatch.setattr(module, "load_bist_universe", lambda *args, **kwargs: (["AAA"], "stale_cache"))
    monkeypatch.setattr(module, "write_period_outputs", lambda result, cfg: {"period_strategy_summary.csv": str(Path(cfg.output_dir) / "period_strategy_summary.csv")})

    argv = [
        "--period-starts",
        "2026-01-01",
        "2026-02-01",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", ["x", *argv])
    module.main()
    err = tmp_path / "period_checkpoints" / "2026-02-01.error.json"
    assert err.exists()
    payload = json.loads(err.read_text(encoding="utf-8"))
    assert payload["period_error"] is True


def test_resume_uses_symbol_checkpoint_subset(tmp_path, monkeypatch) -> None:
    module = _load_module()
    seen_only_symbols: list[list[str] | None] = []

    def _fake_run(cfg):
        seen_only_symbols.append(cfg.only_symbols)
        # Return result for whichever symbol subset is requested.
        target_symbol = (cfg.only_symbols or ["AAA"])[0]
        result = _fake_result(cfg.period_starts[0], 1)
        result.holdings["symbol"] = target_symbol
        return result

    monkeypatch.setattr(module, "run_period_backtest", _fake_run)
    monkeypatch.setattr(module, "load_bist_universe", lambda *args, **kwargs: (["AAA", "BBB"], "stale_cache"))
    monkeypatch.setattr(module, "write_period_outputs", lambda result, cfg: {"period_strategy_summary.csv": str(Path(cfg.output_dir) / "period_strategy_summary.csv")})

    cp_dir = tmp_path / "period_checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / "2026-01-01.symbols.csv").write_text(
        "period_start,symbol,status\n2026-01-01,AAA,completed\n",
        encoding="utf-8",
    )

    argv = [
        "--period-starts",
        "2026-01-01",
        "--output-dir",
        str(tmp_path),
        "--resume",
    ]
    monkeypatch.setattr("sys.argv", ["x", *argv])
    module.main()
    # Should run only for remaining symbol BBB.
    assert seen_only_symbols
    assert seen_only_symbols[0] == ["BBB"]


def test_symbols_per_run_limits_subset(tmp_path, monkeypatch) -> None:
    module = _load_module()
    seen_only_symbols: list[list[str] | None] = []

    def _fake_run(cfg):
        seen_only_symbols.append(cfg.only_symbols)
        result = _fake_result(cfg.period_starts[0], 1)
        if cfg.only_symbols:
            result.holdings["symbol"] = cfg.only_symbols[0]
        return result

    monkeypatch.setattr(module, "run_period_backtest", _fake_run)
    monkeypatch.setattr(module, "load_bist_universe", lambda *args, **kwargs: (["AAA", "BBB", "CCC"], "stale_cache"))
    monkeypatch.setattr(module, "write_period_outputs", lambda result, cfg: {"period_strategy_summary.csv": str(Path(cfg.output_dir) / "period_strategy_summary.csv")})

    argv = [
        "--period-starts",
        "2026-01-01",
        "--output-dir",
        str(tmp_path),
        "--symbols-per-run",
        "2",
    ]
    monkeypatch.setattr("sys.argv", ["x", *argv])
    module.main()
    assert seen_only_symbols
    assert len(seen_only_symbols[0]) == 2


def test_symbols_per_run_progress_and_full_completion(tmp_path, monkeypatch) -> None:
    module = _load_module()
    symbols = [f"S{i:03d}" for i in range(100)]

    def _fake_run(cfg):
        subset = cfg.only_symbols or []
        pstart = cfg.period_starts[0]
        pend = cfg.period_ends[0] if cfg.period_ends else "2026-02-01"
        holdings = pd.DataFrame(
            [
                {
                    "period_start": pstart,
                    "period_end": pend,
                    "strategy": "volume_spike_strict",
                    "symbol": sym,
                    "signal_date": pstart,
                    "entry_price": 10.0,
                }
                for sym in subset
            ]
        )
        period = pd.DataFrame(
            [
                {
                    "period_start": pstart,
                    "period_end": pend,
                    "strategy": "volume_spike_strict",
                    "signal_count": len(subset),
                    "signal_count_before_quality_filter": len(subset),
                    "signal_count_after_quality_filter": len(subset),
                    "filtered_out_count": 0,
                    "avg_basket_alpha_to_current": 1.0,
                }
            ]
        )
        stability = pd.DataFrame([{"strategy": "volume_spike_strict", "period_count": 1, "active_period_count": 1, "empty_period_count": 0, "avg_basket_alpha_to_current": 1.0}])
        diag = pd.DataFrame([{"period_start": pstart, "symbols_with_required_lookback": len(subset)}])
        reasons = pd.DataFrame([{"period_start": pstart, "failed_rsi_14": 0}])
        coverage = pd.DataFrame([{"symbol": sym, "row_count": 100} for sym in subset])
        return PeriodBacktestResult(
            holdings=holdings,
            period_strategy_summary=period,
            strategy_stability_summary=stability,
            period_diagnostics_summary=diag,
            quality_filter_reason_summary=reasons,
            data_coverage_summary=coverage,
            run_summary={"period_count": 1, "failed_symbols": []},
        )

    monkeypatch.setattr(module, "run_period_backtest", _fake_run)
    monkeypatch.setattr(module, "load_bist_universe", lambda *args, **kwargs: (symbols, "stale_cache"))
    monkeypatch.setattr(module, "write_period_outputs", lambda result, cfg: {"period_strategy_summary.csv": str(Path(cfg.output_dir) / "period_strategy_summary.csv")})

    base_argv = [
        "--period-starts",
        "2026-01-01",
        "--output-dir",
        str(tmp_path),
        "--symbols-per-run",
        "25",
        "--resume",
    ]

    for i in range(4):
        monkeypatch.setattr("sys.argv", ["x", *base_argv])
        module.main()
        progress = json.loads((tmp_path / "period_progress.json").read_text(encoding="utf-8"))
        if i == 0:
            assert progress["fully_completed_periods"] == 0
            assert progress["active_period_processed_symbols"] == 25
        if i == 1:
            assert progress["active_period_processed_symbols"] == 50
        if i == 3:
            assert progress["fully_completed_periods"] == 1
            assert (tmp_path / "period_checkpoints" / "2026-01-01.json").exists()

from market_radar.backtesting.signal_definitions import SIGNAL_DEFINITIONS, resolve_strategies


def test_resolve_all_strategies() -> None:
    names = resolve_strategies(["all"])
    assert "positive_money_flow" in names
    assert "strong_momentum" in names


def test_positive_money_flow_match() -> None:
    strategy = SIGNAL_DEFINITIONS["positive_money_flow"]
    metrics = {
        "volume_ratio_20d": 2.0,
        "turnover_ratio_20d": 1.5,
        "daily_return_pct": 0.5,
        "close_position": 0.7,
        "cmf_20": 0.1,
        "obv_slope_5d": 10,
        "mfi_14": 60,
        "accumulation_score": 70,
    }
    assert strategy.match(metrics)

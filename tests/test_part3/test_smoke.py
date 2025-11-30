import pandas as pd
import numpy as np

from src.part3_strategy.task7_backtest import LongShortStrategy
from src.part3_strategy.task8_performance import (
    calculate_performance_metrics,
    compare_with_benchmarks,
    generate_performance_report,
)
from src.part3_strategy.task9_analysis import (
    analyze_strategy_performance,
    identify_drawdown_periods,
    generate_improvement_proposals,
)


def _synthetic_prices(n_bars: int = 120, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="5min")
    steps = rng.normal(0, 0.001, size=(n_bars, n_assets))
    prices = np.cumprod(1 + steps, axis=0)
    cols = [f"S{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_backtest_runs_and_shapes():
    prices = _synthetic_prices()
    returns = prices.pct_change().fillna(0)

    strat = LongShortStrategy(
        signal_type="bollinger",
        rebalance_periods=12,
        long_quantile=0.5,
        short_quantile=0.5,
    )
    res = strat.backtest(returns=returns, prices=prices)

    assert set(res.keys()) >= {
        "returns",
        "nav",
        "weights",
        "turnover",
        "transaction_costs",
        "gross_exposure",
        "capital_used",
        "trade_log",
    }
    assert len(res["returns"]) == len(returns)
    assert res["weights"].shape == returns.shape
    assert not res["nav"].isna().any()


def test_performance_and_report():
    prices = _synthetic_prices(seed=1)
    returns = prices.pct_change().fillna(0)
    strat = LongShortStrategy(signal_type="bollinger", rebalance_periods=10, long_quantile=0.5, short_quantile=0.5)
    res = strat.backtest(returns=returns, prices=prices)

    metrics = calculate_performance_metrics(res["returns"])
    assert "sharpe_ratio" in metrics

    bench = res["returns"].rolling(3).mean().fillna(0)
    cmp = compare_with_benchmarks(res["returns"], bench)
    assert "information_ratio" in cmp

    report = generate_performance_report(res)
    assert "metrics" in report and "summary" in report


def test_analysis_and_improvements():
    prices = _synthetic_prices(seed=2)
    returns = prices.pct_change().fillna(0)
    strat = LongShortStrategy(signal_type="bollinger", rebalance_periods=8, long_quantile=0.5, short_quantile=0.5)
    res = strat.backtest(returns=returns, prices=prices)

    analysis = analyze_strategy_performance(res)
    assert "drawdown" in analysis

    dd_periods = identify_drawdown_periods(res["nav"])
    assert isinstance(dd_periods, list)

    proposals = generate_improvement_proposals(analysis)
    assert isinstance(proposals, dict)

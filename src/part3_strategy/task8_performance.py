"""
Task 8: Performance Evaluation & Tearsheet Generation

Minimal performance metrics and simple report helpers to evaluate backtest
results produced by Task 7. Focus on essential metrics and a compact API
that is easy to use in examples and tests.

Author: ELEC4546/7079 Course
Date: December 2024
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def calculate_performance_metrics(returns_series: pd.Series) -> Dict[str, float]:
    """
    Calculate key performance metrics for a return series.

    Args:
        returns_series (pd.Series): Strategy returns per period

    Returns:
        Dict[str, float]: Metrics such as total_return, sharpe_ratio, max_drawdown
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints for performance metrics:
    # 1. Preprocess: remove NaNs via returns_series.dropna()
    # 2. Return metrics:
    #    - total return: (1 + r).prod() - 1
    #    - annualized return: (1 + r.mean()) ** 252 - 1  (assume 252 trading days)
    # 3. Risk metrics:
    #    - annualized volatility: r.std() * sqrt(252)
    #    - max drawdown: max((running peak - cumulative) / running peak)
    # 4. Risk-adjusted return:
    #    - Sharpe ratio: annualized return / annualized volatility
    # 5. Build result dict:
    #    - Include total_return, annualized_return, annualized_volatility
    #    - sharpe_ratio, max_drawdown, turnover, etc.
    #
    # Expected output: Dict[str, float] containing all key metrics
    
    raise NotImplementedError("Please implement performance metrics calculation logic")


def compare_with_benchmarks(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Compare strategy with a benchmark using simple metrics.

    Args:
        strategy_returns (pd.Series): Strategy returns
        benchmark_returns (pd.Series): Benchmark returns

    Returns:
        Dict[str, float]: Comparison metrics
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Benchmark comparison implementation hints:
    # 1. Align and clean data:
    #    - Drop NaNs in strategy returns
    #    - Reindex benchmark to strategy index: benchmark_returns.reindex(s.index)
    #    - Ensure equal lengths and sufficient data
    # 2. Excess return: excess = strategy_returns - benchmark_returns
    # 3. Information ratio:
    #    - tracking_error = excess.std() * sqrt(252)
    #    - IR = annualized excess return / tracking_error
    # 4. Correlation: strategy_returns.corr(benchmark_returns)
    # 5. Return a metrics dict
    #
    # Expected output: Dict[str, float] containing information_ratio, correlation, etc.
    
    raise NotImplementedError("Please implement benchmark comparison logic")


def generate_performance_report(strategy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a compact performance report dictionary from Task 7 results.

    Args:
        strategy_results (Dict[str, Any]): Output of Task 7 backtest

    Returns:
        Dict[str, Any]: Report with metrics and final nav
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Report generation implementation hints:
    # 1. Extract from backtest results:
    #    - returns = strategy_results.get("returns", ...)
    #    - nav = strategy_results.get("nav", ...)
    #    - optionally weights history, trade log, etc.
    # 2. Call performance calculation:
    #    - use calculate_performance_metrics()
    # 3. Construct report:
    #    - include metrics dict
    #    - final nav: nav.iloc[-1]
    #    - number of periods: len(returns)
    #    - other summary info
    # 4. Return the report dict
    #
    # Expected output: Dict[str, Any] containing a complete performance report
    
    raise NotImplementedError("Please implement performance report generation logic")


# Minimal demo
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.part3_strategy.task7_backtest import LongShortStrategy

    loader = DataLoader()
    data_5m = loader.load_5min_data()
    prices = data_5m.xs("close_px", axis=1, level=1)
    rets = prices.pct_change().fillna(0.0)

    strat = LongShortStrategy(signal_type="macd", rebalance_periods=12)
    results = strat.backtest(returns=rets, prices=prices)

    report = generate_performance_report(results)
    print("Report:", report)



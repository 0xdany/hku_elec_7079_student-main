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

    rets = returns_series.dropna()  # drop nans, obvs
    if len(rets) == 0:
        return {}

    total_return = float((1 + rets).prod() - 1)  # total ret calc
    mean_ret = float(rets.mean())  # mean ret, duh
    vol = float(rets.std())  # std dev, prob fine

    annualized_return = (1 + mean_ret) ** 252 - 1
    annualized_volatility = vol * np.sqrt(252)

    # calc drawdown, probs should check this later lol
    cumulative = (1 + rets).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0  # sharpe, hope this is right

    downside = rets[rets < 0]  # only neg rets
    if len(downside) > 0:
        downside_vol = float(downside.std()) * np.sqrt(252)
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino_ratio = float("inf") if annualized_return > 0 else 0.0

    win_rate = float((rets > 0).sum() / len(rets))  # win rate calc
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0  # avg win, seems ok
    avg_loss = float(abs(rets[rets < 0].mean())) if (rets < 0).any() else 0.0  # avg loss, abs val
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_loss_ratio": float(profit_loss_ratio),
    }


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

    strat = strategy_returns.dropna()  # clean strat rets
    if strat.empty:
        return {}

    bench = benchmark_returns.reindex(strat.index).fillna(0.0)  # align bench to strat idx
    if len(bench) != len(strat) or len(strat) < 2:
        return {}

    excess = strat - bench  # excess rets
    excess_mean = float(excess.mean())  # mean excess
    excess_std = float(excess.std())  # std of excess

    annualized_excess = (1 + excess_mean) ** 252 - 1  # ann excess ret
    tracking_error = excess_std * np.sqrt(252)  # track err, sqrt(252) for ann
    information_ratio = annualized_excess / tracking_error if tracking_error > 0 else 0.0  # IR calc

    correlation = float(strat.corr(bench))  # corr btw strat n bench

    if float(bench.std()) > 0:  # check if bench has vol
        beta = float(strat.cov(bench) / bench.var())  # beta calc, cov/var
    else:
        beta = 0.0  # no vol means no beta ig

    strat_ann = (1 + float(strat.mean())) ** 252 - 1  # ann strat ret
    bench_ann = (1 + float(bench.mean())) ** 252 - 1  # ann bench ret
    alpha = strat_ann - beta * bench_ann  # alpha, jensen's alpha i think?

    return {
        "information_ratio": float(information_ratio),
        "tracking_error": float(tracking_error),
        "beta": float(beta),
        "alpha": float(alpha),
        "correlation": correlation,
        "excess_return": float(annualized_excess),
    }


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

    returns = strategy_results.get("returns", pd.Series())  # get rets from results
    nav = strategy_results.get("nav", pd.Series())  # nav series
    turnover = strategy_results.get("turnover", pd.Series())  # turnover data
    tx_costs = strategy_results.get("transaction_costs", pd.Series())  # tx costs, fees n stuff
    trade_log = strategy_results.get("trade_log", [])

    if returns is None or len(returns) == 0:
        return {"error": "No returns data available"}

    metrics = calculate_performance_metrics(returns)
    final_nav = float(nav.iloc[-1]) if nav is not None and len(nav) > 0 else 1.0

    report = {
        "metrics": metrics,
        "summary": {
            "final_nav": final_nav,
            "num_periods": int(len(returns)),
            "total_turnover": float(turnover.sum()) if hasattr(turnover, "sum") else 0.0,
            "average_turnover_per_period": float(turnover.mean()) if hasattr(turnover, "mean") else 0.0,
            "total_transaction_costs": float(tx_costs.sum()) if hasattr(tx_costs, "sum") else 0.0,
            "num_trades": int(len(trade_log)),
        },
        "time_series": {
            "returns": returns.to_dict(),
            "nav": nav.to_dict() if nav is not None else {},
            "turnover": turnover.to_dict() if hasattr(turnover, "to_dict") else {},
        },
    }

    return report


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

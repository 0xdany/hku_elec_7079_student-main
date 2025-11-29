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
    rets = returns_series.dropna()
    if len(rets) == 0:
        return {}

    total_return = float((1 + rets).prod() - 1)
    mean_ret = float(rets.mean())
    vol = float(rets.std())

    annualized_return = (1 + mean_ret) ** 252 - 1
    annualized_volatility = vol * np.sqrt(252)

    # Drawdown
    cumulative = (1 + rets).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

    downside = rets[rets < 0]
    if len(downside) > 0:
        downside_vol = float(downside.std()) * np.sqrt(252)
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino_ratio = float("inf") if annualized_return > 0 else 0.0

    win_rate = float((rets > 0).sum() / len(rets))
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(abs(rets[rets < 0].mean())) if (rets < 0).any() else 0.0
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
    strat = strategy_returns.dropna()
    if strat.empty:
        return {}

    bench = benchmark_returns.reindex(strat.index).fillna(0.0)
    if len(bench) != len(strat) or len(strat) < 2:
        return {}

    excess = strat - bench
    excess_mean = float(excess.mean())
    excess_std = float(excess.std())

    annualized_excess = (1 + excess_mean) ** 252 - 1
    tracking_error = excess_std * np.sqrt(252)
    information_ratio = annualized_excess / tracking_error if tracking_error > 0 else 0.0

    correlation = float(strat.corr(bench))

    if float(bench.std()) > 0:
        beta = float(strat.cov(bench) / bench.var())
    else:
        beta = 0.0

    strat_ann = (1 + float(strat.mean())) ** 252 - 1
    bench_ann = (1 + float(bench.mean())) ** 252 - 1
    alpha = strat_ann - beta * bench_ann

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
    returns = strategy_results.get("returns", pd.Series())
    nav = strategy_results.get("nav", pd.Series())
    turnover = strategy_results.get("turnover", pd.Series())
    tx_costs = strategy_results.get("transaction_costs", pd.Series())
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

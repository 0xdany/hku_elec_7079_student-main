"""
Task 9: Critical Analysis & Proposed Improvements

Provide simple analysis helpers to inspect strategy performance and produce
concise improvement suggestions automatically as a starting point.

Author: ELEC4546/7079 Course
Date: December 2024
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def analyze_strategy_performance(strategy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze key characteristics of the strategy performance to identify
    strengths and weaknesses.

    Args:
        strategy_results (Dict[str, Any]): Output from Task 7 backtest

    Returns:
        Dict[str, Any]: Analysis summary containing periods of drawdown, streaks, etc.
    """
    returns = strategy_results.get("returns", pd.Series()).dropna()
    nav = strategy_results.get("nav", pd.Series())
    weights = strategy_results.get("weights", pd.DataFrame())

    if returns.empty:
        return {"error": "Insufficient data"}

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin()
    max_dd_start = running_max[:max_dd_date].idxmax()

    def _longest_streak(series: pd.Series, condition) -> int:
        best = 0
        curr = 0
        for v in series:
            if condition(v):
                curr += 1
                best = max(best, curr)
            else:
                curr = 0
        return best

    longest_win = _longest_streak(returns, lambda x: x > 0)
    longest_loss = _longest_streak(returns, lambda x: x < 0)

    monthly = returns.resample("M").sum()
    winning_months = int((monthly > 0).sum()) if len(monthly) else 0
    total_months = int(len(monthly))
    monthly_win_rate = float(winning_months / total_months) if total_months > 0 else 0.0

    if not weights.empty:
        long_exposure = weights.clip(lower=0).sum(axis=1)
        short_exposure = weights.clip(upper=0).abs().sum(axis=1)
        avg_long_exp = float(long_exposure.mean())
        avg_short_exp = float(short_exposure.mean())
    else:
        avg_long_exp = 0.0
        avg_short_exp = 0.0

    analysis = {
        "drawdown": {
            "max_drawdown": max_dd,
            "max_drawdown_date": str(max_dd_date),
            "max_drawdown_start": str(max_dd_start),
        },
        "streaks": {
            "longest_winning_streak": int(longest_win),
            "longest_losing_streak": int(longest_loss),
        },
        "exposure": {
            "avg_long_exposure": avg_long_exp,
            "avg_short_exposure": avg_short_exp,
        },
        "time_series": {
            "winning_months": winning_months,
            "total_months": total_months,
            "monthly_win_rate": monthly_win_rate,
        },
        "final_nav": float(nav.iloc[-1]) if nav is not None and len(nav) > 0 else 1.0,
        "total_return": float(cumulative.iloc[-1] - 1) if len(cumulative) else 0.0,
        "win_rate": float((returns > 0).mean()),
    }

    return analysis


def identify_drawdown_periods(nav_series, threshold=0.05) -> list:
    """
    Identify drawdown periods from a NAV curve.
    """
    if nav_series is None or len(nav_series) < 2:
        return []

    running_max = nav_series.expanding().max()
    drawdown = (nav_series - running_max) / running_max
    in_dd = drawdown < -threshold

    periods = []
    start_idx = None
    start_date = None

    for idx, (date, flag) in enumerate(zip(nav_series.index, in_dd)):
        if flag and start_idx is None:
            start_idx = idx
            start_date = date
        elif not flag and start_idx is not None:
            end_date = date
            dd_slice = drawdown.iloc[start_idx:idx + 1]
            max_dd = float(dd_slice.min())
            max_dd_date = dd_slice.idxmin()
            duration = (end_date - start_date).days
            recovery = (end_date - max_dd_date).days
            periods.append(
                {
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "max_drawdown": max_dd,
                    "max_drawdown_date": str(max_dd_date),
                    "duration_days": duration,
                    "recovery_days": recovery,
                }
            )
            start_idx = None
            start_date = None

    if start_idx is not None:
        end_date = nav_series.index[-1]
        dd_slice = drawdown.iloc[start_idx:]
        max_dd = float(dd_slice.min())
        max_dd_date = dd_slice.idxmin()
        duration = (end_date - start_date).days
        periods.append(
            {
                "start_date": str(start_date),
                "end_date": str(end_date),
                "max_drawdown": max_dd,
                "max_drawdown_date": str(max_dd_date),
                "duration_days": duration,
                "recovery_days": None,
            }
        )

    return periods


def generate_improvement_proposals(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate simple, actionable suggestions based on analysis findings.

    Args:
        analysis_results (Dict[str, Any]): Output of analyze_strategy_performance

    Returns:
        Dict[str, Any]: Suggested improvements
    """
    proposals = {
        "risk_management": [],
        "return_enhancement": [],
        "cost_optimization": [],
        "operational_improvement": [],
    }

    dd = abs(analysis_results.get("drawdown", {}).get("max_drawdown", 0))
    longest_loss = analysis_results.get("streaks", {}).get("longest_losing_streak", 0)
    total_return = analysis_results.get("total_return", 0.0)
    monthly_win = analysis_results.get("time_series", {}).get("monthly_win_rate", 0.0)

    if dd > 0.20:
        proposals["risk_management"].append(
            "Max drawdown exceeds 20%; consider portfolio-level stop-loss or volatility scaling."
        )
    if longest_loss >= 5:
        proposals["risk_management"].append(
            "Long losing streaks detected; add market regime filters or pause trading in adverse conditions."
        )
    if dd > 0.15:
        proposals["risk_management"].append(
            "Implement dynamic position sizing to reduce exposure after large drawdowns."
        )

    if total_return < 0:
        proposals["return_enhancement"].append(
            "Negative cumulative return; reassess signal quality and factor robustness."
        )
    if monthly_win < 0.5:
        proposals["return_enhancement"].append(
            "Monthly win rate below 50%; refine entry timing or add confirmation filters."
        )
    proposals["return_enhancement"].append(
        "Combine multiple uncorrelated signals to improve risk-adjusted returns."
    )

    proposals["cost_optimization"].append(
        "Review rebalance frequency and execution to reduce turnover and transaction costs."
    )
    proposals["cost_optimization"].append(
        "Use smarter execution (VWAP/TWAP/limit orders) to minimize slippage on large trades."
    )

    proposals["operational_improvement"].append(
        "Expand and regularly refresh the tradable universe with liquidity filters."
    )
    proposals["operational_improvement"].append(
        "Conduct sensitivity and robustness tests across periods and parameter sets."
    )

    # remove empty categories
    return {k: v for k, v in proposals.items() if v}


# Minimal demo
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.part3_strategy.task7_backtest import LongShortStrategy

    loader = DataLoader()
    data_5m = loader.load_5min_data()
    prices = data_5m.xs("close_px", axis=1, level=1)
    rets = prices.pct_change().fillna(0.0)

    strat = LongShortStrategy(signal_type="bollinger", rebalance_periods=12)
    results = strat.backtest(returns=rets, prices=prices)

    analysis = analyze_strategy_performance(results)
    ideas = generate_improvement_proposals(analysis)
    print("Analysis:", analysis)
    print("Proposals:", ideas)


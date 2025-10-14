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
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Strategy performance analysis hints:
    # 1. Data extraction and validation:
    #    - Extract returns and nav series
    #    - Drop NaNs and validate data
    # 2. Drawdown analysis:
    #    - cum = (1 + returns).cumprod()
    #    - roll_max = cum.expanding().max()
    #    - drawdown = (cum - roll_max) / roll_max
    #    - identify max drawdown and start/end
    # 3. Win/loss streaks:
    #    - signs = np.sign(returns)
    #    - implement _longest_streak()
    #    - find longest winning and losing streaks
    # 4. Other metrics:
    #    - final nav, win rate, average profit-loss ratio, etc.
    # 5. Build result dictionary
    #
    # Expected output: Dict[str, Any] with drawdown, streaks, and key metrics
    
    raise NotImplementedError("Please implement strategy performance analysis logic")


def generate_improvement_proposals(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate simple, actionable suggestions based on analysis findings.

    Args:
        analysis_results (Dict[str, Any]): Output of analyze_strategy_performance

    Returns:
        Dict[str, Any]: Suggested improvements
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Improvement suggestion hints:
    # 1. Risk-based:
    #    - If max drawdown too large (>20%): add risk controls
    #    - If long losing streaks (>=5): add market regime filters
    #    - If volatility high: use dynamic position sizing
    # 2. Return-based:
    #    - If final return negative: reassess signal quality
    #    - If Sharpe ratio low: optimize risk-adjusted returns
    #    - If win rate low: improve entry timing
    # 3. Trading-based:
    #    - If turnover high: optimize rebalance frequency
    #    - If costs high: reduce trading frequency
    # 4. General:
    #    - If no glaring issues: run sensitivity analysis
    # 5. Organize suggestions by category: risk, return, cost, operations
    #
    # Expected output: Dict[str, Any] with categorized suggestions
    
    raise NotImplementedError("Please implement improvement suggestion logic")


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



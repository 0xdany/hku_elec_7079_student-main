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
    returns = strategy_results.get("returns", pd.Series()).dropna()  # get rets, drop nans
    nav = strategy_results.get("nav", pd.Series())  # nav series
    weights = strategy_results.get("weights", pd.DataFrame())  # weights df

    if returns.empty:  # check if empty
        return {"error": "Insufficient data"}

    cumulative = (1 + returns).cumprod()  # cum rets
    running_max = cumulative.expanding().max()  # running peak
    drawdown = (cumulative - running_max) / running_max  # calc dd

    max_dd = float(drawdown.min())  # max dd val
    max_dd_date = drawdown.idxmin()  # when max dd happened
    max_dd_start = running_max[:max_dd_date].idxmax()  # start of dd period

    def _longest_streak(series: pd.Series, condition) -> int:  # helper for streaks
        best = 0  # best streak so far
        curr = 0  # current streak
        for v in series:  # iterate thru vals
            if condition(v):  # if condition met
                curr += 1  # incr curr
                best = max(best, curr)  # update best
            else:
                curr = 0  # reset curr
        return best  # return longest

    longest_win = _longest_streak(returns, lambda x: x > 0)  # longest win streak
    longest_loss = _longest_streak(returns, lambda x: x < 0)  # longest loss streak

    monthly = returns.resample("M").sum()  # resample to monthly
    winning_months = int((monthly > 0).sum()) if len(monthly) else 0  # count winning months
    total_months = int(len(monthly))  # total months
    monthly_win_rate = float(winning_months / total_months) if total_months > 0 else 0.0  # monthly win rate

    if not weights.empty:  # if we have weights
        long_exposure = weights.clip(lower=0).sum(axis=1)  # sum of long pos
        short_exposure = weights.clip(upper=0).abs().sum(axis=1)  # sum of short pos
        avg_long_exp = float(long_exposure.mean())  # avg long exp
        avg_short_exp = float(short_exposure.mean())  # avg short exp
    else:
        avg_long_exp = 0.0  # no weights means 0
        avg_short_exp = 0.0  # same here

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
    
    if nav_series is None or len(nav_series) < 2:  # check if valid
        return []

    running_max = nav_series.expanding().max()  # running peak nav
    drawdown = (nav_series - running_max) / running_max  # calc dd
    in_dd = drawdown < -threshold  # flag if in dd period

    periods = []  # list to store dd periods
    start_idx = None  # start idx of current dd
    start_date = None  # start date

    for idx, (date, flag) in enumerate(zip(nav_series.index, in_dd)):  # iterate thru dates
        if flag and start_idx is None:  # entering dd period
            start_idx = idx  # mark start
            start_date = date  # mark start date
        elif not flag and start_idx is not None:  # exiting dd period
            end_date = date  # end date
            dd_slice = drawdown.iloc[start_idx:idx + 1]  # slice of dd vals
            max_dd = float(dd_slice.min())  # max dd in this period
            max_dd_date = dd_slice.idxmin()  # when max dd happened
            duration = (end_date - start_date).days  # duration in days
            recovery = (end_date - max_dd_date).days  # recovery time
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
            start_idx = None  # reset start
            start_date = None  # reset date

    if start_idx is not None:  # if still in dd at end
        end_date = nav_series.index[-1]  # use last date
        dd_slice = drawdown.iloc[start_idx:]  # slice from start to end
        max_dd = float(dd_slice.min())  # max dd
        max_dd_date = dd_slice.idxmin()  # max dd date
        duration = (end_date - start_date).days  # duration
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
    proposals = {  # dict to store proposals
        "risk_management": [],
        "return_enhancement": [],
        "cost_optimization": [],
        "operational_improvement": [],
    }

    dd = abs(analysis_results.get("drawdown", {}).get("max_drawdown", 0))  # get max dd
    longest_loss = analysis_results.get("streaks", {}).get("longest_losing_streak", 0)  # longest loss streak
    total_return = analysis_results.get("total_return", 0.0)  # total ret
    monthly_win = analysis_results.get("time_series", {}).get("monthly_win_rate", 0.0)  # monthly win rate

    if dd > 0.20:  # if dd > 20%
        proposals["risk_management"].append(
            "Max drawdown exceeds 20%; consider portfolio-level stop-loss or volatility scaling."
        )
    if longest_loss >= 5:  # if losing streak >= 5
        proposals["risk_management"].append(
            "Long losing streaks detected; add market regime filters or pause trading in adverse conditions."
        )
    if dd > 0.15:  # if dd > 15%
        proposals["risk_management"].append(
            "Implement dynamic position sizing to reduce exposure after large drawdowns."
        )

    if total_return < 0:  # if neg ret
        proposals["return_enhancement"].append(
            "Negative cumulative return; reassess signal quality and factor robustness."
        )
    if monthly_win < 0.5:  # if win rate < 50%
        proposals["return_enhancement"].append(
            "Monthly win rate below 50%; refine entry timing or add confirmation filters."
        )
    proposals["return_enhancement"].append(  # always add this one
        "Combine multiple uncorrelated signals to improve risk-adjusted returns."
    )

    proposals["cost_optimization"].append(  # cost opt stuff
        "Review rebalance frequency and execution to reduce turnover and transaction costs."
    )
    proposals["cost_optimization"].append(  # more cost stuff
        "Use smarter execution (VWAP/TWAP/limit orders) to minimize slippage on large trades."
    )

    proposals["operational_improvement"].append(  # ops stuff
        "Expand and regularly refresh the tradable universe with liquidity filters."
    )
    proposals["operational_improvement"].append(  # more ops
        "Conduct sensitivity and robustness tests across periods and parameter sets."
    )

    # remove empty cats, only return non-empty ones
    return {k: v for k, v in proposals.items() if v}


# Minimal demo
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.part3_strategy.task7_backtest import LongShortStrategy

    loader = DataLoader()  # init loader
    data_5m = loader.load_5min_data()  # load 5min data
    prices = data_5m.xs("close_px", axis=1, level=1)  # extract close prices
    rets = prices.pct_change().fillna(0.0)  # calc rets, fill nans

    strat = LongShortStrategy(signal_type="bollinger", rebalance_periods=12)  # init strat
    results = strat.backtest(returns=rets, prices=prices)  # run backtest

    analysis = analyze_strategy_performance(results)  # analyze results
    ideas = generate_improvement_proposals(analysis)  # gen proposals
    print("Analysis:", analysis)  # print analysis
    print("Proposals:", ideas)  # print proposals


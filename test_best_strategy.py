#!/usr/bin/env python
"""
Test script with the BEST configuration found from parameter sweep.

Best Config:
  - lookback: 3 bars (15 minutes)
  - rebalance_periods: 9600 bars (~200 trading days)
  - quantile: 0.05 (top/bottom 5%)
  - partial_threshold: 0.05 (5% min trade size)
  
Expected Performance:
  - Total Return: ~52%
  - Sharpe Ratio: ~0.094
  - Max Drawdown: ~-13.7%

Usage:
    uv run python test_best_strategy.py
"""

import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from part3_strategy.task7_backtest import (
    LongShortStrategy, 
    _extract_close_prices, 
    _pct_change_returns
)
from part3_strategy.task8_performance import calculate_performance_metrics


def main():
    print("=" * 70)
    print("BEST STRATEGY CONFIGURATION TEST")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    prices = _extract_close_prices(data_5min)
    returns = _pct_change_returns(prices)
    
    print(f"  Samples: {len(prices):,}")
    print(f"  Stocks: {len(prices.columns)}")
    
    # =========================================================================
    # BEST CONFIGURATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING BEST STRATEGY")
    print("=" * 70)
    
    best_strategy = LongShortStrategy(
        signal_type='reversal',
        rebalance_periods=9600,          # ~200 trading days
        signal_params={'lookback': 3},   # 15-minute lookback
        long_quantile=0.05,              # Top 5%
        short_quantile=0.05,             # Bottom 5%
        use_partial_rebalancing=True,
        min_trade_threshold=0.05,        # 5% minimum trade
        transaction_cost=0.0005,
    )
    
    print("\nConfiguration:")
    print(f"  Signal Type: reversal")
    print(f"  Lookback: 3 bars (15 min)")
    print(f"  Rebalance: every 9600 bars (~200 days)")
    print(f"  Quantile: 5% long / 5% short")
    print(f"  Partial Rebalancing: 5% threshold")
    
    # Run backtest
    print("\nRunning backtest...")
    results = best_strategy.backtest(returns=returns, prices=prices)
    metrics = calculate_performance_metrics(results['returns'])
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    
    print(f"\n  Total Return:        {metrics['total_return']*100:+.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return']*100:+.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:+.3f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:+.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:            {metrics['win_rate']*100:.1f}%")
    print(f"  Volatility (ann.):   {metrics['annualized_volatility']*100:.2f}%")
    
    # Transaction costs
    total_tc = results['transaction_costs'].sum()
    total_turnover = results['turnover'].sum()
    num_rebalances = (results['turnover'] > 0).sum()
    
    print(f"\n  Transaction Costs:   {total_tc*100:.2f}%")
    print(f"  Total Turnover:      {total_turnover:.2f}x")
    print(f"  Rebalance Events:    {num_rebalances}")
    
    # =========================================================================
    # COMPARISON WITH BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE")
    print("=" * 70)
    
    baseline = LongShortStrategy(
        signal_type='reversal',
        rebalance_periods=4800,
        signal_params={'lookback': 6},
        long_quantile=0.10,
        short_quantile=0.10,
    )
    baseline_results = baseline.backtest(returns=returns, prices=prices)
    baseline_metrics = calculate_performance_metrics(baseline_results['returns'])
    
    print(f"\n{'Metric':<25} {'Best':<15} {'Baseline':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {metrics['total_return']*100:+.2f}%{'':<8} {baseline_metrics['total_return']*100:+.2f}%{'':<8} {(metrics['total_return']-baseline_metrics['total_return'])*100:+.2f}%")
    print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:+.3f}{'':<10} {baseline_metrics['sharpe_ratio']:+.3f}{'':<10} {metrics['sharpe_ratio']-baseline_metrics['sharpe_ratio']:+.3f}")
    print(f"{'Max Drawdown':<25} {metrics['max_drawdown']*100:.2f}%{'':<8} {baseline_metrics['max_drawdown']*100:.2f}%{'':<8} {(metrics['max_drawdown']-baseline_metrics['max_drawdown'])*100:+.2f}%")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return metrics, results


if __name__ == "__main__":
    main()


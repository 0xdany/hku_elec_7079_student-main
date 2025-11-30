#!/usr/bin/env python
"""
Generate figures for the final report.

This script creates all visualizations needed for the report:
1. Cumulative NAV comparison (baseline vs enhanced)
2. Drawdown curves
3. Parameter sensitivity heatmap

Usage:
    uv run python generate_report_figures.py

Output:
    - report/figures/cumulative_nav.png
    - report/figures/drawdown_curve.png
    - report/figures/parameter_sensitivity.png
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import DataLoader
from part3_strategy.task7_backtest import (
    LongShortStrategy, 
    _extract_close_prices, 
    _pct_change_returns
)
from part3_strategy.task8_performance import calculate_performance_metrics


def setup_style():
    """Set up matplotlib style for report-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    prices = _extract_close_prices(data_5min)
    returns = _pct_change_returns(prices)
    return prices, returns


def run_strategies(returns, prices):
    """Run baseline and enhanced strategies."""
    print("\nRunning baseline strategy...")
    baseline = LongShortStrategy(
        signal_type='reversal',
        rebalance_periods=4800,
        signal_params={'lookback': 6},
        long_quantile=0.10,
        short_quantile=0.10,
    )
    baseline_results = baseline.backtest(returns=returns, prices=prices)
    
    print("Running enhanced strategy...")
    enhanced = LongShortStrategy(
        signal_type='reversal',
        rebalance_periods=9600,
        signal_params={'lookback': 3},
        long_quantile=0.05,
        short_quantile=0.05,
        use_partial_rebalancing=True,
        min_trade_threshold=0.05,
    )
    enhanced_results = enhanced.backtest(returns=returns, prices=prices)
    
    return baseline_results, enhanced_results


def plot_cumulative_nav(baseline_results, enhanced_results, output_dir):
    """Plot cumulative NAV comparison."""
    print("\nGenerating cumulative NAV chart...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate cumulative NAV
    baseline_nav = (1 + baseline_results['returns']).cumprod()
    enhanced_nav = (1 + enhanced_results['returns']).cumprod()
    
    # Resample to daily for cleaner visualization
    baseline_daily = baseline_nav.resample('D').last().dropna()
    enhanced_daily = enhanced_nav.resample('D').last().dropna()
    
    ax.plot(baseline_daily.index, baseline_daily.values, 
            label=f'Baseline (Return: {(baseline_daily.iloc[-1]-1)*100:+.1f}%)', 
            color='#7f8c8d', linewidth=1.5, alpha=0.8)
    ax.plot(enhanced_daily.index, enhanced_daily.values, 
            label=f'Enhanced (Return: {(enhanced_daily.iloc[-1]-1)*100:+.1f}%)', 
            color='#2ecc71', linewidth=2)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Initial NAV')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative NAV')
    ax.set_title('Strategy Performance: Cumulative Net Asset Value')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save
    filepath = output_dir / 'cumulative_nav.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_drawdown(baseline_results, enhanced_results, output_dir):
    """Plot drawdown curves."""
    print("Generating drawdown chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    def compute_drawdown(returns):
        nav = (1 + returns).cumprod()
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max
        return drawdown
    
    baseline_dd = compute_drawdown(baseline_results['returns'])
    enhanced_dd = compute_drawdown(enhanced_results['returns'])
    
    # Resample to daily
    baseline_dd_daily = baseline_dd.resample('D').last().dropna()
    enhanced_dd_daily = enhanced_dd.resample('D').last().dropna()
    
    ax.fill_between(baseline_dd_daily.index, baseline_dd_daily.values * 100, 0,
                    alpha=0.3, color='#e74c3c', label=f'Baseline (Max DD: {baseline_dd.min()*100:.1f}%)')
    ax.fill_between(enhanced_dd_daily.index, enhanced_dd_daily.values * 100, 0,
                    alpha=0.5, color='#3498db', label=f'Enhanced (Max DD: {enhanced_dd.min()*100:.1f}%)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Strategy Drawdown Comparison')
    ax.legend(loc='lower left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    
    filepath = output_dir / 'drawdown_curve.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_parameter_sensitivity(returns, prices, output_dir):
    """Plot parameter sensitivity heatmap."""
    print("Generating parameter sensitivity heatmap...")
    
    lookbacks = [3, 6, 12, 24]
    rebalance_periods = [2400, 4800, 9600]
    
    results_matrix = np.zeros((len(lookbacks), len(rebalance_periods)))
    
    for i, lb in enumerate(lookbacks):
        for j, reb in enumerate(rebalance_periods):
            strat = LongShortStrategy(
                signal_type='reversal',
                rebalance_periods=reb,
                signal_params={'lookback': lb},
                long_quantile=0.05,
                short_quantile=0.05,
                use_partial_rebalancing=True,
                min_trade_threshold=0.05,
            )
            res = strat.backtest(returns=returns, prices=prices)
            metrics = calculate_performance_metrics(res['returns'])
            results_matrix[i, j] = metrics['total_return'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(results_matrix, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Total Return (%)', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(rebalance_periods)))
    ax.set_yticks(np.arange(len(lookbacks)))
    ax.set_xticklabels([f'{r//48} days' for r in rebalance_periods])
    ax.set_yticklabels([f'{lb} bars ({lb*5} min)' for lb in lookbacks])
    
    ax.set_xlabel('Rebalance Frequency')
    ax.set_ylabel('Lookback Period')
    ax.set_title('Parameter Sensitivity: Total Return (%)')
    
    # Add text annotations
    for i in range(len(lookbacks)):
        for j in range(len(rebalance_periods)):
            text = ax.text(j, i, f'{results_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    # Highlight best cell
    best_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
    rect = plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1, 
                         fill=False, edgecolor='gold', linewidth=3)
    ax.add_patch(rect)
    
    filepath = output_dir / 'parameter_sensitivity.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_monthly_returns(enhanced_results, output_dir):
    """Plot monthly returns heatmap."""
    print("Generating monthly returns heatmap...")
    
    returns = enhanced_results['returns']
    
    # Resample to monthly
    monthly = returns.resample('ME').sum() * 100  # Convert to percentage
    
    # Create year-month matrix
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Monthly Return (%)', rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title('Enhanced Strategy: Monthly Returns (%)')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j] if j < len(pivot.columns) else np.nan
            if not np.isnan(val):
                color = 'white' if abs(val) > 2.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha="center", va="center", 
                       color=color, fontsize=9)
    
    filepath = output_dir / 'monthly_returns.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    setup_style()
    
    # Create output directory
    output_dir = Path('report/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING REPORT FIGURES")
    print("=" * 70)
    
    # Load data
    prices, returns = load_data()
    
    # Run strategies
    baseline_results, enhanced_results = run_strategies(returns, prices)
    
    # Generate figures
    plot_cumulative_nav(baseline_results, enhanced_results, output_dir)
    plot_drawdown(baseline_results, enhanced_results, output_dir)
    plot_parameter_sensitivity(returns, prices, output_dir)
    plot_monthly_returns(enhanced_results, output_dir)
    
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()


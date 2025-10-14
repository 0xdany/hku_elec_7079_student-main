#!/usr/bin/env python3
"""
Simple Part 1 Test Script

This script provides a quick test of all Part 1 functionality
without generating plots or saving files.

Author: ELEC4546/7079 Course
Date: June 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from data_loader import DataLoader, calculate_daily_returns
from part1_data_analysis import (
    calculate_forward_returns,
    calculate_weekly_returns,
    analyze_return_properties,
    calculate_rolling_volatility,
    build_equal_weight_index,
    calculate_volatility_statistics,
    compare_individual_vs_market,
    calculate_correlation_matrix,
    analyze_correlation_structure
)


def test_task1():
    """Test Task 1 functions."""
    print("Testing Task 1: Target Engineering & Return Calculation")
    print("-" * 55)
    
    # Load data
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    data_daily = loader.load_daily_data()
    
    # Test forward returns calculation
    forward_returns = calculate_forward_returns(data_5min, forward_periods=12)
    print(f"✓ Forward returns shape: {forward_returns.shape}")
    
    # Test weekly returns calculation
    weekly_returns = calculate_weekly_returns(data_daily)
    print(f"✓ Weekly returns shape: {weekly_returns.shape}")
    
    # Test return properties analysis
    return_properties = analyze_return_properties(forward_returns)
    print(f"✓ Return properties analyzed for {len(return_properties)} stocks")
    
    # Show some statistics
    print(f"  - Average daily return: {return_properties['mean'].mean():.6f}")
    print(f"  - Average volatility: {return_properties['std'].mean():.6f}")
    
    return True


def test_task2():
    """Test Task 2 functions."""
    print("\nTesting Task 2: Market & Asset Characterization")
    print("-" * 45)
    
    # Load data
    loader = DataLoader()
    data_daily = loader.load_daily_data()
    daily_returns = calculate_daily_returns(data_daily)
    
    # Test rolling volatility calculation
    rolling_vol = calculate_rolling_volatility(daily_returns, window=20)
    print(f"✓ Rolling volatility shape: {rolling_vol.shape}")
    
    # Test equal-weight index construction
    ew_index = build_equal_weight_index(daily_returns)
    print(f"✓ Equal-weight index length: {len(ew_index)}")
    
    # Test volatility statistics
    vol_stats = calculate_volatility_statistics(daily_returns, ew_index)
    print(f"✓ Volatility statistics calculated")
    print(f"  - Average stock volatility: {vol_stats['avg_stock_volatility']:.4f}")
    print(f"  - Index volatility: {vol_stats['index_volatility']:.4f}")
    print(f"  - Diversification ratio: {vol_stats['diversification_ratio']:.2f}")
    
    # Test individual vs market comparison
    market_comparison = compare_individual_vs_market(daily_returns, ew_index)
    print(f"✓ Market comparison completed for {len(market_comparison)} stocks")
    
    return True


def test_task3():
    """Test Task 3 functions."""
    print("\nTesting Task 3: Cross-Sectional Analysis")
    print("-" * 40)
    
    # Load data
    loader = DataLoader()
    data_daily = loader.load_daily_data()
    daily_returns = calculate_daily_returns(data_daily)
    
    # Test correlation matrix calculation
    corr_matrix = calculate_correlation_matrix(daily_returns)
    print(f"✓ Correlation matrix shape: {corr_matrix.shape}")
    
    # Test correlation structure analysis
    corr_analysis = analyze_correlation_structure(corr_matrix, threshold=0.5)
    print(f"✓ Correlation structure analyzed")
    print(f"  - Average correlation: {corr_analysis['avg_correlation']:.4f}")
    print(f"  - High correlation pairs: {corr_analysis['n_high_corr_pairs']}")
    print(f"  - Positive correlations: {corr_analysis['pct_positive_corr']:.1f}%")
    
    # Test rolling correlation (for a sample pair)
    if len(daily_returns.columns) >= 2:
        stock1 = daily_returns.columns[0]
        stock2 = daily_returns.columns[1]
        from part1_data_analysis import calculate_rolling_correlation
        rolling_corr = calculate_rolling_correlation(
            daily_returns[stock1], 
            daily_returns[stock2], 
            window=60
        )
        print(f"✓ Rolling correlation calculated for {stock1} vs {stock2}")
        print(f"  - Average rolling correlation: {rolling_corr.mean():.4f}")
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("Part 1: Data Analysis & Feature Exploration - Quick Test")
    print("=" * 60)
    
    try:
        # Test all tasks
        test_task1()
        test_task2()
        test_task3()
        
        print("\n" + "=" * 60)
        print("✓ All Part 1 tests completed successfully!")
        print("=" * 60)
        
        print("\nSummary:")
        print("- Task 1: Return calculation and analysis ✓")
        print("- Task 2: Volatility analysis and market index ✓")
        print("- Task 3: Correlation analysis and structure ✓")
        print("\nPart 1 implementation is ready for student use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
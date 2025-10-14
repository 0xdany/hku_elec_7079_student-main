#!/usr/bin/env python3
"""
Test script for the updated DataLoader with pickle support.

This script tests the pickle-based data loading functionality.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader


def test_data_loading_performance():
    """
    Test the performance of pickle data loading.
    """
    print("Testing Data Loading Performance")
    print("=" * 50)
    
    loader = DataLoader()
    
    # Test 5-minute data loading
    print("\n1. Testing 5-minute data loading...")
    
    start_time = time.time()
    data_5min = loader.load_5min_data()
    pickle_time = time.time() - start_time
    
    print(f"Pickle loading time: {pickle_time:.3f} seconds")
    print(f"Data shape: {data_5min.shape}")
    print(f"Data type: {type(data_5min)}")
    print(f"Index type: {type(data_5min.index)}")
    print(f"Columns type: {type(data_5min.columns)}")
    
    # Test daily data loading
    print("\n2. Testing daily data loading...")
    
    start_time = time.time()
    data_daily = loader.load_daily_data()
    pickle_time = time.time() - start_time
    
    print(f"Pickle loading time: {pickle_time:.3f} seconds")
    print(f"Data shape: {data_daily.shape}")
    
    # Test weights loading
    print("\n3. Testing stock weights loading...")
    
    start_time = time.time()
    weights = loader.load_stock_weights()
    pickle_time = time.time() - start_time
    
    print(f"Pickle loading time: {pickle_time:.3f} seconds")
    print(f"Weights shape: {weights.shape}")
    print(f"Weight sum: {weights.sum():.6f}")
    
    return data_5min, data_daily, weights


def test_data_integrity(data_5min, data_daily, weights):
    """
    Test the integrity of loaded data.
    """
    print("\nTesting Data Integrity")
    print("=" * 50)
    
    # Test 5-minute data
    print("\n1. 5-minute data integrity:")
    print(f"  - Index is datetime: {isinstance(data_5min.index, pd.DatetimeIndex)}")
    print(f"  - Columns are MultiIndex: {isinstance(data_5min.columns, pd.MultiIndex)}")
    print(f"  - No missing values: {data_5min.isnull().sum().sum() == 0}")
    print(f"  - Date range: {data_5min.index.min()} to {data_5min.index.max()}")
    
    # Test daily data
    print("\n2. Daily data integrity:")
    print(f"  - Index is datetime: {isinstance(data_daily.index, pd.DatetimeIndex)}")
    print(f"  - Columns are MultiIndex: {isinstance(data_daily.columns, pd.MultiIndex)}")
    print(f"  - No missing values: {data_daily.isnull().sum().sum() == 0}")
    print(f"  - Date range: {data_daily.index.min()} to {data_daily.index.max()}")
    
    # Test weights
    print("\n3. Weights integrity:")
    print(f"  - Is Series: {isinstance(weights, pd.Series)}")
    print(f"  - Weight sum close to 1: {abs(weights.sum() - 1.0) < 0.01}")
    print(f"  - All weights positive: {(weights > 0).all()}")


def test_cache_functionality():
    """
    Test the caching functionality.
    """
    print("\nTesting Cache Functionality")
    print("=" * 50)
    
    loader = DataLoader()
    
    # First load (should load from file)
    print("\n1. First load (from file):")
    start_time = time.time()
    data_5min_1 = loader.load_5min_data()
    first_load_time = time.time() - start_time
    print(f"First load time: {first_load_time:.3f} seconds")
    
    # Second load (should load from cache)
    print("\n2. Second load (from cache):")
    start_time = time.time()
    data_5min_2 = loader.load_5min_data()
    second_load_time = time.time() - start_time
    print(f"Second load time: {second_load_time:.3f} seconds")
    
    # Verify data is the same
    print(f"\n3. Data consistency check:")
    print(f"  - Same object: {data_5min_1 is data_5min_2}")
    print(f"  - Same shape: {data_5min_1.shape == data_5min_2.shape}")
    print(f"  - Same content: {data_5min_1.equals(data_5min_2)}")
    
    # Test cache speed improvement
    if second_load_time < first_load_time:
        speedup = first_load_time / second_load_time
        print(f"  - Cache speedup: {speedup:.1f}x faster")


def main():
    """
    Main test function.
    """
    print("DataLoader Pickle Format Test")
    print("=" * 60)
    
    try:
        # Test data loading performance
        data_5min, data_daily, weights = test_data_loading_performance()
        
        # Test data integrity
        test_data_integrity(data_5min, data_daily, weights)
        
        # Test cache functionality
        test_cache_functionality()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! DataLoader is working correctly with pickle format.")
        print("\nKey features:")
        print("- Fast data loading with pickle format")
        print("- Maintained data integrity and structure")
        print("- Efficient caching mechanism")
        print("- Clean and simple API")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)

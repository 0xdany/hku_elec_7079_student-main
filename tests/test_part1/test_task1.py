"""
Unit Tests for Task 1: Target Engineering & Return Calculation

This module contains unit tests for the functions implemented in task1_returns.py

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from part1_data_analysis.task1_returns import (
    calculate_forward_returns,
    calculate_weekly_returns,
    plot_return_distribution,
    analyze_return_properties
)


class TestTask1Returns:
    """Test class for Task 1 return calculation functions."""
    
    @pytest.fixture
    def sample_5min_data(self):
        """Create sample 5-minute data for testing."""
        # Create sample date range (5-minute intervals)
        dates = pd.date_range('2019-01-02 09:30:00', '2019-01-02 15:00:00', freq='5T')
        
        # Sample stock symbols and fields
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        fields = ['open_px', 'high_px', 'low_px', 'close_px', 'volume', 'vwap']
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_product([stocks, fields])
        
        # Generate sample price data with some trend
        np.random.seed(42)
        n_rows = len(dates)
        n_cols = len(columns)
        
        # Base prices around 100
        base_price = 100
        data = np.zeros((n_rows, n_cols))
        
        for i, stock in enumerate(stocks):
            # Generate price series with random walk
            price_changes = np.random.normal(0, 0.001, n_rows)  # Small price changes
            prices = base_price + (i * 10) + np.cumsum(price_changes)  # Different base prices
            
            # Fill in OHLC data
            for j, field in enumerate(['open_px', 'high_px', 'low_px', 'close_px']):
                col_idx = columns.get_loc((stock, field))
                if field == 'close_px':
                    data[:, col_idx] = prices
                elif field == 'open_px':
                    data[:, col_idx] = prices + np.random.normal(0, 0.0005, n_rows)
                elif field == 'high_px':
                    data[:, col_idx] = prices + np.abs(np.random.normal(0, 0.001, n_rows))
                else:  # low_px
                    data[:, col_idx] = prices - np.abs(np.random.normal(0, 0.001, n_rows))
            
            # Add volume and vwap
            vol_idx = columns.get_loc((stock, 'volume'))
            vwap_idx = columns.get_loc((stock, 'vwap'))
            data[:, vol_idx] = np.random.exponential(1000, n_rows)
            data[:, vwap_idx] = prices + np.random.normal(0, 0.0002, n_rows)
        
        return pd.DataFrame(data, index=dates, columns=columns)
    
    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily data for testing."""
        # Create sample date range (daily)
        dates = pd.date_range('2019-01-02', '2019-12-31', freq='D')
        
        # Sample stock symbols and fields
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        fields = ['open_px', 'high_px', 'low_px', 'close_px', 'volume', 'vwap']
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_product([stocks, fields])
        
        # Generate sample price data
        np.random.seed(42)
        n_rows = len(dates)
        n_cols = len(columns)
        
        base_price = 100
        data = np.zeros((n_rows, n_cols))
        
        for i, stock in enumerate(stocks):
            # Generate price series with random walk
            daily_returns = np.random.normal(0.001, 0.02, n_rows)  # Daily returns
            prices = base_price + (i * 10)
            price_series = [prices]
            
            for ret in daily_returns[1:]:
                prices = prices * (1 + ret)
                price_series.append(prices)
            
            price_series = np.array(price_series)
            
            # Fill in OHLC data
            for j, field in enumerate(['open_px', 'high_px', 'low_px', 'close_px']):
                col_idx = columns.get_loc((stock, field))
                if field == 'close_px':
                    data[:, col_idx] = price_series
                elif field == 'open_px':
                    data[:, col_idx] = price_series * (1 + np.random.normal(0, 0.001, n_rows))
                elif field == 'high_px':
                    data[:, col_idx] = price_series * (1 + np.abs(np.random.normal(0, 0.005, n_rows)))
                else:  # low_px
                    data[:, col_idx] = price_series * (1 - np.abs(np.random.normal(0, 0.005, n_rows)))
            
            # Add volume and vwap
            vol_idx = columns.get_loc((stock, 'volume'))
            vwap_idx = columns.get_loc((stock, 'vwap'))
            data[:, vol_idx] = np.random.exponential(10000, n_rows)
            data[:, vwap_idx] = price_series * (1 + np.random.normal(0, 0.0005, n_rows))
        
        return pd.DataFrame(data, index=dates, columns=columns)
    
    def test_calculate_forward_returns_basic(self, sample_5min_data):
        """Test basic forward returns calculation."""
        # Test with default parameters
        forward_returns = calculate_forward_returns(sample_5min_data)
        
        # Check that result is DataFrame
        assert isinstance(forward_returns, pd.DataFrame)
        
        # Check dimensions
        expected_stocks = sample_5min_data.columns.get_level_values(0).unique()
        assert list(forward_returns.columns) == list(expected_stocks)
        assert len(forward_returns) == len(sample_5min_data)
        
        # Check that last 12 rows are NaN (due to forward looking)
        assert forward_returns.iloc[-12:].isnull().all().all()
        
        # Check that some non-NaN values exist
        assert not forward_returns.iloc[:-12].isnull().all().all()
    
    def test_calculate_forward_returns_custom_periods(self, sample_5min_data):
        """Test forward returns calculation with custom periods."""
        forward_periods = 6  # 30 minutes
        forward_returns = calculate_forward_returns(sample_5min_data, forward_periods=forward_periods)
        
        # Check that last 6 rows are NaN
        assert forward_returns.iloc[-6:].isnull().all().all()
        
        # Check that we have more non-NaN values than with default periods
        non_nan_count = forward_returns.count().sum()
        assert non_nan_count > 0
    
    def test_calculate_forward_returns_values(self, sample_5min_data):
        """Test that forward returns are calculated correctly."""
        forward_periods = 2  # Simple test with 2 periods
        forward_returns = calculate_forward_returns(sample_5min_data, forward_periods=forward_periods)
        
        # Manually calculate expected return for first stock, first time period
        stock_1_prices = sample_5min_data[('STOCK_1', 'close_px')]
        expected_return = (stock_1_prices.iloc[2] / stock_1_prices.iloc[0]) - 1
        actual_return = forward_returns['STOCK_1'].iloc[0]
        
        # Check that calculated return matches expected (within tolerance)
        assert abs(actual_return - expected_return) < 1e-10
    
    def test_calculate_weekly_returns_basic(self, sample_daily_data):
        """Test basic weekly returns calculation."""
        weekly_returns = calculate_weekly_returns(sample_daily_data)
        
        # Check that result is DataFrame
        assert isinstance(weekly_returns, pd.DataFrame)
        
        # Check dimensions
        expected_stocks = sample_daily_data.columns.get_level_values(0).unique()
        assert list(weekly_returns.columns) == list(expected_stocks)
        
        # Check that we have fewer rows than daily data (weekly aggregation)
        assert len(weekly_returns) < len(sample_daily_data)
        
        # Check that index is datetime
        assert isinstance(weekly_returns.index, pd.DatetimeIndex)
    
    def test_calculate_weekly_returns_values(self, sample_daily_data):
        """Test that weekly returns are reasonable."""
        weekly_returns = calculate_weekly_returns(sample_daily_data)
        
        # Check that returns are not all zero
        assert not (weekly_returns == 0).all().all()
        
        # Check that returns are reasonable (not extremely large)
        assert (weekly_returns.abs() < 1.0).all().all()  # Less than 100% weekly return
    
    @patch('matplotlib.pyplot.show')
    def test_plot_return_distribution_basic(self, mock_show, sample_5min_data):
        """Test basic return distribution plotting."""
        # Calculate forward returns first
        forward_returns = calculate_forward_returns(sample_5min_data)
        
        # Test plotting
        sample_stocks = ['STOCK_1', 'STOCK_2']
        stats = plot_return_distribution(forward_returns, sample_stocks)
        
        # Check that statistics are returned
        assert isinstance(stats, dict)
        assert len(stats) == len(sample_stocks)
        
        # Check that each stock has expected statistics
        for stock in sample_stocks:
            assert stock in stats
            stock_stats = stats[stock]
            
            # Check for expected keys
            expected_keys = ['mean', 'std', 'skewness', 'kurtosis', 'jarque_bera_pvalue', 'count']
            for key in expected_keys:
                assert key in stock_stats
                assert isinstance(stock_stats[key], (int, float))
    
    def test_plot_return_distribution_invalid_stocks(self, sample_5min_data):
        """Test plotting with invalid stock symbols."""
        forward_returns = calculate_forward_returns(sample_5min_data)
        
        # Test with non-existent stocks
        invalid_stocks = ['INVALID_STOCK']
        
        with pytest.raises(ValueError):
            plot_return_distribution(forward_returns, invalid_stocks)
    
    def test_analyze_return_properties_basic(self, sample_5min_data):
        """Test basic return properties analysis."""
        forward_returns = calculate_forward_returns(sample_5min_data)
        
        summary = analyze_return_properties(forward_returns)
        
        # Check that result is DataFrame
        assert isinstance(summary, pd.DataFrame)
        
        # Check that all stocks are included
        expected_stocks = forward_returns.columns
        assert len(summary) == len(expected_stocks)
        
        # Check for expected columns
        expected_columns = ['count', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 
                          'q25', 'q50', 'q75', 'jarque_bera_pvalue', 'is_normal']
        for col in expected_columns:
            assert col in summary.columns
    
    def test_analyze_return_properties_values(self, sample_5min_data):
        """Test that return properties analysis produces reasonable values."""
        forward_returns = calculate_forward_returns(sample_5min_data)
        summary = analyze_return_properties(forward_returns)
        
        # Check that means are reasonable (not extremely large)
        assert (summary['mean'].abs() < 0.1).all()  # Less than 10% mean return per period
        
        # Check that standard deviations are positive
        assert (summary['std'] > 0).all()
        
        # Check that counts are reasonable
        assert (summary['count'] > 0).all()
        assert (summary['count'] <= len(forward_returns)).all()
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        result = analyze_return_properties(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
        # Test with single column
        single_col_data = pd.DataFrame({'A': [0.01, 0.02, -0.01, 0.005]})
        result = analyze_return_properties(single_col_data)
        assert len(result) == 1
        assert 'A' in result.index
    
    def test_missing_data_handling(self, sample_5min_data):
        """Test handling of missing data."""
        # Introduce some missing data
        data_with_nan = sample_5min_data.copy()
        data_with_nan.iloc[10:15, 0] = np.nan  # Set some values to NaN
        
        # Test forward returns calculation
        forward_returns = calculate_forward_returns(data_with_nan)
        
        # Should still produce results
        assert isinstance(forward_returns, pd.DataFrame)
        assert len(forward_returns) == len(data_with_nan)
        
        # Test return analysis
        summary = analyze_return_properties(forward_returns)
        assert isinstance(summary, pd.DataFrame)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])



"""
Unit Tests for Task 2: Market & Asset Characterization

This module contains unit tests for the functions implemented in task2_volatility.py

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

from part1_data_analysis.task2_volatility import (
    calculate_rolling_volatility,
    build_equal_weight_index,
    plot_volatility_analysis,
    calculate_volatility_statistics,
    compare_individual_vs_market
)


class TestTask2Volatility:
    """Test class for Task 2 volatility analysis functions."""
    
    @pytest.fixture
    def sample_daily_returns(self):
        """Create sample daily returns for testing."""
        # Create sample date range
        dates = pd.date_range('2019-01-02', '2020-12-31', freq='D')
        
        # Generate sample returns for 5 stocks
        np.random.seed(42)
        n_stocks = 5
        stock_names = [f'STOCK_{i+1}' for i in range(n_stocks)]
        
        # Generate correlated returns
        base_returns = np.random.normal(0.0005, 0.015, len(dates))  # Market factor
        
        returns_data = {}
        for i, stock in enumerate(stock_names):
            # Each stock has some correlation with market + idiosyncratic component
            stock_returns = (
                0.7 * base_returns +  # Market component
                0.3 * np.random.normal(0, 0.01, len(dates))  # Idiosyncratic component
            )
            returns_data[stock] = stock_returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def sample_high_vol_returns(self):
        """Create sample returns with time-varying volatility."""
        dates = pd.date_range('2019-01-02', '2020-12-31', freq='D')
        
        # Create regime-switching volatility
        n_days = len(dates)
        volatility = np.ones(n_days) * 0.01  # Base volatility
        
        # High volatility period in the middle
        high_vol_start = n_days // 3
        high_vol_end = 2 * n_days // 3
        volatility[high_vol_start:high_vol_end] = 0.03
        
        # Generate returns with time-varying volatility
        np.random.seed(42)
        returns = np.random.normal(0, 1, n_days) * volatility
        
        return pd.DataFrame({'STOCK_1': returns}, index=dates)
    
    def test_calculate_rolling_volatility_basic(self, sample_daily_returns):
        """Test basic rolling volatility calculation."""
        window = 20
        rolling_vol = calculate_rolling_volatility(sample_daily_returns, window=window)
        
        # Check that result is DataFrame
        assert isinstance(rolling_vol, pd.DataFrame)
        
        # Check dimensions
        assert rolling_vol.shape == sample_daily_returns.shape
        assert list(rolling_vol.columns) == list(sample_daily_returns.columns)
        
        # Check that first (window-1) rows are NaN
        assert rolling_vol.iloc[:window-1].isnull().all().all()
        
        # Check that subsequent rows have values
        assert not rolling_vol.iloc[window:].isnull().all().all()
    
    def test_calculate_rolling_volatility_annualized(self, sample_daily_returns):
        """Test annualized vs non-annualized volatility."""
        window = 20
        
        # Calculate both annualized and non-annualized
        vol_annualized = calculate_rolling_volatility(sample_daily_returns, window=window, annualize=True)
        vol_not_annualized = calculate_rolling_volatility(sample_daily_returns, window=window, annualize=False)
        
        # Annualized should be roughly sqrt(252) times larger
        scaling_factor = np.sqrt(252)
        
        # Compare non-NaN values
        valid_mask = ~vol_not_annualized.isnull()
        ratio = (vol_annualized[valid_mask] / vol_not_annualized[valid_mask]).mean().mean()
        
        # Should be close to sqrt(252) ≈ 15.87
        assert abs(ratio - scaling_factor) < 1.0
    
    def test_calculate_rolling_volatility_values(self, sample_high_vol_returns):
        """Test that rolling volatility captures volatility changes."""
        window = 10
        rolling_vol = calculate_rolling_volatility(sample_high_vol_returns, window=window, annualize=False)
        
        # The high volatility period should show higher rolling volatility
        n_days = len(sample_high_vol_returns)
        high_vol_start = n_days // 3
        high_vol_end = 2 * n_days // 3
        
        # Get volatility values for different periods
        pre_high_vol = rolling_vol.iloc[window:high_vol_start]['STOCK_1'].mean()
        during_high_vol = rolling_vol.iloc[high_vol_start+window:high_vol_end]['STOCK_1'].mean()
        post_high_vol = rolling_vol.iloc[high_vol_end+window:]['STOCK_1'].mean()
        
        # During high vol period should have higher volatility
        assert during_high_vol > pre_high_vol
        assert during_high_vol > post_high_vol
    
    def test_build_equal_weight_index_basic(self, sample_daily_returns):
        """Test basic equal-weight index construction."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        
        # Check that result is Series
        assert isinstance(ew_index, pd.Series)
        
        # Check dimensions
        assert len(ew_index) == len(sample_daily_returns)
        assert ew_index.index.equals(sample_daily_returns.index)
        
        # Check that values are reasonable
        assert not ew_index.isnull().all()
    
    def test_build_equal_weight_index_calculation(self, sample_daily_returns):
        """Test that equal-weight index is calculated correctly."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        
        # Manually calculate expected values for first few rows
        expected_values = sample_daily_returns.mean(axis=1, skipna=True)
        
        # Compare with calculated index
        pd.testing.assert_series_equal(ew_index, expected_values)
    
    def test_build_equal_weight_index_missing_data(self):
        """Test equal-weight index with missing data."""
        # Create data with missing values
        dates = pd.date_range('2019-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'STOCK_1': [0.01, 0.02, np.nan, 0.01, 0.02, 0.01, np.nan, 0.01, 0.02, 0.01],
            'STOCK_2': [0.02, np.nan, 0.01, 0.02, np.nan, 0.02, 0.01, 0.02, np.nan, 0.02],
            'STOCK_3': [0.01, 0.01, 0.01, np.nan, 0.01, np.nan, 0.01, 0.01, 0.01, np.nan]
        }, index=dates)
        
        ew_index = build_equal_weight_index(data)
        
        # Should handle missing data by averaging available stocks
        assert isinstance(ew_index, pd.Series)
        assert len(ew_index) == len(data)
        
        # Check that no values are NaN (should average available stocks)
        assert not ew_index.isnull().any()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_volatility_analysis_basic(self, mock_show, sample_daily_returns):
        """Test basic volatility analysis plotting."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        sample_stocks = ['STOCK_1', 'STOCK_2']
        
        # Should run without errors
        plot_volatility_analysis(sample_daily_returns, sample_stocks, ew_index)
        
        # Check that show was called (plot was created)
        mock_show.assert_called_once()
    
    def test_plot_volatility_analysis_invalid_stocks(self, sample_daily_returns):
        """Test plotting with invalid stock symbols."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        invalid_stocks = ['INVALID_STOCK']
        
        with pytest.raises(ValueError):
            plot_volatility_analysis(sample_daily_returns, invalid_stocks, ew_index)
    
    def test_calculate_volatility_statistics_basic(self, sample_daily_returns):
        """Test basic volatility statistics calculation."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        stats = calculate_volatility_statistics(sample_daily_returns, ew_index)
        
        # Check that result is dictionary
        assert isinstance(stats, dict)
        
        # Check for expected keys
        expected_keys = [
            'avg_stock_volatility', 'median_stock_volatility', 'min_stock_volatility',
            'max_stock_volatility', 'vol_of_volatilities', 'index_volatility',
            'index_vol_std', 'diversification_ratio', 'n_stocks_analyzed'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
    
    def test_calculate_volatility_statistics_diversification(self, sample_daily_returns):
        """Test that diversification ratio makes sense."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        stats = calculate_volatility_statistics(sample_daily_returns, ew_index)
        
        # Diversification ratio should be > 1 (individual stocks more volatile than index)
        assert stats['diversification_ratio'] > 1.0
        
        # Index volatility should be less than average stock volatility
        assert stats['index_volatility'] < stats['avg_stock_volatility']
    
    def test_compare_individual_vs_market_basic(self, sample_daily_returns):
        """Test basic individual vs market comparison."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        comparison = compare_individual_vs_market(sample_daily_returns, ew_index)
        
        # Check that result is DataFrame
        assert isinstance(comparison, pd.DataFrame)
        
        # Check dimensions
        assert len(comparison) <= len(sample_daily_returns.columns)  # May skip stocks with insufficient data
        
        # Check for expected columns
        expected_columns = [
            'annualized_return', 'annualized_volatility', 'beta', 'alpha',
            'correlation', 'sharpe_ratio', 'information_ratio', 'tracking_error', 'observations'
        ]
        
        for col in expected_columns:
            assert col in comparison.columns
    
    def test_compare_individual_vs_market_values(self, sample_daily_returns):
        """Test that comparison values are reasonable."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        comparison = compare_individual_vs_market(sample_daily_returns, ew_index)
        
        if len(comparison) > 0:
            # Beta should be reasonable (typically between 0.5 and 2.0 for most stocks)
            assert (comparison['beta'].abs() < 5.0).all()
            
            # Correlation should be between -1 and 1
            assert (comparison['correlation'].abs() <= 1.0).all()
            
            # Volatility should be positive
            assert (comparison['annualized_volatility'] > 0).all()
            
            # Observations should be positive
            assert (comparison['observations'] > 0).all()
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            ew_index = build_equal_weight_index(empty_df)
            assert len(ew_index) == 0
        except Exception:
            pass  # Empty data might raise exception, which is acceptable
        
        # Test with single stock
        single_stock_data = pd.DataFrame({
            'STOCK_1': np.random.normal(0.001, 0.02, 100)
        }, index=pd.date_range('2019-01-01', periods=100))
        
        ew_index = build_equal_weight_index(single_stock_data)
        assert len(ew_index) == len(single_stock_data)
        
        # For single stock, index should equal the stock returns
        pd.testing.assert_series_equal(ew_index, single_stock_data['STOCK_1'])
    
    def test_rolling_volatility_different_windows(self, sample_daily_returns):
        """Test rolling volatility with different window sizes."""
        windows = [5, 10, 20, 60]
        
        for window in windows:
            rolling_vol = calculate_rolling_volatility(sample_daily_returns, window=window)
            
            # Check that first (window-1) rows are NaN
            assert rolling_vol.iloc[:window-1].isnull().all().all()
            
            # Check that subsequent rows have values
            if len(sample_daily_returns) > window:
                assert not rolling_vol.iloc[window:].isnull().all().all()
    
    def test_volatility_statistics_consistency(self, sample_daily_returns):
        """Test consistency of volatility statistics."""
        ew_index = build_equal_weight_index(sample_daily_returns)
        stats = calculate_volatility_statistics(sample_daily_returns, ew_index)
        
        # Min should be <= median <= max
        assert stats['min_stock_volatility'] <= stats['median_stock_volatility']
        assert stats['median_stock_volatility'] <= stats['max_stock_volatility']
        
        # Number of stocks analyzed should match input
        assert stats['n_stocks_analyzed'] == len(sample_daily_returns.columns)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])



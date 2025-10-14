"""
Unit Tests for Task 3: Cross-Sectional Analysis

This module contains unit tests for the functions implemented in task3_correlation.py

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

from part1_data_analysis.task3_correlation import (
    calculate_correlation_matrix,
    calculate_rolling_correlation,
    plot_correlation_heatmap,
    plot_rolling_correlation_analysis,
    analyze_correlation_structure,
    plot_correlation_distribution
)


class TestTask3Correlation:
    """Test class for Task 3 correlation analysis functions."""
    
    @pytest.fixture
    def sample_daily_returns(self):
        """Create sample daily returns with known correlation structure."""
        dates = pd.date_range('2019-01-02', '2020-12-31', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        
        # Create base market factor
        market_factor = np.random.normal(0, 0.015, n_days)
        
        # Create stocks with different correlations to market
        returns_data = {}
        
        # High correlation stock
        returns_data['STOCK_1'] = (
            0.8 * market_factor + 0.2 * np.random.normal(0, 0.01, n_days)
        )
        
        # Medium correlation stock
        returns_data['STOCK_2'] = (
            0.5 * market_factor + 0.5 * np.random.normal(0, 0.012, n_days)
        )
        
        # Low correlation stock
        returns_data['STOCK_3'] = (
            0.2 * market_factor + 0.8 * np.random.normal(0, 0.008, n_days)
        )
        
        # Negatively correlated stock
        returns_data['STOCK_4'] = (
            -0.3 * market_factor + 0.7 * np.random.normal(0, 0.01, n_days)
        )
        
        # Independent stock
        returns_data['STOCK_5'] = np.random.normal(0, 0.01, n_days)
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def sample_time_varying_returns(self):
        """Create sample returns with time-varying correlation."""
        dates = pd.date_range('2019-01-02', '2020-12-31', freq='D')
        n_days = len(dates)
        
        np.random.seed(42)
        
        # Create regime-switching correlation
        stock1_returns = np.random.normal(0, 0.01, n_days)
        stock2_returns = np.zeros(n_days)
        
        # First half: positive correlation
        mid_point = n_days // 2
        stock2_returns[:mid_point] = (
            0.7 * stock1_returns[:mid_point] + 
            0.3 * np.random.normal(0, 0.01, mid_point)
        )
        
        # Second half: negative correlation
        stock2_returns[mid_point:] = (
            -0.7 * stock1_returns[mid_point:] + 
            0.3 * np.random.normal(0, 0.01, n_days - mid_point)
        )
        
        return pd.DataFrame({
            'STOCK_A': stock1_returns,
            'STOCK_B': stock2_returns
        }, index=dates)
    
    def test_calculate_correlation_matrix_basic(self, sample_daily_returns):
        """Test basic correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        
        # Check that result is DataFrame
        assert isinstance(corr_matrix, pd.DataFrame)
        
        # Check dimensions (should be square)
        n_stocks = len(sample_daily_returns.columns)
        assert corr_matrix.shape == (n_stocks, n_stocks)
        
        # Check that diagonal is all 1s
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(n_stocks))
        
        # Check that matrix is symmetric
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.T.values)
        
        # Check that all values are between -1 and 1
        assert (corr_matrix.abs() <= 1.0).all().all()
    
    def test_calculate_correlation_matrix_methods(self, sample_daily_returns):
        """Test correlation matrix with different methods."""
        methods = ['pearson', 'spearman', 'kendall']
        
        for method in methods:
            corr_matrix = calculate_correlation_matrix(sample_daily_returns, method=method)
            
            # Should produce valid correlation matrix
            assert isinstance(corr_matrix, pd.DataFrame)
            assert corr_matrix.shape[0] == corr_matrix.shape[1]
            assert (corr_matrix.abs() <= 1.0).all().all()
    
    def test_calculate_correlation_matrix_values(self, sample_daily_returns):
        """Test that correlation matrix captures expected relationships."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        
        # STOCK_1 and STOCK_2 should be positively correlated (both correlated with market)
        assert corr_matrix.loc['STOCK_1', 'STOCK_2'] > 0.2
        
        # STOCK_1 and STOCK_4 should have lower/negative correlation
        assert corr_matrix.loc['STOCK_1', 'STOCK_4'] < corr_matrix.loc['STOCK_1', 'STOCK_2']
        
        # STOCK_5 (independent) should have lower correlations with others
        stock5_corrs = corr_matrix.loc['STOCK_5'].drop('STOCK_5')  # Exclude self-correlation
        other_avg_corr = corr_matrix.drop('STOCK_5', axis=0).drop('STOCK_5', axis=1).abs().mean().mean()
        assert stock5_corrs.abs().mean() < other_avg_corr
    
    def test_calculate_rolling_correlation_basic(self, sample_daily_returns):
        """Test basic rolling correlation calculation."""
        stock1 = sample_daily_returns['STOCK_1']
        stock2 = sample_daily_returns['STOCK_2']
        
        window = 60
        rolling_corr = calculate_rolling_correlation(stock1, stock2, window=window)
        
        # Check that result is Series
        assert isinstance(rolling_corr, pd.Series)
        
        # Check dimensions
        assert len(rolling_corr) == len(stock1)
        
        # Check that first (window-1) values are NaN
        assert rolling_corr.iloc[:window-1].isnull().all()
        
        # Check that subsequent values are not all NaN
        assert not rolling_corr.iloc[window:].isnull().all()
        
        # Check that values are between -1 and 1
        valid_corrs = rolling_corr.dropna()
        assert (valid_corrs.abs() <= 1.0).all()
    
    def test_calculate_rolling_correlation_time_varying(self, sample_time_varying_returns):
        """Test rolling correlation captures time-varying relationships."""
        stock_a = sample_time_varying_returns['STOCK_A']
        stock_b = sample_time_varying_returns['STOCK_B']
        
        window = 30
        rolling_corr = calculate_rolling_correlation(stock_a, stock_b, window=window)
        
        # Get correlations for different periods
        n_days = len(sample_time_varying_returns)
        mid_point = n_days // 2
        
        # First half should be positive correlation
        first_half_corr = rolling_corr.iloc[window:mid_point].mean()
        
        # Second half should be negative correlation
        second_half_corr = rolling_corr.iloc[mid_point+window:].mean()
        
        # Should detect the regime change
        assert first_half_corr > 0.3
        assert second_half_corr < -0.3
        assert first_half_corr > second_half_corr
    
    def test_calculate_rolling_correlation_parameters(self, sample_daily_returns):
        """Test rolling correlation with different parameters."""
        stock1 = sample_daily_returns['STOCK_1']
        stock2 = sample_daily_returns['STOCK_2']
        
        # Test different window sizes
        for window in [20, 40, 60]:
            rolling_corr = calculate_rolling_correlation(stock1, stock2, window=window)
            assert rolling_corr.iloc[:window-1].isnull().all()
            assert not rolling_corr.iloc[window:].isnull().all()
        
        # Test with custom min_periods
        rolling_corr = calculate_rolling_correlation(stock1, stock2, window=60, min_periods=30)
        # Should have fewer NaN values at the beginning
        assert rolling_corr.iloc[30:60].isnull().sum() < 30
    
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_heatmap_basic(self, mock_show, sample_daily_returns):
        """Test basic correlation heatmap plotting."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        
        # Should run without errors
        plot_correlation_heatmap(corr_matrix)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_rolling_correlation_analysis_basic(self, mock_show, sample_daily_returns):
        """Test basic rolling correlation analysis plotting."""
        stock_pairs = [('STOCK_1', 'STOCK_2'), ('STOCK_3', 'STOCK_4')]
        
        rolling_corrs = plot_rolling_correlation_analysis(sample_daily_returns, stock_pairs)
        
        # Check that result is dictionary
        assert isinstance(rolling_corrs, dict)
        assert len(rolling_corrs) == len(stock_pairs)
        
        # Check that each pair has a Series
        for pair in stock_pairs:
            assert pair in rolling_corrs
            assert isinstance(rolling_corrs[pair], pd.Series)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    def test_plot_rolling_correlation_analysis_invalid_pairs(self, sample_daily_returns):
        """Test rolling correlation analysis with invalid stock pairs."""
        invalid_pairs = [('INVALID_1', 'INVALID_2')]
        
        with pytest.raises(ValueError):
            plot_rolling_correlation_analysis(sample_daily_returns, invalid_pairs)
    
    def test_analyze_correlation_structure_basic(self, sample_daily_returns):
        """Test basic correlation structure analysis."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        analysis = analyze_correlation_structure(corr_matrix)
        
        # Check that result is dictionary
        assert isinstance(analysis, dict)
        
        # Check for expected keys
        expected_keys = [
            'avg_correlation', 'median_correlation', 'std_correlation',
            'min_correlation', 'max_correlation', 'n_stocks',
            'high_corr_pairs', 'n_high_corr_pairs', 'pct_high_corr',
            'pct_positive_corr', 'pct_negative_corr', 'q25', 'q75'
        ]
        
        for key in expected_keys:
            assert key in analysis
    
    def test_analyze_correlation_structure_values(self, sample_daily_returns):
        """Test that correlation structure analysis produces reasonable values."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        analysis = analyze_correlation_structure(corr_matrix, threshold=0.3)
        
        # Basic checks
        assert -1 <= analysis['avg_correlation'] <= 1
        assert -1 <= analysis['min_correlation'] <= 1
        assert -1 <= analysis['max_correlation'] <= 1
        assert analysis['n_stocks'] == len(sample_daily_returns.columns)
        
        # Percentages should be between 0 and 100
        assert 0 <= analysis['pct_positive_corr'] <= 100
        assert 0 <= analysis['pct_negative_corr'] <= 100
        assert abs(analysis['pct_positive_corr'] + analysis['pct_negative_corr'] - 100) < 1e-6
        
        # High correlation pairs should be reasonable
        assert analysis['n_high_corr_pairs'] >= 0
        assert 0 <= analysis['pct_high_corr'] <= 100
        
        # Check high correlation pairs structure
        if analysis['n_high_corr_pairs'] > 0:
            high_corr_pairs = analysis['high_corr_pairs']
            assert isinstance(high_corr_pairs, list)
            
            for pair in high_corr_pairs:
                assert isinstance(pair, dict)
                assert 'stock1' in pair and 'stock2' in pair and 'correlation' in pair
                assert abs(pair['correlation']) >= 0.3  # Above threshold
    
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_distribution_basic(self, mock_show, sample_daily_returns):
        """Test basic correlation distribution plotting."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        
        # Should run without errors
        plot_correlation_distribution(corr_matrix)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single stock
        single_stock_data = pd.DataFrame({
            'STOCK_1': np.random.normal(0, 0.01, 100)
        })
        
        corr_matrix = calculate_correlation_matrix(single_stock_data)
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix.iloc[0, 0] == 1.0
        
        # Test with perfectly correlated stocks
        perfect_corr_data = pd.DataFrame({
            'STOCK_1': [0.01, 0.02, -0.01, 0.005],
            'STOCK_2': [0.02, 0.04, -0.02, 0.01]  # Exactly 2x STOCK_1
        })
        
        corr_matrix = calculate_correlation_matrix(perfect_corr_data)
        assert abs(corr_matrix.loc['STOCK_1', 'STOCK_2'] - 1.0) < 1e-10
    
    def test_missing_data_handling(self, sample_daily_returns):
        """Test handling of missing data in correlation calculations."""
        # Introduce missing data
        data_with_nan = sample_daily_returns.copy()
        data_with_nan.iloc[10:20, 0] = np.nan  # Missing data in first stock
        data_with_nan.iloc[50:60, 1] = np.nan  # Missing data in second stock
        
        # Test correlation matrix calculation
        corr_matrix = calculate_correlation_matrix(data_with_nan)
        
        # Should still produce valid correlation matrix
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert (corr_matrix.abs() <= 1.0).all().all()
        
        # Test rolling correlation with missing data
        stock1 = data_with_nan.iloc[:, 0]
        stock2 = data_with_nan.iloc[:, 1]
        
        rolling_corr = calculate_rolling_correlation(stock1, stock2, window=30)
        assert isinstance(rolling_corr, pd.Series)
    
    def test_correlation_symmetry_and_properties(self, sample_daily_returns):
        """Test mathematical properties of correlation calculations."""
        corr_matrix = calculate_correlation_matrix(sample_daily_returns)
        
        # Test symmetry
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10
        
        # Test that diagonal elements are 1
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
        
        # Test rolling correlation properties
        stock1 = sample_daily_returns.iloc[:, 0]
        stock2 = sample_daily_returns.iloc[:, 1]
        
        # Self-correlation should be 1
        rolling_self_corr = calculate_rolling_correlation(stock1, stock1, window=30)
        valid_self_corrs = rolling_self_corr.dropna()
        assert (abs(valid_self_corrs - 1.0) < 1e-10).all()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])



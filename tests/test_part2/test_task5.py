"""
Task 5 Unit Tests

Test cases for information coefficient (IC) analysis functionality.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from part2_alpha_modeling.task5_ic_analysis import (
    calculate_information_coefficient,
    calculate_ic_for_multiple_factors,
    analyze_ic_stability,
    analyze_ic_decay,
    calculate_half_life,
    rank_factors_by_ic,
    plot_ic_analysis,
    generate_ic_report
)


class TestInformationCoefficient:
    """Tests for Information Coefficient (IC) calculation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock factor data - single factor time series
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        
        np.random.seed(42)
        
        # Create factor time series (partially correlated with future returns)
        factor_values = []
        forward_returns_values = []
        
        for i, date in enumerate(dates):
            # Generate factor value
            factor_value = np.random.normal(0, 1)
            
            # Generate forward return (partially correlated with factor value)
            correlation = 0.3  # correlation strength
            noise = np.random.normal(0, 0.8)
            forward_return = correlation * factor_value + noise
            
            factor_values.append(factor_value)
            forward_returns_values.append(forward_return)
        
        self.factor_series = pd.Series(factor_values, index=dates, name='momentum_3')
        self.forward_returns = pd.Series(forward_returns_values, index=dates, name='returns')
    
    def test_calculate_information_coefficient_cross_sectional(self):
        """Test cross-sectional IC calculation"""
        ic_results = calculate_information_coefficient(
            self.factor_series, 
            self.forward_returns, 
            method='cross_sectional'
        )
        
        # Check return structure
        assert isinstance(ic_results, dict)
        assert 'ic_value' in ic_results
        assert 't_stat' in ic_results
        assert 'p_value' in ic_results
        assert 'n_observations' in ic_results
        assert 'method' in ic_results
        
        # Check IC value range
        assert -1 <= ic_results['ic_value'] <= 1
        
        # Check number of observations
        assert ic_results['n_observations'] > 0
        
        # Check method field
        assert ic_results['method'] == 'cross_sectional'
    
    def test_calculate_information_coefficient_time_series(self):
        """Test time-series IC calculation"""
        ic_results = calculate_information_coefficient(
            self.factor_series, 
            self.forward_returns, 
            method='time_series'
        )
        
        # Check return structure
        assert isinstance(ic_results, dict)
        assert 'ic_value' in ic_results
        assert 't_stat' in ic_results
        assert 'p_value' in ic_results
        assert 'n_observations' in ic_results
        assert 'method' in ic_results
        
        # Check IC value range
        assert -1 <= ic_results['ic_value'] <= 1
        
        # Check method field
        assert ic_results['method'] == 'time_series'
    
    def test_calculate_information_coefficient_with_missing_data(self):
        """Test IC calculation with missing data"""
        # Add missing values
        factor_series_with_missing = self.factor_series.copy()
        factor_series_with_missing.iloc[10:15] = np.nan
        
        returns_with_missing = self.forward_returns.copy()
        returns_with_missing.iloc[10:15] = np.nan
        
        ic_results = calculate_information_coefficient(
            factor_series_with_missing, 
            returns_with_missing, 
            method='cross_sectional'
        )
        
        # Ensure missing values are handled
        assert 'ic_value' in ic_results
        assert not pd.isna(ic_results['ic_value'])
    
    def test_calculate_information_coefficient_insufficient_data(self):
        """Test insufficient data case"""
        # Create insufficient data case
        small_factor_series = self.factor_series.iloc[:5]  # only 5 timestamps
        small_returns_series = self.forward_returns.iloc[:5]
        
        with pytest.raises(ValueError, match="Insufficient valid data points"):
            calculate_information_coefficient(
                small_factor_series, 
                small_returns_series, 
                method='cross_sectional'
            )
    
    def test_calculate_information_coefficient_invalid_input(self):
        """Test invalid inputs"""
        # Empty data
        with pytest.raises(ValueError, match="Input series is empty"):
            calculate_information_coefficient(pd.Series(), self.forward_returns)
        
        with pytest.raises(ValueError, match="Input series is empty"):
            calculate_information_coefficient(self.factor_series, pd.Series())
        
        # Non-Series inputs
        with pytest.raises(ValueError, match="Input must be pandas.Series"):
            calculate_information_coefficient(
                pd.DataFrame({'factor': [1, 2, 3]}), 
                self.forward_returns
            )
        
        # Unsupported method
        with pytest.raises(ValueError, match="Unsupported IC calculation method"):
            calculate_information_coefficient(
                self.factor_series, 
                self.forward_returns, 
                method='invalid_method'
            )


class TestMultipleFactorsIC:
    """Tests for multiple-factor IC calculation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock multi-factor data
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3', 'STOCK_4', 'STOCK_5']
        
        np.random.seed(42)
        
        # Create factor data (partially correlated with forward returns)
        factor_data = []
        forward_returns_data = []
        
        for i, date in enumerate(dates):
            # Generate factor values
            factor_values = np.random.normal(0, 1, len(stocks))
            
            # Generate forward returns (partially correlated)
            correlation = 0.3  # correlation strength
            noise = np.random.normal(0, 0.8, len(stocks))
            forward_returns = correlation * factor_values + noise
            
            factor_data.append(factor_values)
            forward_returns_data.append(forward_returns)
        
        self.factor_data = pd.DataFrame(
            factor_data, 
            index=dates, 
            columns=stocks
        )
        
        self.forward_returns = pd.DataFrame(
            forward_returns_data, 
            index=dates, 
            columns=stocks
        )
    
    def test_calculate_ic_for_multiple_factors_basic(self):
        """Test basic multi-factor IC calculation"""
        ic_results = calculate_ic_for_multiple_factors(
            self.factor_data, 
            self.forward_returns, 
            method='cross_sectional'
        )
        
        # Check return structure
        assert isinstance(ic_results, dict)
        assert 'ic_series' in ic_results
        assert 't_stats' in ic_results
        assert 'p_values' in ic_results
        assert 'mean_ic' in ic_results
        assert 'std_ic' in ic_results
        assert 'ir_ratio' in ic_results
        assert 'hit_rate' in ic_results
        
        # Check IC time series
        ic_series = ic_results['ic_series']
        assert isinstance(ic_series, pd.Series)
        assert len(ic_series) == len(self.factor_data)
        
        # Check IC value range
        valid_ic = ic_series.dropna()
        if len(valid_ic) > 0:
            assert valid_ic.min() >= -1
            assert valid_ic.max() <= 1
        
        # Check summary statistics
        assert not pd.isna(ic_results['mean_ic'])
        assert not pd.isna(ic_results['std_ic'])
        assert 0 <= ic_results['hit_rate'] <= 1


class TestICStabilityAnalysis:
    """Tests for IC stability analysis functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock IC time series
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        np.random.seed(42)
        
        # Create IC series with trend and noise
        trend = np.linspace(0.1, 0.05, len(dates))  # downward trend
        noise = np.random.normal(0, 0.1, len(dates))
        ic_series = trend + noise
        
        self.ic_series = pd.Series(ic_series, index=dates)
    
    def test_analyze_ic_stability_basic(self):
        """Test basic IC stability analysis"""
        stability_analysis = analyze_ic_stability(self.ic_series, window=20)
        
        # Check return structure
        assert isinstance(stability_analysis, dict)
        assert 'rolling_mean' in stability_analysis
        assert 'rolling_std' in stability_analysis
        assert 'rolling_ir' in stability_analysis
        assert 'ic_stability' in stability_analysis
        assert 'ic_decay' in stability_analysis
        assert 'ic_distribution' in stability_analysis
        
        # Check rolling statistics
        rolling_mean = stability_analysis['rolling_mean']
        rolling_std = stability_analysis['rolling_std']
        rolling_ir = stability_analysis['rolling_ir']
        
        assert isinstance(rolling_mean, pd.Series)
        assert isinstance(rolling_std, pd.Series)
        assert isinstance(rolling_ir, pd.Series)
        
        # Check stability metrics
        stability_metrics = stability_analysis['ic_stability']
        assert 'mean_ic' in stability_metrics
        assert 'std_ic' in stability_metrics
        assert 'ir_ratio' in stability_metrics
        assert 'hit_rate' in stability_metrics
        assert 'positive_ic_ratio' in stability_metrics
        assert 'negative_ic_ratio' in stability_metrics
        
        # Validate metric ranges
        assert 0 <= stability_metrics['hit_rate'] <= 1
        assert 0 <= stability_metrics['positive_ic_ratio'] <= 1
        assert 0 <= stability_metrics['negative_ic_ratio'] <= 1
    
    def test_analyze_ic_stability_with_different_windows(self):
        """Test IC stability analysis with different windows"""
        # Small window
        stability_small = analyze_ic_stability(self.ic_series, window=10)
        
        # Large window
        stability_large = analyze_ic_stability(self.ic_series, window=50)
        
        # Ensure both return results
        assert 'rolling_mean' in stability_small
        assert 'rolling_mean' in stability_large
        
        # Larger window should be smoother
        small_rolling = stability_small['rolling_mean'].dropna()
        large_rolling = stability_large['rolling_mean'].dropna()
        
        if len(small_rolling) > 0 and len(large_rolling) > 0:
            # Standard deviation should be smaller (smoother) for large window
            assert large_rolling.std() <= small_rolling.std()
    
    def test_analyze_ic_stability_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError, match="IC series is empty"):
            analyze_ic_stability(pd.Series())
        
        with pytest.raises(ValueError, match="Input must be pandas.Series"):
            analyze_ic_stability(pd.DataFrame({'ic': [1, 2, 3]}))


class TestICDecayAnalysis:
    """Tests for IC decay analysis functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create IC series with decay characteristics
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        np.random.seed(42)
        
        # Create an autocorrelated series
        ic_series = np.zeros(100)
        ic_series[0] = 0.1
        
        for i in range(1, 100):
            ic_series[i] = 0.8 * ic_series[i-1] + np.random.normal(0, 0.1)
        
        self.ic_series = pd.Series(ic_series, index=dates)
    
    def test_analyze_ic_decay_basic(self):
        """Test basic IC decay analysis"""
        decay_analysis = analyze_ic_decay(self.ic_series)
        
        # Check return structure
        assert isinstance(decay_analysis, dict)
        assert 'autocorr_1' in decay_analysis
        assert 'autocorr_5' in decay_analysis
        assert 'autocorr_10' in decay_analysis
        assert 'half_life' in decay_analysis
        
        # Check autocorrelation coefficient ranges
        for key in ['autocorr_1', 'autocorr_5', 'autocorr_10']:
            if not pd.isna(decay_analysis[key]):
                assert -1 <= decay_analysis[key] <= 1
        
        # Check half-life
        if not pd.isna(decay_analysis['half_life']):
            assert decay_analysis['half_life'] > 0
    
    def test_analyze_ic_decay_empty_series(self):
        """Test IC decay analysis for empty series"""
        empty_series = pd.Series()
        decay_analysis = analyze_ic_decay(empty_series)
        
        assert decay_analysis == {}
    
    def test_analyze_ic_decay_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError, match="Input must be pandas.Series"):
            analyze_ic_decay(pd.DataFrame({'ic': [1, 2, 3]}))
    
    def test_calculate_half_life_basic(self):
        """Test half-life calculation"""
        # Create a series with clear decay
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        
        # Create a decaying series
        decay_series = []
        value = 1.0
        for i in range(50):
            decay_series.append(value)
            value = 0.9 * value + np.random.normal(0, 0.01)  # decay rate 0.9
        
        series = pd.Series(decay_series, index=dates)
        
        half_life = calculate_half_life(series)
        
        # Check reasonableness of half-life
        if not pd.isna(half_life):
            assert half_life > 0
            # For decay rate 0.9, theoretical half-life ≈ 6.6
            assert 1 <= half_life <= 20
    
    def test_calculate_half_life_edge_cases(self):
        """Test half-life edge cases"""
        # Empty series
        empty_series = pd.Series()
        half_life = calculate_half_life(empty_series)
        assert pd.isna(half_life)
        
        # All-NaN series
        nan_series = pd.Series([np.nan] * 10)
        half_life = calculate_half_life(nan_series)
        assert pd.isna(half_life)
        
        # Too-short series
        short_series = pd.Series([1, 2, 3])
        half_life = calculate_half_life(short_series)
        assert pd.isna(half_life)
        
        # Non-Series input
        with pytest.raises(ValueError, match="Input must be pandas.Series"):
            calculate_half_life(pd.DataFrame({'data': [1, 2, 3]}))


class TestFactorRanking:
    """Tests for factor ranking functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock IC results
        factors = ['momentum_3', 'momentum_9', 'mean_reversion_12', 'volume_ratio_24']
        ic_values = [0.05, 0.03, 0.08, 0.02]
        
        self.ic_results = pd.Series(ic_values, index=factors)
    
    def test_rank_factors_by_ic_basic(self):
        """Test basic factor ranking"""
        ranking = rank_factors_by_ic(self.ic_results, min_ic_threshold=0.02)
        
        # Check return structure
        assert isinstance(ranking, pd.DataFrame)
        assert 'factor' in ranking.columns
        assert 'mean_ic' in ranking.columns
        assert 'abs_ic' in ranking.columns
        assert 'effectiveness_score' in ranking.columns
        assert 'is_effective' in ranking.columns
        assert 'rank' in ranking.columns
        
        # Check ranking logic
        assert len(ranking) == len(self.ic_results)
        
        # Check ranking order (descending by effectiveness score)
        effectiveness_scores = ranking['effectiveness_score'].values
        assert np.all(np.diff(effectiveness_scores) <= 0)  # descending order
        
        # Check effective factors
        effective_factors = ranking[ranking['is_effective']]
        assert len(effective_factors) >= 0  # there may be no effective factors
    
    def test_rank_factors_by_ic_with_threshold(self):
        """Test factor ranking with different thresholds"""
        # Low threshold
        ranking_low = rank_factors_by_ic(self.ic_results, min_ic_threshold=0.01)
        
        # High threshold
        ranking_high = rank_factors_by_ic(self.ic_results, min_ic_threshold=0.05)
        
        # Higher threshold should yield fewer effective factors
        effective_low = ranking_low[ranking_low['is_effective']]
        effective_high = ranking_high[ranking_high['is_effective']]
        
        assert len(effective_high) <= len(effective_low)
    
    def test_rank_factors_by_ic_with_nan_values(self):
        """Test factor ranking with NaN values"""
        # Insert NaN values
        ic_results_with_nan = self.ic_results.copy()
        ic_results_with_nan.iloc[1] = np.nan
        
        ranking = rank_factors_by_ic(ic_results_with_nan, min_ic_threshold=0.02)
        
        # Ensure NaNs handled
        assert len(ranking) == len(ic_results_with_nan)
        
        # NaN factor should be marked ineffective
        nan_factor = ranking[ranking['factor'] == ic_results_with_nan.index[1]]
        assert not nan_factor['is_effective'].iloc[0]
    
    def test_rank_factors_by_ic_invalid_input(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError, match="IC results are empty"):
            rank_factors_by_ic(pd.Series())
        
        with pytest.raises(ValueError, match="Input must be pandas.Series"):
            rank_factors_by_ic(pd.DataFrame({'ic': [1, 2, 3]}))


class TestICVisualization:
    """Tests for IC visualization functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock IC analysis results
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        
        np.random.seed(42)
        
        # IC time series
        ic_series = np.random.normal(0.05, 0.1, len(dates))
        self.ic_results = {
            'ic_series': pd.Series(ic_series, index=dates),
            'mean_ic_by_factor': pd.Series([0.05, 0.03, 0.08, 0.02], 
                                          index=['factor1', 'factor2', 'factor3', 'factor4']),
            'rolling_mean': pd.Series(ic_series * 0.8, index=dates),
            'rolling_std': pd.Series(np.ones(len(dates)) * 0.08, index=dates)
        }
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_ic_analysis_basic(self, mock_savefig, mock_show):
        """Test basic IC analysis plotting"""
        # Test without saving figure
        plot_ic_analysis(self.ic_results)
        
        # Verify show called
        mock_show.assert_called()
        
        # Test saving figure
        plot_ic_analysis(self.ic_results, save_path='test_ic_plot.png')
        
        # Verify savefig called
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_ic_analysis_missing_data(self, mock_show):
        """Test plotting with missing IC analysis data"""
        # Create IC results missing some fields
        incomplete_ic_results = {
            'ic_series': self.ic_results['ic_series']
            # other fields missing
        }
        
        # Should not raise
        plot_ic_analysis(incomplete_ic_results)
        mock_show.assert_called()


class TestICReporting:
    """Tests for IC report generation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        # Create mock IC analysis results
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        
        np.random.seed(42)
        
        # IC time series
        ic_series = np.random.normal(0.05, 0.1, len(dates))
        
        self.ic_results = {
            'ic_stability': {
                'mean_ic': 0.05,
                'std_ic': 0.1,
                'ir_ratio': 0.5,
                'hit_rate': 0.6,
                'positive_ic_ratio': 0.3,
                'negative_ic_ratio': 0.2
            },
            'ic_decay': {
                'autocorr_1': 0.8,
                'autocorr_5': 0.6,
                'autocorr_10': 0.4,
                'half_life': 5.2
            }
        }
        
        # Create factor ranking
        ranking_data = [
            {'factor': 'factor1', 'mean_ic': 0.08, 'rank': 1, 'is_effective': True},
            {'factor': 'factor2', 'mean_ic': 0.05, 'rank': 2, 'is_effective': True},
            {'factor': 'factor3', 'mean_ic': 0.03, 'rank': 3, 'is_effective': False},
            {'factor': 'factor4', 'mean_ic': 0.01, 'rank': 4, 'is_effective': False}
        ]
        
        self.factor_ranking = pd.DataFrame(ranking_data)
    
    def test_generate_ic_report_basic(self):
        """Test basic IC report generation"""
        report = generate_ic_report(self.ic_results, self.factor_ranking)
        
        # Check report content
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check key sections
        assert "IC Analysis Report" in report
        assert "IC Summary Statistics" in report
        assert "Factor Effectiveness Ranking" in report
        assert "Effective Factor Statistics" in report
        assert "IC Decay Analysis" in report
        
        # Check numeric content
        assert "0.0500" in report  # mean_ic
        assert "0.1000" in report  # std_ic
        assert "0.5000" in report  # ir_ratio
    
    @patch('builtins.open', create=True)
    def test_generate_ic_report_save_file(self, mock_open):
        """Test saving IC report to file"""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        report = generate_ic_report(
            self.ic_results, 
            self.factor_ranking, 
            save_path='test_ic_report.txt'
        )
        
        # Verify file write
        mock_file.write.assert_called()
        
        # Check report content
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_generate_ic_report_missing_data(self):
        """Test IC report generation with missing data"""
        # Create IC results missing some fields
        incomplete_ic_results = {
            'ic_stability': {
                'mean_ic': 0.05,
                'std_ic': 0.1,
                'ir_ratio': 0.5,
                'hit_rate': 0.6
                # other fields missing
            }
            # missing ic_decay
        }
        
        # Should not raise
        report = generate_ic_report(incomplete_ic_results, self.factor_ranking)
        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__])

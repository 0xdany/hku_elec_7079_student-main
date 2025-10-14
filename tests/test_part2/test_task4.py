"""
Task 4 Unit Tests

Test cases for alpha factor engineering functionality.

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

from part2_alpha_modeling.task4_factors import (
    calculate_momentum_factors,
    calculate_mean_reversion_factors,
    calculate_volume_factors,
    calculate_intraday_factors,
    create_factor_dataset,
    handle_outliers,
    standardize_factor,
    check_factor_quality
)


class TestMomentumFactors:
    """Test momentum factor calculations"""
    
    def setup_method(self):
        """Set up test data"""
        # Create synthetic price data - single stock time series
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        # Generate synthetic prices (trend + noise)
        np.random.seed(42)
        base_price = 100.0
        price_data = []
        
        for i, date in enumerate(dates):
            # Add trend and noise
            trend = 0.001 * i  # slight upward trend
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            price_data.append(price)
        
        self.price_series = pd.Series(price_data, index=dates, name='close_px')
    
    def test_calculate_momentum_factors_basic(self):
        """Test basic momentum factor calculation"""
        factors = calculate_momentum_factors(self.price_series, periods=[3, 9])
        
        # Check return type
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        
        # Check columns
        expected_columns = ['momentum_3', 'momentum_9']
        for col in expected_columns:
            assert col in factors.columns
        
        # Check index alignment
        assert factors.index.equals(self.price_series.index)
        
        # Data quality check
        assert not factors.isnull().all().all()
    
    def test_calculate_momentum_factors_with_missing_data(self):
        """Test handling missing data in momentum factors"""
        # Add some missing values
        price_series_with_missing = self.price_series.copy()
        price_series_with_missing.iloc[10:15] = np.nan
        
        factors = calculate_momentum_factors(price_series_with_missing, periods=[3])
        
        # Check missing handled
        assert not factors.isnull().all().all()
    
    def test_calculate_momentum_factors_with_outliers(self):
        """Test outlier handling in momentum factors"""
        # Add outlier
        price_series_with_outliers = self.price_series.copy()
        price_series_with_outliers.iloc[20] = 1000  # outlier
        
        factors = calculate_momentum_factors(price_series_with_outliers, periods=[3])
        
        # Check outliers handled
        assert not factors.isnull().all().all()
        
        # Standardized value range
        for col in factors.columns:
            col_data = factors[col].dropna()
            if len(col_data) > 0:
                # Standardized values should be within reasonable range
                assert col_data.abs().max() < 10
    
    def test_calculate_momentum_factors_different_methods(self):
        """Test different calculation methods"""
        # Test pct-change method
        factors_pct = calculate_momentum_factors(self.price_series, periods=[3], method='pct_change')
        
        # Test log-return method
        factors_log = calculate_momentum_factors(self.price_series, periods=[3], method='log_return')
        
        # Both methods should return data
        assert not factors_pct.empty
        assert not factors_log.empty
        
        # Check column names
        assert 'momentum_3' in factors_pct.columns
        assert 'momentum_3' in factors_log.columns
    
    def test_calculate_momentum_factors_invalid_input(self):
        """Test invalid input"""
        # Empty data
        with pytest.raises(ValueError, match="(?i)empty"):
            calculate_momentum_factors(pd.Series())
        
        # Unsupported method
        with pytest.raises(ValueError, match="(?i)unsupported"):
            calculate_momentum_factors(self.price_series, method='invalid_method')
        
        # Non-Series input
        with pytest.raises(ValueError, match="(?i)pandas\.Series"):
            calculate_momentum_factors(pd.DataFrame({'price': [1, 2, 3]}))


class TestMeanReversionFactors:
    """Test mean reversion factor calculations"""
    
    def setup_method(self):
        """Set up test data"""
        # Use same test data as momentum factors
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        np.random.seed(42)
        base_price = 100.0
        price_data = []
        
        for i, date in enumerate(dates):
            trend = 0.001 * i
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            price_data.append(price)
        
        self.price_series = pd.Series(price_data, index=dates, name='close_px')
    
    def test_calculate_mean_reversion_factors_basic(self):
        """Test basic mean reversion factor calculation"""
        factors = calculate_mean_reversion_factors(self.price_series, ma_periods=[12, 24])
        
        # Check return type
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        
        # Check column names
        expected_columns = ['mean_reversion_12', 'mean_reversion_24']
        for col in expected_columns:
            assert col in factors.columns
        
        # Check index alignment
        assert factors.index.equals(self.price_series.index)
    
    def test_calculate_mean_reversion_factors_data_quality(self):
        """Test mean reversion factor data quality"""
        factors = calculate_mean_reversion_factors(self.price_series, ma_periods=[12])
        
        # Distribution checks
        for col in factors.columns:
            col_data = factors[col].dropna()
            if len(col_data) > 0:
                # Standardization effect
                assert abs(col_data.mean()) < 0.1
                assert abs(col_data.std() - 1.0) < 0.1
    
    def test_calculate_mean_reversion_factors_invalid_input(self):
        """Test invalid input"""
        with pytest.raises(ValueError, match="(?i)empty"):
            calculate_mean_reversion_factors(pd.Series())
        
        with pytest.raises(ValueError, match="(?i)pandas\.Series"):
            calculate_mean_reversion_factors(pd.DataFrame({'price': [1, 2, 3]}))


class TestVolumeFactors:
    """Test volume factor calculations"""
    
    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        np.random.seed(42)
        base_volume = 1000.0
        volume_data = []
        
        for i, date in enumerate(dates):
            # Add cyclical volume pattern
            pattern = 1 + 0.5 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.2)
            volume = base_volume * pattern * (1 + noise)
            volume = max(volume, 1)  # ensure positive volume
            volume_data.append(volume)
        
        self.volume_series = pd.Series(volume_data, index=dates, name='volume')
    
    def test_calculate_volume_factors_basic(self):
        """Test basic volume factor calculation"""
        factors = calculate_volume_factors(self.volume_series, lookback_periods=[12, 24])
        
        # Check return type
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        
        # Check column names
        expected_columns = ['volume_ratio_12', 'volume_ratio_24', 
                           'volume_change_12', 'volume_change_24']
        for col in expected_columns:
            assert col in factors.columns
        
        # Check index alignment
        assert factors.index.equals(self.volume_series.index)
    
    def test_calculate_volume_factors_zero_volume_handling(self):
        """Test zero volume handling"""
        # Add zero volumes
        volume_series_with_zeros = self.volume_series.copy()
        volume_series_with_zeros.iloc[10:15] = 0
        
        factors = calculate_volume_factors(volume_series_with_zeros, lookback_periods=[12])
        
        # Check zero volume handled
        assert not factors.isnull().all().all()
        
        # Check ratio factors
        ratio_cols = [col for col in factors.columns if 'ratio' in col]
        for col in ratio_cols:
            col_data = factors[col].dropna()
            if len(col_data) > 0:
                # Standardized values should be within reasonable range
                assert col_data.abs().max() < 10
    
    def test_calculate_volume_factors_data_quality(self):
        """Test volume factor data quality"""
        factors = calculate_volume_factors(self.volume_series, lookback_periods=[12])
        
        for col in factors.columns:
            col_data = factors[col].dropna()
            if len(col_data) > 0:
                # Standardization effect
                assert abs(col_data.mean()) < 0.1
                assert abs(col_data.std() - 1.0) < 0.1
    
    def test_calculate_volume_factors_invalid_input(self):
        """Test invalid input"""
        with pytest.raises(ValueError, match="(?i)empty"):
            calculate_volume_factors(pd.Series())
        
        with pytest.raises(ValueError, match="(?i)pandas\.Series"):
            calculate_volume_factors(pd.DataFrame({'volume': [1, 2, 3]}))


class TestIntradayFactors:
    """Test intraday feature calculations"""
    
    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range('2024-01-01 09:30:00', periods=100, freq='5min')
        
        np.random.seed(42)
        base_price = 100.0
        
        price_data = []
        vwap_data = []
        open_data = []
        
        for i, date in enumerate(dates):
            # Generate price data
            trend = 0.001 * i
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            
            # VWAP slightly below close
            vwap = price * (1 + np.random.normal(-0.001, 0.002))
            
            # Open price
            open_px = price * (1 + np.random.normal(0, 0.005))
            
            price_data.append(price)
            vwap_data.append(vwap)
            open_data.append(open_px)
        
        self.price_series = pd.Series(price_data, index=dates, name='close_px')
        self.vwap_series = pd.Series(vwap_data, index=dates, name='vwap')
        self.open_series = pd.Series(open_data, index=dates, name='open_px')
    
    def test_calculate_intraday_factors_basic(self):
        """Test basic intraday feature calculation"""
        factors = calculate_intraday_factors(self.price_series, self.vwap_series, self.open_series)
        
        # Check return type
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        
        # Check basic features
        expected_features = ['open_to_close_return', 'vwap_deviation', 'intraday_time_ratio']
        for feature in expected_features:
            assert feature in factors.columns
        
        # Check index alignment
        assert factors.index.equals(self.price_series.index)
    
    def test_calculate_intraday_factors_time_features(self):
        """Test intraday time features"""
        factors = calculate_intraday_factors(self.price_series, self.vwap_series, self.open_series)
        
        # Check time-related features
        time_features = [col for col in factors.columns if 'time_ratio' in col]
        assert len(time_features) > 0
        
        # Check time ratio range (standardized)
        for col in time_features:
            col_data = factors[col].dropna()
            if len(col_data) > 0:
                # Standardized values should be within reasonable range
                assert col_data.abs().max() < 10
    
    def test_calculate_intraday_factors_data_quality(self):
        """Test intraday feature data quality"""
        factors = calculate_intraday_factors(self.price_series, self.vwap_series, self.open_series)
        
        # Standardization effect
        for col in factors.columns:
            if col not in [c for c in factors.columns if 'time_ratio' in c]:
                col_data = factors[col].dropna()
                if len(col_data) > 0:
                    assert abs(col_data.mean()) < 0.1
                    assert abs(col_data.std() - 1.0) < 0.1
    
    def test_calculate_intraday_factors_invalid_input(self):
        """Test invalid input"""
        with pytest.raises(ValueError, match="(?i)empty"):
            calculate_intraday_factors(pd.Series(), self.vwap_series, self.open_series)
        
        with pytest.raises(ValueError, match="(?i)pandas\.Series"):
            calculate_intraday_factors(pd.DataFrame({'price': [1, 2, 3]}), self.vwap_series, self.open_series)


class TestFactorDataset:
    """Test factor dataset creation"""
    
    def setup_method(self):
        """Set up test data"""
        dates = pd.date_range('2024-01-01 09:30:00', periods=50, freq='5min')
        stocks = ['STOCK_1', 'STOCK_2']
        
        # Create various data matrices
        price_multi_index = pd.MultiIndex.from_product([stocks, ['close_px', 'high_px', 'low_px']], 
                                                      names=['symbol', 'field'])
        volume_multi_index = pd.MultiIndex.from_product([stocks, ['volume']], 
                                                       names=['symbol', 'field'])
        vwap_multi_index = pd.MultiIndex.from_product([stocks, ['vwap']], 
                                                     names=['symbol', 'field'])
        open_multi_index = pd.MultiIndex.from_product([stocks, ['open_px']], 
                                                     names=['symbol', 'field'])
        
        np.random.seed(42)
        base_prices = np.array([100, 50])
        
        price_data = []
        volume_data = []
        vwap_data = []
        open_data = []
        
        for i, date in enumerate(dates):
            # Prices
            trend = 0.001 * i
            noise = np.random.normal(0, 0.01, len(stocks))
            prices = base_prices * (1 + trend + noise)
            
            # High/Low
            high_prices = prices * (1 + np.random.uniform(0, 0.02, len(stocks)))
            low_prices = prices * (1 - np.random.uniform(0, 0.02, len(stocks)))
            
            # Volume
            volumes = base_prices * 10 * (1 + np.random.normal(0, 0.3, len(stocks)))
            volumes = np.maximum(volumes, 1)
            
            # VWAP and Open
            vwap = prices * (1 + np.random.normal(-0.001, 0.002, len(stocks)))
            open_px = prices * (1 + np.random.normal(0, 0.005, len(stocks)))
            
            # Combine price fields
            price_row = []
            for j in range(len(stocks)):
                price_row.extend([prices[j], high_prices[j], low_prices[j]])
            
            price_data.append(price_row)
            volume_data.append(volumes)
            vwap_data.append(vwap)
            open_data.append(open_px)
        
        self.price_data = pd.DataFrame(price_data, index=dates, columns=price_multi_index)
        self.volume_data = pd.DataFrame(volume_data, index=dates, columns=volume_multi_index)
        self.vwap_data = pd.DataFrame(vwap_data, index=dates, columns=vwap_multi_index)
        self.open_data = pd.DataFrame(open_data, index=dates, columns=open_multi_index)
    
    @patch('part2_alpha_modeling.task4_factors.save_results')
    @patch('part2_alpha_modeling.task4_factors.ensure_directory')
    def test_create_factor_dataset_basic(self, mock_ensure_dir, mock_save_results):
        """Test basic factor dataset creation"""
        mock_ensure_dir.return_value = Path("results/part2")
        
        factor_dataset = create_factor_dataset(
            self.price_data, self.volume_data, self.vwap_data, self.open_data
        )
        
        # Check return type
        assert isinstance(factor_dataset, pd.DataFrame)
        assert not factor_dataset.empty
        
        # Check factor groups exist
        momentum_cols = [col for col in factor_dataset.columns if 'momentum' in col]
        mean_reversion_cols = [col for col in factor_dataset.columns if 'mean_reversion' in col]
        volume_cols = [col for col in factor_dataset.columns if 'volume' in col]
        intraday_cols = [col for col in factor_dataset.columns if any(x in col for x in ['open_to_close', 'vwap_deviation', 'intraday_time_ratio'])]
        
        assert len(momentum_cols) > 0
        assert len(mean_reversion_cols) > 0
        assert len(volume_cols) > 0
        assert len(intraday_cols) > 0
        
        # Check index alignment
        assert factor_dataset.index.equals(self.price_data.index)
        
        # Ensure save called
        mock_save_results.assert_called()
    
    def test_create_factor_dataset_data_quality(self):
        """Test factor dataset data quality"""
        factor_dataset = create_factor_dataset(
            self.price_data, self.volume_data, self.vwap_data, self.open_data
        )
        
        # Missing ratio should not be too high
        missing_ratio = factor_dataset.isnull().sum().sum() / (factor_dataset.shape[0] * factor_dataset.shape[1])
        assert missing_ratio < 0.5
        
        # Check outliers
        for col in factor_dataset.columns:
            col_data = factor_dataset[col].dropna()
            if len(col_data) > 0:
                # Check extreme values
                q99 = col_data.quantile(0.99)
                q01 = col_data.quantile(0.01)
                assert abs(q99) < 10
                assert abs(q01) < 10


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_handle_outliers(self):
        """Test outlier handling"""
        # Create data with outliers
        data = pd.DataFrame({
            'col1': [1, 2, 3, 100, 5, 6, -50, 8, 9, 10],
            'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
        })
        
        cleaned_data = handle_outliers(data, method='std_cutoff', threshold=3)
        
        # Outliers should be constrained within reasonable bounds
        assert cleaned_data['col1'].max() <= 100
        assert cleaned_data['col1'].min() >= -50
        assert cleaned_data['col2'].max() <= 1000
        
        # Truncation reduces dispersion
        original_std = data.std()
        cleaned_std = cleaned_data.std()
        assert (cleaned_std <= original_std).all()
    
    def test_standardize_factor(self):
        """Test factor standardization"""
        # Create test data
        data = pd.DataFrame({
            'factor1': [1, 2, 3, 4, 5],
            'factor2': [10, 20, 30, 40, 50]
        })
        
        standardized = standardize_factor(data)
        
        # Check standardization
        for col in standardized.columns:
            assert abs(standardized[col].mean()) < 1e-10
            assert abs(standardized[col].std() - 1.0) < 1e-10
    
    def test_check_factor_quality(self):
        """Test factor quality check"""
        # Create test data
        data = pd.DataFrame({
            'factor1': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 1000],  # with NaNs and outliers
            'factor2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        quality_report = check_factor_quality(data)
        
        # Check report structure
        assert 'missing_ratio' in quality_report
        assert 'outlier_ratio' in quality_report
        assert 'distribution_stats' in quality_report
        assert 'total_missing_ratio' in quality_report
        assert 'avg_outlier_ratio' in quality_report
        
        # Basic sanity
        assert quality_report['total_missing_ratio'] > 0
        assert quality_report['avg_outlier_ratio'] > 0


if __name__ == "__main__":
    pytest.main([__file__])

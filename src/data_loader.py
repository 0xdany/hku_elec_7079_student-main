"""
Data Loader Module

This module provides functions for loading and preprocessing stock K-bar data
for the quantitative strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path
import warnings
import pickle
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader class for handling stock K-bar data loading and preprocessing.
    
    This class provides methods to load 5-minute and daily K-bar data,
    stock weights, and perform basic data validation and preprocessing.
    """
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_root (Optional[str]): Root directory containing data files
                                     If None, uses default data directory
        """
        if data_root is None:
            # Default to project data directory
            project_root = Path(__file__).parent.parent
            self.data_root = project_root / "data" / "raw"
        else:
            self.data_root = Path(data_root)
        
        self.data_cache = {}  # Cache for loaded data
    
    def load_5min_data(
        self, 
        file_path: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load 5-minute K-bar data from pickle format.
        
        Args:
            file_path (Optional[str]): Path to 5-minute data file
                                     If None, uses default path
            use_cache (bool): Whether to use cached data if available
        
        Returns:
            pd.DataFrame: Multi-level DataFrame with 5-minute K-bar data
                         Columns: MultiIndex (stock_symbol, data_field)
                         Index: timestamp
        
        Example:
            >>> loader = DataLoader()
            >>> data_5min = loader.load_5min_data()
            >>> print(data_5min.head())
        
        Notes:
            - Expected data fields: open_px, high_px, low_px, close_px, volume, vwap
            - Data should be sorted by timestamp
            - Missing values are handled during loading
        """
        cache_key = '5min_data'
        
        if use_cache and cache_key in self.data_cache:
            print("Loading 5-minute data from cache...")
            return self.data_cache[cache_key]
        
        if file_path is None:
            file_path = self.data_root / "Train_IntraDayData_5minute.pkl"
        
        print(f"Loading 5-minute data from: {file_path}")
        
        try:
            # Load pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Ensure data is DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Loaded data is not a DataFrame")
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort by timestamp
            data = data.sort_index()
            
            # Basic data validation
            self._validate_data(data, "5-minute")
            
            # Cache the data
            if use_cache:
                self.data_cache[cache_key] = data
            
            print(f"Loaded 5-minute data: {data.shape[0]} rows, {data.shape[1]} columns")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Please ensure the data file is in the correct location.")
            # Return sample data structure for development
            return self._create_sample_5min_data()
        
        except Exception as e:
            print(f"Error loading 5-minute data: {e}")
            return self._create_sample_5min_data()
    
    def load_daily_data(
        self, 
        file_path: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load daily K-bar data from pickle format.
        
        Args:
            file_path (Optional[str]): Path to daily data file
                                     If None, uses default path
            use_cache (bool): Whether to use cached data if available
        
        Returns:
            pd.DataFrame: Multi-level DataFrame with daily K-bar data
                         Columns: MultiIndex (stock_symbol, data_field)
                         Index: date
        
        Example:
            >>> loader = DataLoader()
            >>> data_daily = loader.load_daily_data()
            >>> print(data_daily.head())
        """
        cache_key = 'daily_data'
        
        if use_cache and cache_key in self.data_cache:
            print("Loading daily data from cache...")
            return self.data_cache[cache_key]
        
        if file_path is None:
            file_path = self.data_root / "Train_DailyData.pkl"
        
        print(f"Loading daily data from: {file_path}")
        
        try:
            # Load pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Ensure data is DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Loaded data is not a DataFrame")
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            # Basic data validation
            self._validate_data(data, "daily")
            
            # Cache the data
            if use_cache:
                self.data_cache[cache_key] = data
            
            print(f"Loaded daily data: {data.shape[0]} rows, {data.shape[1]} columns")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Please ensure the data file is in the correct location.")
            # Return sample data structure for development
            return self._create_sample_daily_data()
        
        except Exception as e:
            print(f"Error loading daily data: {e}")
            return self._create_sample_daily_data()
    
    def load_stock_weights(
        self, 
        file_path: Optional[str] = None
    ) -> pd.Series:
        """
        Load stock weights data from pickle format.
        
        Args:
            file_path (Optional[str]): Path to stock weights file
                                     If None, uses default path
        
        Returns:
            pd.Series: Series with stock weights
                      Index: stock symbols
                      Values: weights
        
        Example:
            >>> loader = DataLoader()
            >>> weights = loader.load_stock_weights()
            >>> print(weights.head())
        """
        if file_path is None:
            file_path = self.data_root / "stock_weight.pkl"
        
        print(f"Loading stock weights from: {file_path}")
        
        try:
            # Load pickle file
            with open(file_path, 'rb') as f:
                weights = pickle.load(f)
            
            # Ensure weights is Series
            if not isinstance(weights, pd.Series):
                raise ValueError("Loaded weights is not a Series")
            
            print(f"Loaded weights for {len(weights)} stocks")
            print(f"Weight sum: {weights.sum():.6f}")
            
            return weights
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            # Return equal weights for development
            return self._create_sample_weights()
        
        except Exception as e:
            print(f"Error loading stock weights: {e}")
            return self._create_sample_weights()
    
    def get_stock_list(self, data_type: str = "5min") -> List[str]:
        """
        Get list of available stock symbols.
        
        Args:
            data_type (str): Type of data ("5min" or "daily")
        
        Returns:
            List[str]: List of stock symbols
        """
        if data_type == "5min":
            data = self.load_5min_data()
        elif data_type == "daily":
            data = self.load_daily_data()
        else:
            raise ValueError("data_type must be '5min' or 'daily'")
        
        if isinstance(data.columns, pd.MultiIndex):
            return list(data.columns.get_level_values(0).unique())
        else:
            return list(data.columns)
    
    def _validate_data(self, data: pd.DataFrame, data_type: str) -> None:
        """
        Validate loaded data structure and content.
        
        Args:
            data (pd.DataFrame): Data to validate
            data_type (str): Type of data for validation context
        """
        # Check if data is empty
        if data.empty:
            raise ValueError(f"{data_type} data is empty")
        
        # Check for MultiIndex columns
        if not isinstance(data.columns, pd.MultiIndex):
            print(f"Warning: {data_type} data does not have MultiIndex columns")
        
        # Check for expected data fields
        expected_fields = ['open_px', 'high_px', 'low_px', 'close_px', 'volume', 'vwap']
        
        if isinstance(data.columns, pd.MultiIndex):
            available_fields = data.columns.get_level_values(1).unique()
            missing_fields = [field for field in expected_fields if field not in available_fields]
            
            if missing_fields:
                print(f"Warning: Missing expected fields in {data_type} data: {missing_fields}")
        
        # Check for missing values
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        if missing_pct > 10:
            print(f"Warning: {data_type} data has {missing_pct:.2f}% missing values")
        
        print(f"{data_type} data validation completed")
    
    def _create_sample_5min_data(self) -> pd.DataFrame:
        """Create sample 5-minute data for development/testing."""
        print("Creating sample 5-minute data for development...")
        
        # Create sample date range
        dates = pd.date_range('2019-01-02 09:30:00', '2019-01-03 15:00:00', freq='5T')
        dates = dates[dates.time >= pd.Timestamp('09:30:00').time()]
        dates = dates[dates.time <= pd.Timestamp('15:00:00').time()]
        
        # Sample stock symbols
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        fields = ['open_px', 'high_px', 'low_px', 'close_px', 'volume', 'vwap']
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_product([stocks, fields])
        
        # Generate sample data
        np.random.seed(42)
        n_rows = len(dates)
        n_cols = len(columns)
        
        data = np.random.randn(n_rows, n_cols) * 0.01 + 100  # Around 100 with 1% volatility
        
        # Adjust volume to be larger
        for i, stock in enumerate(stocks):
            vol_idx = columns.get_loc((stock, 'volume'))
            data[:, vol_idx] = np.random.exponential(1000, n_rows)
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        return df
    
    def _create_sample_daily_data(self) -> pd.DataFrame:
        """Create sample daily data for development/testing."""
        print("Creating sample daily data for development...")
        
        # Create sample date range
        dates = pd.date_range('2019-01-02', '2024-12-31', freq='D')
        
        # Sample stock symbols
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        fields = ['open_px', 'high_px', 'low_px', 'close_px', 'volume', 'vwap']
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_product([stocks, fields])
        
        # Generate sample data
        np.random.seed(42)
        n_rows = len(dates)
        n_cols = len(columns)
        
        data = np.random.randn(n_rows, n_cols) * 0.02 + 100  # Around 100 with 2% daily volatility
        
        # Adjust volume to be larger
        for i, stock in enumerate(stocks):
            vol_idx = columns.get_loc((stock, 'volume'))
            data[:, vol_idx] = np.random.exponential(10000, n_rows)
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        return df
    
    def _create_sample_weights(self) -> pd.Series:
        """Create sample stock weights for development/testing."""
        stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        weights = pd.Series([0.4, 0.35, 0.25], index=stocks)
        return weights


def calculate_daily_returns(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from daily price data.
    
    Args:
        daily_data (pd.DataFrame): Daily K-bar data with MultiIndex columns
    
    Returns:
        pd.DataFrame: Daily returns for each stock
    
    Example:
        >>> loader = DataLoader()
        >>> daily_data = loader.load_daily_data()
        >>> daily_returns = calculate_daily_returns(daily_data)
    """
    daily_returns = pd.DataFrame()
    
    if isinstance(daily_data.columns, pd.MultiIndex):
        stock_symbols = daily_data.columns.get_level_values(0).unique()
        
        for stock in stock_symbols:
            try:
                close_prices = daily_data[(stock, 'close_px')]
                stock_returns = close_prices.pct_change().dropna()
                daily_returns[stock] = stock_returns
            except KeyError:
                print(f"Warning: close_px not found for {stock}")
                continue
    
    return daily_returns


# Convenience functions for easy data access
def load_data(data_type: str = "both", data_root: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load data.
    
    Args:
        data_type (str): Type of data to load ("5min", "daily", or "both")
        data_root (Optional[str]): Root directory for data files
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing loaded data
    
    Example:
        >>> data = load_data("both")
        >>> print(data.keys())
    """
    loader = DataLoader(data_root)
    result = {}
    
    if data_type in ["5min", "both"]:
        result["5min"] = loader.load_5min_data()
    
    if data_type in ["daily", "both"]:
        result["daily"] = loader.load_daily_data()
    
    if data_type == "both":
        result["weights"] = loader.load_stock_weights()
    
    return result


# Example usage
def main():
    """
    Main function to demonstrate data loading functionality.
    """
    print("Data Loader Module")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load different types of data
    print("\n1. Loading 5-minute data...")
    data_5min = loader.load_5min_data()
    print(f"5-minute data shape: {data_5min.shape}")
    
    print("\n2. Loading daily data...")
    data_daily = loader.load_daily_data()
    print(f"Daily data shape: {data_daily.shape}")
    
    print("\n3. Loading stock weights...")
    weights = loader.load_stock_weights()
    print(f"Weights shape: {weights.shape}")
    
    print("\n4. Getting stock list...")
    stocks = loader.get_stock_list("daily")
    print(f"Available stocks: {stocks[:5]}...")  # Show first 5
    
    print("\n5. Calculating daily returns...")
    daily_returns = calculate_daily_returns(data_daily)
    print(f"Daily returns shape: {daily_returns.shape}")
    
    print("\nData loading demonstration completed!")


if __name__ == "__main__":
    main()

"""
Task 4: Alpha Factor Engineering

This module implements various alpha factors for quantitative strategy development.
It includes momentum factors, mean-reversion factors, volume-based factors, and
intraday features.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import ensure_directory, save_results

warnings.filterwarnings('ignore')


def calculate_momentum_factors(
    price_series: pd.Series, 
    periods: List[int] = [3, 9, 18],
    method: str = 'pct_change'
) -> pd.DataFrame:
    """
    Calculate momentum factors for a single stock.
    
    Calculates the price change rate over the past N periods to capture price trends.
    
    Args:
        price_series (pd.Series): Price time series for a single stock, with timestamps as index
        periods (List[int]): Time window list, default [3, 9, 18] corresponding to 15min, 45min, 90min
        method (str): Calculation method, 'pct_change' for percentage change, 'log_return' for log returns
    
    Returns:
        pd.DataFrame: Momentum factor DataFrame, column names as momentum_{period}, each column for one period
    
    Example:
        >>> # Get price data for a single stock
        >>> stock_prices = price_data.xs('STOCK_1', level=0, axis=1)['close_px']
        >>> momentum_factors = calculate_momentum_factors(stock_prices)
        >>> print(momentum_factors.head())
    
    Notes:
        - Uses close price to calculate momentum factors
        - Forward fills missing values
        - Applies 3-sigma outlier truncation
        - Uses Z-score standardization
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation: Check data type and whether it's empty
    # 2. Data preprocessing: Use ffill() for forward filling missing values
    # 3. Calculate momentum factors for each time window period:
    #    - pct_change method: price_series.pct_change(periods=period) 
    #    - log_return method: np.log(price_series / price_series.shift(periods=period))
    # 4. Data cleaning:
    #    - Fill missing values with 0.0
    #    - Call handle_outliers_series() to handle outliers
    #    - Call standardize_series() for Z-score standardization
    # 5. Organize results: Create DataFrame with column name format 'momentum_{period}'
    #
    # Expected output: DataFrame with time as rows, momentum factors for different periods as columns
    
    raise NotImplementedError("Please implement momentum factor calculation logic")


def calculate_mean_reversion_factors(
    price_series: pd.Series, 
    ma_periods: List[int] = [12, 24, 48]
) -> pd.DataFrame:
    """
    Calculate mean reversion factors for a single stock.
    
    Calculates the deviation of stock price from its moving average to capture mean reversion opportunities.
    
    Args:
        price_series (pd.Series): Price time series for a single stock
        ma_periods (List[int]): Moving average period list, default [12, 24, 48] corresponding to 1h, 2h, 4h
    
    Returns:
        pd.DataFrame: Mean reversion factor DataFrame, column names as mean_reversion_{period}, each column for one period
    
    Example:
        >>> stock_prices = price_data.xs('STOCK_1', level=0, axis=1)['close_px']
        >>> mean_reversion_factors = calculate_mean_reversion_factors(stock_prices)
        >>> print(mean_reversion_factors.head())
    
    Notes:
        - Uses close price to calculate moving averages
        - Calculates percentage deviation of price from moving average
        - Applies standardization to results
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation and preprocessing (refer to momentum factor implementation)
    # 2. Calculate mean reversion factors for each moving average period:
    #    - Calculate moving average: price_series.rolling(window=period, min_periods=1).mean()
    #    - Calculate deviation: (price_series - ma) / ma
    # 3. Data cleaning:
    #    - Fill missing values with 0.0
    #    - Outlier handling and standardization (using existing helper functions)
    # 4. Organize results: Column name format as 'mean_reversion_{period}'
    #
    # Expected output: DataFrame with time as rows, mean reversion factors for different periods as columns
    
    raise NotImplementedError("Please implement mean reversion factor calculation logic")


def calculate_volume_factors(
    volume_series: pd.Series, 
    lookback_periods: List[int] = [12, 24, 48]
) -> pd.DataFrame:
    """
    Calculate volume factors for a single stock.
    
    Calculates the ratio and change rate of current volume relative to historical average volume.
    
    Args:
        volume_series (pd.Series): Volume time series for a single stock
        lookback_periods (List[int]): Lookback period list, default [12, 24, 48]
    
    Returns:
        pd.DataFrame: Volume factor DataFrame containing volume_ratio_{period} and volume_change_{period}
    
    Example:
        >>> stock_volume = volume_data.xs('STOCK_1', level=0, axis=1)['volume']
        >>> volume_factors = calculate_volume_factors(stock_volume)
        >>> print(volume_factors.head())
    
    Notes:
        - Calculates volume ratio: current volume / historical average volume
        - Calculates volume change rate: current volume change rate relative to historical average
        - Handles zero volume situations
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation and preprocessing
    # 2. Calculate volume factors for each lookback period:
    #    - Calculate historical average volume: volume_series.rolling(window=period, min_periods=1).mean()
    #    - Calculate volume ratio: volume_series / avg_volume
    #    - Calculate volume change rate: (volume_series - avg_volume) / avg_volume
    # 3. Special handling:
    #    - Handle zero volume: replace([np.inf, -np.inf], appropriate values)
    #    - Fill volume ratio missing values with 1.0 (normal level)
    #    - Fill volume change rate missing values with 0.0 (no change)
    # 4. Data cleaning and standardization
    # 5. Organize results: Each period corresponds to two columns 'volume_ratio_{period}' and 'volume_change_{period}'
    #
    # Expected output: DataFrame with time as rows, volume factors for different periods as columns
    
    raise NotImplementedError("Please implement volume factor calculation logic")


def calculate_intraday_factors(
    price_series: pd.Series, 
    vwap_series: pd.Series,
    open_series: pd.Series
) -> pd.DataFrame:
    """
    Calculate intraday features for a single stock.
    
    Calculates intraday features such as price change from open to current, price deviation from daily VWAP, etc.
    
    Args:
        price_series (pd.Series): Close price time series for a single stock
        vwap_series (pd.Series): VWAP time series for a single stock
        open_series (pd.Series): Open price time series for a single stock
    
    Returns:
        pd.DataFrame: Intraday feature DataFrame containing open_to_close_return, vwap_deviation, intraday_time_ratio, etc.
    
    Example:
        >>> stock_prices = price_data.xs('STOCK_1', level=0, axis=1)['close_px']
        >>> stock_vwap = price_data.xs('STOCK_1', level=0, axis=1)['vwap']
        >>> stock_open = price_data.xs('STOCK_1', level=0, axis=1)['open_px']
        >>> intraday_factors = calculate_intraday_factors(stock_prices, stock_vwap, stock_open)
        >>> print(intraday_factors.head())
    
    Notes:
        - Calculates post-opening returns
        - Calculates price deviation from VWAP
        - Calculates intraday time features
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation: Check if all input series are empty and of correct type
    # 2. Data preprocessing: Forward fill all series
    # 3. Calculate intraday features:
    #    a) Post-opening return: (price_series - open_series) / open_series
    #    b) VWAP deviation: (price_series - vwap_series) / vwap_series  
    #    c) Intraday time feature: Calculate time ratio from market open
    #       - Market open time: 9:30
    #       - Market close time: 15:00
    #       - Time ratio: (current time - open time) / (close time - open time)
    # 4. Data cleaning:
    #    - Fill missing values with 0.0
    #    - Outlier handling (3-sigma truncation)
    #    - Z-score standardization
    # 5. Organize results: Column names as 'open_to_close_return', 'vwap_deviation', 'intraday_time_ratio'
    #
    # Expected output: DataFrame with time as rows, various intraday features as columns
    
    raise NotImplementedError("Please implement intraday factor calculation logic")


def calculate_factors_for_stock(
    stock_data: pd.DataFrame,
    stock_symbol: str,
    periods: List[int] = [3, 9, 18],
    ma_periods: List[int] = [12, 24, 48],
    volume_periods: List[int] = [12, 24, 48]
) -> pd.DataFrame:
    """
    Calculate all factors for a single stock.
    
    Args:
        stock_data (pd.DataFrame): DataFrame containing all data for the stock in MultiIndex format
        stock_symbol (str): Stock symbol
        periods (List[int]): Momentum factor periods
        ma_periods (List[int]): Mean reversion factor periods
        volume_periods (List[int]): Volume factor periods
    
    Returns:
        pd.DataFrame: DataFrame containing all factors
    """
    # Extract data for the stock
    stock_prices = stock_data.xs(stock_symbol, level=0, axis=1)['close_px']
    stock_volume = stock_data.xs(stock_symbol, level=0, axis=1)['volume']
    stock_vwap = stock_data.xs(stock_symbol, level=0, axis=1)['vwap']
    stock_open = stock_data.xs(stock_symbol, level=0, axis=1)['open_px']
    
    # Calculate various factors
    momentum_factors = calculate_momentum_factors(stock_prices, periods=periods)
    mean_reversion_factors = calculate_mean_reversion_factors(stock_prices, ma_periods=ma_periods)
    volume_factors = calculate_volume_factors(stock_volume, lookback_periods=volume_periods)
    intraday_factors = calculate_intraday_factors(stock_prices, stock_vwap, stock_open)
    
    # Combine all factors
    factors_df = pd.concat([
        momentum_factors,
        mean_reversion_factors,
        volume_factors,
        intraday_factors
    ], axis=1)
    
    return factors_df


def create_factor_dataset(
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    vwap_data: pd.DataFrame,
    open_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a complete factor dataset.
    
    Integrates all factors into a single DataFrame and performs quality checks.
    
    Args:
        price_data (pd.DataFrame): Price data
        volume_data (pd.DataFrame): Volume data
        vwap_data (pd.DataFrame): VWAP data
        open_data (pd.DataFrame): Open price data
    
    Returns:
        pd.DataFrame: Complete factor dataset
    
    Example:
        >>> data_5min = loader.load_5min_data()
        >>> factor_dataset = create_factor_dataset(data_5min, data_5min, data_5min, data_5min)
        >>> print(factor_dataset.shape)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Data merging and validation:
    #    - Use pd.concat() to merge all input data
    #    - Check if data is in MultiIndex format
    #    - Extract all stock symbols: columns.get_level_values(0).unique()
    # 2. Calculate factors by stock:
    #    - Iterate through each stock symbol
    #    - Call calculate_factors_for_stock() to calculate all factors for that stock
    #    - Add stock prefix to factor column names: f"{stock_symbol}_{col}"
    # 3. Merge all factors:
    #    - Use pd.concat(all_factors, axis=1) for horizontal merging
    #    - Handle empty data cases
    # 4. Quality check:
    #    - Call check_factor_quality() function
    #    - Print quality report summary
    # 5. Save results:
    #    - Create results/part2 directory
    #    - Save as CSV file
    #    - Return factor dataset
    #
    # Expected output: DataFrame with time as rows, all factors for all stocks as columns
    
    raise NotImplementedError("Please implement factor dataset creation logic")


def handle_outliers_series(
    data: pd.Series, 
    method: str = 'std_cutoff', 
    threshold: float = 3.0
) -> pd.Series:
    """
    Handle outliers in time series data.
    
    Args:
        data (pd.Series): Input time series
        method (str): Processing method, 'std_cutoff' for standard deviation truncation
        threshold (float): Threshold value, default 3 times standard deviation
    
    Returns:
        pd.Series: Processed time series
    """
    if method == 'std_cutoff':
        # Calculate mean and standard deviation
        mean_val = data.mean()
        std_val = data.std()
        
        if std_val == 0:
            return data
        
        # Create upper and lower bounds
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val
        
        # Truncate outliers
        data_clean = data.clip(lower=lower_bound, upper=upper_bound)
        
        return data_clean
    else:
        raise ValueError(f"Unsupported outlier handling method: {method}")


def standardize_series(data: pd.Series) -> pd.Series:
    """
    Standardize time series data.
    
    Uses Z-score standardization method.
    
    Args:
        data (pd.Series): Time series data
    
    Returns:
        pd.Series: Standardized time series
    """
    # Calculate mean and standard deviation
    mean_val = data.mean()
    std_val = data.std()
    
    # Avoid division by zero
    if std_val == 0:
        return pd.Series(0, index=data.index)
    
    # Z-score standardization
    standardized = (data - mean_val) / std_val
    
    return standardized


def handle_outliers(
    data: pd.DataFrame, 
    method: str = 'std_cutoff', 
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Handle outliers in DataFrame.
    
    Args:
        data (pd.DataFrame): Input data
        method (str): Processing method, 'std_cutoff' for standard deviation truncation
        threshold (float): Threshold value, default 3 times standard deviation
    
    Returns:
        pd.DataFrame: Processed data
    """
    if method == 'std_cutoff':
        # Calculate mean and standard deviation for each column
        mean_vals = data.mean()
        std_vals = data.std()
        
        # Create upper and lower bounds
        lower_bound = mean_vals - threshold * std_vals
        upper_bound = mean_vals + threshold * std_vals
        
        # Truncate outliers
        data_clean = data.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        return data_clean
    else:
        raise ValueError(f"Unsupported outlier handling method: {method}")


def standardize_factor(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize factor data.
    
    Uses Z-score standardization method.
    
    Args:
        factor_data (pd.DataFrame): Factor data
    
    Returns:
        pd.DataFrame: Standardized data
    """
    # Calculate mean and standard deviation for each column
    mean_vals = factor_data.mean()
    std_vals = factor_data.std()
    
    # Avoid division by zero
    std_vals = std_vals.replace(0, 1)
    
    # Z-score standardization
    standardized = (factor_data - mean_vals) / std_vals
    
    return standardized


def check_factor_quality(factor_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check factor quality.
    
    Checks quality indicators such as missing values, outliers, and distribution.
    
    Args:
        factor_data (pd.DataFrame): Factor data
    
    Returns:
        Dict[str, Any]: Quality report
    """
    quality_report = {}
    
    # Missing value check
    missing_ratio = factor_data.isnull().sum() / len(factor_data)
    quality_report['missing_ratio'] = missing_ratio.to_dict()
    quality_report['total_missing_ratio'] = missing_ratio.mean()
    
    # Outlier check (using IQR method)
    outlier_ratios = {}
    for col in factor_data.columns:
        Q1 = factor_data[col].quantile(0.25)
        Q3 = factor_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((factor_data[col] < lower_bound) | (factor_data[col] > upper_bound)).sum()
        outlier_ratios[col] = outliers / len(factor_data)
    
    quality_report['outlier_ratio'] = outlier_ratios
    quality_report['avg_outlier_ratio'] = np.mean(list(outlier_ratios.values()))
    
    # Distribution check
    distribution_stats = {}
    for col in factor_data.columns:
        distribution_stats[col] = {
            'mean': factor_data[col].mean(),
            'std': factor_data[col].std(),
            'skewness': factor_data[col].skew(),
            'kurtosis': factor_data[col].kurtosis()
        }
    
    quality_report['distribution_stats'] = distribution_stats
    
    return quality_report


def plot_factor_distributions(
    factor_data: pd.DataFrame, 
    sample_factors: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot factor distribution charts.
    
    Args:
        factor_data (pd.DataFrame): Factor data
        sample_factors (Optional[List[str]]): Factor samples to plot, if None plots all factors
        save_path (Optional[str]): Save path
    """
    if sample_factors is None:
        # Select first 10 factors for visualization
        sample_factors = factor_data.columns[:10].tolist()
    
    n_factors = len(sample_factors)
    n_cols = 3
    n_rows = (n_factors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, factor in enumerate(sample_factors):
        if i < len(axes):
            # Plot histogram
            axes[i].hist(factor_data[factor].dropna(), bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{factor} Distribution')
            axes[i].set_xlabel('Factor Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_factors, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Factor distribution chart saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test code
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    
    # Create factor dataset
    factor_dataset = create_factor_dataset(
        price_data=data_5min,
        volume_data=data_5min,
        vwap_data=data_5min,
        open_data=data_5min
    )
    
    print("Factor engineering completed!")

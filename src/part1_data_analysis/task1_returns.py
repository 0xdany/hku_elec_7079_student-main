"""
Task 1: Target Engineering & Return Calculation

This module implements functions for calculating forward returns and analyzing return distributions
for the quantitative strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_forward_returns(
    data: pd.DataFrame, 
    forward_periods: int = 12,
    price_column: str = 'close_px'
) -> pd.DataFrame:
    """
    Calculate forward returns for each stock at every 5-minute interval.
    
    This function computes the forward return over a specified number of periods
    (default 12 periods = 1 hour for 5-minute data) for each stock in the dataset.
    The forward return is calculated as (future_price / current_price - 1).
    
    Args:
        data (pd.DataFrame): Multi-level DataFrame with stock data
                           Expected structure: MultiIndex columns (stock_symbol, data_fields)
                           Index should be timestamp
        forward_periods (int): Number of periods to look forward (default 12 for 1-hour)
        price_column (str): Column name for price data (default 'close_px')
    
    Returns:
        pd.DataFrame: DataFrame with forward returns for each stock
                     Columns: stock symbols
                     Index: timestamps
                     
    Example:
        >>> data = load_5min_data()  # Load your 5-minute data
        >>> forward_returns = calculate_forward_returns(data, forward_periods=12)
        >>> print(forward_returns.head())
    
    Notes:
        - The last 'forward_periods' rows will contain NaN values
        - Returns are calculated as simple returns (not log returns)
        - Data should be properly sorted by timestamp before calling this function
    """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Check if input data has MultiIndex format
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Input data must have MultiIndex columns")

    # 2. Extract all stock symbols: data.columns.get_level_values(0).unique()
    stock_symbols: List[str] = data.columns.get_level_values(0).unique()

    # 3. For each stock:
    #    - Get current prices: data[(stock, price_column)]
    #    - Get future prices: current_prices.shift(-forward_periods)
    #    - Calculate forward returns: (future_price / current_price) - 1
    # 4. Combine all stock returns into DataFrame
    forward_returns: pd.DataFrame = pd.DataFrame()
    for stock in stock_symbols:
        try:
            current_prices: pd.Series = data[(stock, price_column)]
            future_prices: pd.Series = current_prices.shift(-forward_periods) # negative means later time
            forward_returns[stock] = (future_prices / current_prices) - 1

    # 5. Handle KeyError and missing value situations
            forward_returns.iloc[-forward_periods:] = np.nan # hadnle last row
        except KeyError:
            raise ValueError(f"{price_column} not found for {stock}")

    # Expected output: DataFrame with time as rows, stock symbols as columns, forward returns as values
    return forward_returns

    


def calculate_weekly_returns(daily_data: pd.DataFrame, price_column: str = 'close_px') -> pd.DataFrame:
    """
    Calculate weekly returns for each stock using daily K-bar data.
    
    This function computes weekly returns by resampling daily data to weekly frequency
    and calculating the return from Monday's open to Friday's close.
    
    Args:
        daily_data (pd.DataFrame): Multi-level DataFrame with daily stock data
                                 Expected structure: MultiIndex columns (stock_symbol, data_fields)
                                 Index should be date
        price_column (str): Column name for price data (default 'close_px')
    
    Returns:
        pd.DataFrame: DataFrame with weekly returns for each stock
                     Columns: stock symbols
                     Index: week ending dates
                     
    Example:
        >>> daily_data = load_daily_data()  # Load your daily data
        >>> weekly_returns = calculate_weekly_returns(daily_data)
        >>> print(weekly_returns.head())
    
    Notes:
        - Weekly returns are calculated as (week_end_price / week_start_price) - 1
        - Weeks are defined as Monday to Friday
        - Missing data is handled by forward-filling within the week
    """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Check if input data has MultiIndex format
    if not isinstance(daily_data.columns, pd.MultiIndex):
        raise ValueError("Input data must have MultiIndex columns")

    # 2. Extract all stock symbols
    stock_symbols: List[str] = daily_data.columns.get_level_values(0).unique()

    # 3. For each stock:
    #    - Get daily price data: daily_data[(stock, price_column)]
    #    - Resample to weekly frequency using resample('W')
    #    - Get last trading day price of each week: .last()
    #    - Calculate weekly returns: .pct_change()
    # 5. Combine results into DataFrame
    # 4. Handle missing data and exceptional situations
    weekly_returns: pd.DataFrame = pd.DataFrame()

    for stock in stock_symbols:
        try:
            weekly_price = daily_data[(stock, price_column)].ffill().resample('W')
            weekly_price_closing = weekly_price.last()
            weekly_price_opening = weekly_price.first()
            weekly_returns[stock] = (weekly_price_closing / weekly_price_opening) - 1

        except KeyError:
            raise ValueError(f"{price_column} not found for {stock}")
    #
    # Expected output: DataFrame with week end dates as rows, stock symbols as columns, weekly returns as values
    return weekly_returns
    



def plot_return_distribution(
    returns_data: pd.DataFrame, 
    sample_stocks: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Dict[str, Dict[str, float]]:
    """
    Visualize and analyze the distribution of returns for sample stocks.
    
    This function creates distribution plots and calculates statistical properties
    (mean, std, skewness, kurtosis) for the return distributions. It also performs
    normality tests to assess if returns follow a normal distribution.
    
    Args:
        returns_data (pd.DataFrame): DataFrame with returns for each stock
        sample_stocks (List[str]): List of stock symbols to analyze
        save_path (Optional[str]): Path to save the plot (optional)
        figsize (Tuple[int, int]): Figure size for the plot
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing statistical properties
                                   for each stock including:
                                   - mean: Average return
                                   - std: Standard deviation
                                   - skewness: Measure of asymmetry
                                   - kurtosis: Measure of tail heaviness
                                   - jarque_bera_pvalue: P-value for normality test
                                   
    Example:
        >>> forward_returns = calculate_forward_returns(data)
        >>> sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        >>> stats = plot_return_distribution(forward_returns, sample_stocks)
        >>> print(stats['STOCK_1'])
    
    Notes:
        - Jarque-Bera test is used to test for normality
        - P-value < 0.05 suggests non-normal distribution
        - Skewness > 0 indicates right tail, < 0 indicates left tail
        - Kurtosis > 3 indicates heavy tails compared to normal distribution
    """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Verify sample stocks exist in the data
    if missing_stocks := set(sample_stocks) - set(returns_data.columns):
        raise ValueError(f"The following sample stocks were not found in the data: {', '.join(missing_stocks)}")

    # 2. Create 2×n_stocks subplot layout:
    #    - First row: histogram + normal distribution overlay
    #    - Second row: Q-Q plots
    n_stocks = len(sample_stocks)
    fig, axes = plt.subplots(nrows=2, ncols=n_stocks, figsize=figsize, constrained_layout=True)

    # 3. Calculate statistics for each stock:
    #    - Basic statistics: mean(), std()
    #    - Distribution characteristics: skew(), kurtosis()
    #    - Normality test: stats.jarque_bera()
    stats_dict = {}
    for i, stock in enumerate(sample_stocks):
        returns = returns_data[stock].dropna() # Drop NaN values for accurate stats
        
        if returns.empty:
            raise ValueError(f"No valid return data found for stock '{stock}'")

        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        count = int(returns.count())
        _, jarque_bera_pvalue = stats.jarque_bera(returns)
        
        stats_dict[stock] = {
            'mean': mean_return,
            'std': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_pvalue': jarque_bera_pvalue,
            'count': count
        }
        
        # 4. Create visualizations:
        #    - plt.hist() for histogram
        #    - stats.norm.pdf() for normal distribution overlay
        #    - stats.probplot() for Q-Q plots
        ax_hist = axes[0, i] if n_stocks > 1 else axes[0]
        
        n, bins, _ = ax_hist.hist(returns, bins=50, density=True, alpha=0.6, label='Returns')
 
        xmin, xmax = ax_hist.get_xlim() 
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_return, std_return) # normal distribution overlay
        ax_hist.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
           
        ax_qq = axes[1, i] if n_stocks > 1 else axes[1]
        stats.probplot(returns, dist="norm", plot=ax_qq)  #  for Q-Q plots

        # 5. Add chart labels, legends, statistical info text boxes
        ax_hist.set_title(f'Return Distribution: {stock}')
        ax_hist.set_xlabel('Return')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()

        stats_text = (
            f"Mean: {mean_return:.4f}\n"
            f"Std: {std_return:.4f}\n"
            f"Skew: {skewness:.4f}\n"
            f"Kurt: {kurtosis:.4f}\n"
            f"JB p-value: {jarque_bera_pvalue:.4f}"
        )
        ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        ax_qq.set_title(f'Q-Q Plot: {stock}')
    # Add a main title for the figure
    fig.suptitle('Return Distribution Analysis', fontsize=16) # add a main title for the figure
    
    # 6. Save plots (if path specified)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    plt.show()

    # Expected output: Dict[stock_symbol, Dict[statistic_name, value]]
    return stats_dict
    
    

def analyze_return_properties(returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze statistical properties of returns for all stocks.
    
    This function calculates comprehensive statistics for return distributions
    across all stocks in the dataset.
    
    Args:
        returns_data (pd.DataFrame): DataFrame with returns for each stock
    
    Returns:
        pd.DataFrame: Summary statistics for all stocks
    
    Example:
        >>> returns = calculate_forward_returns(data)
        >>> summary = analyze_return_properties(returns)
        >>> print(summary.head())
    """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Calculate statistics for each stock's return series:
    #    - Basic statistics: count, mean, std, min, max
    #    - Quantiles: q25, q50 (median), q75
    #    - Distribution shape: skewness, kurtosis
    # 2. Perform normality test (when sample size >= 8):
    #    - Use stats.jarque_bera()
    #    - Determine normality based on p-value (>0.05 for normal)
    # 3. Organize results into DataFrame with stock symbols as index
    # 4. Handle edge cases with empty data
    #
    # Expected output: DataFrame with stocks as rows, statistical indicators as columns

   # Handle empty DataFrame on edge case
    if returns_data.empty:
        # Define the expected columns for an empty result
        expected_columns = [
            'count', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis',
            'q25', 'q50', 'q75', 'jarque_bera_pvalue', 'is_normal'
        ]
        return pd.DataFrame(columns=expected_columns)
    
    # 1. Calculate statistics for each stock's return series:
    #    - Basic statistics: count, mean, std, min, max
    #    - Quantiles: q25, q50 (median), q75
    # transpose the data so that it will match the dataframe format
    summary_df = returns_data.describe(percentiles=[0.25, 0.5, 0.75]).transpose()
    # Rename the percentile columns to match the test requirements
    summary_df = summary_df.rename(columns={'25%': 'q25', '50%': 'q50', '75%': 'q75'})
    #    - Distribution shape: skewness, kurtosis
    summary_df['skewness'] = returns_data.skew()
    summary_df['kurtosis'] = returns_data.kurtosis()
    # 2. Perform normality test (when sample size >= 8):
    # define a helper function for a jarque bera for better readability
    def apply_jarque_bera(series):
        """Helper function to apply Jarque-Bera test."""
        # Check if the series has at least 8 non-NaN values
        if series.count() < 8:
            return pd.Series({'jarque_bera_pvalue': np.nan, 'is_normal': np.nan})
        
        # Drop NaNs before applying the test
        non_nan_series = series.dropna()
        
        #    - Use stats.jarque_bera()
        _, p_value = stats.jarque_bera(non_nan_series)
        
        #    - Determine normality based on p-value (>0.05 for normal)
        is_normal = p_value > 0.05
        
        return pd.Series({'jarque_bera_pvalue': p_value, 'is_normal': is_normal})
    # Apply the function to each column and join the results
    jb_results = returns_data.apply(apply_jarque_bera)
    summary_df = summary_df.join(jb_results.transpose())
    # 3. Organize results into DataFrame with stock symbols as index
    # The .transpose() on the describe() result already sets the stock symbols as the index.
    # The .join() operation correctly adds the new columns.
    # 4. Handle edge cases with empty data
    # The use of .describe() and .dropna() inside the apply_jarque_bera function
    # handles empty data by producing NaNs where necessary.
    
    return summary_df


# Example usage and testing functions
def main():
    """
    Main function to demonstrate the usage of return calculation functions.
    
    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 1: Target Engineering & Return Calculation")
    print("=" * 50)
    
    # Note: This is a template - actual data loading will depend on your data structure
    print("To use these functions:")
    print("1. Load your 5-minute data using the data loader")
    print("2. Calculate forward returns using calculate_forward_returns()")
    print("3. Load daily data and calculate weekly returns")
    print("4. Analyze return distributions for sample stocks")
    
    print("\nExample code:")
    print("""
    # Load data (implement based on your data structure)
    # data_5min = load_5min_data()
    # data_daily = load_daily_data()
    
    # Calculate forward returns
    # forward_returns = calculate_forward_returns(data_5min, forward_periods=12)
    
    # Calculate weekly returns
    # weekly_returns = calculate_weekly_returns(data_daily)
    
    # Analyze distributions
    # sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
    # stats = plot_return_distribution(forward_returns, sample_stocks)
    """)


if __name__ == "__main__":
    main()



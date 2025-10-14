"""
Task 2: Market & Asset Characterization

This module implements functions for volatility analysis and market index construction
for the quantitative strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def calculate_rolling_volatility(
    daily_returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation) for stock returns.
    
    This function computes the rolling standard deviation of daily returns
    for each stock over a specified window period. Volatility can be annualized
    by multiplying by sqrt(252) for daily data.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns for each stock
                                    Columns: stock symbols
                                    Index: dates
        window (int): Rolling window size in days (default 20 for monthly volatility)
        annualize (bool): Whether to annualize volatility (default True)
    
    Returns:
        pd.DataFrame: DataFrame with rolling volatility for each stock
                     Same structure as input but with volatility values
                     
    Example:
        >>> daily_returns = calculate_daily_returns(data)
        >>> volatility = calculate_rolling_volatility(daily_returns, window=20)
        >>> print(volatility.head())
    
    Notes:
        - Annualized volatility assumes 252 trading days per year
        - The first (window-1) rows will contain NaN values
        - Volatility is calculated as the standard deviation of returns
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Use pandas rolling() function to calculate rolling standard deviation:
    #    - daily_returns.rolling(window=window).std()
    # 2. If annualize=True, annualize the results:
    #    - Multiply by np.sqrt(252), assuming 252 trading days per year
    # 3. Note that the first (window-1) rows will be NaN, which is normal
    #
    # Expected output: DataFrame with same structure as input, values are rolling volatilities
    
    vol = daily_returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def build_equal_weight_index(
    stock_returns: pd.DataFrame,
    rebalance_freq: str = 'D'
) -> pd.Series:
    """
    Create an equal-weighted index from individual stock returns.
    
    This function constructs a market index by equally weighting all stocks
    and rebalancing at the specified frequency. The index serves as a
    benchmark for strategy performance evaluation.
    
    Args:
        stock_returns (pd.DataFrame): DataFrame with returns for each stock
                                    Columns: stock symbols
                                    Index: timestamps
        rebalance_freq (str): Rebalancing frequency ('D' for daily, 'W' for weekly)
    
    Returns:
        pd.Series: Equal-weighted index returns
                  Index: timestamps
                  Values: index returns
                  
    Example:
        >>> daily_returns = calculate_daily_returns(data)
        >>> ew_index = build_equal_weight_index(daily_returns)
        >>> print(f"Index average return: {ew_index.mean():.4f}")
    
    Notes:
        - Equal weighting means each stock gets 1/N weight where N is number of stocks
        - Missing data is handled by excluding stocks with NaN returns for that period
        - Index is rebalanced at specified frequency to maintain equal weights
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Calculate equal-weight returns (simple average of all stocks):
    #    - Use stock_returns.mean(axis=1, skipna=True)
    if stock_returns.empty:
        return pd.Series(index=stock_returns.index, dtype=float)
    index_returns = stock_returns.mean(axis=1, skipna=True)

    # 2. axis=1 means calculate across rows (average across stocks)
    if stock_returns.shape[1] == 1:
        index_returns.name = stock_returns.columns[0]

    # 3. skipna=True automatically handles missing values, only calculates average for valid data
    else:
        index_returns.name = None

    # 4. Returns a time series (pd.Series)
    #
    # Expected output: Series with time index and equal-weight index returns as values
    
    return index_returns


def plot_volatility_analysis(
    daily_returns: pd.DataFrame,
    sample_stocks: List[str],
    equal_weight_index: pd.Series,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create comprehensive volatility analysis plots.
    
    This function generates multiple visualizations to analyze volatility patterns:
    1. Rolling volatility time series for sample stocks
    2. Volatility distribution comparison
    3. Index vs individual stock volatility comparison
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        sample_stocks (List[str]): List of stock symbols to analyze
        equal_weight_index (pd.Series): Equal-weighted index returns
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size for the plot
    
    Example:
        >>> daily_returns = calculate_daily_returns(data)
        >>> ew_index = build_equal_weight_index(daily_returns)
        >>> sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
        >>> plot_volatility_analysis(daily_returns, sample_stocks, ew_index)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Validate sample stocks and calculate volatility:
    #    - Use the implemented calculate_rolling_volatility() function
    #    - Calculate 20-day rolling volatility for both individual stocks and index
    missing = set(sample_stocks) - set(daily_returns.columns)
    if missing:
        raise ValueError(f"Missing stocks: {', '.join(sorted(missing))}")
    selected = daily_returns[sample_stocks]
    rolling_vol = calculate_rolling_volatility(selected, window=20, annualize=False)
    index_rolling_vol = equal_weight_index.rolling(window=20).std()

    # 2. Create 2×2 subplot layout:
    #    - Top-left: Volatility time series line plot
    #    - Top-right: Volatility distribution box plot
    #    - Bottom-left: Risk-return scatter plot
    #    - Bottom-right: Volatility clustering analysis (squared returns)
    ann_ret = selected.mean() * 252
    ann_vol = selected.std() * np.sqrt(252)
    squared_roll = selected.pow(2).rolling(window=20).mean()
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_ts = axes[0, 0]
    
    # 3. Draw each subplot:
    #    - ax.plot() for time series
    #    - ax.boxplot() for distribution
    #    - ax.scatter() for scatter plot
    # 4. Add legends, labels, grids, etc.
    for col in rolling_vol.columns:
        ax_ts.plot(rolling_vol.index, rolling_vol[col], label=col)
    ax_ts.plot(index_rolling_vol.index, index_rolling_vol, label='INDEX')
    ax_ts.set_title('20-day Rolling Volatility')
    ax_ts.legend()
    ax_box = axes[0, 1]
    data_for_box = [rolling_vol[col].dropna().values for col in rolling_vol.columns]

    if len(data_for_box) > 0 and all(len(x) > 0 for x in data_for_box):
        ax_box.boxplot(data_for_box, labels=rolling_vol.columns)
    ax_box.set_title('Volatility Distribution')
    ax_scatter = axes[1, 0]
    ax_scatter.scatter(ann_vol.values, ann_ret.values)

    for i, name in enumerate(ann_ret.index):
        ax_scatter.annotate(name, (ann_vol.values[i], ann_ret.values[i]))
    ax_scatter.set_xlabel('Annualized Volatility')
    ax_scatter.set_ylabel('Annualized Return')
    ax_scatter.set_title('Risk-Return')

    ax_sq = axes[1, 1]
    for col in squared_roll.columns:
        ax_sq.plot(squared_roll.index, squared_roll[col], label=col)
    ax_sq.set_title('Rolling Mean of Squared Returns')

    # 5. Save the plot (if path is specified)
    #
    # No return value, directly display the chart
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def calculate_volatility_statistics(
    daily_returns: pd.DataFrame,
    equal_weight_index: pd.Series,
    window: int = 20
) -> Dict[str, Any]:
    """
    Calculate comprehensive volatility statistics for stocks and index.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        equal_weight_index (pd.Series): Equal-weighted index returns
        window (int): Rolling window for volatility calculation
    
    Returns:
        Dict[str, Any]: Dictionary containing volatility statistics
    
    Example:
        >>> stats = calculate_volatility_statistics(daily_returns, ew_index)
        >>> print(f"Average stock volatility: {stats['avg_stock_volatility']:.4f}")
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Compute rolling volatility for stocks and index:
    #    - Use calculate_rolling_volatility()
    #    - Convert equal_weight_index to DataFrame when needed
    stock_vol = calculate_rolling_volatility(daily_returns, window=window, annualize=True)
    stock_vol_means = stock_vol.mean(skipna=True)
    index_vol_series = equal_weight_index.rolling(window=window).std() * np.sqrt(252)

    # 2. Stock volatility statistics:
    #    - Average volatility: stock_vol_means.mean()
    #    - Median, min, max
    #    - Volatility of volatilities: stock_vol_means.std()
    avg_stock_volatility = float(stock_vol_means.mean()) if len(stock_vol_means) else float('nan')
    median_stock_volatility = float(stock_vol_means.median()) if len(stock_vol_means) else float('nan')
    min_stock_volatility = float(stock_vol_means.min()) if len(stock_vol_means) else float('nan')
    max_stock_volatility = float(stock_vol_means.max()) if len(stock_vol_means) else float('nan')
    vol_of_volatilities = float(stock_vol_means.std()) if len(stock_vol_means) else float('nan')
    
    # 3. Index volatility statistics
    index_volatility = float(index_vol_series.mean()) if len(index_vol_series) else float('nan')
    index_vol_std = float(index_vol_series.std()) if len(index_vol_series) else float('nan')
    
    # 4. Diversification benefit:
    #    - diversification_ratio = avg_stock_volatility / index_volatility
    # 5. Correlation between stock and market volatilities
    diversification_ratio = float(avg_stock_volatility / index_volatility) if index_volatility and not np.isnan(index_volatility) and index_volatility != 0 else float('nan')
    n_stocks_analyzed = int(len(daily_returns.columns))
    
    # 6. Return statistics dict
    #
    # Expected output: Dict of volatility statistics
    
    return {
        'avg_stock_volatility': avg_stock_volatility,
        'median_stock_volatility': median_stock_volatility,
        'min_stock_volatility': min_stock_volatility,
        'max_stock_volatility': max_stock_volatility,
        'vol_of_volatilities': vol_of_volatilities,
        'index_volatility': index_volatility,
        'index_vol_std': index_vol_std,
        'diversification_ratio': diversification_ratio,
        'n_stocks_analyzed': n_stocks_analyzed
    }


def compare_individual_vs_market(
    daily_returns: pd.DataFrame,
    equal_weight_index: pd.Series
) -> pd.DataFrame:
    """
    Compare individual stock performance with market index.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        equal_weight_index (pd.Series): Equal-weighted index returns
    
    Returns:
        pd.DataFrame: Comparison statistics for each stock
    
    Example:
        >>> comparison = compare_individual_vs_market(daily_returns, ew_index)
        >>> print(comparison.head())
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. For each stock vs market index:
    #    - Align data: reindex() to ensure time alignment
    #    - Filter insufficient data (<50 observations)
    results: Dict[str, Dict[str, Any]] = {}
    for stock in daily_returns.columns:
        aligned = pd.concat([daily_returns[stock], equal_weight_index], axis=1, join='inner').dropna()
        if aligned.shape[0] < 50:
            continue
        aligned.columns = ['stock', 'market']
        
    # 2. Compute key metrics:
    #    - Annualized return and volatility (×252, ×√252)
    #    - Beta: covariance / market variance
    #    - Alpha: stock return - Beta × market return
    #    - Correlation: stock.corr(index)
    # 3. Risk-adjusted metrics:
    #    - Sharpe ratio: annualized return / annualized volatility
    #    - Tracking error: std of (stock - Beta × market)
    #    - Information ratio: alpha / tracking error
    # 4. Organize results as DataFrame
    #
    # Expected output: DataFrame, rows=stocks, columns=metrics
    
        stock_mean = aligned['stock'].mean()
        market_mean = aligned['market'].mean()
        market_var = aligned['market'].var()
        beta = float(aligned['stock'].cov(aligned['market']) / market_var) if market_var != 0 else np.nan
        alpha_daily = stock_mean - (beta * market_mean if not np.isnan(beta) else 0.0)
        ann_return = float(stock_mean * 252)
        ann_vol = float(aligned['stock'].std() * np.sqrt(252))
        corr = float(aligned['stock'].corr(aligned['market']))
        residual = aligned['stock'] - (beta * aligned['market'] if not np.isnan(beta) else 0.0)
        tracking_error = float(residual.std() * np.sqrt(252))
        sharpe_ratio = float(ann_return / ann_vol) if ann_vol != 0 else np.nan
        information_ratio = float((alpha_daily * 252) / tracking_error) if tracking_error != 0 else np.nan
        results[stock] = {
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'beta': beta,
            'alpha': float(alpha_daily * 252),
            'correlation': corr,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'observations': int(aligned.shape[0])
        }
    return pd.DataFrame.from_dict(results, orient='index')


# Example usage and testing functions
def main():
    """
    Main function to demonstrate the usage of volatility analysis functions.
    
    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 2: Market & Asset Characterization")
    print("=" * 50)
    
    print("Functions implemented:")
    print("1. calculate_rolling_volatility() - Calculate rolling volatility for stocks")
    print("2. build_equal_weight_index() - Create equal-weighted market index")
    print("3. plot_volatility_analysis() - Comprehensive volatility visualization")
    print("4. calculate_volatility_statistics() - Volatility statistics summary")
    print("5. compare_individual_vs_market() - Individual vs market comparison")
    
    print("\nExample usage:")
    print("""
    # Calculate daily returns first
    # daily_returns = daily_data.pct_change().dropna()
    
    # Calculate rolling volatility
    # volatility = calculate_rolling_volatility(daily_returns, window=20)
    
    # Build equal-weight index
    # ew_index = build_equal_weight_index(daily_returns)
    
    # Create volatility analysis plots
    # sample_stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3']
    # plot_volatility_analysis(daily_returns, sample_stocks, ew_index)
    
    # Calculate statistics
    # stats = calculate_volatility_statistics(daily_returns, ew_index)
    # comparison = compare_individual_vs_market(daily_returns, ew_index)
    """)


if __name__ == "__main__":
    main()



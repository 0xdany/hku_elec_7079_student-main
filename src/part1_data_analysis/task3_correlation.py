"""
Task 3: Cross-Sectional Analysis

This module implements functions for correlation analysis and cross-sectional
relationship analysis for the quantitative strategy development project.

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


def calculate_correlation_matrix(
    daily_returns: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix of daily returns for all stocks.
    
    This function computes the pairwise correlation between all stocks
    in the dataset over the entire period. The correlation matrix shows
    how stock returns move together.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns for each stock
                                    Columns: stock symbols
                                    Index: dates
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
    
    Returns:
        pd.DataFrame: Correlation matrix with stocks as both rows and columns
                     Values range from -1 (perfect negative correlation) to 
                     +1 (perfect positive correlation)
                     
    Example:
        >>> daily_returns = calculate_daily_returns(data)
        >>> corr_matrix = calculate_correlation_matrix(daily_returns)
        >>> print(f"Average correlation: {corr_matrix.mean().mean():.4f}")
    
    Notes:
        - Diagonal elements are always 1.0 (perfect self-correlation)
        - Matrix is symmetric
        - Missing data is handled by pairwise deletion
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Use pandas corr() function to calculate correlation matrix:
    #    - daily_returns.corr(method=method)
    # 2. method parameter supports: 'pearson', 'spearman', 'kendall'
    # 3. Function automatically handles missing values (pairwise deletion)
    # 4. Returned matrix should be symmetric with diagonal = 1.0
    #
    # Expected output: DataFrame with stock codes as rows and columns, values as correlation coefficients
    
    if daily_returns.empty:
        return pd.DataFrame(index=daily_returns.columns, columns=daily_returns.columns, dtype=float)
    corr = daily_returns.corr(method=method)
    return corr


def calculate_rolling_correlation(
    stock1_returns: pd.Series,
    stock2_returns: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling correlation between two specific stocks.
    
    This function computes the correlation between two stocks over a rolling
    window to analyze how their relationship changes over time.
    
    Args:
        stock1_returns (pd.Series): Returns for first stock
        stock2_returns (pd.Series): Returns for second stock
        window (int): Rolling window size in days (default 60 for ~3 months)
        min_periods (Optional[int]): Minimum periods required for calculation
    
    Returns:
        pd.Series: Rolling correlation time series
                  Index: dates
                  Values: correlation coefficients
                  
    Example:
        >>> stock1 = daily_returns['STOCK_1']
        >>> stock2 = daily_returns['STOCK_2']
        >>> rolling_corr = calculate_rolling_correlation(stock1, stock2, window=60)
        >>> print(f"Average rolling correlation: {rolling_corr.mean():.4f}")
    
    Notes:
        - The first (window-1) observations will be NaN
        - Correlation can vary significantly over time
        - Values range from -1 to +1
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Align two return series:
    #    - Create DataFrame with two columns
    #    - Use dropna() to remove missing values
    if min_periods is None:
        min_periods = window
    df = pd.concat([stock1_returns, stock2_returns], axis=1).dropna()
    if df.empty:
        return pd.Series(index=stock1_returns.index, dtype=float)

    # 2. Calculate rolling correlation:
    #    - Use stock1.rolling(window).corr(stock2)
    #    - Set appropriate min_periods parameter
    # 3. Handle parameters:
    #    - If min_periods is None, set to window
    rolling_corr = df.iloc[:, 0].rolling(window=window, min_periods=min_periods).corr(df.iloc[:, 1])
    rolling_corr = rolling_corr.reindex(stock1_returns.index)

    # 4. Return time series
    #
    # Expected output: Series with time index and rolling correlation coefficients as values
    
    return rolling_corr


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Stock Returns Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm'
) -> None:
    """
    Create a heatmap visualization of the correlation matrix.
    
    This function generates a color-coded heatmap to visualize the correlation
    structure between all stocks in the dataset.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to visualize
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        cmap (str): Colormap for the heatmap
    
    Example:
        >>> corr_matrix = calculate_correlation_matrix(daily_returns)
        >>> plot_correlation_heatmap(corr_matrix)
    
    Notes:
        - Red colors indicate positive correlation
        - Blue colors indicate negative correlation
        - Diagonal is always dark red (correlation = 1)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Create figure: plt.figure(figsize=figsize)
    # 2. Optional: create upper-triangular mask to avoid duplicate display
    #    - mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=figsize)
    mask = None
    if correlation_matrix.shape[0] > 1:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # 3. Plot heatmap with seaborn:
    #    - sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, ...)
    #    - set center=0 for neutral zero
    #    - set square=True for square cells
    # 4. Add title and labels:
    #    - plt.title(), plt.xlabel(), plt.ylabel()
    #    - rotate x-axis labels for readability
    # 5. Save and show figure
    #
    # No return value; show heatmap directly
    
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0, square=True, cbar_kws={"shrink": 0.8}, annot=False)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_rolling_correlation_analysis(
    daily_returns: pd.DataFrame,
    stock_pairs: List[Tuple[str, str]],
    window: int = 60,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Analyze and visualize rolling correlations for multiple stock pairs.
    
    This function calculates and plots rolling correlations for specified
    stock pairs to show how relationships change over time.
    
    Args:
        daily_returns (pd.DataFrame): DataFrame with daily returns
        stock_pairs (List[Tuple[str, str]]): List of stock pairs to analyze
        window (int): Rolling window size
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    
    Returns:
        Dict[Tuple[str, str], pd.Series]: Dictionary of rolling correlation series
    
    Example:
        >>> stock_pairs = [('STOCK_1', 'STOCK_2'), ('STOCK_3', 'STOCK_4')]
        >>> rolling_corrs = plot_rolling_correlation_analysis(daily_returns, stock_pairs)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Validate stock pairs exist in data
    all_symbols = set(daily_returns.columns)
    for a, b in stock_pairs:
        if a not in all_symbols or b not in all_symbols:
            raise ValueError("Invalid stock pair provided")

    # 2. Compute rolling correlation for each pair:
    #    - Use implemented calculate_rolling_correlation()
    n = len(stock_pairs)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    # 3. Create subplot layout:
    #    - Determine n_rows, n_cols
    #    - Use plt.subplots() to create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    # 4. Plot each pair's rolling correlation:
    #    - ax.plot() time series
    #    - Add zero line and mean line as references
    #    - Set y-limits to (-1.1, 1.1)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    results: Dict[Tuple[str, str], pd.Series] = {}
    for idx, pair in enumerate(stock_pairs):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        s1 = daily_returns[pair[0]]
        s2 = daily_returns[pair[1]]
        rc = calculate_rolling_correlation(s1, s2, window=window)
        results[pair] = rc
        ax.plot(rc.index, rc.values, label=f"{pair[0]} vs {pair[1]}")
        ax.axhline(0.0, color='black', linewidth=1, linestyle='--')
        if rc.dropna().size > 0:
            ax.axhline(rc.dropna().mean(), color='orange', linewidth=1, linestyle=':')
            
    # 5. Add titles, labels, legend, grid
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"Rolling Corr ({window}): {pair[0]} vs {pair[1]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    total_axes = n_rows * n_cols

    # 6. Hide extra subplots
    for j in range(n, total_axes):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])

    # 7. Save and show figure
    #
    # Expected output: Dict[Tuple[pair], Series[rolling_correlation]]
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return results


def analyze_correlation_structure(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the correlation structure of the stock universe.
    
    This function provides insights into the correlation patterns
    among stocks, including clustering and distribution analysis.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to analyze
        threshold (float): Correlation threshold for high correlation pairs
    
    Returns:
        Dict[str, Any]: Dictionary containing correlation analysis results
    
    Example:
        >>> corr_matrix = calculate_correlation_matrix(daily_returns)
        >>> analysis = analyze_correlation_structure(corr_matrix)
        >>> print(f"Average correlation: {analysis['avg_correlation']:.4f}")
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Extract off-diagonal correlation values:
    #    - Use np.eye() to create diagonal mask
    #    - Extract off-diagonal elements
    if correlation_matrix.empty:
        return {
            'avg_correlation': np.nan,
            'median_correlation': np.nan,
            'std_correlation': np.nan,
            'min_correlation': np.nan,
            'max_correlation': np.nan,
            'n_stocks': 0,
            'high_corr_pairs': [],
            'n_high_corr_pairs': 0,
            'pct_high_corr': 0.0,
            'pct_positive_corr': 0.0,
            'pct_negative_corr': 0.0,
            'q25': np.nan,
            'q75': np.nan
        }
    n = correlation_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)

    # 2. Compute basic statistics:
    #    - mean, median, std, min, max
    vals = correlation_matrix.values[mask]
    avg_corr = float(np.nanmean(vals)) if vals.size else np.nan
    med_corr = float(np.nanmedian(vals)) if vals.size else np.nan
    std_corr = float(np.nanstd(vals)) if vals.size else np.nan
    min_corr = float(np.nanmin(vals)) if vals.size else np.nan
    max_corr = float(np.nanmax(vals)) if vals.size else np.nan
    
    # 3. Identify high-correlation pairs:
    #    - Iterate upper triangle
    #    - Find pairs with abs(corr) >= threshold
    # 4. Distribution characteristics:
    #    - Positive vs negative proportions
    #    - Count and ratio of high-corr pairs
    # 5. Quantiles
    q25 = float(np.nanpercentile(vals, 25)) if vals.size else np.nan
    q75 = float(np.nanpercentile(vals, 75)) if vals.size else np.nan
    pos_pct = float((np.sum(vals > 0) / vals.size) * 100) if vals.size else 0.0
    neg_pct = 100.0 - pos_pct if vals.size else 0.0
    high_pairs = []
    stocks = list(correlation_matrix.index)
    count_high = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_ij = correlation_matrix.iloc[i, j]
            if abs(corr_ij) >= threshold:
                count_high += 1
                high_pairs.append({
                    'stock1': stocks[i],
                    'stock2': stocks[j],
                    'correlation': float(corr_ij)
                })
    total_pairs = n * (n - 1) / 2
    pct_high = float((count_high / total_pairs) * 100) if total_pairs > 0 else 0.0

    # 6. Return result dictionary
    #
    # Expected output: Dict with correlation structure metrics
    
    return {
        'avg_correlation': avg_corr,
        'median_correlation': med_corr,
        'std_correlation': std_corr,
        'min_correlation': min_corr,
        'max_correlation': max_corr,
        'n_stocks': int(n),
        'high_corr_pairs': high_pairs,
        'n_high_corr_pairs': int(count_high),
        'pct_high_corr': pct_high,
        'pct_positive_corr': pos_pct,
        'pct_negative_corr': neg_pct,
        'q25': q25,
        'q75': q75
    }


def plot_correlation_distribution(
    correlation_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the distribution of pairwise correlations.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix to analyze
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    # 
    # Implementation hints:
    # 1. Extract off-diagonal correlation values:
    #    - Use diagonal mask to drop diagonal entries
    n = correlation_matrix.shape[0]
    if n <= 1:
        plt.figure(figsize=figsize)
        plt.show()
        return
    mask = ~np.eye(n, dtype=bool)
    vals = correlation_matrix.values[mask]

    # 2. Create 1x2 subplot layout:
    #    - Left: histogram of correlation distribution
    #    - Right: boxplot summary
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_hist = axes[0]
    ax_box = axes[1]
    
    # 3. Plot histogram:
    #    - plt.hist()
    #    - Add mean/median reference lines
    #    - Add legend and labels
    ax_hist.hist(vals, bins=30, alpha=0.7, edgecolor='k')
    mean_v = float(np.nanmean(vals)) if vals.size else np.nan
    median_v = float(np.nanmedian(vals)) if vals.size else np.nan
    ax_hist.axvline(mean_v, color='red', linestyle='--', label='Mean')
    ax_hist.axvline(median_v, color='green', linestyle=':', label='Median')
    ax_hist.set_title('Correlation Distribution')
    ax_hist.legend()

    # 4. Plot boxplot:
    #    - plt.boxplot()
    ax_box.boxplot(vals, vert=True)
    ax_box.set_title('Summary')

    # 5. Add stats textbox: count, mean, std, min, max
    stats_text = (
        f"Count: {vals.size}\n"
        f"Mean: {mean_v:.4f}\n"
        f"Std: {float(np.nanstd(vals)):.4f}\n"
        f"Min: {float(np.nanmin(vals)):.4f}\n"
        f"Max: {float(np.nanmax(vals)):.4f}"
    )
    ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
                 
    # 6. Save and show
    #
    # No return value; show the distribution plots
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Example usage and testing functions
def main():
    """
    Main function to demonstrate the usage of correlation analysis functions.
    
    This function provides examples of how to use the implemented functions
    and can be used for testing purposes.
    """
    print("Task 3: Cross-Sectional Analysis")
    print("=" * 50)
    
    print("Functions implemented:")
    print("1. calculate_correlation_matrix() - Calculate correlation matrix")
    print("2. calculate_rolling_correlation() - Rolling correlation for stock pairs")
    print("3. plot_correlation_heatmap() - Visualize correlation matrix")
    print("4. plot_rolling_correlation_analysis() - Rolling correlation analysis")
    print("5. analyze_correlation_structure() - Correlation structure analysis")
    print("6. plot_correlation_distribution() - Correlation distribution plots")
    
    print("\nExample usage:")
    print("""
    # Calculate daily returns first
    # daily_returns = daily_data.pct_change().dropna()
    
    # Calculate correlation matrix
    # corr_matrix = calculate_correlation_matrix(daily_returns)
    
    # Visualize correlation matrix
    # plot_correlation_heatmap(corr_matrix)
    
    # Analyze rolling correlations
    # stock_pairs = [('STOCK_1', 'STOCK_2'), ('STOCK_3', 'STOCK_4')]
    # rolling_corrs = plot_rolling_correlation_analysis(daily_returns, stock_pairs)
    
    # Analyze correlation structure
    # analysis = analyze_correlation_structure(corr_matrix)
    # plot_correlation_distribution(corr_matrix)
    """)


if __name__ == "__main__":
    main()



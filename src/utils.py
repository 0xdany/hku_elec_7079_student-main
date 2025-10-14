"""
Utility Functions Module

This module provides common utility functions used across the quantitative
strategy development project.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any, Union
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, create if it doesn't.
    
    Args:
        path (Union[str, Path]): Directory path to ensure
    
    Returns:
        Path: Path object of the ensured directory
    
    Example:
        >>> results_dir = ensure_directory("results/figures")
        >>> print(f"Directory ensured: {results_dir}")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    filename: str,
    results_dir: str = "results",
    file_format: str = "csv"
) -> str:
    """
    Save analysis results to file.
    
    Args:
        data: Data to save (DataFrame, Series, or dictionary)
        filename (str): Name of the file (without extension)
        results_dir (str): Directory to save results
        file_format (str): File format ("csv", "xlsx", "json", "pickle")
    
    Returns:
        str: Path to saved file
    
    Example:
        >>> save_results(correlation_matrix, "correlation_matrix", "results/part1")
    """
    # Ensure results directory exists
    results_path = ensure_directory(results_dir)
    
    # Construct full file path
    file_path = results_path / f"{filename}.{file_format}"
    
    try:
        if file_format == "csv":
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_csv(file_path)
            else:
                pd.DataFrame([data]).to_csv(file_path)
        
        elif file_format == "xlsx":
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_excel(file_path)
            else:
                pd.DataFrame([data]).to_excel(file_path)
        
        elif file_format == "json":
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_json(file_path)
            else:
                import json
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        elif file_format == "pickle":
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        print(f"Results saved to: {file_path}")
        return str(file_path)
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return ""


def load_results(
    filename: str,
    results_dir: str = "results",
    file_format: str = "csv"
) -> Union[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load previously saved analysis results.
    
    Args:
        filename (str): Name of the file (without extension)
        results_dir (str): Directory containing results
        file_format (str): File format ("csv", "xlsx", "json", "pickle")
    
    Returns:
        Union[pd.DataFrame, pd.Series, Dict]: Loaded data
    
    Example:
        >>> correlation_matrix = load_results("correlation_matrix", "results/part1")
    """
    results_path = Path(results_dir)
    file_path = results_path / f"{filename}.{file_format}"
    
    try:
        if file_format == "csv":
            return pd.read_csv(file_path, index_col=0)
        
        elif file_format == "xlsx":
            return pd.read_excel(file_path, index_col=0)
        
        elif file_format == "json":
            import json
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif file_format == "pickle":
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def setup_plotting_style(style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Set up consistent plotting style for the project.
    
    Args:
        style (str): Matplotlib style to use
        figsize (Tuple[int, int]): Default figure size
    
    Example:
        >>> setup_plotting_style("seaborn", (15, 10))
    """
    try:
        plt.style.use(style)
    except:
        plt.style.use('default')
    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value (float): Decimal value to format
        decimals (int): Number of decimal places
    
    Returns:
        str: Formatted percentage string
    
    Example:
        >>> format_percentage(0.1234, 2)
        '12.34%'
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_summary_statistics(
    data: Union[pd.DataFrame, pd.Series],
    percentiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> pd.DataFrame:
    """
    Calculate comprehensive summary statistics for data.
    
    Args:
        data: Data to analyze
        percentiles (List[float]): Percentiles to calculate
    
    Returns:
        pd.DataFrame: Summary statistics
    
    Example:
        >>> stats = calculate_summary_statistics(returns_data)
        >>> print(stats)
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Basic statistics
    summary = data.describe(percentiles=percentiles)
    
    # Additional statistics
    additional_stats = pd.DataFrame(index=['skewness', 'kurtosis', 'missing_count', 'missing_pct'])
    
    for col in data.columns:
        series = data[col].dropna()
        
        if len(series) > 0:
            from scipy import stats
            additional_stats.loc['skewness', col] = stats.skew(series)
            additional_stats.loc['kurtosis', col] = stats.kurtosis(series, fisher=True)
        else:
            additional_stats.loc['skewness', col] = np.nan
            additional_stats.loc['kurtosis', col] = np.nan
        
        missing_count = data[col].isnull().sum()
        additional_stats.loc['missing_count', col] = missing_count
        additional_stats.loc['missing_pct', col] = missing_count / len(data) * 100
    
    # Combine statistics
    full_summary = pd.concat([summary, additional_stats])
    
    return full_summary


def validate_data_quality(
    data: pd.DataFrame,
    max_missing_pct: float = 20.0,
    min_observations: int = 100
) -> Dict[str, Any]:
    """
    Validate data quality and provide quality report.
    
    Args:
        data (pd.DataFrame): Data to validate
        max_missing_pct (float): Maximum acceptable missing data percentage
        min_observations (int): Minimum required observations
    
    Returns:
        Dict[str, Any]: Data quality report
    
    Example:
        >>> quality_report = validate_data_quality(stock_data)
        >>> print(f"Data quality score: {quality_report['quality_score']}")
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'date_range': (data.index.min(), data.index.max()) if hasattr(data.index, 'min') else None,
        'issues': [],
        'warnings': [],
        'quality_score': 100.0
    }
    
    # Check for missing data
    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    report['missing_data_pct'] = missing_pct
    
    if missing_pct > max_missing_pct:
        report['issues'].append(f"High missing data: {missing_pct:.2f}%")
        report['quality_score'] -= 30
    elif missing_pct > max_missing_pct / 2:
        report['warnings'].append(f"Moderate missing data: {missing_pct:.2f}%")
        report['quality_score'] -= 10
    
    # Check number of observations
    if len(data) < min_observations:
        report['issues'].append(f"Insufficient observations: {len(data)} < {min_observations}")
        report['quality_score'] -= 25
    
    # Check for duplicate indices
    if hasattr(data.index, 'duplicated'):
        duplicate_indices = data.index.duplicated().sum()
        if duplicate_indices > 0:
            report['issues'].append(f"Duplicate indices: {duplicate_indices}")
            report['quality_score'] -= 15
    
    # Check for constant columns
    constant_columns = []
    for col in data.columns:
        if data[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        report['warnings'].append(f"Constant columns: {constant_columns}")
        report['quality_score'] -= 5
    
    # Check for extreme outliers (beyond 5 standard deviations)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    extreme_outliers = 0
    
    for col in numeric_columns:
        series = data[col].dropna()
        if len(series) > 0:
            z_scores = np.abs((series - series.mean()) / series.std())
            extreme_outliers += (z_scores > 5).sum()
    
    if extreme_outliers > 0:
        outlier_pct = (extreme_outliers / (len(data) * len(numeric_columns))) * 100
        if outlier_pct > 1:
            report['warnings'].append(f"Extreme outliers: {outlier_pct:.2f}%")
            report['quality_score'] -= 5
    
    # Ensure quality score doesn't go below 0
    report['quality_score'] = max(0, report['quality_score'])
    
    return report


def create_performance_summary(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns (pd.Series): Return series to analyze
        benchmark_returns (Optional[pd.Series]): Benchmark returns for comparison
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    
    Example:
        >>> performance = create_performance_summary(strategy_returns, market_returns)
        >>> print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {}
    
    # Basic performance metrics
    total_return = (1 + returns_clean).prod() - 1
    annualized_return = (1 + returns_clean.mean()) ** 252 - 1
    annualized_volatility = returns_clean.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns_clean).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Additional metrics
    positive_periods = (returns_clean > 0).sum()
    win_rate = positive_periods / len(returns_clean)
    
    # Downside deviation (for Sortino ratio)
    negative_returns = returns_clean[returns_clean < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    performance_metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'skewness': returns_clean.skew(),
        'kurtosis': returns_clean.kurtosis()
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_clean = benchmark_returns.reindex(returns_clean.index).dropna()
        aligned_returns = returns_clean.reindex(benchmark_clean.index).dropna()
        
        if len(aligned_returns) > 10 and len(benchmark_clean) > 10:
            # Beta calculation
            covariance = np.cov(aligned_returns, benchmark_clean)[0, 1]
            benchmark_variance = np.var(benchmark_clean)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation
            benchmark_annual_return = (1 + benchmark_clean.mean()) ** 252 - 1
            alpha = annualized_return - beta * benchmark_annual_return
            
            # Information ratio
            excess_returns = aligned_returns - benchmark_clean
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            performance_metrics.update({
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'correlation': aligned_returns.corr(benchmark_clean)
            })
    
    return performance_metrics


def print_performance_report(performance_metrics: Dict[str, float]) -> None:
    """
    Print a formatted performance report.
    
    Args:
        performance_metrics (Dict[str, float]): Performance metrics dictionary
    
    Example:
        >>> metrics = create_performance_summary(returns)
        >>> print_performance_report(metrics)
    """
    print("Performance Report")
    print("=" * 50)
    
    # Return metrics
    if 'total_return' in performance_metrics:
        print(f"Total Return:        {format_percentage(performance_metrics['total_return'])}")
    if 'annualized_return' in performance_metrics:
        print(f"Annualized Return:   {format_percentage(performance_metrics['annualized_return'])}")
    
    # Risk metrics
    if 'annualized_volatility' in performance_metrics:
        print(f"Annualized Vol:      {format_percentage(performance_metrics['annualized_volatility'])}")
    if 'max_drawdown' in performance_metrics:
        print(f"Max Drawdown:        {format_percentage(performance_metrics['max_drawdown'])}")
    
    # Risk-adjusted metrics
    if 'sharpe_ratio' in performance_metrics:
        print(f"Sharpe Ratio:        {performance_metrics['sharpe_ratio']:.3f}")
    if 'sortino_ratio' in performance_metrics:
        print(f"Sortino Ratio:       {performance_metrics['sortino_ratio']:.3f}")
    
    # Additional metrics
    if 'win_rate' in performance_metrics:
        print(f"Win Rate:            {format_percentage(performance_metrics['win_rate'])}")
    
    # Benchmark comparison metrics
    if 'beta' in performance_metrics:
        print("\nBenchmark Comparison")
        print("-" * 20)
        print(f"Beta:                {performance_metrics['beta']:.3f}")
        print(f"Alpha:               {format_percentage(performance_metrics['alpha'])}")
        if 'information_ratio' in performance_metrics:
            print(f"Information Ratio:   {performance_metrics['information_ratio']:.3f}")
        if 'correlation' in performance_metrics:
            print(f"Correlation:         {performance_metrics['correlation']:.3f}")


# Example usage
def main():
    """
    Main function to demonstrate utility functions.
    """
    print("Utility Functions Module")
    print("=" * 50)
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252), 
                              index=pd.date_range('2023-01-01', periods=252))
    
    # Calculate performance metrics
    performance = create_performance_summary(sample_returns)
    print_performance_report(performance)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(sample_returns)
    print(f"\nSummary Statistics Shape: {summary_stats.shape}")
    
    # Validate data quality
    sample_data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    quality_report = validate_data_quality(sample_data)
    print(f"\nData Quality Score: {quality_report['quality_score']:.1f}")
    
    print("\nUtility functions demonstration completed!")


if __name__ == "__main__":
    main()



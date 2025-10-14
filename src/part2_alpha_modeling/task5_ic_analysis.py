"""
Task 5: Signal Quality Evaluation

This module implements information coefficient (IC) analysis for evaluating
the predictive power of alpha factors.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import ensure_directory, save_results

warnings.filterwarnings('ignore')


def calculate_information_coefficient(
    factor_series: pd.Series,
    forward_returns: pd.Series,
    method: str = 'cross_sectional'
) -> Dict[str, Any]:
    """
    Calculate information coefficient (IC) for a single factor.
    
    Calculates the correlation coefficient between factor values and future returns to evaluate the predictive power of the factor.
    
    Args:
        factor_series (pd.Series): Time series of a single factor
        forward_returns (pd.Series): Corresponding future returns time series
        method (str): Calculation method, 'cross_sectional' for cross-sectional IC, 'time_series' for time series IC
    
    Returns:
        Dict[str, Any]: IC analysis results, including IC value, t-statistic, p-value, etc.
    
    Example:
        >>> # Get single factor data
        >>> factor_data = factor_dataset['momentum_3']
        >>> forward_returns = calculate_forward_returns(price_data)
        >>> ic_results = calculate_information_coefficient(factor_data, forward_returns)
        >>> print(ic_results)
    
    Notes:
        - Cross-sectional IC: Calculates correlation coefficient between factor values and returns
        - Time series IC: Calculates correlation coefficient between factor time series and return time series
        - Calculates statistical significance of IC
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation:
    #    - Check if series are empty and of correct type
    #    - Ensure sufficient data points (at least 10)
    # 2. Data alignment and cleaning:
    #    - Use index.intersection() to align time indices
    #    - Remove missing values: ~(factor_aligned.isnull() | returns_aligned.isnull())
    # 3. IC calculation:
    #    - Use scipy.stats.pearsonr(factor_valid, returns_valid) to calculate correlation coefficient and p-value
    #    - Calculate t-statistic: correlation * sqrt((n-2)/(1-correlation²))
    # 4. Result organization:
    #    - Create dictionary containing ic_value, t_stat, p_value, n_observations, method
    #    - For this task, cross_sectional and time_series methods have the same implementation
    # 5. Exception handling:
    #    - Check if correlation coefficient is 1 (to avoid division by zero)
    #    - Return NaN if calculation is not feasible
    #
    # Expected output: Dict containing IC value and statistical significance information
    
    raise NotImplementedError("Please implement single factor information coefficient calculation logic")


def calculate_ic_for_multiple_factors(
    factor_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'cross_sectional'
) -> Dict[str, Any]:
    """
    Calculate information coefficient for multiple factors.
    
    Supports two calculation methods:
    1. Time series IC (time_series): N feature factor columns correspond to N label series, output N IC values
    2. Cross-sectional IC (cross_sectional): Each row as a cross-section, each row corresponds to one IC value
    
    Args:
        factor_data (pd.DataFrame): Factor data DataFrame, each column is a factor
        forward_returns (pd.DataFrame): Future returns data DataFrame
        method (str): Calculation method, 'time_series' or 'cross_sectional'
    
    Returns:
        Dict[str, Any]: Data containing IC analysis results
    """
    if factor_data.empty or forward_returns.empty:
        raise ValueError("Input data is empty")
    
    if not isinstance(factor_data, pd.DataFrame) or not isinstance(forward_returns, pd.DataFrame):
        raise ValueError("Input must be pandas.DataFrame type")
    
    # Ensure data alignment
    common_index = factor_data.index.intersection(forward_returns.index)
    if len(common_index) == 0:
        raise ValueError("Factor data and return data have no common time index")
    
    factor_data = factor_data.loc[common_index]
    forward_returns = forward_returns.loc[common_index]
    
    ic_results = {}
    
    if method == 'time_series':
        """
        Time series IC calculation:
        - Calculate time series correlation coefficient between each factor column and corresponding return column
        - Output N IC values (N is the number of factors)
        """
        ic_values = {}
        t_stats = {}
        p_values = {}
        
        # Ensure factor data and return data have consistent column names
        common_columns = factor_data.columns.intersection(forward_returns.columns)
        if len(common_columns) == 0:
            raise ValueError("Factor data and return data have no common column names")
        
        for column in common_columns:
            factor_series = factor_data[column]
            return_series = forward_returns[column]
            
            # Remove missing values
            valid_mask = ~(factor_series.isnull() | return_series.isnull())
            if valid_mask.sum() < 10:
                ic_values[column] = np.nan
                t_stats[column] = np.nan
                p_values[column] = np.nan
                continue
            
            factor_valid = factor_series[valid_mask]
            return_valid = return_series[valid_mask]
            
            # Calculate correlation coefficient
            correlation, p_value = stats.pearsonr(factor_valid, return_valid)
            ic_values[column] = correlation
            
            # Calculate t-statistic
            n = len(factor_valid)
            if n > 2 and abs(correlation) < 1:
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                t_stats[column] = t_stat
                p_values[column] = p_value
            else:
                t_stats[column] = np.nan
                p_values[column] = np.nan
        
        # Create results
        ic_series = pd.Series(ic_values)
        t_stats_series = pd.Series(t_stats)
        p_values_series = pd.Series(p_values)
        
        ic_results = {
            'ic_series': ic_series,
            't_stats': t_stats_series,
            'p_values': p_values_series,
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'ir_ratio': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
            'hit_rate': (ic_series > 0).mean(),
            'method': method
        }
        
    elif method == 'cross_sectional':
        """
        Cross-sectional IC calculation:
        - Each row as a cross-section, calculate cross-sectional correlation coefficient at that time point
        - Output IC value count equals the number of rows
        """
        ic_series = pd.Series(index=common_index, dtype=float)
        t_stats = pd.Series(index=common_index, dtype=float)
        p_values = pd.Series(index=common_index, dtype=float)
        
        # Calculate cross-sectional IC for each time point
        for timestamp in common_index:
            factor_t = factor_data.loc[timestamp]
            return_t = forward_returns.loc[timestamp]
            
            # Remove missing values
            valid_mask = ~(factor_t.isnull() | return_t.isnull())
            if valid_mask.sum() < 3:  # Lower minimum observation requirement as cross-sectional data usually has fewer stocks
                ic_series[timestamp] = np.nan
                t_stats[timestamp] = np.nan
                p_values[timestamp] = np.nan
                continue
            
            factor_valid = factor_t[valid_mask]
            return_valid = return_t[valid_mask]
            
            # Calculate correlation coefficient
            correlation, p_value = stats.pearsonr(factor_valid, return_valid)
            ic_series[timestamp] = correlation
            
            # Calculate t-statistic
            n = len(factor_valid)
            if n > 2 and abs(correlation) < 1:
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                t_stats[timestamp] = t_stat
                p_values[timestamp] = p_value
            else:
                t_stats[timestamp] = np.nan
                p_values[timestamp] = np.nan
        
        # Calculate summary statistics
        ic_results = {
            'ic_series': ic_series,
            't_stats': t_stats,
            'p_values': p_values,
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'ir_ratio': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
            'hit_rate': (ic_series > 0).mean(),
            'method': method
        }
    
    else:
        raise ValueError(f"Unsupported IC calculation method: {method}")
    
    return ic_results


def analyze_ic_stability(
    ic_series: pd.Series,
    window: int = 60
) -> Dict[str, Any]:
    """
    Analyze IC stability.
    
    Calculate rolling IC and analyze IC stability indicators.
    
    Args:
        ic_series (pd.Series): IC time series
        window (int): Rolling window size, default 60 days
    
    Returns:
        Dict[str, Any]: IC stability analysis results
    
    Example:
        >>> ic_results = calculate_ic_for_multiple_factors(factor_data, forward_returns)
        >>> stability_analysis = analyze_ic_stability(ic_results['ic_series'], window=60)
        >>> print(stability_analysis)
    
    Notes:
        - Calculate rolling IC mean and standard deviation
        - Calculate IC stability indicators
        - Analyze IC decay characteristics
    """
    if ic_series.empty:
        raise ValueError("IC time series is empty")
    
    if not isinstance(ic_series, pd.Series):
        raise ValueError("Input must be pandas.Series type")
    
    stability_analysis = {}
    
    # Calculate rolling IC statistics
    rolling_mean = ic_series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = ic_series.rolling(window=window, min_periods=window//2).std()
    rolling_ir = rolling_mean / rolling_std
    
    stability_analysis['rolling_mean'] = rolling_mean
    stability_analysis['rolling_std'] = rolling_std
    stability_analysis['rolling_ir'] = rolling_ir
    
    # Calculate stability indicators
    stability_analysis['ic_stability'] = {
        'mean_ic': ic_series.mean(),
        'std_ic': ic_series.std(),
        'ir_ratio': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
        'hit_rate': (ic_series > 0).mean(),
        'positive_ic_ratio': (ic_series > 0.02).mean(),
        'negative_ic_ratio': (ic_series < -0.02).mean()
    }
    
    # Calculate IC decay analysis
    stability_analysis['ic_decay'] = analyze_ic_decay(ic_series)
    
    # Calculate IC distribution characteristics
    stability_analysis['ic_distribution'] = {
        'skewness': ic_series.skew(),
        'kurtosis': ic_series.kurtosis(),
        'quantiles': ic_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    }
    
    return stability_analysis


def analyze_ic_decay(ic_series: pd.Series) -> Dict[str, float]:
    """
    Analyze IC decay.
    
    Analyze the time decay characteristics of IC.
    
    Args:
        ic_series (pd.Series): IC time series
    
    Returns:
        Dict[str, float]: IC decay analysis results
    """
    if ic_series.empty:
        return {}
    
    if not isinstance(ic_series, pd.Series):
        raise ValueError("Input must be pandas.Series type")
    
    # Calculate autocorrelation coefficients
    autocorr_1 = ic_series.autocorr(lag=1)
    autocorr_5 = ic_series.autocorr(lag=5)
    autocorr_10 = ic_series.autocorr(lag=10)
    
    # Calculate IC persistence
    ic_persistence = {
        'autocorr_1': autocorr_1,
        'autocorr_5': autocorr_5,
        'autocorr_10': autocorr_10,
        'half_life': calculate_half_life(ic_series)
    }
    
    return ic_persistence


def calculate_half_life(series: pd.Series) -> float:
    """
    Calculate half-life.
    
    Calculate the half-life of a time series.
    
    Args:
        series (pd.Series): Time series
    
    Returns:
        float: Half-life
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be pandas.Series type")
    
    if series.empty or series.isnull().all():
        return np.nan
    
    # Remove missing values
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return np.nan
    
    # Calculate first-order autoregression
    lagged = series_clean.shift(1)
    valid_mask = ~(series_clean.isnull() | lagged.isnull())
    
    if valid_mask.sum() < 5:
        return np.nan
    
    y = series_clean[valid_mask]
    x = lagged[valid_mask]
    
    # Add constant term
    x_with_const = np.column_stack([np.ones(len(x)), x])
    
    try:
        # Least squares regression
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        phi = beta[1]  # Autoregression coefficient
        
        if phi > 0 and phi < 1:
            half_life = -np.log(2) / np.log(phi)
            return half_life
        else:
            return np.nan
    except:
        return np.nan


def rank_factors_by_ic(
    ic_results: pd.Series,
    min_ic_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Rank factors based on IC values.
    
    Sort factors according to IC values to identify the most effective factors.
    
    Args:
        ic_results (pd.Series): IC analysis results, index as factor names, values as IC values
        min_ic_threshold (float): Minimum IC threshold, default 0.02
    
    Returns:
        pd.DataFrame: Factor ranking results
    
    Example:
        >>> ic_results = calculate_ic_for_multiple_factors(factor_data, forward_returns)
        >>> factor_ranking = rank_factors_by_ic(ic_results['mean_ic_by_factor'])
        >>> print(factor_ranking)
    
    Notes:
        - Sort by IC value in descending order
        - Filter out factors with IC values below threshold
        - Calculate effectiveness score for factors
    """
    if ic_results.empty:
        raise ValueError("IC results are empty")
    
    if not isinstance(ic_results, pd.Series):
        raise ValueError("Input must be pandas.Series type")
    
    # Create factor ranking DataFrame
    ranking_data = []
    
    for factor in ic_results.index:
        mean_ic = ic_results.loc[factor]
        
        # Calculate effectiveness score
        effectiveness_score = abs(mean_ic) if not pd.isna(mean_ic) else 0
        
        ranking_data.append({
            'factor': factor,
            'mean_ic': mean_ic,
            'abs_ic': abs(mean_ic) if not pd.isna(mean_ic) else 0,
            'effectiveness_score': effectiveness_score,
            'is_effective': abs(mean_ic) >= min_ic_threshold if not pd.isna(mean_ic) else False
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Sort by effectiveness score in descending order
    ranking_df = ranking_df.sort_values('effectiveness_score', ascending=False)
    
    # Add ranking
    ranking_df['rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def plot_ic_analysis(
    ic_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Plot IC analysis charts.
    
    Args:
        ic_results (Dict[str, Any]): IC analysis results
        save_path (Optional[str]): Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. IC time series chart
    if 'ic_series' in ic_results:
        ic_series = ic_results['ic_series']
        axes[0, 0].plot(ic_series.index, ic_series.values, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('IC Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Information Coefficient')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. IC distribution histogram
    if 'ic_series' in ic_results:
        ic_series = ic_results['ic_series'].dropna()
        axes[0, 1].hist(ic_series, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('IC Distribution')
        axes[0, 1].set_xlabel('Information Coefficient')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Factor IC ranking chart
    if 'mean_ic_by_factor' in ic_results:
        mean_ic = ic_results['mean_ic_by_factor'].sort_values(ascending=True)
        axes[1, 0].barh(range(len(mean_ic)), mean_ic.values)
        axes[1, 0].set_yticks(range(len(mean_ic)))
        axes[1, 0].set_yticklabels(mean_ic.index)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Factor IC Ranking')
        axes[1, 0].set_xlabel('Mean IC')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rolling IC chart
    if 'rolling_mean' in ic_results:
        rolling_mean = ic_results['rolling_mean']
        rolling_std = ic_results['rolling_std']
        
        axes[1, 1].plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean IC', alpha=0.7)
        axes[1, 1].fill_between(rolling_mean.index, 
                               rolling_mean.values - rolling_std.values,
                               rolling_mean.values + rolling_std.values,
                               alpha=0.3, label='±1 Std')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Rolling IC Analysis')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Information Coefficient')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IC analysis chart saved to: {save_path}")
    
    plt.show()


def generate_ic_report(
    ic_results: Dict[str, Any],
    factor_ranking: pd.DataFrame,
    save_path: Optional[str] = None
) -> str:
    """
    Generate IC analysis report.
    
    Args:
        ic_results (Dict[str, Any]): IC analysis results
        factor_ranking (pd.DataFrame): Factor ranking results
        save_path (Optional[str]): Save path
    
    Returns:
        str: Report content
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("IC Analysis Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary statistics
    if 'ic_stability' in ic_results:
        stability = ic_results['ic_stability']
        report_lines.append("IC Summary Statistics:")
        report_lines.append(f"  Average IC: {stability.get('mean_ic', np.nan):.4f}")
        report_lines.append(f"  IC Standard Deviation: {stability.get('std_ic', np.nan):.4f}")
        report_lines.append(f"  IR Ratio: {stability.get('ir_ratio', np.nan):.4f}")
        report_lines.append(f"  Positive IC Ratio: {stability.get('hit_rate', np.nan):.2%}")
        report_lines.append(f"  High Positive IC Ratio: {stability.get('positive_ic_ratio', np.nan):.2%}")
        report_lines.append(f"  High Negative IC Ratio: {stability.get('negative_ic_ratio', np.nan):.2%}")
        report_lines.append("")
    
    # Factor ranking
    if not factor_ranking.empty and 'factor' in factor_ranking.columns:
        report_lines.append("Factor Effectiveness Ranking (Top 10):")
        top_factors = factor_ranking.head(10)
        for _, row in top_factors.iterrows():
            report_lines.append(f"  {row['rank']:2d}. {row['factor']:<30} IC: {row['mean_ic']:6.4f}")
        report_lines.append("")
        
        # Effective factor statistics
        if 'is_effective' in factor_ranking.columns:
            effective_factors = factor_ranking[factor_ranking['is_effective']]
            report_lines.append(f"Effective Factor Statistics:")
            report_lines.append(f"  Total Factors: {len(factor_ranking)}")
            report_lines.append(f"  Effective Factors: {len(effective_factors)}")
            report_lines.append(f"  Effectiveness Rate: {len(effective_factors)/len(factor_ranking):.2%}")
            report_lines.append("")
    
    # IC decay analysis
    if 'ic_decay' in ic_results:
        decay = ic_results['ic_decay']
        report_lines.append("IC Decay Analysis:")
        report_lines.append(f"  1-period Autocorrelation: {decay.get('autocorr_1', np.nan):.4f}")
        report_lines.append(f"  5-period Autocorrelation: {decay.get('autocorr_5', np.nan):.4f}")
        report_lines.append(f"  10-period Autocorrelation: {decay.get('autocorr_10', np.nan):.4f}")
        report_lines.append(f"  Half-life: {decay.get('half_life', np.nan):.2f} periods")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"IC analysis report saved to: {save_path}")
    
    return report_content


if __name__ == "__main__":
    # Test code
    from data_loader import DataLoader
    from task4_factors import create_factor_dataset
    
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
    
    # Calculate future returns (need to import related functions from Part1)
    # forward_returns = calculate_forward_returns(data_5min)
    
    print("IC analysis module test completed!")

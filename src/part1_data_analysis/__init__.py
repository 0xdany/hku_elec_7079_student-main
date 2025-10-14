"""
Part 1: Data Analysis & Feature Exploration

This package contains modules for the first part of the quantitative strategy
development project, focusing on data analysis and feature exploration.

Modules:
    task1_returns: Target engineering and return calculation functions
    task2_volatility: Market and asset characterization functions  
    task3_correlation: Cross-sectional analysis functions

Author: ELEC4546/7079 Course
Date: December 2024
"""

from .task1_returns import (
    calculate_forward_returns,
    calculate_weekly_returns,
    plot_return_distribution,
    analyze_return_properties
)

from .task2_volatility import (
    calculate_rolling_volatility,
    build_equal_weight_index,
    plot_volatility_analysis,
    calculate_volatility_statistics,
    compare_individual_vs_market
)

from .task3_correlation import (
    calculate_correlation_matrix,
    calculate_rolling_correlation,
    plot_correlation_heatmap,
    plot_rolling_correlation_analysis,
    analyze_correlation_structure,
    plot_correlation_distribution
)

__all__ = [
    # Task 1 functions
    'calculate_forward_returns',
    'calculate_weekly_returns', 
    'plot_return_distribution',
    'analyze_return_properties',
    
    # Task 2 functions
    'calculate_rolling_volatility',
    'build_equal_weight_index',
    'plot_volatility_analysis',
    'calculate_volatility_statistics',
    'compare_individual_vs_market',
    
    # Task 3 functions
    'calculate_correlation_matrix',
    'calculate_rolling_correlation',
    'plot_correlation_heatmap',
    'plot_rolling_correlation_analysis',
    'analyze_correlation_structure',
    'plot_correlation_distribution'
]



"""
Part 2: Signal Prediction & Alpha Modeling

This package contains modules for alpha factor engineering, signal quality evaluation,
and machine learning model development for quantitative strategy development.

Author: ELEC4546/7079 Course
Date: December 2024
"""

# Import main functions from task modules
from .task4_factors import (
    calculate_momentum_factors,
    calculate_mean_reversion_factors,
    calculate_volume_factors,
    calculate_intraday_factors,
    create_factor_dataset,
    handle_outliers,
    standardize_factor,
    check_factor_quality,
    plot_factor_distributions
)

from .task5_ic_analysis import (
    calculate_information_coefficient,
    analyze_ic_stability,
    analyze_ic_decay,
    calculate_half_life,
    rank_factors_by_ic,
    plot_ic_analysis,
    generate_ic_report
)

from .task6_models import (
    LinearRankingModel,
    TreeRankingModel,
    evaluate_model_performance,
    walk_forward_validation,
    time_series_cv_validation,
    simple_validation,
    calculate_performance_metrics,
    calculate_ranking_metrics,
    compare_models,
    plot_model_comparison,
    save_model_results
)

__all__ = [
    # Task 4: Factor Engineering
    'calculate_momentum_factors',
    'calculate_mean_reversion_factors',
    'calculate_volume_factors',
    'calculate_intraday_factors',
    'create_factor_dataset',
    'handle_outliers',
    'standardize_factor',
    'check_factor_quality',
    'plot_factor_distributions',
    
    # Task 5: IC Analysis
    'calculate_information_coefficient',
    'analyze_ic_stability',
    'analyze_ic_decay',
    'calculate_half_life',
    'rank_factors_by_ic',
    'plot_ic_analysis',
    'generate_ic_report',
    
    # Task 6: Model Development
    'LinearRankingModel',
    'TreeRankingModel',
    'evaluate_model_performance',
    'walk_forward_validation',
    'time_series_cv_validation',
    'simple_validation',
    'calculate_performance_metrics',
    'calculate_ranking_metrics',
    'compare_models',
    'plot_model_comparison',
    'save_model_results'
]

__version__ = '1.0.0'
__author__ = 'ELEC4546/7079 Course'
__email__ = 'course@hku.hk'

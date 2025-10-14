#!/usr/bin/env python3
"""
Simple Part 2 Test Script

This script provides a quick test of all Part 2 functionality
without generating plots or saving files.

Author: ELEC4546/7079 Course
Date: June 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from data_loader import DataLoader
from part2_alpha_modeling.task4_factors import (
    calculate_momentum_factors,
    calculate_mean_reversion_factors,
    calculate_volume_factors,
    calculate_intraday_factors,
    create_factor_dataset,
    check_factor_quality
)
from part2_alpha_modeling.task5_ic_analysis import (
    calculate_ic_for_multiple_factors,
    analyze_ic_stability,
    rank_factors_by_ic,
    generate_ic_report
)
from part2_alpha_modeling.task6_models import (
    LinearRankingModel,
    TreeRankingModel,
    evaluate_model_performance,
    compare_models
)


def test_task4():
    """Test Task 4 functions."""
    print("Testing Task 4: Alpha Factor Engineering")
    print("-" * 45)
    
    # Load data
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    # Use only the first 1000 rows to speed up tests
    data_5min = data_5min.iloc[:1000]
    
    # Select one stock and extract Series inputs
    stocks = data_5min.columns.get_level_values(0).unique()
    stock = stocks[0]
    close_series = data_5min[(stock, 'close_px')]
    volume_series = data_5min[(stock, 'volume')]
    vwap_series = data_5min[(stock, 'vwap')]
    open_series = data_5min[(stock, 'open_px')]
    
    # Test momentum factors
    momentum_factors = calculate_momentum_factors(close_series, periods=[3, 9])
    print(f"✓ Momentum factors shape: {momentum_factors.shape}")
    
    # Test mean reversion factors
    mean_reversion_factors = calculate_mean_reversion_factors(close_series, ma_periods=[12, 24])
    print(f"✓ Mean reversion factors shape: {mean_reversion_factors.shape}")
    
    # Test volume factors
    volume_factors = calculate_volume_factors(volume_series, lookback_periods=[12, 24])
    print(f"✓ Volume factors shape: {volume_factors.shape}")
    
    # Test intraday factors
    intraday_factors = calculate_intraday_factors(close_series, vwap_series, open_series)
    print(f"✓ Intraday factors shape: {intraday_factors.shape}")
    
    # Test factor dataset creation (slice distinct fields to avoid duplicate columns)
    price_only = data_5min.xs('close_px', level=1, axis=1, drop_level=False)
    volume_only = data_5min.xs('volume', level=1, axis=1, drop_level=False)
    vwap_only = data_5min.xs('vwap', level=1, axis=1, drop_level=False)
    open_only = data_5min.xs('open_px', level=1, axis=1, drop_level=False)
    factor_dataset = create_factor_dataset(price_only, volume_only, vwap_only, open_only)
    print(f"✓ Complete factor dataset shape: {factor_dataset.shape}")
    
    # Test factor quality check
    quality_report = check_factor_quality(factor_dataset)
    print(f"✓ Factor quality report generated")
    print(f"  - Total missing ratio: {quality_report['total_missing_ratio']:.4f}")
    print(f"  - Average outlier ratio: {quality_report['avg_outlier_ratio']:.4f}")
    
    return True


def test_task5():
    """Test Task 5 functions."""
    print("\nTesting Task 5: Signal Quality Evaluation")
    print("-" * 42)
    
    # Load data and create factor dataset
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    data_5min = data_5min.iloc[:1000]
    price_only = data_5min.xs('close_px', level=1, axis=1, drop_level=False)
    volume_only = data_5min.xs('volume', level=1, axis=1, drop_level=False)
    vwap_only = data_5min.xs('vwap', level=1, axis=1, drop_level=False)
    open_only = data_5min.xs('open_px', level=1, axis=1, drop_level=False)
    factor_dataset = create_factor_dataset(price_only, volume_only, vwap_only, open_only)
    
    # Create mock forward returns aligned with factor columns
    np.random.seed(42)
    forward_returns = pd.DataFrame(
        np.random.normal(0, 0.01, factor_dataset.shape),
        index=factor_dataset.index,
        columns=factor_dataset.columns
    )
    
    # Test IC calculation for multiple factors (cross-sectional)
    ic_results = calculate_ic_for_multiple_factors(
        factor_dataset, forward_returns, method='cross_sectional'
    )
    print(f"✓ IC calculation completed")
    if 'ic_series' in ic_results:
        print(f"  - Mean IC: {ic_results['mean_ic']:.4f}")
        print(f"  - IC std: {ic_results['std_ic']:.4f}")
        print(f"  - IR ratio: {ic_results['ir_ratio']:.4f}")
        print(f"  - Hit rate: {ic_results['hit_rate']:.2%}")
    
    # Test IC stability analysis
    if 'ic_series' in ic_results:
        stability_analysis = analyze_ic_stability(ic_results['ic_series'], window=20)
        print(f"✓ IC stability analysis completed")
        print(f"  - IC stability metrics calculated")
    
    # Test factor ranking
    # Create mock IC results for ranking
    mock_ic_results = pd.Series([0.05, 0.03, 0.08, 0.02, 0.06], 
                               index=['factor1', 'factor2', 'factor3', 'factor4', 'factor5'])
    factor_ranking = rank_factors_by_ic(mock_ic_results, min_ic_threshold=0.02)
    print(f"✓ Factor ranking completed")
    print(f"  - Top factor: {factor_ranking.iloc[0]['factor']} (IC: {factor_ranking.iloc[0]['mean_ic']:.4f})")
    print(f"  - Effective factors: {factor_ranking['is_effective'].sum()}/{len(factor_ranking)}")
    
    # Test IC report generation
    mock_ic_results_dict = {
        'ic_stability': {
            'mean_ic': 0.05,
            'std_ic': 0.1,
            'ir_ratio': 0.5,
            'hit_rate': 0.6,
            'positive_ic_ratio': 0.3,
            'negative_ic_ratio': 0.2
        },
        'ic_decay': {
            'autocorr_1': 0.8,
            'autocorr_5': 0.6,
            'autocorr_10': 0.4,
            'half_life': 5.2
        }
    }
    ic_report = generate_ic_report(mock_ic_results_dict, factor_ranking)
    print(f"✓ IC report generated ({len(ic_report)} characters)")
    
    return True


def test_task6():
    """Test Task 6 functions."""
    print("\nTesting Task 6: Building a Predictive Ranking Model")
    print("-" * 50)
    
    # Load data and create factor dataset
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    data_5min = data_5min.iloc[:1000]
    price_only = data_5min.xs('close_px', level=1, axis=1, drop_level=False)
    volume_only = data_5min.xs('volume', level=1, axis=1, drop_level=False)
    vwap_only = data_5min.xs('vwap', level=1, axis=1, drop_level=False)
    open_only = data_5min.xs('open_px', level=1, axis=1, drop_level=False)
    factor_dataset = create_factor_dataset(price_only, volume_only, vwap_only, open_only)
    
    # Create mock target variable (forward returns)
    np.random.seed(42)
    target_variable = pd.Series(
        np.random.normal(0, 0.01, len(factor_dataset)),
        index=factor_dataset.index
    )
    
    # Test linear ranking model
    print("Testing Linear Ranking Model...")
    linear_model = LinearRankingModel(alpha=0.01, model_type='ridge')
    linear_model.fit(factor_dataset, target_variable)
    print(f"✓ Linear model trained")
    
    # Test predictions
    linear_predictions = linear_model.predict(factor_dataset)
    print(f"✓ Linear model predictions shape: {linear_predictions.shape}")
    
    # Test feature importance
    linear_importance = linear_model.get_feature_importance()
    print(f"✓ Linear model feature importance calculated")
    print(f"  - Top feature: {linear_importance.index[0]} (importance: {linear_importance.iloc[0]:.4f})")
    
    # Test tree ranking model
    print("\nTesting Tree Ranking Model...")
    try:
        tree_model = TreeRankingModel(model_type='lightgbm', num_leaves=31)
        tree_model.fit(factor_dataset, target_variable)
    except Exception as _:
        print("LightGBM not available or failed, falling back to XGBoost...")
        tree_model = TreeRankingModel(model_type='xgboost', max_depth=6)
        tree_model.fit(factor_dataset, target_variable)
    print(f"✓ Tree model trained")
    
    # Test predictions
    tree_predictions = tree_model.predict(factor_dataset)
    print(f"✓ Tree model predictions shape: {tree_predictions.shape}")
    
    # Test feature importance
    tree_importance = tree_model.get_feature_importance()
    print(f"✓ Tree model feature importance calculated")
    print(f"  - Top feature: {tree_importance.index[0]} (importance: {tree_importance.iloc[0]:.4f})")
    
    # Test model evaluation
    print("\nTesting Model Evaluation...")
    linear_performance = evaluate_model_performance(
        linear_model, factor_dataset, target_variable, validation_method='simple'
    )
    print(f"✓ Linear model performance evaluated")
    print(f"  - R²: {linear_performance['r2']:.4f}")
    print(f"  - RMSE: {linear_performance['rmse']:.4f}")
    print(f"  - Mean IC: {linear_performance['mean_ic']:.4f}")
    
    tree_performance = evaluate_model_performance(
        tree_model, factor_dataset, target_variable, validation_method='simple'
    )
    print(f"✓ Tree model performance evaluated")
    print(f"  - R²: {tree_performance['r2']:.4f}")
    print(f"  - RMSE: {tree_performance['rmse']:.4f}")
    print(f"  - Mean IC: {tree_performance['mean_ic']:.4f}")
    
    # Test model comparison
    print("\nTesting Model Comparison...")
    models = {
        'ridge': linear_model,
        tree_model.model_type: tree_model
    }
    comparison_df = compare_models(models, factor_dataset, target_variable, validation_method='simple')
    print(f"✓ Model comparison completed")
    print(f"  - Comparison shape: {comparison_df.shape}")
    print(f"  - Best R²: {comparison_df['r2'].max():.4f}")
    print(f"  - Best IC: {comparison_df['mean_ic'].max():.4f}")
    
    return True


def test_integration():
    """Test integration of all tasks."""
    print("\nTesting Part 2 Integration")
    print("-" * 30)
    
    # Load data
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    data_5min = data_5min.iloc[:1000]
    
    # Create factor dataset
    print("Creating factor dataset...")
    price_only = data_5min.xs('close_px', level=1, axis=1, drop_level=False)
    volume_only = data_5min.xs('volume', level=1, axis=1, drop_level=False)
    vwap_only = data_5min.xs('vwap', level=1, axis=1, drop_level=False)
    open_only = data_5min.xs('open_px', level=1, axis=1, drop_level=False)
    factor_dataset = create_factor_dataset(price_only, volume_only, vwap_only, open_only)
    
    # Create target variable
    np.random.seed(42)
    target_variable = pd.Series(
        np.random.normal(0, 0.01, len(factor_dataset)),
        index=factor_dataset.index
    )
    
    # Train models
    print("Training models...")
    linear_model = LinearRankingModel(alpha=0.01, model_type='ridge')
    try:
        tree_model = TreeRankingModel(model_type='lightgbm')
        tree_model.fit(factor_dataset, target_variable)
    except Exception as _:
        print("LightGBM not available or failed, falling back to XGBoost...")
        tree_model = TreeRankingModel(model_type='xgboost')
        tree_model.fit(factor_dataset, target_variable)
    
    linear_model.fit(factor_dataset, target_variable)
    
    # Evaluate models
    print("Evaluating models...")
    linear_performance = evaluate_model_performance(
        linear_model, factor_dataset, target_variable, validation_method='simple'
    )
    tree_performance = evaluate_model_performance(
        tree_model, factor_dataset, target_variable, validation_method='simple'
    )
    
    # Compare models
    models = {'ridge': linear_model, 'lightgbm': tree_model}
    comparison_df = compare_models(models, factor_dataset, target_variable, validation_method='simple')
    
    print(f"✓ Integration test completed")
    print(f"  - Factor dataset: {factor_dataset.shape}")
    print(f"  - Linear model R²: {linear_performance['r2']:.4f}")
    print(f"  - Tree model R²: {tree_performance['r2']:.4f}")
    print(f"  - Best model: {comparison_df['r2'].idxmax()}")
    
    return True


def main():
    """Run all tests."""
    print("Part 2: Signal Prediction & Alpha Modeling")
    print("=" * 50)
    
    try:
        # Test individual tasks
        test_task4()
        test_task5()
        test_task6()
        
        # Test integration
        test_integration()
        
        print("\n" + "=" * 50)
        print("All Part 2 tests completed successfully! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

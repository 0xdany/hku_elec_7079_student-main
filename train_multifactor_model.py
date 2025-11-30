#!/usr/bin/env python
"""
Multi-Factor ML Model Training Script

This script trains a multi-factor ML model for stock prediction and saves
the predictions for use in backtesting.

Usage:
    uv run python train_multifactor_model.py

Output:
    - results/part3/ml_predictions.pkl - Prediction scores DataFrame
    - results/part3/ml_model_info.json - Model metadata
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader
from part3_strategy.task7_backtest import (
    LongShortStrategy, _extract_close_prices, _pct_change_returns
)
from part3_strategy.task8_performance import calculate_performance_metrics


def compute_factors(prices: pd.DataFrame, volumes: pd.DataFrame = None) -> dict:
    """Compute all factors for the entire price matrix."""
    print("\n[1/4] Computing factors...")
    
    factors = {}
    
    # Reversal factors (negative past returns)
    for lb in tqdm([6, 12, 24, 48], desc="  Reversal factors"):
        factors[f'reversal_{lb}'] = -prices.pct_change(lb)
    
    # Mean reversion factors
    for window in tqdm([24, 48, 96], desc="  Mean reversion"):
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        factors[f'ma_dev_{window}'] = -(prices - ma) / std.replace(0, np.nan)
    
    # Volatility (lower = better for reversal)
    for window in tqdm([12, 24, 48], desc="  Volatility"):
        factors[f'volatility_{window}'] = -prices.pct_change().rolling(window).std()
    
    # Volume factors (if available)
    if volumes is not None:
        for window in tqdm([24, 48], desc="  Volume factors"):
            vol_ma = volumes.rolling(window).mean()
            factors[f'vol_ratio_{window}'] = -(volumes / vol_ma.replace(0, np.nan) - 1)
    
    print(f"  Total factors: {len(factors)}")
    return factors


def prepare_training_data(factors: dict, forward_returns: pd.DataFrame, 
                          train_end_idx: int) -> tuple:
    """Prepare training data by stacking all stocks."""
    print("\n[2/4] Preparing training data...")
    
    X_list = []
    y_list = []
    factor_names = list(factors.keys())
    symbols = list(forward_returns.columns)
    
    for sym in tqdm(symbols, desc="  Processing stocks"):
        # Stack factors for this stock
        factor_data = pd.DataFrame({
            fname: factors[fname][sym] for fname in factor_names
        })
        
        # Get labels
        y = forward_returns[sym]
        
        # Training slice
        factor_train = factor_data.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        
        # Valid mask (no NaNs)
        valid_mask = factor_train.notna().all(axis=1) & y_train.notna()
        
        if valid_mask.sum() > 100:
            X_list.append(factor_train[valid_mask])
            y_list.append(y_train[valid_mask])
    
    X_train = pd.concat(X_list, axis=0)
    y_train = pd.concat(y_list, axis=0)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {len(factor_names)}")
    
    return X_train, y_train, factor_names


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                model_type: str = 'linear') -> tuple:
    """Train the prediction model."""
    print("\n[3/4] Training model...")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.fillna(0))
    
    # Train
    if model_type == 'linear':
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_train.fillna(0))
        print(f"  Model: Ridge Regression (alpha=1.0)")
    else:
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05,
                verbose=-1
            )
            model.fit(X_scaled, y_train.fillna(0))
            print(f"  Model: LightGBM")
        except ImportError:
            print("  LightGBM not available, using Ridge")
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_train.fillna(0))
    
    # Feature importance
    if hasattr(model, 'coef_'):
        importance = pd.Series(np.abs(model.coef_), index=X_train.columns)
        print(f"\n  Top 5 features:")
        for fname, imp in importance.nlargest(5).items():
            print(f"    {fname}: {imp:.4f}")
    
    return model, scaler


def generate_predictions(factors: dict, model, scaler, 
                        train_end_idx: int, time_index, symbols) -> pd.DataFrame:
    """Generate out-of-sample predictions."""
    print("\n[4/4] Generating predictions...")
    
    factor_names = list(factors.keys())
    n_samples = len(time_index)
    
    # Initialize predictions DataFrame
    predictions = pd.DataFrame(np.nan, index=time_index, columns=symbols)
    
    # Only predict from train_end_idx onwards
    for t_idx in tqdm(range(train_end_idx, n_samples), desc="  Predicting"):
        ts = time_index[t_idx]
        
        for sym in symbols:
            # Get factor values at this timestamp
            factor_vals = []
            valid = True
            
            for fname in factor_names:
                val = factors[fname][sym].iloc[t_idx]
                if pd.isna(val):
                    valid = False
                    break
                factor_vals.append(val)
            
            if valid:
                X = scaler.transform([factor_vals])
                predictions.loc[ts, sym] = model.predict(X)[0]
    
    # Count valid predictions
    valid_preds = predictions.notna().sum().sum()
    total_possible = (n_samples - train_end_idx) * len(symbols)
    print(f"  Valid predictions: {valid_preds:,} / {total_possible:,} ({valid_preds/total_possible*100:.1f}%)")
    
    return predictions


def run_backtest(prices, returns, predictions):
    """Run backtest and return metrics."""
    print("\n[5/5] Running backtest...")
    
    strat = LongShortStrategy(
        long_quantile=0.10,
        short_quantile=0.10,
        rebalance_periods=4800,
        transaction_cost=0.0005,
        signal_type='predictions',
    )
    
    results = strat.backtest(returns=returns, prices=prices, predictions=predictions)
    metrics = calculate_performance_metrics(results['returns'])
    
    return results, metrics


def main():
    print("="*70)
    print("MULTI-FACTOR ML MODEL TRAINING")
    print("="*70)
    
    # Configuration
    TRAIN_RATIO = 0.6
    FORWARD_PERIODS = 12
    MODEL_TYPE = 'linear'  # 'linear' or 'tree'
    
    print(f"\nConfiguration:")
    print(f"  Train ratio: {TRAIN_RATIO}")
    print(f"  Forward periods: {FORWARD_PERIODS}")
    print(f"  Model type: {MODEL_TYPE}")
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    data_5min = loader.load_5min_data()
    
    prices = _extract_close_prices(data_5min)
    returns = _pct_change_returns(prices)
    volumes = data_5min.xs('volume', axis=1, level=1)
    
    symbols = list(prices.columns)
    time_index = prices.index
    n_samples = len(time_index)
    train_end_idx = int(n_samples * TRAIN_RATIO)
    
    print(f"  Samples: {n_samples:,}")
    print(f"  Stocks: {len(symbols)}")
    print(f"  Training samples: {train_end_idx:,}")
    print(f"  Test samples: {n_samples - train_end_idx:,}")
    
    # Compute factors
    factors = compute_factors(prices, volumes)
    
    # Compute forward returns (labels)
    forward_returns = prices.pct_change(FORWARD_PERIODS).shift(-FORWARD_PERIODS)
    
    # Prepare training data
    X_train, y_train, factor_names = prepare_training_data(
        factors, forward_returns, train_end_idx
    )
    
    # Train model
    model, scaler = train_model(X_train, y_train, MODEL_TYPE)
    
    # Generate predictions
    predictions = generate_predictions(
        factors, model, scaler, train_end_idx, time_index, symbols
    )
    
    # Save results
    output_dir = Path("results/part3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = output_dir / "ml_predictions.pkl"
    predictions.to_pickle(pred_path)
    print(f"\nPredictions saved to: {pred_path}")
    
    # Save model info
    info = {
        'train_ratio': TRAIN_RATIO,
        'forward_periods': FORWARD_PERIODS,
        'model_type': MODEL_TYPE,
        'factor_names': factor_names,
        'n_train_samples': len(X_train),
        'n_stocks': len(symbols),
        'train_end_date': str(time_index[train_end_idx]),
    }
    info_path = output_dir / "ml_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved to: {info_path}")
    
    # Run backtest
    results, metrics = run_backtest(prices, returns, predictions)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nMulti-Factor ML Strategy:")
    print(f"  Total Return: {metrics['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:+.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    
    # Compare with baseline
    print("\nComparing with baseline reversal...")
    strat_rev = LongShortStrategy(
        signal_type='reversal', 
        rebalance_periods=4800, 
        signal_params={'lookback': 6}
    )
    results_rev = strat_rev.backtest(returns=returns, prices=prices)
    metrics_rev = calculate_performance_metrics(results_rev['returns'])
    
    print(f"\nBaseline Reversal Strategy:")
    print(f"  Total Return: {metrics_rev['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {metrics_rev['sharpe_ratio']:+.3f}")
    print(f"  Max Drawdown: {metrics_rev['max_drawdown']*100:.2f}%")
    
    print(f"\nImprovement:")
    print(f"  Return: {(metrics['total_return'] - metrics_rev['total_return'])*100:+.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio'] - metrics_rev['sharpe_ratio']:+.3f}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()


"""
Task 6: Building a Predictive Ranking Model

This module implements machine learning models for stock ranking prediction
using engineered alpha factors.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import ensure_directory, save_results

warnings.filterwarnings('ignore')


class LinearRankingModel:
    """
    Linear ranking model.
    
    Uses linear regression models for stock ranking prediction with regularization support.
    
    Attributes:
        alpha (float): Regularization strength
        l1_ratio (float): L1 regularization ratio (0 for L2, 1 for L1)
        model: Trained model
        scaler: Feature standardizer
        feature_names (List[str]): Feature name list
    """
    
    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, model_type: str = 'elastic_net'):
        """
        Initialize linear ranking model.
        
        Args:
            alpha (float): Regularization strength, default 0.01
            l1_ratio (float): L1 regularization ratio, default 0.5 (ElasticNet)
            model_type (str): Model type, 'ridge', 'lasso', 'elastic_net'
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Select model based on model type
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        else:
            raise ValueError(f"Unsupported linear model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRankingModel':
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target variable (future returns)
        
        Returns:
            LinearRankingModel: Trained model
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Input validation: Check if X and y are empty
        # 2. Save feature names: self.feature_names = X.columns.tolist()
        # 3. Data preprocessing:
        #    - Handle missing values: X.fillna(X.median()), y.fillna(y.median())
        #    - Data alignment: Use index.intersection() to ensure X and y indices are consistent
        # 4. Feature standardization:
        #    - Use self.scaler.fit_transform(X_clean) to standardize features
        # 5. Model training:
        #    - Call self.model.fit(X_scaled, y_clean) to train the model
        #    - Set self.is_fitted = True to mark model as trained
        # 6. Return self to support chaining
        #
        # Expected output: Trained LinearRankingModel instance
        
        raise NotImplementedError("Please implement linear model training logic")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores.
        
        Args:
            X (pd.DataFrame): Feature data
        
        Returns:
            np.ndarray: Predicted ranking scores
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Model status check: Check self.is_fitted to ensure model is trained
        # 2. Input validation: Check if X is empty
        # 3. Data preprocessing:
        #    - Handle missing values: X.fillna(X.median())
        # 4. Feature standardization:
        #    - Use self.scaler.transform(X_clean) to standardize features
        #    - Note: Use transform instead of fit_transform (model already trained)
        # 5. Model prediction:
        #    - Call self.model.predict(X_scaled) to get prediction results
        # 6. Return prediction scores in numpy array format
        #
        # Expected output: numpy.ndarray containing predicted ranking scores for each sample
        
        raise NotImplementedError("Please implement linear model prediction logic")
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance.
        
        Returns:
            pd.Series: Feature importance
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Status check: Ensure model is trained and feature_names are saved
        # 2. Get model coefficients: coefficients = self.model.coef_
        # 3. Create feature importance Series:
        #    - Create Series with coefficients: pd.Series(coefficients, index=self.feature_names)
        #    - Take absolute values: .abs() (linear model importance usually looks at absolute coefficient values)
        # 4. Return feature importance Series
        #
        # Expected output: pandas.Series with feature names as index and importance scores (absolute values) as values
        
        raise NotImplementedError("Please implement feature importance calculation logic")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model.
        
        Args:
            filepath (str): Save path
        """
        if not self.is_fitted:
            raise ValueError("Model not yet trained")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LinearRankingModel':
        """
        Load the model.
        
        Args:
            filepath (str): Model file path
        
        Returns:
            LinearRankingModel: Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            alpha=model_data['alpha'],
            l1_ratio=model_data['l1_ratio'],
            model_type=model_data['model_type']
        )
        
        # Restore model state
        model.model = model_data['model']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.is_fitted = True
        
        return model


class TreeRankingModel:
    """
    Tree model ranking.
    
    Uses LightGBM or XGBoost for stock ranking prediction.
    
    Attributes:
        model_type (str): Model type, 'lightgbm' or 'xgboost'
        model: Trained model
        feature_names (List[str]): Feature name list
        params (Dict): Model parameters
    """
    
    def __init__(self, model_type: str = 'lightgbm', **params):
        """
        Initialize tree model.
        
        Args:
            model_type (str): Model type, 'lightgbm' or 'xgboost'
            **params: Model parameters
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.params = params
        self.is_fitted = False
        
        # Set default parameters
        if model_type == 'lightgbm':
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        elif model_type == 'xgboost':
            default_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'random_state': 42
            }
        else:
            raise ValueError(f"Unsupported tree model type: {model_type}")
        
        # Update parameters
        self.params.update(default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TreeRankingModel':
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target variable (future returns)
        
        Returns:
            TreeRankingModel: Trained model
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Input validation and data preprocessing (refer to LinearRankingModel implementation)
        # 2. Save feature names: self.feature_names = X.columns.tolist()
        # 3. Choose training method based on model_type:
        #    a) If model_type == 'lightgbm':
        #       - Create LightGBM dataset: train_data = lgb.Dataset(X_clean, label=y_clean)
        #       - Train model: lgb.train(self.params, train_data, num_boost_round=100, ...)
        #       - Add early stopping: callbacks=[lgb.early_stopping(stopping_rounds=10)]
        #    b) If model_type == 'xgboost':
        #       - Create XGBoost regressor: self.model = xgb.XGBRegressor(**self.params)
        #       - Train model: self.model.fit(X_clean, y_clean)
        # 4. Set self.is_fitted = True and return self
        #
        # Expected output: Trained TreeRankingModel instance
        
        raise NotImplementedError("Please implement tree model training logic")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores.
        
        Args:
            X (pd.DataFrame): Feature data
        
        Returns:
            np.ndarray: Predicted ranking scores
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Model status check: Ensure self.is_fitted is True
        # 2. Input validation: Check if X is empty
        # 3. Data preprocessing: Handle missing values X.fillna(X.median())
        # 4. Make predictions based on model_type:
        #    - If model_type == 'lightgbm': predictions = self.model.predict(X_clean)
        #    - If model_type == 'xgboost': predictions = self.model.predict(X_clean)
        # 5. Return prediction results in numpy array format
        #
        # Expected output: numpy.ndarray containing predicted ranking scores for each sample
        
        raise NotImplementedError("Please implement tree model prediction logic")
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance.
        
        Returns:
            pd.Series: Feature importance
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Implementation hints:
        # 1. Status check: Ensure model is trained and feature_names are saved
        # 2. Get feature importance based on model_type:
        #    - If model_type == 'lightgbm':
        #      importance = self.model.feature_importance(importance_type='gain')
        #    - If model_type == 'xgboost':
        #      importance = self.model.feature_importances_
        # 3. Create feature importance Series:
        #    - Create with importance values and feature names: pd.Series(importance, index=self.feature_names)
        # 4. Return feature importance Series
        #
        # Expected output: pandas.Series with feature names as index and importance scores as values
        
        raise NotImplementedError("Please implement tree model feature importance calculation logic")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model.
        
        Args:
            filepath (str): Save path
        """
        if not self.is_fitted:
            raise ValueError("Model not yet trained")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TreeRankingModel':
        """
        Load the model.
        
        Args:
            filepath (str): Model file path
        
        Returns:
            TreeRankingModel: Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            model_type=model_data['model_type'],
            **model_data['params']
        )
        
        # Restore model state
        model.model = model_data['model']
        model.feature_names = model_data['feature_names']
        model.is_fitted = True
        
        return model


def evaluate_model_performance(
    model: Union[LinearRankingModel, TreeRankingModel],
    X: pd.DataFrame,
    y: pd.Series,
    validation_method: str = 'walk_forward',
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Use Walk-Forward validation and other methods to evaluate model performance.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature data
        y (pd.Series): Target variable
        validation_method (str): Validation method, 'walk_forward' or 'cross_validation'
        **kwargs: Other parameters
    
    Returns:
        Dict[str, Any]: Performance evaluation results
    
    Example:
        >>> model = LinearRankingModel()
        >>> model.fit(X_train, y_train)
        >>> performance = evaluate_model_performance(model, X_test, y_test)
        >>> print(performance)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    #
    # Implementation hints:
    # 1. Input validation: Check if X and y are empty
    # 2. Data alignment: Use index.intersection() to ensure X and y time indices are consistent
    # 3. Choose validation method based on validation_method:
    #    - 'walk_forward': Call walk_forward_validation(model, X, y, **kwargs)
    #    - 'cross_validation': Call time_series_cv_validation(model, X, y, **kwargs)
    #    - Others: Call simple_validation(model, X, y)
    # 4. Return performance evaluation result dictionary
    #
    # Expected output: Dict[str, Any] containing various performance metrics
    
    raise NotImplementedError("Please implement model performance evaluation logic")


def walk_forward_validation(
    model: Union[LinearRankingModel, TreeRankingModel],
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    step_size: int = 100
) -> Dict[str, Any]:
    """
    Walk-Forward validation.
    
    Args:
        model: Model
        X (pd.DataFrame): Feature data
        y (pd.Series): Target variable
        train_size (float): Training set ratio
        step_size (int): Step size
    
    Returns:
        Dict[str, Any]: Validation results
    """
    total_samples = len(X)
    train_samples = int(total_samples * train_size)
    
    predictions = []
    actuals = []
    ic_scores = []
    
    for i in range(train_samples, total_samples, step_size):
        # Split training and test sets
        train_end = i
        test_end = min(i + step_size, total_samples)
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        
        # Train model
        model_copy = type(model)()
        if isinstance(model, LinearRankingModel):
            model_copy = LinearRankingModel(
                alpha=model.alpha,
                l1_ratio=model.l1_ratio,
                model_type=model.model_type
            )
        elif isinstance(model, TreeRankingModel):
            model_copy = TreeRankingModel(
                model_type=model.model_type,
                **model.params
            )
        
        model_copy.fit(X_train, y_train)
        
        # Predict
        pred = model_copy.predict(X_test)
        predictions.extend(pred)
        actuals.extend(y_test.values)
        
        # Calculate IC
        if len(pred) > 10:
            ic = np.corrcoef(pred, y_test.values)[0, 1]
            if not np.isnan(ic):
                ic_scores.append(ic)
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(
        np.array(predictions), 
        np.array(actuals),
        ic_scores
    )
    
    return performance


def time_series_cv_validation(
    model: Union[LinearRankingModel, TreeRankingModel],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Time series cross-validation.
    
    Args:
        model: Model
        X (pd.DataFrame): Feature data
        y (pd.Series): Target variable
        n_splits (int): Number of splits
    
    Returns:
        Dict[str, Any]: Validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    predictions = []
    actuals = []
    ic_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model_copy = type(model)()
        if isinstance(model, LinearRankingModel):
            model_copy = LinearRankingModel(
                alpha=model.alpha,
                l1_ratio=model.l1_ratio,
                model_type=model.model_type
            )
        elif isinstance(model, TreeRankingModel):
            model_copy = TreeRankingModel(
                model_type=model.model_type,
                **model.params
            )
        
        model_copy.fit(X_train, y_train)
        
        # Predict
        pred = model_copy.predict(X_test)
        predictions.extend(pred)
        actuals.extend(y_test.values)
        
        # Calculate IC
        if len(pred) > 10:
            ic = np.corrcoef(pred, y_test.values)[0, 1]
            if not np.isnan(ic):
                ic_scores.append(ic)
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(
        np.array(predictions), 
        np.array(actuals),
        ic_scores
    )
    
    return performance


def simple_validation(
    model: Union[LinearRankingModel, TreeRankingModel],
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, Any]:
    """
    Simple validation.
    
    Args:
        model: Model
        X (pd.DataFrame): Feature data
        y (pd.Series): Target variable
    
    Returns:
        Dict[str, Any]: Validation results
    """
    # Predict
    predictions = model.predict(X)
    actuals = y.values
    
    # Calculate IC
    ic = np.corrcoef(predictions, actuals)[0, 1]
    ic_scores = [ic] if not np.isnan(ic) else []
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(
        predictions, 
        actuals,
        ic_scores
    )
    
    return performance


def calculate_performance_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    ic_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculate performance metrics
    
    Args:
        predictions (np.ndarray): Predicted values
        actuals (np.ndarray): Actual values
        ic_scores (List[float]): IC scores list
    
    Returns:
        Dict[str, Any]: Performance metrics
    """
    # Remove missing values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    pred_clean = predictions[valid_mask]
    act_clean = actuals[valid_mask]
    
    if len(pred_clean) == 0:
        return {}
    
    # Regression metrics
    mse = mean_squared_error(act_clean, pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(act_clean, pred_clean)
    r2 = r2_score(act_clean, pred_clean)
    
    # IC metrics
    mean_ic = np.mean(ic_scores) if ic_scores else np.nan
    std_ic = np.std(ic_scores) if ic_scores else np.nan
    # When there's only one IC value, set IR ratio to 0 (no variation)
    ir_ratio = mean_ic / std_ic if std_ic != 0 and not np.isnan(std_ic) and not np.isnan(mean_ic) else (0.0 if not np.isnan(mean_ic) else np.nan)
    
    # Ranking metrics
    ranking_metrics = calculate_ranking_metrics(pred_clean, act_clean)
    
    performance = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ir_ratio': ir_ratio,
        'hit_rate': (mean_ic > 0) if not np.isnan(mean_ic) else np.nan,
        'ranking_metrics': ranking_metrics
    }
    
    return performance


def calculate_ranking_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Calculate ranking-related metrics
    
    Args:
        predictions (np.ndarray): Predicted values
        actuals (np.ndarray): Actual values
    
    Returns:
        Dict[str, float]: Ranking metrics
    """
    # Calculate ranking correlation
    pred_ranks = pd.Series(predictions).rank(ascending=False)
    act_ranks = pd.Series(actuals).rank(ascending=False)
    
    # Spearman correlation coefficient
    spearman_corr = pred_ranks.corr(act_ranks, method='spearman')
    
    # Calculate Top-K hit rates
    k_values = [5, 10, 20]
    top_k_hit_rates = {}
    
    for k in k_values:
        if len(predictions) >= k:
            top_k_pred = np.argsort(predictions)[-k:]
            top_k_act = np.argsort(actuals)[-k:]
            hit_rate = len(set(top_k_pred) & set(top_k_act)) / k
            top_k_hit_rates[f'top_{k}_hit_rate'] = hit_rate
    
    ranking_metrics = {
        'spearman_correlation': spearman_corr,
        **top_k_hit_rates
    }
    
    return ranking_metrics


def compare_models(
    models: Dict[str, Union[LinearRankingModel, TreeRankingModel]],
    X: pd.DataFrame,
    y: pd.Series,
    validation_method: str = 'walk_forward'
) -> pd.DataFrame:
    """
    Compare performance of multiple models
    
    Args:
        models (Dict): Model dictionary
        X (pd.DataFrame): Feature data
        y (pd.Series): Target variable
        validation_method (str): Validation method
    
    Returns:
        pd.DataFrame: Model comparison results
    """
    comparison_results = []
    
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        
        # Evaluate model performance
        performance = evaluate_model_performance(
            model, X, y, validation_method
        )
        
        # Add model name
        performance['model_name'] = model_name
        
        comparison_results.append(performance)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.set_index('model_name')
    
    return comparison_df


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison charts.
    
    Args:
        comparison_df (pd.DataFrame): Model comparison results
        save_path (Optional[str]): Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RMSE comparison
    if 'rmse' in comparison_df.columns:
        axes[0, 0].bar(comparison_df.index, comparison_df['rmse'])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. R² comparison
    if 'r2' in comparison_df.columns:
        axes[0, 1].bar(comparison_df.index, comparison_df['r2'])
        axes[0, 1].set_title('R² Comparison')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. IC comparison
    if 'mean_ic' in comparison_df.columns:
        axes[1, 0].bar(comparison_df.index, comparison_df['mean_ic'])
        axes[1, 0].set_title('Mean IC Comparison')
        axes[1, 0].set_ylabel('Mean IC')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. IR ratio comparison
    if 'ir_ratio' in comparison_df.columns:
        axes[1, 1].bar(comparison_df.index, comparison_df['ir_ratio'])
        axes[1, 1].set_title('IR Ratio Comparison')
        axes[1, 1].set_ylabel('IR Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison chart saved to: {save_path}")
    
    plt.show()


def save_model_results(
    model: Union[LinearRankingModel, TreeRankingModel],
    performance: Dict[str, Any],
    model_name: str,
    save_dir: str = "results/models/part2"
) -> None:
    """
    Save model results.
    
    Args:
        model: Trained model
        performance (Dict[str, Any]): Performance metrics
        model_name (str): Model name
        save_dir (str): Save directory
    """
    # Ensure directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / f"{model_name}.pkl"
    model.save_model(str(model_path))
    
    # Save performance metrics
    performance_path = save_dir / f"{model_name}_performance.json"
    with open(performance_path, 'w') as f:
        json.dump(performance, f, indent=2, default=str)
    
    # Save feature importance
    feature_importance = model.get_feature_importance()
    importance_path = save_dir / f"{model_name}_feature_importance.csv"
    feature_importance.to_csv(importance_path)
    
    print(f"Model results saved to: {save_dir}")


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
    
    print("Model building module test completed!")

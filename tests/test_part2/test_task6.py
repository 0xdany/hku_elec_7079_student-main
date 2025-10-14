"""
Task 6 Unit Tests

Test cases for machine learning model functionality.

Author: ELEC4546/7079 Course
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import tempfile
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from part2_alpha_modeling.task6_models import (
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


class TestLinearRankingModel:
    """Test linear ranking model"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic feature data
        n_samples = 100
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target variable (with correlation to features)
        self.y = pd.Series(
            np.random.normal(0, 0.1, n_samples) + 0.1 * np.sum(self.X.values, axis=1),
            index=self.X.index
        )
    
    def test_linear_ranking_model_initialization(self):
        """Test linear model initialization"""
        # Test Ridge model
        ridge_model = LinearRankingModel(alpha=0.1, model_type='ridge')
        assert ridge_model.alpha == 0.1
        assert ridge_model.model_type == 'ridge'
        assert not ridge_model.is_fitted
        
        # Test Lasso model
        lasso_model = LinearRankingModel(alpha=0.01, model_type='lasso')
        assert lasso_model.model_type == 'lasso'
        
        # Test ElasticNet model
        elastic_model = LinearRankingModel(alpha=0.01, l1_ratio=0.7, model_type='elastic_net')
        assert elastic_model.l1_ratio == 0.7
        
        # Unsupported model type
        with pytest.raises(ValueError, match="(?i)unsupported"):
            LinearRankingModel(model_type='invalid')
    
    def test_linear_ranking_model_fit(self):
        """Test linear model training"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        
        # Fit model
        fitted_model = model.fit(self.X, self.y)
        
        # Check model state
        assert fitted_model.is_fitted
        assert fitted_model.feature_names == list(self.X.columns)
        assert fitted_model.model is not None
        
        # Should return self
        assert fitted_model is model
    
    def test_linear_ranking_model_fit_with_missing_data(self):
        """Test training with missing data"""
        # Add missing values
        X_with_missing = self.X.copy()
        X_with_missing.iloc[0, 0] = np.nan
        
        y_with_missing = self.y.copy()
        y_with_missing.iloc[1] = np.nan
        
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        fitted_model = model.fit(X_with_missing, y_with_missing)
        
        # Model should fit successfully
        assert fitted_model.is_fitted
    
    def test_linear_ranking_model_predict(self):
        """Test linear model prediction"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        # Predict
        predictions = model.predict(self.X)
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.X)
        assert not np.any(np.isnan(predictions))
        
        # Predict with untrained model
        untrained_model = LinearRankingModel(alpha=0.01, model_type='ridge')
        with pytest.raises(ValueError, match="(?i)not\s*trained"):
            untrained_model.predict(self.X)
        
        # Predict with empty data
        with pytest.raises(ValueError, match="(?i)empty"):
            model.predict(pd.DataFrame())
    
    def test_linear_ranking_model_feature_importance(self):
        """Test linear model feature importance"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Check feature importance
        assert isinstance(feature_importance, pd.Series)
        assert len(feature_importance) == len(self.X.columns)
        assert all(feature_importance >= 0)  # importance should be non-negative
        assert feature_importance.index.equals(pd.Index(self.X.columns))
        
        # Untrained model feature importance
        untrained_model = LinearRankingModel(alpha=0.01, model_type='ridge')
        with pytest.raises(ValueError, match="(?i)not\s*trained"):
            untrained_model.get_feature_importance()
    
    def test_linear_ranking_model_save_load(self):
        """Test saving and loading linear ranking model"""
        model = LinearRankingModel(alpha=0.01, l1_ratio=0.5, model_type='elastic_net')
        model.fit(self.X, self.y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            model.save_model(model_path)
            
            # Load model
            loaded_model = LinearRankingModel.load_model(model_path)
            
            # Check loaded model
            assert loaded_model.is_fitted
            assert loaded_model.alpha == model.alpha
            assert loaded_model.l1_ratio == model.l1_ratio
            assert loaded_model.model_type == model.model_type
            assert loaded_model.feature_names == model.feature_names
            
            # Predictions should match
            original_predictions = model.predict(self.X)
            loaded_predictions = loaded_model.predict(self.X)
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
            
        finally:
            # Cleanup temp file
            if os.path.exists(model_path):
                os.unlink(model_path)
        
        # Saving before training should fail
        untrained_model = LinearRankingModel(alpha=0.01, model_type='ridge')
        with pytest.raises(ValueError, match="(?i)not\s*trained"):
            untrained_model.save_model('test.pkl')


class TestTreeRankingModel:
    """Test tree-based ranking model"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic feature data
        n_samples = 100
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target variable
        self.y = pd.Series(
            np.random.normal(0, 0.1, n_samples) + 0.1 * np.sum(self.X.values, axis=1),
            index=self.X.index
        )
    
    def test_tree_ranking_model_initialization(self):
        """Test tree model initialization"""
        # Test LightGBM model
        lgb_model = TreeRankingModel(model_type='lightgbm', num_leaves=31)
        assert lgb_model.model_type == 'lightgbm'
        assert lgb_model.params['num_leaves'] == 31
        assert not lgb_model.is_fitted
        
        # Test XGBoost model
        xgb_model = TreeRankingModel(model_type='xgboost', max_depth=6)
        assert xgb_model.model_type == 'xgboost'
        assert xgb_model.params['max_depth'] == 6
        
        # Unsupported model type
        with pytest.raises(ValueError, match="(?i)unsupported"):
            TreeRankingModel(model_type='invalid')
    
    def test_tree_ranking_model_fit(self):
        """Test tree model training"""
        model = TreeRankingModel(model_type='lightgbm', num_leaves=31)
        
        # Fit model
        fitted_model = model.fit(self.X, self.y)
        
        # Check model state
        assert fitted_model.is_fitted
        assert fitted_model.feature_names == list(self.X.columns)
        assert fitted_model.model is not None
        
        # Should return self
        assert fitted_model is model
    
    def test_tree_ranking_model_fit_with_missing_data(self):
        """Test tree model training with missing data"""
        # Add missing values
        X_with_missing = self.X.copy()
        X_with_missing.iloc[0, 0] = np.nan
        
        y_with_missing = self.y.copy()
        y_with_missing.iloc[1] = np.nan
        
        model = TreeRankingModel(model_type='lightgbm')
        fitted_model = model.fit(X_with_missing, y_with_missing)
        
        # Model should fit successfully
        assert fitted_model.is_fitted
    
    def test_tree_ranking_model_predict(self):
        """Test tree model prediction"""
        model = TreeRankingModel(model_type='lightgbm')
        model.fit(self.X, self.y)
        
        # Predict
        predictions = model.predict(self.X)
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.X)
        assert not np.any(np.isnan(predictions))
        
        # Predict with untrained model
        untrained_model = TreeRankingModel(model_type='lightgbm')
        with pytest.raises(ValueError, match="(?i)not\s*trained"):
            untrained_model.predict(self.X)
    
    def test_tree_ranking_model_feature_importance(self):
        """Test tree model feature importance"""
        model = TreeRankingModel(model_type='lightgbm')
        model.fit(self.X, self.y)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Check feature importance
        assert isinstance(feature_importance, pd.Series)
        assert len(feature_importance) == len(self.X.columns)
        assert all(feature_importance >= 0)
        assert feature_importance.index.equals(pd.Index(self.X.columns))
    
    def test_tree_ranking_model_save_load(self):
        """Test saving and loading tree model"""
        model = TreeRankingModel(model_type='lightgbm', num_leaves=31)
        model.fit(self.X, self.y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            model.save_model(model_path)
            
            # Load model
            loaded_model = TreeRankingModel.load_model(model_path)
            
            # Check loaded model
            assert loaded_model.is_fitted
            assert loaded_model.model_type == model.model_type
            assert loaded_model.params['num_leaves'] == model.params['num_leaves']
            assert loaded_model.feature_names == model.feature_names
            
            # Predictions should match
            original_predictions = model.predict(self.X)
            loaded_predictions = loaded_model.predict(self.X)
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
            
        finally:
            # Cleanup temp file
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestModelEvaluation:
    """Test model evaluation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        n_samples = 200
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        self.y = pd.Series(
            np.random.normal(0, 0.1, n_samples) + 0.1 * np.sum(self.X.values, axis=1),
            index=self.X.index
        )
    
    def test_evaluate_model_performance_walk_forward(self):
        """Test walk-forward validation"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        performance = evaluate_model_performance(
            model, self.X, self.y, validation_method='walk_forward'
        )
        
        # Check performance keys
        assert isinstance(performance, dict)
        assert 'mse' in performance
        assert 'rmse' in performance
        assert 'mae' in performance
        assert 'r2' in performance
        assert 'mean_ic' in performance
        assert 'ir_ratio' in performance
        
        # Sanity checks
        assert performance['mse'] >= 0
        assert performance['rmse'] >= 0
        assert performance['mae'] >= 0
        assert -1 <= performance['r2'] <= 1
    
    def test_evaluate_model_performance_cross_validation(self):
        """Test time-series cross-validation"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        performance = evaluate_model_performance(
            model, self.X, self.y, validation_method='cross_validation'
        )
        
        # Check performance keys
        assert isinstance(performance, dict)
        assert 'mse' in performance
        assert 'rmse' in performance
        assert 'mae' in performance
        assert 'r2' in performance
    
    def test_evaluate_model_performance_simple(self):
        """Test simple validation"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        performance = evaluate_model_performance(
            model, self.X, self.y, validation_method='simple'
        )
        
        # Check performance keys
        assert isinstance(performance, dict)
        assert 'mse' in performance
        assert 'rmse' in performance
        assert 'mae' in performance
        assert 'r2' in performance
    
    def test_evaluate_model_performance_invalid_input(self):
        """Test invalid input"""
        model = LinearRankingModel(alpha=0.01, model_type='ridge')
        model.fit(self.X, self.y)
        
        # Empty data
        with pytest.raises(ValueError, match="(?i)empty"):
            evaluate_model_performance(model, pd.DataFrame(), self.y)
        
        with pytest.raises(ValueError, match="(?i)empty"):
            evaluate_model_performance(model, self.X, pd.Series())


class TestPerformanceMetrics:
    """Test performance metrics"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic predictions and actuals
        n_samples = 100
        
        self.predictions = np.random.normal(0, 1, n_samples)
        self.actuals = self.predictions + np.random.normal(0, 0.1, n_samples)  # high correlation
        self.ic_scores = [0.1, 0.2, 0.15, 0.18, 0.12]
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        performance = calculate_performance_metrics(
            self.predictions, self.actuals, self.ic_scores
        )
        
        # Check metrics
        assert isinstance(performance, dict)
        assert 'mse' in performance
        assert 'rmse' in performance
        assert 'mae' in performance
        assert 'r2' in performance
        assert 'mean_ic' in performance
        assert 'std_ic' in performance
        assert 'ir_ratio' in performance
        assert 'hit_rate' in performance
        assert 'ranking_metrics' in performance
        
        # Sanity checks
        assert performance['mse'] >= 0
        assert performance['rmse'] >= 0
        assert performance['mae'] >= 0
        assert performance['r2'] > 0
        assert not pd.isna(performance['mean_ic'])
        assert not pd.isna(performance['ir_ratio'])
    
    def test_calculate_performance_metrics_with_nan(self):
        """Test metrics calculation with NaNs"""
        # Add NaNs
        predictions_with_nan = self.predictions.copy()
        predictions_with_nan[0] = np.nan
        
        actuals_with_nan = self.actuals.copy()
        actuals_with_nan[1] = np.nan
        
        performance = calculate_performance_metrics(
            predictions_with_nan, actuals_with_nan, self.ic_scores
        )
        
        # Check NaNs handled
        assert isinstance(performance, dict)
        assert 'mse' in performance
    
    def test_calculate_performance_metrics_empty_data(self):
        """Test metrics calculation with empty data"""
        performance = calculate_performance_metrics(
            np.array([]), np.array([]), []
        )
        
        assert performance == {}
    
    def test_calculate_ranking_metrics(self):
        """Test ranking metrics calculation"""
        ranking_metrics = calculate_ranking_metrics(self.predictions, self.actuals)
        
        # Check ranking metrics
        assert isinstance(ranking_metrics, dict)
        assert 'spearman_correlation' in ranking_metrics
        assert 'top_5_hit_rate' in ranking_metrics
        assert 'top_10_hit_rate' in ranking_metrics
        assert 'top_20_hit_rate' in ranking_metrics
        
        # Sanity checks
        assert -1 <= ranking_metrics['spearman_correlation'] <= 1
        assert 0 <= ranking_metrics['top_5_hit_rate'] <= 1
        assert 0 <= ranking_metrics['top_10_hit_rate'] <= 1
        assert 0 <= ranking_metrics['top_20_hit_rate'] <= 1


class TestModelComparison:
    """Test model comparison"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        n_samples = 100
        n_features = 10
        
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        self.y = pd.Series(
            np.random.normal(0, 0.1, n_samples) + 0.1 * np.sum(self.X.values, axis=1),
            index=self.X.index
        )
        
        # Create multiple models
        self.models = {
            'ridge': LinearRankingModel(alpha=0.01, model_type='ridge'),
            'lasso': LinearRankingModel(alpha=0.01, model_type='lasso'),
            'elastic_net': LinearRankingModel(alpha=0.01, l1_ratio=0.5, model_type='elastic_net')
        }
        
        # Fit models
        for model in self.models.values():
            model.fit(self.X, self.y)
    
    def test_compare_models(self):
        """Test model comparison"""
        comparison_df = compare_models(
            self.models, self.X, self.y, validation_method='simple'
        )
        
        # Check comparison result
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(self.models)
        assert 'ridge' in comparison_df.index
        assert 'lasso' in comparison_df.index
        assert 'elastic_net' in comparison_df.index
        
        # Check metric columns
        expected_columns = ['mse', 'rmse', 'mae', 'r2', 'mean_ic', 'ir_ratio']
        for col in expected_columns:
            if col in comparison_df.columns:
                assert not comparison_df[col].isnull().all()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_model_comparison(self, mock_savefig, mock_show):
        """Test model comparison plotting"""
        comparison_df = compare_models(
            self.models, self.X, self.y, validation_method='simple'
        )
        
        # Plot without saving
        plot_model_comparison(comparison_df)
        mock_show.assert_called()
        
        # Plot and save image
        plot_model_comparison(comparison_df, save_path='test_comparison.png')
        mock_savefig.assert_called()


class TestModelSaving:
    """Test model saving utilities"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic data
        n_samples = 50
        n_features = 5
        
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        self.y = pd.Series(
            np.random.normal(0, 0.1, n_samples) + 0.1 * np.sum(self.X.values, axis=1),
            index=self.X.index
        )
        
        # Create model
        self.model = LinearRankingModel(alpha=0.01, model_type='ridge')
        self.model.fit(self.X, self.y)
        
        # Create performance dict
        self.performance = {
            'mse': 0.1,
            'rmse': 0.316,
            'mae': 0.25,
            'r2': 0.8,
            'mean_ic': 0.05,
            'ir_ratio': 0.5
        }
    
    @patch('part2_alpha_modeling.task6_models.ensure_directory')
    def test_save_model_results(self, mock_ensure_dir):
        """Test saving model results"""
        mock_ensure_dir.return_value = Path("test_results")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model_results(
                self.model, 
                self.performance, 
                'test_model', 
                save_dir=temp_dir
            )
            
            # Files should be created
            model_file = Path(temp_dir) / "test_model.pkl"
            performance_file = Path(temp_dir) / "test_model_performance.json"
            importance_file = Path(temp_dir) / "test_model_feature_importance.csv"
            
            assert model_file.exists()
            assert performance_file.exists()
            assert importance_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])

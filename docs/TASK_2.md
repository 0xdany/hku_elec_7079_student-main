# TASK 2: Signal Prediction & Alpha Modeling - Student Implementation

## 🎯 Learning Objectives

Through this task, students will master:
- Alpha factor engineering methodologies and techniques
- Information Coefficient (IC) analysis theory and practice
- Machine learning applications in quantitative finance
- Model validation and performance evaluation methods
- Feature engineering and data preprocessing techniques

## 📚 Task Overview

**Part 2** contains 3 subtasks with medium to high difficulty, designed to develop students' feature engineering and machine learning modeling capabilities:

### Task 2.1: Alpha Factor Engineering
- **File Location**: `src/part2_alpha_modeling/task4_factors.py`
- **Core Objective**: Build predictive features (Alpha factors) from raw data

### Task 2.2: Information Coefficient (IC) Analysis
- **File Location**: `src/part2_alpha_modeling/task5_ic_analysis.py`
- **Core Objective**: Evaluate Alpha factors' predictive power

### Task 2.3: Machine Learning Model Development
- **File Location**: `src/part2_alpha_modeling/task6_models.py`
- **Core Objective**: Build stock ranking prediction models

## 🔧 Core Functions to Implement

### Task 2.1: Alpha Factor Engineering

#### `calculate_momentum_factors()` - Momentum Factor Calculation
```python
def calculate_momentum_factors(price_series: pd.Series, periods: List[int] = [3, 9, 18], method: str = 'pct_change') -> pd.DataFrame:
    """
    Calculate momentum factors for a single stock
    
    Implementation Points:
    1. Calculate price changes over different time windows
    2. Support both percentage change and log return methods
    3. Handle missing values with forward fill
    4. Apply outlier treatment with 3-sigma truncation
    5. Z-score standardization
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_mean_reversion_factors()` - Mean Reversion Factors
```python
def calculate_mean_reversion_factors(price_series: pd.Series, ma_periods: List[int] = [12, 24, 48]) -> pd.DataFrame:
    """
    Calculate mean reversion factors for a single stock
    
    Implementation Points:
    1. Calculate moving averages for different periods
    2. Compute price deviation from moving averages: (price - ma) / ma
    3. Handle missing values and outliers
    4. Standardize the results
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_volume_factors()` - Volume Factors
```python
def calculate_volume_factors(volume_series: pd.Series, lookback_periods: List[int] = [12, 24, 48]) -> pd.DataFrame:
    """
    Calculate volume factors for a single stock
    
    Implementation Points:
    1. Calculate volume ratios: current_volume / historical_average_volume
    2. Calculate volume changes: (current_volume - historical_average) / historical_average
    3. Handle zero volume and division by zero cases
    4. Apply outlier treatment and standardization
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_intraday_factors()` - Intraday Features
```python
def calculate_intraday_factors(price_series: pd.Series, vwap_series: pd.Series, open_series: pd.Series) -> pd.DataFrame:
    """
    Calculate intraday features for a single stock
    
    Implementation Points:
    1. Open-to-close returns: (close - open) / open
    2. VWAP deviation: (close - vwap) / vwap
    3. Intraday time features: time elapsed since market open
    4. Standardize all features
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `create_factor_dataset()` - Factor Dataset Creation
```python
def create_factor_dataset(price_data: pd.DataFrame, volume_data: pd.DataFrame, vwap_data: pd.DataFrame, open_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive factor dataset
    
    Implementation Points:
    1. Calculate all factor types for each stock
    2. Consolidate all factors into unified DataFrame
    3. Perform quality checks: missing values, outliers, distribution analysis
    4. Save results to results/part2/ directory
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

### Task 2.2: IC Analysis

#### `calculate_information_coefficient()` - Single Factor IC Calculation
```python
def calculate_information_coefficient(factor_series: pd.Series, forward_returns: pd.Series, method: str = 'cross_sectional') -> Dict[str, Any]:
    """
    Calculate information coefficient for a single factor
    
    Implementation Points:
    1. Data alignment: ensure consistent time indices
    2. Correlation calculation: use scipy.stats.pearsonr()
    3. Statistical significance: calculate t-statistic and p-value
    4. Return comprehensive analysis results dictionary
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_ic_for_multiple_factors()` - Multi-Factor IC Analysis
```python
def calculate_ic_for_multiple_factors(factor_data: pd.DataFrame, forward_returns: pd.DataFrame, method: str = 'cross_sectional') -> Dict[str, Any]:
    """
    Calculate information coefficients for multiple factors
    
    Supports two methods:
    1. Time series IC: time series correlation for each factor column
    2. Cross-sectional IC: cross-sectional correlation at each time point
    
    Implementation Points:
    1. Choose calculation method based on method parameter
    2. Calculate IC mean, standard deviation, IR ratio, hit rate
    3. Return comprehensive analysis results
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `analyze_ic_stability()` - IC Stability Analysis
```python
def analyze_ic_stability(ic_series: pd.Series, window: int = 60) -> Dict[str, Any]:
    """
    Analyze IC stability characteristics
    
    Implementation Points:
    1. Calculate rolling IC statistics: mean, std, IR ratio
    2. IC decay analysis: autocorrelation and half-life
    3. IC distribution characteristics: skewness, kurtosis, quantiles
    4. Return complete stability analysis report
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

### Task 2.3: Machine Learning Models

#### `LinearRankingModel` Class - Linear Ranking Model
```python
class LinearRankingModel:
    """
    Linear ranking model class
    
    Methods to implement:
    1. __init__(): Initialize model parameters
    2. fit(): Train the model
    3. predict(): Make predictions
    4. get_feature_importance(): Get feature importance
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRankingModel':
        """
        Train linear model
        
        Implementation Points:
        1. Data preprocessing: missing value handling, standardization
        2. Model training: support Ridge, Lasso, ElasticNet
        3. Feature importance calculation
        4. Model validation and performance evaluation
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        pass
```

#### `TreeRankingModel` Class - Tree-based Model
```python
class TreeRankingModel:
    """
    Tree-based ranking model class
    
    Methods to implement:
    1. __init__(): Support LightGBM and XGBoost
    2. fit(): Train tree model
    3. predict(): Make predictions
    4. get_feature_importance(): Feature importance
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TreeRankingModel':
        """
        Train tree model
        
        Implementation Points:
        1. Parameter setting: learning rate, tree depth, regularization
        2. Early stopping mechanism: prevent overfitting
        3. Feature importance analysis
        4. Cross-validation
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        pass
```

#### `evaluate_model_performance()` - Model Evaluation
```python
def evaluate_model_performance(model, X: pd.DataFrame, y: pd.Series, validation_method: str = 'walk_forward', **kwargs) -> Dict[str, Any]:
    """
    Model performance evaluation
    
    Implementation Points:
    1. Validation method selection: Walk-Forward, time series cross-validation
    2. Performance metrics calculation: MSE, RMSE, MAE, R², IC, IR
    3. Ranking metrics: Spearman correlation, Top-K hit rate
    4. Return comprehensive evaluation report
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

## 📊 Data Flow and Dependencies

```
Raw Data (Part 1)
    ↓
Factor Engineering (Task 2.1)
    ↓
Factor Dataset + Forward Returns
    ↓
IC Analysis (Task 2.2) → Factor Screening and Ranking
    ↓
Machine Learning Modeling (Task 2.3)
    ↓
Trained Prediction Models → Part 3 Strategy Development
```

## 🧪 Testing and Validation

### Running Tests
```bash
# Run all Part 2 tests
python -m pytest tests/test_part2/ -v

# Run specific task tests
python -m pytest tests/test_part2/test_task4.py -v  # Factor engineering
python -m pytest tests/test_part2/test_task5.py -v  # IC analysis
python -m pytest tests/test_part2/test_task6.py -v  # Model development

# Run simple verification
python test_part2_simple.py
```

### Validation Checkpoints

#### Task 2.1: Factor Engineering
- [ ] Factor dataset shape is reasonable (time × number of factors)
- [ ] All factors are standardized (mean ≈ 0, std ≈ 1)
- [ ] Missing value ratio is acceptable (<20%)
- [ ] Factor distributions are reasonable (skewness and kurtosis)

#### Task 2.2: IC Analysis
- [ ] IC values are within reasonable range (-1 to 1)
- [ ] Reasonable number of effective factors (|IC| > 0.02)
- [ ] IR ratios are positive and stable
- [ ] IC time series show no obvious trends (stationarity)

#### Task 2.3: Model Development
- [ ] Model training completes successfully
- [ ] Predictions show positive correlation with actual returns
- [ ] Model performance metrics are within reasonable range
- [ ] Feature importance analysis is meaningful

## 🎨 Expected Visualizations

Students should implement visualization functions for:

1. **Factor Distribution Plots**: Show distribution characteristics of various factors
2. **IC Time Series Plots**: Display IC changes over time
3. **Factor IC Ranking Chart**: Horizontal bar chart showing factor effectiveness
4. **Model Performance Comparison**: Performance comparison across different models
5. **Feature Importance Plot**: Importance of various factors in models

## 🚀 Getting Started

### 1. Data Preparation
```python
# Ensure Part 1 is completed with forward returns data
from src.part1_data_analysis.task1_returns import calculate_forward_returns
from src.data_loader import DataLoader

loader = DataLoader()
data_5min = loader.load_5min_data()
forward_returns = calculate_forward_returns(data_5min, forward_periods=12)
```

### 2. Implementation Steps

#### Step 1: Factor Engineering
```python
# Start implementing individual factor functions
momentum_factors = calculate_momentum_factors(stock_price_series)
print(f"Momentum factors shape: {momentum_factors.shape}")
```

#### Step 2: IC Analysis
```python
# Calculate IC for individual factors
ic_result = calculate_information_coefficient(factor_series, return_series)
print(f"IC value: {ic_result['ic_value']:.4f}")
```

#### Step 3: Model Development
```python
# Start with simple linear model
model = LinearRankingModel(alpha=0.01, model_type='ridge')
model.fit(factor_data, forward_returns)
```

## 📝 Implementation Tips

### Common Issues and Solutions

1. **Memory Management**
```python
# Handle large datasets with batch processing
for stock_batch in np.array_split(stock_list, n_batches):
    batch_factors = calculate_factors_batch(stock_batch)
```

2. **Data Alignment**
```python
# Ensure factor and return data time alignment
common_index = factor_data.index.intersection(return_data.index)
factor_aligned = factor_data.loc[common_index]
return_aligned = return_data.loc[common_index]
```

3. **Model Overfitting Prevention**
```python
# Use time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

4. **Performance Optimization**
```python
# Use vectorized operations
# Avoid: manual loops for calculations
# Use: pandas/numpy vectorized functions
rolling_means = data.rolling(window).mean()  # vectorized
```

## 🎯 Assessment Criteria

### Code Quality (40%)
- [ ] Complete and correct function implementation
- [ ] Clear code structure with comprehensive comments
- [ ] Error handling and edge case management
- [ ] Pass all unit tests

### Factor Quality (30%)
- [ ] Reasonable factor engineering methods
- [ ] In-depth factor effectiveness analysis
- [ ] Credible IC analysis results
- [ ] Factor interpretation and economic intuition

### Model Performance (30%)
- [ ] Reasonable model selection and parameter tuning
- [ ] Correct validation methodology
- [ ] Accurate performance metric calculations
- [ ] Result interpretation and improvement suggestions

## 🔗 Related Resources

- [Alpha Factor Theory Fundamentals](https://en.wikipedia.org/wiki/Alpha_(finance))
- [Information Coefficient Analysis Methods](https://www.investopedia.com/terms/i/informationratio.asp)
- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

Upon completing Task 2, you will have mastered the core skills for developing practical trading strategies!

---
**Estimated Completion Time**: 15-20 hours  
**Difficulty Level**: ⭐⭐⭐⭐☆  
**Prerequisites**: Task 1 completed, machine learning basics, statistical knowledge
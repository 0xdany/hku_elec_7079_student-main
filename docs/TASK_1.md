# TASK 1: Data Analysis & Feature Exploration - Student Implementation

## 🎯 Learning Objectives

Through this task, students will master:
- Fundamentals of financial time series data processing
- Return calculation and statistical analysis
- Volatility analysis and market index construction
- Correlation analysis and data visualization
- Usage of Python libraries: pandas, numpy, matplotlib, etc.

## 📚 Task Overview

**Part 1** contains 3 subtasks with progressive difficulty, designed to help students familiarize with fundamental tools and methods in quantitative analysis:

### Task 1.1: Target Variable Engineering & Return Calculation
- **File Location**: `src/part1_data_analysis/task1_returns.py`
- **Core Objective**: Calculate forward returns and weekly returns for stock prediction models

### Task 1.2: Market & Asset Characteristic Analysis
- **File Location**: `src/part1_data_analysis/task2_volatility.py`
- **Core Objective**: Compute volatility metrics and construct equal-weight market index

### Task 1.3: Cross-sectional Analysis
- **File Location**: `src/part1_data_analysis/task3_correlation.py`
- **Core Objective**: Analyze correlation structure and cross-sectional relationships

## 🔧 Core Functions to Implement

### Task 1.1: Return Calculation

#### `calculate_forward_returns()` - Forward Return Calculation
```python
def calculate_forward_returns(data: pd.DataFrame, forward_periods: int = 12, price_column: str = 'close_px') -> pd.DataFrame:
    """
    Calculate forward returns for each stock at every 5-minute interval.
    
    Implementation Points:
    1. Extract stock symbols and price data from MultiIndex DataFrame
    2. Use shift(-forward_periods) to get future prices
    3. Calculate returns: (future_price / current_price) - 1
    4. Handle missing values and boundary cases
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_weekly_returns()` - Weekly Return Calculation  
```python
def calculate_weekly_returns(daily_data: pd.DataFrame, price_column: str = 'close_px') -> pd.DataFrame:
    """
    Calculate weekly returns based on daily data.
    
    Implementation Points:
    1. Use resample('W') to resample to weekly frequency
    2. Calculate weekly price change rates
    3. Handle missing data and edge cases
    4. Return properly formatted DataFrame
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `plot_return_distribution()` - Return Distribution Analysis
```python
def plot_return_distribution(returns_data: pd.DataFrame, sample_stocks: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """
    Visualize and analyze return distribution characteristics.
    
    Implementation Points:
    1. Select sample stocks for analysis
    2. Create histogram and Q-Q plots
    3. Calculate statistical metrics (mean, std, skew, kurtosis)
    4. Perform normality tests
    5. Save plots and display results
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `analyze_return_properties()` - Comprehensive Return Analysis
```python
def analyze_return_properties(returns_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of return characteristics.
    
    Implementation Points:
    1. Calculate basic statistics for all stocks
    2. Test for normality and other distribution properties
    3. Identify outliers and extreme values
    4. Generate summary statistics and insights
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

### Task 1.2: Volatility Analysis

#### `calculate_rolling_volatility()` - Rolling Volatility Calculation
```python
def calculate_rolling_volatility(returns_data: pd.DataFrame, window: int = 60, annualize: bool = True) -> pd.DataFrame:
    """
    Calculate rolling volatility for each stock.
    
    Implementation Points:
    1. Use rolling window to calculate standard deviation
    2. Annualize volatility if requested (multiply by sqrt(periods))
    3. Handle missing values appropriately
    4. Return DataFrame with same structure as input
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `build_equal_weight_index()` - Equal Weight Index Construction
```python
def build_equal_weight_index(returns_data: pd.DataFrame) -> pd.Series:
    """
    Construct equal-weight market index from stock returns.
    
    Implementation Points:
    1. Calculate equal weights for all stocks (1/N)
    2. Compute weighted average returns across stocks
    3. Handle missing data by adjusting weights
    4. Return time series of index returns
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `plot_volatility_analysis()` - Comprehensive Volatility Visualization
```python
def plot_volatility_analysis(daily_returns: pd.DataFrame, equal_weight_index: pd.Series, sample_stocks: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """
    Create comprehensive volatility analysis plots.
    
    Implementation Points:
    1. Plot rolling volatility time series
    2. Compare individual stock vs index volatility
    3. Create volatility distribution histograms
    4. Add statistical annotations and insights
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

### Task 1.3: Correlation Analysis

#### `calculate_correlation_matrix()` - Correlation Matrix Computation
```python
def calculate_correlation_matrix(daily_returns: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate pairwise correlation matrix of stock returns.
    
    Implementation Points:
    1. Use pandas corr() function with specified method
    2. Handle missing data appropriately (pairwise deletion)
    3. Ensure symmetric matrix with diagonal = 1.0
    4. Support different correlation methods (pearson, spearman, kendall)
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `calculate_rolling_correlation()` - Rolling Correlation Analysis
```python
def calculate_rolling_correlation(stock1_returns: pd.Series, stock2_returns: pd.Series, window: int = 60, min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate rolling correlation between two stocks.
    
    Implementation Points:
    1. Align two return series by index
    2. Use pandas rolling correlation function
    3. Set appropriate min_periods parameter
    4. Return time series of correlation coefficients
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

#### `plot_correlation_heatmap()` - Correlation Heatmap Visualization
```python
def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str = "Stock Returns Correlation Matrix", cmap: str = 'RdBu_r', figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None) -> None:
    """
    Create correlation matrix heatmap visualization.
    
    Implementation Points:
    1. Use seaborn heatmap function
    2. Apply appropriate color scheme and formatting
    3. Add title and axis labels
    4. Handle large matrices with proper scaling
    """
    # TODO: STUDENT IMPLEMENTATION REQUIRED
    pass
```

## 📊 Data Flow and Dependencies

```
Raw K-bar Data (Part 1)
    ↓
Forward Returns (Task 1.1) → Target Variables for Part 2
    ↓
Volatility Analysis (Task 1.2) → Market Understanding
    ↓
Correlation Analysis (Task 1.3) → Cross-sectional Structure
    ↓
Foundation for Factor Engineering (Part 2)
```

## 🧪 Testing and Validation

### Running Tests
```bash
# Run all Part 1 tests
python -m pytest tests/test_part1/ -v

# Run specific task tests
python -m pytest tests/test_part1/test_task1.py -v  # Return calculation
python -m pytest tests/test_part1/test_task2.py -v  # Volatility analysis
python -m pytest tests/test_part1/test_task3.py -v  # Correlation analysis

# Run simple verification
python test_part1_simple.py
```

### Validation Checkpoints

#### Task 1.1: Return Calculation
- [ ] Forward returns shape matches input data dimensions
- [ ] Weekly returns properly aggregated to weekly frequency
- [ ] No future data leakage in calculations
- [ ] Missing values handled appropriately

#### Task 1.2: Volatility Analysis
- [ ] Rolling volatility calculated correctly
- [ ] Equal-weight index properly constructed
- [ ] Volatility statistics make financial sense
- [ ] Visualizations are clear and informative

#### Task 1.3: Correlation Analysis
- [ ] Correlation matrix is symmetric with diagonal = 1
- [ ] Rolling correlations show time-varying patterns
- [ ] Heatmap visualization is readable
- [ ] Statistical analysis provides insights

## 🎨 Expected Visualizations

Students should produce the following plots:

1. **Return Distribution Plots**: Histograms and Q-Q plots for sample stocks
2. **Volatility Time Series**: Rolling volatility over time
3. **Volatility Comparison**: Individual stocks vs market index
4. **Correlation Heatmap**: Pairwise correlation matrix
5. **Rolling Correlation**: Time-varying correlation patterns

## 🚀 Getting Started

### 1. Data Preparation
```python
# Load data using the data loader
from src.data_loader import DataLoader

loader = DataLoader()
data_5min = loader.load_5min_data()
data_daily = loader.load_daily_data()
```

### 2. Implementation Approach

#### Step 1: Start with Return Calculation
```python
# Begin with the forward returns function
forward_returns = calculate_forward_returns(data_5min, forward_periods=12)
print(f"Forward returns shape: {forward_returns.shape}")
```

#### Step 2: Volatility Analysis
```python
# Calculate daily returns first
daily_returns = data_daily.pct_change().dropna()
rolling_vol = calculate_rolling_volatility(daily_returns, window=60)
```

#### Step 3: Correlation Analysis
```python
# Calculate correlation matrix
corr_matrix = calculate_correlation_matrix(daily_returns)
print(f"Correlation matrix shape: {corr_matrix.shape}")
```

## 📝 Implementation Tips

### Common Issues and Solutions

1. **MultiIndex Data Handling**
```python
# Access specific stock data
stock_data = data_5min.xs('STOCK_1', level=0, axis=1)
# Or access specific field
close_prices = data_5min.xs('close_px', level=1, axis=1)
```

2. **Missing Data Treatment**
```python
# Forward fill for price data
prices_filled = prices.fillna(method='ffill')
# Drop NaN for returns
returns_clean = returns.dropna()
```

3. **Performance Optimization**
```python
# Use vectorized operations
# Avoid: for loops over DataFrame rows
# Use: pandas built-in functions and numpy operations
```

4. **Memory Management**
```python
# For large datasets, process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

## 🎯 Assessment Criteria

### Code Quality (40%)
- [ ] Functions implemented completely and correctly
- [ ] Code structure is clear with appropriate comments
- [ ] Error handling and edge case management
- [ ] Passes all unit tests

### Analysis Quality (35%)
- [ ] Correct understanding of financial data characteristics
- [ ] Appropriate statistical analysis methods
- [ ] Clear and informative visualizations
- [ ] Meaningful interpretation of results

### Technical Skills (25%)
- [ ] Efficient use of pandas and numpy
- [ ] Proper handling of time series data
- [ ] Professional-quality plots and charts
- [ ] Code optimization and best practices

## 🔗 Related Resources

- [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Financial Data Analysis with Python](https://www.quantstart.com/)
- [Volatility Modeling Concepts](https://en.wikipedia.org/wiki/Volatility_(finance))
- [Correlation Analysis in Finance](https://www.investopedia.com/terms/c/correlation.asp)

Good luck with your implementation! This foundational analysis will prepare you for advanced factor engineering and strategy development in the following parts.

---
**Estimated Completion Time**: 8-12 hours  
**Difficulty Level**: ⭐⭐☆☆☆  
**Prerequisites**: Python basics, pandas fundamentals, basic statistics
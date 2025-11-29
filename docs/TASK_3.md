# Part 3 Student Tasks - Detailed Implementation Guide

## I. Task Overview

Part 3 requires students to build a complete quantitative trading strategy system, including:
1. **Strategy Construction & Backtesting** - Implement a complete backtesting engine for long-short strategies
2. **Performance Evaluation** - Calculate strategy performance metrics and risk measures
3. **Strategy Analysis & Improvement** - Conduct in-depth analysis of strategy performance and propose optimization recommendations

This is the final component of the course project, requiring integration of results from Part 1 (Data Analysis) and Part 2 (Alpha Modeling).

---

## II. Core Functions/Classes to Implement

Based on code file analysis, students must complete **9 core functions/classes**:

### 📌 Task 3.1: Strategy Construction & Backtesting (`task7_backtest.py`)

#### 1. `_BollingerSingleAsset.update()` - Bollinger Bands Single Asset Implementation

**Current Status**: Empty implementation, raises `NotImplementedError`

**Required Implementation Logic**:
```python
def update(self, price: float, volume: Optional[float] = None) -> float:
    """
    Update Bollinger Bands strategy state and generate trading signals at each time point
    
    Implementation Points:
    1. Maintain price history: self.prices.append(float(price))
    2. Check sufficient history: if len(self.prices) < self.window: return 0.0
    3. Compute statistics:
       - Moving average: ma = window_prices.mean()
       - Standard deviation: vol = window_prices.std(ddof=0)
       - Upper band: upper = ma + self.num_std * vol
       - Lower band: lower = ma - self.num_std * vol
    4. Detect breakout signals:
       - Buy signal: price crosses from below lower band to above
       - Sell signal: price crosses from above upper band to below
    5. Update state flags: prev_below_lower, prev_above_upper
    6. Return signal: +1.0(buy), -1.0(sell), 0.0(hold)
    """
```

**Key Challenges**:
- Correctly maintain rolling window of price history
- Must avoid look-ahead bias (only use current and prior data)
- Accurately detect price crossings with Bollinger Bands

---

#### 2. `_MACDSingleAsset.update()` - MACD Single Asset Implementation

**Current Status**: Empty implementation, raises `NotImplementedError`

**Required Implementation Logic**:
```python
def update(self, price: float, volume: Optional[float] = None) -> float:
    """
    Update MACD strategy state
    
    Implementation Points:
    1. Update EMAs:
       - self.ema_fast = self._ema(self.ema_fast, price, self.fast)
       - self.ema_slow = self._ema(self.ema_slow, price, self.slow)
    2. Calculate MACD line: macd = self.ema_fast - self.ema_slow
    3. Update signal line: self.macd_sig = self._ema(self.macd_sig, macd, self.sig)
    4. Detect crossovers:
       - Bullish crossover (buy): MACD crosses above signal line
       - Bearish crossover (sell): MACD crosses below signal line
    5. Update state flags
    6. Return signal: +1.0(buy), -1.0(sell), 0.0(hold)
    """
```

**Key Challenges**:
- Correctly implement incremental EMA calculation (helper method `_ema` already provided)
- Accurately detect crossovers between MACD line and signal line

---

#### 3. `LongShortStrategy.backtest()` - Long-Short Strategy Backtesting Engine

**Current Status**: Empty implementation - **This is the most critical and complex function**

**Complete Backtesting Flow to Implement**:

```python
def backtest(self, returns, prices, predictions) -> Dict[str, Any]:
    """
    Execute complete strategy backtesting workflow
    
    Core Implementation Steps:
    
    1. Data Preprocessing:
       - Extract close price matrix (if MultiIndex format)
       - Compute returns (if not provided)
       - Get stock list and time index
    
    2. Initialization:
       - Initialize technical indicator strategy instances (if used)
       - Create containers: weights history, turnover, transaction costs, portfolio returns, etc.
       - Set initial state: current_weights = 0, pending_weights = 0
    
    3. Main Loop (iterate over time points):
       for t in time_index:
           a) Calculate current period returns
           b) Execute pending weight changes:
              - delta = pending_weights - current_weights
              - turnover = abs(delta).sum()
              - transaction_cost = turnover * cost_rate
              - Record trade log
           c) Calculate portfolio return:
              - gross_return = (current_weights * returns[t]).sum()
              - net_return = gross_return - transaction_cost
           d) Generate next period signals:
              - Technical strategies: iterate over stocks, call update() to get signals
              - Prediction strategies: use provided predictions
              - Determine rebalancing based on rebalance_periods
           e) Update state variables
    
    4. Post-processing:
       - nav = (1 + returns).cumprod()
       - capital_used = gross_exposure * nav
       - Organize trade log
    """
```

**Return Result Structure**:
```python
{
    "returns": pd.Series,           # Portfolio returns time series
    "nav": pd.Series,                # Net asset value curve
    "weights": pd.DataFrame,         # Weights history [time x stocks]
    "turnover": pd.Series,           # Turnover time series
    "transaction_costs": pd.Series,  # Transaction costs time series
    "gross_exposure": pd.Series,     # Gross exposure time series
    "capital_used": pd.Series,       # Capital usage
    "trade_log": List[Dict],         # Trade records
}
```

**Key Challenges**:
- Correctly construct long-short positions (based on quantile stock selection)
- Accurately calculate turnover and transaction costs
- Avoid look-ahead bias
- Handle suspended trading and abnormal data
- Memory optimization (handling large-scale data)

---

### 📌 Task 3.2: Performance Evaluation (`task8_performance.py`)

#### 4. `calculate_performance_metrics()` - Performance Metrics Calculation

**Current Status**: Empty implementation

**Required Metrics**:

```python
def calculate_performance_metrics(returns_series: pd.Series) -> Dict[str, float]:
    """
    Calculate key performance metrics
    
    Required Metrics:
    
    1. Return Metrics:
       - total_return: Total return = (1 + r).prod() - 1
       - annualized_return: Annualized return = (1 + r.mean())^252 - 1
    
    2. Risk Metrics:
       - annualized_volatility: Annualized volatility = r.std() * sqrt(252)
       - max_drawdown: Maximum drawdown = max((peak - current) / peak)
    
    3. Risk-Adjusted Returns:
       - sharpe_ratio: Sharpe ratio = annualized return / annualized volatility
       - sortino_ratio: Sortino ratio (downside volatility)
    
    4. Other Metrics:
       - win_rate: Win rate = positive return periods / total periods
       - profit_loss_ratio: Average profit / average loss
    """
```

**Key Formulas**:
- **Annualized Return**: `(1 + mean_return) ** 252 - 1`
- **Annualized Volatility**: `return_std * sqrt(252)`
- **Maximum Drawdown**: `max((cumulative_peak - current_cumulative) / cumulative_peak)`
- **Sharpe Ratio**: `annualized_return / annualized_volatility`

---

#### 5. `compare_with_benchmarks()` - Benchmark Comparison

**Current Status**: Empty implementation

**Required Comparison Metrics**:

```python
def compare_with_benchmarks(strategy_returns, benchmark_returns) -> Dict[str, float]:
    """
    Strategy vs benchmark comparison
    
    Metrics to Calculate:
    
    1. excess_return: Excess return = strategy - benchmark
    2. information_ratio: Information ratio = annualized excess return / tracking error
    3. tracking_error: Tracking error = excess.std() * sqrt(252)
    4. beta: Beta coefficient = Cov(strategy, benchmark) / Var(benchmark)
    5. correlation: Correlation coefficient
    """
```

---

#### 6. `generate_performance_report()` - Performance Report Generation

**Current Status**: Empty implementation

**Required Functionality**:
```python
def generate_performance_report(strategy_results) -> Dict[str, Any]:
    """
    Generate comprehensive performance report
    
    Implementation Steps:
    1. Extract data from backtest results (returns, nav, weights, etc.)
    2. Call calculate_performance_metrics()
    3. Build report structure:
       - metrics: All performance metrics
       - final_nav: Final net asset value
       - num_periods: Number of periods
       - summary: Text summary
    4. Optional: Generate HTML format report
    """
```

---

### 📌 Task 3.3: Strategy Analysis & Improvement (`task9_analysis.py`)

#### 7. `analyze_strategy_performance()` - Strategy Performance Analysis

**Current Status**: Empty implementation

**Required Analysis Dimensions**:

```python
def analyze_strategy_performance(strategy_results) -> Dict[str, Any]:
    """
    In-depth strategy performance analysis
    
    Analysis Dimensions:
    
    1. Drawdown Analysis:
       - Identify all drawdown periods
       - Calculate drawdown depth, duration, recovery time
       - Analyze market environment during drawdown periods
    
    2. Return Decomposition:
       - Long contribution vs short contribution
       - Timing returns vs stock selection returns
       - Performance across different periods (monthly/quarterly)
    
    3. Trading Analysis:
       - Turnover statistics
       - Transaction cost impact
       - Position concentration
    
    4. Time Series Analysis:
       - Consecutive profit/loss streaks
       - Strategy effectiveness decay
       - Rolling performance metrics
    """
```

---

#### 8. `identify_drawdown_periods()` - Drawdown Period Identification

**Required Implementation**: Identify and analyze drawdown periods

```python
def identify_drawdown_periods(nav_series, threshold=0.05) -> List[Dict]:
    """
    Identify drawdown periods
    
    Return Format:
    [
        {
            'start_date': Drawdown start date,
            'end_date': Drawdown end date,
            'max_drawdown': Maximum drawdown magnitude,
            'duration_days': Duration in days,
            'recovery_days': Recovery time in days
        },
        ...
    ]
    """
```

---

#### 9. `generate_improvement_proposals()` - Improvement Recommendations Generation

**Current Status**: Empty implementation

**Required Logic**:

```python
def generate_improvement_proposals(analysis_results) -> Dict[str, List[str]]:
    """
    Generate improvement recommendations based on analysis results
    
    Recommendation Categories:
    
    1. risk_management: Risk management recommendations
       - If max drawdown > 20%: Add stop-loss mechanism
       - If consecutive losses >= 5 periods: Add market regime filter
       - If high volatility: Use dynamic position sizing
    
    2. return_enhancement: Return enhancement recommendations
       - If low Sharpe ratio: Optimize factor combination
       - If low win rate: Improve entry timing
       - If negative returns: Reassess signal quality
    
    3. cost_optimization: Cost optimization recommendations
       - If high turnover: Optimize rebalancing frequency
       - If high transaction cost ratio: Reduce trading frequency
    
    4. operational_improvement: Operational improvement recommendations
       - Expand stock universe
       - Improve execution algorithms
       - Liquidity management
    """
```

---

## III. Critical Technical Points

### 🔴 Key Errors to Avoid

#### 1. Look-Ahead Bias
```python
# ❌ WRONG: Using t+1 data to generate signal at time t
signal_t = calculate_signal(data[:t+2])

# ✅ CORRECT: Only use data up to and including time t
signal_t = calculate_signal(data[:t+1])
```

#### 2. Transaction Cost Calculation
```python
# Turnover = sum of absolute weight changes
turnover = abs(new_weights - old_weights).sum()
transaction_cost = turnover * cost_rate

# Net return = gross return - transaction costs
net_return = gross_return - transaction_cost
```

#### 3. Abnormal Data Handling
```python
# Limit extreme returns
returns_clean = returns.clip(-0.1, 0.1)  # Limit to ±10%
returns_clean = returns_clean.replace([np.inf, -np.inf], 0)
```

---

### 📊 Long-Short Position Construction Logic

```python
# Long: Top quantile by signal score
long_stocks = signal_scores.quantile(1 - long_quantile)
# Short: Bottom quantile by signal score
short_stocks = signal_scores.quantile(short_quantile)

# Equal weight allocation
long_weight = 0.5 / n_long   # Long total weight 50%
short_weight = -0.5 / n_short # Short total weight -50%
```

---

### 🎯 Performance Metric Annualization

Assuming 252 trading days per year:
```python
annual_return = (1 + returns.mean()) ** 252 - 1
annual_volatility = returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility
```

---

## IV. Data Dependencies and Integration

### Dependency Chain
```
Part 1 (Data Analysis)
  ↓ Provides: Return data, price data
Part 2 (Alpha Modeling)
  ↓ Provides: Prediction scores (predictions)
Part 3 (Strategy Backtesting)
  ↓ Uses above data for strategy construction
```

### Data Format Requirements

**Input Data**:
- `returns`: pd.DataFrame `[time x stocks]` - Returns matrix
- `prices`: pd.DataFrame - Price data (can be MultiIndex or wide format)
- `predictions`: pd.DataFrame `[time x stocks]` - Prediction scores (optional)

**Output Results**:
- Structured dictionary containing all backtest results and analysis reports

---

## V. Assessment Criteria

### Code Implementation (60%)
- ✅ Strategy logic is correct with no obvious bugs
- ✅ Backtesting framework is complete and functional
- ✅ Code structure is clear with comprehensive comments

### Strategy Effectiveness (15%)
- ✅ Strategy generates positive returns with controllable risk
- ✅ Strategy logic is reasonable with economic intuition

### Analysis Quality (25%)
- ✅ Performance analysis is comprehensive and thorough
- ✅ Improvement recommendations are specific and actionable
- ✅ Visualizations are clear and professional
- ✅ Conclusions are supported by data

---

## VI. Recommended Implementation Path

### 🎯 Phase 1: Implement Basic Backtesting Framework

**Priority Order**:
1. **Implement simple Bollinger Bands strategy first** (`_BollingerSingleAsset.update()`)
   - Logic is relatively simple
   - Easy to debug and verify
   
2. **Implement core backtesting engine** (`LongShortStrategy.backtest()`)
   - Start with most simplified version
   - First handle prediction signals (skip technical indicators)
   - Gradually add features

3. **Add MACD strategy** (`_MACDSingleAsset.update()`)

### 🎯 Phase 2: Complete Performance Evaluation

4. **Implement performance metrics calculation** (`calculate_performance_metrics()`)
5. **Implement benchmark comparison** (`compare_with_benchmarks()`)
6. **Generate performance report** (`generate_performance_report()`)

### 🎯 Phase 3: In-Depth Analysis and Optimization

7. **Strategy analysis** (`analyze_strategy_performance()`)
8. **Drawdown identification** (`identify_drawdown_periods()`)
9. **Improvement recommendations** (`generate_improvement_proposals()`)

---

## VII. Available Code Resources

### Helper Functions Already Provided

1. **`_extract_close_prices()`** - Extract close prices from MultiIndex
2. **`_pct_change_returns()`** - Calculate returns
3. **`_BaseSingleAssetStrategy`** - Base class for single asset strategies
4. **`LongShortStrategy._construct_weights_from_scores_once()`** - Single timestamp weight construction
5. **`_MACDSingleAsset._ema()`** - EMA calculation helper method

### Modules Ready to Use

- `DataLoader` - Data loader
- Part 1 return calculations
- Part 2 model predictions

---

## VIII. Potential Challenges and Solutions

### Challenge 1: High Backtesting Engine Complexity
**Solutions**:
- Use incremental development, start with simplest version
- Test with small dataset first (first 5000 rows, 20 stocks)
- Use print statements to debug key steps

### Challenge 2: Many Performance Metric Details
**Solutions**:
- Reference standard financial library implementations (e.g., empyrical)
- Test each metric individually for validation
- Use known data to verify calculation correctness

### Challenge 3: Strategy Performance May Not Be Ideal
**Solutions**:
- Simple technical indicator strategies may perform mediocrely - this is normal
- Focus on correct code implementation, not absolute strategy returns
- Analysis section should honestly reflect issues and propose improvements

---

## IX. Questions to Clarify

Based on analysis, the following questions may need clarification with students or course administrators:

1. **Missing Test Cases** - All three test files are empty, how can students verify implementation correctness?
   - Recommendation: Provide at least basic unit tests
   
2. **Data Availability** - Do students have complete data access?
   - 5-minute data (2019-2024, 100 stocks)
   - Daily data
   - Stock weights data
   
3. **Part 1 and Part 2 Completion** - Have students completed the first two parts?
   - If not completed, the predictions parameter in Part 3 cannot be used
   
4. **Visualization Requirements Specificity** - TASK_3.md mentions 6 types of charts, must all be implemented?

5. **Report Format** - Is HTML report optional or mandatory?

---

## X. Summary

### Core Task Checklist

Functions students must **implement from scratch**:

| Task | File | Function/Class | Difficulty |
|------|------|----------------|------------|
| 3.1.1 | task7_backtest.py | `_BollingerSingleAsset.update()` | ⭐⭐⭐ |
| 3.1.2 | task7_backtest.py | `_MACDSingleAsset.update()` | ⭐⭐⭐ |
| 3.1.3 | task7_backtest.py | `LongShortStrategy.backtest()` | ⭐⭐⭐⭐⭐ |
| 3.2.1 | task8_performance.py | `calculate_performance_metrics()` | ⭐⭐⭐⭐ |
| 3.2.2 | task8_performance.py | `compare_with_benchmarks()` | ⭐⭐⭐ |
| 3.2.3 | task8_performance.py | `generate_performance_report()` | ⭐⭐ |
| 3.3.1 | task9_analysis.py | `analyze_strategy_performance()` | ⭐⭐⭐⭐ |
| 3.3.2 | task9_analysis.py | `identify_drawdown_periods()` | ⭐⭐⭐ |
| 3.3.3 | task9_analysis.py | `generate_improvement_proposals()` | ⭐⭐⭐ |

### Most Critical Implementation

**`LongShortStrategy.backtest()`** is the core of entire Part 3, accounting for approximately 40% of the workload. Students are advised to:
1. Understand the overall workflow first
2. Implement module by module
3. Test frequently with small datasets
4. Ensure no look-ahead bias

---

## XI. Detailed Implementation Specifications

### `_BollingerSingleAsset.update()` Specification

**Input Parameters**:
- `price` (float): Current bar close price
- `volume` (Optional[float]): Current bar volume (not used in basic implementation)

**Return Value**:
- `float`: Trading signal (+1.0 = buy, -1.0 = sell, 0.0 = no signal/hold)

**State Variables to Maintain**:
- `self.prices`: List of historical prices (append-only)
- `self.prev_below_lower`: Boolean flag for previous state (price below lower band)
- `self.prev_above_upper`: Boolean flag for previous state (price above upper band)

**Algorithm Steps**:
```
1. Append current price to history
2. If history length < window size:
   - Return 0.0 (insufficient data)
3. Get most recent window_size prices
4. Calculate:
   - mean = average of window prices
   - std = standard deviation of window prices
   - upper_band = mean + (num_std * std)
   - lower_band = mean - (num_std * std)
5. Determine current state:
   - is_below_lower = (price < lower_band)
   - is_above_upper = (price > upper_band)
6. Generate signal based on crossover:
   - If was_below_lower AND now (price >= lower_band): return +1.0 (buy signal)
   - If was_above_upper AND now (price <= upper_band): return -1.0 (sell signal)
   - Otherwise: return 0.0
7. Update state flags for next iteration
```

**Important Notes**:
- Use `np.std(ddof=0)` for population standard deviation
- Only trigger signals on **crossovers**, not sustained conditions
- First window_size-1 bars will return 0.0 signals

---

### `_MACDSingleAsset.update()` Specification

**Input Parameters**:
- `price` (float): Current bar close price
- `volume` (Optional[float]): Current bar volume (not used)

**Return Value**:
- `float`: Trading signal (+1.0 = buy, -1.0 = sell, 0.0 = no signal/hold)

**State Variables to Maintain**:
- `self.ema_fast`: Fast EMA value (initialized to None)
- `self.ema_slow`: Slow EMA value (initialized to None)
- `self.macd_sig`: Signal line EMA value (initialized to None)
- `self.prev_macd_lt_sig`: Boolean flag (MACD < signal line in previous bar)
- `self.prev_macd_gt_sig`: Boolean flag (MACD > signal line in previous bar)

**Algorithm Steps**:
```
1. Update fast and slow EMAs using helper method _ema():
   - ema_fast = _ema(prev_ema_fast, price, fast_period)
   - ema_slow = _ema(prev_ema_slow, price, slow_period)
2. Calculate MACD line:
   - macd = ema_fast - ema_slow
3. Update signal line EMA:
   - macd_sig = _ema(prev_macd_sig, macd, signal_period)
4. Determine current state:
   - is_lt = (macd < macd_sig)
   - is_gt = (macd > macd_sig)
5. Generate signal based on crossover:
   - If was_lt AND now (macd >= macd_sig): return +1.0 (bullish crossover)
   - If was_gt AND now (macd <= macd_sig): return -1.0 (bearish crossover)
   - Otherwise: return 0.0
6. Update state flags for next iteration
```

**EMA Formula** (already implemented in `_ema()` helper):
```
alpha = 2.0 / (span + 1.0)
new_ema = alpha * current_price + (1.0 - alpha) * prev_ema
If prev_ema is None: new_ema = current_price
```

---

### `LongShortStrategy.backtest()` Detailed Specification

**Input Parameters**:
- `returns` (Optional[pd.DataFrame]): Returns matrix [time x stocks]. If None, compute from prices.
- `prices` (Optional[pd.DataFrame]): Price data (MultiIndex or wide format).
- `predictions` (Optional[pd.DataFrame]): Prediction scores [time x stocks]. Required if signal_type='predictions'.

**Return Value**:
- `Dict[str, Any]` containing:
  - `'returns'`: pd.Series - Portfolio returns
  - `'nav'`: pd.Series - Net asset value (starts at 1.0)
  - `'weights'`: pd.DataFrame - Weights history
  - `'turnover'`: pd.Series - Turnover per period
  - `'transaction_costs'`: pd.Series - Transaction costs per period
  - `'gross_exposure'`: pd.Series - Sum of absolute weights
  - `'capital_used'`: pd.Series - Actual capital deployed
  - `'trade_log'`: List[Dict] - Trade records

**Algorithm Pseudocode**:

```python
# 1. PREPROCESSING
if prices is None and returns is None:
    raise ValueError("Must provide either prices or returns")

if returns is None:
    price_matrix = _extract_close_prices(prices)
    returns = _pct_change_returns(price_matrix)
else:
    price_matrix = _extract_close_prices(prices) if prices is not None else None

symbols = list(returns.columns)
time_index = returns.index

# 2. INITIALIZATION
if signal_type in ['bollinger', 'macd']:
    self._init_single_asset_strategies(symbols)

# Initialize containers
weights_hist = pd.DataFrame(0.0, index=time_index, columns=symbols)
turnover_series = pd.Series(0.0, index=time_index)
tx_cost_series = pd.Series(0.0, index=time_index)
port_ret_series = pd.Series(0.0, index=time_index)
gross_exp_series = pd.Series(0.0, index=time_index)
trade_log = []

# State variables
current_weights = pd.Series(0.0, index=symbols)
pending_weights = pd.Series(0.0, index=symbols)
bars_since_rebalance = 0

# 3. MAIN LOOP
for i, timestamp in enumerate(time_index):
    # a) Get current period returns
    curr_returns = returns.loc[timestamp]
    
    # b) Execute pending weight changes (apply from previous period)
    delta = pending_weights - current_weights
    turnover = abs(delta).sum()
    tx_cost = turnover * self.transaction_cost
    
    # Record trades if turnover > 0
    if turnover > 0:
        for sym in symbols:
            if abs(delta[sym]) > 1e-8:
                trade_log.append({
                    'timestamp': timestamp,
                    'symbol': sym,
                    'weight_change': delta[sym],
                    'turnover_contribution': abs(delta[sym])
                })
    
    # Update current weights
    current_weights = pending_weights.copy()
    
    # c) Calculate portfolio return
    gross_ret = (current_weights * curr_returns).sum()
    net_ret = gross_ret - tx_cost
    
    # Store results for this period
    port_ret_series.loc[timestamp] = net_ret
    weights_hist.loc[timestamp] = current_weights
    turnover_series.loc[timestamp] = turnover
    tx_cost_series.loc[timestamp] = tx_cost
    gross_exp_series.loc[timestamp] = abs(current_weights).sum()
    
    # d) Generate signals for next period
    bars_since_rebalance += 1
    
    if bars_since_rebalance >= self.rebalance_periods:
        # Time to rebalance
        bars_since_rebalance = 0
        
        if self.signal_type == 'predictions':
            # Use provided prediction scores
            if predictions is None:
                raise ValueError("predictions required for signal_type='predictions'")
            scores = predictions.loc[timestamp]
            pending_weights = self._construct_weights_from_scores_once(scores, symbols)
            
        elif self.signal_type in ['bollinger', 'macd']:
            # Generate technical signals
            if price_matrix is None:
                raise ValueError("prices required for technical signal strategies")
            
            signals = pd.Series(0.0, index=symbols)
            curr_prices = price_matrix.loc[timestamp]
            
            for sym in symbols:
                if pd.notna(curr_prices[sym]):
                    signal = self._single_asset[sym].update(float(curr_prices[sym]))
                    signals[sym] = signal
            
            # Convert signals to weights
            pending_weights = self._construct_weights_from_scores_once(signals, symbols)
    
    # else: keep pending_weights unchanged (no rebalancing yet)

# 4. POST-PROCESSING
nav = (1.0 + port_ret_series).cumprod()
capital_used = gross_exp_series * nav

results = {
    "returns": port_ret_series,
    "nav": nav,
    "weights": weights_hist,
    "turnover": turnover_series,
    "transaction_costs": tx_cost_series,
    "gross_exposure": gross_exp_series,
    "capital_used": capital_used,
    "trade_log": trade_log,
}
return results
```

**Critical Implementation Details**:

1. **Weight Application Timing**:
   - Weights computed at time `t` are applied at time `t+1`
   - This ensures no look-ahead bias
   - Transaction costs are incurred when weights change

2. **Rebalancing Logic**:
   - Only recompute weights every `rebalance_periods` bars
   - Between rebalances, maintain existing positions
   - This reduces turnover and transaction costs

3. **Position Construction** (via `_construct_weights_from_scores_once`):
   - Already implemented - calculates equal-weight long/short positions
   - Long: top `long_quantile` by score
   - Short: bottom `short_quantile` by score
   - Each side gets 50% gross weight (normalized to `max_gross_leverage`)

4. **Edge Cases**:
   - Handle NaN returns (replace with 0.0)
   - Handle suspended stocks (skip in signal generation)
   - Ensure weights sum to near-zero (dollar-neutral)

---

### `calculate_performance_metrics()` Detailed Specification

**Input Parameters**:
- `returns_series` (pd.Series): Portfolio returns (one value per time period)

**Return Value**:
- `Dict[str, float]` containing all performance metrics

**Required Metrics and Formulas**:

```python
# Clean data
rets = returns_series.dropna()
if len(rets) == 0:
    return {}  # or raise error

# 1. RETURN METRICS
total_return = (1 + rets).prod() - 1
mean_return = rets.mean()
annualized_return = (1 + mean_return) ** 252 - 1

# 2. RISK METRICS
volatility = rets.std()
annualized_volatility = volatility * np.sqrt(252)

# Maximum drawdown
cumulative = (1 + rets).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()  # Most negative value

# 3. RISK-ADJUSTED RETURNS
if annualized_volatility > 0:
    sharpe_ratio = annualized_return / annualized_volatility
else:
    sharpe_ratio = 0.0

# Sortino ratio (downside deviation)
negative_rets = rets[rets < 0]
if len(negative_rets) > 0:
    downside_std = negative_rets.std()
    annualized_downside = downside_std * np.sqrt(252)
    if annualized_downside > 0:
        sortino_ratio = annualized_return / annualized_downside
    else:
        sortino_ratio = 0.0
else:
    sortino_ratio = float('inf') if annualized_return > 0 else 0.0

# 4. WIN RATE AND PROFIT-LOSS RATIO
winning_periods = (rets > 0).sum()
losing_periods = (rets < 0).sum()
total_periods = len(rets)
win_rate = winning_periods / total_periods if total_periods > 0 else 0.0

avg_win = rets[rets > 0].mean() if winning_periods > 0 else 0.0
avg_loss = abs(rets[rets < 0].mean()) if losing_periods > 0 else 0.0
profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

# 5. RETURN DICTIONARY
return {
    'total_return': float(total_return),
    'annualized_return': float(annualized_return),
    'annualized_volatility': float(annualized_volatility),
    'sharpe_ratio': float(sharpe_ratio),
    'sortino_ratio': float(sortino_ratio),
    'max_drawdown': float(max_drawdown),
    'win_rate': float(win_rate),
    'profit_loss_ratio': float(profit_loss_ratio),
}
```

**Important Constants**:
- **252**: Number of trading days per year (standard assumption)
- **SQRT(252) ≈ 15.87**: Annualization factor for daily volatility

---

### `compare_with_benchmarks()` Detailed Specification

**Input Parameters**:
- `strategy_returns` (pd.Series): Strategy returns
- `benchmark_returns` (pd.Series): Benchmark returns

**Return Value**:
- `Dict[str, float]` containing comparison metrics

**Implementation**:

```python
# 1. ALIGN DATA
strat = strategy_returns.dropna()
if strat.empty:
    return {}

bench = benchmark_returns.reindex(strat.index).fillna(0.0)

if len(strat) != len(bench) or len(strat) < 2:
    return {}  # Insufficient data

# 2. EXCESS RETURNS
excess = strat - bench
excess_mean = excess.mean()
excess_std = excess.std()

annualized_excess = (1 + excess_mean) ** 252 - 1
tracking_error = excess_std * np.sqrt(252)

# 3. INFORMATION RATIO
if tracking_error > 0:
    information_ratio = annualized_excess / tracking_error
else:
    information_ratio = 0.0

# 4. BETA
if bench.std() > 0:
    covariance = strat.cov(bench)
    variance_bench = bench.var()
    beta = covariance / variance_bench
else:
    beta = 0.0

# 5. CORRELATION
correlation = strat.corr(bench)

# 6. ALPHA (Jensen's alpha)
# alpha = strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
# Assuming risk_free_rate = 0 for simplicity
strat_ann_ret = (1 + strat.mean()) ** 252 - 1
bench_ann_ret = (1 + bench.mean()) ** 252 - 1
alpha = strat_ann_ret - beta * bench_ann_ret

return {
    'information_ratio': float(information_ratio),
    'tracking_error': float(tracking_error),
    'beta': float(beta),
    'alpha': float(alpha),
    'correlation': float(correlation),
    'excess_return': float(annualized_excess),
}
```

---

### `generate_performance_report()` Detailed Specification

**Input Parameters**:
- `strategy_results` (Dict[str, Any]): Output from `LongShortStrategy.backtest()`

**Return Value**:
- `Dict[str, Any]` containing comprehensive report

**Implementation**:

```python
# 1. EXTRACT DATA
returns = strategy_results.get('returns', pd.Series())
nav = strategy_results.get('nav', pd.Series())
weights = strategy_results.get('weights', pd.DataFrame())
turnover = strategy_results.get('turnover', pd.Series())
tx_costs = strategy_results.get('transaction_costs', pd.Series())
trade_log = strategy_results.get('trade_log', [])

if returns.empty:
    return {'error': 'No returns data available'}

# 2. CALCULATE METRICS
metrics = calculate_performance_metrics(returns)

# 3. AGGREGATE STATISTICS
final_nav = float(nav.iloc[-1]) if not nav.empty else 1.0
num_periods = len(returns)
total_turnover = float(turnover.sum())
total_tx_costs = float(tx_costs.sum())
avg_turnover = float(turnover.mean())
num_trades = len(trade_log)

# 4. BUILD REPORT
report = {
    'metrics': metrics,
    'summary': {
        'final_nav': final_nav,
        'num_periods': num_periods,
        'total_turnover': total_turnover,
        'average_turnover_per_period': avg_turnover,
        'total_transaction_costs': total_tx_costs,
        'num_trades': num_trades,
    },
    'time_series': {
        'returns': returns.to_dict(),
        'nav': nav.to_dict(),
        'turnover': turnover.to_dict(),
    },
    # Optional: Add more sections as needed
}

return report
```

---

### `analyze_strategy_performance()` Detailed Specification

**Input Parameters**:
- `strategy_results` (Dict[str, Any]): Output from backtest

**Return Value**:
- `Dict[str, Any]` containing analysis results

**Implementation Structure**:

```python
# 1. EXTRACT DATA
returns = strategy_results.get('returns', pd.Series()).dropna()
nav = strategy_results.get('nav', pd.Series())
weights = strategy_results.get('weights', pd.DataFrame())

if returns.empty:
    return {'error': 'Insufficient data'}

# 2. DRAWDOWN ANALYSIS
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

max_dd = float(drawdown.min())
max_dd_date = drawdown.idxmin()

# Find start of max drawdown period
max_dd_start = running_max[:max_dd_date].idxmax()

# 3. WIN/LOSS STREAK ANALYSIS
def longest_streak(series, condition):
    """Find longest consecutive streak matching condition"""
    count = 0
    max_count = 0
    for val in series:
        if condition(val):
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count

longest_win_streak = longest_streak(returns, lambda x: x > 0)
longest_loss_streak = longest_streak(returns, lambda x: x < 0)

# 4. RETURN DECOMPOSITION (if weights available)
if not weights.empty:
    # Separate long and short positions
    long_weights = weights.clip(lower=0)
    short_weights = weights.clip(upper=0)
    
    # Calculate contributions (need returns matrix, not just portfolio returns)
    # This is approximate - full decomposition requires per-asset returns
    avg_long_exposure = float(long_weights.sum(axis=1).mean())
    avg_short_exposure = float(abs(short_weights.sum(axis=1)).mean())
else:
    avg_long_exposure = 0.0
    avg_short_exposure = 0.0

# 5. TIME SERIES METRICS
monthly_returns = returns.resample('M').sum()
winning_months = (monthly_returns > 0).sum()
total_months = len(monthly_returns)

# 6. BUILD ANALYSIS RESULT
analysis = {
    'drawdown': {
        'max_drawdown': max_dd,
        'max_drawdown_date': str(max_dd_date),
        'max_drawdown_start': str(max_dd_start),
    },
    'streaks': {
        'longest_winning_streak': int(longest_win_streak),
        'longest_losing_streak': int(longest_loss_streak),
    },
    'exposure': {
        'avg_long_exposure': avg_long_exposure,
        'avg_short_exposure': avg_short_exposure,
    },
    'time_series': {
        'winning_months': int(winning_months),
        'total_months': int(total_months),
        'monthly_win_rate': float(winning_months / total_months) if total_months > 0 else 0.0,
    },
    'final_nav': float(nav.iloc[-1]) if not nav.empty else 1.0,
}

return analysis
```

---

### `identify_drawdown_periods()` Detailed Specification

**Input Parameters**:
- `nav_series` (pd.Series): Net asset value time series
- `threshold` (float): Minimum drawdown magnitude to report (default: 0.05 = 5%)

**Return Value**:
- `List[Dict[str, Any]]`: List of drawdown period dictionaries

**Implementation**:

```python
if nav_series.empty or len(nav_series) < 2:
    return []

# Calculate drawdown series
running_max = nav_series.expanding().max()
drawdown = (nav_series - running_max) / running_max

# Identify periods where drawdown exceeds threshold
in_drawdown = drawdown < -threshold

drawdown_periods = []
start_idx = None

for i, (date, is_dd) in enumerate(zip(nav_series.index, in_drawdown)):
    if is_dd and start_idx is None:
        # Start of new drawdown
        start_idx = i
        start_date = date
        
    elif not is_dd and start_idx is not None:
        # End of drawdown
        end_date = date
        dd_slice = drawdown.iloc[start_idx:i+1]
        max_dd = float(dd_slice.min())
        max_dd_date = dd_slice.idxmin()
        
        # Calculate duration
        duration_days = (end_date - start_date).days
        
        # Calculate recovery (from max dd to exit)
        recovery_start = max_dd_date
        recovery_end = end_date
        recovery_days = (recovery_end - recovery_start).days
        
        drawdown_periods.append({
            'start_date': str(start_date),
            'end_date': str(end_date),
            'max_drawdown': max_dd,
            'max_drawdown_date': str(max_dd_date),
            'duration_days': duration_days,
            'recovery_days': recovery_days,
        })
        
        start_idx = None

# Handle case where still in drawdown at end
if start_idx is not None:
    end_date = nav_series.index[-1]
    dd_slice = drawdown.iloc[start_idx:]
    max_dd = float(dd_slice.min())
    max_dd_date = dd_slice.idxmin()
    duration_days = (end_date - start_date).days
    
    drawdown_periods.append({
        'start_date': str(start_date),
        'end_date': str(end_date),
        'max_drawdown': max_dd,
        'max_drawdown_date': str(max_dd_date),
        'duration_days': duration_days,
        'recovery_days': None,  # Not yet recovered
    })

return drawdown_periods
```

---

### `generate_improvement_proposals()` Detailed Specification

**Input Parameters**:
- `analysis_results` (Dict[str, Any]): Output from `analyze_strategy_performance()`

**Return Value**:
- `Dict[str, List[str]]`: Categorized list of improvement suggestions

**Implementation Logic**:

```python
proposals = {
    'risk_management': [],
    'return_enhancement': [],
    'cost_optimization': [],
    'operational_improvement': [],
}

# Extract key metrics
max_dd = abs(analysis_results.get('drawdown', {}).get('max_drawdown', 0))
longest_loss = analysis_results.get('streaks', {}).get('longest_losing_streak', 0)
final_nav = analysis_results.get('final_nav', 1.0)
total_return = final_nav - 1.0

# RISK MANAGEMENT RECOMMENDATIONS
if max_dd > 0.20:  # 20% drawdown threshold
    proposals['risk_management'].append(
        "Implement stop-loss mechanism: Maximum drawdown exceeds 20%. "
        "Consider adding portfolio-level stop-loss to limit losses during adverse periods."
    )

if longest_loss >= 5:
    proposals['risk_management'].append(
        "Add market regime filter: Strategy experiences long consecutive loss streaks (>=5 periods). "
        "Consider identifying and avoiding unfavorable market conditions."
    )

if max_dd > 0.15:
    proposals['risk_management'].append(
        "Implement dynamic position sizing: Reduce position sizes during high volatility periods "
        "or after drawdowns to better control risk."
    )

# RETURN ENHANCEMENT RECOMMENDATIONS
if total_return < 0:
    proposals['return_enhancement'].append(
        "Reassess signal quality: Strategy shows negative total return. "
        "Review and potentially redesign the underlying signals or factors."
    )

# If we had Sharpe ratio in analysis_results, we could check:
# if sharpe_ratio < 1.0:
#     proposals['return_enhancement'].append(...)

proposals['return_enhancement'].append(
    "Optimize factor combination: Consider combining multiple uncorrelated signals "
    "to improve risk-adjusted returns."
)

if analysis_results.get('time_series', {}).get('monthly_win_rate', 0) < 0.5:
    proposals['return_enhancement'].append(
        "Improve entry timing: Monthly win rate below 50%. "
        "Consider adding timing filters or entry rules to improve success rate."
    )

# COST OPTIMIZATION RECOMMENDATIONS
# (Would need turnover data from strategy_results, not analysis_results)
proposals['cost_optimization'].append(
    "Evaluate rebalancing frequency: Consider reducing rebalancing frequency "
    "to lower transaction costs if turnover is high."
)

proposals['cost_optimization'].append(
    "Optimize execution: Implement smarter order execution to reduce slippage "
    "and market impact, especially for large positions."
)

# OPERATIONAL IMPROVEMENT RECOMMENDATIONS
proposals['operational_improvement'].append(
    "Expand stock universe: Consider including more stocks to improve diversification "
    "and potentially find better alpha opportunities."
)

proposals['operational_improvement'].append(
    "Add liquidity filters: Ensure sufficient liquidity for all traded stocks "
    "to reduce execution risk and improve realism of backtest."
)

if max_dd < 0.10 and total_return > 0.20:
    # Strategy performing well
    proposals['operational_improvement'].append(
        "Conduct sensitivity analysis: Strategy shows good performance. "
        "Test robustness across different market periods and parameter settings."
    )

# Remove empty categories
return {k: v for k, v in proposals.items() if v}
```

---

## XII. Testing and Validation Guidelines

### Unit Testing Strategy

Even without provided test files, students should validate each function:

1. **Test with Simple Synthetic Data**:
```python
# Example: Test Bollinger Bands with known price pattern
bb = _BollingerSingleAsset(window=5, num_std=2.0)
prices = [10, 10, 10, 10, 10, 15, 15, 15, 15, 5, 5, 5, 5]
signals = [bb.update(p) for p in prices]
# Expect: first 5 signals = 0, then check for expected buy/sell signals
```

2. **Validate Metrics with Known Results**:
```python
# Example: Known return stream
returns = pd.Series([0.01, 0.02, -0.01, 0.015])  # 4 periods
metrics = calculate_performance_metrics(returns)
# Manually verify:
# total_return = (1.01 * 1.02 * 0.99 * 1.015) - 1 ≈ 0.0344
assert abs(metrics['total_return'] - 0.0344) < 0.001
```

3. **Check Edge Cases**:
- Empty data
- Single data point
- All zero returns
- NaN values

### Integration Testing

Test complete workflow:
```python
# Load small dataset
loader = DataLoader()
data = loader.load_5min_data().iloc[:1000, :20]  # Small subset

# Run full backtest pipeline
strategy = LongShortStrategy(signal_type='bollinger')
results = strategy.backtest(prices=data)
metrics = calculate_performance_metrics(results['returns'])
analysis = analyze_strategy_performance(results)
proposals = generate_improvement_proposals(analysis)

# Verify all outputs have expected structure
assert 'sharpe_ratio' in metrics
assert 'drawdown' in analysis
assert 'risk_management' in proposals
```

---

## XIII. Common Pitfalls and How to Avoid Them

### Pitfall 1: Index Alignment Issues
**Problem**: Mismatched indices when combining dataframes/series
**Solution**: Always use `.loc[]` and `.reindex()` explicitly
```python
# ❌ WRONG: Direct indexing can cause misalignment
portfolio_return = (weights * returns).sum()

# ✅ CORRECT: Ensure alignment
curr_returns = returns.loc[timestamp]
portfolio_return = (weights * curr_returns).sum()
```

### Pitfall 2: Off-by-One Errors in Signal Timing
**Problem**: Signals applied immediately cause look-ahead bias
**Solution**: Maintain separate "current" and "pending" weights
```python
# Weights computed at time t are pending
# They get applied (become current) at time t+1
```

### Pitfall 3: Incorrect Turnover Calculation
**Problem**: Forgetting to sum absolute changes
**Solution**: 
```python
# ✅ CORRECT
turnover = abs(new_weights - old_weights).sum()

# ❌ WRONG (can cancel out)
turnover = (new_weights - old_weights).sum()
```

### Pitfall 4: Not Handling NaN Values
**Problem**: NaN propagation breaks calculations
**Solution**: Explicitly handle NaN at each step
```python
returns_clean = returns.fillna(0.0)
weights_clean = weights.fillna(0.0)
```

### Pitfall 5: Memory Issues with Large Datasets
**Problem**: Loading entire 5-year, 100-stock dataset
**Solution**: 
- Test with small subsets first
- Consider batch processing for production
- Use efficient data types (float32 instead of float64)

---

## XIV. Visualization Requirements (Optional Enhancement)

While not part of core implementation, students may want to add visualizations:

### 1. NAV Curve Chart
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
nav.plot(label='Strategy NAV', linewidth=2)
plt.axhline(y=1.0, color='gray', linestyle='--', label='Initial Value')
plt.title('Strategy Net Asset Value Over Time')
plt.xlabel('Date')
plt.ylabel('NAV')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Drawdown Chart
```python
cumulative = nav
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

plt.figure(figsize=(12, 6))
drawdown.plot(color='red', linewidth=2)
plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
plt.title('Strategy Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()
```

### 3. Returns Distribution
```python
plt.figure(figsize=(10, 6))
returns.hist(bins=50, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('Distribution of Portfolio Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

### 4. Rolling Sharpe Ratio
```python
window = 252  # 1 year rolling window
rolling_returns = returns.rolling(window).mean() * 252
rolling_vol = returns.rolling(window).std() * np.sqrt(252)
rolling_sharpe = rolling_returns / rolling_vol

plt.figure(figsize=(12, 6))
rolling_sharpe.plot(linewidth=2)
plt.axhline(y=1.0, color='green', linestyle='--', label='Sharpe = 1.0')
plt.axhline(y=0.0, color='red', linestyle='--', label='Sharpe = 0.0')
plt.title(f'Rolling {window}-Day Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.show()
```

### 5. Turnover Analysis
```python
plt.figure(figsize=(12, 6))
turnover.plot(linewidth=1, alpha=0.7)
plt.axhline(y=turnover.mean(), color='red', linestyle='--', 
            label=f'Mean Turnover: {turnover.mean():.4f}')
plt.title('Portfolio Turnover Over Time')
plt.xlabel('Date')
plt.ylabel('Turnover')
plt.legend()
plt.grid(True)
plt.show()
```

---

## XV. Final Checklist for Students

### Before Considering Task Complete:

- [ ] **Task 3.1 - Backtesting**
  - [ ] `_BollingerSingleAsset.update()` implemented and tested
  - [ ] `_MACDSingleAsset.update()` implemented and tested
  - [ ] `LongShortStrategy.backtest()` implemented with all required outputs
  - [ ] Can run backtest with both technical signals and predictions
  - [ ] No look-ahead bias verified
  - [ ] Transaction costs correctly calculated
  - [ ] Trade log properly populated

- [ ] **Task 3.2 - Performance Evaluation**
  - [ ] `calculate_performance_metrics()` returns all required metrics
  - [ ] Sharpe ratio, max drawdown calculations verified
  - [ ] `compare_with_benchmarks()` implements all comparison metrics
  - [ ] `generate_performance_report()` produces complete report
  - [ ] All annualization factors correct (252 trading days)

- [ ] **Task 3.3 - Analysis & Improvement**
  - [ ] `analyze_strategy_performance()` covers all analysis dimensions
  - [ ] `identify_drawdown_periods()` correctly identifies drawdown events
  - [ ] `generate_improvement_proposals()` provides actionable recommendations
  - [ ] Recommendations are specific and data-driven

- [ ] **Code Quality**
  - [ ] All functions have docstrings
  - [ ] Critical sections have comments
  - [ ] No obvious bugs or errors
  - [ ] Handles edge cases (empty data, NaNs, etc.)
  - [ ] Code follows project style conventions

- [ ] **Validation**
  - [ ] Tested with small synthetic datasets
  - [ ] Tested with actual course data
  - [ ] Results are reasonable (no extreme values)
  - [ ] Can run examples in `if __name__ == "__main__"` blocks

---

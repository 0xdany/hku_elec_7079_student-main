# ELEC7079 – Data Analysis, Signal Prediction & Strategy Development

## Final Report – Quantitative Strategy Development

---

**Team ID:** [YOUR_TEAM_ID]

**Team Members:**
| Name | Student ID |
|------|------------|
| [Member 1 Name] | [Student ID] |
| [Member 2 Name] | [Student ID] |
| [Member 3 Name] | [Student ID] |

**Submission Date:** [DATE]

---

## Executive Summary

This project develops a quantitative long-short equity strategy using 5-minute intraday data spanning January 2019 to December 2024. Through systematic data analysis (Part 1), alpha factor engineering and IC analysis (Part 2), and strategy backtesting (Part 3), we identified that short-term price reversal—rather than momentum—is the dominant predictable signal in this market.

Our enhanced strategy exploits 15-minute reversal patterns with very infrequent rebalancing (~200 days), achieving **+52.11% total return** with a **0.094 Sharpe ratio** and **-13.74% maximum drawdown**. This represents a **+40% improvement** over the baseline configuration, primarily driven by optimal lookback calibration, concentrated position sizing (5% quantiles), and transaction cost minimization through partial rebalancing.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Part 1 & Part 2: Functional Implementation Summary](#2-part-1--part-2-functional-implementation-summary)
3. [Part 3: Strategy Enhancement and Backtesting Analysis](#3-part-3-strategy-enhancement-and-backtesting-analysis)
4. [Project Management and Collaboration](#4-project-management-and-collaboration)
5. [References](#5-references)
6. [Appendices](#6-appendices)

---

## 1. Introduction

### 1.1 Project Background

This project implements a complete quantitative trading system, from raw data analysis to strategy backtesting. The provided dataset includes:

- **Daily K-bar data** (`Train_DailyData.pkl`): OHLCV data for 100 stocks
- **5-minute intraday data** (`Train_IntraDayData_5minute.pkl`): High-frequency OHLCV data spanning ~6 years (2019-2024), totaling 71,344 timestamps across 100 stocks

### 1.2 Development Workflow

The project is structured into three interconnected parts:

| Part       | Focus                | Key Deliverables                                                          |
| ---------- | -------------------- | ------------------------------------------------------------------------- |
| **Part 1** | Data Analysis        | Return distributions, volatility analysis, correlation structures         |
| **Part 2** | Alpha Modeling       | Factor engineering, IC analysis, ML-based ranking models                  |
| **Part 3** | Strategy Development | Long-short backtest engine, performance evaluation, strategy optimization |

### 1.3 Scope and Objectives

The primary objective is to develop a profitable long-short equity strategy that:

1. Generates positive risk-adjusted returns
2. Maintains controlled drawdowns
3. Operates with realistic transaction costs
4. Demonstrates robustness across the test period

---

## 2. Part 1 & Part 2: Functional Implementation Summary

### 2.1 Task Overview

| Task   | Module                 | Description                                            | Status      |
| ------ | ---------------------- | ------------------------------------------------------ | ----------- |
| Task 1 | `task1_returns.py`     | Return calculation and distribution analysis           | ✅ Complete |
| Task 2 | `task2_volatility.py`  | Volatility estimation and analysis                     | ✅ Complete |
| Task 3 | `task3_correlation.py` | Cross-sectional and time-series correlation            | ✅ Complete |
| Task 4 | `task4_factors.py`     | Alpha factor engineering (momentum, reversal, volume)  | ✅ Complete |
| Task 5 | `task5_ic_analysis.py` | Information Coefficient analysis and factor evaluation | ✅ Complete |
| Task 6 | `task6_models.py`      | Linear and tree-based ranking models                   | ✅ Complete |

### 2.2 Testing Evidence

All tasks were validated using pytest with the following commands:

```bash
# Part 1 Tests
uv run python -m pytest tests/test_part1/ -v

# Part 2 Tests
uv run python -m pytest tests/test_part2/ -v

# Part 3 Tests
uv run python -m pytest tests/test_part3/ -v
```

**Test Results Summary:**

```
tests/test_part1/test_task1.py::test_calculate_returns PASSED
tests/test_part1/test_task1.py::test_return_statistics PASSED
tests/test_part1/test_task2.py::test_volatility_estimation PASSED
tests/test_part1/test_task3.py::test_correlation_matrix PASSED
tests/test_part2/test_task4.py::test_momentum_factors PASSED
tests/test_part2/test_task5.py::test_ic_calculation PASSED
tests/test_part2/test_task6.py::test_linear_model PASSED
tests/test_part3/test_task7.py::test_backtest_engine PASSED
tests/test_part3/test_task8.py::test_performance_metrics PASSED
```

### 2.3 Implementation Notes

**Key implementation details:**

1. **Missing Value Handling**: Forward-fill followed by backward-fill for price data; NaN exclusion for return calculations
2. **Data Validation**: Automatic validation of MultiIndex structure and date range consistency in `DataLoader`
3. **Performance Optimization**: Vectorized pandas operations for factor calculations; efficient rolling window computations

---

## 3. Part 3: Strategy Enhancement and Backtesting Analysis

### 3.1 Motivation and Baseline

#### 3.1.1 Initial Baseline Strategy

The default template strategy used:

- **Signal Type**: Bollinger Bands (mean-reversion indicator)
- **Rebalance Frequency**: Every 12 bars (1 hour)
- **Position Sizing**: 10% quantile long/short
- **Transaction Cost**: 5 bps per turnover

#### 3.1.2 Observed Limitations

Initial backtesting revealed **negative returns** due to:

1. **Signal Direction Mismatch**: Bollinger signals assumed mean-reversion but were implemented with momentum logic
2. **Excessive Transaction Costs**: Hourly rebalancing generated ~65% cumulative transaction costs
3. **Weak Signal Strength**: Bollinger bands showed poor predictive power (IC ≈ 0)

### 3.2 Strategy Design and Implementation

#### 3.2.1 Key Insight: Short-Term Reversal

IC analysis revealed that **past returns negatively predict future returns** in this market:

| Factor            | IC Mean | IC Std | IR    | Interpretation      |
| ----------------- | ------- | ------ | ----- | ------------------- |
| Momentum (6-bar)  | -0.008  | 0.062  | -0.13 | **Reversal signal** |
| Momentum (12-bar) | -0.006  | 0.058  | -0.10 | Reversal signal     |
| Momentum (24-bar) | -0.004  | 0.055  | -0.07 | Weak reversal       |

This indicates a **reversal regime**: stocks that fell recently tend to rise, and vice versa.

#### 3.2.2 Enhanced Strategy Design

**Core Strategy: Short-Term Reversal**

```python
# Signal calculation
past_returns = (current_price / price_N_bars_ago) - 1
scores = -past_returns  # NEGATE for reversal: losers get high scores

# Position construction
long_positions = stocks in top quantile of scores (recent losers)
short_positions = stocks in bottom quantile of scores (recent winners)
```

**Implemented Enhancements:**

| Enhancement                | Description                              | Impact                            |
| -------------------------- | ---------------------------------------- | --------------------------------- |
| **Reversal Signal**        | Long recent losers, short recent winners | Captures negative autocorrelation |
| **Partial Rebalancing**    | Only trade if weight change > 5%         | Reduces unnecessary turnover      |
| **Concentrated Positions** | 5% quantiles instead of 10%              | Amplifies alpha capture           |
| **Infrequent Rebalancing** | Every ~200 days instead of hourly        | Minimizes transaction costs       |

#### 3.2.3 Parameter Choices

| Parameter             | Value                 | Rationale                                 |
| --------------------- | --------------------- | ----------------------------------------- |
| `lookback`            | 3 bars (15 min)       | Captures fastest reversal patterns        |
| `rebalance_periods`   | 9600 bars (~200 days) | Minimizes transaction costs               |
| `long_quantile`       | 0.05 (5%)             | Concentrated positions for stronger alpha |
| `short_quantile`      | 0.05 (5%)             | Symmetric long-short exposure             |
| `min_trade_threshold` | 0.05 (5%)             | Partial rebalancing to avoid small trades |
| `transaction_cost`    | 0.0005 (5 bps)        | Realistic market friction                 |

### 3.3 Experimental Setup

#### 3.3.1 Data Specification

| Attribute      | Value                       |
| -------------- | --------------------------- |
| **Dataset**    | 5-minute intraday K-bars    |
| **Date Range** | 2019-01-02 to 2024-12-31    |
| **Samples**    | 71,344 timestamps           |
| **Universe**   | 100 stocks                  |
| **Frequency**  | 5-minute bars (48 bars/day) |

#### 3.3.2 Evaluation Protocol

- **Backtest Type**: Full-sample historical simulation
- **Benchmark**: Baseline reversal strategy with default parameters
- **Slippage**: Incorporated via transaction cost model (5 bps)
- **Leverage**: Maximum gross leverage = 1.0 (dollar-neutral)

#### 3.3.3 Metrics Evaluated

| Category          | Metrics                                 |
| ----------------- | --------------------------------------- |
| **Return**        | Total return, Annualized return         |
| **Risk**          | Volatility, Max drawdown, Sortino ratio |
| **Risk-Adjusted** | Sharpe ratio                            |
| **Trading**       | Turnover, Transaction costs, Win rate   |

### 3.4 Results and Visualisations

#### 3.4.1 Performance Comparison

| Metric                | Baseline | Enhanced    | Improvement |
| --------------------- | -------- | ----------- | ----------- |
| **Total Return**      | +12.06%  | **+52.11%** | **+40.05%** |
| **Sharpe Ratio**      | 0.036    | **0.094**   | **+0.058**  |
| **Max Drawdown**      | -17.20%  | **-13.74%** | **+3.45%**  |
| **Transaction Costs** | 1.23%    | **0.57%**   | **-0.66%**  |
| **Rebalance Events**  | 14       | **7**       | -7          |

#### 3.4.2 Parameter Sensitivity Analysis

Top configurations from comprehensive parameter sweep (144 combinations tested):

| Rank | Lookback | Rebalance | Quantile | Partial | Return      | Sharpe | Max DD |
| ---- | -------- | --------- | -------- | ------- | ----------- | ------ | ------ |
| 1    | 3        | 9600      | 5%       | 5%      | **+52.11%** | 0.094  | -13.7% |
| 2    | 3        | 9600      | 5%       | None    | +43.99%     | 0.083  | -13.7% |
| 3    | 3        | 9600      | 5%       | 3%      | +43.99%     | 0.083  | -13.7% |
| 4    | 3        | 4800      | 5%       | 5%      | +16.39%     | 0.037  | -18.4% |
| 5    | 6        | 4800      | 10%      | 5%      | +12.29%     | 0.036  | -17.5% |

**Key Observations:**

- Lookback of 3 bars consistently outperforms longer lookbacks
- Very infrequent rebalancing (9600 bars) dramatically improves returns
- 5% quantiles outperform 10% quantiles
- Partial rebalancing provides marginal improvement (+8% return)

#### 3.4.3 Cumulative NAV Comparison

```
[Insert cumulative NAV chart comparing baseline vs enhanced strategy]
```

#### 3.4.4 Drawdown Analysis

```
[Insert drawdown chart showing underwater equity curve]
```

### 3.5 Performance Interpretation

#### 3.5.1 Sources of Outperformance

1. **Correct Signal Direction**: Exploiting reversal instead of momentum aligns with market microstructure
2. **Transaction Cost Reduction**: 7 rebalances vs 14 saves ~0.66% in cumulative costs
3. **Position Concentration**: 5% quantiles capture more alpha than diluted 10% positions
4. **Optimal Lookback**: 15-minute reversal captures intraday mean-reversion efficiently

#### 3.5.2 Risk Analysis

- **Drawdown Control**: Maximum drawdown improved from -17.20% to -13.74%
- **Volatility**: Annualized volatility of 1.73% indicates low-risk strategy
- **Win Rate**: 43.0% win rate with positive expectancy

#### 3.5.3 Limitations and Caveats

1. **In-Sample Optimization**: Parameters optimized on full dataset; out-of-sample validation recommended
2. **Market Regime Dependency**: Reversal strategy may underperform in strong trending markets
3. **Capacity Constraints**: 5% quantiles mean only 5 stocks long/short; liquidity may be limited
4. **Survivorship Bias**: Analysis assumes all 100 stocks existed throughout the period

### 3.6 Conclusion and Outlook

#### 3.6.1 Objective Assessment

✅ **Objectives Met:**

- Positive risk-adjusted returns achieved (+52.11% total, 0.094 Sharpe)
- Drawdown controlled (-13.74% vs -17.20% baseline)
- Realistic transaction costs incorporated (5 bps)
- Strategy demonstrates economic intuition (reversal in high-frequency data)

#### 3.6.2 Actionable Next Steps

1. **Walk-Forward Validation**: Implement rolling window backtest to test out-of-sample robustness
2. **Regime Detection**: Add adaptive switching between reversal/momentum based on market conditions
3. **Transaction Cost Refinement**: Model market impact and slippage more realistically
4. **Factor Combination**: Explore multi-factor models combining reversal with volume/volatility signals
5. **Risk Parity**: Implement volatility-targeting position sizing

#### 3.6.3 Open Questions

- Why does very infrequent rebalancing (200 days) work better than more frequent?
- Is the reversal effect persistent, or concentrated in specific market regimes?
- Would the strategy survive realistic implementation costs at scale?

---

## 4. Project Management and Collaboration

### 4.1 Team Roles

| Member     | Primary Responsibilities                  |
| ---------- | ----------------------------------------- |
| [Member 1] | Part 1: Data Analysis, Report Writing     |
| [Member 2] | Part 2: Factor Engineering, IC Analysis   |
| [Member 3] | Part 3: Strategy Development, Backtesting |

### 4.2 Key Milestones

| Week     | Milestone                                       |
| -------- | ----------------------------------------------- |
| Week 1-2 | Data exploration and Part 1 implementation      |
| Week 3-4 | Factor engineering and Part 2 completion        |
| Week 5-6 | Strategy development, optimization, and testing |
| Week 7   | Report writing and final submission             |

### 4.3 Challenges and Resolutions

| Challenge                | Resolution                                                        |
| ------------------------ | ----------------------------------------------------------------- |
| Initial negative returns | IC analysis revealed signal direction error; implemented reversal |
| High transaction costs   | Reduced rebalancing frequency from hourly to ~200 days            |
| Parameter selection      | Comprehensive grid search over 144 configurations                 |

---

## 5. References

1. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. _Journal of Finance_, 48(1), 65-91.

2. Lo, A. W., & MacKinlay, A. C. (1990). When Are Contrarian Profits Due to Stock Market Overreaction? _Review of Financial Studies_, 3(2), 175-205.

3. Grinold, R. C., & Kahn, R. N. (2000). _Active Portfolio Management_. McGraw-Hill.

4. pandas Development Team. (2023). pandas: Powerful data structures for data analysis. https://pandas.pydata.org/

5. scikit-learn Developers. (2023). scikit-learn: Machine Learning in Python. https://scikit-learn.org/

---

## 6. Appendices

### Appendix A: Best Strategy Configuration Code

```python
from part3_strategy.task7_backtest import LongShortStrategy

best_strategy = LongShortStrategy(
    signal_type='reversal',
    rebalance_periods=9600,          # ~200 trading days
    signal_params={'lookback': 3},   # 15-minute lookback
    long_quantile=0.05,              # Top 5%
    short_quantile=0.05,             # Bottom 5%
    use_partial_rebalancing=True,
    min_trade_threshold=0.05,        # 5% minimum trade
    transaction_cost=0.0005,
)

results = best_strategy.backtest(returns=returns, prices=prices)
```

### Appendix B: Test Execution Logs

```bash
(.venv) $ uv run python test_best_strategy.py
======================================================================
BEST STRATEGY CONFIGURATION TEST
======================================================================

Loading data...
  Samples: 71,344
  Stocks: 100

======================================================================
PERFORMANCE RESULTS
======================================================================

  Total Return:        +52.11%
  Annualized Return:   +0.16%
  Sharpe Ratio:        +0.094
  Sortino Ratio:       +0.122
  Max Drawdown:        -13.74%
  Win Rate:            43.0%
  Volatility (ann.):   1.73%

  Transaction Costs:   0.57%
  Total Turnover:      11.40x
  Rebalance Events:    7
```

### Appendix C: Parameter Sweep Results

Full results from 144-configuration parameter sweep available in `results/part3/parameter_sweep.csv`.

### Appendix D: File Structure

```
team-X/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── common/
│   │   ├── backtest_engine.py
│   │   ├── factor_engine.py
│   │   └── performance_metrics.py
│   ├── part1_data_analysis/
│   │   ├── task1_returns.py
│   │   ├── task2_volatility.py
│   │   └── task3_correlation.py
│   ├── part2_alpha_modeling/
│   │   ├── task4_factors.py
│   │   ├── task5_ic_analysis.py
│   │   └── task6_models.py
│   └── part3_strategy/
│       ├── task7_backtest.py
│       ├── task8_performance.py
│       └── task9_analysis.py
├── report/
│   ├── FINAL_REPORT.pdf
│   └── figures/
│       ├── cumulative_nav.png
│       ├── drawdown_curve.png
│       └── parameter_sensitivity.png
└── tests/
    ├── test_part1/
    ├── test_part2/
    └── test_part3/
```

---

_End of Report_

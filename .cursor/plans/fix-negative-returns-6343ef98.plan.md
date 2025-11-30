<!-- 6343ef98-34c3-4f01-82c0-44170abd60a4 fcf0c1e2-23e6-4eb1-97ba-34636da7f14c -->
# Fix Negative Returns Strategy

## Root Cause Analysis Results

**Problem 1: Signal Direction is INVERTED**

- Current implementation uses momentum logic (long past winners, short past losers)
- This market exhibits **short-term reversal** with IC = -0.13 (past returns anti-predict future)
- Bollinger buy signals actually produce -0.0192% avg next return (losing proposition)

**Problem 2: Transaction Costs Too High**

- Current rebalancing every 12 bars (1 hour) generates ~65% cumulative TC
- This eats all potential alpha

**Problem 3: Wrong Factor Choice**

- Bollinger bands are noisy for this data
- Simple reversal factor has much stronger IC

---

## Proposed Code Changes

### Fix 1: Add Reversal Signal Type (task7_backtest.py)

Add a new signal type `'reversal'` that INVERTS the ranking:

```python
# In LongShortStrategy.backtest(), around line 346:
elif self.signal_type == "reversal":
    curr_prices = price_matrix.loc[ts]
    lookback = self.signal_params.get("lookback", 12)
    if i >= lookback:
        past_prices = price_matrix.iloc[i - lookback]
        scores = -(curr_prices / past_prices - 1)  # NEGATIVE for reversal
        scores = scores.dropna()
        pending_w = self._construct_weights_from_scores_once(scores, symbols)
```

### Fix 2: Increase Default Rebalance Period

Change default in `LongShortStrategy.__init__()`:

- From `rebalance_periods: int = 12` 
- To `rebalance_periods: int = 240` (weekly)

### Fix 3: Optimal Configuration

Based on backtesting analysis, the best parameters are:

- `signal_type='reversal'` with `signal_params={'lookback': 12}`
- `rebalance_periods=240` (weekly)
- `long_quantile=0.10`, `short_quantile=0.10`
- `transaction_cost=0.0005`

**Expected Performance**: ~36% total return, 0.49 Sharpe, -18% max DD over 5 years

---

## Alternative: Use Part 2 ML Model

Train `LinearRankingModel` or `TreeRankingModel` with reversal factors:

1. Use `calculate_momentum_factors()` with negative sign
2. Train model to predict forward returns
3. Use model predictions as `signal_type='predictions'`

---

## Implementation Order

1. Add `'reversal'` signal type to backtest engine
2. Update default `rebalance_periods` to 240
3. Run full backtest to verify positive returns
4. (Optional) Integrate Part 2 ML model for improved alpha

### To-dos

- [ ] Add 'reversal' signal type to LongShortStrategy.backtest()
- [ ] Change default rebalance_periods from 12 to 240 (weekly)
- [ ] Run full backtest with optimal config to verify positive returns
- [ ] (Optional) Integrate Part 2 ML model with reversal factors
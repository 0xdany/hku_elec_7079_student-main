"""
Task 7: Strategy Construction & Backtesting

This module provides a minimal yet complete multi-asset long-short strategy
implementation based on simple technical signals (Bollinger Bands or MACD),
including position construction, turnover and transaction cost modeling,
capital usage tracking, and trade logs, with a small runnable example.

Functions/classes follow simple, explicit interfaces to be easily reused by
other parts. If real data is not available, it can run with the sample data
from `DataLoader` for demonstration.

Author: ELEC4546/7079 Course
Date: December 2024
"""

from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import pandas as pd


def _extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a wide DataFrame of close prices `[time x symbols]` from
    a MultiIndex K-bar DataFrame `(symbol, field)`.

    Args:
        data (pd.DataFrame): MultiIndex columns: (symbol, field)

    Returns:
        pd.DataFrame: Close price matrix with symbols as columns
    """
    if isinstance(data.columns, pd.MultiIndex):
        symbols = data.columns.get_level_values(0).unique()
        close = {}
        for s in symbols:
            if (s, "close_px") in data.columns:
                close[s] = data[(s, "close_px")]
        if not close:
            raise KeyError("No close_px field found in provided data.")
        return pd.DataFrame(close)
    # Already wide-format
    return data.copy()


def _pct_change_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert close prices into simple returns by period.

    Args:
        prices (pd.DataFrame): Prices `[time x symbols]`

    Returns:
        pd.DataFrame: Returns `[time x symbols]`
    """
    ret = prices.pct_change().fillna(0.0)
    return ret


class _BaseSingleAssetStrategy:
    """
    Base class for single-asset strategies maintaining rolling state.
    Each call to update(price, ... ) advances the internal state by one bar
    and produces a signal based ONLY on information up to the current bar.
    The trade will be executed at the NEXT bar by the backtest engine.
    """

    def update(self, price: float, volume: Optional[float] = None) -> float:
        raise NotImplementedError


class _BollingerSingleAsset(_BaseSingleAssetStrategy):
    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = int(window)
        self.num_std = float(num_std)
        self.prices: List[float] = []
        self.prev_below_lower: Optional[bool] = None
        self.prev_above_upper: Optional[bool] = None

    def update(self, price: float, volume: Optional[float] = None) -> float:
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Bollinger Bands implementation hints:
        # 1. Update price history: self.prices.append(float(price))
        # 2. Check history length: if len(self.prices) < self.window: return 0.0
        # 3. Compute statistics:
        #    - window_prices = np.array(self.prices[-self.window:])
        #    - ma = window_prices.mean()
        #    - vol = window_prices.std(ddof=0)
        #    - upper = ma + self.num_std * vol
        #    - lower = ma - self.num_std * vol
        # 4. Detect crossings:
        #    - was_below_lower, was_above_upper (previous state)
        #    - is_below_lower = price < lower, is_above_upper = price > upper (current)
        #    - buy signal: was_below_lower and price >= lower
        #    - sell signal: was_above_upper and price <= upper
        # 5. Update flags: self.prev_below_lower, self.prev_above_upper
        # 6. Return signal: +1.0 (buy), -1.0 (sell), 0.0 (no signal)

        raise NotImplementedError("Please implement Bollinger Bands signal generation logic")


class _MACDSingleAsset(_BaseSingleAssetStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, sig: int = 9) -> None:
        self.fast = int(fast)
        self.slow = int(slow)
        self.sig = int(sig)
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.macd_sig: Optional[float] = None
        self.prev_macd_lt_sig: Optional[bool] = None
        self.prev_macd_gt_sig: Optional[bool] = None

    def _ema(self, prev: Optional[float], price: float, span: int) -> float:
        if prev is None:
            return float(price)
        alpha = 2.0 / (span + 1.0)
        return alpha * float(price) + (1.0 - alpha) * prev

    def update(self, price: float, volume: Optional[float] = None) -> float:
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # MACD implementation hints:
        # 1. Update EMAs:
        #    - self.ema_fast = self._ema(self.ema_fast, price, self.fast)
        #    - self.ema_slow = self._ema(self.ema_slow, price, self.slow)
        # 2. Compute MACD line: macd = self.ema_fast - self.ema_slow
        # 3. Update signal line EMA: self.macd_sig = self._ema(self.macd_sig, macd, self.sig)
        # 4. Detect crossovers:
        #    - Previous state: was_lt, was_gt (MACD vs signal)
        #    - Current state: is_lt = macd < self.macd_sig, is_gt = macd > self.macd_sig
        #    - Bullish crossover: was_lt and macd >= self.macd_sig
        #    - Bearish crossover: was_gt and macd <= self.macd_sig
        # 5. Update flags: self.prev_macd_lt_sig, self.prev_macd_gt_sig
        # 6. Return signal: +1.0 (buy), -1.0 (sell), 0.0 (no signal)

        raise NotImplementedError("Please implement MACD signal generation logic")


class LongShortStrategy:
    """
    Multi-asset long-short strategy with simple signal-to-weights mapping and
    turnover-based transaction costs.

    Args:
        long_quantile (float): Top quantile to long (e.g., 0.1 for top 10%)
        short_quantile (float): Bottom quantile to short
        rebalance_periods (int): Rebalance frequency in bars
        transaction_cost (float): Cost per unit turnover (e.g., 0.0005 = 5bps)
        max_gross_leverage (float): Sum(|weights|) target at rebalance (e.g., 1.0)
        signal_type (str): 'predictions', 'bollinger', or 'macd'
        signal_params (Optional[Dict[str, Any]]): Parameters for signal functions
    """

    def __init__(
        self,
        long_quantile: float = 0.1,
        short_quantile: float = 0.1,
        rebalance_periods: int = 12,
        transaction_cost: float = 0.0005,
        max_gross_leverage: float = 1.0,
        signal_type: str = "predictions",
        signal_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.long_quantile = float(long_quantile)
        self.short_quantile = float(short_quantile)
        self.rebalance_periods = int(rebalance_periods)
        self.transaction_cost = float(transaction_cost)
        self.max_gross_leverage = float(max_gross_leverage)
        self.signal_type = signal_type
        self.signal_params = signal_params or {}
        self._single_asset: Dict[str, _BaseSingleAssetStrategy] = {}

    def _init_single_asset_strategies(self, symbols: List[str]) -> None:
        self._single_asset = {}
        if self.signal_type == "bollinger":
            w = int(self.signal_params.get("window", 20))
            nstd = float(self.signal_params.get("num_std", 2.0))
            for s in symbols:
                self._single_asset[s] = _BollingerSingleAsset(window=w, num_std=nstd)
        elif self.signal_type == "macd":
            f = int(self.signal_params.get("fast", 12))
            sl = int(self.signal_params.get("slow", 26))
            sg = int(self.signal_params.get("signal", 9))
            for s in symbols:
                self._single_asset[s] = _MACDSingleAsset(fast=f, slow=sl, sig=sg)
        elif self.signal_type == "predictions":
            # No per-asset strategy instances needed
            pass
        else:
            raise ValueError("Unsupported signal_type: %s" % self.signal_type)

    def _construct_weights_from_scores_once(self, scores: pd.Series, symbols: List[str]) -> pd.Series:
        """
        Build equal-weight long/short weights cross-sectionally from a single
        timestamp score vector.
        """
        s = scores.dropna()
        if s.empty:
            return pd.Series(0.0, index=symbols)
        q_long = s.quantile(1.0 - self.long_quantile)
        q_short = s.quantile(self.short_quantile)
        long_names = list(s[s >= q_long].index)
        short_names = list(s[s <= q_short].index)
        gross_side = self.max_gross_leverage / 2.0
        wl = gross_side / max(len(long_names), 1)
        ws = -gross_side / max(len(short_names), 1)
        w = pd.Series(0.0, index=symbols)
        if long_names:
            w.loc[long_names] = wl
        if short_names:
            w.loc[short_names] = ws
        return w

    def _construct_target_weights(self, scores: pd.DataFrame) -> pd.DataFrame:
        # Deprecated in row-wise engine; kept for backward compatibility if needed
        index = scores.index
        symbols = list(scores.columns)
        out = pd.DataFrame(0.0, index=index, columns=symbols)
        for ts in index:
            out.loc[ts] = self._construct_weights_from_scores_once(scores.loc[ts], symbols)
        return out

    def backtest(
        self,
        returns: Optional[pd.DataFrame] = None,
        prices: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest of the long-short strategy.

        Args:
            returns (Optional[pd.DataFrame]): Asset returns `[time x symbols]`. If None, computed from prices.
            prices (Optional[pd.DataFrame]): K-bar data (MultiIndex columns) or wide close price matrix.
            predictions (Optional[pd.DataFrame]): Cross-sectional scores `[time x symbols]`. If None, derive from signals.

        Returns:
            Dict[str, Any]: Results including returns, nav, weights, turnover, costs, and trade log.
        """
        # TODO: STUDENT IMPLEMENTATION REQUIRED
        #
        # Long-short backtest implementation hints:
        # 1. Preprocessing:
        #    - Ensure either prices or returns is provided
        #    - Extract price matrix via _extract_close_prices()
        #    - Get symbol list and time index
        # 2. Initialize strategy and containers:
        #    - If using technical indicators, init per-asset strategies: _init_single_asset_strategies()
        #    - Create containers: weights history, turnover, costs, portfolio returns, etc.
        # 3. Main loop (iterate over time):
        #    a) Compute current-period returns
        #    b) Apply pending weight changes:
        #       - delta = pending_w - current_w
        #       - turnover = abs(delta).sum()
        #       - cost = turnover * transaction_cost
        #       - record trade log
        #    c) Portfolio return: (current_w * returns).sum() - transaction_cost
        #    d) Generate next-period signals:
        #       - Technical strategies: update per-asset strategies, get signals
        #       - Prediction strategies: use provided predictions
        #       - Rebalance by rebalance_periods
        #    e) Update state variables
        # 4. Post-processing:
        #    - nav = (1 + returns).cumprod()
        #    - capital_used = gross_exposure * nav
        #    - tidy trade log

        raise NotImplementedError("Please implement long-short backtest logic")

        results = {
            "returns": port_ret,
            "nav": nav,
            "weights": weights_hist,
            "turnover": turnover,
            "transaction_costs": tx_costs,
            "gross_exposure": gross_exposure,
            "capital_used": capital_used,
            "trade_log": trade_log,
        }
        return results


def run_backtest(
    strategy: LongShortStrategy,
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = 0.0005,
) -> Dict[str, Any]:
    """
    Required wrapper to run backtest with provided predictions and returns.

    Args:
        strategy (LongShortStrategy): Configured strategy instance
        predictions (pd.DataFrame): Cross-sectional scores `[time x symbols]`
        returns (pd.DataFrame): Asset returns `[time x symbols]`
        transaction_cost (float): Cost per unit turnover

    Returns:
        Dict[str, Any]: Backtest results dictionary
    """
    strategy.transaction_cost = float(transaction_cost)
    return strategy.backtest(returns=returns, predictions=predictions)


# Minimal example for quick manual run
if __name__ == "__main__":
    import sys, os
    _cur_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_cur_dir, os.pardir, os.pardir))
    _src_dir = os.path.join(_project_root, "src")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    try:
        from src.data_loader import DataLoader
    except ModuleNotFoundError:
        from data_loader import DataLoader

    loader = DataLoader()
    data_5m = loader.load_5min_data()
    prices_wide = _extract_close_prices(data_5m)
    # Subset for quick demo: first 5000 rows and first 20 symbols
    if len(prices_wide) > 5000:
        prices_wide = prices_wide.iloc[:5000]
    if prices_wide.shape[1] > 20:
        prices_wide = prices_wide.iloc[:, :20]
    rets = _pct_change_returns(prices_wide)

    # Example 1: Bollinger signal-based strategy (no external predictions)
    strat_sig = LongShortStrategy(
        long_quantile=0.2,
        short_quantile=0.2,
        rebalance_periods=12,
        transaction_cost=0.0005,
        max_gross_leverage=1.0,
        signal_type="bollinger",
        signal_params={"window": 20, "num_std": 2.0},
    )
    res_sig = strat_sig.backtest(returns=rets, prices=prices_wide, predictions=None)
    
    print("\n=== Bollinger Strategy Results ===")
    print(f"Final NAV: {float(res_sig['nav'].iloc[-1]):.4f}")
    print(f"Total Return: {(float(res_sig['nav'].iloc[-1]) - 1.0) * 100:.2f}%")
    print(f"Total Periods: {len(res_sig['returns'])}")
    print(f"Average Daily Turnover: {float(res_sig['turnover'].mean()):.4f}")
    print(f"Total Transaction Costs: {float(res_sig['transaction_costs'].sum()):.6f}")
    print(f"Max Gross Exposure: {float(res_sig['gross_exposure'].max()):.4f}")
    print(f"Number of Trades: {len(res_sig['trade_log'])}")
    
    # Show some performance metrics
    from src.part3_strategy.task8_performance import calculate_performance_metrics
    try:
        from part3_strategy.task8_performance import calculate_performance_metrics
    except ImportError:
        pass
    else:
        metrics = calculate_performance_metrics(res_sig["returns"])
        if metrics:
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
            print(f"Annualized Return: {metrics.get('annualized_return', 0) * 100:.2f}%")
            print(f"Annualized Volatility: {metrics.get('annualized_volatility', 0) * 100:.2f}%")

    # Example 2: If you had model scores (here we just reuse prices to create dummy scores)
    dummy_scores = prices_wide.rank(axis=1, method="first")
    strat_pred = LongShortStrategy(
        long_quantile=0.2,
        short_quantile=0.2,
        rebalance_periods=12,
        transaction_cost=0.0005,
        max_gross_leverage=1.0,
        signal_type="predictions",
    )
    res_pred = run_backtest(strat_pred, predictions=dummy_scores, returns=rets, transaction_cost=0.0005)
    
    print("\n=== Predictions-based Strategy Results ===")
    print(f"Final NAV: {float(res_pred['nav'].iloc[-1]):.4f}")
    print(f"Total Return: {(float(res_pred['nav'].iloc[-1]) - 1.0) * 100:.2f}%")
    print(f"Number of Trades: {len(res_pred['trade_log'])}")
    print(f"Total Transaction Costs: {float(res_pred['transaction_costs'].sum()):.6f}")



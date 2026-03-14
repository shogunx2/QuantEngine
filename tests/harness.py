import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from strategy import MLStrategy
from backtester import Backtester
from config import TRAIN_RATIO


def load_and_split(ticker: str, start: str, end: str, train_ratio: float = TRAIN_RATIO):
    """
    Load historical data for a ticker and split into train/test segments.

    Returns (full_data, train_df, test_df).
    """
    loader = DataLoader(ticker, start, end)
    data = loader.fetch()
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    print(f"Loaded {ticker} — Train: {len(train)} | Test: {len(test)}")
    if len(test) > 0:
        print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")
    return data, train, test


def run_strategy(train, test):
    """
    Fit MLStrategy on train data, generate signals on test data,
    and backtest using the current Backtester API.

    Returns (results, baseline, signals, confidence).
    """
    strat = MLStrategy()
    strat.fit(train)
    signals, confidence = strat.generate_signals(test)

    bt = Backtester()
    close = test["Close"]
    open_ = test["Open"]
    high = test["High"]
    low = test["Low"]
    atr = test["ATR"]

    results = bt.run(close, open_, high, low, atr, signals)
    baseline = bt.buy_and_hold(close, open_)

    return results, baseline, signals, confidence


def print_results(label, results, baseline):
    """
    Convenience printer for ML vs buy-and-hold results.
    """
    print(f"\n--- {label} ---")
    print(f"ML Return:    {results['total_return']:+.2f}% (₹{results['final_value']:,.2f})")
    print(f"Buy & Hold:   {baseline['total_return']:+.2f}% (₹{baseline['final_value']:,.2f})")
    print(f"Round Trips:  {results['num_trades']}")
    alpha = results["total_return"] - baseline["total_return"]
    print(f"Alpha:        {alpha:+.2f}%")
    return alpha

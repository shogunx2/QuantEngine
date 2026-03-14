import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data_loader import DataLoader
from strategy import MLStrategy
from backtester import Backtester
from config import TRAIN_RATIO, FEATURE_COLS


def load_and_split(ticker, start, end, train_ratio=TRAIN_RATIO):
    loader = DataLoader(ticker, start, end)
    data = loader.fetch()
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    print(f"Loaded {ticker} — Train: {len(train)} | Test: {len(test)}")
    if len(test) > 0:
        print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")
    return data, train, test


def run_strategy(train, test, threshold=None):
    from config import MIN_SIGNAL_THRESHOLD
    threshold = threshold or MIN_SIGNAL_THRESHOLD

    strategy = MLStrategy()
    strategy.fit(train)
    predictions = strategy.predict(test)
    regime = test["Regime_ok"].values
    signals = strategy.generate_signals(predictions, threshold=threshold, regime=regime)

    bt = Backtester()
    results = bt.run(test["Close"], test["Open"], signals)
    baseline = bt.buy_and_hold(test["Close"], test["Open"])

    return results, baseline, signals, predictions


def print_results(label, results, baseline):
    print(f"\n--- {label} ---")
    print(f"ML Return:    {results['total_return']:+.2f}% (₹{results['final_value']:,.2f})")
    print(f"Buy & Hold:   {baseline['total_return']:+.2f}% (₹{baseline['final_value']:,.2f})")
    print(f"Round Trips:  {results['num_trades']}")
    alpha = results["total_return"] - baseline["total_return"]
    print(f"Alpha:        {alpha:+.2f}%")
    return alpha

import sys
import numpy as np
import pandas as pd
from harness import load_and_split, run_strategy, print_results
from backtester import Backtester

DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"


def run(ticker=None, start=None, end=None):
    ticker = ticker or DEFAULT_TICKER
    start = start or DEFAULT_START
    end = end or DEFAULT_END
    print("=" * 50)
    print("STRESS TEST: Best Day Removal (Fragility Check)")
    print("=" * 50)

    _, train, test = load_and_split(ticker, start, end)

    results, baseline, signals, _ = run_strategy(train, test)
    full_alpha = print_results("FULL BACKTEST", results, baseline)

    prices = test["Close"].values.astype(float)
    daily_portfolio = []
    cash = 20000.0
    holdings = 0

    for i in range(len(prices)):
        sig = int(signals[i])
        if sig == 1 and holdings == 0:
            qty = int(cash // (prices[i] * 1.0015))
            if qty > 0:
                cash -= qty * prices[i] * 1.0015
                holdings = qty
        elif sig != 1 and holdings > 0:
            cash += holdings * prices[i] * 0.9975
            holdings = 0
        daily_portfolio.append(cash + holdings * prices[i])

    daily_returns = pd.Series(daily_portfolio).pct_change().fillna(0)
    best_day_idx = daily_returns.idxmax()
    best_day_return = daily_returns.iloc[best_day_idx]
    best_date = test.index[best_day_idx] if best_day_idx < len(test.index) else "N/A"

    print(f"\nBest single day: index {best_day_idx} ({best_date})")
    print(f"Best day return: {best_day_return*100:+.2f}%")

    drop_indices = list(range(len(test)))
    drop_indices.remove(best_day_idx)

    test_trimmed = test.iloc[drop_indices].copy()
    signals_trimmed = signals[drop_indices]

    bt = Backtester()
    trimmed_results = bt.run(test_trimmed["Close"], test_trimmed["Open"], signals_trimmed)
    trimmed_baseline = bt.buy_and_hold(test_trimmed["Close"], test_trimmed["Open"])
    trimmed_alpha = print_results("WITHOUT BEST DAY", trimmed_results, trimmed_baseline)

    print("\n" + "=" * 50)
    if trimmed_alpha > 0:
        print("PASS — Strategy survives best-day removal. Not fragile.")
        return True
    else:
        diff = full_alpha - trimmed_alpha
        print(f"FAIL — Removing one day costs {diff:.2f}% alpha. Strategy is fragile.")
        return False


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    run(ticker, start, end)


if __name__ == "__main__":
    main()

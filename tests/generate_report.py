import sys
import os
from datetime import datetime
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_loader import DataLoader
from strategy import MLStrategy
from backtester import Backtester
from config import (
    TRAIN_RATIO, INITIAL_CAPITAL, MIN_SIGNAL_THRESHOLD,
    FEATURE_COLS, BROKERAGE_RATE, BROKERAGE_CAP,
    SLIPPAGE_RATE, STT_RATE, DP_CHARGE,
)

DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
THRESHOLDS = [0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018]


def generate_report(ticker=None, start=None, end=None):
    ticker = ticker or DEFAULT_TICKER
    start = start or DEFAULT_START
    end = end or DEFAULT_END

    lines = []
    w = lines.append

    w(f"# QuantEngine Stress Test Report")
    w(f"")
    w(f"**Ticker:** {ticker} | **Period:** {start} to {end} | **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"")
    w(f"## Configuration")
    w(f"")
    w(f"| Parameter | Value |")
    w(f"|---|---|")
    w(f"| Initial Capital | ₹{INITIAL_CAPITAL:,.2f} |")
    w(f"| Signal Threshold | {MIN_SIGNAL_THRESHOLD*100:.1f}% |")
    w(f"| Train/Test Split | {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f} |")
    w(f"| Brokerage | {BROKERAGE_RATE*100:.2f}% (cap ₹{BROKERAGE_CAP:.0f}) |")
    w(f"| Slippage | {SLIPPAGE_RATE*100:.1f}% |")
    w(f"| STT | {STT_RATE*100:.1f}% |")
    w(f"| DP Charge | ₹{DP_CHARGE:.2f}/sell |")
    w(f"| Execution | T+1 Open |")
    w(f"| Features | {len(FEATURE_COLS)} |")
    w(f"")

    print(f"Loading data for {ticker}...")
    loader = DataLoader(ticker, start, end)
    data = loader.fetch()
    split_idx = int(len(data) * TRAIN_RATIO)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    strategy = MLStrategy()
    strategy.fit(train)
    predictions = strategy.predict(test)
    signals = strategy.generate_signals(predictions)

    bt = Backtester()
    results = bt.run(test["Close"], test["Open"], signals)
    baseline = bt.buy_and_hold(test["Close"], test["Open"])
    alpha = results["total_return"] - baseline["total_return"]

    w(f"## Baseline Results")
    w(f"")
    w(f"**Test Period:** {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
    w(f"")
    w(f"| | ML Strategy | Buy & Hold |")
    w(f"|---|---|---|")
    w(f"| Final Value | ₹{results['final_value']:,.2f} | ₹{baseline['final_value']:,.2f} |")
    w(f"| Total Return | {results['total_return']:+.2f}% | {baseline['total_return']:+.2f}% |")
    w(f"| Round Trips | {results['num_trades']} | 1 |")
    w(f"| **Alpha** | **{alpha:+.2f}%** | — |")
    w(f"")

    w(f"---")
    w(f"")
    w(f"## Stress Test 1: Feature Shuffle")
    w(f"")
    print("Running feature shuffle test...")
    rng = np.random.default_rng(42)
    train_shuf = train.copy()
    test_shuf = test.copy()
    for col in FEATURE_COLS:
        train_shuf[col] = rng.permutation(train_shuf[col].values)
        test_shuf[col] = rng.permutation(test_shuf[col].values)

    s2 = MLStrategy()
    s2.fit(train_shuf)
    pred_shuf = s2.predict(test_shuf)
    sig_shuf = s2.generate_signals(pred_shuf)
    res_shuf = bt.run(test_shuf["Close"], test_shuf["Open"], sig_shuf)

    shuf_pass = results["total_return"] > res_shuf["total_return"] and alpha > 0
    w(f"| | Real Features | Shuffled (Noise) |")
    w(f"|---|---|---|")
    w(f"| Return | {results['total_return']:+.2f}% | {res_shuf['total_return']:+.2f}% |")
    w(f"| Trades | {results['num_trades']} | {res_shuf['num_trades']} |")
    w(f"")
    w(f"**Result: {'PASS' if shuf_pass else 'FAIL'}** — {'Real features outperform noise.' if shuf_pass else 'Model may be overfitting.'}")
    w(f"")

    w(f"---")
    w(f"")
    w(f"## Stress Test 2: Threshold Sensitivity")
    w(f"")
    print("Running sensitivity test...")
    w(f"| Threshold | ML Return | B&H | Alpha | Trades | |")
    w(f"|---|---|---|---|---|---|")
    pass_count = 0
    for t in THRESHOLDS:
        sig_t = strategy.generate_signals(predictions, threshold=t)
        res_t = bt.run(test["Close"], test["Open"], sig_t)
        a = res_t["total_return"] - baseline["total_return"]
        ok = a > 0
        if ok:
            pass_count += 1
        w(f"| {t*100:.1f}% | {res_t['total_return']:+.2f}% | {baseline['total_return']:+.2f}% | {a:+.2f}% | {res_t['num_trades']} | {'✓' if ok else '✗'} |")

    sens_pass = pass_count >= len(THRESHOLDS) * 0.7
    w(f"")
    w(f"**Result: {'PASS' if sens_pass else 'FAIL'}** — Alpha positive in {pass_count}/{len(THRESHOLDS)} thresholds.")
    w(f"")

    w(f"---")
    w(f"")
    w(f"## Stress Test 3: Best Day Removal")
    w(f"")
    print("Running best-day removal test...")
    import pandas as pd
    portfolio_vals = results["portfolio_values"]
    daily_rets = pd.Series(portfolio_vals).pct_change().fillna(0)
    best_idx = daily_rets.idxmax()
    best_ret = daily_rets.iloc[best_idx]
    best_date = test.index[best_idx] if best_idx < len(test.index) else "N/A"

    drop = list(range(len(test)))
    drop.remove(best_idx)
    test_trim = test.iloc[drop].copy()
    sig_trim = signals[drop]
    res_trim = bt.run(test_trim["Close"], test_trim["Open"], sig_trim)
    bl_trim = bt.buy_and_hold(test_trim["Close"], test_trim["Open"])
    trim_alpha = res_trim["total_return"] - bl_trim["total_return"]

    best_pass = trim_alpha > 0
    w(f"Best single day: **{best_date}** ({best_ret*100:+.2f}%)")
    w(f"")
    w(f"| | Full | Without Best Day |")
    w(f"|---|---|---|")
    w(f"| ML Return | {results['total_return']:+.2f}% | {res_trim['total_return']:+.2f}% |")
    w(f"| Alpha | {alpha:+.2f}% | {trim_alpha:+.2f}% |")
    w(f"")
    w(f"**Result: {'PASS' if best_pass else 'FAIL'}** — {'Strategy survives best-day removal.' if best_pass else 'Strategy is fragile.'}")
    w(f"")

    w(f"---")
    w(f"")
    w(f"## Stress Test 4: Out-of-Sample (2026)")
    w(f"")
    print("Running out-of-sample test...")
    try:
        oos_loader = DataLoader(ticker, "2026-01-01", datetime.now().strftime("%Y-%m-%d"))
        oos_data = oos_loader.fetch()
        if len(oos_data) < 5:
            w(f"Only {len(oos_data)} OOS trading days available — skipped.")
            oos_pass = None
        else:
            res_oos = bt.run(oos_data["Close"], oos_data["Open"],
                             strategy.generate_signals(strategy.predict(oos_data)))
            bl_oos = bt.buy_and_hold(oos_data["Close"], oos_data["Open"])
            oos_alpha = res_oos["total_return"] - bl_oos["total_return"]

            w(f"**OOS Period:** {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} days)")
            w(f"")
            w(f"| | ML Strategy | Buy & Hold |")
            w(f"|---|---|---|")
            w(f"| Return | {res_oos['total_return']:+.2f}% | {bl_oos['total_return']:+.2f}% |")
            w(f"| Alpha | {oos_alpha:+.2f}% | — |")
            w(f"")
            oos_pass = oos_alpha > 0
            w(f"**Result: {'PASS' if oos_pass else 'FAIL'}**")
    except Exception as e:
        w(f"Could not run OOS test: {e}")
        oos_pass = None
    w(f"")

    w(f"---")
    w(f"")
    w(f"## Summary")
    w(f"")
    w(f"| Test | Result |")
    w(f"|---|---|")
    w(f"| Feature Shuffle | {'PASS' if shuf_pass else 'FAIL'} |")
    w(f"| Threshold Sensitivity | {'PASS' if sens_pass else 'FAIL'} |")
    w(f"| Best Day Removal | {'PASS' if best_pass else 'FAIL'} |")
    oos_label = "PASS" if oos_pass else ("FAIL" if oos_pass is False else "SKIPPED")
    w(f"| Out-of-Sample 2026 | {oos_label} |")
    w(f"")

    report = "\n".join(lines)

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{ticker}_{timestamp}.md"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {filepath}")
    return filepath


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    generate_report(ticker, start, end)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from data_loader import DataLoader
from strategy import MLStrategy, run_cv
from backtester import Backtester
from config import (
    TRAIN_RATIO, INITIAL_CAPITAL, MIN_SIGNAL_THRESHOLD, SWING_HORIZON,
    MAX_HOLD_DAYS, TP_ATR_MULT, SL_ATR_MULT, TRAIL_ATR_MULT, META_CONFIDENCE,
    CV_SPLITS, FEATURE_COLS,
)


def main():
    ticker = input("Enter NSE ticker (e.g., RELIANCE): ").strip()
    start = input("Start date (YYYY-MM-DD): ").strip()
    end = input("End date (YYYY-MM-DD): ").strip()

    print(f"\nFetching data for {ticker}...")
    loader = DataLoader(ticker, start, end)
    data = loader.fetch()
    print(f"Loaded {len(data)} trading days")

    print(f"\n{'=' * 55}")
    print(f"  PURGED WALK-FORWARD CV ({CV_SPLITS} folds)")
    print(f"{'=' * 55}")

    fold_results, avg_importance = run_cv(data)
    for fr in fold_results:
        print(
            f"  Fold {fr['fold']}: train={fr['train_size']} test={fr['test_size']} "
            f"buys={fr['buy_signals']} win={fr['win_rate']:.1f}% avg_ret={fr['avg_return']:+.2f}%"
        )

    if avg_importance:
        print(f"\n{'=' * 55}")
        print("  FEATURE IMPORTANCE (avg across folds)")
        print(f"{'=' * 55}")
        sorted_imp = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp:
            bar = "█" * int(imp * 200)
            print(f"  {feat:<20s} {imp:.4f} {bar}")

    print(f"\n{'=' * 55}")
    print("  FINAL BACKTEST (80/20 split)")
    print(f"{'=' * 55}")

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"  Train: {len(train_data)} days | Test: {len(test_data)} days")
    print(f"  Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}\n")

    strategy = MLStrategy()
    strategy.fit(train_data)
    signals, confidence = strategy.generate_signals(test_data)

    counts = pd.Series(signals).value_counts()
    print(
        f"  Signals — Buy: {counts.get(1, 0)} | "
        f"Sell: {counts.get(-1, 0)} | "
        f"Hold: {counts.get(0, 0)}"
    )
    buy_mask = signals == 1
    if buy_mask.sum() > 0:
        avg_conf = confidence[buy_mask].mean()
        print(f"  Avg meta-confidence on buys: {avg_conf:.3f}")

    bt = Backtester()
    test_close = test_data["Close"]
    test_open = test_data["Open"]
    test_high = test_data["High"]
    test_low = test_data["Low"]
    test_atr = test_data["ATR"]

    results = bt.run(test_close, test_open, test_high, test_low, test_atr, signals)
    baseline = bt.buy_and_hold(test_close, test_open)

    print(f"\n{'=' * 55}")
    print(f"  Initial Capital:      ₹{INITIAL_CAPITAL:,.2f}")
    print(f"  Signal Threshold:     CPP_score > 0.3 + meta > {META_CONFIDENCE}")
    print(f"  Swing Horizon:        {SWING_HORIZON}-day triple-barrier")
    print(f"  TP activation: {TP_ATR_MULT}×ATR → trail: {TRAIL_ATR_MULT}×ATR  SL: {SL_ATR_MULT}×ATR  Max Hold: {MAX_HOLD_DAYS}d")
    print(f"{'=' * 55}")

    print("\n  --- ML Strategy (Meta-Labeled) ---")
    print(f"  Final Value:  ₹{results['final_value']:,.2f}")
    print(f"  Total Return: {results['total_return']:+.2f}%")
    print(f"  Round Trips:  {results['num_trades']}")
    print(f"  Exits — Trail: {results['trail_hits']} | SL: {results['sl_hits']} | Time: {results['time_exits']}")

    print("\n  --- Buy & Hold Baseline ---")
    print(f"  Final Value:  ₹{baseline['final_value']:,.2f}")
    print(f"  Total Return: {baseline['total_return']:+.2f}%")

    print(f"\n{'=' * 55}")
    alpha = results["total_return"] - baseline["total_return"]
    if alpha > 0:
        print(f"  ML Strategy beats Buy & Hold by {alpha:+.2f}%")
    else:
        print(f"  Buy & Hold wins by {abs(alpha):.2f}%")


if __name__ == "__main__":
    main()

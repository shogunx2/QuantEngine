import sys
import numpy as np
from harness import load_and_split, run_strategy, print_results
from config import FEATURE_COLS

DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"


def run(ticker=None, start=None, end=None):
    ticker = ticker or DEFAULT_TICKER
    start = start or DEFAULT_START
    end = end or DEFAULT_END
    print("=" * 50)
    print("STRESS TEST: Feature Shuffle (Noise Detection)")
    print("=" * 50)

    _, train, test = load_and_split(ticker, start, end)

    real_results, baseline, _, _ = run_strategy(train, test)
    real_alpha = print_results("REAL FEATURES", real_results, baseline)

    rng = np.random.default_rng(42)
    train_shuffled = train.copy()
    test_shuffled = test.copy()
    for col in FEATURE_COLS:
        train_shuffled[col] = rng.permutation(train_shuffled[col].values)
        test_shuffled[col] = rng.permutation(test_shuffled[col].values)

    shuf_results, shuf_baseline, _, _ = run_strategy(train_shuffled, test_shuffled)
    shuf_alpha = print_results("SHUFFLED FEATURES", shuf_results, shuf_baseline)

    print("\n" + "=" * 50)
    if real_alpha > 0 and shuf_results["total_return"] < real_results["total_return"]:
        print("PASS — Real features outperform noise.")
        return True
    else:
        print("FAIL — Model may be overfitting.")
        return False


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    run(ticker, start, end)


if __name__ == "__main__":
    main()

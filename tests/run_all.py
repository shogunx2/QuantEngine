import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_feature_shuffle import run as run_shuffle
from test_sensitivity import run as run_sensitivity
from test_best_day import run as run_best_day
from test_out_of_sample import run as run_oos

DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"

TESTS = [
    ("Feature Shuffle", run_shuffle),
    ("Threshold Sensitivity", run_sensitivity),
    ("Best Day Removal", run_best_day),
    ("Out-of-Sample 2026", run_oos),
]


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    start = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START
    end = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_END

    print("=" * 60)
    print(f"  QUANTENGINE STRESS TEST SUITE — {ticker}")
    print("=" * 60)

    results = {}
    for name, test_fn in TESTS:
        print(f"\n\n{'#' * 60}")
        print(f"  Running: {name}")
        print(f"{'#' * 60}\n")
        if test_fn == run_oos:
            passed = test_fn(ticker=ticker, train_start=start, train_end=end)
        else:
            passed = test_fn(ticker=ticker, start=start, end=end)
        results[name] = passed

    print(f"\n\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

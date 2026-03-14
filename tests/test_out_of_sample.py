import sys
from harness import load_and_split, run_strategy, print_results

DEFAULT_TICKER = "ATGL"
DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END = "2025-12-31"
DEFAULT_OOS_START = "2026-01-01"
DEFAULT_OOS_END = "2026-03-12"


def run(ticker=None, train_start=None, train_end=None, oos_start=None, oos_end=None):
    ticker = ticker or DEFAULT_TICKER
    train_start = train_start or DEFAULT_TRAIN_START
    train_end = train_end or DEFAULT_TRAIN_END
    oos_start = oos_start or DEFAULT_OOS_START
    oos_end = oos_end or DEFAULT_OOS_END
    print("=" * 50)
    print("STRESS TEST: Out-of-Sample (2026)")
    print("=" * 50)

    from data_loader import DataLoader

    print("\nLoading training data...")
    train_loader = DataLoader(ticker, train_start, train_end)
    train_data = train_loader.fetch()
    print(f"Train: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")

    print("\nLoading OOS data...")
    oos_loader = DataLoader(ticker, oos_start, oos_end)
    oos_data = oos_loader.fetch()

    if len(oos_data) < 5:
        print(f"Only {len(oos_data)} OOS days available — not enough to test.")
        return

    print(f"OOS: {len(oos_data)} days ({oos_data.index[0].date()} to {oos_data.index[-1].date()})")

    results, baseline, _, _ = run_strategy(train_data, oos_data)
    alpha = print_results("OUT-OF-SAMPLE 2026", results, baseline)

    print("\n" + "=" * 50)
    if alpha > 0:
        print("PASS — Alpha holds on unseen 2026 data.")
        return True
    else:
        print("FAIL — No alpha on unseen data. Model may not generalize.")
        return False


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    train_start = sys.argv[2] if len(sys.argv) > 2 else None
    train_end = sys.argv[3] if len(sys.argv) > 3 else None
    oos_start = sys.argv[4] if len(sys.argv) > 4 else None
    oos_end = sys.argv[5] if len(sys.argv) > 5 else None
    run(ticker, train_start, train_end, oos_start, oos_end)


if __name__ == "__main__":
    main()

import sys
from harness import load_and_split, run_strategy, print_results

DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"
THRESHOLDS = [0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018]


def run(ticker=None, start=None, end=None):
    ticker = ticker or DEFAULT_TICKER
    start = start or DEFAULT_START
    end = end or DEFAULT_END
    print("=" * 50)
    print("STRESS TEST: Threshold Sensitivity")
    print("=" * 50)

    _, train, test = load_and_split(ticker, start, end)

    results_table = []

    for t in THRESHOLDS:
        results, baseline, _, _ = run_strategy(train, test, threshold=t)
        alpha = results["total_return"] - baseline["total_return"]
        results_table.append({
            "threshold": t,
            "ml_return": results["total_return"],
            "bh_return": baseline["total_return"],
            "alpha": alpha,
            "trades": results["num_trades"],
        })

    print(f"\n{'Threshold':>10} {'ML Return':>10} {'B&H':>10} {'Alpha':>10} {'Trades':>7}")
    print("-" * 52)

    pass_count = 0
    for r in results_table:
        marker = "✓" if r["alpha"] > 0 else "✗"
        if r["alpha"] > 0:
            pass_count += 1
        print(
            f"{r['threshold']*100:>9.1f}% "
            f"{r['ml_return']:>+9.2f}% "
            f"{r['bh_return']:>+9.2f}% "
            f"{r['alpha']:>+9.2f}% "
            f"{r['trades']:>6}  {marker}"
        )

    print("\n" + "=" * 50)
    if pass_count == len(THRESHOLDS):
        print(f"PASS — Alpha positive across all {len(THRESHOLDS)} thresholds. Robust.")
        return True
    elif pass_count >= len(THRESHOLDS) * 0.7:
        print(f"PASS (marginal) — Alpha positive in {pass_count}/{len(THRESHOLDS)} thresholds.")
        return True
    else:
        print(f"FAIL — Alpha positive in only {pass_count}/{len(THRESHOLDS)} thresholds. Brittle.")
        return False


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    start = sys.argv[2] if len(sys.argv) > 2 else None
    end = sys.argv[3] if len(sys.argv) > 3 else None
    run(ticker, start, end)


if __name__ == "__main__":
    main()

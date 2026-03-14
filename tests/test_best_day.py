import numpy as np
import pandas as pd
import pytest

from harness import load_and_split, run_strategy
from backtester import Backtester


DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"


@pytest.mark.slow
def test_best_day_removal_fragility():
    """
    Stress test: removing the single best portfolio day should not
    destroy all alpha; strategy should remain profitable.
    """
    _, train, test = load_and_split(DEFAULT_TICKER, DEFAULT_START, DEFAULT_END)

    results, baseline, signals, _ = run_strategy(train, test)
    full_alpha = results["total_return"] - baseline["total_return"]

    portfolio_values = np.asarray(results["portfolio_values"], dtype=float)
    daily_returns = pd.Series(portfolio_values).pct_change().fillna(0.0)
    best_day_idx = int(daily_returns.idxmax())

    drop_indices = list(range(len(test)))
    if best_day_idx in drop_indices:
        drop_indices.remove(best_day_idx)

    test_trimmed = test.iloc[drop_indices].copy()
    signals_trimmed = signals[drop_indices]

    bt = Backtester()
    trimmed_results = bt.run(
        test_trimmed["Close"],
        test_trimmed["Open"],
        test_trimmed["High"],
        test_trimmed["Low"],
        test_trimmed["ATR"],
        signals_trimmed,
    )
    trimmed_baseline = bt.buy_and_hold(test_trimmed["Close"], test_trimmed["Open"])
    trimmed_alpha = trimmed_results["total_return"] - trimmed_baseline["total_return"]

    assert trimmed_alpha > 0, (
        f"Removing best single day eliminates alpha "
        f"(full alpha={full_alpha:.2f}%, trimmed alpha={trimmed_alpha:.2f}%)."
    )

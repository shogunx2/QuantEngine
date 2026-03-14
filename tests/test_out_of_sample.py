import pytest

from data_loader import DataLoader
from harness import run_strategy


DEFAULT_TICKER = "ATGL"
DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END = "2025-12-31"
DEFAULT_OOS_START = "2026-01-01"
DEFAULT_OOS_END = "2026-03-12"


@pytest.mark.slow
def test_out_of_sample_2026_alpha():
    """
    Stress test: strategy trained on 2020–2025 data should
    still have positive alpha on unseen 2026 data.
    """
    train_loader = DataLoader(DEFAULT_TICKER, DEFAULT_TRAIN_START, DEFAULT_TRAIN_END)
    train_data = train_loader.fetch()

    oos_loader = DataLoader(DEFAULT_TICKER, DEFAULT_OOS_START, DEFAULT_OOS_END)
    oos_data = oos_loader.fetch()

    if len(oos_data) < 5:
        pytest.skip(f"Only {len(oos_data)} OOS days available — not enough to test.")

    results, baseline, _, _ = run_strategy(train_data, oos_data)
    alpha = results["total_return"] - baseline["total_return"]

    assert alpha > 0, f"No positive alpha on unseen 2026 data (alpha={alpha:.2f}%). Model may not generalize."

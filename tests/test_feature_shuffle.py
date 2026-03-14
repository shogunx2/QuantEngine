import numpy as np
import pytest

from harness import load_and_split, run_strategy
from config import FEATURE_COLS


DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"


@pytest.mark.slow
def test_feature_shuffle_noise_detection():
    """
    Stress test: model with real features should outperform the same
    model trained on fully shuffled feature columns.
    """
    _, train, test = load_and_split(DEFAULT_TICKER, DEFAULT_START, DEFAULT_END)

    real_results, real_baseline, _, _ = run_strategy(train, test)
    real_alpha = real_results["total_return"] - real_baseline["total_return"]

    rng = np.random.default_rng(42)
    train_shuffled = train.copy()
    test_shuffled = test.copy()
    for col in FEATURE_COLS:
        if col in train_shuffled and col in test_shuffled:
            train_shuffled[col] = rng.permutation(train_shuffled[col].values)
            test_shuffled[col] = rng.permutation(test_shuffled[col].values)

    shuf_results, shuf_baseline, _, _ = run_strategy(train_shuffled, test_shuffled)
    shuf_alpha = shuf_results["total_return"] - shuf_baseline["total_return"]

    assert real_alpha > 0, "Real-feature alpha is not positive; baseline strategy underperforms buy & hold."
    assert (
        real_results["total_return"] > shuf_results["total_return"]
    ), "Strategy with real features does not beat shuffled-feature noise baseline; may be overfitting."

import math

import pytest

from harness import load_and_split, run_strategy
import strategy as strategy_module


DEFAULT_TICKER = "ATGL"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"

# Sensible meta-confidence thresholds to probe robustness of the filter.
CONF_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@pytest.mark.slow
def test_meta_confidence_sensitivity():
    """
    Stress test: strategy alpha should be reasonably robust
    to changes in the meta-model confidence threshold.
    """
    _, train, test = load_and_split(DEFAULT_TICKER, DEFAULT_START, DEFAULT_END)

    results_table = []

    original_threshold = strategy_module.META_CONFIDENCE
    try:
        for thr in CONF_THRESHOLDS:
            strategy_module.META_CONFIDENCE = thr
            results, baseline, _, _ = run_strategy(train, test)
            alpha = results["total_return"] - baseline["total_return"]
            results_table.append(
                {
                    "threshold": thr,
                    "ml_return": results["total_return"],
                    "bh_return": baseline["total_return"],
                    "alpha": alpha,
                    "trades": results["num_trades"],
                }
            )
    finally:
        strategy_module.META_CONFIDENCE = original_threshold

    positive_alpha = [r for r in results_table if r["alpha"] > 0]
    pass_count = len(positive_alpha)

    required = math.ceil(len(CONF_THRESHOLDS) * 0.7)
    assert (
        pass_count >= required
    ), f"Alpha positive in only {pass_count}/{len(CONF_THRESHOLDS)} meta-thresholds; strategy is brittle."

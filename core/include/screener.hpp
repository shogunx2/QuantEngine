#pragma once
#include "types.hpp"
#include <vector>

namespace qe {

struct ScreenerConfig {
    int rsi_period = 14;
    int atr_period = 14;
    int bb_period = 20;
    double bb_std = 2.0;
    int macd_fast = 12;
    int macd_slow = 26;
    int macd_signal = 9;
    int ma_short = 5;
    int ma_long = 20;
    double rsi_oversold = 35.0;
    double rsi_overbought = 65.0;
    double min_score = 0.5;
};

SwingSignal compute_signal(const Bar* bars, size_t n, const ScreenerConfig& cfg);

std::vector<ScreenResult> screen_batch(
    const std::vector<std::vector<Bar>>& all_stocks,
    const ScreenerConfig& cfg
);

}

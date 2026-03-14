#pragma once
#include <cstddef>
#include <vector>

namespace qe {

struct Bar {
    double open;
    double high;
    double low;
    double close;
    double volume;
};

struct SwingSignal {
    double score;
    double rsi;
    double atr;
    double bb_pct;
    double macd_hist;
    double ma_trend;
    int direction;
};

struct ScreenResult {
    int stock_idx;
    SwingSignal signal;
};

}

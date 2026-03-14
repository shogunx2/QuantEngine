#include "indicators.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace qe {

void sma(const double* close, size_t n, int period, double* out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += close[i];
        if (i >= static_cast<size_t>(period)) {
            sum -= close[i - period];
        }
        if (i >= static_cast<size_t>(period - 1)) {
            out[i] = sum / period;
        } else {
            out[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

void ema(const double* close, size_t n, int period, double* out) {
    double k = 2.0 / (period + 1);
    out[0] = close[0];
    for (size_t i = 1; i < n; ++i) {
        out[i] = close[i] * k + out[i - 1] * (1.0 - k);
    }
}

void rsi(const double* close, size_t n, int period, double* out) {
    if (n < 2) return;

    double avg_gain = 0.0, avg_loss = 0.0;

    for (int i = 1; i <= period && i < static_cast<int>(n); ++i) {
        double delta = close[i] - close[i - 1];
        if (delta > 0) avg_gain += delta;
        else avg_loss -= delta;
        out[i] = std::numeric_limits<double>::quiet_NaN();
    }
    out[0] = std::numeric_limits<double>::quiet_NaN();

    avg_gain /= period;
    avg_loss /= period;

    if (static_cast<int>(n) > period) {
        double rs = (avg_loss == 0.0) ? 100.0 : avg_gain / avg_loss;
        out[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    for (size_t i = period + 1; i < n; ++i) {
        double delta = close[i] - close[i - 1];
        double gain = delta > 0 ? delta : 0.0;
        double loss = delta < 0 ? -delta : 0.0;
        avg_gain = (avg_gain * (period - 1) + gain) / period;
        avg_loss = (avg_loss * (period - 1) + loss) / period;
        double rs = (avg_loss == 0.0) ? 100.0 : avg_gain / avg_loss;
        out[i] = 100.0 - (100.0 / (1.0 + rs));
    }
}

void atr(const Bar* bars, size_t n, int period, double* out) {
    if (n < 2) return;
    out[0] = std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    for (size_t i = 1; i < n; ++i) {
        double tr = std::max({
            bars[i].high - bars[i].low,
            std::abs(bars[i].high - bars[i - 1].close),
            std::abs(bars[i].low - bars[i - 1].close)
        });
        if (i < static_cast<size_t>(period)) {
            sum += tr;
            out[i] = std::numeric_limits<double>::quiet_NaN();
        } else if (i == static_cast<size_t>(period)) {
            sum += tr;
            out[i] = sum / period;
        } else {
            out[i] = (out[i - 1] * (period - 1) + tr) / period;
        }
    }
}

void bollinger_pct(const double* close, size_t n, int period, double num_std, double* out) {
    for (size_t i = 0; i < n; ++i) {
        if (i < static_cast<size_t>(period - 1)) {
            out[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        double sum = 0.0, sq_sum = 0.0;
        for (int j = 0; j < period; ++j) {
            double v = close[i - j];
            sum += v;
            sq_sum += v * v;
        }
        double mean = sum / period;
        double var = sq_sum / period - mean * mean;
        double std = std::sqrt(std::max(var, 0.0));
        double upper = mean + num_std * std;
        double lower = mean - num_std * std;
        double range = upper - lower;
        out[i] = (range > 1e-10) ? (close[i] - lower) / range : 0.5;
    }
}

void macd_histogram(const double* close, size_t n, int fast, int slow, int signal, double* out) {
    std::vector<double> ema_fast(n), ema_slow(n), macd_line(n), sig_line(n);
    ema(close, n, fast, ema_fast.data());
    ema(close, n, slow, ema_slow.data());
    for (size_t i = 0; i < n; ++i) {
        macd_line[i] = ema_fast[i] - ema_slow[i];
    }
    ema(macd_line.data(), n, signal, sig_line.data());
    for (size_t i = 0; i < n; ++i) {
        out[i] = macd_line[i] - sig_line[i];
    }
}

void ma_trend(const double* close, size_t n, int short_p, int long_p, double* out) {
    std::vector<double> ma_short(n), ma_long(n);
    sma(close, n, short_p, ma_short.data());
    sma(close, n, long_p, ma_long.data());
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(ma_short[i]) || std::isnan(ma_long[i]) || ma_long[i] == 0.0) {
            out[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            out[i] = (ma_short[i] - ma_long[i]) / ma_long[i];
        }
    }
}

void obv(const double* close, const double* volume, size_t n, double* out) {
    if (n == 0) return;
    out[0] = volume[0];
    for (size_t i = 1; i < n; ++i) {
        if (close[i] > close[i - 1]) {
            out[i] = out[i - 1] + volume[i];
        } else if (close[i] < close[i - 1]) {
            out[i] = out[i - 1] - volume[i];
        } else {
            out[i] = out[i - 1];
        }
    }
}

}

#include "screener.hpp"
#include "indicators.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>

namespace qe {

SwingSignal compute_signal(const Bar* bars, size_t n, const ScreenerConfig& cfg) {
    SwingSignal sig{};
    if (n < 30) return sig;

    std::vector<double> closes(n);
    for (size_t i = 0; i < n; ++i) closes[i] = bars[i].close;

    std::vector<double> rsi_out(n), atr_out(n), bb_out(n), macd_out(n), trend_out(n);

    rsi(closes.data(), n, cfg.rsi_period, rsi_out.data());
    atr(bars, n, cfg.atr_period, atr_out.data());
    bollinger_pct(closes.data(), n, cfg.bb_period, cfg.bb_std, bb_out.data());
    macd_histogram(closes.data(), n, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal, macd_out.data());
    ma_trend(closes.data(), n, cfg.ma_short, cfg.ma_long, trend_out.data());

    size_t last = n - 1;
    sig.rsi = rsi_out[last];
    sig.atr = atr_out[last];
    sig.bb_pct = bb_out[last];
    sig.macd_hist = macd_out[last];
    sig.ma_trend = trend_out[last];

    if (std::isnan(sig.rsi) || std::isnan(sig.atr)) return sig;

    double score = 0.0;

    if (sig.rsi < cfg.rsi_oversold) {
        score += (cfg.rsi_oversold - sig.rsi) / cfg.rsi_oversold;
    } else if (sig.rsi > cfg.rsi_overbought) {
        score -= (sig.rsi - cfg.rsi_overbought) / (100.0 - cfg.rsi_overbought);
    }

    if (sig.bb_pct < 0.2) {
        score += (0.2 - sig.bb_pct) * 2.0;
    } else if (sig.bb_pct > 0.8) {
        score -= (sig.bb_pct - 0.8) * 2.0;
    }

    if (sig.macd_hist > 0 && !std::isnan(sig.macd_hist)) {
        double norm = sig.atr > 0 ? sig.macd_hist / sig.atr : 0;
        score += std::clamp(norm, -1.0, 1.0) * 0.5;
    } else if (sig.macd_hist < 0 && !std::isnan(sig.macd_hist)) {
        double norm = sig.atr > 0 ? sig.macd_hist / sig.atr : 0;
        score += std::clamp(norm, -1.0, 1.0) * 0.5;
    }

    if (!std::isnan(sig.ma_trend)) {
        score += std::clamp(sig.ma_trend * 10.0, -0.5, 0.5);
    }

    sig.score = score;
    sig.direction = (score > cfg.min_score) ? 1 : (score < -cfg.min_score) ? -1 : 0;

    return sig;
}

std::vector<ScreenResult> screen_batch(
    const std::vector<std::vector<Bar>>& all_stocks,
    const ScreenerConfig& cfg
) {
    size_t num_stocks = all_stocks.size();
    std::vector<ScreenResult> results(num_stocks);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& bars = all_stocks[i];
            results[i].stock_idx = static_cast<int>(i);
            results[i].signal = compute_signal(bars.data(), bars.size(), cfg);
        }
    };

    if (num_stocks <= num_threads) {
        worker(0, num_stocks);
    } else {
        std::vector<std::thread> threads;
        size_t chunk = num_stocks / num_threads;
        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk;
            size_t end = (t == num_threads - 1) ? num_stocks : start + chunk;
            threads.emplace_back(worker, start, end);
        }
        for (auto& th : threads) th.join();
    }

    std::vector<ScreenResult> filtered;
    for (auto& r : results) {
        if (r.signal.direction != 0) {
            filtered.push_back(r);
        }
    }

    std::sort(filtered.begin(), filtered.end(), [](const ScreenResult& a, const ScreenResult& b) {
        return std::abs(a.signal.score) > std::abs(b.signal.score);
    });

    return filtered;
}

}

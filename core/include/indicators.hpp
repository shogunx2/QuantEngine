#pragma once
#include "types.hpp"
#include <vector>

namespace qe {

void sma(const double* close, size_t n, int period, double* out);
void ema(const double* close, size_t n, int period, double* out);
void rsi(const double* close, size_t n, int period, double* out);
void atr(const Bar* bars, size_t n, int period, double* out);
void bollinger_pct(const double* close, size_t n, int period, double num_std, double* out);
void macd_histogram(const double* close, size_t n, int fast, int slow, int signal, double* out);
void ma_trend(const double* close, size_t n, int short_p, int long_p, double* out);
void obv(const double* close, const double* volume, size_t n, double* out);

}

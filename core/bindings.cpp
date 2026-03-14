#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "include/types.hpp"
#include "include/indicators.hpp"
#include "include/screener.hpp"
#include <vector>
#include <stdexcept>

namespace py = pybind11;

static std::vector<qe::Bar> numpy_to_bars(
    py::array_t<double> open,
    py::array_t<double> high,
    py::array_t<double> low,
    py::array_t<double> close,
    py::array_t<double> volume
) {
    auto o = open.unchecked<1>();
    auto h = high.unchecked<1>();
    auto l = low.unchecked<1>();
    auto c = close.unchecked<1>();
    auto v = volume.unchecked<1>();
    size_t n = static_cast<size_t>(o.shape(0));
    if (static_cast<size_t>(h.shape(0)) != n ||
        static_cast<size_t>(l.shape(0)) != n ||
        static_cast<size_t>(c.shape(0)) != n ||
        static_cast<size_t>(v.shape(0)) != n) {
        throw std::invalid_argument("All arrays must have the same length");
    }
    std::vector<qe::Bar> bars(n);
    for (size_t i = 0; i < n; ++i) {
        bars[i] = {o(i), h(i), l(i), c(i), v(i)};
    }
    return bars;
}

static py::array_t<double> run_indicator_close(
    py::array_t<double> close, int period,
    void (*fn)(const double*, size_t, int, double*)
) {
    auto c = close.unchecked<1>();
    size_t n = static_cast<size_t>(c.shape(0));
    py::array_t<double> out(n);
    fn(c.data(0), n, period, out.mutable_data(0));
    return out;
}

PYBIND11_MODULE(qe_core, m) {
    m.doc() = "QuantEngine C++ core";

    py::class_<qe::Bar>(m, "Bar")
        .def(py::init<>())
        .def_readwrite("open", &qe::Bar::open)
        .def_readwrite("high", &qe::Bar::high)
        .def_readwrite("low", &qe::Bar::low)
        .def_readwrite("close", &qe::Bar::close)
        .def_readwrite("volume", &qe::Bar::volume);

    py::class_<qe::SwingSignal>(m, "SwingSignal")
        .def(py::init<>())
        .def_readwrite("score", &qe::SwingSignal::score)
        .def_readwrite("rsi", &qe::SwingSignal::rsi)
        .def_readwrite("atr", &qe::SwingSignal::atr)
        .def_readwrite("bb_pct", &qe::SwingSignal::bb_pct)
        .def_readwrite("macd_hist", &qe::SwingSignal::macd_hist)
        .def_readwrite("ma_trend", &qe::SwingSignal::ma_trend)
        .def_readwrite("direction", &qe::SwingSignal::direction);

    py::class_<qe::ScreenResult>(m, "ScreenResult")
        .def(py::init<>())
        .def_readwrite("stock_idx", &qe::ScreenResult::stock_idx)
        .def_readwrite("signal", &qe::ScreenResult::signal);

    py::class_<qe::ScreenerConfig>(m, "ScreenerConfig")
        .def(py::init<>())
        .def_readwrite("rsi_period", &qe::ScreenerConfig::rsi_period)
        .def_readwrite("atr_period", &qe::ScreenerConfig::atr_period)
        .def_readwrite("bb_period", &qe::ScreenerConfig::bb_period)
        .def_readwrite("bb_std", &qe::ScreenerConfig::bb_std)
        .def_readwrite("macd_fast", &qe::ScreenerConfig::macd_fast)
        .def_readwrite("macd_slow", &qe::ScreenerConfig::macd_slow)
        .def_readwrite("macd_signal", &qe::ScreenerConfig::macd_signal)
        .def_readwrite("ma_short", &qe::ScreenerConfig::ma_short)
        .def_readwrite("ma_long", &qe::ScreenerConfig::ma_long)
        .def_readwrite("rsi_oversold", &qe::ScreenerConfig::rsi_oversold)
        .def_readwrite("rsi_overbought", &qe::ScreenerConfig::rsi_overbought)
        .def_readwrite("min_score", &qe::ScreenerConfig::min_score);

    m.def("sma", [](py::array_t<double> close, int period) {
        return run_indicator_close(close, period, qe::sma);
    }, py::arg("close"), py::arg("period"));

    m.def("ema", [](py::array_t<double> close, int period) {
        return run_indicator_close(close, period, qe::ema);
    }, py::arg("close"), py::arg("period"));

    m.def("rsi", [](py::array_t<double> close, int period) {
        return run_indicator_close(close, period, qe::rsi);
    }, py::arg("close"), py::arg("period"));

    m.def("atr", [](py::array_t<double> open, py::array_t<double> high,
                     py::array_t<double> low, py::array_t<double> close,
                     py::array_t<double> volume, int period) {
        auto bars = numpy_to_bars(open, high, low, close, volume);
        size_t n = bars.size();
        py::array_t<double> out(n);
        qe::atr(bars.data(), n, period, out.mutable_data(0));
        return out;
    }, py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"),
       py::arg("volume"), py::arg("period"));

    m.def("bollinger_pct", [](py::array_t<double> close, int period, double num_std) {
        auto c = close.unchecked<1>();
        size_t n = static_cast<size_t>(c.shape(0));
        py::array_t<double> out(n);
        qe::bollinger_pct(c.data(0), n, period, num_std, out.mutable_data(0));
        return out;
    }, py::arg("close"), py::arg("period") = 20, py::arg("num_std") = 2.0);

    m.def("macd_histogram", [](py::array_t<double> close, int fast, int slow, int signal) {
        auto c = close.unchecked<1>();
        size_t n = static_cast<size_t>(c.shape(0));
        py::array_t<double> out(n);
        qe::macd_histogram(c.data(0), n, fast, slow, signal, out.mutable_data(0));
        return out;
    }, py::arg("close"), py::arg("fast") = 12, py::arg("slow") = 26, py::arg("signal") = 9);

    m.def("obv", [](py::array_t<double> close, py::array_t<double> volume) {
        auto c = close.unchecked<1>();
        auto v = volume.unchecked<1>();
        size_t n = static_cast<size_t>(c.shape(0));
        py::array_t<double> out(n);
        qe::obv(c.data(0), v.data(0), n, out.mutable_data(0));
        return out;
    }, py::arg("close"), py::arg("volume"));

    m.def("compute_signal", [](py::array_t<double> open, py::array_t<double> high,
                                py::array_t<double> low, py::array_t<double> close,
                                py::array_t<double> volume, const qe::ScreenerConfig& cfg) {
        auto bars = numpy_to_bars(open, high, low, close, volume);
        return qe::compute_signal(bars.data(), bars.size(), cfg);
    }, py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"),
       py::arg("volume"), py::arg("config") = qe::ScreenerConfig());

    m.def("screen_batch", [](py::list stock_data, const qe::ScreenerConfig& cfg) {
        std::vector<std::vector<qe::Bar>> all_stocks;
        all_stocks.reserve(py::len(stock_data));
        for (auto item : stock_data) {
            auto tup = item.cast<py::tuple>();
            auto o = tup[0].cast<py::array_t<double>>();
            auto h = tup[1].cast<py::array_t<double>>();
            auto l = tup[2].cast<py::array_t<double>>();
            auto c = tup[3].cast<py::array_t<double>>();
            auto v = tup[4].cast<py::array_t<double>>();
            all_stocks.push_back(numpy_to_bars(o, h, l, c, v));
        }
        return qe::screen_batch(all_stocks, cfg);
    }, py::arg("stock_data"), py::arg("config") = qe::ScreenerConfig());
}

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import yfinance as yf
import pandas as pd
import numpy as np
import qe_core
from config import SWING_HORIZON, TP_ATR_MULT, SL_ATR_MULT, ATR_PERIOD


class DataLoader:
    def __init__(self, ticker, start, end):
        self.ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        self.start = start
        self.end = end
        self.data = None

    def _fetch_single(self, ticker):
        raw = yf.download(ticker, start=self.start, end=self.end, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    def fetch(self):
        raw = self._fetch_single(self.ticker)
        if raw.empty:
            raise ValueError(f"No data returned for {self.ticker}")
        self.data = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        vix = self._fetch_single("^INDIAVIX")
        nifty = self._fetch_single("^NSEI")

        if not vix.empty:
            self.data["VIX"] = vix["Close"]
        else:
            self.data["VIX"] = np.nan

        if not nifty.empty:
            self.data["Nifty_Close"] = nifty["Close"]
        else:
            self.data["Nifty_Close"] = np.nan

        self._compute_features()
        return self.data

    def _compute_features(self):
        df = self.data

        close = df["Close"].values.astype(np.float64)
        open_ = df["Open"].values.astype(np.float64)
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        volume = df["Volume"].values.astype(np.float64)

        df["Return_1d"] = df["Close"].pct_change()
        df["Return_5d"] = df["Close"].pct_change(5)
        for w in [5, 10, 20]:
            df[f"MA_{w}_ratio"] = df["Close"] / df["Close"].rolling(w).mean()

        rsi_arr = qe_core.rsi(close, 14)
        df["RSI_14"] = rsi_arr

        df["Volume_change"] = df["Volume"].pct_change()
        df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
        df["VIX_above_18"] = (df["VIX"] > 18).astype(int)
        df["Nifty_Return_5d"] = df["Nifty_Close"].pct_change(5)
        df["Nifty_MA_20_ratio"] = df["Nifty_Close"] / df["Nifty_Close"].rolling(20).mean()

        sma_200 = qe_core.sma(close, 200)
        df["Regime_stock"] = (close >= sma_200).astype(int)

        nifty_close = df["Nifty_Close"].values.astype(np.float64)
        nifty_sma_50 = qe_core.sma(nifty_close, 50)
        df["Regime_nifty"] = (nifty_close >= nifty_sma_50).astype(int)

        df["Regime_ok"] = (df["Regime_stock"] & df["Regime_nifty"]).astype(int)

        df.drop(columns=["Nifty_Close"], inplace=True)

        bb_arr = qe_core.bollinger_pct(close, 20, 2.0)
        df["BB_pct"] = bb_arr

        macd_arr = qe_core.macd_histogram(close, 12, 26, 9)
        df["MACD_hist"] = macd_arr

        sma_5 = qe_core.sma(close, 5)
        sma_20 = qe_core.sma(close, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            ma_t = np.where(sma_20 != 0, (sma_5 - sma_20) / sma_20, np.nan)
        df["MA_trend"] = ma_t

        cfg = qe_core.ScreenerConfig()

        rsi_vals = np.array(rsi_arr)
        bb_vals = np.array(bb_arr)
        macd_vals = np.array(macd_arr)
        ma_t_vals = np.array(df["MA_trend"].values)
        atr_arr_full = qe_core.atr(open_, high, low, close, volume, ATR_PERIOD)
        atr_vals = np.array(atr_arr_full)

        scores = np.zeros(len(close))
        for i in range(len(close)):
            r, b, m, mt, a = rsi_vals[i], bb_vals[i], macd_vals[i], ma_t_vals[i], atr_vals[i]
            if np.isnan(r) or np.isnan(a):
                scores[i] = np.nan
                continue
            s = 0.0
            if r < cfg.rsi_oversold:
                s += (cfg.rsi_oversold - r) / cfg.rsi_oversold
            elif r > cfg.rsi_overbought:
                s -= (r - cfg.rsi_overbought) / (100.0 - cfg.rsi_overbought)
            if b < 0.2:
                s += (0.2 - b) * 2.0
            elif b > 0.8:
                s -= (b - 0.8) * 2.0
            if not np.isnan(m) and a > 0:
                norm = np.clip(m / a, -1.0, 1.0) * 0.5
                s += norm
            if not np.isnan(mt):
                s += np.clip(mt * 10.0, -0.5, 0.5)
            scores[i] = s
        df["CPP_score"] = scores

        obv_arr = qe_core.obv(close, volume)
        obv_sma = qe_core.sma(obv_arr, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["OBV_ratio"] = np.where(obv_sma != 0, obv_arr / obv_sma, np.nan)

        df["ATR"] = atr_vals

        self._compute_triple_barrier(df, atr_vals)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    def _compute_triple_barrier(self, df, atr_arr):
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        n = len(close)
        labels = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
                continue
            entry = close[i]
            tp = entry + TP_ATR_MULT * atr_arr[i]
            sl = entry - SL_ATR_MULT * atr_arr[i]
            end_j = min(i + SWING_HORIZON, n - 1)

            hit = 0
            for j in range(i + 1, end_j + 1):
                if high[j] >= tp:
                    hit = 1
                    break
                if low[j] <= sl:
                    hit = -1
                    break

            if hit == 0:
                ret = (close[end_j] - entry) / entry
                hit = 1 if ret > 0 else (-1 if ret < 0 else 0)

            labels[i] = hit

        df["Target"] = labels
        df["TB_return"] = np.nan
        for i in range(n):
            if np.isnan(labels[i]):
                continue
            entry = close[i]
            tp = entry + TP_ATR_MULT * atr_arr[i]
            sl = entry - SL_ATR_MULT * atr_arr[i]
            end_j = min(i + SWING_HORIZON, n - 1)
            for j in range(i + 1, end_j + 1):
                if high[j] >= tp:
                    df.iloc[i, df.columns.get_loc("TB_return")] = (tp - entry) / entry
                    break
                if low[j] <= sl:
                    df.iloc[i, df.columns.get_loc("TB_return")] = (sl - entry) / entry
                    break
            else:
                df.iloc[i, df.columns.get_loc("TB_return")] = (close[end_j] - entry) / entry

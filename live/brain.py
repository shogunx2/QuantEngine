import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

import logging
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import qe_core
from config import (
    FEATURE_COLS, MIN_SIGNAL_THRESHOLD, SWING_HORIZON,
    N_ESTIMATORS, RANDOM_STATE, TRAIN_RATIO,
)
from strategy import MLStrategy
from config_live import HISTORY_DAYS, MODEL_DIR

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(MODEL_DIR, "live_model.pkl")


class Brain:
    def __init__(self):
        self.model = None
        self._hist_cache = {}
        self._nifty_cache = None
        self._vix_cache = None

    def load_or_train(self, symbols):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            log.info(f"Loaded pre-trained model from {MODEL_PATH}")
            self._warm_history(symbols)
            return

        log.info("No saved model — training from scratch on watchlist history")
        from data_loader import DataLoader
        from datetime import datetime, timedelta

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=int(HISTORY_DAYS * 2))).strftime("%Y-%m-%d")

        all_train = []
        for sym in symbols:
            try:
                loader = DataLoader(sym, start, end)
                data = loader.fetch()
                split = int(len(data) * TRAIN_RATIO)
                all_train.append(data.iloc[:split])
                log.info(f"  {sym}: {len(data)} days loaded, {split} for training")
            except Exception as e:
                log.warning(f"  {sym}: skipped — {e}")

        if not all_train:
            raise RuntimeError("No training data available for any symbol")

        combined = pd.concat(all_train, ignore_index=True)
        self.model = MLStrategy()
        self.model.fit(combined)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        log.info(f"Model trained on {len(combined)} rows, saved to {MODEL_PATH}")
        self._warm_history(symbols)

    def _warm_history(self, symbols):
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=HISTORY_DAYS + 50)).strftime("%Y-%m-%d")

        for sym in symbols:
            ticker = sym if sym.endswith(".NS") else f"{sym}.NS"
            try:
                raw = yf.download(ticker, start=start, end=end, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if not raw.empty:
                    self._hist_cache[sym] = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                    log.info(f"  History warm: {sym} — {len(raw)} days")
            except Exception as e:
                log.warning(f"  History warm failed for {sym}: {e}")

        try:
            vix = yf.download("^INDIAVIX", start=start, end=end, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if not vix.empty:
                self._vix_cache = vix[["Close"]].rename(columns={"Close": "VIX"})
        except Exception:
            pass

        try:
            nifty = yf.download("^NSEI", start=start, end=end, progress=False)
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            if not nifty.empty:
                self._nifty_cache = nifty[["Close"]].rename(columns={"Close": "Nifty_Close"})
        except Exception:
            pass

    def evaluate(self, symbol, live_candle):
        hist = self._hist_cache.get(symbol)
        if hist is None or len(hist) < 201:
            log.warning(f"{symbol}: insufficient history ({0 if hist is None else len(hist)} days)")
            return None

        today = pd.DataFrame(
            [live_candle.to_dict()],
            index=[pd.Timestamp.now().normalize()],
        )
        df = pd.concat([hist, today])

        if self._vix_cache is not None:
            df = df.join(self._vix_cache, how="left")
            df["VIX"] = df["VIX"].ffill()
        else:
            df["VIX"] = np.nan

        if self._nifty_cache is not None:
            df = df.join(self._nifty_cache, how="left")
            df["Nifty_Close"] = df["Nifty_Close"].ffill()
        else:
            df["Nifty_Close"] = np.nan

        close = df["Close"].values.astype(np.float64)
        open_ = df["Open"].values.astype(np.float64)
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        volume = df["Volume"].values.astype(np.float64)

        row = {}
        row["Return_1d"] = (close[-1] / close[-2] - 1) if len(close) >= 2 else 0.0
        row["Return_5d"] = (close[-1] / close[-6] - 1) if len(close) >= 6 else 0.0

        for w in [5, 10, 20]:
            sma_w = qe_core.sma(close, w)
            row[f"MA_{w}_ratio"] = close[-1] / sma_w[-1] if sma_w[-1] != 0 else 1.0

        rsi_arr = qe_core.rsi(close, 14)
        row["RSI_14"] = rsi_arr[-1]

        row["Volume_change"] = (volume[-1] / volume[-2] - 1) if len(volume) >= 2 and volume[-2] != 0 else 0.0

        returns = np.diff(close) / close[:-1]
        row["Volatility_10d"] = float(np.std(returns[-10:])) if len(returns) >= 10 else 0.0

        vix_vals = df["VIX"].values
        row["VIX"] = float(vix_vals[-1]) if not np.isnan(vix_vals[-1]) else 15.0
        row["VIX_above_18"] = 1 if row["VIX"] > 18 else 0

        nifty_vals = df["Nifty_Close"].values
        if not np.isnan(nifty_vals[-1]) and not np.isnan(nifty_vals[-6]):
            row["Nifty_Return_5d"] = nifty_vals[-1] / nifty_vals[-6] - 1
        else:
            row["Nifty_Return_5d"] = 0.0

        nifty_clean = nifty_vals[~np.isnan(nifty_vals)].astype(np.float64)
        if len(nifty_clean) >= 20:
            nifty_sma_20 = qe_core.sma(nifty_clean, 20)
            row["Nifty_MA_20_ratio"] = nifty_clean[-1] / nifty_sma_20[-1] if nifty_sma_20[-1] != 0 else 1.0
        else:
            row["Nifty_MA_20_ratio"] = 1.0

        bb_arr = qe_core.bollinger_pct(close, 20, 2.0)
        row["BB_pct"] = bb_arr[-1]

        macd_arr = qe_core.macd_histogram(close, 12, 26, 9)
        row["MACD_hist"] = macd_arr[-1]

        sma_5 = qe_core.sma(close, 5)
        sma_20 = qe_core.sma(close, 20)
        row["MA_trend"] = (sma_5[-1] - sma_20[-1]) / sma_20[-1] if sma_20[-1] != 0 else 0.0

        cfg = qe_core.ScreenerConfig()
        cpp_sig = qe_core.compute_signal(open_, high, low, close, volume, cfg)
        row["CPP_score"] = cpp_sig.score

        sma_200 = qe_core.sma(close, 200)
        regime_stock = 1 if close[-1] >= sma_200[-1] else 0

        regime_nifty = 1
        if len(nifty_clean) >= 50:
            nifty_sma_50 = qe_core.sma(nifty_clean, 50)
            regime_nifty = 1 if nifty_clean[-1] >= nifty_sma_50[-1] else 0

        regime_ok = regime_stock & regime_nifty

        for col in FEATURE_COLS:
            if col not in row:
                row[col] = 0.0

        feature_row = pd.DataFrame([row])[FEATURE_COLS]
        prediction = self.model.predict(feature_row)[0]

        signal_raw = 1 if prediction > MIN_SIGNAL_THRESHOLD else (-1 if prediction < -MIN_SIGNAL_THRESHOLD else 0)
        signal = 0 if (signal_raw == 1 and regime_ok == 0) else signal_raw

        return {
            "symbol": symbol,
            "prediction": prediction,
            "signal_raw": signal_raw,
            "signal": signal,
            "regime_stock": regime_stock,
            "regime_nifty": regime_nifty,
            "regime_ok": regime_ok,
            "rsi": row["RSI_14"],
            "bb_pct": row["BB_pct"],
            "cpp_score": row["CPP_score"],
            "ltp": close[-1],
        }

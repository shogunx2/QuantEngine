import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger(__name__)


class Candle:
    __slots__ = ("open", "high", "low", "close", "volume", "timestamp")

    def __init__(self, open_=0.0, high=0.0, low=0.0, close=0.0, volume=0, timestamp=None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "Volume": self.volume,
        }

    def __repr__(self):
        return (
            f"Candle(O={self.open:.2f} H={self.high:.2f} "
            f"L={self.low:.2f} C={self.close:.2f} V={self.volume})"
        )


class Aggregator:
    def __init__(self):
        self._lock = threading.Lock()
        self._daily = {}
        self._minute = defaultdict(dict)

    def on_tick(self, tick):
        sym = tick.symbol
        ltp = tick.ltp
        vol = tick.volume
        ts = tick.timestamp

        with self._lock:
            if sym not in self._daily:
                self._daily[sym] = Candle(
                    open_=ltp, high=ltp, low=ltp, close=ltp,
                    volume=vol, timestamp=ts,
                )
            else:
                c = self._daily[sym]
                c.high = max(c.high, ltp)
                c.low = min(c.low, ltp)
                c.close = ltp
                c.volume = vol
                c.timestamp = ts

            minute_key = ts.replace(second=0, microsecond=0) if isinstance(ts, datetime) else ts
            if minute_key not in self._minute[sym]:
                self._minute[sym][minute_key] = Candle(
                    open_=ltp, high=ltp, low=ltp, close=ltp,
                    volume=vol, timestamp=minute_key,
                )
            else:
                mc = self._minute[sym][minute_key]
                mc.high = max(mc.high, ltp)
                mc.low = min(mc.low, ltp)
                mc.close = ltp
                mc.volume = vol

    def get_daily_candle(self, symbol):
        with self._lock:
            return self._daily.get(symbol)

    def get_all_daily(self):
        with self._lock:
            return dict(self._daily)

    def get_minute_candles(self, symbol):
        with self._lock:
            bars = self._minute.get(symbol, {})
            return [bars[k] for k in sorted(bars.keys())]

    def reset_day(self):
        with self._lock:
            self._daily.clear()
            self._minute.clear()
            log.info("Aggregator reset for new trading day")

    def tick_count(self):
        with self._lock:
            return sum(1 for sym in self._daily if self._daily[sym].volume > 0)

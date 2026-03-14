import abc
import time
import random
import threading
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class Tick:
    __slots__ = ("symbol", "token", "ltp", "volume", "timestamp")

    def __init__(self, symbol, token, ltp, volume, timestamp):
        self.symbol = symbol
        self.token = token
        self.ltp = ltp
        self.volume = volume
        self.timestamp = timestamp

    def __repr__(self):
        return f"Tick({self.symbol} {self.ltp:.2f} vol={self.volume} @ {self.timestamp})"


class BaseStreamer(abc.ABC):
    def __init__(self, watchlist, on_tick):
        self.watchlist = watchlist
        self.on_tick = on_tick
        self._running = False

    @abc.abstractmethod
    def connect(self):
        ...

    @abc.abstractmethod
    def disconnect(self):
        ...


class AngelStreamer(BaseStreamer):
    def connect(self):
        from SmartApi import SmartConnect
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        import pyotp
        from config_live import ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET

        smart = SmartConnect(api_key=ANGEL_API_KEY)
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        session = smart.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if session["status"] is False:
            raise ConnectionError(f"Angel login failed: {session}")

        auth_token = session["data"]["jwtToken"]
        feed_token = smart.getfeedToken()

        token_map = {w["token"]: w["symbol"] for w in self.watchlist}
        token_list = [
            {"exchangeType": 1, "tokens": [w["token"] for w in self.watchlist]}
        ]

        self._ws = SmartWebSocketV2(
            auth_token, ANGEL_API_KEY, ANGEL_CLIENT_ID, feed_token
        )

        def on_data(wsapp, msg):
            tok = str(msg.get("token", ""))
            sym = token_map.get(tok, tok)
            tick = Tick(
                symbol=sym,
                token=tok,
                ltp=msg.get("last_traded_price", 0) / 100.0,
                volume=msg.get("volume_trade_for_the_day", 0),
                timestamp=datetime.now(),
            )
            self.on_tick(tick)

        def on_open(wsapp):
            log.info("Angel WebSocket connected")
            self._ws.subscribe("abc", 1, token_list)

        def on_error(wsapp, error):
            log.error(f"Angel WS error: {error}")

        def on_close(wsapp):
            log.info("Angel WebSocket closed")

        self._ws.on_data = on_data
        self._ws.on_open = on_open
        self._ws.on_error = on_error
        self._ws.on_close = on_close

        self._running = True
        self._ws.connect()

    def disconnect(self):
        self._running = False
        if hasattr(self, "_ws"):
            self._ws.close_connection()


class DhanStreamer(BaseStreamer):
    def connect(self):
        from dhanhq import marketfeed
        from config_live import DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID

        token_map = {}
        instruments = []
        for w in self.watchlist:
            sec_id = int(w["token"])
            token_map[sec_id] = w["symbol"]
            instruments.append((0, str(sec_id), marketfeed.Ticker))

        self._feed = marketfeed.DhanFeed(
            DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, instruments
        )

        self._running = True

        def poll():
            self._feed.run_forever()
            while self._running:
                try:
                    resp = self._feed.get_data()
                    if resp and "LTP" in resp:
                        sec_id = resp.get("security_id", 0)
                        sym = token_map.get(int(sec_id), str(sec_id))
                        tick = Tick(
                            symbol=sym,
                            token=str(sec_id),
                            ltp=float(resp["LTP"]),
                            volume=int(resp.get("volume", 0)),
                            timestamp=datetime.now(),
                        )
                        self.on_tick(tick)
                except Exception as e:
                    log.error(f"Dhan feed error: {e}")
                time.sleep(0.5)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def disconnect(self):
        self._running = False
        if hasattr(self, "_feed"):
            self._feed.disconnect()


class SimulatedStreamer(BaseStreamer):
    def __init__(self, watchlist, on_tick, speed=1.0):
        super().__init__(watchlist, on_tick)
        self._speed = speed
        self._thread = None

    def connect(self):
        import yfinance as yf

        prices = {}
        for w in self.watchlist:
            sym = w["symbol"]
            ticker = sym if sym.endswith(".NS") else f"{sym}.NS"
            hist = yf.download(ticker, period="5d", interval="1m", progress=False)
            if hasattr(hist.columns, "get_level_values"):
                hist.columns = hist.columns.get_level_values(0)
            if not hist.empty:
                prices[sym] = hist
                log.info(f"Loaded {len(hist)} 1-min bars for {sym}")

        if not prices:
            raise ValueError("No historical 1-min data for simulation")

        self._running = True

        def replay():
            all_times = set()
            for df in prices.values():
                all_times.update(df.index.tolist())
            all_times = sorted(all_times)

            for ts in all_times:
                if not self._running:
                    break
                for sym, df in prices.items():
                    if ts in df.index:
                        row = df.loc[ts]
                        tick = Tick(
                            symbol=sym,
                            token=next(
                                (w["token"] for w in self.watchlist if w["symbol"] == sym), ""
                            ),
                            ltp=float(row["Close"]),
                            volume=int(row["Volume"]),
                            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        )
                        self.on_tick(tick)
                time.sleep(0.05 / self._speed)
            log.info("Simulation replay finished")

        self._thread = threading.Thread(target=replay, daemon=True)
        self._thread.start()

    def disconnect(self):
        self._running = False


def create_streamer(broker, watchlist, on_tick):
    if broker == "angel":
        return AngelStreamer(watchlist, on_tick)
    elif broker == "dhan":
        return DhanStreamer(watchlist, on_tick)
    else:
        return SimulatedStreamer(watchlist, on_tick, speed=10.0)

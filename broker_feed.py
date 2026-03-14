"""
Price feed abstraction.

- YFinanceFeed: ~15-min delayed for NSE. Fine for EOD, dangerous for intraday.
- AngelOneFeed: Real-time via Angel One SmartAPI WebSocket.
- DhanFeed: Real-time via Dhan HTTP API.

Set PRICE_FEED in config.py and provide API credentials via environment variables.
"""

import random
import threading
import time
import requests
import yfinance as yf
from config import (
    PRICE_FEED,
    ANGELONE_API_KEY, ANGELONE_CLIENT_ID, ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET,
    ANGELONE_WS_STALE_SECS, ANGELONE_WS_RECONNECT_MIN_SECS, ANGELONE_WS_RECONNECT_MAX_SECS,
    ANGELONE_WS_FAILBACK_TICKS, ANGELONE_WS_FALLBACK_DWELL_SECS,
    DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID,
)


class YFinanceFeed:
    """~15 minute delayed prices from Yahoo Finance. Paper-trading only."""

    DELAY_WARNING = (
        "WARNING: yfinance NSE data is ~15 min delayed. "
        "Stop-losses may fire late. Use a broker feed for production."
    )

    def __init__(self):
        self._warned = False

    def fetch(self, ticker):
        if not self._warned:
            print(f"\n  {self.DELAY_WARNING}")
            self._warned = True

        t = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        tk = yf.Ticker(t)
        data = tk.history(period="1d", interval="1m")
        if data.empty:
            return None
        return {
            "time": data.index[-1].strftime("%H:%M"),
            "open": float(data["Open"].iloc[0]),
            "high": float(data["High"].max()),
            "low": float(data["Low"].min()),
            "last": float(data.iloc[-1]["Close"]),
            "delayed": True,
        }


class AngelOneFeed:
    """
    Real-time prices via Angel One SmartAPI.

    Requires: pip install smartapi-python pyotp
    Set env vars: ANGELONE_API_KEY, ANGELONE_CLIENT_ID,
                  ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET
    """

    def __init__(self):
        if not ANGELONE_API_KEY:
            raise RuntimeError(
                "Angel One credentials not set. "
                "Export ANGELONE_API_KEY, ANGELONE_CLIENT_ID, "
                "ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET."
            )
        from SmartApi import SmartConnect
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        import pyotp

        self._SmartWebSocketV2 = SmartWebSocketV2
        self._pyotp = pyotp
        self.obj = SmartConnect(api_key=ANGELONE_API_KEY)
        self._login()

        self._token_map = {}
        self._build_token_map()

        self._lock = threading.Lock()
        self._mode = "HTTP_FALLBACK"
        self._mode_reason = "init"
        self._last_mode_switch = time.time()
        self._ws_connected = False
        self._ws_stable_ticks = 0
        self._last_ws_tick_ts = 0.0
        self._tick_cache = {}
        self._subscribed_tokens = set()
        self._ws_live_tokens = set()
        self._ws = None

        self._start_ws_manager()

    def _login(self):
        totp = self._pyotp.TOTP(ANGELONE_TOTP_SECRET).now()
        session = self.obj.generateSession(ANGELONE_CLIENT_ID, ANGELONE_PASSWORD, totp)
        if not session or not session.get("status"):
            raise RuntimeError(f"Angel login failed: {session}")
        data = session.get("data") or {}
        self._auth_token = data.get("jwtToken")
        self._feed_token = self.obj.getfeedToken()
        if not self._auth_token or not self._feed_token:
            raise RuntimeError("Angel login succeeded but jwt/feed token was missing.")

    def _build_ws_instance(self):
        ws = self._SmartWebSocketV2(
            auth_token=self._auth_token,
            api_key=ANGELONE_API_KEY,
            client_code=ANGELONE_CLIENT_ID,
            feed_token=self._feed_token,
            max_retry_attempt=1,
            retry_delay=1,
            retry_multiplier=1,
            retry_duration=5,
        )
        ws.on_open = self._on_ws_open
        ws.on_data = self._on_ws_data
        ws.on_error = self._on_ws_error
        ws.on_close = self._on_ws_close
        return ws

    def _switch_mode(self, new_mode, reason):
        with self._lock:
            if self._mode != new_mode:
                self._mode = new_mode
                self._mode_reason = reason
                self._last_mode_switch = time.time()
                print(f"  AngelOne mode -> {new_mode} ({reason})")
            else:
                self._mode_reason = reason

    def _start_ws_manager(self):
        manager = threading.Thread(target=self._ws_manager_loop, daemon=True)
        manager.start()

    def _ws_manager_loop(self):
        retry = 0
        while True:
            try:
                self._login()
                ws = self._build_ws_instance()
                with self._lock:
                    self._ws = ws
                ws.connect()
                retry = 0
            except Exception as e:
                self._switch_mode("HTTP_FALLBACK", f"ws_connect_error:{e}")
                with self._lock:
                    self._ws_connected = False
                retry += 1
                base = min(
                    ANGELONE_WS_RECONNECT_MAX_SECS,
                    ANGELONE_WS_RECONNECT_MIN_SECS * (2 ** max(0, retry - 1)),
                )
                sleep_for = min(ANGELONE_WS_RECONNECT_MAX_SECS, base + random.uniform(0, 0.4))
                time.sleep(max(ANGELONE_WS_RECONNECT_MIN_SECS, sleep_for))

    def _subscribe_tokens(self, tokens):
        ws = None
        with self._lock:
            ws = self._ws
            is_connected = self._ws_connected
        if not ws or not is_connected or not tokens:
            return
        token_list = [{"exchangeType": 1, "tokens": sorted(tokens)}]
        ws.subscribe("qe_ltp", ws.LTP_MODE, token_list)

    def _on_ws_open(self, _wsapp):
        with self._lock:
            self._ws_connected = True
            self._ws_live_tokens = set()
            tokens = set(self._subscribed_tokens)
        if tokens:
            try:
                self._subscribe_tokens(tokens)
                with self._lock:
                    self._ws_live_tokens.update(tokens)
            except Exception as e:
                self._switch_mode("HTTP_FALLBACK", f"ws_subscribe_error:{e}")

    def _on_ws_data(self, _wsapp, data):
        token = str(data.get("token") or "").strip()
        raw_ltp = data.get("last_traded_price")
        if not token or raw_ltp is None:
            return

        price = float(raw_ltp) / 100.0
        now = time.time()
        with self._lock:
            prev = self._tick_cache.get(token)
            day_open = prev["open"] if prev else price
            day_high = max(prev["high"], price) if prev else price
            day_low = min(prev["low"], price) if prev else price
            self._tick_cache[token] = {
                "time": "",
                "open": round(day_open, 2),
                "high": round(day_high, 2),
                "low": round(day_low, 2),
                "last": round(price, 2),
                "delayed": False,
                "ts": now,
            }
            self._last_ws_tick_ts = now
            self._ws_stable_ticks += 1

            can_failback = (
                self._mode == "HTTP_FALLBACK"
                and (now - self._last_mode_switch) >= ANGELONE_WS_FALLBACK_DWELL_SECS
                and self._ws_stable_ticks >= ANGELONE_WS_FAILBACK_TICKS
            )
        if can_failback:
            self._switch_mode("WS_PRIMARY", "ws_stable")

    def _on_ws_error(self, _wsapp, error):
        with self._lock:
            self._ws_connected = False
            self._ws_stable_ticks = 0
        self._switch_mode("HTTP_FALLBACK", f"ws_error:{error}")

    def _on_ws_close(self, _wsapp):
        with self._lock:
            self._ws_connected = False
            self._ws_stable_ticks = 0
            self._ws_live_tokens = set()
        self._switch_mode("HTTP_FALLBACK", "ws_closed")

    def _read_ws_tick(self, token):
        with self._lock:
            tick = self._tick_cache.get(token)
            mode = self._mode
            ws_connected = self._ws_connected
        if mode != "WS_PRIMARY" or not ws_connected or not tick:
            return None
        if (time.time() - tick["ts"]) > ANGELONE_WS_STALE_SECS:
            self._switch_mode("HTTP_FALLBACK", "ws_stale")
            return None
        out = dict(tick)
        out.pop("ts", None)
        return out

    def _build_token_map(self):
        """Build symbol -> token map for NSE cash equities from Angel instrument master."""
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            instruments = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to download Angel instrument master: {e}")

        token_map = {}
        for item in instruments:
            if item.get("exch_seg") != "NSE":
                continue

            symbol = (item.get("symbol") or "").upper()
            token = str(item.get("token") or "").strip()
            if not symbol.endswith("-EQ") or not token:
                continue

            base_symbol = symbol[:-3]
            token_map[base_symbol] = token

        if not token_map:
            raise RuntimeError("Angel instrument master parsed, but no NSE -EQ tokens were found.")

        self._token_map = token_map

    def _lookup_token(self, ticker):
        key = ticker.replace(".NS", "").upper()
        token = self._token_map.get(key)
        if token:
            return token
        raise KeyError(f"Angel symbol token not found for {key}-EQ")

    def _fetch_http(self, symbol, token):
        ltp_data = self.obj.ltpData("NSE", symbol + "-EQ", token)
        if ltp_data and ltp_data.get("status"):
            d = ltp_data["data"]
            return {
                "time": "",
                "open": float(d.get("open", d["ltp"])),
                "high": float(d.get("high", d["ltp"])),
                "low": float(d.get("low", d["ltp"])),
                "last": float(d["ltp"]),
                "delayed": False,
            }
        return None

    def fetch(self, ticker):
        symbol = ticker.replace(".NS", "").upper()
        try:
            token = self._lookup_token(symbol)

            with self._lock:
                self._subscribed_tokens.add(token)
                ws_connected = self._ws_connected
                already_live = token in self._ws_live_tokens

            if ws_connected and not already_live:
                try:
                    self._subscribe_tokens({token})
                    with self._lock:
                        self._ws_live_tokens.add(token)
                except Exception:
                    pass

            ws_tick = self._read_ws_tick(token)
            if ws_tick:
                return ws_tick

            return self._fetch_http(symbol, token)
        except Exception as e:
            print(f"  AngelOne fetch error for {symbol}: {e}")
        return None


class DhanFeed:
    """
    Real-time prices via Dhan HTTP API.

    Requires: pip install dhanhq
    Set env vars: DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID
    """

    def __init__(self):
        if not DHAN_ACCESS_TOKEN:
            raise RuntimeError(
                "Dhan credentials not set. "
                "Export DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID."
            )
        from dhanhq import dhanhq
        self.dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)

    def fetch(self, ticker):
        symbol = ticker.replace(".NS", "")
        try:
            resp = self.dhan.intraday_daily_minute_charts(
                security_id=symbol,
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
            )
            if resp and resp.get("status") == "success":
                candles = resp["data"]
                last = candles[-1]
                return {
                    "time": last.get("start_Time", ""),
                    "open": float(candles[0]["open"]),
                    "high": float(max(c["high"] for c in candles)),
                    "low": float(min(c["low"] for c in candles)),
                    "last": float(last["close"]),
                    "delayed": False,
                }
        except Exception as e:
            print(f"  Dhan fetch error for {symbol}: {e}")
        return None


def get_feed():
    """Return the configured price feed instance."""
    feed_name = PRICE_FEED.lower()
    if feed_name == "angelone":
        return AngelOneFeed()
    elif feed_name == "dhan":
        return DhanFeed()
    else:
        return YFinanceFeed()

import os


def _load_dotenv_if_present() -> None:
    """Load KEY=VALUE pairs from .env if present, without overriding existing env vars."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_if_present()

BROKERAGE_RATE = 0.0005
BROKERAGE_CAP = 20.0
SLIPPAGE_RATE = 0.001
STT_RATE = 0.001
DP_CHARGE = 20.0
MIN_SIGNAL_THRESHOLD = 0.03
INITIAL_CAPITAL = 20000.0
TRAIN_RATIO = 0.8
N_ESTIMATORS = 200
RANDOM_STATE = 42

SWING_HORIZON = 5
MIN_HOLD_DAYS = 2
MAX_HOLD_DAYS = 5

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5
TRAIL_ATR_MULT = 1.0
ATR_PERIOD = 14

META_CONFIDENCE = 0.50

# --- Price Feed ---
# "yfinance" (15-min delayed, paper-trading only)
# "angelone" or "dhan" (real-time, requires API keys below)
PRICE_FEED = os.environ.get("PRICE_FEED", "angelone")

# Angel One SmartAPI credentials (set via env vars for security)
ANGELONE_API_KEY = os.environ.get("ANGELONE_API_KEY", "")
ANGELONE_CLIENT_ID = os.environ.get("ANGELONE_CLIENT_ID", "")
ANGELONE_PASSWORD = os.environ.get("ANGELONE_PASSWORD", "")
ANGELONE_TOTP_SECRET = os.environ.get("ANGELONE_TOTP_SECRET", "")

# Angel One feed resilience knobs
ANGELONE_WS_STALE_SECS = float(os.environ.get("ANGELONE_WS_STALE_SECS", "3"))
ANGELONE_WS_RECONNECT_MIN_SECS = float(os.environ.get("ANGELONE_WS_RECONNECT_MIN_SECS", "1"))
ANGELONE_WS_RECONNECT_MAX_SECS = float(os.environ.get("ANGELONE_WS_RECONNECT_MAX_SECS", "10"))
ANGELONE_WS_FAILBACK_TICKS = int(os.environ.get("ANGELONE_WS_FAILBACK_TICKS", "3"))
ANGELONE_WS_FALLBACK_DWELL_SECS = float(os.environ.get("ANGELONE_WS_FALLBACK_DWELL_SECS", "15"))

# Dhan API credentials
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")

# --- EOD Timing ---
# NSE closing price = VWAP of last 30 min, not settled until ~3:40-3:45 PM.
# EOD scripts must run AFTER this time for accurate daily candles.
EOD_SAFE_HOUR = 15
EOD_SAFE_MINUTE = 45

PURGE_EMBARGO_DAYS = 5
CV_SPLITS = 5

FEATURE_COLS = [
    "Return_5d",
    "MA_5_ratio",
    "MA_10_ratio",
    "MA_20_ratio",
    "RSI_14",
    "Volume_change",
    "Volatility_10d",
    "VIX",
    "Nifty_Return_5d",
    "Nifty_MA_20_ratio",
    "BB_pct",
    "MACD_hist",
    "MA_trend",
    "OBV_ratio",
    "CPP_score",
    "Regime_ok",
]

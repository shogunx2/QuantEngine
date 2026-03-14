import os

BROKER = os.environ.get("QE_BROKER", "simulate")

ANGEL_API_KEY = os.environ.get("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD = os.environ.get("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET", "")

DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")

WATCHLIST = [
    {"symbol": "ADANIENT", "token": "25", "exchange": "NSE"},
    {"symbol": "ATGL", "token": "10794", "exchange": "NSE"},
    {"symbol": "RELIANCE", "token": "2885", "exchange": "NSE"},
    {"symbol": "HDFCBANK", "token": "1333", "exchange": "NSE"},
    {"symbol": "TATASTEEL", "token": "3499", "exchange": "NSE"},
]

NIFTY_TOKEN = "99926000"
NIFTY_SYMBOL = "NIFTY 50"

MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
DECISION_H, DECISION_M = 15, 15

HISTORY_DAYS = 250

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

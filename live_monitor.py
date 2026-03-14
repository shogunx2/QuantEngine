import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import json
import time
import pandas as pd
from datetime import datetime, timedelta
from config import (
    TP_ATR_MULT, SL_ATR_MULT, TRAIL_ATR_MULT, MAX_HOLD_DAYS, INITIAL_CAPITAL,
)
from broker_feed import get_feed

LEDGER_PATH = "paper_trades/ledger.json"
POLL_INTERVAL = 60
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)


def load_ledger():
    with open(LEDGER_PATH, "r") as f:
        return json.load(f)


def save_ledger(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


# Price feed is created once at module level via config.PRICE_FEED
_feed = None


def _get_feed():
    global _feed
    if _feed is None:
        _feed = get_feed()
    return _feed


def fetch_live_price(ticker):
    return _get_feed().fetch(ticker)


def is_market_hours():
    now = datetime.now()
    market_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0)
    market_close = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0)
    return market_open <= now <= market_close


def check_trade(trade, price_data):
    high = price_data["high"]
    low = price_data["low"]
    last = price_data["last"]
    atr = trade["atr"]
    events = []

    if not trade["trailing_active"] and high >= trade["tp_activation"]:
        trade["trailing_active"] = True
        trade["trail_stop"] = round(high - TRAIL_ATR_MULT * atr, 2)
        events.append(("TRAIL_ACTIVATED", trade["trail_stop"]))

    if trade["trailing_active"]:
        new_trail = round(high - TRAIL_ATR_MULT * atr, 2)
        if new_trail > trade["trail_stop"]:
            trade["trail_stop"] = new_trail
            events.append(("TRAIL_RAISED", new_trail))

    exit_price = None
    exit_reason = None
    day_open = price_data["open"]

    # Gap-through logic: if the day opened below the stop level,
    # we can't get the theoretical price — exit at the actual open.
    if not trade["trailing_active"] and low <= trade["sl"]:
        exit_price = day_open if day_open <= trade["sl"] else trade["sl"]
        exit_reason = "STOP LOSS (GAP)" if day_open <= trade["sl"] else "STOP LOSS"
    elif trade["trailing_active"] and low <= trade["trail_stop"]:
        exit_price = day_open if day_open <= trade["trail_stop"] else trade["trail_stop"]
        exit_reason = "TRAIL STOP (GAP)" if day_open <= trade["trail_stop"] else "TRAILING STOP"

    trade["current_price"] = round(last, 2)
    trade["unrealized_pnl"] = round(last - trade["entry_price"], 2)

    return exit_price, exit_reason, events


def print_dashboard(ledger, prices):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n  QuantEngine Live Monitor  [{now}]")
    print(f"  {'=' * 62}")
    print(f"  Capital: ₹{ledger['capital']:,.2f}   "
          f"P&L: ₹{ledger['capital'] - INITIAL_CAPITAL:+,.2f}   "
          f"Open: {len(ledger['open_trades'])}   "
          f"Closed: {len(ledger['closed_trades'])}")
    print(f"  {'=' * 62}")

    if not ledger["open_trades"]:
        print(f"\n  No open trades.")
        return

    print(f"\n  {'Ticker':<10s} {'Entry':>8s} {'Now':>8s} {'PnL':>8s} {'PnL%':>7s}  "
          f"{'SL':>8s} {'TP/Trail':>8s}  {'Status'}")
    print(f"  {'-' * 62}")

    for t in ledger["open_trades"]:
        entry = t["entry_price"]
        now_p = t["current_price"]
        pnl = now_p - entry
        pnl_pct = (pnl / entry) * 100

        if t["trailing_active"]:
            barrier = t["trail_stop"]
            status = "\033[96mTRAILING\033[0m"
        else:
            barrier = t["tp_activation"]
            status = f"Day {t['hold_days']}/{MAX_HOLD_DAYS}"

        if pnl > 0:
            color = "\033[92m"
        elif pnl < 0:
            color = "\033[91m"
        else:
            color = "\033[0m"

        dist_sl = ((now_p - t["sl"]) / now_p) * 100

        print(f"  {t['ticker']:<10s} {entry:>8.2f} {now_p:>8.2f} "
              f"{color}{pnl:>+8.2f} {pnl_pct:>+6.2f}%\033[0m  "
              f"{t['sl']:>8.2f} {barrier:>8.2f}  {status}")

    if ledger["closed_trades"]:
        recent = ledger["closed_trades"][-3:]
        print(f"\n  Recent Closes:")
        for c in recent:
            print(f"    {c['ticker']:<10s} {c['exit_reason']:<15s} "
                  f"PnL=₹{c['pnl']:+.2f} ({c['pnl_pct']:+.2f}%)")


def run_monitor():
    print(f"\n  QuantEngine Live Monitor starting...")

    if not os.path.exists(LEDGER_PATH):
        print(f"  No ledger found. Run paper_trade.py first.")
        return

    while True:
        if not is_market_hours():
            now = datetime.now()
            market_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0)
            if now < market_open:
                wait = (market_open - now).seconds
                print(f"  Market opens in {wait // 60} minutes. Waiting...")
                time.sleep(min(wait, 300))
                continue
            else:
                print(f"  Market closed. Final update done.")
                break

        ledger = load_ledger()

        if not ledger["open_trades"]:
            print_dashboard(ledger, {})
            print(f"\n  No open trades to monitor. Exiting.")
            break

        prices = {}
        for trade in ledger["open_trades"]:
            ticker = trade["ticker"]
            try:
                p = fetch_live_price(ticker)
                if p:
                    prices[ticker] = p
            except Exception:
                pass

        still_open = []
        for trade in ledger["open_trades"]:
            ticker = trade["ticker"]
            if ticker not in prices:
                still_open.append(trade)
                continue

            exit_price, exit_reason, events = check_trade(trade, prices[ticker])

            for event_type, val in events:
                if event_type == "TRAIL_ACTIVATED":
                    print(f"\n  *** {ticker} TRAIL ACTIVATED — trail stop set at ₹{val:.2f} ***")
                elif event_type == "TRAIL_RAISED":
                    print(f"\n  *** {ticker} trail stop raised to ₹{val:.2f} ***")

            if exit_price:
                pnl = exit_price - trade["entry_price"]
                pnl_pct = (pnl / trade["entry_price"]) * 100
                ledger["capital"] += pnl
                closed = {
                    **trade,
                    "exit_date": datetime.now().strftime("%Y-%m-%d"),
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                }
                ledger["closed_trades"].append(closed)
                print(f"\n  *** {ticker} {exit_reason} @ ₹{exit_price:.2f}  "
                      f"PnL=₹{pnl:+.2f} ({pnl_pct:+.2f}%) ***")
            else:
                still_open.append(trade)

        ledger["open_trades"] = still_open
        save_ledger(ledger)
        print_dashboard(ledger, prices)
        print(f"\n  Next update in {POLL_INTERVAL}s... (Ctrl+C to stop)")

        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n  Monitor stopped.")
            break


if __name__ == "__main__":
    run_monitor()

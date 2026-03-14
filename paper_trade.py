import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from data_loader import DataLoader
from strategy import MLStrategy
from costs import entry_cost, exit_cost
from config import (
    TP_ATR_MULT,
    SL_ATR_MULT,
    TRAIL_ATR_MULT,
    META_CONFIDENCE,
    INITIAL_CAPITAL,
    MAX_HOLD_DAYS,
    EOD_SAFE_HOUR,
    EOD_SAFE_MINUTE,
)
from risk import apply_trailing, evaluate_exit

WATCHLIST = [
    "ADANIENT", "ATGL", "RELIANCE", "TATASTEEL", "HDFCBANK",
    "TATAPOWER", "BAJFINANCE", "SBIN", "ITC", "INFY",
]

LOOKBACK_YEARS = 5
LEDGER_PATH = "paper_trades/ledger.json"
EXCEL_PATH = "paper_trades/portfolio.xlsx"

# Volatility-adjusted position sizing:
# - risk a fixed fraction of current capital per trade
# - scale quantity inversely with ATR-stop distance
RISK_PER_TRADE = 0.01  # 1% of current capital per trade
MAX_POSITION_FRACTION = 0.2  # cap notional per trade at 20% of capital


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            return json.load(f)
    return {"capital": INITIAL_CAPITAL, "open_trades": [], "closed_trades": []}


def save_ledger(ledger):
    os.makedirs("paper_trades", exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def fetch_today_prices(ticker, end_date):
    t = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    raw = yf.download(t, start=(end_date - timedelta(days=10)).strftime("%Y-%m-%d"),
                      end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        return None
    last = raw.iloc[-1]
    return {
        "date": raw.index[-1].strftime("%Y-%m-%d"),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": float(last["Close"]),
    }


def scan_stock(ticker, end_date):
    start_date = (end_date - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    loader = DataLoader(ticker, start_date, end_str)
    data = loader.fetch()
    if len(data) < 300:
        return None

    strat = MLStrategy()
    strat.fit(data.iloc[:-1])
    today = data.iloc[[-1]]
    signals, confidence = strat.generate_signals(today)

    signal = int(signals[0])
    conf = float(confidence[0])
    row = today.iloc[0]

    return {
        "signal": signal,
        "confidence": conf,
        "close": float(row["Close"]),
        "atr": float(row["ATR"]),
        "rsi": float(row["RSI_14"]),
        "cpp_score": float(row["CPP_score"]),
        "regime": int(row["Regime_ok"]),
    }


def update_open_trades(ledger, today_str):
    still_open = []
    for trade in ledger["open_trades"]:
        ticker = trade["ticker"]
        prices = fetch_today_prices(ticker, datetime.strptime(today_str, "%Y-%m-%d"))
        if prices is None:
            still_open.append(trade)
            continue

        trade["hold_days"] += 1
        high = prices["high"]
        low = prices["low"]
        close = prices["close"]
        day_open = prices["open"]
        atr = trade["atr"]

        trade["trailing_active"], trade["trail_stop"], _ = apply_trailing(
            high=high,
            atr=atr,
            trailing_active=trade["trailing_active"],
            tp_activation=trade["tp_activation"],
            current_trailing_stop=trade["trail_stop"],
            trail_mult=TRAIL_ATR_MULT,
        )

        exit_price, reason = evaluate_exit(
            day_open=day_open,
            low=low,
            close=close,
            base_sl=trade["sl"],
            trailing_active=trade["trailing_active"],
            trailing_stop=trade["trail_stop"],
            hold_days=trade["hold_days"],
            max_hold_days=MAX_HOLD_DAYS,
        )

        if exit_price is not None:
            if reason == "SL_GAP":
                exit_reason = "STOP LOSS (GAP)"
            elif reason == "SL":
                exit_reason = "STOP LOSS"
            elif reason == "TRAIL_GAP":
                exit_reason = "TRAIL STOP (GAP)"
            elif reason == "TRAIL":
                exit_reason = "TRAILING STOP"
            elif reason == "TIME":
                exit_reason = "TIME EXIT"
            else:
                exit_reason = "EXIT"

            qty = trade.get("quantity")
            if qty is None or "allocated_capital" not in trade:
                # Legacy trades: approximate per-share PnL without detailed costs.
                pnl = exit_price - trade["entry_price"]
                pnl_pct = (pnl / trade["entry_price"]) * 100 if trade["entry_price"] else 0.0
                ledger["capital"] += exit_price
            else:
                trade_value = exit_price * qty
                total_exit_cost = exit_cost(trade_value)
                exit_total = trade_value - total_exit_cost
                entry_total = trade["allocated_capital"]
                pnl = exit_total - entry_total
                pnl_pct = (pnl / entry_total) * 100 if entry_total else 0.0
                ledger["capital"] += exit_total

            closed = {
                **trade,
                "exit_date": today_str,
                "exit_price": round(exit_price, 2),
                "exit_reason": exit_reason,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
            ledger["closed_trades"].append(closed)
            print(f"    CLOSED {ticker}: {exit_reason} @ ₹{exit_price:.2f}  "
                  f"PnL=₹{pnl:+.2f} ({pnl_pct:+.2f}%)")
        else:
            qty = trade.get("quantity", 1)
            unr_pnl = (close - trade["entry_price"]) * qty
            trade["unrealized_pnl"] = round(unr_pnl, 2)
            trade["current_price"] = round(close, 2)
            still_open.append(trade)

    ledger["open_trades"] = still_open


def open_new_trades(ledger, signals, today_str):
    open_tickers = {t["ticker"] for t in ledger["open_trades"]}

    for sig in signals:
        if sig["signal"] != 1:
            continue
        if sig["ticker"] in open_tickers:
            continue

        entry = sig["close"]
        atr = sig["atr"]

        if atr <= 0:
            continue

        capital = ledger["capital"]
        if capital <= 0:
            break

        risk_capital = capital * RISK_PER_TRADE
        stop_distance = SL_ATR_MULT * atr
        if stop_distance <= 0:
            continue

        qty_risk = int(risk_capital // stop_distance)
        max_notional = capital * MAX_POSITION_FRACTION
        qty_cap = int(max_notional // entry) if entry > 0 else 0

        qty = min(qty_risk, qty_cap)
        if qty <= 0:
            continue

        notional = qty * entry
        total_entry_cost = entry_cost(notional)
        total_used = notional + total_entry_cost
        if total_used > capital:
            continue

        ledger["capital"] -= total_used

        trade = {
            "ticker": sig["ticker"],
            "entry_date": today_str,
            "entry_price": round(entry, 2),
            "quantity": qty,
            "allocated_capital": round(total_used, 2),
            "atr": round(atr, 2),
            "sl": round(entry - SL_ATR_MULT * atr, 2),
            "tp_activation": round(entry + TP_ATR_MULT * atr, 2),
            "trail_stop": 0.0,
            "trailing_active": False,
            "hold_days": 0,
            "confidence": sig["confidence"],
            "regime": "BULL" if sig["regime"] == 1 else "BEAR",
            "unrealized_pnl": 0.0,
            "current_price": round(entry, 2),
        }
        ledger["open_trades"].append(trade)
        print(f"    OPENED {sig['ticker']}: BUY x{qty} @ ₹{entry:.2f}  "
              f"SL=₹{trade['sl']:.2f}  TP_Act=₹{trade['tp_activation']:.2f}  "
              f"Risk≈₹{risk_capital:.0f}  Conf={sig['confidence']:.3f}")


def save_excel(ledger):
    os.makedirs("paper_trades", exist_ok=True)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        summary = pd.DataFrame([{
            "Starting Capital": INITIAL_CAPITAL,
            "Current Capital": round(ledger["capital"], 2),
            "Total P&L": round(ledger["capital"] - INITIAL_CAPITAL, 2),
            "Total P&L %": round((ledger["capital"] / INITIAL_CAPITAL - 1) * 100, 2),
            "Open Trades": len(ledger["open_trades"]),
            "Closed Trades": len(ledger["closed_trades"]),
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }])
        summary.to_excel(writer, sheet_name="Summary", index=False)

        if ledger["open_trades"]:
            open_df = pd.DataFrame(ledger["open_trades"])
            cols = ["ticker", "entry_date", "entry_price", "quantity", "allocated_capital",
                    "current_price", "unrealized_pnl", "sl", "tp_activation", "trail_stop",
                    "trailing_active", "hold_days", "confidence", "regime"]
            open_df = open_df[[c for c in cols if c in open_df.columns]]
            open_df.to_excel(writer, sheet_name="Open Trades", index=False)

        if ledger["closed_trades"]:
            closed_df = pd.DataFrame(ledger["closed_trades"])
            cols = ["ticker", "entry_date", "exit_date", "entry_price",
                    "exit_price", "pnl", "pnl_pct", "exit_reason",
                    "hold_days", "confidence", "regime"]
            closed_df = closed_df[[c for c in cols if c in closed_df.columns]]
            closed_df.to_excel(writer, sheet_name="Closed Trades", index=False)


def main():
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    # --- EOD settlement guard ---
    # NSE closing price = VWAP of last 30 min; not settled until ~3:40-3:45 PM.
    safe_time = today.replace(hour=EOD_SAFE_HOUR, minute=EOD_SAFE_MINUTE, second=0)
    if today < safe_time and "--force" not in sys.argv:
        print(f"\n  ⚠  It's {today.strftime('%H:%M')} — NSE closing price isn't settled yet.")
        print(f"  The official close is a VWAP of 3:00-3:30 PM,")
        print(f"  finalized around 3:40-3:45 PM.")
        print(f"  Run after {EOD_SAFE_HOUR}:{EOD_SAFE_MINUTE:02d} for accurate data,")
        print(f"  or use --force to override.\n")
        return

    print(f"\n  QuantEngine Paper Trader")
    print(f"  {today_str}")
    print(f"  {'=' * 50}")

    ledger = load_ledger()

    print(f"\n  [1/3] Updating {len(ledger['open_trades'])} open trades...")
    if ledger["open_trades"]:
        update_open_trades(ledger, today_str)
    else:
        print("    No open trades.")

    print(f"\n  [2/3] Scanning watchlist for new signals...")
    signals = []
    for ticker in WATCHLIST:
        print(f"    {ticker}...", end=" ", flush=True)
        try:
            result = scan_stock(ticker, today)
            if result:
                result["ticker"] = ticker
                signals.append(result)
                label = {1: "BUY", -1: "SELL", 0: "—"}[result["signal"]]
                print(f"{label} (conf={result['confidence']:.3f})")
            else:
                print("skip")
        except Exception as e:
            print(f"error: {e}")

    open_new_trades(ledger, signals, today_str)

    save_ledger(ledger)
    save_excel(ledger)

    print(f"\n  [3/3] Portfolio Status")
    print(f"  {'=' * 50}")
    print(f"  Capital:       ₹{ledger['capital']:,.2f}")
    print(f"  Total P&L:     ₹{ledger['capital'] - INITIAL_CAPITAL:+,.2f} "
          f"({(ledger['capital'] / INITIAL_CAPITAL - 1) * 100:+.2f}%)")
    print(f"  Open Trades:   {len(ledger['open_trades'])}")
    print(f"  Closed Trades: {len(ledger['closed_trades'])}")

    if ledger["open_trades"]:
        print(f"\n  Open Positions:")
        for t in ledger["open_trades"]:
            status = "TRAILING" if t["trailing_active"] else f"Day {t['hold_days']}/{MAX_HOLD_DAYS}"
            print(f"    {t['ticker']:<12s} Entry=₹{t['entry_price']:>8.2f}  "
                  f"Now=₹{t['current_price']:>8.2f}  "
                  f"PnL=₹{t['unrealized_pnl']:>+7.2f}  {status}")

    if ledger["closed_trades"]:
        wins = [t for t in ledger["closed_trades"] if t["pnl"] > 0]
        total = len(ledger["closed_trades"])
        print(f"\n  Win Rate: {len(wins)}/{total} ({len(wins)/total*100:.0f}%)")

    print(f"\n  Saved: {EXCEL_PATH}")
    print(f"  Ledger: {LEDGER_PATH}\n")


if __name__ == "__main__":
    main()

import os
import json
import logging
from datetime import datetime
from config_live import LOG_DIR

log = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self, initial_capital=20000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_log = []
        self.hold_days = {}
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log_file = os.path.join(
            LOG_DIR, f"paper_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    def on_decision(self, verdict):
        sym = verdict["symbol"]
        signal = verdict["signal"]
        ltp = verdict["ltp"]
        ts = datetime.now().isoformat()

        if signal == 1 and sym not in self.positions:
            qty = int(self.cash * 0.95 // ltp)
            if qty > 0:
                cost = qty * ltp
                self.cash -= cost
                self.positions[sym] = {"qty": qty, "entry": ltp, "entry_date": ts}
                self.hold_days[sym] = 0
                entry = {
                    "time": ts, "action": "BUY", "symbol": sym,
                    "qty": qty, "price": ltp, "cash_left": round(self.cash, 2),
                    "verdict": verdict,
                }
                self.trade_log.append(entry)
                self._write(entry)
                log.info(f"PAPER BUY  {sym} x{qty} @ ₹{ltp:.2f}")

        elif signal != 1 and sym in self.positions:
            pos = self.positions[sym]
            proceeds = pos["qty"] * ltp
            self.cash += proceeds
            pnl = (ltp / pos["entry"] - 1) * 100
            entry = {
                "time": ts, "action": "SELL", "symbol": sym,
                "qty": pos["qty"], "price": ltp,
                "entry_price": pos["entry"], "pnl_pct": round(pnl, 2),
                "hold_days": self.hold_days.get(sym, 0),
                "cash_left": round(self.cash, 2),
                "verdict": verdict,
            }
            self.trade_log.append(entry)
            self._write(entry)
            del self.positions[sym]
            self.hold_days.pop(sym, None)
            log.info(f"PAPER SELL {sym} x{pos['qty']} @ ₹{ltp:.2f} PnL={pnl:+.2f}%")

        else:
            entry = {
                "time": ts, "action": "HOLD", "symbol": sym,
                "signal": signal, "ltp": ltp,
                "regime_ok": verdict.get("regime_ok"),
            }
            self._write(entry)

    def tick_hold_days(self):
        for sym in list(self.hold_days.keys()):
            self.hold_days[sym] += 1

    def portfolio_value(self, prices):
        val = self.cash
        for sym, pos in self.positions.items():
            val += pos["qty"] * prices.get(sym, pos["entry"])
        return val

    def summary(self, prices):
        pv = self.portfolio_value(prices)
        ret = (pv / self.initial_capital - 1) * 100
        lines = [
            f"Cash: ₹{self.cash:,.2f}",
            f"Positions: {len(self.positions)}",
        ]
        for sym, pos in self.positions.items():
            cur = prices.get(sym, pos["entry"])
            pnl = (cur / pos["entry"] - 1) * 100
            lines.append(
                f"  {sym}: {pos['qty']} x ₹{cur:.2f} "
                f"(entry ₹{pos['entry']:.2f}, {pnl:+.2f}%, "
                f"day {self.hold_days.get(sym, '?')})"
            )
        lines.append(f"Portfolio: ₹{pv:,.2f} ({ret:+.2f}%)")
        return "\n".join(lines)

    def _write(self, entry):
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

import numpy as np
from config import (
    INITIAL_CAPITAL,
    MAX_HOLD_DAYS,
    TP_ATR_MULT,
    SL_ATR_MULT,
    TRAIL_ATR_MULT,
)
from costs import entry_cost, exit_cost
from risk import apply_trailing, evaluate_exit


class Backtester:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital

    def _entry_cost_rate(self, trade_value):
        if trade_value <= 0:
            return 0.0
        return entry_cost(trade_value) / trade_value

    def _exit_cost(self, trade_value):
        if trade_value <= 0:
            return 0.0
        return exit_cost(trade_value)

    def _trade_return(self, entry_price, exit_price, quantity):
        entry_value = entry_price * quantity
        exit_value = exit_price * quantity
        cost_entry = self._entry_cost_rate(entry_value)
        cost_exit = self._exit_cost(exit_value)
        return (exit_value - cost_exit) / (entry_value + entry_value * cost_entry) - 1

    def run(self, close_prices, open_prices, high_prices, low_prices, atr_values, signals):
        cash = self.initial_capital
        holdings = 0
        entry_price = 0.0
        hold_days = 0
        base_sl = 0.0
        tp_activation = 0.0
        trailing_active = False
        trailing_stop = 0.0
        portfolio_values = []
        trades = []
        pending_signal = None
        trail_hits = 0
        sl_hits = 0
        time_exits = 0

        for i in range(len(close_prices)):
            close = float(close_prices.iloc[i])
            open_ = float(open_prices.iloc[i])
            high = float(high_prices.iloc[i])
            low = float(low_prices.iloc[i])
            cur_atr = float(atr_values.iloc[i])

            if pending_signal == "BUY" and holdings == 0:
                cost_rate = self._entry_cost_rate(cash)
                effective_price = open_ * (1 + cost_rate)
                qty = int(cash // effective_price)
                if qty > 0:
                    entry_price = open_
                    entry_atr = cur_atr
                    tp_activation = entry_price + TP_ATR_MULT * entry_atr
                    base_sl = entry_price - SL_ATR_MULT * entry_atr
                    trailing_stop = base_sl
                    trailing_active = False
                    cost = qty * open_ * cost_rate
                    cash -= qty * open_ + cost
                    holdings = qty
                    hold_days = 0
                    trades.append(("BUY", close_prices.index[i], open_, qty))
                pending_signal = None

            elif pending_signal == "SELL" and holdings > 0:
                trade_value = holdings * open_
                cost = self._exit_cost(trade_value)
                cash += trade_value - cost
                ret = self._trade_return(entry_price, open_, holdings)
                trades.append(("SELL", close_prices.index[i], open_, holdings, ret, "time"))
                time_exits += 1
                holdings = 0
                hold_days = 0
                trailing_active = False
                pending_signal = None

            if holdings > 0:
                hold_days += 1

                trailing_active, trailing_stop, _ = apply_trailing(
                    high=high,
                    atr=cur_atr,
                    trailing_active=trailing_active,
                    tp_activation=tp_activation,
                    current_trailing_stop=trailing_stop,
                    trail_mult=TRAIL_ATR_MULT,
                )

                exit_price, reason = evaluate_exit(
                    day_open=open_,
                    low=low,
                    close=close,
                    base_sl=base_sl,
                    trailing_active=trailing_active,
                    trailing_stop=trailing_stop,
                    hold_days=hold_days,
                    max_hold_days=None,  # time exits handled via pending SELL signal
                )

                if exit_price is not None:
                    trade_value = holdings * exit_price
                    cost = self._exit_cost(trade_value)
                    cash += trade_value - cost
                    ret = self._trade_return(entry_price, exit_price, holdings)
                    if reason in ("TRAIL", "TRAIL_GAP"):
                        trades.append(("SELL", close_prices.index[i], exit_price, holdings, ret, "trail"))
                        trail_hits += 1
                    else:
                        trades.append(("SELL", close_prices.index[i], exit_price, holdings, ret, "sl"))
                        sl_hits += 1
                    holdings = 0
                    hold_days = 0
                    trailing_active = False

            portfolio_values.append(cash + holdings * close)

            signal = int(signals[i])
            if signal == 1 and holdings == 0:
                pending_signal = "BUY"
            elif holdings > 0 and not trailing_active and hold_days >= MAX_HOLD_DAYS:
                pending_signal = "SELL"
            else:
                pending_signal = None

        if holdings > 0:
            final_price = float(close_prices.iloc[-1])
            trade_value = holdings * final_price
            cost = self._exit_cost(trade_value)
            cash += trade_value - cost
            ret = self._trade_return(entry_price, final_price, holdings)
            trades.append(("SELL", close_prices.index[-1], final_price, holdings, ret, "end"))
            holdings = 0
            portfolio_values[-1] = cash

        total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100

        return {
            "portfolio_values": portfolio_values,
            "total_return": total_return,
            "final_value": portfolio_values[-1],
            "num_trades": sum(1 for t in trades if t[0] == "BUY"),
            "trades": trades,
            "trail_hits": trail_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
        }

    def buy_and_hold(self, close_prices, open_prices):
        entry_price = float(open_prices.iloc[0])
        cash = self.initial_capital

        cost_rate = self._entry_cost_rate(cash)
        effective_price = entry_price * (1 + cost_rate)
        qty = int(cash // effective_price)

        if qty == 0:
            return {"total_return": 0.0, "final_value": cash}

        entry_cost = qty * entry_price * cost_rate
        remaining_cash = cash - qty * entry_price - entry_cost

        exit_price = float(close_prices.iloc[-1])
        trade_value = qty * exit_price
        exit_cost = self._exit_cost(trade_value)

        final_value = remaining_cash + trade_value - exit_cost
        total_return = (final_value / self.initial_capital - 1) * 100

        return {"total_return": total_return, "final_value": final_value}

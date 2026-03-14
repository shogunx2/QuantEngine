#!/usr/bin/env python3
import sys
import os
import time
import logging
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_live import (
    BROKER, WATCHLIST, DECISION_H, DECISION_M,
    MARKET_OPEN_H, MARKET_OPEN_M, MARKET_CLOSE_H, MARKET_CLOSE_M,
)
from streamer import create_streamer
from aggregator import Aggregator
from brain import Brain
from paper_trader import PaperTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QuantEngine.Live")


def run_live():
    symbols = [w["symbol"] for w in WATCHLIST]
    log.info(f"Broker: {BROKER}")
    log.info(f"Watchlist: {symbols}")

    aggregator = Aggregator()
    brain = Brain()
    paper = PaperTrader()

    log.info("Training / loading model...")
    brain.load_or_train(symbols)

    tick_count = [0]

    def on_tick(tick):
        aggregator.on_tick(tick)
        tick_count[0] += 1
        if tick_count[0] % 500 == 0:
            candles = aggregator.get_all_daily()
            active = sum(1 for c in candles.values() if c.close > 0)
            log.info(f"Ticks: {tick_count[0]} | Active symbols: {active}")

    log.info("Connecting to market feed...")
    streamer = create_streamer(BROKER, WATCHLIST, on_tick)
    streamer.connect()
    log.info("Feed connected — streaming ticks")

    decision_fired = False

    def decision_loop():
        nonlocal decision_fired
        while True:
            now = datetime.now()
            if BROKER == "simulate":
                time.sleep(5)
                if tick_count[0] > 100 and not decision_fired:
                    fire_decision(brain, aggregator, paper, symbols)
                    decision_fired = True
                    break
            else:
                if (now.hour == DECISION_H and now.minute >= DECISION_M
                        and not decision_fired):
                    log.info("=== 3:15 PM — DECISION TIME ===")
                    fire_decision(brain, aggregator, paper, symbols)
                    decision_fired = True

                if now.hour >= MARKET_CLOSE_H and now.minute >= MARKET_CLOSE_M:
                    log.info("Market closed. Resetting for next day.")
                    paper.tick_hold_days()
                    aggregator.reset_day()
                    decision_fired = False
                    time.sleep(3600 * 14)

                time.sleep(30)

    decision_thread = threading.Thread(target=decision_loop, daemon=True)
    decision_thread.start()

    try:
        if BROKER == "simulate":
            decision_thread.join(timeout=120)
            time.sleep(2)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        streamer.disconnect()
        prices = {}
        for sym in symbols:
            candle = aggregator.get_daily_candle(sym)
            if candle:
                prices[sym] = candle.close
        log.info(f"\n{paper.summary(prices)}")


def fire_decision(brain, aggregator, paper, symbols):
    log.info("-" * 60)
    prices = {}
    for sym in symbols:
        candle = aggregator.get_daily_candle(sym)
        if candle is None or candle.close == 0:
            log.warning(f"  {sym}: no candle data — skip")
            continue
        prices[sym] = candle.close
        verdict = brain.evaluate(sym, candle)
        if verdict is None:
            continue

        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        regime = "OK" if verdict["regime_ok"] else "BLOCKED"
        raw = signal_map[verdict["signal_raw"]]
        final = signal_map[verdict["signal"]]

        log.info(
            f"  {sym:12s} LTP=₹{verdict['ltp']:>9.2f} | "
            f"Pred={verdict['prediction']:+.4f} Raw={raw:4s} "
            f"Regime={regime:7s} => {final}"
        )
        paper.on_decision(verdict)

    log.info("-" * 60)
    log.info(paper.summary(prices))


if __name__ == "__main__":
    run_live()

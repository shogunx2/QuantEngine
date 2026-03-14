from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TrailEvent:
    """Structured event for trailing-stop updates."""

    type: str  # "TRAIL_ACTIVATED" | "TRAIL_RAISED"
    level: float


def apply_trailing(
    high: float,
    atr: float,
    trailing_active: bool,
    tp_activation: float,
    current_trailing_stop: float,
    trail_mult: float,
) -> Tuple[bool, float, List[TrailEvent]]:
    """
    Shared trailing-stop logic.

    - If price first reaches TP activation level, turn on trailing stop.
    - Once active, ratchet the trail up as new highs are made.
    """
    events: List[TrailEvent] = []

    if not trailing_active and high >= tp_activation:
        trailing_active = True
        current_trailing_stop = high - trail_mult * atr
        events.append(TrailEvent("TRAIL_ACTIVATED", round(current_trailing_stop, 2)))

    if trailing_active:
        new_trail = high - trail_mult * atr
        if new_trail > current_trailing_stop:
            current_trailing_stop = new_trail
            events.append(TrailEvent("TRAIL_RAISED", round(current_trailing_stop, 2)))

    return trailing_active, current_trailing_stop, events


def evaluate_exit(
    day_open: float,
    low: float,
    close: float,
    base_sl: float,
    trailing_active: bool,
    trailing_stop: float,
    hold_days: int,
    max_hold_days: Optional[int],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Shared exit decision logic for SL / trailing / time exits.

    Returns (exit_price, reason_code) where reason_code is one of:
    - "SL", "SL_GAP"
    - "TRAIL", "TRAIL_GAP"
    - "TIME"
    - None if no exit.
    """
    # Hard stop loss (pre-trailing), with gap handling.
    if not trailing_active and low <= base_sl:
        if day_open <= base_sl:
            return day_open, "SL_GAP"
        return base_sl, "SL"

    # Trailing stop, with gap handling.
    if trailing_active and low <= trailing_stop:
        if day_open <= trailing_stop:
            return day_open, "TRAIL_GAP"
        return trailing_stop, "TRAIL"

    # Time-based exit (only when trailing has not activated).
    if max_hold_days is not None and not trailing_active and hold_days >= max_hold_days:
        return close, "TIME"

    return None, None


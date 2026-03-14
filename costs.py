from __future__ import annotations

from config import (
    BROKERAGE_RATE,
    BROKERAGE_CAP,
    SLIPPAGE_RATE,
    STT_RATE,
    DP_CHARGE,
)


def entry_cost(trade_value: float) -> float:
    """
    Total entry-side costs in rupees for a given notional.

    Includes:
    - brokerage (capped)
    - slippage (as a simple percentage of notional)
    """
    brokerage = min(trade_value * BROKERAGE_RATE, BROKERAGE_CAP)
    slippage = trade_value * SLIPPAGE_RATE
    return brokerage + slippage


def exit_cost(trade_value: float) -> float:
    """
    Total exit-side costs in rupees for a given notional.

    Includes:
    - brokerage (capped)
    - slippage
    - STT (on sell)
    - DP charge
    """
    brokerage = min(trade_value * BROKERAGE_RATE, BROKERAGE_CAP)
    slippage = trade_value * SLIPPAGE_RATE
    stt = trade_value * STT_RATE
    return brokerage + slippage + stt + DP_CHARGE


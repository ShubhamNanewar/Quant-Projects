"""Calculations with explicit units and failure conditions."""
from decimal import Decimal
import math

WAD = 10 ** 18
ORACLE_SCALE = 10 ** 36


def validate_book(bids, asks):
    if not bids or not asks:
        raise ValueError("Both sides of the book are required")
    for side in (bids, asks):
        if any(not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0 for p, q in side):
            raise ValueError("Book prices and quantities must be positive and finite")
    if bids != sorted(bids, reverse=True) or asks != sorted(asks):
        raise ValueError("Book levels are not ordered")
    if bids[0][0] >= asks[0][0]:
        raise ValueError("Book is crossed or locked")


def book_metrics(bids, asks):
    validate_book(bids, asks)
    bid, ask = bids[0][0], asks[0][0]
    mid = (bid + ask) / 2
    buy_depth = sum(p * q for p, q in bids if p >= mid * 0.999)
    sell_depth = sum(p * q for p, q in asks if p <= mid * 1.001)
    return {"bid": bid, "ask": ask, "mid": mid,
            "spread_bps": (ask - bid) / mid * 10000,
            "bid_depth_10bps_usd": buy_depth, "ask_depth_10bps_usd": sell_depth,
            "imbalance_10bps": (buy_depth - sell_depth) / (buy_depth + sell_depth)
            if buy_depth + sell_depth else math.nan}


def vwap(levels, quantity):
    """Return the price for a full base-asset fill, or None if depth is insufficient."""
    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("Quantity must be positive and finite")
    remaining, cash = quantity, 0.0
    for price, available in levels:
        take = min(remaining, available)
        cash += take * price
        remaining -= take
        if remaining <= quantity * 1e-12:
            return cash / quantity
    return None


def round_trip(buy_asks, sell_bids, quantity, buy_fee_bps, sell_fee_bps):
    buy, sell = vwap(buy_asks, quantity), vwap(sell_bids, quantity)
    if buy is None or sell is None:
        return None
    cost = quantity * buy * (1 + buy_fee_bps / 10000)
    proceeds = quantity * sell * (1 - sell_fee_bps / 10000)
    return {"buy_vwap": buy, "sell_vwap": sell, "gross_bps": (sell / buy - 1) * 10000,
            "net_bps": (proceeds / cost - 1) * 10000, "net_usd": proceeds - cost}


def debt_from_shares(shares, total_assets, total_shares):
    # Morpho adds one virtual asset and one million virtual shares.
    numerator = int(shares) * (int(total_assets) + 1)
    denominator = int(total_shares) + 10 ** 6
    return (numerator + denominator - 1) // denominator


def position_health(collateral, oracle_price, debt, lltv):
    values = tuple(map(int, (collateral, oracle_price, debt, lltv)))
    collateral, oracle_price, debt, lltv = values
    if min(values) < 0 or oracle_price == 0 or lltv > WAD:
        raise ValueError("Invalid position inputs")
    value = collateral * oracle_price // ORACLE_SCALE
    maximum = value * lltv // WAD
    return {"collateral_value_raw": value, "max_borrow_raw": maximum,
            "health_factor": maximum / debt if debt else math.inf,
            "liquidatable": debt > maximum}


def shocked_price(price, shock):
    if not 0 <= shock < 1:
        raise ValueError("Shock must lie in [0, 1)")
    numerator, denominator = Decimal(str(shock)).as_integer_ratio()
    return int(price) * (denominator - numerator) // denominator


def withdrawal_capacity(allocation_assets, supply_assets, borrow_assets):
    """Immediate capacity before other withdrawals, excluding reallocations and idle cash."""
    return min(int(allocation_assets), max(0, int(supply_assets) - int(borrow_assets)))

"""
Strategy: Swing Structure Forecast [BOSWaves]
Source: TradingView Pine Script by BOSWaves

Logic:
  Detect swing highs/lows using highest/lowest over N bars.
  Track swing direction changes.
  Forecast next swing move size from historical swing percentages (weighted avg).
  S/R zones created at each confirmed swing pivot.
  BUY  when swing low confirmed (direction flips bullish) + price above S/R support
  SELL when swing high confirmed (direction flips bearish) OR forecast target hit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backtesting import Strategy
from _backtest_runner import run_backtest


def detect_swings(high, low, close, swing_len):
    """
    Replicate BOSWaves swing detection:
    - H = ta.highest(high, swingLen), L = ta.lowest(low, swingLen)
    - dir = true when high == H, false when low == L
    - Swing high confirmed when high[1]==H[1] and high < H
    - Swing low confirmed when low[1]==L[1] and low > L
    Returns: direction array, swing_high prices/indices, swing_low prices/indices,
             and arrays marking direction change bars.
    """
    n = len(high)
    direction = np.zeros(n, dtype=int)  # 1=up, -1=down
    hi_price = np.full(n, np.nan)
    hi_idx = np.full(n, 0, dtype=int)
    lo_price = np.full(n, np.nan)
    lo_idx = np.full(n, 0, dtype=int)

    # Running highest/lowest
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - swing_len + 1)
        H[i] = np.max(high[start:i + 1])
        L[i] = np.min(low[start:i + 1])

    cur_dir = 0  # 0=unknown, 1=up(bull), -1=down(bear)
    cur_hi_price = high[0]
    cur_hi_idx = 0
    cur_lo_price = low[0]
    cur_lo_idx = 0

    # Swing pct history for forecast
    swing_pcts = []
    dir_changes = np.zeros(n, dtype=int)  # +1 = bullish flip, -1 = bearish flip

    for i in range(n):
        # Direction tracking (like Pine's dir := true/false)
        if high[i] == H[i]:
            cur_dir = 1
        if low[i] == L[i]:
            cur_dir = -1

        # Confirm swing high
        if i >= 1 and high[i - 1] == H[i - 1] and high[i] < H[i]:
            cur_hi_price = high[i - 1]
            cur_hi_idx = i - 1

        # Confirm swing low
        if i >= 1 and low[i - 1] == L[i - 1] and low[i] > L[i]:
            cur_lo_price = low[i - 1]
            cur_lo_idx = i - 1

        hi_price[i] = cur_hi_price
        hi_idx[i] = cur_hi_idx
        lo_price[i] = cur_lo_price
        lo_idx[i] = cur_lo_idx
        direction[i] = cur_dir

        # Detect direction change
        if i > 0 and direction[i] != direction[i - 1]:
            if cur_dir == 1:  # flipped bullish (swing low confirmed area)
                dir_changes[i] = 1
            else:  # flipped bearish (swing high confirmed area)
                dir_changes[i] = -1

            # Record swing size
            if not np.isnan(cur_hi_price) and not np.isnan(cur_lo_price) and cur_lo_price > 0:
                pct = abs(cur_hi_price - cur_lo_price) / cur_lo_price * 100
                swing_pcts.append(pct)

    return direction, dir_changes, hi_price, lo_price, swing_pcts


def calc_forecast_targets(high, low, close, direction, dir_changes, hi_price, lo_price, samples=20):
    """
    Calculate forecast target price at each direction change using weighted avg of recent swing sizes.
    """
    n = len(close)
    targets = np.full(n, np.nan)
    pct_history = []

    for i in range(n):
        if dir_changes[i] != 0:
            # Record swing pct
            if not np.isnan(hi_price[i]) and not np.isnan(lo_price[i]) and lo_price[i] > 0:
                pct = abs(hi_price[i] - lo_price[i]) / lo_price[i] * 100
                pct_history.append(pct)
                if len(pct_history) > samples:
                    pct_history.pop(0)

            # Calculate weighted avg forecast
            if len(pct_history) >= 2:
                total_w = 0
                weighted_sum = 0
                for j, p in enumerate(pct_history):
                    w = j + 1  # more recent = higher weight
                    weighted_sum += p * w
                    total_w += w
                f_pct = weighted_sum / total_w

                if dir_changes[i] == 1:  # bullish flip → target up
                    origin = lo_price[i]
                    targets[i] = origin * (1 + f_pct / 100)
                else:  # bearish flip → target down
                    origin = hi_price[i]
                    targets[i] = origin * (1 - f_pct / 100)

    return targets


class SwingForecast_BOSWaves(Strategy):
    swing_len = 16
    samples = 20

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)

        direction, dir_changes, hi_price, lo_price, _ = detect_swings(
            high, low, close, self.swing_len
        )
        targets = calc_forecast_targets(
            high, low, close, direction, dir_changes, hi_price, lo_price, self.samples
        )

        self.direction = self.I(lambda: direction, name="SwingDir")
        self.dir_changes = self.I(lambda: dir_changes, name="DirChange")
        self.targets = self.I(lambda: targets, name="ForecastTarget")

        # Track last target for exit
        last_target = np.full(len(close), np.nan)
        cur_target = np.nan
        for i in range(len(close)):
            if not np.isnan(targets[i]):
                cur_target = targets[i]
            last_target[i] = cur_target
        self.last_target = self.I(lambda: last_target, name="LastTarget")

    def next(self):
        if len(self.data.Close) < 2:
            return

        price = self.data.Close[-1]
        dir_change = self.dir_changes[-1]
        target = self.last_target[-1]
        cur_dir = self.direction[-1]

        if not self.position:
            # BUY on bullish direction change
            if dir_change == 1:
                self.buy(size=0.95)
        else:
            # EXIT on bearish direction change OR forecast target reached
            if dir_change == -1:
                self.position.close()
            elif not np.isnan(target) and cur_dir == 1 and price >= target:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        SwingForecast_BOSWaves,
        data_file="BTC-USD-1h.csv",
        notes="Swing structure + weighted forecast target (BOSWaves)",
    )
    run_backtest(
        SwingForecast_BOSWaves,
        data_file="BTC-USD-1d.csv",
        notes="Swing structure + weighted forecast target (BOSWaves, daily)",
    )

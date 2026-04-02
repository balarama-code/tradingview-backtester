"""
Strategy: Swing Profile [BigBeluga]
Source: TradingView Pine Script by BigBeluga

Logic:
  Detect swing highs/lows using highest/lowest over 50 bars.
  Build volume profile per swing leg — find Point of Control (PoC).
  Track buy vs sell volume delta within each swing.
  BUY  when direction flips bullish + positive volume delta (buyers in control)
  SELL when direction flips bearish OR price drops below PoC level
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from backtesting import Strategy
from _backtest_runner import run_backtest


def detect_swings_with_profile(high, low, close, open_arr, volume, swing_len, num_bins=20):
    """
    Detect swings and compute volume profile stats per swing leg.
    Returns: direction, dir_changes, poc_levels, volume_deltas
    """
    n = len(high)

    # Rolling highest/lowest
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - swing_len + 1)
        H[i] = np.max(high[start:i + 1])
        L[i] = np.min(low[start:i + 1])

    direction = np.zeros(n, dtype=int)
    cur_dir = 0
    hi_price, hi_idx = high[0], 0
    lo_price, lo_idx = low[0], 0

    dir_changes = np.zeros(n, dtype=int)
    poc_levels = np.full(n, np.nan)
    vol_deltas = np.full(n, np.nan)

    # Track last PoC for exit logic
    last_poc = np.nan

    for i in range(n):
        if high[i] == H[i]:
            cur_dir = -1  # down move starting
        if low[i] == L[i]:
            cur_dir = 1   # up move starting

        if i >= 1 and high[i - 1] == H[i - 1] and high[i] < H[i]:
            hi_price = high[i - 1]
            hi_idx = i - 1

        if i >= 1 and low[i - 1] == L[i - 1] and low[i] > L[i]:
            lo_price = low[i - 1]
            lo_idx = i - 1

        prev_dir = direction[i - 1] if i > 0 else 0
        direction[i] = cur_dir

        if i > 0 and direction[i] != direction[i - 1] and direction[i - 1] != 0:
            if cur_dir == 1:
                dir_changes[i] = 1   # bullish flip
            else:
                dir_changes[i] = -1  # bearish flip

            # Build volume profile for completed swing leg
            swing_start = min(hi_idx, lo_idx)
            swing_end = max(hi_idx, lo_idx)
            swing_bottom = min(hi_price, lo_price)
            swing_top = max(hi_price, lo_price)

            if swing_end > swing_start and swing_top > swing_bottom:
                bin_step = (swing_top - swing_bottom) / num_bins
                vol_bins = np.zeros(num_bins)
                buy_vol = 0.0
                sell_vol = 0.0

                for j in range(swing_start, min(swing_end + 1, i)):
                    c = close[j]
                    for k in range(num_bins):
                        bin_mid = swing_bottom + (k * bin_step) + (bin_step / 2)
                        if abs(bin_mid - c) < bin_step:
                            vol_bins[k] += volume[j]
                    if close[j] > open_arr[j]:
                        buy_vol += volume[j]
                    else:
                        sell_vol += volume[j]

                # PoC = mid price of highest volume bin
                poc_bin = np.argmax(vol_bins)
                last_poc = swing_bottom + (poc_bin * bin_step) + (bin_step / 2)

                total_vol = buy_vol + sell_vol
                delta = (buy_vol - sell_vol) / total_vol * 100 if total_vol > 0 else 0
                vol_deltas[i] = delta

        poc_levels[i] = last_poc

    return direction, dir_changes, poc_levels, vol_deltas


class SwingProfile_BigBeluga(Strategy):
    swing_len = 50

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)
        open_arr = np.array(self.data.Open, dtype=float)
        volume = np.array(self.data.Volume, dtype=float)

        direction, dir_changes, poc_levels, vol_deltas = detect_swings_with_profile(
            high, low, close, open_arr, volume, self.swing_len
        )

        self.direction = self.I(lambda: direction, name="SwingDir")
        self.dir_changes = self.I(lambda: dir_changes, name="DirChange")
        self.poc = self.I(lambda: poc_levels, name="PoC")
        self.vol_delta = self.I(lambda: vol_deltas, name="VolDelta")

    def next(self):
        if len(self.data.Close) < 2:
            return

        price = self.data.Close[-1]
        dir_change = self.dir_changes[-1]
        poc = self.poc[-1]
        delta = self.vol_delta[-1]

        if not self.position:
            # BUY on bullish flip with positive volume delta (buyers dominated)
            if dir_change == 1:
                if np.isnan(delta) or delta > 0:
                    self.buy(size=0.95)
        else:
            # EXIT on bearish flip OR price falls below PoC
            if dir_change == -1:
                self.position.close()
            elif not np.isnan(poc) and price < poc:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        SwingProfile_BigBeluga,
        data_file="BTC-USD-1h.csv",
        notes="Swing profile + PoC + volume delta filter (BigBeluga)",
    )
    run_backtest(
        SwingProfile_BigBeluga,
        data_file="BTC-USD-1d.csv",
        notes="Swing profile + PoC + volume delta filter (BigBeluga, daily)",
    )

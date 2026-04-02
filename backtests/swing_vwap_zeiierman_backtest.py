"""
Strategy: Dynamic Swing Anchored VWAP (Zeiierman)
Source: TradingView Pine Script by Zeiierman

Logic:
  Detect swing highs/lows using highest/lowest bars over N period.
  Direction: if swing high bar is more recent → uptrend, else downtrend.
  BUY  when direction flips to uptrend (swing low confirmed = HL/LL)
  SELL when direction flips to downtrend (swing high confirmed = HH/LH)
  Uses anchored VWAP from swing points as dynamic support/resistance filter.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import ta
from backtesting import Strategy
from _backtest_runner import run_backtest


def highest_bars_ago(high_arr, period):
    """For each bar, return how many bars ago the highest high was (0 = current bar is highest)."""
    result = np.full(len(high_arr), np.nan)
    for i in range(period - 1, len(high_arr)):
        window = high_arr[i - period + 1:i + 1]
        idx = np.argmax(window)
        result[i] = -(period - 1 - idx)  # negative offset like Pine's highestbars
    return result


def lowest_bars_ago(low_arr, period):
    """For each bar, return how many bars ago the lowest low was (0 = current bar is lowest)."""
    result = np.full(len(low_arr), np.nan)
    for i in range(period - 1, len(low_arr)):
        window = low_arr[i - period + 1:i + 1]
        idx = np.argmin(window)
        result[i] = -(period - 1 - idx)
    return result


def calc_swing_direction(high_arr, low_arr, period):
    """
    Track swing high/low bar indices and determine direction.
    dir = 1 (uptrend) when most recent pivot is a swing high.
    dir = -1 (downtrend) when most recent pivot is a swing low.
    """
    n = len(high_arr)
    direction = np.zeros(n)
    ph_bar = 0  # bar index of last swing high
    pl_bar = 0  # bar index of last swing low

    hb = highest_bars_ago(high_arr, period)
    lb = lowest_bars_ago(low_arr, period)

    for i in range(n):
        if not np.isnan(hb[i]) and hb[i] == 0:
            ph_bar = i
        if not np.isnan(lb[i]) and lb[i] == 0:
            pl_bar = i
        direction[i] = 1 if ph_bar > pl_bar else -1

    return direction


def calc_anchored_vwap(hlc3_arr, volume_arr, direction, base_apt=20.0):
    """Calculate EWMA-based anchored VWAP that resets at each swing change."""
    n = len(hlc3_arr)
    vwap = np.full(n, np.nan)
    decay = np.exp(-np.log(2.0) / max(1.0, base_apt))
    alpha = 1.0 - decay

    p = hlc3_arr[0] * volume_arr[0]
    vol = volume_arr[0]

    for i in range(n):
        if i > 0 and direction[i] != direction[i - 1]:
            # Reset on direction change
            p = hlc3_arr[i] * volume_arr[i]
            vol = volume_arr[i]
        else:
            pxv = hlc3_arr[i] * volume_arr[i]
            p = (1.0 - alpha) * p + alpha * pxv
            vol = (1.0 - alpha) * vol + alpha * volume_arr[i]

        vwap[i] = p / vol if vol > 0 else np.nan

    return vwap


class SwingVWAP_Zeiierman(Strategy):
    swing_period = 50
    base_apt = 20

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)
        volume = np.array(self.data.Volume, dtype=float)
        hlc3 = (high + low + close) / 3.0

        direction = calc_swing_direction(high, low, self.swing_period)
        vwap = calc_anchored_vwap(hlc3, volume, direction, self.base_apt)

        self.direction = self.I(lambda: direction, name="SwingDir")
        self.vwap = self.I(lambda: vwap, name="AnchoredVWAP")

    def next(self):
        if len(self.data.Close) < 2:
            return
        if np.isnan(self.direction[-1]) or np.isnan(self.direction[-2]):
            return

        cur_dir = self.direction[-1]
        prev_dir = self.direction[-2]
        price = self.data.Close[-1]
        vwap_val = self.vwap[-1]

        if not self.position:
            # BUY: direction flips to uptrend + price above VWAP
            if prev_dir == -1 and cur_dir == 1 and price > vwap_val:
                self.buy(size=0.95)
        else:
            # SELL: direction flips to downtrend OR price drops below VWAP
            if (prev_dir == 1 and cur_dir == -1) or price < vwap_val:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        SwingVWAP_Zeiierman,
        data_file="BTC-USD-1h.csv",
        notes="Swing direction + anchored VWAP filter (Zeiierman)",
    )
    run_backtest(
        SwingVWAP_Zeiierman,
        data_file="BTC-USD-1d.csv",
        notes="Swing direction + anchored VWAP filter (Zeiierman, daily)",
    )

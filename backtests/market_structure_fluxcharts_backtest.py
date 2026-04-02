"""
Strategy: Market Structure Dashboard [Flux Charts]
Source: TradingView Pine Script by Flux Charts

Logic:
  1. Swing detection: identify swing highs and swing lows (pivot points)
  2. Structure classification:
     - HH (Higher High) / HL (Higher Low) = bullish structure
     - LH (Lower High) / LL (Lower Low) = bearish structure
  3. Structure bias:
     - Bullish when last swing high is HH AND last swing low is HL
     - Bearish when last swing high is LH AND last swing low is LL
  4. EMA trend filter: close > EMA(9) for longs
  BUY  when structure bias flips bullish + close > EMA
  SELL when structure bias flips bearish OR close < EMA while bearish structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backtesting import Strategy
from _backtest_runner import run_backtest


def detect_swings(high, low, swing_len=5):
    """
    Detect swing highs and swing lows using pivot logic.
    A swing high at bar i: high[i] is highest in [i-swing_len, i+swing_len]
    A swing low at bar i: low[i] is lowest in [i-swing_len, i+swing_len]
    Returns arrays of swing_high_idx, swing_high_val, swing_low_idx, swing_low_val.
    """
    n = len(high)
    swing_highs = []  # (index, value)
    swing_lows = []   # (index, value)

    for i in range(swing_len, n - swing_len):
        # Swing high: high[i] >= all highs in window
        is_sh = True
        for j in range(i - swing_len, i + swing_len + 1):
            if j != i and high[j] > high[i]:
                is_sh = False
                break
        if is_sh:
            swing_highs.append((i, high[i]))

        # Swing low: low[i] <= all lows in window
        is_sl = True
        for j in range(i - swing_len, i + swing_len + 1):
            if j != i and low[j] < low[i]:
                is_sl = False
                break
        if is_sl:
            swing_lows.append((i, low[i]))

    return swing_highs, swing_lows


def calc_structure_bias(high, low, close, swing_len=5, ema_len=9):
    """
    Calculate market structure bias per bar.
    Returns:
      bias: +1 bullish, -1 bearish, 0 neutral (per bar)
      ema: EMA values
    """
    n = len(close)

    # Detect swings
    swing_highs, swing_lows = detect_swings(high, low, swing_len)

    # Classify each swing relative to previous
    # high_type: +1 = HH, -1 = LH
    # low_type: +1 = HL, -1 = LL
    bias = np.zeros(n, dtype=int)

    last_sh_val = None
    last_sl_val = None
    high_type = 0  # +1=HH, -1=LH
    low_type = 0   # +1=HL, -1=LL

    sh_idx = 0
    sl_idx = 0

    for i in range(n):
        # Process any swing highs confirmed at or before bar i
        # A swing at index j is confirmed at j + swing_len
        while sh_idx < len(swing_highs) and swing_highs[sh_idx][0] + swing_len <= i:
            j, val = swing_highs[sh_idx]
            if last_sh_val is not None:
                if val > last_sh_val:
                    high_type = 1   # HH
                else:
                    high_type = -1  # LH
            last_sh_val = val
            sh_idx += 1

        while sl_idx < len(swing_lows) and swing_lows[sl_idx][0] + swing_len <= i:
            j, val = swing_lows[sl_idx]
            if last_sl_val is not None:
                if val > last_sl_val:
                    low_type = 1   # HL
                else:
                    low_type = -1  # LL
            last_sl_val = val
            sl_idx += 1

        # Structure bias
        if high_type == 1 and low_type == 1:
            bias[i] = 1   # Bullish: HH + HL
        elif high_type == -1 and low_type == -1:
            bias[i] = -1  # Bearish: LH + LL
        else:
            bias[i] = bias[i-1] if i > 0 else 0  # Mixed: keep previous

    # EMA
    ema = pd.Series(close).ewm(span=ema_len, adjust=False).mean().values

    return bias, ema


class MarketStructure_Flux(Strategy):
    swing_len = 5
    ema_len = 9

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)

        bias, ema = calc_structure_bias(high, low, close, self.swing_len, self.ema_len)

        self.bias = self.I(lambda: bias, name="StructureBias")
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_len, adjust=False).mean(),
                         self.data.Close, name="EMA9")

    def next(self):
        if len(self.data.Close) < self.swing_len * 2 + 5:
            return

        cur_bias = self.bias[-1]
        prev_bias = self.bias[-2]
        price = self.data.Close[-1]
        ema = self.ema[-1]

        if np.isnan(ema) or cur_bias == 0:
            return

        if not self.position:
            # BUY: bias flips to bullish + price above EMA
            if cur_bias == 1 and prev_bias != 1 and price > ema:
                self.buy(size=0.95)
            # Also enter if already bullish bias and price crosses above EMA
            elif cur_bias == 1 and price > ema and self.data.Close[-2] <= self.ema[-2]:
                self.buy(size=0.95)
        else:
            # SELL: bias flips bearish
            if cur_bias == -1 and prev_bias != -1:
                self.position.close()
            # Also exit if price drops below EMA while bias is not bullish
            elif cur_bias != 1 and price < ema:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        MarketStructure_Flux,
        data_file="BTC-USD-1h.csv",
        notes="Market Structure HH/HL/LH/LL + EMA filter (Flux Charts)",
    )
    run_backtest(
        MarketStructure_Flux,
        data_file="BTC-USD-1d.csv",
        notes="Market Structure HH/HL/LH/LL + EMA filter (Flux Charts, daily)",
    )

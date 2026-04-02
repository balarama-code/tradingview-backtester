"""
Strategy: Machine Learning Adaptive SuperTrend (AlgoAlpha)
Source: TradingView Pine Script by AlgoAlpha

Logic:
  1. K-Means clustering (3 clusters) on ATR values over training period
     to classify current volatility as High / Medium / Low.
  2. Assign the nearest cluster centroid as the adaptive ATR value.
  3. Feed adaptive ATR into a SuperTrend indicator.
  BUY  when SuperTrend flips bullish (dir crosses under 0)
  SELL when SuperTrend flips bearish (dir crosses over 0)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backtesting import Strategy
from _backtest_runner import run_backtest


def calc_atr(high, low, close, period=10):
    """Calculate Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    # RMA (Wilder's smoothing)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def kmeans_3_clusters(atr_vals, training_period, high_pct=0.75, mid_pct=0.5, low_pct=0.25):
    """
    Run K-Means (3 clusters) on ATR over a rolling training window.
    Returns assigned centroid for each bar.
    """
    n = len(atr_vals)
    assigned = np.full(n, np.nan)

    for i in range(training_period - 1, n):
        window = atr_vals[i - training_period + 1:i + 1]
        window = window[~np.isnan(window)]
        if len(window) < 3:
            continue

        upper = np.max(window)
        lower = np.min(window)
        rng = upper - lower
        if rng == 0:
            assigned[i] = upper
            continue

        # Initial centroids
        c_high = lower + rng * high_pct
        c_mid = lower + rng * mid_pct
        c_low = lower + rng * low_pct

        # K-Means iterations
        for _ in range(20):
            hv, mv, lv = [], [], []
            for v in window:
                d1 = abs(v - c_high)
                d2 = abs(v - c_mid)
                d3 = abs(v - c_low)
                min_d = min(d1, d2, d3)
                if d1 == min_d:
                    hv.append(v)
                elif d2 == min_d:
                    mv.append(v)
                else:
                    lv.append(v)

            new_high = np.mean(hv) if hv else c_high
            new_mid = np.mean(mv) if mv else c_mid
            new_low = np.mean(lv) if lv else c_low

            if new_high == c_high and new_mid == c_mid and new_low == c_low:
                break
            c_high, c_mid, c_low = new_high, new_mid, new_low

        # Assign current ATR to nearest centroid
        cur = atr_vals[i]
        dists = [abs(cur - c_high), abs(cur - c_mid), abs(cur - c_low)]
        centroids = [c_high, c_mid, c_low]
        assigned[i] = centroids[np.argmin(dists)]

    return assigned


def calc_supertrend(high, low, close, factor, atr_values):
    """
    SuperTrend using provided ATR values (adaptive centroid).
    Returns (supertrend_line, direction).
    direction: -1 = bullish, 1 = bearish
    """
    n = len(close)
    src = (high + low) / 2.0
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)

    for i in range(n):
        if np.isnan(atr_values[i]):
            direction[i] = direction[i - 1] if i > 0 else 1
            continue

        ub = src[i] + factor * atr_values[i]
        lb = src[i] - factor * atr_values[i]

        if i > 0 and not np.isnan(lower_band[i - 1]):
            lb = lb if (lb > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]) else lower_band[i - 1]
        if i > 0 and not np.isnan(upper_band[i - 1]):
            ub = ub if (ub < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]) else upper_band[i - 1]

        upper_band[i] = ub
        lower_band[i] = lb

        if i == 0 or np.isnan(supertrend[i - 1]):
            direction[i] = 1
        elif supertrend[i - 1] == upper_band[i - 1]:
            direction[i] = -1 if close[i] > ub else 1
        else:
            direction[i] = 1 if close[i] < lb else -1

        supertrend[i] = lb if direction[i] == -1 else ub

    return supertrend, direction


class ML_AdaptiveSuperTrend(Strategy):
    atr_len = 10
    factor = 3.0
    training_period = 100

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)

        # ATR
        atr = calc_atr(high, low, close, self.atr_len)

        # K-Means adaptive ATR
        adaptive_atr = kmeans_3_clusters(atr, self.training_period)

        # SuperTrend with adaptive ATR
        st_line, direction = calc_supertrend(high, low, close, self.factor, adaptive_atr)

        self.st = self.I(lambda: st_line, name="SuperTrend")
        self.direction = self.I(lambda: direction, name="Direction")

    def next(self):
        if len(self.data.Close) < 2:
            return
        cur_dir = self.direction[-1]
        prev_dir = self.direction[-2]

        if np.isnan(cur_dir) or np.isnan(prev_dir):
            return

        if not self.position:
            # BUY: direction flips to bullish (-1)
            if prev_dir == 1 and cur_dir == -1:
                self.buy(size=0.95)
        else:
            # SELL: direction flips to bearish (1)
            if prev_dir == -1 and cur_dir == 1:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        ML_AdaptiveSuperTrend,
        data_file="BTC-USD-1h.csv",
        notes="K-Means adaptive SuperTrend (AlgoAlpha)",
    )
    run_backtest(
        ML_AdaptiveSuperTrend,
        data_file="BTC-USD-1d.csv",
        notes="K-Means adaptive SuperTrend (AlgoAlpha, daily)",
    )

"""
Strategy: StealthTrail SuperTrend ML Pro [WillyAlgoTrader] v5.1.0
Source: TradingView Pine Script by WillyAlgoTrader

Logic:
  1. Instrument profiling: efficiency ratio, normalized volatility, vol cluster
  2. Regime classification: TRENDING / RANGING / VOLATILE
  3. Auto-tune ATR length + SuperTrend multiplier based on regime weights
  4. Adaptive SuperTrend with flip cushion + cooldown
  5. RSI momentum filter on signals
  BUY  when SuperTrend flips bullish + RSI momentum OK
  SELL when SuperTrend flips bearish + RSI momentum OK OR trailing SL hit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import ta
from backtesting import Strategy
from _backtest_runner import run_backtest


def calc_atr(high, low, close, period):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calc_efficiency_ratio(close, lookback):
    n = len(close)
    er = np.full(n, 0.5)
    for i in range(lookback, n):
        price_change = abs(close[i] - close[i - lookback])
        total_path = sum(abs(close[j] - close[j-1]) for j in range(i - lookback + 1, i + 1))
        er[i] = price_change / total_path if total_path > 0 else 0.5
    return er


def calc_adaptive_supertrend(high, low, close, profile_lookback=100):
    """
    Full adaptive SuperTrend with regime-based auto-tuning.
    """
    n = len(close)

    # Efficiency ratio for regime
    er = calc_efficiency_ratio(close, profile_lookback)

    # Normalized volatility
    atr_profile = calc_atr(high, low, close, min(profile_lookback, 50))
    norm_vol = np.where(close > 0, atr_profile / close * 100, 1.0)

    # Vol cluster (short ATR / long ATR)
    atr_short = calc_atr(high, low, close, 20)
    atr_long = calc_atr(high, low, close, min(profile_lookback, 50))
    vol_cluster = np.where(atr_long > 0, atr_short / atr_long, 1.0)

    # Regime scores
    trend_dir = np.zeros(n, dtype=int)
    st_band = np.full(n, np.nan)
    bars_since_flip = np.full(n, 100)

    for i in range(profile_lookback, n):
        # Smoothed regime inputs
        er_smooth = np.nanmean(er[max(0, i-30):i+1])
        vc_smooth = np.nanmean(vol_cluster[max(0, i-30):i+1])

        trend_score = er_smooth
        range_score = (1.0 - er_smooth)
        volat_score = max(0, vc_smooth - 1.0)
        total = trend_score + range_score + volat_score
        if total == 0:
            total = 1.0

        wT = trend_score / total
        wR = range_score / total
        wV = volat_score / total

        # Auto-tune parameters
        nv = np.clip(norm_vol[i] / 1.5, 0.7, 1.8) if not np.isnan(norm_vol[i]) else 1.0
        eff_atr_len = int(np.clip((wT * 10 + wR * 16 + wV * 21) * nv, 5, 50))
        nv2 = np.clip(norm_vol[i] / 1.0, 0.8, 1.5) if not np.isnan(norm_vol[i]) else 1.0
        eff_mult = np.clip((wT * 2.0 + wR * 3.2 + wV * 3.8) * nv2, 1.2, 6.0)
        eff_cushion = np.clip(wT * 0.05 + wR * 0.25 + wV * 0.15, 0.0, 0.5)
        eff_cooldown = int(np.clip(wT * 2 + wR * 5 + wV * 3, 1, 10))

        # ATR at effective length
        start = max(0, i - eff_atr_len + 1)
        tr_window = np.zeros(i - start + 1)
        for j in range(start, i + 1):
            if j == 0:
                tr_window[j - start] = high[j] - low[j]
            else:
                tr_window[j - start] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_val = np.mean(tr_window[-eff_atr_len:]) if len(tr_window) >= eff_atr_len else np.mean(tr_window)
        if np.isnan(atr_val) or atr_val == 0:
            atr_val = high[i] - low[i]

        # Adaptive multiplier with vol ratio
        adapt_smooth_len = int(np.clip(eff_atr_len * 4, 20, 200))
        atr_sma_start = max(0, i - adapt_smooth_len + 1)
        atr_sma = np.nanmean(atr_short[atr_sma_start:i+1])
        vol_ratio = atr_val / atr_sma if atr_sma > 0 else 1.0
        adaptive_mult = np.clip(eff_mult * vol_ratio, 1.0, 5.0)

        # SuperTrend bands
        hl2 = (high[i] + low[i]) / 2.0
        upper = hl2 + adaptive_mult * atr_val
        lower = hl2 - adaptive_mult * atr_val

        prev_dir = trend_dir[i-1] if i > 0 else 1
        prev_band = st_band[i-1] if i > 0 and not np.isnan(st_band[i-1]) else (lower if prev_dir == 1 else upper)
        prev_bars = bars_since_flip[i-1] if i > 0 else 100
        cushion = eff_cushion * atr_val

        if prev_dir == 1:
            band = max(lower, prev_band)
            if close[i] < (band - cushion) and prev_bars >= eff_cooldown:
                trend_dir[i] = -1
                st_band[i] = upper
                bars_since_flip[i] = 0
            else:
                trend_dir[i] = 1
                st_band[i] = band
                bars_since_flip[i] = prev_bars + 1
        else:
            band = min(upper, prev_band)
            if close[i] > (band + cushion) and prev_bars >= eff_cooldown:
                trend_dir[i] = 1
                st_band[i] = lower
                bars_since_flip[i] = 0
            else:
                trend_dir[i] = -1
                st_band[i] = band
                bars_since_flip[i] = prev_bars + 1

    return trend_dir, st_band


class StealthTrail_ML(Strategy):
    profile_lookback = 100
    rsi_len = 13
    rsi_thresh = 45.0

    def init(self):
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)
        close = np.array(self.data.Close, dtype=float)

        trend_dir, st_band = calc_adaptive_supertrend(high, low, close, self.profile_lookback)

        self.trend_dir = self.I(lambda: trend_dir, name="TrendDir")
        self.st_band = self.I(lambda: st_band, name="STBand")

        # RSI
        self.rsi = self.I(
            lambda x: ta.momentum.RSIIndicator(pd.Series(x), window=self.rsi_len).rsi(),
            self.data.Close, name="RSI"
        )

    def next(self):
        if len(self.data.Close) < self.profile_lookback + 2:
            return

        cur_dir = self.trend_dir[-1]
        prev_dir = self.trend_dir[-2]
        rsi = self.rsi[-1]
        band = self.st_band[-1]

        if np.isnan(rsi) or np.isnan(band) or cur_dir == 0:
            return

        # Flip detection
        bull_flip = prev_dir != 1 and cur_dir == 1
        bear_flip = prev_dir != -1 and cur_dir == -1

        # RSI momentum filter
        rsi_ok_bull = rsi >= self.rsi_thresh
        rsi_ok_bear = rsi <= (100 - self.rsi_thresh)

        if not self.position:
            if bull_flip and rsi_ok_bull:
                self.buy(size=0.95, sl=band)
        else:
            if bear_flip and rsi_ok_bear:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        StealthTrail_ML,
        data_file="BTC-USD-1h.csv",
        notes="Adaptive SuperTrend + regime auto-tune + RSI filter (WillyAlgo StealthTrail)",
    )
    run_backtest(
        StealthTrail_ML,
        data_file="BTC-USD-1d.csv",
        notes="Adaptive SuperTrend + regime auto-tune + RSI filter (WillyAlgo StealthTrail, daily)",
    )

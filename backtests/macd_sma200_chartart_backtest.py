"""
Strategy: MACD + SMA 200 Strategy (by ChartArt)
Source: TradingView Pine Script v2 by ChartArt

Logic:
  fastMA = SMA(12), slowMA = SMA(26), veryslowMA = SMA(200)
  MACD = fastMA - slowMA, Signal = SMA(MACD, 9), Histogram = MACD - Signal
  BUY  when histogram crosses above 0 AND MACD > 0 AND fastMA > slowMA
       AND close[26] > SMA200
  SELL when histogram crosses below 0 AND MACD < 0 AND fastMA < slowMA
       AND close[26] < SMA200
  Cancel long if slowMA < SMA200, cancel short if slowMA > SMA200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backtesting import Strategy
from _backtest_runner import run_backtest


class MACD_SMA200(Strategy):
    fast_len = 12
    slow_len = 26
    signal_len = 9
    veryslow_len = 200

    def init(self):
        close = pd.Series(self.data.Close)

        # SMA
        self.fast_ma = self.I(lambda x: pd.Series(x).rolling(self.fast_len).mean(), self.data.Close, name="FastMA")
        self.slow_ma = self.I(lambda x: pd.Series(x).rolling(self.slow_len).mean(), self.data.Close, name="SlowMA")
        self.veryslow_ma = self.I(lambda x: pd.Series(x).rolling(self.veryslow_len).mean(), self.data.Close, name="SMA200")

        # MACD
        fast_sma = close.rolling(self.fast_len).mean()
        slow_sma = close.rolling(self.slow_len).mean()
        macd = fast_sma - slow_sma
        signal = macd.rolling(self.signal_len).mean()
        hist = macd - signal

        self.macd = self.I(lambda: macd.values, name="MACD")
        self.signal = self.I(lambda: signal.values, name="Signal")
        self.hist = self.I(lambda: hist.values, name="Histogram")

    def next(self):
        if len(self.data.Close) < self.veryslow_len + 2:
            return

        hist = self.hist[-1]
        prev_hist = self.hist[-2]
        macd = self.macd[-1]
        fast = self.fast_ma[-1]
        slow = self.slow_ma[-1]
        veryslow = self.veryslow_ma[-1]
        price = self.data.Close[-1]

        if any(np.isnan(x) for x in [hist, prev_hist, macd, fast, slow, veryslow]):
            return

        # Price 26 bars ago above/below SMA200
        price_lagged = self.data.Close[-self.slow_len] if len(self.data.Close) > self.slow_len else price

        # Crossover/crossunder of histogram over 0
        hist_cross_above = prev_hist <= 0 and hist > 0
        hist_cross_below = prev_hist >= 0 and hist < 0

        if not self.position:
            # BUY: hist crosses above 0 + MACD > 0 + fastMA > slowMA + lagged price > SMA200
            if hist_cross_above and macd > 0 and fast > slow and price_lagged > veryslow:
                # Additional filter: slowMA > SMA200 (cancel long if not)
                if slow > veryslow:
                    self.buy(size=0.95)
        else:
            # SELL: hist crosses below 0 + MACD < 0 + fastMA < slowMA + lagged price < SMA200
            # OR cancel condition: slowMA < SMA200
            if slow < veryslow:
                self.position.close()
            elif hist_cross_below and macd < 0 and fast < slow:
                self.position.close()


if __name__ == "__main__":
    run_backtest(
        MACD_SMA200,
        data_file="BTC-USD-1h.csv",
        notes="MACD + SMA200 trend filter (ChartArt)",
    )
    run_backtest(
        MACD_SMA200,
        data_file="BTC-USD-1d.csv",
        notes="MACD + SMA200 trend filter (ChartArt, daily)",
    )

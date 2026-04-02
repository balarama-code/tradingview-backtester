"""
Strategy: RSI + Bollinger Bands Breakout
Source: Common TradingView indicator pattern

Logic:
  BUY  when RSI < 30 AND price touches lower Bollinger Band
  SELL when RSI > 70 AND price touches upper Bollinger Band
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import ta
from backtesting import Strategy
from _backtest_runner import run_backtest


class RSI_BollingerBands(Strategy):
    # Parameters
    rsi_period = 14
    bb_period = 20
    bb_std = 2
    rsi_oversold = 30
    rsi_overbought = 70

    def init(self):
        close = pd.Series(self.data.Close)

        # RSI
        self.rsi = self.I(
            lambda x: ta.momentum.RSIIndicator(pd.Series(x), window=self.rsi_period).rsi(),
            self.data.Close,
            name="RSI",
        )

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=self.bb_period, window_dev=self.bb_std)
        self.bb_upper = self.I(lambda: bb.bollinger_hband().values, name="BB_Upper")
        self.bb_lower = self.I(lambda: bb.bollinger_lband().values, name="BB_Lower")

    def next(self):
        if not self.position:
            # BUY: RSI oversold + price at/below lower band
            if self.rsi[-1] < self.rsi_oversold and self.data.Close[-1] <= self.bb_lower[-1]:
                self.buy(size=0.95)
        else:
            # SELL: RSI overbought + price at/above upper band
            if self.rsi[-1] > self.rsi_overbought and self.data.Close[-1] >= self.bb_upper[-1]:
                self.position.close()


if __name__ == "__main__":
    # Run on BTC 1h
    run_backtest(
        RSI_BollingerBands,
        data_file="BTC-USD-1h.csv",
        notes="RSI<30 + lower BB buy, RSI>70 + upper BB sell",
    )
    # Run on BTC 1d
    run_backtest(
        RSI_BollingerBands,
        data_file="BTC-USD-1d.csv",
        notes="RSI<30 + lower BB buy, RSI>70 + upper BB sell (daily)",
    )

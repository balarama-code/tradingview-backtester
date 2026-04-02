"""
Strategy: Sniper Entry/Exit with SL & TP by KhanSaab V.02
Source: TradingView Pine Script by KhanSaab

Logic:
  BUY  when EMA9 crosses above EMA21 (only if not already long)
  EXIT when EMA9 crosses below EMA21 OR stop loss hit (ATR * 1.5 below entry)
  Uses 7-factor bull/bear scoring (VWAP, RSI, MACD, EMA, ADX, Volume, 5m RSI)
  as confirmation — only buy when bull score > bear score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import ta
from backtesting import Strategy
from _backtest_runner import run_backtest


class SniperKhanSaab(Strategy):
    # Parameters
    ema_fast = 9
    ema_slow = 21
    atr_period = 14
    atr_mult = 1.5
    rsi_period = 14

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # EMA 9 & 21
        self.ema9 = self.I(
            lambda x: ta.trend.EMAIndicator(pd.Series(x), window=self.ema_fast).ema_indicator(),
            self.data.Close, name="EMA9"
        )
        self.ema21 = self.I(
            lambda x: ta.trend.EMAIndicator(pd.Series(x), window=self.ema_slow).ema_indicator(),
            self.data.Close, name="EMA21"
        )

        # ATR for stop loss
        self.atr = self.I(
            lambda h, l, c: ta.volatility.AverageTrueRange(
                pd.Series(h), pd.Series(l), pd.Series(c), window=self.atr_period
            ).average_true_range(),
            self.data.High, self.data.Low, self.data.Close, name="ATR"
        )

        # RSI
        self.rsi = self.I(
            lambda x: ta.momentum.RSIIndicator(pd.Series(x), window=self.rsi_period).rsi(),
            self.data.Close, name="RSI"
        )

        # MACD
        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.macd_line = self.I(lambda: macd_obj.macd().values, name="MACD")
        self.macd_signal = self.I(lambda: macd_obj.macd_signal().values, name="MACD_Signal")

        # ADX
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
        self.adx = self.I(lambda: adx_obj.adx().values, name="ADX")

        # Volume SMA
        self.vol_avg = self.I(
            lambda v: pd.Series(v).rolling(20).mean(),
            self.data.Volume, name="VolAvg"
        )

    def next(self):
        price = self.data.Close[-1]
        ema9 = self.ema9[-1]
        ema21 = self.ema21[-1]
        atr_val = self.atr[-1]
        rsi = self.rsi[-1]
        macd = self.macd_line[-1]
        signal = self.macd_signal[-1]
        adx = self.adx[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        open_price = self.data.Open[-1]

        # Skip if indicators not ready
        if np.isnan(ema9) or np.isnan(ema21) or np.isnan(atr_val) or np.isnan(adx):
            return

        # EMA crossover detection
        if len(self.data.Close) < 2:
            return
        prev_ema9 = self.ema9[-2]
        prev_ema21 = self.ema21[-2]
        if np.isnan(prev_ema9) or np.isnan(prev_ema21):
            return

        buy_cross = prev_ema9 <= prev_ema21 and ema9 > ema21
        sell_cross = prev_ema9 >= prev_ema21 and ema9 < ema21

        # Bull score (6 factors — skip 5m RSI since we can't do multi-timeframe)
        bull_score = 0
        bull_score += 1 if rsi > 50 else 0
        bull_score += 1 if macd > signal else 0
        bull_score += 1 if ema9 > ema21 else 0
        bull_score += 1 if adx > 25 and price > ema9 else 0
        bull_score += 1 if vol > vol_avg and price > open_price else 0

        if not self.position:
            # BUY: EMA9 crosses above EMA21, bull score >= 3 (majority bullish)
            if buy_cross and bull_score >= 3:
                risk = atr_val * self.atr_mult
                sl = price - risk
                self.buy(size=0.95, sl=sl)
        else:
            # EXIT: EMA9 crosses below EMA21
            if sell_cross:
                self.position.close()


if __name__ == "__main__":
    # Run on BTC 1h
    run_backtest(
        SniperKhanSaab,
        data_file="BTC-USD-1h.csv",
        notes="EMA9/21 cross + bull score filter + ATR SL (KhanSaab Sniper)",
    )
    # Run on BTC 1d
    run_backtest(
        SniperKhanSaab,
        data_file="BTC-USD-1d.csv",
        notes="EMA9/21 cross + bull score filter + ATR SL (KhanSaab Sniper, daily)",
    )

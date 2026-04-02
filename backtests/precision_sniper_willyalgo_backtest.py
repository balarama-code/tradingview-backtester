"""
Strategy: Precision Sniper [WillyAlgoTrader] v1.1.0
Source: TradingView Pine Script by WillyAlgoTrader

Logic:
  10-factor confluence scoring engine:
    1. EMA fast > slow (+1)      2. Close > EMA trend (+1)
    3. RSI 50-75 (+1)            4. MACD histogram > 0 (+1)
    5. MACD > signal (+1)        6. Close > VWAP (+1)
    7. Volume above avg (+1)     8. ADX>20 & DI+>DI- (+1)
    9. Close > EMA fast (+0.5)   (HTF bias skipped = single TF)
  BUY  when EMA fast crosses above slow + bull momentum + score >= threshold
  SELL when EMA fast crosses below slow + bear momentum + score >= threshold
  SL: ATR-based with swing structure option, trailing after TP1/TP2 hit
  Uses "Default" preset for 1h, "Swing" preset for 1d (matches Auto mode)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import ta
from backtesting import Strategy
from _backtest_runner import run_backtest


class PrecisionSniper(Strategy):
    # Default preset (overridden per timeframe in __main__)
    ema_fast_len = 9
    ema_slow_len = 21
    ema_trend_len = 55
    rsi_len = 13
    atr_len = 14
    min_score = 5
    sl_mult = 1.5
    tp1_mult = 1.0
    tp2_mult = 2.0
    tp3_mult = 3.0
    swing_lookback = 10

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # EMAs
        self.ema_fast = self.I(
            lambda x: ta.trend.EMAIndicator(pd.Series(x), window=self.ema_fast_len).ema_indicator(),
            self.data.Close, name="EMA_Fast"
        )
        self.ema_slow = self.I(
            lambda x: ta.trend.EMAIndicator(pd.Series(x), window=self.ema_slow_len).ema_indicator(),
            self.data.Close, name="EMA_Slow"
        )
        self.ema_trend = self.I(
            lambda x: ta.trend.EMAIndicator(pd.Series(x), window=self.ema_trend_len).ema_indicator(),
            self.data.Close, name="EMA_Trend"
        )

        # RSI
        self.rsi = self.I(
            lambda x: ta.momentum.RSIIndicator(pd.Series(x), window=self.rsi_len).rsi(),
            self.data.Close, name="RSI"
        )

        # ATR
        self.atr = self.I(
            lambda h, l, c: ta.volatility.AverageTrueRange(
                pd.Series(h), pd.Series(l), pd.Series(c), window=self.atr_len
            ).average_true_range(),
            self.data.High, self.data.Low, self.data.Close, name="ATR"
        )

        # MACD
        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.macd_line = self.I(lambda: macd_obj.macd().values, name="MACD")
        self.macd_signal = self.I(lambda: macd_obj.macd_signal().values, name="MACD_Sig")
        self.macd_hist = self.I(lambda: macd_obj.macd_diff().values, name="MACD_Hist")

        # ADX
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
        self.adx = self.I(lambda: adx_obj.adx().values, name="ADX")
        self.di_plus = self.I(lambda: adx_obj.adx_pos().values, name="DI+")
        self.di_minus = self.I(lambda: adx_obj.adx_neg().values, name="DI-")

        # Volume SMA
        self.vol_avg = self.I(
            lambda v: pd.Series(v).rolling(20).mean(),
            self.data.Volume, name="VolAvg"
        )

        # VWAP approximation (cumulative typical price * volume / cumulative volume)
        tp = (np.array(self.data.High) + np.array(self.data.Low) + np.array(self.data.Close)) / 3.0
        cum_tpv = np.cumsum(tp * np.array(self.data.Volume))
        cum_vol = np.cumsum(np.array(self.data.Volume))
        vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, tp)
        self.vwap = self.I(lambda: vwap, name="VWAP")

    def next(self):
        if len(self.data.Close) < max(self.ema_trend_len, 50) + 2:
            return

        price = self.data.Close[-1]
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_t = self.ema_trend[-1]
        rsi = self.rsi[-1]
        atr_val = self.atr[-1]
        macd = self.macd_line[-1]
        macd_sig = self.macd_signal[-1]
        macd_hist = self.macd_hist[-1]
        adx = self.adx[-1]
        di_p = self.di_plus[-1]
        di_m = self.di_minus[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        vwap_val = self.vwap[-1]
        open_price = self.data.Open[-1]

        if any(np.isnan(x) for x in [ema_f, ema_s, ema_t, rsi, atr_val, adx]):
            return

        # EMA crossover
        prev_ema_f = self.ema_fast[-2]
        prev_ema_s = self.ema_slow[-2]
        if np.isnan(prev_ema_f) or np.isnan(prev_ema_s):
            return

        bull_cross = prev_ema_f <= prev_ema_s and ema_f > ema_s
        bear_cross = prev_ema_f >= prev_ema_s and ema_f < ema_s

        # Momentum
        bull_momentum = price > ema_f and price > ema_s
        bear_momentum = price < ema_f and price < ema_s

        # Bull confluence score (9 factors, max ~9.5 without HTF)
        bull_score = 0.0
        bull_score += 1.0 if ema_f > ema_s else 0.0
        bull_score += 1.0 if price > ema_t else 0.0
        bull_score += 1.0 if 50 < rsi < 75 else 0.0
        bull_score += 1.0 if macd_hist > 0 else 0.0
        bull_score += 1.0 if macd > macd_sig else 0.0
        bull_score += 1.0 if price > vwap_val else 0.0
        bull_score += 1.0 if vol > vol_avg * 1.2 else 0.0
        bull_score += 1.0 if adx > 20 and di_p > di_m else 0.0
        bull_score += 0.5 if price > ema_f else 0.0

        # Structure-based SL
        lookback = min(self.swing_lookback, len(self.data.Close) - 1)
        swing_low = min(self.data.Low[-i] for i in range(1, lookback + 1))

        if not self.position:
            # BUY: EMA cross + momentum + RSI not OB + score threshold
            if bull_cross and bull_momentum and rsi < 75 and bull_score >= self.min_score:
                risk = atr_val * self.sl_mult
                atr_sl = price - risk
                struct_sl = swing_low - atr_val * 0.2
                sl = max(atr_sl, struct_sl)
                min_dist = atr_val * 0.5
                if abs(price - sl) < min_dist:
                    sl = price - min_dist
                self.buy(size=0.95, sl=sl)
        else:
            # EXIT: bearish EMA cross + bear momentum + score
            bear_score = 0.0
            bear_score += 1.0 if ema_f < ema_s else 0.0
            bear_score += 1.0 if price < ema_t else 0.0
            bear_score += 1.0 if 25 < rsi < 50 else 0.0
            bear_score += 1.0 if macd_hist < 0 else 0.0
            bear_score += 1.0 if macd < macd_sig else 0.0
            bear_score += 1.0 if price < vwap_val else 0.0
            bear_score += 1.0 if vol > vol_avg * 1.2 else 0.0
            bear_score += 1.0 if adx > 20 and di_m > di_p else 0.0
            bear_score += 0.5 if price < ema_f else 0.0

            if bear_cross and bear_momentum and rsi > 25 and bear_score >= self.min_score:
                self.position.close()


if __name__ == "__main__":
    # 1h — Auto resolves to "Default" preset
    run_backtest(
        PrecisionSniper,
        data_file="BTC-USD-1h.csv",
        notes="10-factor confluence + EMA cross, Default preset (WillyAlgo Sniper)",
    )

    # 1d — Auto resolves to "Swing" preset
    class PrecisionSniper_Swing(PrecisionSniper):
        ema_fast_len = 13
        ema_slow_len = 34
        ema_trend_len = 89
        rsi_len = 21
        atr_len = 20
        min_score = 6
        sl_mult = 2.5

    run_backtest(
        PrecisionSniper_Swing,
        data_file="BTC-USD-1d.csv",
        notes="10-factor confluence + EMA cross, Swing preset (WillyAlgo Sniper, daily)",
    )

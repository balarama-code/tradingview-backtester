# TradingView Indicator Backtester

Automated backtesting framework for TradingView Pine Script indicators on crypto data.

## Structure

```
tradingview/
    README.md               # You're here
    backtest_results.csv    # Master results log
    pine_scripts/           # Raw Pine Script source code (.pine files)
    backtests/              # Python backtest files (_backtest.py)
    data/                   # OHLCV data
        BTC-USD-1h.csv      # 1-hour candles
        BTC-USD-1d.csv      # Daily candles
        ETH-USD-1h.csv
        ETH-USD-1d.csv
    errors.log              # Failed conversions
```

## Workflow

1. Extract Pine Script indicators from TradingView (Editors Picks, Top, Trending)
2. Convert each to Python using backtesting.py
3. Run backtests on BTC/ETH data
4. Log results to CSV and code comments
5. Commit after EACH backtest

## Rules (LOCKED IN)

1. **NO FAST-TRACKING** - Every indicator gets evaluated
2. **Skip ONLY** true visualization tools (volume profiles, drawing tools, etc.)
3. **Be LOOSE** with "backtestable" criteria - Don't skip opportunities

## Results Format

CSV columns: `strategy_name, return_pct, sharpe_ratio, max_drawdown, win_rate, num_trades, timeframe, notes`

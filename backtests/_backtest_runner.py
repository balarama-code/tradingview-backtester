"""
Backtest Runner — loads data, runs a strategy, logs results to CSV.

Usage:
    from _backtest_runner import run_backtest
    run_backtest(MyStrategy, data_file="BTC-USD-1h.csv", cash=100_000)
"""

import csv
import os
import sys
from datetime import datetime

import pandas as pd
from backtesting import Backtest
from backtesting.lib import FractionalBacktest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "backtest_results.csv")


def load_data(filename: str) -> pd.DataFrame:
    """Load OHLCV CSV and prepare for backtesting.py (needs Open, High, Low, Close, Volume columns)."""
    path = os.path.join(DATA_DIR, filename)

    # yfinance saves with multi-row headers — skip ticker/datetime rows
    df = pd.read_csv(path, header=0)

    # Drop rows where 'Close' is not numeric (header artifacts)
    first_col = df.columns[0]
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()].copy()

    # Parse datetime index
    df.index = pd.to_datetime(df[first_col], utc=True)
    df.index.name = "Date"
    df.drop(columns=[first_col], inplace=True)

    # Cast to float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


def run_backtest(
    strategy_class,
    data_file: str = "BTC-USD-1h.csv",
    cash: float = 10_000_000,
    commission: float = 0.001,
    exclusive_orders: bool = True,
    trade_on_close: bool = True,
    notes: str = "",
):
    """Run backtest and log results."""
    strategy_name = strategy_class.__name__
    timeframe = data_file.replace(".csv", "").split("-")[-1]

    print(f"\n{'='*60}")
    print(f"  Backtesting: {strategy_name}")
    print(f"  Data: {data_file}  |  Cash: ${cash:,.0f}  |  Commission: {commission*100:.1f}%")
    print(f"{'='*60}\n")

    df = load_data(data_file)
    print(f"  Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})\n")

    bt = Backtest(
        df,
        strategy_class,
        cash=cash,
        commission=commission,
        exclusive_orders=exclusive_orders,
        trade_on_close=trade_on_close,
    )
    stats = bt.run()

    # Print key results
    ret = stats["Return [%]"]
    sharpe = stats.get("Sharpe Ratio", 0) or 0
    maxdd = stats.get("Max. Drawdown [%]", 0) or 0
    winrate = stats.get("Win Rate [%]", 0) or 0
    trades = stats.get("# Trades", 0) or 0

    print(f"  Results:")
    print(f"  {'Return:':<20} {ret:>10.2f}%")
    print(f"  {'Sharpe Ratio:':<20} {sharpe:>10.2f}")
    print(f"  {'Max Drawdown:':<20} {maxdd:>10.2f}%")
    print(f"  {'Win Rate:':<20} {winrate:>10.2f}%")
    print(f"  {'Trades:':<20} {trades:>10}")
    print(f"\n{'='*60}\n")

    # Log to CSV
    row = {
        "strategy_name": strategy_name,
        "return_pct": round(ret, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(maxdd, 2),
        "win_rate": round(winrate, 2),
        "num_trades": int(trades),
        "timeframe": timeframe,
        "notes": notes,
    }

    file_exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 50
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Results appended to backtest_results.csv")
    return stats

import numpy as np
import pandas as pd


def run(signals: pd.DataFrame, holding_period: int = 4, cost_bps: float = 5.0) -> dict:
    """
    holding_period: subsample every N timestamps to avoid overlapping returns.
                    Must match the forward_return horizon in engineer.py (default 4h).
    cost_bps:       roundtrip transaction cost per rebalance in basis points (default 5 bps).
    """
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    # Only evaluate non-overlapping windows
    timestamps = timestamps[::holding_period]

    portfolio_returns = []
    active_timestamps = []

    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        if len(longs) == 0:
            continue
        period_return = longs["forward_return"].mean() - cost_bps / 10_000
        portfolio_returns.append(period_return)
        active_timestamps.append(ts)

    returns_series = pd.Series(portfolio_returns, index=active_timestamps)

    equity_full = (1 + returns_series).cumprod()

    total_return = float(equity_full.iloc[-1] - 1.0)

    n = len(returns_series)
    # Annualize: 6 market bars/day, last `holding_period` bars each day have no valid target
    periods_per_year = 252 * (6 - holding_period)
    if n > 1 and returns_series.std() != 0:
        sharpe = float(returns_series.mean() / returns_series.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    rolling_max = equity_full.cummax()
    drawdowns = (equity_full - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    win_rate = float((returns_series > 0).sum() / n) if n > 0 else 0.0

    n_periods = int(n)

    return {
        "equity_curve": equity_full,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_periods": n_periods,
        "holding_period": holding_period,
        "cost_bps": cost_bps,
    }


def trade_log(signals: pd.DataFrame, holding_period: int = 4, cost_bps: float = 5.0,
              sector_map: dict = None) -> pd.DataFrame:
    """Return a DataFrame with one row per executed long trade."""
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    timestamps = timestamps[::holding_period]

    rows = []
    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        for symbol, row in longs.iterrows():
            rows.append({
                "entry_time":  ts,
                "symbol":      symbol,
                "sector":      sector_map.get(symbol, "Unknown") if sector_map else "Unknown",
                "rank":        int(row["rank"]),
                "proba":       round(float(row["proba"]), 4),
                "return":      round(float(row["forward_return"]), 6),
                "net_return":  round(float(row["forward_return"]) - cost_bps / 10_000, 6),
                "win":         float(row["forward_return"]) > 0,
            })

    return pd.DataFrame(rows)


def trade_summary(log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade log by symbol — times selected, avg return, win rate, total PnL."""
    summary = (
        log.groupby("symbol")
        .agg(
            sector=("sector", "first"),
            n_trades=("net_return", "count"),
            avg_return_pct=("net_return", lambda x: round(x.mean() * 100, 4)),
            win_rate=("win", lambda x: round(x.mean() * 100, 1)),
            total_pnl_pct=("net_return", lambda x: round(x.sum() * 100, 4)),
            best_trade_pct=("return", lambda x: round(x.max() * 100, 4)),
            worst_trade_pct=("return", lambda x: round(x.min() * 100, 4)),
        )
        .sort_values("total_pnl_pct", ascending=False)
        .reset_index()
    )
    return summary


def print_summary(results: dict) -> None:
    eq = results["equity_curve"]
    print(f"Equity Curve Start : {eq.index[0]}  |  Value: {eq.iloc[0]:.4f}")
    print(f"Equity Curve End   : {eq.index[-1]}  |  Value: {eq.iloc[-1]:.4f}")
    print(f"Total Return       : {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
    print(f"Sharpe Ratio       : {results['sharpe']:.4f}")
    print(f"Max Drawdown       : {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")
    print(f"Win Rate           : {results['win_rate']:.4f} ({results['win_rate']*100:.2f}%)")
    print(f"N Periods          : {results['n_periods']}")
    print(f"Holding Period     : {results.get('holding_period', '?')}h  |  Cost: {results.get('cost_bps', '?')} bps/trade")


if __name__ == "__main__":
    np.random.seed(42)

    n_symbols = 20
    n_timestamps = 300
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    timestamps = pd.date_range(start="2024-01-01", periods=n_timestamps, freq="h")

    index = pd.MultiIndex.from_product([timestamps, symbols], names=["datetime", "symbol"])
    n_rows = len(index)

    raw_signals = np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    forward_returns = np.random.normal(loc=0.001, scale=0.01, size=n_rows)

    signals = pd.DataFrame(
        {"signal": raw_signals, "forward_return": forward_returns},
        index=index,
    )

    results = run(signals)
    print_summary(results)

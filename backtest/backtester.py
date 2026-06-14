import numpy as np
import pandas as pd


def run(signals: pd.DataFrame, holding_period: int = 1, cost_bps: float = 2.0,
        capital: float = 100_000.0) -> dict:
    """
    capital:        starting portfolio value in dollars.
    holding_period: bars between rebalances (1 = every bar, non-overlapping).
    cost_bps:       roundtrip bid-ask cost per position in basis points.
                    Applied per stock: each position pays cost_bps on entry + exit.
                    Portfolio-level cost per period = cost_bps (equal-weight averaging).
    """
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    timestamps = timestamps[::holding_period]

    portfolio_value = capital
    period_records = []

    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        if len(longs) == 0:
            continue

        n_pos = len(longs)
        position_size = portfolio_value / n_pos          # dollars per stock

        # Per-stock net return: gross return minus roundtrip spread cost
        gross = longs["forward_return"]
        net   = gross - cost_bps / 10_000               # cost applied per position

        period_gross_ret = gross.mean()
        period_net_ret   = net.mean()
        dollar_pnl       = portfolio_value * period_net_ret
        dollar_cost      = portfolio_value * (cost_bps / 10_000)   # total cost this period

        portfolio_value  = portfolio_value * (1 + period_net_ret)

        period_records.append({
            "timestamp":       ts,
            "n_positions":     n_pos,
            "position_size":   round(position_size, 2),
            "gross_return":    period_gross_ret,
            "net_return":      period_net_ret,
            "dollar_pnl":      round(dollar_pnl, 2),
            "dollar_cost":     round(dollar_cost, 2),
            "portfolio_value": round(portfolio_value, 2),
            "win":             period_net_ret > 0,
        })

    if not period_records:
        return {}

    periods_df = pd.DataFrame(period_records).set_index("timestamp")
    equity = periods_df["portfolio_value"]

    total_dollar_pnl = portfolio_value - capital
    total_return     = total_dollar_pnl / capital

    n = len(periods_df)
    periods_per_year = 252 * (6 - holding_period)
    net_rets = periods_df["net_return"]
    sharpe = float(net_rets.mean() / net_rets.std() * np.sqrt(periods_per_year)) \
             if n > 1 and net_rets.std() != 0 else 0.0

    rolling_max  = equity.cummax()
    max_drawdown = float(((equity - rolling_max) / rolling_max).min())
    max_dd_dollar = float((equity - rolling_max).min())

    win_rate = float(periods_df["win"].mean())

    return {
        "equity_curve":      equity,
        "periods_df":        periods_df,
        "capital":           capital,
        "final_value":       round(portfolio_value, 2),
        "total_dollar_pnl":  round(total_dollar_pnl, 2),
        "total_return":      total_return,
        "sharpe":            sharpe,
        "max_drawdown":      max_drawdown,
        "max_dd_dollar":     round(max_dd_dollar, 2),
        "win_rate":          win_rate,
        "n_periods":         n,
        "holding_period":    holding_period,
        "cost_bps":          cost_bps,
        "total_cost_dollars": round(periods_df["dollar_cost"].sum(), 2),
    }


def trade_log(signals: pd.DataFrame, holding_period: int = 1, cost_bps: float = 2.0,
              capital: float = 100_000.0, sector_map: dict = None) -> pd.DataFrame:
    """One row per executed long position. Includes dollar P&L based on equal-weight sizing."""
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    timestamps = timestamps[::holding_period]

    portfolio_value = capital
    rows = []

    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        if len(longs) == 0:
            continue

        n_pos = len(longs)
        position_size = portfolio_value / n_pos

        period_net = longs["forward_return"].mean() - cost_bps / 10_000
        portfolio_value = portfolio_value * (1 + period_net)

        for symbol, row in longs.iterrows():
            gross_ret = float(row["forward_return"])
            net_ret   = gross_ret - cost_bps / 10_000
            rows.append({
                "entry_time":    ts,
                "symbol":        symbol,
                "sector":        sector_map.get(symbol, "Unknown") if sector_map else "Unknown",
                "rank":          int(row["rank"]),
                "proba":         round(float(row["proba"]), 4),
                "position_size": round(position_size, 2),
                "gross_return":  round(gross_ret, 6),
                "cost_bps":      cost_bps,
                "net_return":    round(net_ret, 6),
                "dollar_pnl":    round(position_size * net_ret, 2),
                "win":           net_ret > 0,
            })

    return pd.DataFrame(rows)


def trade_summary(log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by symbol — showing both % and $ metrics."""
    summary = (
        log.groupby("symbol")
        .agg(
            sector=("sector", "first"),
            n_trades=("net_return", "count"),
            avg_return_pct=("net_return", lambda x: round(x.mean() * 100, 4)),
            win_rate=("win", lambda x: round(x.mean() * 100, 1)),
            total_dollar_pnl=("dollar_pnl", lambda x: round(x.sum(), 2)),
            avg_dollar_pnl=("dollar_pnl", lambda x: round(x.mean(), 2)),
            best_trade_pct=("gross_return", lambda x: round(x.max() * 100, 4)),
            worst_trade_pct=("gross_return", lambda x: round(x.min() * 100, 4)),
        )
        .sort_values("total_dollar_pnl", ascending=False)
        .reset_index()
    )
    return summary


def print_summary(results: dict) -> None:
    c = results["capital"]
    f = results["final_value"]
    print(f"Starting Capital   : ${c:>12,.2f}")
    print(f"Final Value        : ${f:>12,.2f}")
    print(f"Total P&L          : ${results['total_dollar_pnl']:>+12,.2f}  ({results['total_return']*100:.2f}%)")
    print(f"Total Cost Paid    : ${results['total_cost_dollars']:>12,.2f}")
    print(f"Sharpe Ratio       : {results['sharpe']:.4f}")
    print(f"Max Drawdown       : ${results['max_dd_dollar']:>+12,.2f}  ({results['max_drawdown']*100:.2f}%)")
    print(f"Win Rate           : {results['win_rate']*100:.2f}%  ({results['n_periods']} periods)")
    print(f"Holding Period     : {results['holding_period']}h  |  Cost/position: {results['cost_bps']} bps")

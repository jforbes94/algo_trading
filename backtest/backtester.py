import numpy as np
import pandas as pd


def run(signals: pd.DataFrame, holding_period: int = 1, cost_bps: float = 2.0,
        capital: float = 100_000.0, rebalance: bool = False,
        hold_rank: int = None, kelly: bool = False) -> dict:
    """
    capital:        starting portfolio value in dollars.
    holding_period: bars between rebalances (ignored when rebalance=True).
    cost_bps:       roundtrip bid-ask spread per position in basis points.
    rebalance:      only pay cost when positions enter/exit; holds are free.
    hold_rank:      (requires rebalance=True) keep a held position until its rank
                    exceeds this threshold, reducing unnecessary turnover.
    kelly:          size positions by model edge (2*proba−1) rather than equal weight.
    """
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    if not rebalance:
        timestamps = timestamps[::holding_period]

    portfolio_value = capital
    active_set: set = set()
    active_weights: dict = {}
    period_records = []

    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        if len(longs) == 0:
            if rebalance:
                active_set = set()
                active_weights = {}
            continue

        # Build target portfolio: top-N entries + hold-band retentions
        if hold_rank is not None and rebalance:
            in_band = set(ts_data[ts_data["rank"] <= hold_rank].index)
            still_held = (active_set & in_band) & set(ts_data.index)
            target_set = still_held | set(longs.index)
            target_data = ts_data.loc[sorted(target_set)]
        else:
            target_set = set(longs.index)
            target_data = longs

        n_pos = len(target_set)

        # Position weights: Kelly (edge-proportional) or equal
        if kelly:
            edges = (2.0 * target_data["proba"] - 1.0).clip(lower=1e-6)
            weights = edges / edges.sum()
        else:
            weights = pd.Series(1.0 / n_pos, index=target_data.index)

        period_gross_ret = float((target_data["forward_return"] * weights).sum())

        if rebalance:
            entering = target_set - active_set
            exiting  = active_set - target_set
            n_prev   = len(active_set)

            # Cost proportional to each position's portfolio weight
            exit_cost  = sum(
                active_weights.get(s, 1.0 / max(n_prev, 1)) * (cost_bps / 2)
                for s in exiting
            ) / 10_000
            entry_cost = float(
                weights.reindex(list(entering)).sum() * (cost_bps / 2)
            ) / 10_000
            period_cost = exit_cost + entry_cost

            active_set     = target_set
            active_weights = dict(zip(weights.index, weights.values))
        else:
            period_cost = cost_bps / 10_000

        period_net_ret  = period_gross_ret - period_cost
        dollar_pnl      = portfolio_value * period_net_ret
        dollar_cost     = portfolio_value * period_cost
        portfolio_value = portfolio_value * (1 + period_net_ret)

        period_records.append({
            "timestamp":       ts,
            "n_positions":     n_pos,
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
    periods_per_year = 252 * 5 if rebalance else 252 * (6 - holding_period)
    net_rets = periods_df["net_return"]
    sharpe = float(net_rets.mean() / net_rets.std() * np.sqrt(periods_per_year)) \
             if n > 1 and net_rets.std() != 0 else 0.0

    rolling_max  = equity.cummax()
    max_drawdown = float(((equity - rolling_max) / rolling_max).min())
    max_dd_dollar = float((equity - rolling_max).min())
    win_rate = float(periods_df["win"].mean())

    return {
        "equity_curve":       equity,
        "periods_df":         periods_df,
        "capital":            capital,
        "final_value":        round(portfolio_value, 2),
        "total_dollar_pnl":   round(total_dollar_pnl, 2),
        "total_return":       total_return,
        "sharpe":             sharpe,
        "max_drawdown":       max_drawdown,
        "max_dd_dollar":      round(max_dd_dollar, 2),
        "win_rate":           win_rate,
        "n_periods":          n,
        "holding_period":     1 if rebalance else holding_period,
        "cost_bps":           cost_bps,
        "total_cost_dollars": round(periods_df["dollar_cost"].sum(), 2),
        "rebalance":          rebalance,
        "hold_rank":          hold_rank,
        "kelly":              kelly,
    }


def trade_log(signals: pd.DataFrame, holding_period: int = 1, cost_bps: float = 2.0,
              capital: float = 100_000.0, sector_map: dict = None,
              rebalance: bool = False, hold_rank: int = None,
              kelly: bool = False) -> pd.DataFrame:
    """One row per (bar × long position).

    rebalance=False: cost_bps charged on every position every bar.
    rebalance=True:  cost_bps charged only on new entries; holds are free.
    hold_rank:       keep held positions until rank > hold_rank.
    kelly:           position weight ∝ model edge (2*proba−1).
    """
    timestamps = signals.index.get_level_values(0).unique().sort_values()
    if not rebalance:
        timestamps = timestamps[::holding_period]

    portfolio_value = capital
    active_set: set = set()
    active_weights: dict = {}
    rows = []

    for ts in timestamps:
        ts_data = signals.loc[ts]
        longs = ts_data[ts_data["signal"] == 1]
        if len(longs) == 0:
            if rebalance:
                active_set = set()
                active_weights = {}
            continue

        if hold_rank is not None and rebalance:
            in_band = set(ts_data[ts_data["rank"] <= hold_rank].index)
            still_held = (active_set & in_band) & set(ts_data.index)
            target_set = still_held | set(longs.index)
            target_data = ts_data.loc[sorted(target_set)]
        else:
            target_set = set(longs.index)
            target_data = longs

        n_pos = len(target_set)

        if kelly:
            edges = (2.0 * target_data["proba"] - 1.0).clip(lower=1e-6)
            weights = edges / edges.sum()
        else:
            weights = pd.Series(1.0 / n_pos, index=target_data.index)

        if rebalance:
            entering = target_set - active_set
            exiting  = active_set - target_set
            n_prev   = len(active_set)
            exit_cost  = sum(
                active_weights.get(s, 1.0 / max(n_prev, 1)) * (cost_bps / 2)
                for s in exiting
            ) / 10_000
            entry_cost = float(
                weights.reindex(list(entering)).sum() * (cost_bps / 2)
            ) / 10_000
            period_cost = exit_cost + entry_cost
        else:
            entering = target_set
            period_cost = cost_bps / 10_000

        period_gross_ret = float((target_data["forward_return"] * weights).sum())
        period_net = period_gross_ret - period_cost

        # Capture position sizes at start of period (before NAV update)
        pos_sizes = {sym: portfolio_value * float(weights[sym]) for sym in target_set}
        portfolio_value *= (1 + period_net)

        for symbol, row in target_data.iterrows():
            gross_ret = float(row["forward_return"])
            w         = float(weights[symbol])
            pos_size  = pos_sizes[symbol]
            is_new    = symbol in entering
            pos_cost  = (cost_bps / 10_000) if is_new else 0.0
            net_ret   = gross_ret - pos_cost
            rows.append({
                "entry_time":    ts,
                "symbol":        symbol,
                "sector":        sector_map.get(symbol, "Unknown") if sector_map else "Unknown",
                "rank":          int(row["rank"]),
                "proba":         round(float(row["proba"]), 4),
                "weight":        round(w, 4),
                "position_size": round(pos_size, 2),
                "gross_return":  round(gross_ret, 6),
                "cost_bps":      cost_bps if is_new else 0.0,
                "net_return":    round(net_ret, 6),
                "dollar_pnl":    round(pos_size * net_ret, 2),
                "win":           net_ret > 0,
                "is_new_entry":  is_new,
            })

        if rebalance:
            active_set     = target_set
            active_weights = dict(zip(weights.index, weights.values))

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
    parts = []
    if results.get("rebalance"):
        parts.append("rebalance")
    else:
        parts.append(f"hold {results['holding_period']}h")
    if results.get("hold_rank"):
        parts.append(f"hold_band≤{results['hold_rank']}")
    if results.get("kelly"):
        parts.append("kelly")
    mode = " | ".join(parts)

    print(f"Mode               : {mode}")
    print(f"Starting Capital   : ${c:>12,.2f}")
    print(f"Final Value        : ${f:>12,.2f}")
    print(f"Total P&L          : ${results['total_dollar_pnl']:>+12,.2f}  ({results['total_return']*100:.2f}%)")
    print(f"Total Cost Paid    : ${results['total_cost_dollars']:>12,.2f}")
    print(f"Sharpe Ratio       : {results['sharpe']:.4f}")
    print(f"Max Drawdown       : ${results['max_dd_dollar']:>+12,.2f}  ({results['max_drawdown']*100:.2f}%)")
    print(f"Win Rate           : {results['win_rate']*100:.2f}%  ({results['n_periods']} periods)")
    print(f"Cost/position      : {results['cost_bps']} bps  |  Mode: {mode}")

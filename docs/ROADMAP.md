# Roadmap

## Completed

- [x] Data layer: fetcher, store, parquet cache, incremental refresh
- [x] Universe: ~501 S&P 500 tickers + GICS sector map
- [x] Bulk download and backfill (up to 5 years of 1h history)
- [x] Sanity check script (duplicates, bad OHLC values, gap detection)
- [x] Market hours filter: drop extended-hours bars (4am–10am, 3pm–8pm ET)
- [x] Feature engineering: 28-feature panel (momentum, trend, volatility, volume,
      calendar, price action, cross-sectional ranks, vs-SPY, vs-sector)
- [x] Time-aware forward_return: uses index lookup at `t+1h` (not positional shift),
      so labels never span overnight gaps
- [x] Cross-sectional target variable (`y = 1` if stock outperforms median at timestamp)
- [x] Walk-forward LightGBM model (4 folds, out-of-sample predictions only)
- [x] Signal generation: top/bottom N per timestamp by model probability
- [x] Backtester: equity curve, Sharpe, max drawdown, win rate, trade log
- [x] Parameterized backtest notebook with labeled output folders
- [x] Transaction costs: 5 bps roundtrip per rebalance
- [x] ETF universe (SPY + 11 GICS sector ETFs) for cross-asset features
- [x] Leakage investigation: root cause was extended-hours data (49.4% of raw bars);
      fixed by market hours filter
- [x] 1h holding period (5 valid entry slots/day: 10am–2pm ET)
- [x] Correct annualization: `252 * (6 - holding_period)` periods/year

---

## Next Up

### Tier 1 — Signal quality (highest priority)

1. **Feature engineering tier 2**
   - Time-of-day normalized volume (`vol_vs_hour_norm`)
   - Intraday cumulative return from open (`ret_from_open`)
   - Candlestick body/wick ratios (`body_ratio`, `upper_wick`, `lower_wick`)
   - High-low spread as bid-ask proxy (`hl_spread_norm`)
   - Cross-sectional return dispersion (regime signal)
   - Volume-direction interaction (`vol_ratio * sign(close - open)`)

2. **Model objective: ranking**
   - Switch from binary classification to LightGBM `lambdarank`
   - Label: quantile rank of forward_return (not binary above/below median)
   - Optimize NDCG@20 directly — this is the portfolio construction objective

3. **Purged walk-forward CV**
   - Add embargo of `holding_period` bars at each fold boundary
   - Prevents label overlap leakage at fold transitions

### Tier 2 — Robustness

4. **Hyperparameter search** with Optuna (learning rate, num_leaves, min_data_in_leaf)
5. **SHAP-driven feature pruning** — drop bottom-quartile features per fold
6. **Survivorship bias mitigation** — point-in-time S&P 500 constituent data

### Tier 3 — Execution

7. **Execution layer**: Alpaca order placement (paper trading)
8. **Scheduler**: run pipeline on market open, rebalance each hour
9. **Risk management**: position sizing, max drawdown circuit breaker, per-stock limits
10. **Live monitoring**: equity curve dashboard, alert on drawdown threshold

---

## Future Considerations

- **Alternative data**: earnings calendar proximity, news sentiment scores
- **Options market signals**: put/call ratio, implied vol surface
- **Regime detection**: HMM or clustering on macro features to condition strategy
- **Daily bar variant**: cleaner strategy, avoids all intraday timestamp complexity

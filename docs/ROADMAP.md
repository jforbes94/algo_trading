# Roadmap

## Completed

### Data Layer
- [x] Fetcher, store, parquet cache, incremental refresh (Alpaca API)
- [x] Universe: ~501 S&P 500 tickers + GICS sector map
- [x] ETF universe: SPY + 11 GICS sector ETFs for cross-asset features
- [x] Bulk download + backfill (up to 5 years of 1h history)
- [x] Sanity check script (duplicates, bad OHLC values, gap detection)
- [x] Market hours filter: keep 14:00–19:30 UTC (10am–3:30pm ET) only
- [x] **Daily macro data layer** (`data/macro_fetcher.py`)
  - Yield curve (10y−short rate), HY spread proxy via HYG/LQD
  - VIX term structure (^VIX, ^VIX3M): level, ratio, 1-day change
  - Economic calendar flags: FOMC day, NFP day, CPI day (2021–2026)
  - Earnings proximity: days to/from earnings per stock (yfinance)
  - Cached to `data/cache/daily/`; auto-refreshes daily/weekly

### Feature Engineering (38 features total)
- [x] **Tier 1 — 28 features**: momentum (1h/4h/1d/5d), RSI, MACD, Bollinger, realized vol,
      VWAP deviation, volume ratio, candlestick range position, vol regime,
      RSI acceleration, overnight gap, market breadth, cross-sectional ranks,
      SPY-relative returns, sector-relative returns, VIX proxy
- [x] **Tier 2 — 10 features**: intraday-hour-normalized volume, return from open,
      candlestick body/wick ratios, signed volume direction, VWAP drift from open,
      intraday range rank, cross-sectional return dispersion, sector-relative open return
- [x] **Daily features — 17 features** (merged from macro_fetcher):
      yield_10y, yield_curve, hy_spread, vix_level, vix_3m, vix_term_ratio, vix_1d_chg,
      fomc_day, nfp_day, cpi_day, macro_event_day,
      days_to_earnings, days_from_earnings, pre_earnings_5d, post_earnings_2d, earnings_week
- [x] Time-aware `forward_return`: index lookup at `t+1h` (not positional shift)
- [x] Cross-sectional label: `y = 1` if stock beats median return at that timestamp
- [x] Feature correlation report (`scripts/feature_correlation_report.py`):
      IC, ICIR, Pearson/Spearman, year-by-year heatmap, redundancy matrix

### Model & Backtest
- [x] Walk-forward LightGBM (4 folds, out-of-sample predictions only)
- [x] Signal generation: top-N per timestamp by model probability
- [x] Dollar-based backtester: equity curve, Sharpe, max drawdown, win rate
- [x] Trade log with per-position dollar P&L, position sizing
- [x] PDF trade report: 3-page visual (equity curve, rolling Sharpe, per-stock breakdown)
- [x] Parameterized backtest notebook with labeled output folders
- [x] Transaction costs: 2 bps per position (roundtrip)
- [x] Correct annualization: `252 × (6 − holding_period)` periods/year

### Diagnostics
- [x] Statistical significance testing: win rate p-values, sample size analysis
- [x] Leakage investigation: root cause was extended-hours bars (49.4% of data)
- [x] Bar timestamp & shift validation scripts
- [x] Feature correlation analysis confirming near-zero IC on tier-1 features

---

## Current Results (2021–2026, 5 years, top-5 positions, 1h hold, 2 bps/pos)

| Metric | Tier-1 only | + Tier-2 features |
|--------|------------|-------------------|
| Win Rate | 50.04% | 50.46% |
| Sharpe | −0.44 | −0.06 |
| Total P&L | −$30,433 | −$9,079 |
| Gross P&L (pre-cost) | +$54k | +$88.5k |
| Max Drawdown | −$41,957 | −$24,828 |

**Key finding**: Gross alpha exists (+$88.5k over 5 years); transaction costs ($97.6k) consume it.
Model has real signal; cost structure and objective function are the primary bottlenecks.

---

## Next Up

### Priority 1 — Model methodology (highest expected impact)

1. **Quintile labels** — replace binary `y` (beats median) with within-timestamp quintile rank
   (0=bottom, 4=top). 4× the discriminative resolution. ~3 lines in `engineer.py`.

2. **Walk-forward embargo** — add 5-timestamp (~1 trading day) gap at each fold boundary.
   Eliminates soft leakage from rolling features spanning the train/test split.
   Diagnostic: if AUC drops after adding this, prior signal was boundary leakage.

3. **Increased regularization + early stopping** — `min_child_samples=200`, `reg_lambda=1.0`,
   `learning_rate=0.02`, `n_estimators=500`, early stopping on held-out temporal slice.

4. **LambdaRank objective** — switch from binary cross-entropy to `rank_xendcg`/`lambdarank`.
   Model currently never sees the timestamp "query group"; LambdaRank optimizes
   cross-sectional ordering directly.

### Priority 2 — Signal exploration

5. **Ridge regression baseline** — 2-line change. If Ridge matches LightGBM, tree complexity
   is wasted on noise and the signal is linear.

6. **Top-1 position sizing** — only trade the single highest-confidence pick per period.
   Reduces costs dramatically; concentrates into the model's best guess.

7. **Daily feature IC analysis** — run correlation report on the 17 new daily features to
   confirm which earn positive IC (especially earnings proximity and VIX term structure).

### Priority 3 — Robustness

8. **Hyperparameter search** with Optuna
9. **SHAP-driven feature pruning** — drop bottom-quartile features per fold
10. **Survivorship bias mitigation** — point-in-time S&P 500 constituent data
11. **Time-decay sample weights** — down-weight 2021–2022 observations for 2025 regime

### Priority 4 — Execution layer

12. Alpaca order placement (paper trading)
13. Hourly scheduler: run pipeline on market open, rebalance each hour
14. Risk management: position sizing, max drawdown circuit breaker
15. Live monitoring: equity curve dashboard, drawdown alerts

---

## Future Considerations

- **News sentiment**: per-stock daily headline sentiment (NewsAPI, Benzinga)
- **Options signals**: put/call ratio per stock, IV rank, unusual options activity
- **Regime detection**: HMM or clustering on macro features to condition strategy
- **Sector models**: separate LightGBM per GICS sector (sparse, but sector dynamics differ)
- **Daily bar variant**: cleaner strategy, avoids all intraday timestamp complexity
- **Short interest**: FINRA bi-monthly, useful as a mean-reversion signal

---

## Known Technical Debt

- `_filter_market_hours()` is duplicated between `data/store.py` and `features/engineer.py` — should be moved to a shared `data/utils.py`
- ~~`scripts/` contains a mix of operational and one-off diagnostic scripts~~ — resolved: one-off diagnostics moved to `scripts/analysis/`
- `data/macro_fetcher.py` downloads full history on each refresh rather than incrementally appending — fast enough for daily data but inconsistent with `store.py` pattern
- HY spread is a proxy (rolling vol of HYG−LQD returns) rather than true OAS bps — acceptable but less precise than FRED series
- Short rate in `yield_curve` uses ^IRX (13-week T-bill) rather than 2-year Treasury (no free 2yr ticker on Yahoo Finance)

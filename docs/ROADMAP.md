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

### Feature Engineering (55 features total)
- [x] **Tier 1 — 28 features**: momentum (1h/4h/1d/5d), RSI, MACD, Bollinger, realized vol,
      VWAP deviation, volume ratio, candlestick range position, vol regime,
      RSI acceleration, overnight gap, market breadth, cross-sectional ranks,
      SPY-relative returns, sector-relative returns, VIX proxy
- [x] **Tier 2 — 10 features**: intraday-hour-normalized volume, return from open,
      candlestick body/wick ratios, signed volume direction, VWAP drift from open,
      intraday range rank, cross-sectional return dispersion, sector-relative open return
- [x] **Daily features — 17 features** (merged from macro_fetcher, now active in notebook and all scripts):
      yield_10y, yield_curve, hy_spread, vix_level, vix_3m, vix_term_ratio, vix_1d_chg,
      fomc_day, nfp_day, cpi_day, macro_event_day,
      days_to_earnings, days_from_earnings, pre_earnings_5d, post_earnings_2d, earnings_week
- [x] Time-aware `forward_return`: index lookup at `t+1h` (not positional shift)
- [x] **Quintile label**: `y ∈ {0,1,2,3,4}` within-timestamp quintile rank of forward return.
      Replaced binary beat-median label. Model now optimizes for genuine outperformers,
      not median crossers. (`features/engineer.py`)
- [x] Feature correlation report (`scripts/feature_correlation_report.py`):
      IC, ICIR, Pearson/Spearman, year-by-year heatmap, redundancy matrix

### Model & Backtest
- [x] Walk-forward LightGBM (4 folds, out-of-sample predictions only)
- [x] **Switched to `LGBMRegressor`** (`objective="regression"`) — predicts quintile score
      directly. Fold metric is now mean IC (Spearman rank correlation) not AUC.
- [x] Signal generation: top-N per timestamp by model score
- [x] Dollar-based backtester: equity curve, Sharpe, max drawdown, win rate
- [x] Trade log with per-position dollar P&L, Kelly weight, position sizing
- [x] PDF trade report: 3-page visual (equity curve, rolling Sharpe, per-stock breakdown)
- [x] Parameterized backtest notebook with labeled output folders
- [x] Transaction costs: 2 bps per position (roundtrip)
- [x] Correct annualization: `252 × (6 − holding_period)` periods/year
- [x] **Rebalancing cost model** — charge cost only on entries/exits, not held positions
- [x] **Hold band** (`HOLD_RANK=25`) — hold until rank drops below 25, not just below TOP_N
- [x] **Kelly position sizing** — weight ∝ model edge `(2×score − 1)`, normalized per bar

### Diagnostics
- [x] Statistical significance testing: win rate p-values, sample size analysis
- [x] Leakage investigation: root cause was extended-hours bars (49.4% of data)
- [x] Bar timestamp & shift validation scripts
- [x] Feature correlation analysis confirming near-zero IC on tier-1 features

---

## Current Results

### Latest: 1-year out-of-sample (Sep 2025 – Jun 2026)
Settings: `TOP_N=5`, `HOLD_RANK=25`, `KELLY=True`, `REBALANCE=True`, `COST_BPS=2.0`, quintile labels

| Metric | Value |
|--------|-------|
| Net P&L | +$27,611 (+27.6%) |
| Gross P&L | +$41,334 |
| Total Costs | $13,723 (33% of gross) |
| Win Rate | 50.6% |
| Avg positions/bar | 6.6 (hold band active) |
| Turnover rate | 61.3% of bars are new entries |
| Top sector | XLK +$18k (65% of total profit) |

### Historical baseline (2021–2026, pre-rebalancing, binary label, equal-weight)

| Metric | Tier-1 only | + Tier-2 features |
|--------|------------|-------------------|
| Win Rate | 50.04% | 50.46% |
| Sharpe | −0.44 | −0.06 |
| Total P&L | −$30,433 | −$9,079 |
| Gross P&L (pre-cost) | +$54k | +$88.5k |
| Total Costs | $84k | $97.6k |

**Key finding**: Binary label was the primary model flaw — the model couldn't distinguish
+4% outperformers from +0.02% median crossers. Quintile regression + rebalancing cost model
eliminated the cost drag and dramatically improved signal quality.

---

## Next Up

### Priority 1 — Model improvements

1. **LambdaRank objective** (`rank_xendcg`) — the regression model predicts quintile scores
   independently per stock. LambdaRank treats each timestamp as a query group and directly
   optimizes the cross-sectional ranking. Requires passing `group` array to LightGBM.
   Likely the single highest-impact remaining change.

2. **Walk-forward embargo** — 5-bar gap at each fold boundary prevents rolling-feature leakage
   across the train/test split. Diagnostic: if IC drops significantly, prior folds had leakage.

3. **Sector concentration risk** — XLK drove 65% of 1yr profits. The model needs either
   sector-neutral position sizing (cap each sector's weight) or a sector exposure monitor.

4. **Full 5-year benchmark** — run with current settings (`LABEL="5yr"`, `START_DATE=None`)
   to get a statistically meaningful performance record vs the historical baseline.

### Priority 2 — Signal quality

5. **SHAP feature importance** — identify which of 55 features are actually driving the
   quintile predictions and prune the bottom quartile per fold.

6. **Daily feature IC analysis** — run feature correlation report on the 17 daily features
   specifically. Earnings proximity and VIX term structure are the most likely to have signal.

7. **Ridge regression baseline** — if Ridge matches LightGBM IC, tree complexity is wasted
   on noise and the signal is linear. 2-line change to validate.

8. **Time-decay sample weights** — down-weight 2021–2022 observations. Market regime
   in 2025–2026 likely differs significantly from COVID-era data.

### Priority 3 — Robustness

9. **Hyperparameter search** with Optuna (learning rate, num_leaves, min_child_samples)
10. **Survivorship bias mitigation** — point-in-time S&P 500 constituent data
11. **Sector-neutral position sizing** — cap each GICS sector to ≤40% of portfolio weight

### Priority 4 — Execution layer

12. Alpaca order placement (paper trading live)
13. Hourly scheduler: run pipeline on market open, rebalance each hour
14. Risk management: max drawdown circuit breaker, position size limits
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

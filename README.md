# Algo Trading — Cross-Sectional Intraday LightGBM

1-hour intraday cross-sectional momentum/factor model on the S&P 500.
Trained walk-forward on 5 years of hourly OHLCV data. Paper trading via Alpaca.

---

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Universe | ~500 S&P 500 constituents |
| Bar frequency | 1-hour |
| Trading hours | 10am–2pm ET (5 valid entry bars/day) |
| Holding period | 1 hour |
| Positions | Top-N stocks by model probability |
| Signal | LightGBM cross-sectional rank of P(outperform median) |
| Cost model | 2 bps per position (roundtrip) |
| Target | 1h forward return vs cross-sectional median |

**Current gross alpha**: ~+$88.5k over 5 years on $100k capital before costs.
Transaction costs remain the binding constraint at 2 bps/position.

---

## Architecture

```
data/
  fetcher.py          — Alpaca API raw OHLCV download
  store.py            — Parquet cache, incremental refresh, bulk operations
  universe.py         — S&P 500 ticker list + GICS sector map
  macro_fetcher.py    — Daily macro: yield curve, VIX term, calendar flags, earnings

features/
  engineer.py         — build_features(): 55-feature panel (38 hourly + 17 daily)

models/
  predictor.py        — WalkForwardModel: expanding-window LightGBM, 4 folds

strategy/
  signals.py          — generate_signals(): rank stocks by proba, emit top/bottom N

backtest/
  backtester.py       — Dollar-based P&L: equity curve, Sharpe, drawdown, trade log

scripts/
  fetch_sp500.py               — Download/refresh full universe
  backfill_history.py          — Prepend older history to existing cache
  feature_correlation_report.py — IC/ICIR analysis PDF report
  feature_analysis.py          — Feature importance + distribution charts (all 55 features)
  trade_report.py              — Standalone trade report PDF generator
  plot_tree.py                 — Visualize a single LightGBM decision tree
  sanity_check.py              — Data quality checks
  run_pipeline.py              — End-to-end pipeline runner
  analysis/                    — One-off research scripts (leakage diagnostics, shift validation, etc.)

docs/
  ROADMAP.md          — Project roadmap and technical debt log
  FEATURES.md         — Full feature documentation (55 features)

backtest.ipynb        — Parameterized backtest notebook (main entry point)
```

---

## Feature Set (55 features)

### Hourly Features (38)

**Tier 1 — Per-ticker time-series (18)**
Momentum returns (1h, 4h, 1d, 5d), RSI, MACD histogram, SMA cross, Bollinger %B and width,
realized volatility, VWAP deviation, volume ratio, hour-of-day, day-of-week,
intraday range position, volatility regime, RSI acceleration, overnight gap.

**Tier 1 — Cross-sectional (5)**
Market breadth (% above SMA20), percentile ranks of RSI, 5d return, realized vol, volume ratio.

**Tier 1 — Cross-asset (5)**
SPY-relative 1h and 4h returns, sector ETF-relative 1h and 4h returns, VIX proxy.

**Tier 2 — Microstructure (10)**
Hour-normalized volume, return from open, candlestick body/upper wick/lower wick ratios,
signed volume direction, VWAP drift from open, intraday range rank,
cross-sectional return dispersion, sector-relative return from open.

### Daily Features (17)

**Macro regime (4)**: 10y yield, yield curve (10y−short rate), HY spread proxy, short rate.

**VIX term structure (4)**: VIX spot level, 3-month VIX, term ratio (VIX3M/VIX), 1-day change.

**Economic calendar (4)**: FOMC day flag, NFP day flag, CPI day flag, any macro event flag.

**Earnings proximity (5)**: Days to next earnings, days from last earnings,
pre-earnings 5-day window flag, post-earnings 2-day flag, earnings week flag.

Full documentation: [docs/FEATURES.md](docs/FEATURES.md)

---

## Setup

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate algotrading

# 2. Add API key
echo "ALPACA_API_KEY=your_key" > .env
echo "ALPACA_SECRET_KEY=your_secret" >> .env

# 3. Download data (first run — takes ~20 min)
python scripts/fetch_sp500.py

# 4. Run backtest
jupyter notebook backtest.ipynb
```

---

## Data Management

```bash
# Refresh all hourly data to latest
python -c "from data.store import refresh_all; refresh_all()"

# Refresh daily macro/earnings data (auto-refreshes when stale)
python -c "
from data.macro_fetcher import load_all_daily
daily = load_all_daily(tickers, '2021-01-01', '2026-06-13')
"

# Sanity check
python scripts/sanity_check.py

# Feature correlation report
python scripts/feature_correlation_report.py
# → outputs/feature_analysis/feature_correlation_report.pdf
```

---

## Key Design Decisions

**Market hours filter**: Extended-hours bars (4am–10am, 3pm–8pm ET) are dropped entirely.
49.4% of raw Alpaca data was outside market hours. These bars produced
untradeable signals and inflated backtest returns.

**Time-aware forward return**: `close(t+1h)` is looked up by timestamp, not positional shift.
Positional `shift(-1)` crosses overnight gaps; time-aware lookup returns NaN
for bars targeting after-hours, which are silently dropped by `dropna`.

**Cross-sectional label**: `y = 1` if the stock's 1h return beats the cross-sectional median
at that timestamp. This frames the task as relative stock-picking (always ~50% base rate),
not market-timing.

**Dollar-based backtester**: Tracks actual portfolio value from starting capital.
Per-position cost deducted from each stock's return. Dollar P&L reported per trade.

---

## Current Performance (2021–2026)

```
Capital         : $100,000
Final Value     : $90,921
Total P&L       : -$9,079  (-9.1%)
Gross P&L       : +$88,537  (before costs)
Total Costs     : $97,617
Sharpe          : -0.055
Max Drawdown    : -$24,828  (-22.8%)
Win Rate        : 50.46%
```

The model has genuine gross alpha. Transaction costs at 2 bps/position are the
primary drag. See [docs/ROADMAP.md](docs/ROADMAP.md) for the improvement plan.

---

## Methodology Notes

The current model is a binary classifier (beats median → 1, doesn't → 0).
The methodology review identified three priority improvements:

1. **Quintile labels** (0–4) instead of binary — 4× more signal resolution
2. **Walk-forward embargo** — 5-bar gap at fold boundaries to remove rolling-feature leakage
3. **LambdaRank objective** — directly optimizes cross-sectional stock ranking

See [docs/ROADMAP.md](docs/ROADMAP.md) for full priority queue.

# algo_trading

An intraday cross-sectional equity trading system for the S&P 500. Fetches hourly bars from
Alpaca, filters to regular market hours, engineers a 28-feature panel, trains a walk-forward
LightGBM classifier, generates long/short signals, and evaluates them with a parametric
backtester. All results output to a labelled folder for side-by-side period comparisons.

---

## Strategy overview

- **Universe**: ~501 current S&P 500 constituents
- **Data**: 1-hour OHLCV bars from Alpaca (up to 5 years history)
- **Market hours**: 10:00 am – 3:00 pm ET only (14:00–19:00 UTC). Extended-hours bars
  (4am–4pm pre/post-market) are excluded — they have 10–100× wider bid-ask spreads and
  produce untradeable signals.
- **Target**: 1-hour forward return (time-aware: looks up `close[t+1h]` in the index, not
  a positional shift). Rows where `t+1h` falls outside market hours are dropped, giving
  5 valid entry slots per day (10am–2pm).
- **Model**: Walk-forward LightGBM binary classifier. Label `y=1` if a stock's 1h return
  exceeds the cross-sectional median at that timestamp.
- **Signal**: Long the top 20 stocks by predicted probability each hour.
- **Cost assumption**: 5 bps roundtrip per rebalance (conservative for liquid large-caps).

---

## Setup

**1. Create and activate a conda environment**

```bash
conda create -n algotrading python=3.12
conda activate algotrading
pip install -r requirements.txt
```

**2. Register the Jupyter kernel** (needed to run `backtest.ipynb`)

```bash
python -m ipykernel install --user --name algotrading --display-name "algotrading"
```

**3. Configure API credentials**

Copy `.env.example` to `.env` and fill in your Alpaca paper-trading keys:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true
```

Alpaca paper-trading keys are free and provide access to historical bar data.

---

## Project structure

```
algo_trading/
├── data/
│   ├── fetcher.py          # Alpaca API client, raw bar fetch, technical indicators
│   ├── store.py            # Parquet cache, incremental refresh, bulk download, backfill
│   ├── universe.py         # S&P 500 ticker list + GICS sector map
│   └── cache/              # Per-symbol parquet files — gitignored, regenerate locally
├── features/
│   └── engineer.py         # Market hours filter, 28-feature panel, time-aware labels
├── models/
│   └── predictor.py        # Walk-forward LightGBM classifier
├── strategy/
│   └── signals.py          # Ranks probabilities per timestamp, assigns long/short/neutral
├── backtest/
│   └── backtester.py       # Equity curve, Sharpe, max drawdown, win rate, trade log
├── scripts/
│   ├── fetch_sp500.py      # One-time download of all ~501 S&P 500 symbols (1 year)
│   ├── backfill_history.py # Prepend 1 additional year of history to all cached symbols
│   ├── backfill_3yr.py     # Prepend 3 additional years (total ~5 years available)
│   ├── diagnose_leakage.py # Diagnostic: extended hours %, AUC per fold, year-by-year perf
│   ├── validate_shifts.py  # Validates that shift() time gaps match intended lookbacks
│   └── sanity_check.py     # Validates cache for duplicates, bad values, intraday gaps
├── docs/
│   └── ROADMAP.md
├── backtest.ipynb          # Parameterized backtest notebook — main execution entrypoint
├── .env.example            # API key template (never commit .env)
└── requirements.txt
```

---

## Running a backtest

Open `backtest.ipynb` in Jupyter with the `algotrading` kernel. Edit the parameters cell:

```python
LABEL          = "full_5yr"   # subfolder under outputs/ for all results
START_DATE     = None         # "YYYY-MM-DD" or None for full history
END_DATE       = None         # "YYYY-MM-DD" or None for today
TOP_N          = 20           # stocks to long per period
BOTTOM_N       = 20           # stocks to short per period (not used in backtest P&L)
HOLDING_PERIOD = 1            # hours (1 = 5 slots/day, 4 = 2 slots/day)
COST_BPS       = 5.0          # roundtrip transaction cost
N_SPLITS       = 4            # walk-forward folds
```

Run all cells. Outputs save to `outputs/{LABEL}/`:

```
outputs/full_5yr/
├── trade_report.pdf          # 3-page visual report
├── page1.png / page2.png / page3.png
├── trade_log.csv             # one row per executed long trade
├── summary.txt               # key metrics as plain text
└── backtest_executed.ipynb   # executed notebook snapshot
```

Or run headlessly from the terminal:

```bash
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output outputs/full_5yr/backtest_executed.ipynb \
  backtest.ipynb
```

---

## Data management

**First-time download** (~501 S&P 500 symbols, 1 year of 1h bars):

```bash
python scripts/fetch_sp500.py
```

**Extend history backwards** (adds older data to existing parquet files without re-downloading):

```bash
python scripts/backfill_history.py  # +1 year
python scripts/backfill_3yr.py      # +3 years (total ~5 years if run after above)
```

**Incremental refresh** (append only new bars since last cached timestamp):

```python
from data.store import refresh_all
refresh_all(interval="1h")
```

**Parquet cache location**: `data/cache/<SYMBOL>_1h.parquet`
- Index: `datetime` (DatetimeTZAware, UTC)
- Columns: `open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`
- Cache is gitignored — each developer runs their own download

---

## Architecture

```
Alpaca API
    │
    ▼
data/fetcher.py       fetch_raw(symbol, start, end)  →  raw OHLCV DataFrame
                      add_indicators()               →  + RSI, SMA20/50, MACD, BB, VWAP
    │
    ▼
data/store.py         load() / bulk_load()           →  parquet cache (data/cache/)
                      backfill() / bulk_backfill()   →  prepend older history
    │
    ▼
features/engineer.py
    _filter_market_hours()       keep 14:00–19:00 UTC (10am–3pm ET), drop extended hours
    load_universe()              dict[symbol → DataFrame]
    build_features(data, etfs)   MultiIndex DataFrame [datetime, symbol]
                                 28 features + forward_return (time-aware, 1h) + y
    │
    ▼
models/predictor.py
    WalkForwardModel.fit_predict(features_df)  →  Series[proba]  (out-of-sample only)
    │
    ▼
strategy/signals.py
    generate_signals(proba, features_df, top_n, bottom_n)
                                               →  DataFrame[signal, rank, proba, forward_return]
    │
    ▼
backtest/backtester.py
    run(signals, holding_period, cost_bps)     →  {equity_curve, total_return, sharpe,
                                                   max_drawdown, win_rate, n_periods}
    trade_log(signals, ...)                    →  DataFrame (one row per trade)
```

---

## Feature set (28 features)

| Category | Feature | Description |
|---|---|---|
| Momentum | `ret_1h` | Log return over 1 bar (1h) |
| Momentum | `ret_4h` | Log return over 4 market bars (~overnight lookback from open) |
| Momentum | `ret_1d` | Log return over 7 market bars (~1 trading day) |
| Momentum | `ret_5d` | Log return over 35 market bars (~5 trading days) |
| Trend | `rsi` | 14-period RSI |
| Trend | `sma_cross` | (SMA20 − SMA50) / close |
| Trend | `macd_hist` | MACD histogram |
| Volatility | `bb_pct` | Bollinger Band position: (close − lower) / (upper − lower) |
| Volatility | `bb_width` | Bollinger Band width: (upper − lower) / mid |
| Volatility | `realized_vol` | 20-bar rolling std of `ret_1h` |
| Volume | `vwap_dev` | (close − VWAP) / VWAP |
| Volume | `vol_ratio` | volume / 20-bar rolling mean volume |
| Calendar | `hour` | Hour of day (UTC) |
| Calendar | `dow` | Day of week (0=Monday) |
| Price action | `range_pos` | (close − low) / (high − low) — intraday range position |
| Regime | `vol_regime` | 5-bar vol / 20-bar vol — short vs long-term volatility ratio |
| Momentum accel | `rsi_accel` | RSI − RSI[4 bars ago] |
| Gap | `overnight_gap` | open / prev_close − 1 — institutional order flow proxy |
| Breadth | `market_breadth` | % of universe stocks above their SMA20 at this timestamp |
| Cross-sectional | `rsi_rank` | Percentile rank of RSI across all stocks at this timestamp |
| Cross-sectional | `ret_5d_rank` | Percentile rank of 5-day return |
| Cross-sectional | `realized_vol_rank` | Percentile rank of realized volatility |
| Cross-sectional | `vol_ratio_rank` | Percentile rank of volume ratio |
| vs SPY | `ret_1h_vs_spy` | ret_1h − SPY ret_1h |
| vs SPY | `ret_4h_vs_spy` | ret_4h − SPY ret_4h |
| Market regime | `vix_proxy` | Rolling realized vol of SPY (VIX approximation) |
| vs Sector | `ret_1h_vs_sector` | ret_1h − sector ETF ret_1h |
| vs Sector | `ret_4h_vs_sector` | ret_4h − sector ETF ret_4h |

---

## Known limitations

**Survivorship bias**: Universe is the *current* S&P 500 list. Stocks removed during the
backtest period (bankruptcy, acquisition, demotion) are absent, which inflates performance.
Point-in-time constituent data (e.g. Tiingo) would fix this.

**Transaction cost assumption**: 5 bps roundtrip is conservative for large-caps at top of
book but does not account for market impact at size or slippage on fast-moving names.

**No execution layer**: The system is research-only. Order placement via Alpaca's trading
API is not yet implemented.

**Model simplicity**: Single LightGBM binary classifier. A ranking objective (lambdarank)
or return regression would more directly optimise the portfolio construction objective.

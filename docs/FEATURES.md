# Feature Documentation

**Strategy**: 1-hour intraday cross-sectional LightGBM on S&P 500 constituents  
**Universe**: ~500 stocks, 5 valid trade bars per day (10am–2pm ET)  
**Target**: `forward_return` = 1-hour forward price return (time-aware; NaN when market is closed)  
**Label**: `y` = 1 if `forward_return` > cross-sectional median at that timestamp, else 0  
**Total features**: 38 (28 tier-1 + 10 tier-2)

---

## Tier 1 — Original Feature Set

### Per-Ticker Time-Series

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 1 | `ret_1h` | `log(close / close.shift(1))` | Prior-bar momentum or reversal | Negative IC observed — 1h momentum anti-predicts |
| 2 | `ret_4h` | `log(close / close.shift(4))` | ~Overnight return from prior afternoon | Positional shift; 4 market bars back ≈ yesterday 2pm close |
| 3 | `ret_1d` | `log(close / close.shift(7))` | ~1-day momentum | 7 bars × 1h ≈ 1 trading day |
| 4 | `ret_5d` | `log(close / close.shift(35))` | ~Weekly momentum | 35 bars × 1h ≈ 5 trading days |
| 5 | `rsi` | RSI(14) from `pandas_ta` | Overbought/oversold mean reversion | Range 0–100; added by `add_indicators()` |
| 6 | `sma_cross` | `(sma_20 - sma_50) / close` | Trend direction | Positive = short-term trend up vs medium-term |
| 7 | `macd_hist` | MACD histogram from `pandas_ta` | Momentum acceleration/deceleration | `macd - signal` |
| 8 | `bb_pct` | `(close - bb_lower) / (bb_upper - bb_lower)` | Position within volatility band | 0 = at lower band, 1 = at upper band |
| 9 | `bb_width` | `(bb_upper - bb_lower) / bb_mid` | Volatility expansion/contraction | Higher = wider bands = more volatile regime |
| 10 | `realized_vol` | `ret_1h.rolling(20).std()` | Recent intraday volatility level | Rolling 20-bar (~4 trading days) std |
| 11 | `vwap_dev` | `(close - vwap) / vwap` | Deviation from average transaction price | Negative = trading below avg price (potential support) |
| 12 | `vol_ratio` | `volume / volume.rolling(20).mean()` | Abnormal volume relative to recent avg | > 1 = above-average activity |
| 13 | `hour` | `index.hour` (UTC) | Intraday seasonality | 14–18 UTC = 10am–2pm ET trading window |
| 14 | `dow` | `index.dayofweek` | Day-of-week seasonality | 0=Monday … 4=Friday |
| 15 | `range_pos` | `(close - low) / (high - low)` | Bar close position within intraday range | 0 = closed at low, 1 = closed at high |
| 16 | `vol_regime` | `vol_5bar.std / vol_20bar.std` | Short-term vol vs medium-term vol | > 1 = volatility expanding; < 1 = contracting |
| 17 | `rsi_accel` | `rsi - rsi.shift(4)` | Rate of change of RSI | Momentum of momentum; 4 bars ≈ 4 hours |
| 18 | `overnight_gap` | `open / close.shift(1) - 1` | Institutional order flow at open | Captures pre-market sentiment gap |

### Cross-Sectional (same timestamp, all stocks)

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 19 | `market_breadth` | `mean(above_sma20)` per timestamp | Broad market health | % of universe trading above 20-bar SMA |
| 20 | `rsi_rank` | `rsi.rank(pct=True)` per timestamp | Relative overbought/oversold position | 0=most oversold, 1=most overbought in universe |
| 21 | `ret_5d_rank` | `ret_5d.rank(pct=True)` per timestamp | Relative weekly momentum | |
| 22 | `realized_vol_rank` | `realized_vol.rank(pct=True)` per timestamp | Relative volatility level | |
| 23 | `vol_ratio_rank` | `vol_ratio.rank(pct=True)` per timestamp | Relative volume activity | |

### Cross-Asset (ETF benchmarks)

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 24 | `ret_1h_vs_spy` | `ret_1h - spy_ret_1h` | Stock-specific 1h move vs market | Idiosyncratic momentum |
| 25 | `ret_4h_vs_spy` | `ret_4h - spy_ret_4h` | Stock-specific ~overnight move vs market | |
| 26 | `vix_proxy` | `spy_ret_1h.rolling(20).std()` | Market volatility regime | Same value for all stocks at each timestamp |
| 27 | `ret_1h_vs_sector` | `ret_1h - sector_etf_ret_1h` | Stock-specific 1h move vs sector | Sector ETF mapped via `sector_map` |
| 28 | `ret_4h_vs_sector` | `ret_4h - sector_etf_ret_4h` | Stock-specific ~overnight move vs sector | |

---

## Tier 2 — Extended Feature Set

### Per-Ticker Time-Series

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 29 | `vol_vs_hour_avg` | `volume / mean(volume at this UTC hour)` | Abnormal activity controlling for intraday seasonality | Addresses U-shaped volume curve (busy at open/close) |
| 30 | `ret_from_open` | `log(close / day_open)` | Cumulative intraday momentum/reversal from open | Anchored to first bar's open price each day |
| 31 | `body_ratio` | `abs(close - open) / (high - low + 1e-8)` | Bar conviction — how directional was this bar | 1 = full-body candle (strong move), 0 = doji |
| 32 | `upper_wick` | `(high - max(open, close)) / (high - low + 1e-8)` | Selling pressure at highs — price rejected | High upper wick = bears pushed price back down |
| 33 | `lower_wick` | `(min(open, close) - low) / (high - low + 1e-8)` | Buying pressure at lows — price supported | High lower wick = bulls defended the low |
| 34 | `vol_direction` | `sign(close - open) × volume / rolling_20_std` | Net buying vs selling pressure, normalized | Positive = net buying volume, negative = selling |
| 35 | `vwap_dev_from_open` | `(vwap - open) / open` | Net transaction pressure since market open | Positive = avg trade above open (bullish tape) |

> **Candlestick invariant**: `body_ratio + upper_wick + lower_wick = 1.0` by construction.

### Cross-Sectional (same timestamp, all stocks)

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 36 | `intraday_range_rank` | `rank((high - low) / close, pct=True)` per timestamp | Which stocks are experiencing widest swings now | 0 = tightest range, 1 = widest range in universe |
| 37 | `cs_return_dispersion` | `std(ret_1h)` per timestamp | Market-wide stock-picking opportunity | High = divergent returns (alpha environment); low = correlated tape |

### Cross-Asset

| # | Feature | Formula | Signal Hypothesis | Notes |
|---|---------|---------|-------------------|-------|
| 38 | `ret_from_open_vs_sector` | `ret_from_open - sector_etf_ret_from_open` | Idiosyncratic intraday move stripping sector drift | Removes broad sector beta from intraday return |

---

## Target Variables (not features — excluded from model inputs)

| Variable | Definition | Notes |
|----------|-----------|-------|
| `forward_return` | `close(t+1h) / close(t) - 1` | Time-aware: uses `reindex(index + 1h)`, not positional shift. Bars targeting after-market get NaN and are dropped. |
| `y` | `1 if forward_return > median(forward_return at t), else 0` | Cross-sectional binary label — beats the median, not just positive return |

---

## Known Signal Diagnostics (from correlation report, full 5-year history)

| Finding | Detail |
|---------|--------|
| All return features have negative IC | `ret_1h`, `ret_4h`, `ret_1d`, `ret_5d` all anti-predict at 1h horizon — momentum reverses |
| Highest |IC mean| among tier-1 | `vwap_dev`: 0.017, ICIR: −0.12 |
| No feature exceeds ICIR ±0.5 | Minimum threshold for a tradeable signal; none of tier-1 clears this |
| `market_breadth` / `vix_proxy` IC ≈ 0 | Cross-sectional constants per timestamp — after ranking, all stocks identical |
| Costs > gross returns | At 2 bps/position and 5 positions, $84k costs on $100k over 5 years |

---

## Feature Engineering Notes

**Market hours filter**: Only 14:00–19:30 UTC bars (10am–3:30pm ET) are kept. Extended-hours bars (pre-market, after-hours) are excluded — they have thin liquidity and produce untradeable signals.

**Positional vs time-aware shifts**: All lookback features (`ret_1h`, `ret_4h`, etc.) use positional `shift(n)` which is correct after the market-hours filter (no overnight gaps within the filtered data). Only `forward_return` uses time-aware `reindex` to prevent leaking across the 3pm–10am gap.

**Cross-sectional label**: `y` is defined relative to the cross-sectional median at each timestamp, not absolute zero. This makes the classification task symmetric (always ~50% positive) and focuses the model on relative stock-picking rather than market-timing.

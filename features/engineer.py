import glob
import os

import numpy as np
import pandas as pd

from data.fetcher import add_indicators
from data.store import CACHE_DIR
from data.universe import get_etf_list


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular US market hours bars (9:30am–4:00pm ET = 13:30–20:00 UTC).

    Alpaca returns extended-hours bars (4am–8pm ET) by default. These have
    wide bid-ask spreads, thin liquidity, and produce forward_returns that
    span illiquid overnight periods — massively inflating backtest returns.
    """
    idx = pd.to_datetime(df.index, utc=True)
    # Keep bars whose hour is 13 (>=:30) through 19 UTC.
    # 1h bars timestamped at :30 cover 9:30-10:30am ET (13:30-14:30 UTC).
    # Bars at 14:00-19:00 UTC cover 10:00am-3:00pm ET.
    # The 19:30 UTC bar (3:30pm ET open, 4:00pm ET close) is the last full bar.
    hour = idx.hour
    minute = idx.minute
    in_market = ((hour == 13) & (minute >= 30)) | ((hour >= 14) & (hour <= 19))
    return df.loc[in_market]


def load_universe(interval="1h") -> dict[str, pd.DataFrame]:
    etf_set = set(get_etf_list())
    pattern = os.path.join(CACHE_DIR, f"*_{interval}.parquet")
    files = glob.glob(pattern)
    result = {}
    for f in files:
        symbol = os.path.basename(f).replace(f"_{interval}.parquet", "")
        if symbol in etf_set:
            continue  # ETFs are loaded separately via load_etf_universe()
        df = pd.read_parquet(f)
        df = _filter_market_hours(df)
        df = add_indicators(df)
        result[symbol] = df
    return result


def build_features(data_dict: dict, etf_dict: dict = None, sector_map: dict = None, daily_features: pd.DataFrame = None) -> pd.DataFrame:
    frames = []
    for symbol, df in data_dict.items():
        f = pd.DataFrame(index=df.index)
        f["ret_1h"] = np.log(df["close"] / df["close"].shift(1))
        f["ret_4h"] = np.log(df["close"] / df["close"].shift(4))  # 4 market bars back (~overnight from open)
        f["ret_1d"] = np.log(df["close"] / df["close"].shift(7))
        f["ret_5d"] = np.log(df["close"] / df["close"].shift(35))
        f["rsi"] = df["rsi"]
        f["sma_cross"] = (df["sma_20"] - df["sma_50"]) / df["close"]
        f["macd_hist"] = df["macd_hist"]
        f["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-8)
        f["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        f["realized_vol"] = f["ret_1h"].rolling(20).std()
        f["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]
        f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        f["hour"] = df.index.hour
        f["dow"] = df.index.dayofweek

        # Category 1: Within-stock features
        # Intraday range position: where did close land in today's high-low range
        f["range_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(lower=1e-8)

        # Volatility regime: short-term vol vs longer-term vol
        vol_5 = f["ret_1h"].rolling(5).std()
        vol_20 = f["ret_1h"].rolling(20).std()
        f["vol_regime"] = vol_5 / vol_20.replace(0, np.nan)

        # RSI acceleration: rate of change of RSI
        f["rsi_accel"] = df["rsi"] - df["rsi"].shift(4)

        # Overnight gap: open vs prior close (institutional order flow signal)
        f["overnight_gap"] = df["open"] / df["close"].shift(1) - 1

        # Above SMA20 flag (used to compute market breadth after concat)
        f["above_sma20"] = (df["close"] > df["sma_20"]).astype(float)

        # ── Tier-2 features (per-ticker) ──

        # 1. vol_vs_hour_avg: volume / stock's mean volume at this specific hour-of-day.
        #    Controls for intraday U-shaped volume seasonality (open and close are busier).
        #    Values > 1 = above-average activity for this time slot.
        hour_of_day = df.index.hour
        vol_hour_avg = df.groupby(hour_of_day)["volume"].transform("mean")
        f["vol_vs_hour_avg"] = df["volume"] / vol_hour_avg.replace(0, np.nan)

        # 2. ret_from_open: log return from first bar of the day to current bar.
        #    First bar open is used as the intraday anchor (10am ET = 14:00 UTC).
        #    Captures intraday momentum or mean-reversion vs. the opening price.
        date_key = df.index.normalize()
        day_open = df.groupby(date_key)["open"].transform("first")
        f["ret_from_open"] = np.log(df["close"] / day_open.replace(0, np.nan))

        # 3. body_ratio: |close - open| / (high - low). Near 1 = strong directional
        #    bar; near 0 = indecisive doji-like bar.
        hl_range = df["high"] - df["low"] + 1e-8
        f["body_ratio"] = np.abs(df["close"] - df["open"]) / hl_range

        # 4. upper_wick: (high - max(open,close)) / hl_range. Selling pressure at highs.
        f["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / hl_range

        # 5. lower_wick: (min(open,close) - low) / hl_range. Buying pressure at lows.
        f["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / hl_range

        # 6. vol_direction: volume × sign(close - open), normalized by its own rolling
        #    20-bar std so the series is scaled comparably across different price levels.
        signed_vol = df["volume"] * np.sign(df["close"] - df["open"])
        signed_vol_std = signed_vol.rolling(20).std().replace(0, np.nan)
        f["vol_direction"] = signed_vol / signed_vol_std

        # 10. vwap_dev_from_open: (vwap - open) / open.
        #     How far has the average transaction price drifted from the day's open?
        #     Positive = net buying pressure since open; negative = net selling.
        f["vwap_dev_from_open"] = (df["vwap"] - df["open"]) / df["open"].replace(0, np.nan)

        # Helper for cross-sectional intraday_range_rank (feature 8).
        # Store (high - low) / close so it's available after concat.
        f["_hl_frac"] = hl_range / df["close"].replace(0, np.nan)

        target_times = df.index + pd.Timedelta(hours=1)
        future_close = df["close"].reindex(target_times).values
        f["forward_return"] = future_close / df["close"].values - 1
        f["symbol"] = symbol
        frames.append(f)

    combined = pd.concat(frames)
    combined = combined.dropna(subset=["forward_return"])
    combined = combined.set_index("symbol", append=True)
    combined.index.names = ["datetime", "symbol"]

    # Category 2: Cross-sectional features
    # Market breadth: % of stocks above their SMA20 at each timestamp
    combined["market_breadth"] = combined.groupby(level="datetime")["above_sma20"].transform("mean")
    combined.drop(columns=["above_sma20"], inplace=True)

    # Cross-sectional percentile ranks (0=worst, 1=best among all stocks at that timestamp)
    for col in ["rsi", "ret_5d", "realized_vol", "vol_ratio"]:
        combined[f"{col}_rank"] = combined.groupby(level="datetime")[col].rank(pct=True)

    # ── Tier-2 cross-sectional features ──

    # 8. intraday_range_rank: cross-sectional percentile rank of (high-low)/close.
    #    At each timestamp, which stocks are experiencing the widest price swings?
    #    0 = narrowest range, 1 = widest range. Uses same pattern as existing _rank cols.
    combined["intraday_range_rank"] = combined.groupby(level="datetime")["_hl_frac"].rank(pct=True)
    combined.drop(columns=["_hl_frac"], inplace=True)

    # 9. cs_return_dispersion: cross-sectional std of ret_1h at each timestamp.
    #    Market-wide feature — same value for every stock at a given timestamp.
    #    High dispersion = rich stock-picking environment; low = correlated tape.
    combined["cs_return_dispersion"] = combined.groupby(level="datetime")["ret_1h"].transform("std")

    # Category 3: Cross-asset features (only if etf_dict and sector_map are provided)
    if etf_dict is not None and sector_map is not None:
        # Compute 1h and 4h log returns for all ETFs
        etf_ret_1h = {}
        etf_ret_4h = {}
        for etf, edf in etf_dict.items():
            etf_ret_1h[etf] = np.log(edf["close"] / edf["close"].shift(1))
            etf_ret_4h[etf] = np.log(edf["close"] / edf["close"].shift(4))
        etf_ret_1h_df = pd.DataFrame(etf_ret_1h)   # index=datetime, columns=ETF tickers
        etf_ret_4h_df = pd.DataFrame(etf_ret_4h)

        # Compute ret_from_open for each sector ETF (for feature 7)
        etf_ret_from_open = {}
        for etf, edf in etf_dict.items():
            edf_filtered = _filter_market_hours(edf)
            etf_date_key = edf_filtered.index.normalize()
            etf_day_open = edf_filtered.groupby(etf_date_key)["open"].transform("first")
            etf_ret_from_open[etf] = np.log(edf_filtered["close"] / etf_day_open.replace(0, np.nan))
        etf_ret_from_open_df = pd.DataFrame(etf_ret_from_open)  # index=datetime, columns=ETF tickers

        # Reset index for merge operations
        combined_reset = combined.reset_index()
        combined_reset["sector_etf"] = combined_reset["symbol"].map(sector_map)

        # SPY-relative return
        if "SPY" in etf_ret_1h_df.columns:
            spy_1h = etf_ret_1h_df[["SPY"]].rename(columns={"SPY": "spy_ret_1h"})
            spy_4h = etf_ret_4h_df[["SPY"]].rename(columns={"SPY": "spy_ret_4h"})
            combined_reset = combined_reset.merge(spy_1h, left_on="datetime", right_index=True, how="left")
            combined_reset = combined_reset.merge(spy_4h, left_on="datetime", right_index=True, how="left")
            combined_reset["ret_1h_vs_spy"] = combined_reset["ret_1h"] - combined_reset["spy_ret_1h"]
            combined_reset["ret_4h_vs_spy"] = combined_reset["ret_4h"] - combined_reset["spy_ret_4h"]
            # VIX proxy: rolling realized vol of SPY (regime signal)
            spy_close = etf_dict["SPY"]["close"]
            spy_rvol = np.log(spy_close / spy_close.shift(1)).rolling(20).std()
            spy_rvol = spy_rvol.rename("vix_proxy")
            combined_reset = combined_reset.merge(spy_rvol, left_on="datetime", right_index=True, how="left")

        # Sector-relative return: melt ETF returns to long format, merge on (datetime, sector_etf)
        etf_long_1h = etf_ret_1h_df.stack().reset_index()
        etf_long_1h.columns = ["datetime", "sector_etf", "sector_ret_1h"]
        etf_long_4h = etf_ret_4h_df.stack().reset_index()
        etf_long_4h.columns = ["datetime", "sector_etf", "sector_ret_4h"]
        combined_reset = combined_reset.merge(etf_long_1h, on=["datetime", "sector_etf"], how="left")
        combined_reset = combined_reset.merge(etf_long_4h, on=["datetime", "sector_etf"], how="left")
        combined_reset["ret_1h_vs_sector"] = combined_reset["ret_1h"] - combined_reset["sector_ret_1h"]
        combined_reset["ret_4h_vs_sector"] = combined_reset["ret_4h"] - combined_reset["sector_ret_4h"]

        # 7. ret_from_open_vs_sector: stock's intraday return from open minus its sector
        #    ETF's intraday return from open. Isolates the idiosyncratic component of the
        #    stock's intraday move, stripping out broad sector drift.
        etf_long_rfo = etf_ret_from_open_df.stack().reset_index()
        etf_long_rfo.columns = ["datetime", "sector_etf", "sector_ret_from_open"]
        combined_reset = combined_reset.merge(etf_long_rfo, on=["datetime", "sector_etf"], how="left")
        combined_reset["ret_from_open_vs_sector"] = (
            combined_reset["ret_from_open"] - combined_reset["sector_ret_from_open"]
        )

        # Drop helper columns, restore MultiIndex
        drop_cols = [
            "sector_etf", "spy_ret_1h", "spy_ret_4h",
            "sector_ret_1h", "sector_ret_4h", "sector_ret_from_open",
        ]
        combined_reset.drop(columns=[c for c in drop_cols if c in combined_reset.columns], inplace=True)
        combined = combined_reset.set_index(["datetime", "symbol"])

    medians = combined.groupby(level="datetime")["forward_return"].transform("median")
    combined["y"] = (combined["forward_return"] > medians).astype(int)

    # ── Merge daily features (if provided) ────────────────────────────────────
    if daily_features is not None:
        # combined has MultiIndex (datetime[UTC-tz-aware], symbol).
        # Normalise datetime to date, strip tz so it matches the daily index.
        dt_level = combined.index.get_level_values("datetime")
        dates_naive = dt_level.normalize().tz_localize(None)
        sym_level = combined.index.get_level_values("symbol")

        # Build a lookup key Series aligned to combined's integer positions.
        lookup_keys = list(zip(dates_naive, sym_level))
        lookup_idx = pd.MultiIndex.from_tuples(lookup_keys, names=["date", "symbol"])

        # Ensure daily_features index is (date, symbol) with tz-naive dates.
        df_daily = daily_features.copy()
        df_daily.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(df_daily.index.get_level_values(0)).tz_localize(None)
             if df_daily.index.get_level_values(0).tz is not None
             else pd.to_datetime(df_daily.index.get_level_values(0)),
             df_daily.index.get_level_values(1)],
            names=["date", "symbol"],
        )

        # Reindex daily_features to match each hourly row's (date, symbol) key.
        daily_aligned = df_daily.reindex(lookup_idx)
        daily_aligned.index = combined.index  # restore original MultiIndex

        # Left-join: only add columns that don't already exist.
        new_cols = [c for c in daily_aligned.columns if c not in combined.columns]
        if new_cols:
            combined = combined.join(daily_aligned[new_cols], how="left")

    return combined


if __name__ == "__main__":
    data = load_universe("1h")
    data = dict(list(data.items())[:10])
    features = build_features(data)
    print("Shape:", features.shape)
    print("\nColumns:", features.columns.tolist())
    print("\nNaN counts:\n", features.isna().sum())
    print("\ny value_counts:\n", features["y"].value_counts())

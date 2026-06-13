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


def build_features(data_dict: dict, etf_dict: dict = None, sector_map: dict = None) -> pd.DataFrame:
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

        # Drop helper columns, restore MultiIndex
        drop_cols = ["sector_etf", "spy_ret_1h", "spy_ret_4h", "sector_ret_1h", "sector_ret_4h"]
        combined_reset.drop(columns=[c for c in drop_cols if c in combined_reset.columns], inplace=True)
        combined = combined_reset.set_index(["datetime", "symbol"])

    medians = combined.groupby(level="datetime")["forward_return"].transform("median")
    combined["y"] = (combined["forward_return"] > medians).astype(int)

    return combined


if __name__ == "__main__":
    data = load_universe("1h")
    data = dict(list(data.items())[:10])
    features = build_features(data)
    print("Shape:", features.shape)
    print("\nColumns:", features.columns.tolist())
    print("\nNaN counts:\n", features.isna().sum())
    print("\ny value_counts:\n", features["y"].value_counts())

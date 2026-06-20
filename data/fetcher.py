import os
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
        )
    return _client

def _parse_period(period: str) -> datetime:
    units = {"d": 1, "mo": 30, "y": 365}
    for suffix, days_per_unit in units.items():
        if period.endswith(suffix):
            n = int(period[: -len(suffix)])
            return datetime.now() - timedelta(days=n * days_per_unit)
    raise ValueError(f"Unrecognised period: {period}. Use e.g. '6mo', '1y', '90d'.")

def _parse_interval(interval: str) -> TimeFrame:
    mapping = {
        "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
        "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
        "1d":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    if interval not in mapping:
        raise ValueError(f"Unrecognised interval: {interval}. Choose from {list(mapping)}")
    return mapping[interval]


RAW_COLS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

# Alpaca uses "/" for share classes (BRK/B); we store and expose them with "-" (BRK-B).
def _to_alpaca(symbol: str) -> str:
    return symbol.replace("-", "/")

def _from_alpaca(symbol: str) -> str:
    return symbol.replace("/", "-")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]        = ta.rsi(df["close"], length=14)
    df["sma_20"]     = ta.sma(df["close"], length=20)
    df["sma_50"]     = ta.sma(df["close"], length=50)
    df["ema_12"]     = ta.ema(df["close"], length=12)
    df["ema_26"]     = ta.ema(df["close"], length=26)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]

    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bb.iloc[:, 0]
    df["bb_mid"]   = bb.iloc[:, 1]
    df["bb_upper"] = bb.iloc[:, 2]
    return df


def fetch_raw(symbol: str, start: datetime, interval: str = "1h", end: datetime = None) -> pd.DataFrame:
    alpaca_sym = _to_alpaca(symbol)
    request = StockBarsRequest(
        symbol_or_symbols=alpaca_sym,
        timeframe=_parse_interval(interval),
        start=start,
        end=end or datetime.now(),
    )
    bars = _get_client().get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[alpaca_sym]
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df.columns = [c.lower() for c in df.columns]
    return df[[c for c in RAW_COLS if c in df.columns]]


def fetch(symbol: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    return add_indicators(fetch_raw(symbol, start=_parse_period(period), interval=interval))


if __name__ == "__main__":
    df = fetch("AAPL", period="6mo", interval="1h")
    print(f"Rows: {len(df)}")
    print(df.tail(5).to_string())

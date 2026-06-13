import io
import json
import os

import pandas as pd
import requests

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (algo-trading-research/1.0)"}
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "sector_map_cache.json")

SECTOR_ETF_MAP = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


def _fetch_sp500_table() -> pd.DataFrame:
    html = requests.get(_SP500_URL, headers=_HEADERS, timeout=15).text
    return pd.read_html(io.StringIO(html))[0]


def get_sp500() -> list[str]:
    table = _fetch_sp500_table()
    tickers = table["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]


def get_sector_map(refresh: bool = False) -> dict[str, str]:
    """Return a mapping of S&P 500 symbol -> GICS sector ETF ticker.

    Cached to data/sector_map_cache.json after the first fetch.
    Pass refresh=True to force a new Wikipedia request (e.g. after index rebalance).
    """
    if not refresh and os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)

    table = _fetch_sp500_table()
    result: dict[str, str] = {}
    for _, row in table.iterrows():
        symbol = str(row["Symbol"]).replace(".", "-")
        sector = str(row["GICS Sector"])
        etf = SECTOR_ETF_MAP.get(sector)
        if etf is not None:
            result[symbol] = etf

    with open(_CACHE_FILE, "w") as f:
        json.dump(result, f)

    return result


def get_etf_list() -> list[str]:
    """Return the full list of ETF tickers used by the data pipeline (SPY + 11 sector ETFs)."""
    return ["SPY", "XLK", "XLV", "XLF", "XLY", "XLC", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB"]

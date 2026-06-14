"""
data/macro_fetcher.py
─────────────────────
Daily-frequency macro, VIX term-structure, economic calendar, and earnings
proximity features.  All daily data is timezone-naive (date-only index).

Usage:
    from data.macro_fetcher import load_all_daily
    daily = load_all_daily(tickers, "2021-01-01", "2026-06-13")
    # daily is a MultiIndex (date, symbol) DataFrame with all daily columns.

Design rules
    - Never fetches live data inside build_features(); pre-fetch and pass in.
    - pandas_datareader / fredapi are not required; falls back to yfinance.
    - All cache paths live under CACHE_DIR/daily/.
    - Datetime indices stripped to date-only (tz-naive) before caching.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from data.store import CACHE_DIR

# ── Cache directory ────────────────────────────────────────────────────────────
DAILY_CACHE = os.path.join(CACHE_DIR, "daily")
os.makedirs(DAILY_CACHE, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(DAILY_CACHE, f"{name}.parquet")


def _is_stale(path: str, max_age_days: float = 1.0) -> bool:
    """Return True if file doesn't exist or is older than max_age_days."""
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age.total_seconds() > max_age_days * 86_400


# ── 1. Macro daily (yield curve + HY spread) ─────────────────────────────────

def fetch_macro_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily macro data: yield curve, HY spread, 10-year Treasury.

    Primary source: yfinance Treasury yield tickers.
        ^TNX  = 10-Year Treasury yield
        ^IRX  = 13-Week T-Bill yield (short-term rate proxy; no 2-yr on Yahoo)
        HYG   = iShares HY Bond ETF yield-spread proxy via price returns

    HY OAS spread proxy: we use the daily log-return of HYG vs LQD to capture
    the HY-IG spread direction.  For a level proxy we use (1 - HYG/LQD price
    ratio normalised to the sample mean) * 100 as a z-scored spread level.
    This avoids needing FRED/fredapi.

    Derived columns returned:
        yield_10y       — 10-yr Treasury yield (%)
        yield_curve     — yield_10y - short_rate (>0 normal, <0 inverted)
        hy_spread       — HY-spread proxy (HYG/LQD log-return rolling 20d std,
                          scaled to roughly match OAS bps / 100)
    """
    cache = _cache_path("macro_daily")
    if not _is_stale(cache):
        df = pd.read_parquet(cache)
        # Trim to requested range
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        return df.loc[mask]

    print("fetch_macro_daily: downloading from yfinance...")
    # Download Treasury yields and HY/IG ETFs
    tickers = ["^TNX", "^IRX", "HYG", "LQD"]
    raw = yf.download(
        tickers,
        start=start_date,
        end=(pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"

    df = pd.DataFrame(index=close.index)

    # 10-year yield (already in %)
    df["yield_10y"] = close["^TNX"]

    # Short-term rate (13-week T-bill, %)
    df["short_rate"] = close["^IRX"]

    # Yield curve = 10y minus short rate
    df["yield_curve"] = df["yield_10y"] - df["short_rate"]

    # HY spread proxy: rolling 20-day realised vol of HYG log returns,
    # normalised so the level is roughly comparable to OAS / 100.
    hyg_ret = np.log(close["HYG"] / close["HYG"].shift(1))
    lqd_ret = np.log(close["LQD"] / close["LQD"].shift(1))
    spread_ret = hyg_ret - lqd_ret  # excess return (negative = spread widening)
    # 20-day rolling vol of spread excess return * 100 → comparable to OAS level
    df["hy_spread"] = spread_ret.rolling(20).std() * 100

    # Forward-fill weekends / holidays (up to 3 days)
    df = df.ffill(limit=3)

    # Cache full history, then return trimmed
    df.to_parquet(cache)
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    return df.loc[mask]


# ── 2. VIX term structure ─────────────────────────────────────────────────────

def fetch_vix_term(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch VIX and VIX3M daily data.

    Columns returned:
        vix_level      — VIX spot (^VIX close)
        vix_3m         — 3-month VIX (^VIX3M close)
        vix_term_ratio — vix_3m / vix_level  (>1 = contango/normal, <1 = stress)
        vix_1d_chg     — vix_level pct change day-over-day
    """
    cache = _cache_path("vix_term")
    if not _is_stale(cache):
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        return df.loc[mask]

    print("fetch_vix_term: downloading from yfinance...")
    raw = yf.download(
        ["^VIX", "^VIX3M"],
        start=start_date,
        end=(pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"

    df = pd.DataFrame(index=close.index)
    df["vix_level"] = close["^VIX"]
    df["vix_3m"] = close["^VIX3M"]
    df["vix_term_ratio"] = df["vix_3m"] / df["vix_level"].replace(0, np.nan)
    df["vix_1d_chg"] = df["vix_level"].pct_change()

    # Forward-fill holidays
    df = df.ffill(limit=3)

    df.to_parquet(cache)
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    return df.loc[mask]


# ── 3. Economic calendar ──────────────────────────────────────────────────────

# Known FOMC announcement dates (source: federalreserve.gov press releases)
_FOMC_DATES: list[str] = [
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
]


def _first_friday(year: int, month: int) -> date:
    """Return the date of the first Friday in the given month."""
    d = date(year, month, 1)
    # weekday(): Monday=0, Friday=4
    days_until_friday = (4 - d.weekday()) % 7
    return d + timedelta(days=days_until_friday)


def _second_wednesday(year: int, month: int) -> date:
    """Return the Wednesday of the 2nd week of the given month.

    'Second week' = days 8–14; first Wednesday in that window.
    """
    d = date(year, month, 8)  # start of 2nd week
    days_until_wed = (2 - d.weekday()) % 7
    return d + timedelta(days=days_until_wed)


def build_economic_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Build daily binary economic event flags.

    Columns:
        fomc_day        — 1 on known FOMC announcement dates
        nfp_day         — 1 on the first Friday of each month (NFP release)
        cpi_day         — 1 on the Wednesday of the 2nd week of each month
        macro_event_day — 1 if any of the above is 1
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    idx = pd.date_range(start, end, freq="D")
    idx.name = "date"

    df = pd.DataFrame(0, index=idx,
                      columns=["fomc_day", "nfp_day", "cpi_day", "macro_event_day"])

    # FOMC days
    fomc_set = {pd.Timestamp(d) for d in _FOMC_DATES}
    df.loc[df.index.isin(fomc_set), "fomc_day"] = 1

    # NFP days — first Friday of each month
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            nfp = pd.Timestamp(_first_friday(year, month))
            if start <= nfp <= end:
                df.loc[nfp, "nfp_day"] = 1

    # CPI days — Wednesday of 2nd week of each month
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            cpi = pd.Timestamp(_second_wednesday(year, month))
            if start <= cpi <= end:
                df.loc[cpi, "cpi_day"] = 1

    df["macro_event_day"] = (
        (df["fomc_day"] | df["nfp_day"] | df["cpi_day"]).astype(int)
    )

    return df


# ── 4. Earnings dates ─────────────────────────────────────────────────────────

def fetch_earnings_dates(
    tickers: list[str],
    cache_days: int = 7,
) -> dict[str, list[date]]:
    """Fetch upcoming + recent earnings dates for all tickers.

    Returns dict: symbol → sorted list of earnings dates (date objects).
    Cached to data/cache/daily/earnings_dates.parquet for cache_days days.
    """
    cache = _cache_path("earnings_dates")

    if not _is_stale(cache, max_age_days=cache_days):
        long_df = pd.read_parquet(cache)
        result: dict[str, list[date]] = {}
        for sym, grp in long_df.groupby("symbol"):
            dates_list = sorted(pd.to_datetime(grp["earnings_date"]).dt.date.tolist())
            result[sym] = dates_list
        return result

    print(f"fetch_earnings_dates: fetching {len(tickers)} tickers from yfinance...")
    rows = []
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            ed_df = t.get_earnings_dates(limit=20)
            if ed_df is None or ed_df.empty:
                continue
            for ts in ed_df.index:
                # Index is tz-aware; convert to plain date
                try:
                    d = pd.Timestamp(ts).tz_localize(None).normalize().date()
                except Exception:
                    d = pd.Timestamp(ts).tz_convert(None).normalize().date()
                rows.append({"symbol": sym, "earnings_date": d})
        except Exception:
            pass  # some tickers have no earnings data (ETFs, etc.)
        time.sleep(0.1)

    if rows:
        long_df = pd.DataFrame(rows)
        long_df.to_parquet(cache, index=False)
    else:
        long_df = pd.DataFrame(columns=["symbol", "earnings_date"])

    result = {}
    for sym, grp in long_df.groupby("symbol"):
        dates_list = sorted(pd.to_datetime(grp["earnings_date"]).dt.date.tolist())
        result[sym] = dates_list
    return result


# ── 5. Earnings features panel ────────────────────────────────────────────────

def build_earnings_features(
    tickers: list[str],
    data_start: str,
    data_end: str,
) -> pd.DataFrame:
    """Build (date, symbol) earnings proximity features.

    Columns:
        days_to_earnings   — calendar days until next earnings (999 if >60d away)
        days_from_earnings — calendar days since last earnings (999 if >60d ago)
        pre_earnings_5d    — 1 if 1 <= days_to_earnings <= 5
        post_earnings_2d   — 1 if 0 <= days_from_earnings <= 2
        earnings_week      — 1 if pre_earnings_5d OR post_earnings_2d

    Index: MultiIndex (date, symbol), both tz-naive.
    """
    cache = _cache_path("earnings_features")
    if os.path.exists(cache) and not _is_stale(cache, max_age_days=7):
        df = pd.read_parquet(cache)
        df.index = pd.MultiIndex.from_frame(
            df.index.to_frame(index=False).assign(
                date=lambda x: pd.to_datetime(x["date"])
            )
        )
        return df

    print("build_earnings_features: computing earnings proximity features...")
    earnings_dict = fetch_earnings_dates(tickers)

    date_range = pd.date_range(data_start, data_end, freq="D")
    records = []

    for sym in tickers:
        earn_dates = earnings_dict.get(sym, [])
        earn_dates_sorted = sorted(earn_dates)

        for d in date_range:
            cal_date = d.date()

            # days_to_earnings: days until next earnings
            future = [e for e in earn_dates_sorted if e >= cal_date]
            if future and (future[0] - cal_date).days <= 60:
                days_to = (future[0] - cal_date).days
            else:
                days_to = 999

            # days_from_earnings: days since last earnings
            past = [e for e in earn_dates_sorted if e < cal_date]
            if past and (cal_date - past[-1]).days <= 60:
                days_from = (cal_date - past[-1]).days
            else:
                days_from = 999

            pre5 = 1 if 1 <= days_to <= 5 else 0
            post2 = 1 if 0 <= days_from <= 2 else 0

            records.append({
                "date": d,
                "symbol": sym,
                "days_to_earnings": days_to,
                "days_from_earnings": days_from,
                "pre_earnings_5d": pre5,
                "post_earnings_2d": post2,
                "earnings_week": int(pre5 or post2),
            })

    result = pd.DataFrame(records).set_index(["date", "symbol"])
    result.to_parquet(cache)
    return result


# ── 6. Master loader ──────────────────────────────────────────────────────────

def load_all_daily(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load all daily features and return a (date, symbol) MultiIndex DataFrame.

    Market-wide features (macro, VIX, calendar) are broadcast to all tickers
    each day.  Per-stock features (earnings) vary by (date, symbol).

    Parameters
    ----------
    tickers    : list of stock symbols (must match keys in data_dict)
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame with MultiIndex (date, symbol), columns:
        From macro     : yield_10y, short_rate, yield_curve, hy_spread
        From VIX       : vix_level, vix_3m, vix_term_ratio, vix_1d_chg
        From calendar  : fomc_day, nfp_day, cpi_day, macro_event_day
        From earnings  : days_to_earnings, days_from_earnings,
                         pre_earnings_5d, post_earnings_2d, earnings_week
    """
    macro = fetch_macro_daily(start_date, end_date)
    vix = fetch_vix_term(start_date, end_date)
    cal = build_economic_calendar(start_date, end_date)
    earn = build_earnings_features(tickers, start_date, end_date)

    # Align market-wide frames on date index
    market_wide = macro.join(vix, how="outer").join(cal, how="outer")
    market_wide = market_wide.ffill(limit=3)
    market_wide.index = pd.to_datetime(market_wide.index)
    market_wide.index.name = "date"

    # Broadcast market-wide data to (date, symbol) MultiIndex
    # by cross-joining with ticker list
    date_range = pd.date_range(start_date, end_date, freq="D")
    idx = pd.MultiIndex.from_product(
        [date_range, tickers], names=["date", "symbol"]
    )
    wide_broadcast = (
        pd.DataFrame(index=idx)
        .join(market_wide, on="date")
    )

    # Join per-stock earnings features
    earn.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(earn.index.get_level_values("date")),
         earn.index.get_level_values("symbol")],
        names=["date", "symbol"],
    )
    result = wide_broadcast.join(earn, how="left")

    return result

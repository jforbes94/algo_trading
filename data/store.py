import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.requests import StockBarsRequest

from data.fetcher import RAW_COLS, _parse_interval, _parse_period, add_indicators, fetch_raw, _get_client, _to_alpaca
from data.universe import get_etf_list


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular US market hours bars (9:30am–4:00pm ET = 13:30–20:00 UTC)."""
    idx = pd.to_datetime(df.index, utc=True)
    hour = idx.hour
    minute = idx.minute
    in_market = ((hour == 13) & (minute >= 30)) | ((hour >= 14) & (hour <= 19))
    return df.loc[in_market]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _path(symbol: str, interval: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")


def load(symbol: str, interval: str = "1h", period: str = "6mo") -> pd.DataFrame:
    path = _path(symbol, interval)

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        last_ts = existing.index.max()
        new = fetch_raw(symbol, start=last_ts, interval=interval)
        new = new[new.index > last_ts]
        if not new.empty:
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined.to_parquet(path)
            print(f"{symbol}: +{len(new)} new bars (total {len(combined)})")
        else:
            combined = existing
            print(f"{symbol}: already up to date ({len(combined)} bars)")
    else:
        start = _parse_period(period)
        combined = fetch_raw(symbol, start=start, interval=interval)
        combined.to_parquet(path)
        print(f"{symbol}: fetched {len(combined)} bars -> {path}")

    return add_indicators(combined)


def bulk_load(symbols: list[str], interval: str = "1h", period: str = "1y", batch_size: int = 50) -> tuple[list, list]:
    start = _parse_period(period)
    saved, failed = [], []
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    for bi, batch in enumerate(batches, 1):
        print(f"Batch {bi}/{len(batches)} ({len(batch)} symbols)...", end=" ", flush=True)
        prev = len(saved)
        try:
            alpaca_batch = [_to_alpaca(s) for s in batch]
            request = StockBarsRequest(
                symbol_or_symbols=alpaca_batch,
                timeframe=_parse_interval(interval),
                start=start,
                end=datetime.now(),
            )
            df_all = _get_client().get_stock_bars(request).df

            for sym in batch:
                try:
                    alpaca_sym = _to_alpaca(sym)
                    sym_df = df_all.loc[alpaca_sym].copy() if isinstance(df_all.index, pd.MultiIndex) else df_all.copy()
                    sym_df.index = pd.to_datetime(sym_df.index)
                    sym_df.index.name = "datetime"
                    sym_df.columns = [c.lower() for c in sym_df.columns]
                    sym_df = sym_df[[c for c in RAW_COLS if c in sym_df.columns]]
                    sym_df.to_parquet(_path(sym, interval))
                    saved.append(sym)
                except KeyError:
                    failed.append(sym)
        except Exception as e:
            print(f"ERROR batch {bi}: {e} — retrying symbols individually...")
            for sym in batch:
                try:
                    sym_df = fetch_raw(sym, start, interval)
                    sym_df.to_parquet(_path(sym, interval))
                    saved.append(sym)
                except Exception as sym_e:
                    print(f"  FAILED {sym}: {sym_e}")
                    failed.append(sym)

        print(f"{len(saved) - prev}/{len(batch)} saved")

    return saved, failed


def load_etf_universe(interval: str = "1h", period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch and cache hourly bars for SPY and the 11 GICS sector ETFs.

    For each ETF the function follows the same incremental-refresh pattern used
    by ``load()``: if a parquet cache already exists only new bars are fetched;
    otherwise the full ``period`` of history is downloaded.  ETFs that fail to
    fetch are skipped with a warning instead of raising an exception.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ETF ticker -> DataFrame with indicators added (same format
        as the per-stock DataFrames returned by ``load()``).
    """
    etfs = get_etf_list()
    result: dict[str, pd.DataFrame] = {}

    for etf in etfs:
        path = _path(etf, interval)
        try:
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                last_ts = existing.index.max()
                new = fetch_raw(etf, start=last_ts, interval=interval)
                new = new[new.index > last_ts]
                if not new.empty:
                    combined = pd.concat([existing, new])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    combined.to_parquet(path)
                    print(f"{etf}: +{len(new)} new bars (total {len(combined)})")
                else:
                    combined = existing
                    print(f"{etf}: already up to date ({len(combined)} bars)")
            else:
                start = _parse_period(period)
                combined = fetch_raw(etf, start=start, interval=interval)
                combined.to_parquet(path)
                print(f"{etf}: fetched {len(combined)} bars -> {path}")

            result[etf] = add_indicators(_filter_market_hours(combined))
        except Exception as e:
            print(f"WARNING: failed to load ETF {etf}: {e} — skipping")

    return result


def backfill(symbol: str, interval: str = "1h", extra_period: str = "1y") -> int:
    """Fetch data *before* the earliest cached timestamp and prepend it.

    Returns the number of bars prepended, or 0 if nothing new was found.
    """
    path = _path(symbol, interval)
    if not os.path.exists(path):
        print(f"{symbol}: no cache found — run load() first")
        return 0

    existing = pd.read_parquet(path)
    earliest_ts = existing.index.min()

    end_dt = earliest_ts
    start_dt = earliest_ts - timedelta(days=int(extra_period.rstrip("y")) * 365
                                        if extra_period.endswith("y")
                                        else int(extra_period.rstrip("mo")) * 30)

    older = fetch_raw(symbol, start=start_dt, interval=interval, end=end_dt)
    older = older[older.index < earliest_ts]

    if older.empty:
        print(f"{symbol}: no older data found before {earliest_ts.date()}")
        return 0

    combined = pd.concat([older, existing])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_parquet(path)
    print(f"{symbol}: prepended {len(older)} bars (now {len(combined)} total, back to {combined.index.min().date()})")
    return len(older)


def bulk_backfill(symbols: list[str], interval: str = "1h", extra_period: str = "1y", batch_size: int = 50) -> dict:
    """Backfill all symbols in batches. Returns dict of symbol -> bars prepended."""
    results = {}
    failed = []

    # Parse extra_period to a timedelta
    if extra_period.endswith("y"):
        delta = timedelta(days=int(extra_period[:-1]) * 365)
    elif extra_period.endswith("mo"):
        delta = timedelta(days=int(extra_period[:-2]) * 30)
    else:
        delta = timedelta(days=int(extra_period[:-1]))

    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    for bi, batch in enumerate(batches, 1):
        print(f"Backfill batch {bi}/{len(batches)} ({len(batch)} symbols)...", end=" ", flush=True)
        batch_total = 0

        # Determine per-symbol date ranges from their caches
        batch_with_ranges = []
        for sym in batch:
            path = _path(sym, interval)
            if not os.path.exists(path):
                failed.append(sym)
                continue
            existing = pd.read_parquet(path)
            earliest = existing.index.min()
            batch_with_ranges.append((sym, existing, earliest, earliest - delta))

        if not batch_with_ranges:
            print("0 cached — skipped")
            continue

        # Use the earliest start across this batch
        global_start = min(r[3] for r in batch_with_ranges)
        global_end   = min(r[2] for r in batch_with_ranges)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=[_to_alpaca(r[0]) for r in batch_with_ranges],
                timeframe=_parse_interval(interval),
                start=global_start,
                end=global_end,
            )
            df_all = _get_client().get_stock_bars(request).df

            for sym, existing, earliest, start_dt in batch_with_ranges:
                try:
                    alpaca_sym = _to_alpaca(sym)
                    sym_df = df_all.loc[alpaca_sym].copy() if isinstance(df_all.index, pd.MultiIndex) else df_all.copy()
                    sym_df.index = pd.to_datetime(sym_df.index)
                    sym_df.index.name = "datetime"
                    sym_df.columns = [c.lower() for c in sym_df.columns]
                    sym_df = sym_df[[c for c in RAW_COLS if c in sym_df.columns]]
                    older = sym_df[sym_df.index < earliest]
                    if older.empty:
                        results[sym] = 0
                        continue
                    combined = pd.concat([older, existing])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    combined.to_parquet(_path(sym, interval))
                    results[sym] = len(older)
                    batch_total += len(older)
                except KeyError:
                    results[sym] = 0
        except Exception as e:
            print(f"ERROR batch {bi}: {e} — falling back to per-symbol...")
            for sym, existing, earliest, start_dt in batch_with_ranges:
                try:
                    older = fetch_raw(sym, start=start_dt, interval=interval, end=earliest)
                    older = older[older.index < earliest]
                    if older.empty:
                        results[sym] = 0
                        continue
                    combined = pd.concat([older, existing])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    combined.to_parquet(_path(sym, interval))
                    results[sym] = len(older)
                    batch_total += len(older)
                except Exception as sym_e:
                    print(f"  FAILED {sym}: {sym_e}")
                    results[sym] = -1
                    failed.append(sym)

        print(f"+{batch_total} bars")

    total_new = sum(v for v in results.values() if v > 0)
    print(f"\nBackfill complete: {total_new:,} bars prepended across {len(results)} symbols")
    if failed:
        print(f"Failed: {failed}")
    return results


def refresh_all(interval: str = "1h") -> dict:
    import glob
    pattern = os.path.join(CACHE_DIR, f"*_{interval}.parquet")
    files = glob.glob(pattern)
    results = {}
    refreshed = 0
    new_bars_total = 0
    up_to_date = 0

    for f in sorted(files):
        basename = os.path.basename(f)
        symbol = basename.replace(f"_{interval}.parquet", "")
        try:
            existing = pd.read_parquet(f)
            last_ts = existing.index.max()
            new = fetch_raw(symbol, start=last_ts, interval=interval)
            new = new[new.index > last_ts]
            if not new.empty:
                combined = pd.concat([existing, new])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                combined.to_parquet(f)
                n = len(new)
                print(f"{symbol}: +{n} new bars")
                results[symbol] = n
                new_bars_total += n
                refreshed += 1
            else:
                print(f"{symbol}: up to date")
                results[symbol] = 0
                up_to_date += 1
        except Exception as e:
            print(f"{symbol}: ERROR — {e}")
            results[symbol] = -1

    print(f"\n{refreshed} symbols refreshed, {new_bars_total} new bars added, {up_to_date} already up to date")
    return results


if __name__ == "__main__":
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    df = load("AAPL", interval="1h", period="6mo")
    print(df.tail(3).to_string())

    path = _path("AAPL", "1h")
    size = _os.path.getsize(path)
    print(f"\nParquet size: {size:,} bytes  ({size/1024:.1f} KB)")

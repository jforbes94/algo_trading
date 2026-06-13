import os
import glob

import pandas as pd

try:
    import pandas_market_calendars as mcal
    _USE_MCal = True
except ImportError:
    _USE_MCal = False

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def _nyse_schedule(start, end):
    nyse = mcal.get_calendar("NYSE")
    return nyse.schedule(start_date=start.date(), end_date=end.date())


def check_symbol(path):
    sym = os.path.basename(path).replace("_1h.parquet", "")
    df = pd.read_parquet(path)
    issues = {}

    dupes = df.index.duplicated().sum()
    if dupes:
        issues["duplicates"] = dupes

    bad = ((df["close"].isna()) | (df["close"] <= 0) | (df["open"] <= 0) |
           (df["high"] <= 0) | (df["low"] <= 0) | (df["high"] < df["low"])).sum()
    if bad:
        issues["bad_values"] = int(bad)

    df_sorted = df.sort_index()
    if _USE_MCal:
        try:
            sched = _nyse_schedule(df_sorted.index[0], df_sorted.index[-1])
            missing_bars = 0
            for _, row in sched.iterrows():
                open_ts, close_ts = row["market_open"], row["market_close"]
                expected = round((close_ts - open_ts).total_seconds() / 3600)
                actual = int(((df_sorted.index >= open_ts) & (df_sorted.index <= close_ts)).sum())
                if actual < expected - 1:
                    missing_bars += expected - actual
            if missing_bars:
                issues["gaps"] = missing_bars
        except Exception:
            pass
    else:
        # Filter to regular market hours only (13:00-21:00 UTC covers 9am-5pm ET in both EST/EDT)
        mkt = df_sorted[df_sorted.index.hour.isin(range(13, 21))]
        if len(mkt) > 1:
            mkt_ts = pd.Series(mkt.index.tz_localize(None) if mkt.index.tz is not None else mkt.index)
            deltas = mkt_ts.diff().dropna()
            # >2h within-session gap, <8h to exclude cross-day transitions (~17h)
            gap_mask = (deltas > pd.Timedelta(hours=2)) & (deltas < pd.Timedelta(hours=8))
            n_gaps = gap_mask.sum()
            if n_gaps:
                gap_details = []
                for i in gap_mask[gap_mask].index:
                    gap_details.append(f"{mkt_ts.iloc[i-1]} -> {mkt_ts.iloc[i]} ({deltas.loc[i]})")
                issues["gaps"] = gap_details[:5]

    return sym, issues


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*_1h.parquet")))
    total = len(files)
    clean = []
    dirty = {}

    for f in files:
        sym, issues = check_symbol(f)
        if issues:
            dirty[sym] = issues
        else:
            clean.append(sym)

    for sym, issues in dirty.items():
        parts = []
        if "duplicates" in issues:
            parts.append(f"duplicates={issues['duplicates']}")
        if "bad_values" in issues:
            parts.append(f"bad_values={issues['bad_values']}")
        if "gaps" in issues:
            g = issues["gaps"]
            if isinstance(g, list):
                parts.append(f"gaps={len(g)} (e.g. {g[0]})")
            else:
                parts.append(f"gaps={g}")
        print(f"  {sym}: {', '.join(parts)}")

    print(f"\nTotal symbols checked : {total}")
    print(f"Symbols with issues   : {len(dirty)}")
    if dirty:
        by_type = {}
        for sym, iss in dirty.items():
            for k in iss:
                by_type.setdefault(k, []).append(sym)
        for kind, syms in by_type.items():
            print(f"  {kind}: {len(syms)} symbols")
    print(f"Symbols clean         : {len(clean)}")
    if not _USE_MCal:
        print("(gap check used weekday heuristic; install pandas_market_calendars for full accuracy)")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()

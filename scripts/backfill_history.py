"""Prepend an extra year of history to every cached parquet file.

Run once before re-running the pipeline to get 2 years of data.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.store import bulk_backfill, CACHE_DIR
from data.universe import get_etf_list, get_sp500

EXTRA_PERIOD = "1y"   # how far back to extend each symbol
INTERVAL     = "1h"

# ── 1. ETFs first (fast, only 12 tickers) ─────────────────────────────────────
print("=== Backfilling ETFs ===")
etfs = get_etf_list()
etf_results = bulk_backfill(etfs, interval=INTERVAL, extra_period=EXTRA_PERIOD, batch_size=12)
print()

# ── 2. All cached stock symbols ────────────────────────────────────────────────
import glob
pattern = os.path.join(CACHE_DIR, f"*_{INTERVAL}.parquet")
cached_files = glob.glob(pattern)
etf_set = set(get_etf_list())
symbols = [
    os.path.basename(f).replace(f"_{INTERVAL}.parquet", "")
    for f in cached_files
    if os.path.basename(f).replace(f"_{INTERVAL}.parquet", "") not in etf_set
]
print(f"=== Backfilling {len(symbols)} stocks ===")
stock_results = bulk_backfill(symbols, interval=INTERVAL, extra_period=EXTRA_PERIOD, batch_size=50)

# ── Summary ────────────────────────────────────────────────────────────────────
all_results = {**etf_results, **stock_results}
prepended = {k: v for k, v in all_results.items() if v > 0}
nothing   = {k: v for k, v in all_results.items() if v == 0}
failed    = {k: v for k, v in all_results.items() if v < 0}

print(f"\n{'='*50}")
print(f"Symbols with new bars : {len(prepended)}")
print(f"Already at full range : {len(nothing)}")
print(f"Failed                : {len(failed)}")
if failed:
    print(f"  Failed tickers: {list(failed.keys())}")

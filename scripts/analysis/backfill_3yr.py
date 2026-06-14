"""Backfill 3 more years of history for all cached symbols."""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.store import bulk_backfill, CACHE_DIR
from data.universe import get_etf_list
import glob

EXTRA_PERIOD = "3y"
INTERVAL     = "1h"

print("=== Backfilling ETFs (3yr) ===")
etfs = get_etf_list()
bulk_backfill(etfs, interval=INTERVAL, extra_period=EXTRA_PERIOD, batch_size=12)

print("\n=== Backfilling stocks (3yr) ===")
pattern = os.path.join(CACHE_DIR, f"*_{INTERVAL}.parquet")
cached_files = glob.glob(pattern)
etf_set = set(get_etf_list())
symbols = [
    os.path.basename(f).replace(f"_{INTERVAL}.parquet", "")
    for f in cached_files
    if os.path.basename(f).replace(f"_{INTERVAL}.parquet", "") not in etf_set
]
print(f"Symbols to backfill: {len(symbols)}")
bulk_backfill(symbols, interval=INTERVAL, extra_period=EXTRA_PERIOD, batch_size=50)
print("\nDone.")

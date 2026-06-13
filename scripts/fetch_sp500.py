import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")

from data.universe import get_sp500
from data.store import bulk_load

tickers = get_sp500()
print(f"Fetching {len(tickers)} S&P 500 symbols  |  interval=1h  |  period=1y\n")

saved, failed = bulk_load(tickers, interval="1h", period="1y")

print(f"\nDone: {len(saved)}/{len(tickers)} saved")
if failed:
    print(f"Failed ({len(failed)}): {failed}")

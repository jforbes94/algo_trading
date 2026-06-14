import pandas as pd
import numpy as np

df = pd.read_parquet(r"E:\Github\algo_trading\data\cache\AAPL_1h.parquet")
df.index = pd.to_datetime(df.index, utc=True)
hour = df.index.hour
minute = df.index.minute
in_market = ((hour == 13) & (minute >= 30)) | ((hour >= 14) & (hour <= 19))
df = df.loc[in_market]
ts = df.index

# What shift(35) actually covers
gaps35 = ts[35:] - ts[:-35]
med35 = gaps35.median().total_seconds() / 3600
print(f"shift(35): median gap = {med35:.1f}h = {med35/6:.1f} trading days (expected 5)")

print()
print("forward_return target times for first 12 bars:")
for i in range(min(12, len(ts) - 4)):
    t = ts[i]
    tgt = ts[i + 4]
    gap = (tgt - t).total_seconds() / 3600
    status = "OVERNIGHT" if gap > 4 else "OK"
    print(f"  {str(t)[:16]} -> {str(tgt)[11:16]} UTC  gap={gap:.0f}h  {status}")

print()
print("Summary: which UTC hours produce valid (same-day) 4h forward returns?")
crosses = pd.Series(
    [(ts[i + 4] - ts[i]).total_seconds() / 3600 > 4 for i in range(len(ts) - 4)],
    index=ts[:-4]
)
by_hour = crosses.groupby(crosses.index.hour).mean()
for h, pct in by_hour.items():
    label = "ALL overnight" if pct == 1.0 else ("NEVER overnight" if pct == 0.0 else f"{pct*100:.0f}% overnight")
    print(f"  Hour {h:02d} UTC ({h-4:02d}:00 ET): {label}")

print()
print("Actual shift() time gaps vs intended:")
rows = [
    ("ret_1h",  "shift(1)",   1,  "1h"),
    ("ret_4h",  "shift(4)",   4,  "4h"),
    ("ret_1d",  "shift(7)",   7,  "1 trading day"),
    ("ret_5d",  "shift(35)",  35, "5 trading days"),
    ("fwd_ret", "shift(-4)",  4,  "4h forward"),
]
for name, label, sh, intended in rows:
    valid = len(ts) - sh
    if sh > 0:
        gaps = ts[sh:] - ts[:valid]
    else:
        gaps = ts[-sh:] - ts[:-(-sh)]  # abs shift
    med_h = gaps.median().total_seconds() / 3600
    max_h = gaps.max().total_seconds() / 3600
    print(f"  {name:10s} ({label:9s}): intended={intended:16s}  actual median={med_h:.0f}h  max={max_h:.0f}h")

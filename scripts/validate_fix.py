import sys
sys.path.insert(0, r"E:\Github\algo_trading")
import pandas as pd
from features.engineer import load_universe, build_features
from data.store import load_etf_universe
from data.universe import get_sector_map

data = load_universe()
data = dict(list(data.items())[:5])
etfs = load_etf_universe()
sector_map = get_sector_map()

df = build_features(data, etfs, sector_map)

dt_idx = df.index.get_level_values("datetime")
hours = dt_idx.hour

print("=== FORWARD_RETURN VALIDATION ===")
fwd_valid = df["forward_return"].notna()
print("forward_return valid by UTC hour:")
for h in sorted(hours.unique()):
    mask = hours == h
    pct = fwd_valid[mask].mean()
    print(f"  {h:02d}:00 UTC ({h-4:02d}:00 ET): {pct*100:.0f}% valid")

print()
print("=== RET_4H VALIDATION ===")
r4h_valid = df["ret_4h"].notna()
print("ret_4h valid by UTC hour:")
for h in sorted(hours.unique()):
    mask = hours == h
    pct = r4h_valid[mask].mean()
    print(f"  {h:02d}:00 UTC ({h-4:02d}:00 ET): {pct*100:.0f}% valid")

print()
total = len(df)
clean = df.dropna(subset=["forward_return"])
print(f"Rows before dropna : {total}")
print(f"Rows after dropna  : {len(clean)}")
print(f"Dropped            : {total - len(clean)} ({(1 - len(clean)/total)*100:.1f}%)")

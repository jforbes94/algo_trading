"""
Analyze a completed backtest run.

Usage:
    python scripts/analyze_backtest.py 1yr
    python scripts/analyze_backtest.py 2024
"""
import sys
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = sys.argv[1] if len(sys.argv) > 1 else "1yr"
OUT_DIR = os.path.join(ROOT, "outputs", LABEL)

# ── Summary ───────────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "summary.txt")
if os.path.exists(summary_path):
    print("=" * 60)
    print(f"SUMMARY  ({LABEL})")
    print("=" * 60)
    with open(summary_path) as f:
        print(f.read())

# ── Load trade log ────────────────────────────────────────────────────────────
log_path = os.path.join(OUT_DIR, "trade_log.csv")
if not os.path.exists(log_path):
    print(f"No trade log found at {log_path}")
    sys.exit(1)

log = pd.read_csv(log_path)
log["entry_time"] = pd.to_datetime(log["entry_time"])
new = log["is_new_entry"]

# ── Basic ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("BASIC")
print("=" * 60)
print(f"Rows            : {len(log):,}")
print(f"Unique symbols  : {log['symbol'].nunique()}")
print(f"Date range      : {log['entry_time'].min().date()} to {log['entry_time'].max().date()}")
print(f"New entries     : {new.sum():,} ({new.mean()*100:.1f}%)  |  Holds: {(~new).sum():,} ({(~new).mean()*100:.1f}%)")
bars_per_ts = log.groupby("entry_time").size()
print(f"Positions/bar   : mean {bars_per_ts.mean():.1f}  min {bars_per_ts.min()}  max {bars_per_ts.max()}")

# ── Returns ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RETURNS")
print("=" * 60)
gr, nr, ps = log["gross_return"], log["net_return"], log["position_size"]
print(f"Gross mean/bar  : {gr.mean()*100:.4f}%")
print(f"Net mean/bar    : {nr.mean()*100:.4f}%")
print(f"Win rate all    : {log['win'].mean()*100:.2f}%")
print(f"Win rate new    : {log[new]['win'].mean()*100:.2f}%")
print(f"Win rate holds  : {log[~new]['win'].mean()*100:.2f}%")
print(f"Net PnL         : ${log['dollar_pnl'].sum():>10,.2f}")
print(f"Gross PnL       : ${(gr * ps).sum():>10,.2f}")
print(f"Costs paid      : ${((gr - nr) * ps).sum():>10,.2f}")

# ── Kelly weights & scores ────────────────────────────────────────────────────
print()
print("=" * 60)
print("KELLY WEIGHTS & SCORES")
print("=" * 60)
w = log["weight"]
print(f"Weight range    : {w.min():.4f} to {w.max():.4f}  (mean {w.mean():.4f})")
p = log["proba"]
print(f"Score range     : {p.min():.4f} to {p.max():.4f}  (mean {p.mean():.4f}  std {p.std():.4f})")

# ── By rank ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("WIN RATE & AVG GROSS RETURN BY RANK")
print("=" * 60)
by_rank = log.groupby("rank").agg(
    n=("win", "count"),
    win_rate=("win", "mean"),
    avg_gross=("gross_return", "mean"),
).head(10)
by_rank["win_rate"] = by_rank["win_rate"] * 100
by_rank["avg_gross"] = by_rank["avg_gross"] * 100
print(by_rank.round(3).to_string())

# ── Top / bottom stocks ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("TOP 10 STOCKS BY NET PnL")
print("=" * 60)
by_sym = log.groupby("symbol").agg(
    n_bars=("dollar_pnl", "count"),
    total_pnl=("dollar_pnl", "sum"),
    win_rate=("win", "mean"),
    avg_gross=("gross_return", "mean"),
).sort_values("total_pnl", ascending=False)
by_sym["win_rate"] = by_sym["win_rate"] * 100
by_sym["avg_gross"] = by_sym["avg_gross"] * 100
print(by_sym.head(10).round(2).to_string())

print()
print("BOTTOM 5 STOCKS BY NET PnL")
print(by_sym.tail(5).round(2).to_string())

# ── Sector ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SECTOR BREAKDOWN")
print("=" * 60)
by_sec = log.groupby("sector").agg(
    n_bars=("dollar_pnl", "count"),
    total_pnl=("dollar_pnl", "sum"),
    win_rate=("win", "mean"),
).sort_values("total_pnl", ascending=False)
by_sec["win_rate"] = by_sec["win_rate"] * 100
print(by_sec.round(2).to_string())

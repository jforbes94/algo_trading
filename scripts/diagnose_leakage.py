"""
Systematic leakage / bias diagnostic.
Checks:
  1. Whether data includes extended hours (pre/post market) bars
  2. Year-by-year backtest performance (2022 bear market should be negative)
  3. Feature-target (forward_return) Pearson correlations
  4. Exact feature list used by the model (confirm no forward_return / y)
  5. AUC per fold
  6. Per-period return distribution stats
"""
import sys, os
ROOT = r"E:\Github\algo_trading"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features.engineer import load_universe, build_features
from models.predictor import WalkForwardModel
from strategy.signals import generate_signals
from backtest.backtester import run
from data.store import load_etf_universe
from data.universe import get_sector_map

print("=" * 60)
print("LEAKAGE DIAGNOSTIC")
print("=" * 60)

# ── 1. Load a sample of data & check timestamps ───────────────────────────────
print("\n[1] Checking timestamps for extended hours...")
data_dict = load_universe()
sample_sym = list(data_dict.keys())[0]
sample_df  = data_dict[sample_sym]
sample_df.index = pd.to_datetime(sample_df.index, utc=True)

hours = sample_df.index.hour
print(f"  Symbol: {sample_sym}")
print(f"  Hour range in data: {hours.min()}:00 – {hours.max()}:00 UTC")
# US market hours = 13:30–20:00 UTC (9:30am–4pm ET)
market_mask = (hours >= 13) & (hours <= 20)
extended_pct = (~market_mask).mean() * 100
print(f"  Extended-hours bars: {extended_pct:.1f}% of total")
print(f"  Total bars: {len(sample_df)}  |  Market-hours bars: {market_mask.sum()}")

# ── 2. Build features (small universe for speed) ──────────────────────────────
print("\n[2] Building features on 20 stocks...")
small_dict  = dict(list(data_dict.items())[:20])
etf_dict    = load_etf_universe()
sector_map  = get_sector_map()
features_df = build_features(small_dict, etf_dict=etf_dict, sector_map=sector_map)

drop_cols   = {"y", "forward_return"}
feat_cols   = [c for c in features_df.columns if c not in drop_cols]
print(f"  Features used by model ({len(feat_cols)}): {feat_cols}")

# ── 3. Feature-target correlations ───────────────────────────────────────────
print("\n[3] Pearson correlation of each feature with forward_return...")
df_clean = features_df[feat_cols + ["forward_return"]].dropna()
corrs = df_clean[feat_cols].corrwith(df_clean["forward_return"]).abs().sort_values(ascending=False)
print(corrs.to_string())
print(f"\n  Max |corr| = {corrs.max():.4f}  (>0.3 is very suspicious)")

# ── 4. Walk-forward AUC per fold ─────────────────────────────────────────────
print("\n[4] Walk-forward AUC per fold (should be 0.50–0.65)...")
model = WalkForwardModel(n_splits=4)
proba = model.fit_predict(features_df)
# AUC already printed inside fit_predict — that's enough

# ── 5. Year-by-year performance on full dataset ───────────────────────────────
print("\n[5] Year-by-year backtest performance (2022 should be negative)...")
signals = generate_signals(proba, features_df, top_n=5, bottom_n=5)  # small top_n for speed
results = run(signals, holding_period=4, cost_bps=5.0)
eq = results["equity_curve"]
eq.index = pd.to_datetime(eq.index, utc=True)

yearly = {}
for year in eq.index.year.unique():
    yr_eq = eq[eq.index.year == year]
    if len(yr_eq) < 2:
        continue
    start_val = yr_eq.iloc[0]
    end_val   = yr_eq.iloc[-1]
    ret = (end_val / start_val - 1) * 100
    yearly[year] = ret
    print(f"  {year}: {ret:+.1f}%")

# ── 6. Per-period return distribution ────────────────────────────────────────
print("\n[6] Per-period return distribution...")
eq_rets = eq.pct_change().dropna()
print(f"  Mean  : {eq_rets.mean()*100:.4f}%")
print(f"  Std   : {eq_rets.std()*100:.4f}%")
print(f"  Min   : {eq_rets.min()*100:.4f}%")
print(f"  Max   : {eq_rets.max()*100:.4f}%")
print(f"  >1%   : {(eq_rets > 0.01).mean()*100:.1f}% of periods")
print(f"  >5%   : {(eq_rets > 0.05).mean()*100:.1f}% of periods (should be near 0 for real trading)")

# ── 7. Check correlation of y with any feature ───────────────────────────────
print("\n[7] Pearson correlation of each feature with y...")
df_y = features_df[feat_cols + ["y"]].dropna()
corrs_y = df_y[feat_cols].corrwith(df_y["y"]).abs().sort_values(ascending=False)
print(corrs_y.head(10).to_string())
print(f"\n  Max |corr with y| = {corrs_y.max():.4f}  (>0.5 means near-perfect prediction)")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

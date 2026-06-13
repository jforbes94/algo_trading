import sys
import os

# Ensure the project root is on sys.path so all packages resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.engineer import load_universe, build_features
from models.predictor import WalkForwardModel
from strategy.signals import generate_signals
from backtest.backtester import run, print_summary, trade_log, trade_summary
from data.store import load_etf_universe
from data.universe import get_sector_map

# ------------------------------------------------------------------
# 1. Load universe and slice to first 20 symbols for speed
# ------------------------------------------------------------------
print("Loading universe...")
data_dict = load_universe()
print(f"Symbols loaded: {len(data_dict)}")

# ------------------------------------------------------------------
# 1b. Load ETF data and sector map
# ------------------------------------------------------------------
print("\nLoading ETF data...")
etf_dict = load_etf_universe()
print(f"ETFs loaded: {list(etf_dict.keys())}")

print("Loading sector map...")
sector_map = get_sector_map()
print(f"Sector map: {len(sector_map)} symbols mapped")

# ------------------------------------------------------------------
# 2. Build features
# ------------------------------------------------------------------
print("\nBuilding features...")
features_df = build_features(data_dict, etf_dict=etf_dict, sector_map=sector_map)
print(f"Features shape : {features_df.shape}")
print(f"Columns        : {features_df.columns.tolist()}")
print(f"Index names    : {features_df.index.names}")

# ------------------------------------------------------------------
# 3. Walk-forward model — fit and predict
# ------------------------------------------------------------------
print("\nRunning walk-forward model (n_splits=4)...")
model = WalkForwardModel(n_splits=4)
proba = model.fit_predict(features_df)
print(f"Predictions shape : {proba.shape}")
print(f"Series name       : {proba.name}")
print(f"Value range       : [{proba.min():.4f}, {proba.max():.4f}]")

# ------------------------------------------------------------------
# 4. Generate signals
# ------------------------------------------------------------------
print("\nGenerating signals (top_n=20, bottom_n=20)...")
signals = generate_signals(proba, features_df, top_n=20, bottom_n=20)
print(f"Signals shape   : {signals.shape}")
print(f"Signals columns : {signals.columns.tolist()}")
print(f"Signal counts   :\n{signals['signal'].value_counts().to_string()}")

# ------------------------------------------------------------------
# 5. Run backtest
# ------------------------------------------------------------------
print("\nRunning backtest (holding_period=4h, cost=5bps)...")
results = run(signals, holding_period=4, cost_bps=5.0)
print("\n=== Backtest Summary ===")
print_summary(results)

# ------------------------------------------------------------------
# 6. Trade log
# ------------------------------------------------------------------
print("\nGenerating trade log...")
log = trade_log(signals, holding_period=4, cost_bps=5.0, sector_map=sector_map)
log_path = os.path.join(ROOT, "trade_log.csv")
log.to_csv(log_path, index=False)
print(f"Trade log saved: {log_path}  ({len(log):,} trades)")

print("\n=== Top 20 stocks by total PnL contribution ===")
summary = trade_summary(log)
print(summary.head(20).to_string(index=False))

print("\n=== Bottom 10 stocks by total PnL contribution ===")
print(summary.tail(10).to_string(index=False))

print(f"\nTotal unique stocks traded: {log['symbol'].nunique()}")
print(f"Most selected stock: {log['symbol'].value_counts().index[0]} "
      f"({log['symbol'].value_counts().iloc[0]} times)")

# ------------------------------------------------------------------
# 7. Feature importance
# ------------------------------------------------------------------
print("\nTop 10 features:")
print(model.feature_importance().head(10).to_string())

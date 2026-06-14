import pandas as pd
import numpy as np
from scipy import stats

log = pd.read_csv(r"E:\Github\algo_trading\outputs\full_5yr\trade_log.csv")
n_trades = len(log)
n_periods = log["entry_time"].nunique()
win_rate = log["win"].mean()

print("=== SAMPLE SIZES ===")
print(f"Trading periods  : {n_periods:,}")
print(f"Individual trades: {n_trades:,}")
print(f"Observed win rate: {win_rate*100:.2f}%")

se = np.sqrt(0.5 * 0.5 / n_trades)
z = (win_rate - 0.5) / se
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
verdict = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
print()
print("=== IS 50.28% DIFFERENT FROM 50%? ===")
print(f"z-score : {z:.3f}")
print(f"p-value : {p_val:.3f}  (need < 0.05)")
print(f"Verdict : {verdict}")

print()
print("=== TRADES NEEDED TO RELIABLY DETECT AN EDGE (95% conf, 80% power) ===")
z_alpha, z_beta = 1.96, 0.84
for edge in [0.01, 0.02, 0.03, 0.05]:
    n_needed = int((z_alpha + z_beta)**2 * 0.25 / edge**2)
    have = "YES" if n_trades >= n_needed else "NO "
    print(f"  {edge*100:.0f}% edge -> win rate {50+edge*100:.0f}%:  need {n_needed:>8,}  [have enough? {have}]")

print()
print("=== BUCKET WIN RATES WITH 95% CONFIDENCE INTERVALS ===")
bins   = [0, 0.54, 0.56, 0.58, 0.60, 1.0]
labels = ["0.51-0.54", "0.54-0.56", "0.56-0.58", "0.58-0.60", ">0.60"]
log["bucket"] = pd.cut(log["proba"], bins=bins, labels=labels)
for bucket, grp in log.groupby("bucket", observed=True):
    n  = len(grp)
    wr = grp["win"].mean()
    se = np.sqrt(wr * (1 - wr) / n)
    lo, hi = (wr - 1.96*se)*100, (wr + 1.96*se)*100
    print(f"  {bucket}: n={n:>6,}  win={wr*100:.1f}%  95% CI [{lo:.1f}%, {hi:.1f}%]")

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from features.engineer import load_universe, build_features
from models.predictor import WalkForwardModel
from data.store import load_etf_universe
from data.universe import get_sector_map

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data_dict = load_universe()
data_dict = dict(list(data_dict.items())[:20])
etf_dict = load_etf_universe()
sector_map = get_sector_map()
features_df = build_features(data_dict, etf_dict=etf_dict, sector_map=sector_map)

feature_cols = [c for c in features_df.columns if c not in {"y", "forward_return"}]
print(f"Features: {len(feature_cols)}  |  Rows: {len(features_df):,}")

# ── Train model for importances ───────────────────────────────────────────────
print("Training walk-forward model...")
model = WalkForwardModel(n_splits=4)
model.fit_predict(features_df)
importance = model.feature_importance()

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
PANEL    = "#313244"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
BLUE     = "#89b4fa"
YELLOW   = "#f9e2af"
BORDER   = "#45475a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.alpha":       0.4,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting feature importance...")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG)

imp_sorted = importance.sort_values(ascending=True)
colors = [BLUE if v >= imp_sorted.median() else SUBTEXT for v in imp_sorted.values]

bars = ax.barh(imp_sorted.index, imp_sorted.values, color=colors, edgecolor="none", height=0.7)

for bar, val in zip(bars, imp_sorted.values):
    ax.text(val + imp_sorted.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=8, color=SUBTEXT)

ax.set_title("Feature Importance  (LightGBM split gain, avg across folds)",
             fontsize=13, color=TEXT, pad=14)
ax.set_xlabel("Importance score", fontsize=10)
ax.axvline(imp_sorted.median(), color=YELLOW, lw=1, linestyle="--", alpha=0.7, label="median")
ax.legend(fontsize=9)
ax.grid(axis="x")
ax.set_facecolor(PANEL)

plt.tight_layout()
out1 = os.path.join(ROOT, "feature_importance.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out1}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Feature distributions: y=0 vs y=1
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting feature distributions...")

n_cols = 5
n_rows = int(np.ceil(len(feature_cols) / n_cols))

fig = plt.figure(figsize=(n_cols * 4, n_rows * 2.8))
fig.patch.set_facecolor(BG)
fig.suptitle("Feature Value Distribution: Outperform (green) vs Underperform (red)",
             fontsize=13, color=TEXT, y=1.01)

df_clean = features_df[feature_cols + ["y"]].dropna()
y0 = df_clean[df_clean["y"] == 0]
y1 = df_clean[df_clean["y"] == 1]

for i, col in enumerate(feature_cols):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.set_facecolor(PANEL)

    # Clip to 1st-99th percentile to avoid extreme outliers crushing the plot
    lo = df_clean[col].quantile(0.01)
    hi = df_clean[col].quantile(0.99)
    bins = np.linspace(lo, hi, 40)

    v0 = y0[col].clip(lo, hi)
    v1 = y1[col].clip(lo, hi)

    ax.hist(v0, bins=bins, density=True, alpha=0.55, color=RED,   label="y=0", linewidth=0)
    ax.hist(v1, bins=bins, density=True, alpha=0.55, color=GREEN, label="y=1", linewidth=0)

    # Median lines
    ax.axvline(v0.median(), color=RED,   lw=1.2, linestyle="--", alpha=0.9)
    ax.axvline(v1.median(), color=GREEN, lw=1.2, linestyle="--", alpha=0.9)

    # Separation score: normalized distance between medians
    spread = abs(v1.median() - v0.median())
    pooled_std = df_clean[col].std()
    sep = spread / pooled_std if pooled_std > 0 else 0

    # Rank by importance for title color
    imp_rank = list(importance.sort_values(ascending=False).index).index(col) + 1 if col in importance.index else 99
    title_color = YELLOW if imp_rank <= 8 else TEXT

    ax.set_title(f"{col}\nsep={sep:.3f}", fontsize=8, color=title_color, pad=4)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", alpha=0.3)

    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
out2 = os.path.join(ROOT, "feature_distributions.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out2}")

print("\nDone. Files written:")
print(f"  {out1}")
print(f"  {out2}")

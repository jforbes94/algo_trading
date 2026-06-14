"""
Feature correlation report.
Computes IC, Pearson, Spearman correlations of every feature with forward_return
over the full history, broken down by year, and saves a PDF + CSV.

Usage:
    python scripts/feature_correlation_report.py
"""

import sys
import os
sys.path.insert(0, r"E:\Github\algo_trading")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from features.engineer import load_universe, build_features
from data.store import load_etf_universe
from data.universe import get_sector_map

OUTPUT_DIR = r"E:\Github\algo_trading\outputs\feature_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG, PANEL, TEXT = "#1e1e2e", "#313244", "#cdd6f4"
SUB, BORDER     = "#a6adc8", "#45475a"
GREEN, RED, BLUE, YELLOW, MAUVE = "#a6e3a1", "#f38ba8", "#89b4fa", "#f9e2af", "#cba6f7"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT, "xtick.color": SUB, "ytick.color": SUB,
    "text.color": TEXT, "grid.color": BORDER, "grid.alpha": 0.4,
    "font.family": "monospace",
})

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if title:  ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    ax.grid(True, alpha=0.3)


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading universe...")
data_dict  = load_universe()
etf_dict   = load_etf_universe()
sector_map = get_sector_map()

print("Building features...")
features = build_features(data_dict, etf_dict, sector_map)
features = features.dropna(subset=["forward_return"])

TARGET = "forward_return"
SKIP   = {TARGET, "y", "above_sma20"}
feature_cols = [
    c for c in features.columns
    if c not in SKIP and pd.api.types.is_numeric_dtype(features[c])
]
n_ts = features.index.get_level_values("datetime").nunique()
print(f"Dataset: {len(features):,} rows | {len(feature_cols)} features | {n_ts:,} timestamps")


# ── 2. Overall Pearson + Spearman correlations ────────────────────────────────
print("Computing overall Pearson / Spearman correlations...")
rows = []
for col in feature_cols:
    valid = features[[col, TARGET]].dropna()
    if len(valid) < 100:
        continue
    pr, pp = stats.pearsonr(valid[col].values, valid[TARGET].values)
    sr, sp = stats.spearmanr(valid[col].values, valid[TARGET].values)
    rows.append({
        "feature": col,
        "pearson_r":   pr, "pearson_p":  pp,
        "spearman_r":  sr, "spearman_p": sp,
        "abs_pearson": abs(pr),
    })
corr_df = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False).reset_index(drop=True)


# ── 3. Per-timestamp IC (cross-sectional Spearman via ranks) ───────────────────
print("Computing per-timestamp IC (cross-sectional Spearman)...")
# Flatten MultiIndex so groupby works unambiguously
flat = features[feature_cols + [TARGET]].reset_index()
datetime_col = "datetime" if "datetime" in flat.columns else flat.columns[0]

# Cross-sectional rank everything at each timestamp
ranked = flat.copy()
ranked[feature_cols + [TARGET]] = (
    flat.groupby(datetime_col)[feature_cols + [TARGET]].rank(pct=True)
)

# Per-timestamp IC: Pearson correlation on ranked data == Spearman on raw
ic_records = {}
for ts, grp in ranked.groupby(datetime_col):
    if len(grp) < 20:
        continue
    target_ranked = grp[TARGET]
    ic_records[ts] = grp[feature_cols].corrwith(target_ranked)

ic_df = pd.DataFrame(ic_records).T
ic_df.index = pd.DatetimeIndex(ic_df.index)
ic_df.index.name = "timestamp"

ic_mean = ic_df[feature_cols].mean()
ic_std  = ic_df[feature_cols].std()
icir    = (ic_mean / ic_std)

# Merge IC metrics into corr_df
ic_stats = ic_mean.rename("ic_mean").to_frame()
ic_stats["ic_std"] = ic_std
ic_stats["icir"]   = icir
ic_stats.index.name = "feature"
ic_stats = ic_stats.reset_index()

corr_df = corr_df.merge(ic_stats, on="feature", how="left")
corr_df = corr_df.sort_values("abs_pearson", ascending=False).reset_index(drop=True)


# ── 4. Year-by-year IC ────────────────────────────────────────────────────────
print("Computing year-by-year IC...")
ic_df["year"] = ic_df.index.year
yearly_ic = ic_df.groupby("year")[feature_cols].mean()


# ── 5. Feature-feature correlation matrix ─────────────────────────────────────
print("Computing feature-feature correlation matrix...")
feat_corr = features[feature_cols].corr(method="pearson")


# ── 6. Save CSV ───────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "feature_correlations.csv")
corr_df.to_csv(csv_path, index=False)
print(f"\nCSV saved: {csv_path}")

print("\n=== TOP 15 FEATURES BY |PEARSON r| WITH forward_return ===")
display = corr_df[["feature", "pearson_r", "spearman_r", "ic_mean", "icir"]].copy()
display["pearson_r"]  = display["pearson_r"].map("{:+.5f}".format)
display["spearman_r"] = display["spearman_r"].map("{:+.5f}".format)
display["ic_mean"]    = display["ic_mean"].map("{:+.5f}".format)
display["icir"]       = display["icir"].map("{:+.4f}".format)
print(display.head(15).to_string(index=False))


# ── 7. PDF report ─────────────────────────────────────────────────────────────
pdf_path = os.path.join(OUTPUT_DIR, "feature_correlation_report.pdf")
pdf = PdfPages(pdf_path)


# ── PAGE 1: Overall correlations ──────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.45,
                        top=0.92, bottom=0.04, left=0.20, right=0.97)
fig.text(0.5, 0.96, "Feature Correlation with forward_return — Full 5-Year History",
         ha="center", fontsize=14, color=TEXT, fontweight="bold")
fig.text(0.5, 0.935, f"n = {len(features):,} row-observations | {n_ts:,} timestamps | "
         f"{features.index.get_level_values('datetime').min().date()} "
         f"→ {features.index.get_level_values('datetime').max().date()}",
         ha="center", fontsize=9, color=SUB)

n_feat = len(corr_df)
y_pos  = np.arange(n_feat)

ax1 = fig.add_subplot(gs[0, 0])
colors = [GREEN if v > 0 else RED for v in corr_df["pearson_r"]]
ax1.barh(y_pos, corr_df["pearson_r"], color=colors, alpha=0.85)
ax1.set_yticks(y_pos); ax1.set_yticklabels(corr_df["feature"], fontsize=7)
ax1.axvline(0, color=BORDER, lw=0.8)
for i, (sig, val) in enumerate(zip(corr_df["pearson_p"] < 0.05, corr_df["pearson_r"])):
    if sig:
        ax1.text(val + np.sign(val) * 0.0001, i, " *",
                 ha="left" if val >= 0 else "right", va="center", color=YELLOW, fontsize=8)
style(ax1, title="Pearson r  (* = p<0.05)", xlabel="Correlation")

ax2 = fig.add_subplot(gs[0, 1])
colors = [GREEN if v > 0 else RED for v in corr_df["spearman_r"]]
ax2.barh(y_pos, corr_df["spearman_r"], color=colors, alpha=0.85)
ax2.set_yticks(y_pos); ax2.set_yticklabels(corr_df["feature"], fontsize=7)
ax2.axvline(0, color=BORDER, lw=0.8)
style(ax2, title="Spearman r (rank correlation)", xlabel="Correlation")

ic_sorted = corr_df.sort_values("ic_mean", key=abs, ascending=False)
ax3 = fig.add_subplot(gs[1, 0])
colors = [GREEN if v > 0 else RED for v in ic_sorted["ic_mean"]]
ax3.barh(np.arange(len(ic_sorted)), ic_sorted["ic_mean"], color=colors, alpha=0.85)
ax3.set_yticks(np.arange(len(ic_sorted))); ax3.set_yticklabels(ic_sorted["feature"], fontsize=7)
ax3.axvline(0, color=BORDER, lw=0.8)
style(ax3, title="IC Mean (avg cross-sectional Spearman per bar)", xlabel="IC")

icir_sorted = corr_df.sort_values("icir", key=abs, ascending=False)
ax4 = fig.add_subplot(gs[1, 1])
colors = [GREEN if v > 0 else RED for v in icir_sorted["icir"]]
ax4.barh(np.arange(len(icir_sorted)), icir_sorted["icir"], color=colors, alpha=0.85)
ax4.set_yticks(np.arange(len(icir_sorted))); ax4.set_yticklabels(icir_sorted["feature"], fontsize=7)
ax4.axvline(0, color=BORDER, lw=0.8)
ax4.axvline( 0.5, color=YELLOW, lw=0.8, linestyle="--", alpha=0.6)
ax4.axvline(-0.5, color=YELLOW, lw=0.8, linestyle="--", alpha=0.6)
style(ax4, title="ICIR = IC Mean / IC Std  (±0.5 threshold dashed)", xlabel="ICIR")

pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)
print("Page 1 saved.")


# ── PAGE 2: Year-by-year IC heatmap ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.97, "Year-by-Year IC per Feature",
         ha="center", fontsize=14, color=TEXT, fontweight="bold")
ax = fig.add_subplot(111)

feat_order = corr_df.sort_values("ic_mean", key=abs, ascending=False)["feature"].tolist()
heat = yearly_ic[feat_order].T   # rows=features, cols=years
vmax = max(abs(heat.values[~np.isnan(heat.values)]).max(), 0.005)

im = ax.imshow(heat.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(yearly_ic.index)))
ax.set_xticklabels(yearly_ic.index.astype(str), fontsize=10, color=TEXT)
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order, fontsize=8)
for r in range(heat.shape[0]):
    for c in range(heat.shape[1]):
        val = heat.values[r, c]
        if not np.isnan(val):
            txt_col = "black" if abs(val) < vmax * 0.65 else "white"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
style(ax, title="IC by Calendar Year  (green = feature predicted correctly, red = anti-predicted)")

pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)
print("Page 2 saved.")


# ── PAGE 3: Feature-feature correlation matrix ────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.97, "Feature-Feature Correlation Matrix (redundancy map)",
         ha="center", fontsize=14, color=TEXT, fontweight="bold")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)

fc = feat_corr.loc[feat_order, feat_order]
im = ax.imshow(fc.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(feat_order)))
ax.set_xticklabels(feat_order, rotation=90, fontsize=7)
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
style(ax, title="|r| > 0.8 = highly redundant pair → candidate for removal")

# Highlight redundant pairs
n = len(feat_order)
for r in range(n):
    for c in range(r+1, n):
        if abs(fc.values[r, c]) > 0.8:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                         fill=False, edgecolor=YELLOW, lw=2))

pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)
print("Page 3 saved.")


# ── PAGE 4: Rolling IC over time for top 10 features ─────────────────────────
top10 = corr_df.sort_values("ic_mean", key=abs, ascending=False)["feature"].head(10).tolist()
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.995, "Rolling IC over Time — Top 10 Features by |IC Mean|  (60-bar window)",
         ha="center", fontsize=13, color=TEXT, fontweight="bold")
fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.97, bottom=0.03)

for ax, feat in zip(axes.flatten(), top10):
    ts_ic   = ic_df[feat].dropna()
    roll    = ts_ic.rolling(60, min_periods=10).mean()
    overall = corr_df.loc[corr_df["feature"] == feat, "ic_mean"].values[0]
    ic_v    = corr_df.loc[corr_df["feature"] == feat, "icir"].values[0]

    ax.plot(ts_ic.index, ts_ic.values, color=BLUE, alpha=0.18, lw=0.5)
    ax.plot(roll.index,  roll.values,  color=MAUVE, lw=1.5, label="Roll-60")
    ax.axhline(0,       color=BORDER, lw=0.8)
    ax.axhline(overall, color=YELLOW, lw=1.0, linestyle="--",
               label=f"Mean {overall:+.4f}  ICIR {ic_v:+.3f}")
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_title(feat, color=TEXT, fontsize=9, pad=4)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)
print("Page 4 saved.")


pdf.close()
print(f"\nReport saved : {pdf_path}")
print(f"CSV    saved : {csv_path}")

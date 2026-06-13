import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT_FORMAT = "png"  # change to "pdf" for a single PDF file
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from features.engineer import load_universe, build_features
from models.predictor import WalkForwardModel
from strategy.signals import generate_signals
from backtest.backtester import run, trade_log, trade_summary
from data.store import load_etf_universe
from data.universe import get_sector_map

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#1e1e2e"
PANEL  = "#313244"
TEXT   = "#cdd6f4"
SUB    = "#a6adc8"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
BLUE   = "#89b4fa"
YELLOW = "#f9e2af"
MAUVE  = "#cba6f7"
BORDER = "#45475a"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUB,
    "ytick.color":       SUB,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.alpha":        0.4,
    "font.family":       "monospace",
})

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("Loading data...")
data_dict    = load_universe()
etf_dict     = load_etf_universe()
sector_map   = get_sector_map()
features_df  = build_features(data_dict, etf_dict=etf_dict, sector_map=sector_map)

print("Training model...")
model  = WalkForwardModel(n_splits=4)
proba  = model.fit_predict(features_df)

print("Generating signals...")
signals = generate_signals(proba, features_df, top_n=20, bottom_n=20)
results = run(signals, holding_period=4, cost_bps=5.0)
log     = trade_log(signals, holding_period=4, cost_bps=5.0, sector_map=sector_map)
summary = trade_summary(log)

eq        = results["equity_curve"]
eq.index  = pd.to_datetime(eq.index)
log["entry_time"] = pd.to_datetime(log["entry_time"])

# Per-period portfolio returns (for distribution + rolling sharpe)
period_rets = eq.pct_change().dropna()

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if title:  ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    ax.grid(True, alpha=0.3)


out_path = os.path.join(ROOT, "trade_report.pdf")
with PdfPages(out_path) as pdf:

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Performance Overview
    # ═════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35,
                             top=0.92, bottom=0.06, left=0.08, right=0.95)

    # -- Header stats --------------------------------------------------------
    stats = [
        ("Total Return",  f"{results['total_return']*100:.2f}%"),
        ("Sharpe Ratio",  f"{results['sharpe']:.2f}"),
        ("Max Drawdown",  f"{results['max_drawdown']*100:.2f}%"),
        ("Win Rate",      f"{results['win_rate']*100:.1f}%"),
        ("N Periods",     str(results['n_periods'])),
        ("N Trades",      f"{len(log):,}"),
        ("Stocks Traded", str(log['symbol'].nunique())),
        ("Cost/Trade",    f"{results['cost_bps']} bps"),
    ]
    fig.text(0.5, 0.96, "Backtest Trade Report", ha="center", va="top",
             fontsize=18, color=TEXT, fontweight="bold")
    fig.text(0.5, 0.935, f"{eq.index[0].date()}  →  {eq.index[-1].date()}  |  "
             f"Holding period: {results['holding_period']}h", ha="center",
             fontsize=10, color=SUB)

    x_pos = [0.08 + i * 0.117 for i in range(8)]
    for (label, val), xp in zip(stats, x_pos):
        color = GREEN if "Return" in label and results['total_return'] > 0 else \
                RED   if "Drawdown" in label else YELLOW
        fig.text(xp, 0.905, val,   fontsize=13, color=color, fontweight="bold")
        fig.text(xp, 0.893, label, fontsize=7,  color=SUB)

    # -- Equity curve --------------------------------------------------------
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.plot(eq.index, eq.values, color=BLUE, lw=1.5, label="Strategy")
    ax_eq.fill_between(eq.index, 1, eq.values,
                        where=eq.values >= 1, alpha=0.15, color=GREEN)
    ax_eq.fill_between(eq.index, 1, eq.values,
                        where=eq.values < 1,  alpha=0.15, color=RED)
    ax_eq.axhline(1, color=BORDER, lw=0.8, linestyle="--")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{(y-1)*100:.0f}%"))
    style(ax_eq, title="Equity Curve (cumulative return)", ylabel="Return")
    ax_eq.legend(fontsize=8)

    # Drawdown below
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.35, color=RED)
    ax_dd.set_ylim(-1, 0.1)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax_dd.tick_params(colors=RED, labelsize=7)
    ax_dd.set_ylabel("Drawdown", color=RED, fontsize=8)

    # -- Monthly returns heatmap ---------------------------------------------
    ax_heat = fig.add_subplot(gs[1, :])
    monthly = eq.resample("ME").last().pct_change().dropna()
    monthly_df = pd.DataFrame({
        "ret":   monthly.values,
        "month": monthly.index.month,
        "year":  monthly.index.year,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns = [month_names[m] for m in pivot.columns]
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax_heat.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(range(len(pivot.columns))); ax_heat.set_xticklabels(pivot.columns, fontsize=8)
    ax_heat.set_yticks(range(len(pivot))); ax_heat.set_yticklabels(pivot.index, fontsize=8)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax_heat.text(c, r, f"{val*100:.1f}%", ha="center", va="center",
                             fontsize=7, color="black" if abs(val) < vmax*0.6 else "white")
    plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02)
    style(ax_heat, title="Monthly Returns Heatmap")

    # -- Return distribution -------------------------------------------------
    ax_dist = fig.add_subplot(gs[2, 0])
    bins = np.linspace(period_rets.quantile(0.01), period_rets.quantile(0.99), 50)
    ax_dist.hist(period_rets[period_rets >= 0], bins=bins, color=GREEN, alpha=0.7, label="Win")
    ax_dist.hist(period_rets[period_rets <  0], bins=bins, color=RED,   alpha=0.7, label="Loss")
    ax_dist.axvline(period_rets.mean(), color=YELLOW, lw=1.5, linestyle="--",
                    label=f"Mean {period_rets.mean()*100:.3f}%")
    ax_dist.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))
    style(ax_dist, title="Per-Period Return Distribution", xlabel="Return")
    ax_dist.legend(fontsize=7)

    # -- Rolling Sharpe -------------------------------------------------------
    ax_rs = fig.add_subplot(gs[2, 1])
    window = 60
    roll_sharpe = (period_rets.rolling(window).mean() /
                   period_rets.rolling(window).std() * np.sqrt(252 * 6.5 / 4))
    ax_rs.plot(roll_sharpe.index, roll_sharpe.values, color=MAUVE, lw=1.2)
    ax_rs.axhline(0, color=BORDER, lw=0.8)
    ax_rs.axhline(results["sharpe"], color=YELLOW, lw=1, linestyle="--",
                  label=f"Overall {results['sharpe']:.2f}")
    style(ax_rs, title=f"Rolling Sharpe ({window}-period window)")
    ax_rs.legend(fontsize=7)

    # -- Trade count by month -------------------------------------------------
    ax_tc = fig.add_subplot(gs[3, 0])
    tc = log.set_index("entry_time").resample("ME")["symbol"].count()
    ax_tc.bar(range(len(tc)), tc.values, color=BLUE, alpha=0.8)
    ax_tc.set_xticks(range(len(tc)))
    ax_tc.set_xticklabels([str(d.date()) for d in tc.index], rotation=45, ha="right", fontsize=6)
    style(ax_tc, title="Trade Count by Month", ylabel="# Trades")

    # -- Win rate over time ---------------------------------------------------
    ax_wr = fig.add_subplot(gs[3, 1])
    wr = log.set_index("entry_time").resample("ME")["win"].mean() * 100
    bar_colors = [GREEN if v >= 50 else RED for v in wr.values]
    ax_wr.bar(range(len(wr)), wr.values, color=bar_colors, alpha=0.8)
    ax_wr.axhline(50, color=YELLOW, lw=1, linestyle="--")
    ax_wr.set_xticks(range(len(wr)))
    ax_wr.set_xticklabels([str(d.date()) for d in wr.index], rotation=45, ha="right", fontsize=6)
    ax_wr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    style(ax_wr, title="Monthly Win Rate", ylabel="Win Rate")

    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Page 1 done.")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Stock & Sector Analysis
    # ═════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                             top=0.94, bottom=0.06, left=0.12, right=0.95)
    fig.text(0.5, 0.97, "Stock & Sector Breakdown", ha="center",
             fontsize=15, color=TEXT, fontweight="bold")

    # -- Top 20 stocks by total PnL ------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    top20  = summary.head(20)
    colors = [GREEN if v >= 0 else RED for v in top20["total_pnl_pct"]]
    bars = ax_top.barh(top20["symbol"], top20["total_pnl_pct"], color=colors, alpha=0.85)
    for bar, row in zip(bars, top20.itertuples()):
        label = f" n={row.n_trades}  wr={row.win_rate:.0f}%  avg={row.avg_return_pct:.3f}%"
        ax_top.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=7, color=SUB)
    style(ax_top, title="Top 20 Stocks by Total PnL Contribution", xlabel="Total PnL (%)")

    # -- Bottom 15 stocks by total PnL ----------------------------------------
    ax_bot = fig.add_subplot(gs[1, :])
    bot15  = summary.tail(15).sort_values("total_pnl_pct")
    colors = [RED if v < 0 else GREEN for v in bot15["total_pnl_pct"]]
    bars = ax_bot.barh(bot15["symbol"], bot15["total_pnl_pct"], color=colors, alpha=0.85)
    for bar, row in zip(bars, bot15.itertuples()):
        label = f" n={row.n_trades}  wr={row.win_rate:.0f}%  avg={row.avg_return_pct:.3f}%"
        ax_bot.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                    label, va="center", ha="right", fontsize=7, color=SUB)
    style(ax_bot, title="Bottom 15 Stocks by Total PnL Contribution", xlabel="Total PnL (%)")

    # -- Sector PnL breakdown ------------------------------------------------
    ax_sec = fig.add_subplot(gs[2, 0])
    sec_pnl = summary.groupby("sector")["total_pnl_pct"].sum().sort_values()
    sec_colors = [GREEN if v >= 0 else RED for v in sec_pnl.values]
    ax_sec.barh(sec_pnl.index, sec_pnl.values, color=sec_colors, alpha=0.85)
    style(ax_sec, title="Total PnL by Sector", xlabel="Total PnL (%)")

    # -- Sector win rate -------------------------------------------------------
    ax_swr = fig.add_subplot(gs[2, 1])
    sec_wr  = log.groupby("sector")["win"].mean().sort_values() * 100
    bar_colors = [GREEN if v >= 50 else RED for v in sec_wr.values]
    ax_swr.barh(sec_wr.index, sec_wr.values, color=bar_colors, alpha=0.85)
    ax_swr.axvline(50, color=YELLOW, lw=1, linestyle="--")
    ax_swr.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style(ax_swr, title="Win Rate by Sector", xlabel="Win Rate")

    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Page 2 done.")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Trade Patterns
    # ═════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                             top=0.94, bottom=0.06, left=0.08, right=0.95)
    fig.text(0.5, 0.97, "Trade Patterns", ha="center",
             fontsize=15, color=TEXT, fontweight="bold")

    # -- Avg return by hour --------------------------------------------------
    ax_hr = fig.add_subplot(gs[0, 0])
    hr = log.copy()
    hr["hour_et"] = (hr["entry_time"].dt.hour - 4) % 24
    by_hour = hr.groupby("hour_et")["net_return"].mean() * 100
    bar_colors = [GREEN if v >= 0 else RED for v in by_hour.values]
    ax_hr.bar(by_hour.index, by_hour.values, color=bar_colors, alpha=0.85)
    ax_hr.axhline(0, color=BORDER, lw=0.8)
    ax_hr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}%"))
    style(ax_hr, title="Avg Net Return by Hour (ET)", xlabel="Hour (ET)", ylabel="Avg Return")

    # -- Avg return by day of week -------------------------------------------
    ax_dw = fig.add_subplot(gs[0, 1])
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    by_dow = log.groupby(log["entry_time"].dt.dayofweek)["net_return"].mean() * 100
    by_dow.index = [dow_names.get(i, str(i)) for i in by_dow.index]
    bar_colors = [GREEN if v >= 0 else RED for v in by_dow.values]
    ax_dw.bar(by_dow.index, by_dow.values, color=bar_colors, alpha=0.85)
    ax_dw.axhline(0, color=BORDER, lw=0.8)
    ax_dw.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}%"))
    style(ax_dw, title="Avg Net Return by Day of Week", ylabel="Avg Return")

    # -- Proba vs actual return scatter --------------------------------------
    ax_sc = fig.add_subplot(gs[1, :])
    sample = log.sample(min(5000, len(log)), random_state=42)
    ax_sc.scatter(sample["proba"], sample["return"] * 100,
                  alpha=0.15, s=8,
                  c=[GREEN if r > 0 else RED for r in sample["return"]])
    ax_sc.axhline(0, color=BORDER, lw=0.8)
    ax_sc.axvline(0.5, color=YELLOW, lw=0.8, linestyle="--")
    # Rolling mean line
    sample_sorted = sample.sort_values("proba")
    roll_mean = sample_sorted["return"].rolling(200).mean() * 100
    ax_sc.plot(sample_sorted["proba"], roll_mean, color=BLUE, lw=2, label="Rolling mean")
    ax_sc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    style(ax_sc, title="Model Probability vs Actual 4h Return (sample of 5,000 trades)",
          xlabel="Model Probability", ylabel="Actual Return")
    ax_sc.legend(fontsize=8)

    # -- Return by rank -------------------------------------------------------
    ax_rk = fig.add_subplot(gs[2, 0])
    by_rank = log.groupby("rank")["net_return"].mean() * 100
    ax_rk.bar(by_rank.index, by_rank.values,
              color=[GREEN if v >= 0 else RED for v in by_rank.values], alpha=0.85)
    ax_rk.axhline(0, color=BORDER, lw=0.8)
    ax_rk.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}%"))
    style(ax_rk, title="Avg Net Return by Rank (1=top pick)", xlabel="Rank", ylabel="Avg Return")

    # -- Cumulative PnL: top 5 vs bottom 5 stocks ----------------------------
    ax_cs = fig.add_subplot(gs[2, 1])
    top5 = summary.head(5)["symbol"].tolist()
    bot5 = summary.tail(5)["symbol"].tolist()
    for sym in top5:
        sym_log = log[log["symbol"] == sym].sort_values("entry_time")
        cum = (1 + sym_log["net_return"]).cumprod()
        ax_cs.plot(range(len(cum)), (cum - 1) * 100, lw=1.2, label=sym, alpha=0.85)
    for sym in bot5:
        sym_log = log[log["symbol"] == sym].sort_values("entry_time")
        cum = (1 + sym_log["net_return"]).cumprod()
        ax_cs.plot(range(len(cum)), (cum - 1) * 100, lw=1, linestyle="--", alpha=0.6, label=sym)
    ax_cs.axhline(0, color=BORDER, lw=0.8)
    ax_cs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    style(ax_cs, title="Cumulative PnL: Top 5 (solid) vs Bottom 5 (dashed)", ylabel="Cum Return")
    ax_cs.legend(fontsize=6, ncol=2)

    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Page 3 done.")

print(f"\nReport saved: {out_path}")

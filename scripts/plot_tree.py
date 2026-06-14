import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import lightgbm as lgb

from features.engineer import load_universe, build_features
from data.store import load_etf_universe
from data.universe import get_sector_map
from data.macro_fetcher import load_all_daily

# ── Train a small interpretable model ────────────────────────────────────────
print("Loading data (20-stock sample for speed)...")
data_dict = load_universe()
data_dict = dict(list(data_dict.items())[:20])
etf_dict = load_etf_universe()
sector_map = get_sector_map()
daily_feats = load_all_daily(list(data_dict.keys()), "2021-01-01", "2026-12-31")
features_df = build_features(data_dict, etf_dict=etf_dict, sector_map=sector_map,
                             daily_features=daily_feats)

feature_cols = [c for c in features_df.columns if c not in {"y", "forward_return"}]
X = features_df[feature_cols].dropna()
y = features_df.loc[X.index, "y"]

TREE_INDEX = 199  # 0-indexed, so this is tree #200

print(f"Training full model (200 trees) to extract tree #{TREE_INDEX + 1}...")
model = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
)
model.fit(X, y)

# ── Extract tree structure ────────────────────────────────────────────────────
tree_json = model.booster_.dump_model()["tree_info"][TREE_INDEX]["tree_structure"]


# ── Recursive layout ─────────────────────────────────────────────────────────
def layout(node, depth=0, x=0.0, x_min=0.0, x_max=1.0):
    """Assign (x, y, label, children) to every node. Returns list of node dicts."""
    nodes = []
    cx = (x_min + x_max) / 2
    cy = -depth

    if "split_feature" in node:
        fname = feature_cols[node["split_feature"]]
        threshold = node["threshold"]
        gain = node.get("split_gain", 0)
        label = f"{fname}\n≤ {threshold:.3f}\ngain={gain:.1f}"
        is_leaf = False
    else:
        val = node.get("leaf_value", 0)
        prob = 1 / (1 + np.exp(-val))  # sigmoid
        label = f"p={prob:.3f}"
        is_leaf = True

    entry = {"x": cx, "y": cy, "label": label, "is_leaf": is_leaf, "children": []}
    nodes.append(entry)

    if not is_leaf:
        mid = (x_min + x_max) / 2
        left_nodes = layout(node["left_child"], depth + 1, cx, x_min, mid)
        right_nodes = layout(node["right_child"], depth + 1, cx, mid, x_max)
        entry["children"] = [left_nodes[0], right_nodes[0]]
        nodes.extend(left_nodes)
        nodes.extend(right_nodes)

    return nodes


nodes = layout(tree_json)


# ── Draw ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 10))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-4.8, 0.6)
ax.axis("off")
fig.patch.set_facecolor("#1e1e2e")
ax.set_facecolor("#1e1e2e")

SPLIT_COLOR = "#313244"
LEAF_COLOR = "#45475a"
SPLIT_TEXT = "#cdd6f4"
LEAF_HIGH = "#a6e3a1"
LEAF_LOW = "#f38ba8"
EDGE_COLOR = "#585b70"


def draw_node(node):
    bw, bh = 0.09, 0.35
    x, y = node["x"], node["y"]

    if node["is_leaf"]:
        prob = float(node["label"].split("=")[1])
        color = LEAF_HIGH if prob > 0.5 else LEAF_LOW
        rect = mpatches.FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="#cdd6f4", linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x, y, node["label"], ha="center", va="center",
                fontsize=8, color="#1e1e2e", fontweight="bold", family="monospace")
    else:
        rect = mpatches.FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.01", facecolor=SPLIT_COLOR, edgecolor="#89b4fa", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, node["label"], ha="center", va="center",
                fontsize=7.5, color=SPLIT_TEXT, family="monospace")

    for i, child in enumerate(node["children"]):
        ax.annotate(
            "", xy=(child["x"], child["y"] + bh / 2),
            xytext=(x, y - bh / 2),
            arrowprops=dict(arrowstyle="-|>", color=EDGE_COLOR, lw=1.2)
        )
        label = "Yes" if i == 0 else "No"
        mx = (x + child["x"]) / 2
        my = (y - bh / 2 + child["y"] + bh / 2) / 2
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7, color="#a6adc8",
                bbox=dict(facecolor="#1e1e2e", edgecolor="none", pad=1))
        draw_node(child)


draw_node(nodes[0])

ax.set_title(
    f"LightGBM — Tree #{TREE_INDEX + 1} of 200  (the final correction tree)\n"
    "Green leaf = predict outperform  |  Red leaf = predict underperform",
    color="#cdd6f4", fontsize=11, pad=12
)

out_path = os.path.join(ROOT, "outputs", "tree_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")

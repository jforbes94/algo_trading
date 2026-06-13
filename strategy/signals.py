import numpy as np
import pandas as pd


def generate_signals(proba, features_df, top_n=20, bottom_n=20):
    df = pd.DataFrame({"proba": proba})
    df["forward_return"] = features_df["forward_return"]

    def _rank_group(group):
        n = len(group)
        group = group.sort_values("proba", ascending=False).copy()
        group["rank"] = range(1, n + 1)
        group["signal"] = 0
        group.loc[group["rank"] <= top_n, "signal"] = 1
        group.loc[group["rank"] > n - bottom_n, "signal"] = -1
        return group

    result = df.groupby(level=0, group_keys=False).apply(_rank_group)
    result["signal"] = result["signal"].astype(int)
    result["rank"] = result["rank"].astype(int)
    return result[["signal", "rank", "proba", "forward_return"]]


def summarize(signals):
    timestamps = signals.index.get_level_values(0).nunique()
    avg_long = signals[signals["signal"] == 1].groupby(level=0).size().mean()
    total = len(signals)
    pct_long = (signals["signal"] == 1).sum() / total * 100
    pct_neutral = (signals["signal"] == 0).sum() / total * 100
    pct_short = (signals["signal"] == -1).sum() / total * 100

    print(f"Total timestamps: {timestamps}")
    print(f"Avg stocks per timestamp with signal==1: {avg_long:.2f}")
    print(f"Signal distribution: {pct_long:.1f}% long, {pct_neutral:.1f}% neutral, {pct_short:.1f}% short")

    if "forward_return" in signals.columns:
        avg_ret_long = signals[signals["signal"] == 1]["forward_return"].mean()
        avg_ret_short = signals[signals["signal"] == -1]["forward_return"].mean()
        print(f"Avg forward_return signal==1: {avg_ret_long:.6f}")
        print(f"Avg forward_return signal==-1: {avg_ret_short:.6f}")


if __name__ == "__main__":
    np.random.seed(42)
    n_symbols = 30
    n_timestamps = 200
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    timestamps = pd.date_range("2023-01-01", periods=n_timestamps, freq="h")

    idx = pd.MultiIndex.from_product([timestamps, symbols], names=["datetime", "symbol"])
    proba = pd.Series(np.random.rand(len(idx)), index=idx, name="proba")
    features_df = pd.DataFrame(
        {"forward_return": np.random.randn(len(idx)) * 0.01},
        index=idx,
    )

    signals = generate_signals(proba, features_df, top_n=5, bottom_n=5)
    summarize(signals)
    print(signals.head(10))

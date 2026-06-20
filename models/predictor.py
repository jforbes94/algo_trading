import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor


class WalkForwardModel:
    def __init__(self, n_splits=6):
        self.n_splits = n_splits
        self._fold_importances = []
        self._feature_names = []

    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        drop_cols = {"y", "forward_return"}
        self._feature_names = [c for c in features_df.columns if c not in drop_cols]

        unique_dates = sorted(features_df.index.get_level_values("datetime").unique())
        chunks = np.array_split(unique_dates, self.n_splits + 1)

        params = {
            "objective":        "regression",
            "metric":           "rmse",
            "n_estimators":     300,
            "learning_rate":    0.04,
            "num_leaves":       31,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "reg_lambda":       1.0,
            "verbose":          -1,
        }

        all_preds = []
        self._fold_importances = []

        for k in range(self.n_splits):
            train_dates = set(np.concatenate(chunks[: k + 1]))
            test_dates  = set(chunks[k + 1])

            train_mask = features_df.index.get_level_values("datetime").isin(train_dates)
            test_mask  = features_df.index.get_level_values("datetime").isin(test_dates)

            train_df = features_df.loc[train_mask]
            test_df  = features_df.loc[test_mask]

            X_train = train_df[self._feature_names]
            y_train = train_df["y"]
            X_test  = test_df[self._feature_names]
            y_test  = test_df["y"]

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)

            scores = model.predict(X_test)

            # Mean IC: per-timestamp Spearman rank correlation (predicted score vs true quintile)
            ts_index = test_df.index.get_level_values("datetime")
            ics = []
            for ts in pd.unique(ts_index):
                mask = ts_index == ts
                if mask.sum() >= 5:
                    ic, _ = spearmanr(scores[mask], y_test.values[mask])
                    if not np.isnan(ic):
                        ics.append(ic)
            mean_ic = float(np.mean(ics)) if ics else 0.0
            print(f"Fold {k + 1} IC: {mean_ic:.4f}  (n_timestamps={len(ics)})")

            fold_preds = pd.Series(scores, index=test_df.index, name="proba")
            all_preds.append(fold_preds)

            importance = pd.Series(model.feature_importances_, index=self._feature_names)
            self._fold_importances.append(importance)

        return pd.concat(all_preds).rename("proba")

    def feature_importance(self) -> pd.Series:
        combined = pd.concat(self._fold_importances, axis=1).mean(axis=1)
        return combined.sort_values(ascending=False)


if __name__ == "__main__":
    np.random.seed(42)
    n_symbols   = 20
    n_timestamps = 500
    n_features  = 14

    timestamps = pd.date_range("2020-01-01", periods=n_timestamps, freq="D")
    symbols    = [f"SYM{i:02d}" for i in range(n_symbols)]
    index = pd.MultiIndex.from_product([timestamps, symbols], names=["datetime", "symbol"])

    feature_cols = {f"feat_{i:02d}": np.random.randn(len(index)) for i in range(n_features)}
    feature_cols["y"]              = np.tile(np.arange(n_symbols) % 5, n_timestamps).astype(float)
    feature_cols["forward_return"] = np.random.randn(len(index))

    features_df = pd.DataFrame(feature_cols, index=index)

    model = WalkForwardModel(n_splits=3)
    preds = model.fit_predict(features_df)

    print(f"\nOutput shape: {preds.shape}")
    print(f"Value range : [{preds.min():.4f}, {preds.max():.4f}]")
    print("\nTop 10 features:")
    print(model.feature_importance().head(10))

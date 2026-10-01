from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CohortData


@dataclass
class FittedPreprocessor:
    standardized: pd.DataFrame
    means: pd.Series
    standard_deviations: pd.Series
    fill_values: dict[str, float]
    feature_types: dict[str, str]

    def matrix(self, features: list[str]) -> np.ndarray:
        missing = sorted(set(features) - set(self.standardized.index))
        if missing:
            raise KeyError(f"Features absent from fitted preprocessing matrix: {missing[:5]}")
        return self.standardized.loc[features].T.to_numpy(np.float32).copy()

    def prs(self) -> np.ndarray:
        return self.standardized.loc["PRS"].to_numpy(np.float32).reshape(-1, 1).copy()


def fit_preprocessor(data: CohortData, teacher_features: list[str]) -> FittedPreprocessor:
    """Fit all imputation/scaling quantities on train80 only.

    NMR uses half the minimum observed training value. Proteins use the training
    median. All rows, including PRS, are z-standardized using train80 moments.
    """
    rows = list(dict.fromkeys([*teacher_features, "PRS"]))
    missing = sorted(set(rows) - set(data.raw_num.index))
    if missing:
        raise KeyError(f"Required raw features are missing: {missing[:10]}")
    train_ids = [data.samples[i] for i in data.idx_train]
    feature_types = {f: "NMR" for f in data.nmr_features}
    feature_types.update({f: "Protein" for f in data.protein_features})
    feature_types["PRS"] = "PRS"
    train_raw = data.raw_num.loc[rows, train_ids].copy()
    fill_values: dict[str, float] = {}
    for feature in rows:
        observed = pd.to_numeric(train_raw.loc[feature], errors="coerce").dropna()
        if observed.empty:
            value = 0.0
        elif feature_types.get(feature) == "NMR":
            value = float(observed.min() / 2.0)
        else:
            value = float(observed.median())
        fill_values[feature] = value
        train_raw.loc[feature] = train_raw.loc[feature].fillna(value)
    means = train_raw.mean(axis=1)
    stds = train_raw.std(axis=1).replace(0, 1).fillna(1)
    all_raw = data.raw_num.loc[rows, data.samples].copy()
    for feature, value in fill_values.items():
        all_raw.loc[feature] = all_raw.loc[feature].fillna(value)
    standardized = all_raw.sub(means, axis=0).div(stds, axis=0).fillna(0)
    return FittedPreprocessor(standardized, means, stds, fill_values, feature_types)

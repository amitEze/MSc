from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        feat_cfg = config.get("features", {})

        strategy = feat_cfg.get("numeric_imputation", "median")
        self.imputer = SimpleImputer(strategy=strategy)

        self.scale_numeric = bool(feat_cfg.get("scale_numeric", True))
        self.scaler = StandardScaler() if self.scale_numeric else None

        self.feature_cols_: Optional[list[str]] = None

    def _select_feature_cols(self, df: pd.DataFrame) -> list[str]:
        data_cfg = self.config.get("data", {})
        feat_cfg = self.config.get("features", {})

        member_id_col = data_cfg.get("member_id_col", "member_id")
        target_col = data_cfg.get("target_col", "y_churn")
        treatment_col = data_cfg.get("treatment_col", "t_outreach")
        drop_cols_cfg = feat_cfg.get("drop_cols", [])

        drop_cols = {member_id_col, target_col, treatment_col, *drop_cols_cfg}

        datetime_cols = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        drop_cols.update(datetime_cols)

        cols = [c for c in df.columns if c not in drop_cols]
        return cols

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        self.feature_cols_ = self._select_feature_cols(df)
        X = df[self.feature_cols_].to_numpy(dtype=float)

        X = self.imputer.fit_transform(X)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_cols_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first.")

        X = df[self.feature_cols_].to_numpy(dtype=float)
        X = self.imputer.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.feature_cols_ = self._select_feature_cols(df)
        X = df[self.feature_cols_].to_numpy(dtype=float)
        X = self.imputer.fit_transform(X)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        return X


def build_matrices(
    df: pd.DataFrame,
    config: Dict[str, Any],
    preprocessor: Optional[FeaturePreprocessor] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, FeaturePreprocessor]:
    data_cfg = config.get("data", {})
    member_id_col = data_cfg.get("member_id_col", "member_id")
    target_col = data_cfg.get("target_col", "y_churn")
    treatment_col = data_cfg.get("treatment_col", "t_outreach")

    member_ids = df[member_id_col].to_numpy()

    y = df[target_col].to_numpy() if target_col in df.columns else None
    t = df[treatment_col].to_numpy() if treatment_col in df.columns else None

    if preprocessor is None:
        preprocessor = FeaturePreprocessor(config)
        X = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)
    else:
        if fit:
            preprocessor.fit(df)
        X = preprocessor.transform(df)

    return X, y, t, member_ids, preprocessor

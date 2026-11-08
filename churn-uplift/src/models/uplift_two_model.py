from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.models.factory import build_model


class UpliftTwoModel:
    def __init__(self, model_type: str, hyperparams: Dict[str, Any]) -> None:
        self.model_type = model_type
        self.hyperparams = hyperparams
        self.model_treated = build_model(model_type, hyperparams)
        self.model_control = build_model(model_type, hyperparams)

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "UpliftTwoModel":
        mask = (~np.isnan(y)) & (~np.isnan(t))
        X = X[mask]
        y = y[mask].astype(int)
        t = t[mask].astype(int)

        X_t = X[t == 1]
        y_t = y[t == 1]

        X_c = X[t == 0]
        y_c = y[t == 0]

        self.model_treated.fit(X_t, y_t)
        self.model_control.fit(X_c, y_c)
        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p0, p1

    def predict_uplift(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p0, p1 = self.predict_proba(X)
        uplift = p0 - p1
        return p0, p1, uplift

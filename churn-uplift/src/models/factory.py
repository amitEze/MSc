from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(model_type: str, hyperparams: Dict[str, Any]):
    if model_type == "logistic_regression":
        return LogisticRegression(**hyperparams)

    if model_type == "xgboost":
        # sensible defaults; can be overridden from config
        defaults = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        merged = {**defaults, **hyperparams}
        return XGBClassifier(**merged)

    raise ValueError(f"Unknown model_type: {model_type}")

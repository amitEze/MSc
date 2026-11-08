# src/evaluation/uplift_metrics.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_uplift_metrics(
    y_true: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
    fractions: Optional[list[float]] = None,
    fixed_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Compute AUC, ATE and selection-based gain metrics.

    Returns a dict with keys:
      - "AUC": ROC AUC of `uplift_scores` predicting `y_true` (nan if undefined)
      - "ATE": average treatment effect (control_mean - treated_mean)
      - "fraction_k": chosen fraction (from `fractions`) that maximizes uplift_hat
      - "gain": uplift_hat * n (expected avoided churns at chosen fraction)
      - "gain/n": uplift_hat (per-member uplift in the top fraction)

    The chosen fraction follows the logic in `notebooks/fraction_selection.ipynb`:
    default fractions are np.linspace(0.05, 0.5, 10) and we pick the fraction
    with the highest `uplift_hat = p_control - p_treat` among fractions where both
    treated and control exist in the top segment.

    If `fixed_fraction` is provided, selection will be evaluated only at that
    single fraction (useful to compute gain on a held-out set using the
    validation-selected fraction).
    """
    mask = (~np.isnan(y_true)) & (~np.isnan(t))
    y = y_true[mask].astype(int)
    t = t[mask].astype(int)
    s = uplift_scores[mask]

    n_total = len(y)

    # AUC: only valid if both classes present
    try:
        auc = float(roc_auc_score(y, s)) if n_total > 0 and len(np.unique(y)) > 1 else np.nan
    except Exception:
        auc = np.nan

    # ATE: control_mean - treated_mean
    ate = np.nan
    if n_total > 0:
        mask_treated = t == 1
        mask_control = t == 0
        if mask_treated.sum() > 0 and mask_control.sum() > 0:
            ate = float(y[mask_control].mean() - y[mask_treated].mean())

    # fraction selection per notebook logic
    # If a fixed fraction is provided, evaluate only that fraction.
    if fixed_fraction is not None:
        fractions = [float(fixed_fraction)]
    elif fractions is None:
        fractions = np.linspace(0.05, 0.5, 10).tolist()

    best_row = None
    rows = []
    if n_total > 0:
        order_global = np.argsort(-s)
        for f in fractions:
            n = int(n_total * f)
            if n < 1:
                continue

            top_idx = order_global[:n]
            treated_idx = top_idx[t[top_idx] == 1]
            control_idx = top_idx[t[top_idx] == 0]

            if len(treated_idx) == 0 or len(control_idx) == 0:
                continue

            p_treat = float(y[treated_idx].mean())
            p_control = float(y[control_idx].mean())

            uplift_hat = p_control - p_treat
            gain = uplift_hat * n

            rows.append({
                "fraction": f,
                "n": n,
                "p_treat": p_treat,
                "p_control": p_control,
                "uplift_hat": uplift_hat,
                "gain": gain,
            })

        if len(rows) > 0:
            # pick fraction with max uplift_hat
            best = max(rows, key=lambda r: r["uplift_hat"])
            best_row = best

    if best_row is None:
        return {"AUC": auc, "ATE": ate, "fraction_k": np.nan, "gain": np.nan, "gain/n": np.nan, "uplift_mean_hat": float(uplift_scores.mean())}

    return {
        "AUC": auc,
        "ATE": ate,
        "fraction_k": float(best_row["fraction"]),
        "gain": float(best_row["gain"]),
        "gain/n": float(best_row["uplift_hat"]),
        "uplift_mean_hat": float(uplift_scores.mean()),
    }

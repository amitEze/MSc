from __future__ import annotations

from typing import Dict, List, Tuple

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def _clean_inputs(y: np.ndarray, t: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = (~np.isnan(y)) & (~np.isnan(t)) & (~np.isnan(s))
    y = y[mask].astype(int)
    t = t[mask].astype(int)
    s = s[mask].astype(float)
    idx = np.nonzero(mask)[0]
    return y, t, s, idx


def qini_curve(y: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute Qini curve data and coefficient.

    Uses unbiased individual uplift estimator z_i = y*(t/p - (1-t)/(1-p)).
    Returns dict with fractions, cumulative_gain, qini_auc and qini_coefficient.
    """
    y, t, s, _ = _clean_inputs(y, t, uplift_scores)
    n = len(y)
    if n == 0:
        return {"fractions": np.array([]), "cum_gain": np.array([]), "qini_auc": math.nan, "qini_coeff": math.nan}

    p = float(np.mean(t))
    # unbiased individual effect estimator
    z = y * (t / p - (1 - t) / (1 - p))

    order = np.argsort(-s)
    z_sorted = z[order]
    cum_gain = np.cumsum(z_sorted)

    fractions = np.arange(1, n + 1) / n

    # compute areas using trapezoidal rule
    model_auc = float(np.trapz(cum_gain, x=fractions))
    random_auc = float(np.trapz(fractions * cum_gain[-1], x=fractions))

    # perfect model: sort by true z
    perfect_order = np.argsort(-z)
    z_perfect_sorted = z[perfect_order]
    cum_perfect = np.cumsum(z_perfect_sorted)
    perfect_auc = float(np.trapz(cum_perfect, x=fractions))

    denom = perfect_auc - random_auc
    qini_coeff = float((model_auc - random_auc) / denom) if denom != 0 else math.nan

    return {"fractions": fractions, "cum_gain": cum_gain, "qini_auc": model_auc, "qini_coeff": qini_coeff}


def uplift_curve(y: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray) -> Dict[str, np.ndarray]:
    """Return uplift curve: fraction vs cumulative incremental churn prevented and per-person uplift."""
    y, t, s, _ = _clean_inputs(y, t, uplift_scores)
    n = len(y)
    if n == 0:
        return {"fractions": np.array([]), "cum_gain": np.array([]), "avg_uplift": np.array([])}

    p = float(np.mean(t))
    z = y * (t / p - (1 - t) / (1 - p))
    order = np.argsort(-s)
    z_sorted = z[order]
    cum_gain = np.cumsum(z_sorted)
    fractions = np.arange(1, n + 1) / n
    avg_uplift = cum_gain / np.arange(1, n + 1)

    return {"fractions": fractions, "cum_gain": cum_gain, "avg_uplift": avg_uplift}


def uplift_by_decile(y: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray) -> List[Dict[str, float]]:
    y, t, s, _ = _clean_inputs(y, t, uplift_scores)
    n = len(y)
    if n == 0:
        return []

    order = np.argsort(-s)
    deciles = []
    for i in range(10):
        start = int(i * n / 10)
        end = int((i + 1) * n / 10)
        idx = order[start:end]
        n_bin = len(idx)
        if n_bin == 0:
            deciles.append({"decile": i + 1, "n": 0, "treated_rate": math.nan, "control_rate": math.nan, "uplift": math.nan})
            continue
        treated_mask = t[idx] == 1
        control_mask = t[idx] == 0
        treated_rate = float(y[idx][treated_mask].mean()) if treated_mask.sum() > 0 else math.nan
        control_rate = float(y[idx][control_mask].mean()) if control_mask.sum() > 0 else math.nan
        uplift = control_rate - treated_rate if (not math.isnan(control_rate) and not math.isnan(treated_rate)) else math.nan
        deciles.append({"decile": i + 1, "n": n_bin, "treated_rate": treated_rate, "control_rate": control_rate, "uplift": uplift})
    return deciles


def arm_prediction_metrics(y: np.ndarray, t: np.ndarray, p_treat: np.ndarray, p_control: np.ndarray) -> Dict[str, object]:
    """Compute ROC and PR metrics for treated and control arms (sanity checks)."""
    metrics_out: Dict[str, object] = {}
    # treated arm: evaluate model_treated on those with t==1
    mask_t = (~np.isnan(y)) & (t == 1)
    if mask_t.sum() > 0:
        y_t = y[mask_t]
        p_t = p_treat[mask_t]
        try:
            fpr_t, tpr_t, _ = metrics.roc_curve(y_t, p_t)
            auc_t = float(metrics.roc_auc_score(y_t, p_t))
        except Exception:
            fpr_t = tpr_t = None
            auc_t = math.nan
        try:
            prec_t, rec_t, _ = metrics.precision_recall_curve(y_t, p_t)
            pr_auc_t = float(metrics.auc(rec_t, prec_t))
        except Exception:
            prec_t = rec_t = None
            pr_auc_t = math.nan
    else:
        fpr_t = tpr_t = prec_t = rec_t = None
        auc_t = pr_auc_t = math.nan

    # control arm
    mask_c = (~np.isnan(y)) & (t == 0)
    if mask_c.sum() > 0:
        y_c = y[mask_c]
        p_c = p_control[mask_c]
        try:
            fpr_c, tpr_c, _ = metrics.roc_curve(y_c, p_c)
            auc_c = float(metrics.roc_auc_score(y_c, p_c))
        except Exception:
            fpr_c = tpr_c = None
            auc_c = math.nan
        try:
            prec_c, rec_c, _ = metrics.precision_recall_curve(y_c, p_c)
            pr_auc_c = float(metrics.auc(rec_c, prec_c))
        except Exception:
            prec_c = rec_c = None
            pr_auc_c = math.nan
    else:
        fpr_c = tpr_c = prec_c = rec_c = None
        auc_c = pr_auc_c = math.nan

    metrics_out.update(
        {
            "roc_treated": {"fpr": fpr_t, "tpr": tpr_t, "auc": auc_t},
            "roc_control": {"fpr": fpr_c, "tpr": tpr_c, "auc": auc_c},
            "pr_treated": {"precision": prec_t, "recall": rec_t, "auc": pr_auc_t},
            "pr_control": {"precision": prec_c, "recall": rec_c, "auc": pr_auc_c},
        }
    )
    return metrics_out


def evaluate_uplift(
    y: np.ndarray,
    t: np.ndarray,
    p_control: np.ndarray,
    p_treat: np.ndarray,
    uplift_scores: np.ndarray,
    out_dir: str | Path,
    plots_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Compute uplift/Qini metrics, plots, and return structured metrics dict.

    Saves PNG plots to `out_dir/figures/<experiment_name>/` and returns a dict of metrics.
    """
    out_dir = Path(out_dir)
    # metrics and structured outputs remain in out_dir (reports under output/*)
    out_dir.mkdir(parents=True, exist_ok=True)

    # plots should be stored under `plots/*` by default unless plots_dir provided
    if plots_dir is None:
        plots_base = Path("plots")
    else:
        plots_base = Path(plots_dir)
    figs_dir = plots_base
    figs_dir.mkdir(parents=True, exist_ok=True)

    y_arr = np.array(y)
    t_arr = np.array(t)
    s_arr = np.array(uplift_scores)

    # Qini
    qini = qini_curve(y_arr, t_arr, s_arr)
    # deciles
    deciles = uplift_by_decile(y_arr, t_arr, s_arr)
    # arm metrics
    arm_metrics = arm_prediction_metrics(y_arr, t_arr, p_treat, p_control)

    # save plots (into plots/ by default)
    # Qini curve
    if qini["fractions"].size > 0:
        plt.figure()
        plt.plot(qini["fractions"], qini["cum_gain"], label="Model Qini")
        plt.plot(qini["fractions"], qini["fractions"] * qini["cum_gain"][-1], label="Random", linestyle="--")
        plt.xlabel("Fraction of population targeted")
        plt.ylabel("Cumulative incremental churn prevented")
        plt.title("Qini curve (uplift, outreach vs no outreach)")
        plt.legend()
        plt.grid()
        p = figs_dir / "qini_curve.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()

    # uplift curve and uplift gain plots removed — keep Qini and ROC/PR plots only

    # ROC/PR plots for arms
    try:
        plt.figure()
        if arm_metrics["roc_treated"]["fpr"] is not None:
            plt.plot(arm_metrics["roc_treated"]["fpr"], arm_metrics["roc_treated"]["tpr"], label=f"Treated (AUC={arm_metrics['roc_treated']['auc']:.3f})")
        if arm_metrics["roc_control"]["fpr"] is not None:
            plt.plot(arm_metrics["roc_control"]["fpr"], arm_metrics["roc_control"]["tpr"], label=f"Control (AUC={arm_metrics['roc_control']['auc']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC – treated vs control models (churn prediction)")
        plt.legend()
        plt.grid()
        p = figs_dir / "roc_treated_vs_control.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    try:
        plt.figure()
        if arm_metrics["pr_treated"]["precision"] is not None:
            plt.plot(arm_metrics["pr_treated"]["recall"], arm_metrics["pr_treated"]["precision"], label=f"Treated (AUC={arm_metrics['pr_treated']['auc']:.3f})")
        if arm_metrics["pr_control"]["precision"] is not None:
            plt.plot(arm_metrics["pr_control"]["recall"], arm_metrics["pr_control"]["precision"], label=f"Control (AUC={arm_metrics['pr_control']['auc']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall – treated vs control models (churn prediction)")
        plt.legend()
        plt.grid()
        p = figs_dir / "pr_treated_vs_control.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # structured metrics
    metrics_out = {
        "qini": {"qini_auc": qini.get("qini_auc"), "qini_coeff": qini.get("qini_coeff")},
        "arm_prediction": {
            "roc_treated_auc": arm_metrics["roc_treated"]["auc"],
            "roc_control_auc": arm_metrics["roc_control"]["auc"],
            "pr_treated_auc": arm_metrics["pr_treated"]["auc"],
            "pr_control_auc": arm_metrics["pr_control"]["auc"],
        },
    }



    return metrics_out

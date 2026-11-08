# src/pipelines/run_uplift_two_model.py
"""Two-model uplift pipeline runner.

Usage:
    python run_uplift_two_model.py --config config/uplift_logreg.yml

Outputs (under `output/uplift/<model-tag>/<experiment_name>`):
 - `test_scores.csv`: per-member predictions and uplift ranking
 - `test_metrics.txt`: simple uplift@k summary
 - `metrics.json`: structured uplift metrics (Qini, deciles, ROC/PR AUCs)
Plots are saved under `plots/uplift/<model-tag>/<experiment_name>/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.loader import load_config
from src.features.featurize import build_matrices, FeaturePreprocessor
from src.models.uplift_two_model import UpliftTwoModel
from src.evaluation.uplift_metrics import calculate_uplift_metrics
import matplotlib.pyplot as plt
from src.evaluation.uplift_eval import evaluate_uplift

def _model_tag(cfg: dict) -> str:
    mtype = cfg["model"]["type"]
    mapping = {
        "logistic_regression": "logreg",
        "xgboost": "xgb",
    }
    return mapping.get(mtype, mtype)

def split_train_val(df: pd.DataFrame, cfg: dict):
    data_cfg = cfg["data"]
    y_col = data_cfg["target_col"]
    t_col = data_cfg["treatment_col"]

    y = df[y_col]
    t = df[t_col]

    strat = (y.astype(int).astype(str) + "_" + t.astype(int).astype(str))
    strat = strat.replace("nan_nan", np.nan)

    # fallback: if strat has NaNs, just stratify by y
    if strat.isna().any():
        strat = y

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.get("training", {}).get("val_size", 0.2),
        random_state=cfg.get("seed", 42),
        stratify=strat,
    )
    return train_df, val_df

def train_and_validate(df: pd.DataFrame, cfg: dict, config_path: str):
    train_df, val_df = split_train_val(df, cfg)

    preproc = FeaturePreprocessor(cfg)

    X_train, y_train, t_train, _, preproc = build_matrices(
        train_df,
        cfg,
        preprocessor=preproc,
        fit=True,
    )
    X_val, y_val, t_val, member_ids_val, _ = build_matrices(
        val_df,
        cfg,
        preprocessor=preproc,
        fit=False,
    )

    model_cfg = cfg["model"]
    uplift_model = UpliftTwoModel(model_cfg["type"], model_cfg.get("hyperparams", {}))
    # Fit final uplift two-model
    uplift_model.fit(X_train, y_train, t_train)

    p0_val, p1_val, uplift_val = uplift_model.predict_uplift(X_val)

    metrics = calculate_uplift_metrics(y_val, t_val, uplift_val)

    print(f"Validation AUC: {metrics.get('AUC', float('nan')):.4f}")
    print(f"Validation ATE: {metrics.get('ATE', float('nan')):.4f}")
    print(
        f"Selected fraction_k: {metrics.get('fraction_k', float('nan'))}, gain: {metrics.get('gain', float('nan'))}")

    tag = _model_tag(cfg)
    out_dir = Path("output") / "uplift" / tag 
    out_dir.mkdir(parents=True, exist_ok=True)

    txt = (
        f"AUC: {metrics.get('AUC', float('nan')):.4f}\n"
        f"empirical_ATE(doesn't depend on the model): {metrics.get('ATE', float('nan')):.4f}\n"
        f"fraction_k: {metrics.get('fraction_k', float('nan'))}\n"
        f"gain: {metrics.get('gain', float('nan'))}\n"
        f"gain/n: {metrics.get('gain/n', float('nan'))}\n"
        f"model_based_ATE: {metrics.get('uplift_mean_hat', float('nan'))}\n"
    )
    (out_dir / f"val_metrics.txt").write_text(txt)

    member_id_col = cfg["data"].get("member_id_col", "member_id")
    treatment_col = cfg["data"].get("treatment_col", "t_outreach")
    target_col = cfg["data"].get("target_col", "y_churn")

    df_val = pd.DataFrame(
        {
            member_id_col: member_ids_val,
            "uplift_score": uplift_val,
            treatment_col: t_val,
            target_col: y_val,
        }
    )

    df_val["rank"] = df_val["uplift_score"].rank(method="first", ascending=False).astype(int)


    cols_out = [member_id_col, "uplift_score", "rank", treatment_col, target_col]
    df_val = df_val[cols_out]

    val_scores_path = out_dir / "val_scores.csv"
    df_val.to_csv(val_scores_path, index=False)
    print(f"Saved validation uplift scores to {val_scores_path}")

    return preproc, uplift_model, metrics.get("fraction_k")

def train_full_and_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    val_fraction: Optional[float] = None,
) -> Path:
    preproc = FeaturePreprocessor(cfg)

    X_train, y_train, t_train, _, preproc = build_matrices(
        train_df,
        cfg,
        preprocessor=preproc,
        fit=True,
    )
    X_test, y_test, t_test, member_ids_test, _ = build_matrices(
        test_df,
        cfg,
        preprocessor=preproc,
        fit=False,
    )

    model_cfg = cfg["model"]
    uplift_model = UpliftTwoModel(model_cfg["type"], model_cfg.get("hyperparams", {}))
    uplift_model.fit(X_train, y_train, t_train)

    p0_test, p1_test, uplift_test = uplift_model.predict_uplift(X_test)

    exp_name = cfg.get("experiment_name", "default_experiment")
    tag = _model_tag(cfg)
    out_dir = Path("output") / "uplift" / tag 
    out_dir.mkdir(parents=True, exist_ok=True)

    member_id_col = cfg["data"]["member_id_col"]
    df_out = pd.DataFrame(
        {
            member_id_col: member_ids_test,
            "p_churn_control": p0_test,
            "p_churn_treated": p1_test,
            "uplift_score": uplift_test,
        }
    )

    df_out["rank"] = df_out["uplift_score"].rank(method="first", ascending=False).astype(int)
    df_out = df_out.sort_values("rank")

    # compute metrics using the validation-selected fraction when available;
    # if val_fraction is not provided or invalid, fall back to config selection.
    metrics_test = None
    if y_test is not None and not np.isnan(y_test).all():
        metrics_test = calculate_uplift_metrics(
            y_test, t_test, uplift_test, fixed_fraction=val_fraction
        )

    chosen_fraction = None
    if metrics_test is not None:
        chosen_fraction = metrics_test.get("fraction_k")
    if chosen_fraction is None or (isinstance(chosen_fraction, float) and np.isnan(chosen_fraction)):
        chosen_fraction = cfg.get("uplift", {}).get("k_fraction_eval", 0.25)

    n_total = len(df_out)
    top_k = max(1, int(chosen_fraction * n_total))
    df_top = df_out.head(top_k).copy()

    member_id_col = cfg["data"]["member_id_col"]
    df_top_out = df_top[[member_id_col, "uplift_score", "rank"]].copy()
    df_top_out = df_top_out.rename(columns={"rank": "member_rank"})

    scores_path = out_dir / f"test_scores.csv"
    df_top_out.to_csv(scores_path, index=False)
    print(f"Saved top-{top_k} uplift scores to {scores_path}")

    if metrics_test is not None:
        txt = (
            f"AUC: {metrics_test.get('AUC', float('nan')):.4f}\n"
            f"empirical_ATE(doesn't depend on the model): {metrics_test.get('ATE', float('nan')):.4f}\n"
            f"fraction_k: {metrics_test.get('fraction_k', float('nan'))}\n"
            f"gain: {metrics_test.get('gain', float('nan'))}\n"
            f"gain/n: {metrics_test.get('gain/n', float('nan'))}\n"
            f"model_based_ATE: {metrics_test.get('uplift_mean_hat', float('nan'))}\n"
        )
        (out_dir / f"test_metrics.txt").write_text(txt)
        print(f"Saved test uplift metrics to {out_dir / f'test_metrics.txt'}")

        try:
            plots_base = Path("plots") / tag
            evaluate_uplift(
                y_test,
                t_test,
                p0_test,
                p1_test,
                uplift_test,
                out_dir=out_dir,
                plots_dir=plots_base,
            )
            print(f"Saved uplift plots to {plots_base / exp_name}")
        except Exception as e:  
            print(f"Could not compute extended uplift evaluation: {e}")

    return scores_path


def run_random_baseline_on_test(
    test_df: pd.DataFrame,
    cfg: dict,
    val_fraction: Optional[float] = None,
) -> Path:
    """Run a random baseline on the test set and save metrics/plots like the model.

    This generates random probabilities / uplift scores and writes the same
    outputs as `train_full_and_predict_test` but under the `random` model tag.
    """
    rng = np.random.RandomState(cfg.get("seed", 42))

    member_id_col = cfg["data"].get("member_id_col", "member_id")
    y_col = cfg["data"].get("target_col", "y_churn")
    t_col = cfg["data"].get("treatment_col", "t_outreach")

    # pull columns if available
    if member_id_col not in test_df.columns:
        raise KeyError(f"member id column '{member_id_col}' not found in test dataframe")

    member_ids = test_df[member_id_col].to_numpy()
    y_test = test_df[y_col].to_numpy() if y_col in test_df.columns else None
    t_test = test_df[t_col].to_numpy() if t_col in test_df.columns else None

    n = len(member_ids)

    uplift_test = rng.rand(n)

    exp_name = cfg.get("experiment_name", "default_experiment")
    tag = "random"
    out_dir = Path("output") / "uplift" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        {
            member_id_col: member_ids,
            "uplift_score": uplift_test,
        }
    )

    df_out["rank"] = df_out["uplift_score"].rank(method="first", ascending=False).astype(int)
    df_out = df_out.sort_values("rank")

    metrics_test = None
    if y_test is not None and not np.isnan(y_test).all():
        metrics_test = calculate_uplift_metrics(y_test, t_test, uplift_test, fixed_fraction=val_fraction)

    chosen_fraction = None
    if metrics_test is not None:
        chosen_fraction = metrics_test.get("fraction_k")
    if chosen_fraction is None or (isinstance(chosen_fraction, float) and np.isnan(chosen_fraction)):
        chosen_fraction = cfg.get("uplift", {}).get("k_fraction_eval", 0.25)

    n_total = len(df_out)
    top_k = max(1, int(chosen_fraction * n_total))
    df_top = df_out.head(top_k).copy()

    df_top_out = df_top[[member_id_col, "uplift_score", "rank"]].copy()
    df_top_out = df_top_out.rename(columns={"rank": "member_rank"})


    if metrics_test is not None:
        txt = (
            f"AUC: {metrics_test.get('AUC', float('nan')):.4f}\n"
            f"ATE: {metrics_test.get('ATE', float('nan')):.4f}\n"
            f"fraction_k: {metrics_test.get('fraction_k', float('nan'))}\n"
            f"gain: {metrics_test.get('gain', float('nan'))}\n"
            f"gain/n: {metrics_test.get('gain/n', float('nan'))}\n"
        )
        try:
            metrics_50 = calculate_uplift_metrics(y_test, t_test, uplift_test, fixed_fraction=0.5)
            gain_50 = metrics_50.get("gain", float("nan"))
            gain_n_50 = metrics_50.get("gain/n", float("nan"))
        except Exception:
            gain_50 = float("nan")
            gain_n_50 = float("nan")

        txt += (
            f"gain_at_0.5: {gain_50}\n"
            f"gain/n_at_0.5: {gain_n_50}\n"
        )
        (out_dir / f"test_metrics.txt").write_text(txt)
        print(f"Saved random baseline test uplift metrics to {out_dir / f'test_metrics.txt'}")

    return None


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    train_path = cfg["data"]["train_base_table"]
    test_path = cfg["data"]["test_base_table"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    print(f"Running experiment: {cfg.get('experiment_name', '')}")

    _, _, val_fraction = train_and_validate(train_df, cfg, config_path)
    train_full_and_predict_test(train_df, test_df, cfg, val_fraction)
    try:
        run_random_baseline_on_test(test_df, cfg, val_fraction)
    except Exception as e:
        print(f"Could not run random baseline: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-model uplift experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)

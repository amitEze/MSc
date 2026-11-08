from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json

from src.config.loader import load_config
from src.features.featurize import build_matrices, FeaturePreprocessor
from src.models.factory import build_model


def _model_tag(cfg: dict) -> str:
    mtype = cfg["model"]["type"]
    mapping = {
        "logistic_regression": "logreg",
        "xgboost": "xgb",
    }
    return mapping.get(mtype, mtype)


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def _flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # convert None to null-literal string for CSV readability
    return {k: ("null" if v is None else v) for k, v in params.items()}


def grid_search_config(config_path: str, n_splits: int = 5, random_state: int | None = 42):
    cfg = load_config(config_path)

    train_path = cfg["data"]["train_base_table"]
    train_df = pd.read_parquet(train_path)

    print(f"Loaded train: {train_df.shape}")

    # build X, y, t (preprocessor fitted on full train set so transforms are consistent)
    preproc = FeaturePreprocessor(cfg)
    X, y, t, member_ids, preproc = build_matrices(train_df, cfg, preprocessor=preproc, fit=True)

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type")
    search_grid = model_cfg.get("search_grid", {})

    combos = _expand_grid(search_grid)
    print(f"Running grid search for {len(combos)} hyperparameter combinations")

    tag = _model_tag(cfg)
    cfg_name = Path(config_path).stem
    out_dir = Path("output") / "grid_search" / tag / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{cfg_name}.csv"

    results = []

    # create splitter stratified by y when possible
    if y is None:
        raise RuntimeError("No target column found in training data for grid search.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for params in combos:
        flat = _flatten_params(params)

        # If config contains an "uplift" section we treat this as two-model uplift
        if "uplift" in cfg:
            # two-model uplift: fit separate models for treated (t==1) and control (t==0)
            t_arr = np.array(t)
            y_arr = np.array(y)

            aucs_t = []
            aucs_c = []

            for train_idx, val_idx in skf.split(X, y_arr):
                # treated
                train_t_mask = (t_arr[train_idx] == 1)
                val_t_mask = (t_arr[val_idx] == 1)

                train_c_mask = (t_arr[train_idx] == 0)
                val_c_mask = (t_arr[val_idx] == 0)

                # treated subset
                if train_t_mask.sum() > 0 and val_t_mask.sum() > 0 and len(np.unique(y_arr[val_idx][val_t_mask])) > 1:
                    try:
                        m_t = build_model(model_type, params)
                        m_t.fit(X[train_idx][train_t_mask], y_arr[train_idx][train_t_mask])
                        p_t = m_t.predict_proba(X[val_idx][val_t_mask])[:, 1]
                        aucs_t.append(roc_auc_score(y_arr[val_idx][val_t_mask], p_t))
                    except Exception as e:
                        print(f"Treated fold error: {e}")
                        aucs_t.append(np.nan)
                else:
                    aucs_t.append(np.nan)

                # control subset
                if train_c_mask.sum() > 0 and val_c_mask.sum() > 0 and len(np.unique(y_arr[val_idx][val_c_mask])) > 1:
                    try:
                        m_c = build_model(model_type, params)
                        m_c.fit(X[train_idx][train_c_mask], y_arr[train_idx][train_c_mask])
                        p_c = m_c.predict_proba(X[val_idx][val_c_mask])[:, 1]
                        aucs_c.append(roc_auc_score(y_arr[val_idx][val_c_mask], p_c))
                    except Exception as e:
                        print(f"Control fold error: {e}")
                        aucs_c.append(np.nan)
                else:
                    aucs_c.append(np.nan)

            mean_t = np.nanmean(aucs_t)
            std_t = np.nanstd(aucs_t)
            mean_c = np.nanmean(aucs_c)
            std_c = np.nanstd(aucs_c)

            combined_mean = np.nanmean([mean_t, mean_c]) if not (np.isnan(mean_t) and np.isnan(mean_c)) else np.nan

            res = {
                **flat,
                "mean_auc_treated": mean_t,
                "std_auc_treated": std_t,
                "mean_auc_control": mean_c,
                "std_auc_control": std_c,
                "combined_mean_auc": combined_mean,
            }
            results.append(res)
            
        else:
            raise ValueError(f"Unsupported model type for grid search: {model_type}")

    # save results
    df_res = pd.DataFrame(results)
    df_res.to_csv(report_path, index=False)
    print(f"Saved grid-search report to {report_path}")

    # determine best hyperparameters and save a small JSON summary
    metric = "combined_mean_auc" if "uplift" in cfg else "mean_auc"
    best_path = out_dir / f"best_{cfg_name}.json"

    try:
        if metric in df_res.columns:
            df_valid = df_res.dropna(subset=[metric])
            if not df_valid.empty:
                best_row = df_valid.sort_values(metric, ascending=False).iloc[0].to_dict()

                # convert numpy / pandas types to native Python types
                serializable = {}
                for k, v in best_row.items():
                    if pd.isna(v):
                        serializable[k] = None
                    elif isinstance(v, (np.floating, np.integer)):
                        serializable[k] = v.item()
                    else:
                        serializable[k] = v

                with best_path.open("w") as f:
                    json.dump(serializable, f, indent=2)
                print(f"Saved best params to {best_path}")
            else:
                print(f"No valid results to choose best params for metric '{metric}'")
        else:
            print(f"Metric '{metric}' not present in results; skipping best-params output")
    except Exception as e:
        print(f"Could not write best params file: {e}")


def main(config_path: str, folds: int):
    grid_search_config(config_path, n_splits=folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search CV for a config file and save AUC results.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    args = parser.parse_args()
    main(args.config, args.folds)

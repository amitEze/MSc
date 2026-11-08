Running the pipelines (xgboost)

Prerequisites
- Python 3.8+
- Install the following
```
pip install numpy pandas scikit-learn xgboost matplotlib pyyaml
```

Workflow
- 1) Build processed base tables for `train` and `test` (required)

Run from the repository root to create processed tables:
command should be
python -m src.data.build_base_table --split <train/test> optional: --web_visits_path --app_usage_path --claims_path --churn_labels_path

```
python -m src.data.build_base_table --split train
python -m src.data.build_base_table --split test
```

- 2) (Optional) Run grid search (xgboost)

```
python -m src.pipelines.grid_search --config config/uplift_xgb.yml --folds 5
```
Outputs
 - `output/grid_search/xgb/uplift_xgb` - contains grid search results CSV files
 - `output/grid_search/xgb/uplift_xgb/best_hyperparams.yml` - contains best hyperparameters found

- 3) Run the two-model uplift pipeline (train/validate + test predictions)

```
python -m src.pipelines.run_uplift_two_model --config config/uplift_xgb.yml
```

Outputs
- Validation scores/metrics and test predictions are written under `output/uplift/xgb/`.
  - `output/uplift/xgb/val_scores.csv`
  - `output/uplift/xgb/val_metrics.txt`
  - `output/uplift/xgb/test_scores.csv` (top-selected members)
  - `output/uplift/xgb/test_metrics.txt`
  - `plots/xgb/` - contains precision-recall, qini curve and roc plots

Questions or changes
- If you want a `requirements.txt`, PowerShell helper scripts, or CI tasks, I can add them.

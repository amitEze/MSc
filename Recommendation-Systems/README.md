# 🎬 MovieLens Recommendation System

A modular, production-ready recommendation system for the MovieLens 32M Kaggle Competition. Optimized for **Weighted RMSE** performance with comprehensive cold-start handling.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Modes](#pipeline-modes)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Cold Start Strategy](#cold-start-strategy)
- [Extending the Pipeline](#extending-the-pipeline)

## 🎯 Overview

This project implements a recommendation system pipeline targeting the MovieLens 32M competition where the evaluation metric is **Weighted RMSE**:

$$W\text{-}RMSE = \sqrt{\frac{\sum_{i} w_i (y_i - \hat{y}_i)^2}{\sum_{i} w_i}}$$

where $w_i = \frac{1}{n_i}$ and $n_i$ is the number of ratings for movie $i$.

**Key Insight:** Niche movies (few ratings) have HIGH weights, so accurate predictions on long-tail items matter more than blockbusters!

## 📁 Project Structure

```
project_root/
├── config/
│   └── config.yaml         # Hyperparameters and file paths
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data ingestion and validation
│   ├── preprocessing.py     # Feature engineering (genres, TF-IDF, BERT)
│   ├── train.py             # Model definitions and training logic
│   └── evaluate.py          # W-RMSE metrics and reporting
├── data/
│   ├── movies.csv           # Movie metadata
│   ├── tags.csv             # User-generated tags
│   ├── train.csv            # Training data (explicit + implicit)
│   └── ratings_submission.csv  # Test data for predictions
├── outputs/                 # Generated outputs (logs, plots, submissions)
├── main.py                  # Pipeline entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## 🚀 Installation

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Advanced models (for better performance)
pip install implicit catboost sentence-transformers
```

## ⚡ Quick Start

### Run Full Pipeline

```bash
python main.py --config config/config.yaml
```

This will:
1. Load and validate all data files
2. Engineer features (genres, TF-IDF tags)
3. Train and cross-validate models
4. Evaluate cold-start performance
5. Generate submission file

### Check Outputs

After running, find results in `outputs/`:
- `submission.csv` - Kaggle submission file
- `evaluation_report.txt` - Model comparison report
- `weight_distribution.png` - W-RMSE weight analysis
- `model_comparison.png` - Model performance visualization
- `pipeline.log` - Detailed execution log

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Key hyperparameters
seed: 42

models:
  bias:
    damping_user: 25      # Regularization for user bias
    damping_movie: 10     # Regularization for movie bias
  
  catboost:
    iterations: 500
    learning_rate: 0.1
    depth: 6

preprocessing:
  tfidf:
    max_features: 500     # Number of TF-IDF features
  bert:
    enabled: false        # Set true for BERT embeddings
```

## 🔄 Pipeline Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Full** | `python main.py --mode full` | Complete pipeline (default) |
| **Train** | `python main.py --mode train` | Train models only |
| **Evaluate** | `python main.py --mode evaluate` | Train + cold start eval |
| **Submit** | `python main.py --mode submit` | Train + generate submission |

### Additional Flags

- `--load-models`: Load pre-trained models from `outputs/models/` instead of training
  ```bash
  python main.py --mode submit --load-models
  ```

## ⚡ Development Mode (Fast Iterations)

For faster development cycles, use data sampling and reduced cross-validation:

```yaml
# config/config.yaml
data:
  sample_fraction: 0.1    # Use 10% of training data (1.0 = full dataset)
  sample_seed: 42         # Random seed for reproducibility

validation:
  fast_mode: true         # Use 2 CV folds instead of 5
  n_splits: 5             # Full CV splits (when fast_mode=false)
```

**Development workflow:**
1. Set `sample_fraction: 0.1` and `fast_mode: true` for quick testing
2. Iterate on models and features
3. Set `sample_fraction: 1.0` and `fast_mode: false` for final training
4. Use `--load-models` to skip retraining for submission generation

**Warning:** Training with sampled data will display:
```
⚠️ DEVELOPMENT MODE: Using 10.0% of training data
```

## 🤖 Models Implemented

### Baseline Models

| Model | Description | Expected W-RMSE |
|-------|-------------|-----------------|
| `GlobalMeanModel` | Predicts global average | ~1.05 |
| `BiasModel` | User + Movie bias (Netflix Prize) | ~0.90-0.95 |
| `BiasGenreModel` | Bias + per-genre preferences | ~0.88-0.92 |

### Advanced Models (Optional)

| Model | Key Advantage |
|-------|---------------|
| `ALSModel` | Matrix factorization via Alternating Least Squares |
| `CatBoostModel` | Native sample weights for W-RMSE optimization |
| `ColdStartContentModel` | Content-based for new items |
| `EnsembleModel` | Weighted averaging or stacking |

## 📊 Evaluation Metrics

### Weighted RMSE (Primary Metric)

The competition metric weights predictions by inverse movie popularity:

```python
from src.evaluate import Evaluator

evaluator = Evaluator(movie_counts)
score = evaluator.weighted_rmse(y_true, y_pred, movie_ids)
```

### Weight Distribution Analysis

```python
# Visualize how weights are distributed
evaluator.analyze_weight_distribution(explicit_train)
```

This reveals that:
- Movies with 1-10 ratings contribute ~70% of total weight
- Top 1% popular movies contribute only ~0.5% of weight

## ❄️ Cold Start Strategy

New/niche movies dominate W-RMSE. Our cold-start approach:

1. **Content Features:** Genre vectors + TF-IDF tag embeddings
2. **Similarity Search:** Find k-nearest neighbors by content
3. **Prediction:** Weighted average of user's ratings on similar movies

### Evaluate Cold Start

```python
from src.train import ColdStartContentModel

content_model = ColdStartContentModel(movies_df, genre_cols)
results = evaluator.evaluate_cold_start(explicit_df, content_model)
```

## 🔧 Extending the Pipeline

### Add a New Model

1. Create class inheriting from `BaseModel` in `src/train.py`:

```python
class MyModel(BaseModel):
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        # Training logic
        pass
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        # Prediction logic
        return predictions
```

2. Add cross-validation wrapper in `main.py`:

```python
def my_model_cv(train_df, val_df):
    model = MyModel()
    model.fit(train_df)
    return model.predict(val_df)

results['my_model'] = trainer.cross_validate(my_model_cv, ...)
```

### Add New Features

Extend `FeatureEngineer` in `src/preprocessing.py`:

```python
def create_my_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering logic
    return df_with_features
```

## 📝 Grading Strategy

| Component | Points | Strategy |
|-----------|--------|----------|
| Beat Baseline (0.9446) | 60% | Bias models achieve this |
| Cold Start Section | 10% | Content-based + evaluation |
| Leaderboard Performance | 30% | Advanced models + ensembling |

## 🔗 Resources

- [Kaggle Competition Page](https://www.kaggle.com/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Netflix Prize Paper](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

## 📄 License

MIT License - feel free to use for academic purposes.

---

**Happy Recommending! 🎬🍿**

"""
MovieLens Recommendation System - Source Package
================================================

This package contains modules for building a movie recommendation system
optimized for Weighted RMSE performance.

Modules:
    - data_loader: Data ingestion and raw data handling
    - preprocessing: Feature engineering and transformations
    - train: Model training logic
    - evaluate: Metrics and evaluation framework
"""

from .data_loader import DataLoader
from .preprocessing import FeatureEngineer
from .train import ModelTrainer
from .evaluate import Evaluator

__all__ = [
    "DataLoader",
    "FeatureEngineer", 
    "ModelTrainer",
    "Evaluator",
]

__version__ = "1.0.0"

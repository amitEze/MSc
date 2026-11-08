"""
Model Training Module
=====================

This module contains all model definitions and training logic for the
MovieLens recommendation system.

Models included:
- GlobalMeanModel: Baseline predicting global average
- BiasModel: User + Movie bias model (Netflix Prize classic)
- BiasGenreModel: Bias model with per-genre adjustments
- ALSModel: Alternating Least Squares matrix factorization
- CatBoostModel: Gradient boosting with sample weights
- ColdStartContentModel: Content-based model for cold start items
- EnsembleModel: Model combination strategies

Classes:
    ModelTrainer: Main class for training and managing models.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all recommendation models."""
    
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for given data."""
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Loaded model instance.
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model


class GlobalMeanModel(BaseModel):
    """
    Baseline model predicting global mean rating for all items.
    
    This serves as the simplest baseline to beat.
    
    Attributes:
        global_mean (float): The mean rating from training data.
    """
    
    def __init__(self) -> None:
        self.global_mean: float = 3.5
    
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Compute global mean from training data.
        
        Args:
            train_df: DataFrame with 'rating' column.
        """
        self.global_mean = train_df['rating'].mean()
        logger.info(f"GlobalMeanModel fitted with mean={self.global_mean:.4f}")
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Predict global mean for all instances.
        
        Args:
            data_df: DataFrame with instances to predict.
            
        Returns:
            Array of predictions (all equal to global mean).
        """
        return np.full(len(data_df), self.global_mean)


class BiasModel(BaseModel):
    """
    User + Movie Bias Model (Netflix Prize Classic).
    
    Predicts: rating = global_mean + user_bias + movie_bias
    
    Uses damping (shrinkage) to prevent overfitting for users/movies
    with few ratings. Formula: bias = sum(residuals) / (count + damping)
    
    Attributes:
        global_mean (float): Global average rating.
        user_bias (Dict): User ID to bias mapping.
        movie_bias (Dict): Movie ID to bias mapping.
        damping_user (int): Regularization for user bias.
        damping_movie (int): Regularization for movie bias.
    """
    
    def __init__(
        self,
        damping_user: int = 25,
        damping_movie: int = 10
    ) -> None:
        """
        Initialize BiasModel with damping parameters.
        
        Args:
            damping_user: Shrinkage factor for user biases.
            damping_movie: Shrinkage factor for movie biases.
        """
        self.damping_user = damping_user
        self.damping_movie = damping_movie
        self.global_mean: float = 3.5
        self.user_bias: Dict[int, float] = {}
        self.movie_bias: Dict[int, float] = {}
    
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Fit bias model to training data.
        
        Computes movie bias first (since movies are more stable),
        then computes user bias from residuals.
        
        Args:
            train_df: DataFrame with 'user_id', 'movie_id', 'rating'.
        """
        train_df = train_df.copy()
        self.global_mean = train_df['rating'].mean()
        
        # Calculate movie bias with shrinkage
        movie_stats = train_df.groupby('movie_id')['rating'].agg(['sum', 'count'])
        movie_stats['bias'] = (
            (movie_stats['sum'] - movie_stats['count'] * self.global_mean) /
            (movie_stats['count'] + self.damping_movie)
        )
        self.movie_bias = movie_stats['bias'].to_dict()
        
        # Calculate residual after movie bias
        train_df['residual'] = (
            train_df['rating'] - self.global_mean -
            train_df['movie_id'].map(self.movie_bias).fillna(0)
        )
        
        # Calculate user bias from residuals
        user_stats = train_df.groupby('user_id')['residual'].agg(['sum', 'count'])
        user_stats['bias'] = user_stats['sum'] / (user_stats['count'] + self.damping_user)
        self.user_bias = user_stats['bias'].to_dict()
        
        logger.info(f"BiasModel fitted: {len(self.user_bias)} users, {len(self.movie_bias)} movies")
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using bias model.
        
        Args:
            data_df: DataFrame with 'user_id', 'movie_id'.
            
        Returns:
            Array of clipped predictions [0.5, 5.0].
        """
        predictions = (
            self.global_mean +
            data_df['movie_id'].map(self.movie_bias).fillna(0) +
            data_df['user_id'].map(self.user_bias).fillna(0)
        )
        return predictions.clip(0.5, 5.0).values


class BiasGenreModel(BaseModel):
    """
    Bias model extended with per-genre user preferences.
    
    Adds a small adjustment based on how much a user tends to
    rate movies of each genre above/below their average.
    
    Attributes:
        bias_model (BiasModel): Base bias model.
        user_genre_prefs (Dict): User-genre preference adjustments.
        genre_cols (List): List of genre column names.
        movies_df (pd.DataFrame): Movies DataFrame with genre features.
    """
    
    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: List[str],
        damping_user: int = 25,
        damping_movie: int = 10,
        genre_damping: int = 5,
        adjustment_factor: float = 0.1
    ) -> None:
        """
        Initialize BiasGenreModel.
        
        Args:
            movies_df: Movies DataFrame with genre columns.
            genre_cols: List of genre column names.
            damping_user: Shrinkage factor for user biases.
            damping_movie: Shrinkage factor for movie biases.
            genre_damping: Shrinkage factor for genre preferences.
            adjustment_factor: Weight for genre adjustments.
        """
        self.movies_df = movies_df
        self.genre_cols = genre_cols
        self.genre_damping = genre_damping
        self.adjustment_factor = adjustment_factor
        self.bias_model = BiasModel(damping_user, damping_movie)
        self.user_genre_prefs: Dict[str, Dict] = {}
    
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Fit bias model and compute user-genre preferences.
        
        Args:
            train_df: DataFrame with 'user_id', 'movie_id', 'rating'.
        """
        # Fit base bias model
        self.bias_model.fit(train_df)
        
        # Merge genre info
        train_with_genres = train_df.merge(
            self.movies_df[['movie_id'] + self.genre_cols],
            on='movie_id',
            how='left'
        )
        
        # Calculate residual after bias prediction
        train_with_genres['residual'] = (
            train_with_genres['rating'] - self.bias_model.global_mean -
            train_with_genres['movie_id'].map(self.bias_model.movie_bias).fillna(0) -
            train_with_genres['user_id'].map(self.bias_model.user_bias).fillna(0)
        )
        
        # Compute user-genre preferences
        for genre in self.genre_cols:
            genre_train = train_with_genres[train_with_genres[genre] == 1]
            if len(genre_train) > 0:
                user_genre_stats = genre_train.groupby('user_id')['residual'].agg(['sum', 'count'])
                user_genre_stats['pref'] = (
                    user_genre_stats['sum'] /
                    (user_genre_stats['count'] + self.genre_damping)
                )
                self.user_genre_prefs[genre] = user_genre_stats['pref'].to_dict()
        
        logger.info(f"BiasGenreModel fitted with {len(self.user_genre_prefs)} genre preferences")
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with genre adjustments.
        
        Args:
            data_df: DataFrame with 'user_id', 'movie_id'.
            
        Returns:
            Array of clipped predictions [0.5, 5.0].
        """
        # Get base bias predictions
        predictions = pd.Series(self.bias_model.predict(data_df), index=data_df.index)
        
        # Merge genre info
        data_with_genres = data_df.merge(
            self.movies_df[['movie_id'] + self.genre_cols],
            on='movie_id',
            how='left'
        )
        
        # Add genre adjustments
        for genre in self.genre_cols:
            if genre in self.user_genre_prefs:
                genre_adj = data_with_genres.apply(
                    lambda x: self.user_genre_prefs[genre].get(x['user_id'], 0) * x.get(genre, 0) * self.adjustment_factor,
                    axis=1
                )
                predictions = predictions + genre_adj.values
        
        return predictions.clip(0.5, 5.0).values


class ALSModel(BaseModel):
    """
    Alternating Least Squares Matrix Factorization.
    
    Uses the implicit library to learn latent factors for users
    and items. Converts explicit ratings to confidence weights.
    
    Attributes:
        n_factors (int): Number of latent factors.
        regularization (float): L2 regularization strength.
        iterations (int): Number of ALS iterations.
        alpha (float): Confidence weight scaling factor.
    """
    
    def __init__(
        self,
        n_factors: int = 100,
        regularization: float = 0.1,
        iterations: int = 15,
        alpha: float = 40,
        random_state: int = 42
    ) -> None:
        """
        Initialize ALS model.
        
        Args:
            n_factors: Number of latent factors.
            regularization: L2 regularization strength.
            iterations: Number of ALS iterations.
            alpha: Confidence weight scaling (confidence = 1 + alpha * rating/5).
            random_state: Random seed for reproducibility.
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.global_mean: float = 3.5
    
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Train ALS model on explicit rating data.
        
        Args:
            train_df: DataFrame with 'user_id', 'movie_id', 'rating'.
        """
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            logger.error("implicit library not installed. Run: pip install implicit")
            raise
        
        train_df = train_df.copy()
        self.global_mean = train_df['rating'].mean()
        
        # Encode user and item IDs
        train_df['user_idx'] = self.user_encoder.fit_transform(train_df['user_id'])
        train_df['item_idx'] = self.item_encoder.fit_transform(train_df['movie_id'])
        
        # Convert ratings to confidence weights
        train_df['confidence'] = 1 + self.alpha * (train_df['rating'] / 5.0)
        
        # Create sparse user-item matrix
        sparse_matrix = csr_matrix(
            (train_df['confidence'], (train_df['user_idx'], train_df['item_idx'])),
            shape=(train_df['user_idx'].max() + 1, train_df['item_idx'].max() + 1)
        )
        
        # Train ALS
        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )
        self.model.fit(sparse_matrix)
        
        logger.info(f"ALSModel fitted with {self.n_factors} factors")
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from ALS model.
        
        Args:
            data_df: DataFrame with 'user_id', 'movie_id'.
            
        Returns:
            Array of clipped predictions [0.5, 5.0].
        """
        predictions = []
        
        for _, row in data_df.iterrows():
            user_id, movie_id = row['user_id'], row['movie_id']
            
            try:
                user_idx = self.user_encoder.transform([user_id])[0]
                item_idx = self.item_encoder.transform([movie_id])[0]
                
                # Dot product of user and item factors
                pred = np.dot(
                    self.model.user_factors[user_idx],
                    self.model.item_factors[item_idx]
                )
                # Scale back to rating range
                pred = pred * 0.5 + self.global_mean
            except (ValueError, IndexError):
                pred = self.global_mean
            
            predictions.append(pred)
        
        return np.clip(predictions, 0.5, 5.0)


class CatBoostModel(BaseModel):
    """
    CatBoost gradient boosting model with sample weights.
    
    Key advantage: Native sample_weight parameter allows direct
    optimization of Weighted RMSE by weighting samples as 1/n_i.
    
    Attributes:
        iterations (int): Number of boosting iterations.
        learning_rate (float): Step size for gradient descent.
        depth (int): Maximum tree depth.
        early_stopping_rounds (int): Early stopping patience.
    """
    
    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.1,
        depth: int = 6,
        early_stopping_rounds: int = 50,
        verbose: int = 100,
        random_state: int = 42
    ) -> None:
        """
        Initialize CatBoost model.
        
        Args:
            iterations: Number of boosting iterations.
            learning_rate: Step size for gradient descent.
            depth: Maximum tree depth.
            early_stopping_rounds: Early stopping patience.
            verbose: Logging frequency.
            random_state: Random seed for reproducibility.
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.feature_cols: List[str] = []
    
    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        movie_counts: Dict[int, int],
        val_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> None:
        """
        Train CatBoost with sample weights for W-RMSE optimization.
        
        Args:
            train_df: Training DataFrame with features and 'rating'.
            feature_cols: List of feature column names.
            movie_counts: Dictionary of movie_id to rating count.
            val_df: Optional validation DataFrame for early stopping.
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            logger.error("catboost not installed. Run: pip install catboost")
            raise
        
        self.feature_cols = feature_cols
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['rating']
        
        # Calculate sample weights for W-RMSE optimization
        sample_weights = train_df['movie_id'].map(
            lambda x: 1.0 / movie_counts.get(x, 1)
        ).values
        
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function='RMSE',
            random_seed=self.random_state,
            verbose=self.verbose
        )
        
        if val_df is not None:
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df['rating']
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.early_stopping_rounds
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        logger.info(f"CatBoostModel fitted with {len(feature_cols)} features")
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from CatBoost model.
        
        Args:
            data_df: DataFrame with feature columns.
            
        Returns:
            Array of clipped predictions [0.5, 5.0].
        """
        X = data_df[self.feature_cols].fillna(0)
        predictions = self.model.predict(X)
        return np.clip(predictions, 0.5, 5.0)


class ColdStartContentModel(BaseModel):
    """
    Content-based model for cold start items.
    
    When a movie has no/few ratings, predicts based on:
    1. Genre similarity (Jaccard/Cosine)
    2. Tag similarity (TF-IDF/BERT)
    
    Uses weighted average of user's ratings on similar movies.
    
    Attributes:
        content_matrix (np.ndarray): Content feature matrix.
        movie_ids (List): List of movie IDs.
        n_similar (int): Number of similar movies to consider.
    """
    
    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_cols: List[str],
        tfidf_cols: Optional[List[str]] = None,
        n_similar: int = 20
    ) -> None:
        """
        Initialize content-based cold start model.
        
        Args:
            movies_df: Movies DataFrame with content features.
            genre_cols: List of genre column names.
            tfidf_cols: Optional list of TF-IDF column names.
            n_similar: Number of similar movies to consider.
        """
        self.n_similar = n_similar
        self.movies_df = movies_df.set_index('movie_id')
        self.genre_cols = genre_cols
        self.tfidf_cols = tfidf_cols or []
        
        # Create content feature matrix
        feature_cols = genre_cols + self.tfidf_cols
        available_cols = [c for c in feature_cols if c in self.movies_df.columns]
        self.content_matrix = self.movies_df[available_cols].fillna(0).values
        self.movie_ids = self.movies_df.index.tolist()
        self.user_ratings_df: Optional[pd.DataFrame] = None
        
        logger.info(f"ColdStartContentModel initialized with {len(available_cols)} content features")
    
    def fit(self, train_df: pd.DataFrame, **kwargs) -> None:
        """
        Store training data for prediction.
        
        Args:
            train_df: Training DataFrame with user ratings.
        """
        self.user_ratings_df = train_df
        logger.info(f"ColdStartContentModel fitted with {len(train_df)} training ratings")
    
    def find_similar_movies(
        self,
        movie_id: int,
        n_similar: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Find n most similar movies by content similarity.
        
        Args:
            movie_id: Target movie ID.
            n_similar: Number of similar movies (uses default if None).
            
        Returns:
            List of (movie_id, similarity_score) tuples.
        """
        n_similar = n_similar or self.n_similar
        
        if movie_id not in self.movie_ids:
            return []
        
        movie_idx = self.movie_ids.index(movie_id)
        movie_vector = self.content_matrix[movie_idx].reshape(1, -1)
        
        # Compute similarity to all movies
        similarities = cosine_similarity(movie_vector, self.content_matrix)[0]
        
        # Get top similar (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar + 1]
        similar_movies = [(self.movie_ids[i], similarities[i]) for i in similar_indices]
        
        return similar_movies
    
    def predict(self, data_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for cold-start items using content similarity.
        
        Args:
            data_df: DataFrame with 'user_id', 'movie_id'.
            
        Returns:
            Array of predictions.
        """
        predictions = []
        
        for _, row in data_df.iterrows():
            pred = self._predict_single(row['user_id'], row['movie_id'])
            predictions.append(pred)
        
        return np.clip(predictions, 0.5, 5.0)
    
    def _predict_single(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a single user-movie pair.
        
        Args:
            user_id: User ID.
            movie_id: Movie ID.
            
        Returns:
            Predicted rating.
        """
        if self.user_ratings_df is None:
            return 3.5
        
        # Get user's rated movies
        user_ratings = self.user_ratings_df[self.user_ratings_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return 3.5
        
        # Find similar movies to the target
        similar_movies = self.find_similar_movies(movie_id)
        
        if not similar_movies:
            return user_ratings['rating'].mean()
        
        # Find overlap between similar movies and user's ratings
        similar_dict = dict(similar_movies)
        user_rated_similar = user_ratings[user_ratings['movie_id'].isin(similar_dict.keys())]
        
        if len(user_rated_similar) == 0:
            return user_ratings['rating'].mean()
        
        # Weighted average by similarity
        weights = user_rated_similar['movie_id'].map(similar_dict).values
        ratings = user_rated_similar['rating'].values
        
        return np.sum(weights * ratings) / np.sum(weights)


class EnsembleModel:
    """
    Ensemble methods for combining multiple model predictions.
    
    Supports weighted averaging and stacking meta-models.
    
    Attributes:
        models (Dict): Dictionary of model name to model instance.
        weights (np.ndarray): Weights for each model in ensemble.
    """
    
    def __init__(self) -> None:
        """Initialize EnsembleModel."""
        self.models: Dict[str, BaseModel] = {}
        self.weights: Optional[np.ndarray] = None
        self.meta_model: Optional[Ridge] = None
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier.
            model: Fitted model instance.
        """
        self.models[name] = model
        logger.info(f"Added model '{name}' to ensemble")
    
    def weighted_average(
        self,
        predictions_list: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Combine predictions using weighted average.
        
        Args:
            predictions_list: List of prediction arrays.
            weights: Optional weights (uniform if None).
            
        Returns:
            Blended predictions.
        """
        predictions_list = [np.array(p) for p in predictions_list]
        
        if weights is None:
            weights = np.ones(len(predictions_list))
        
        weights = np.array(weights) / np.sum(weights)
        
        ensemble = np.zeros(len(predictions_list[0]))
        for pred, w in zip(predictions_list, weights):
            ensemble += w * pred
        
        return np.clip(ensemble, 0.5, 5.0)
    
    def optimize_weights(
        self,
        predictions_list: List[np.ndarray],
        y_true: np.ndarray,
        movie_ids: np.ndarray,
        movie_counts: Dict[int, int],
        wrmse_func: Callable,
        n_trials: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Find optimal ensemble weights using random search.
        
        Args:
            predictions_list: List of prediction arrays.
            y_true: True ratings.
            movie_ids: Movie IDs for weight calculation.
            movie_counts: Movie rating counts.
            wrmse_func: W-RMSE calculation function.
            n_trials: Number of random search iterations.
            
        Returns:
            Tuple of (best_weights, best_score).
        """
        best_weights = None
        best_score = float('inf')
        n_models = len(predictions_list)
        
        for _ in range(n_trials):
            weights = np.random.dirichlet(np.ones(n_models))
            ensemble = self.weighted_average(predictions_list, weights)
            score = wrmse_func(y_true, ensemble, movie_ids, movie_counts)
            
            if score < best_score:
                best_score = score
                best_weights = weights
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {best_weights}, score: {best_score:.4f}")
        
        return best_weights, best_score
    
    def fit_stacking(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        movie_ids: np.ndarray,
        movie_counts: Dict[int, int],
        alpha: float = 1.0
    ) -> None:
        """
        Train a stacking meta-model with sample weights.
        
        Args:
            predictions_dict: Dictionary of model name to predictions.
            y_true: True ratings.
            movie_ids: Movie IDs for weight calculation.
            movie_counts: Movie rating counts.
            alpha: Ridge regularization parameter.
        """
        X_stack = pd.DataFrame()
        
        for model_name, preds in predictions_dict.items():
            X_stack[f'pred_{model_name}'] = preds
        
        # Add interaction features
        if len(predictions_dict) >= 2:
            pred_cols = list(X_stack.columns)
            X_stack['pred_mean'] = X_stack[pred_cols].mean(axis=1)
            X_stack['pred_std'] = X_stack[pred_cols].std(axis=1)
            X_stack['pred_max'] = X_stack[pred_cols].max(axis=1)
            X_stack['pred_min'] = X_stack[pred_cols].min(axis=1)
        
        # Sample weights for W-RMSE
        sample_weights = np.array([1.0 / movie_counts.get(mid, 1) for mid in movie_ids])
        
        self.meta_model = Ridge(alpha=alpha)
        self.meta_model.fit(X_stack, y_true, sample_weight=sample_weights)
        
        logger.info("Stacking meta-model fitted")
    
    def predict_stacking(
        self,
        predictions_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Generate predictions using stacking meta-model.
        
        Args:
            predictions_dict: Dictionary of model name to predictions.
            
        Returns:
            Stacked predictions.
        """
        X_stack = pd.DataFrame()
        
        for model_name, preds in predictions_dict.items():
            X_stack[f'pred_{model_name}'] = preds
        
        if len(predictions_dict) >= 2:
            pred_cols = list(X_stack.columns)
            X_stack['pred_mean'] = X_stack[pred_cols].mean(axis=1)
            X_stack['pred_std'] = X_stack[pred_cols].std(axis=1)
            X_stack['pred_max'] = X_stack[pred_cols].max(axis=1)
            X_stack['pred_min'] = X_stack[pred_cols].min(axis=1)
        
        predictions = self.meta_model.predict(X_stack)
        return np.clip(predictions, 0.5, 5.0)


class ModelTrainer:
    """
    Orchestrates model training and cross-validation.
    
    Provides a unified interface for training different models
    and evaluating them using K-Fold cross-validation.
    
    Attributes:
        random_state (int): Random seed for reproducibility.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.trained_models: Dict[str, BaseModel] = {}
        
        logger.info(f"ModelTrainer initialized with seed={random_state}")
    
    def cross_validate(
        self,
        model_fn: Callable,
        train_df: pd.DataFrame,
        wrmse_func: Callable,
        movie_counts: Dict[int, int],
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation with W-RMSE scoring.
        
        Args:
            model_fn: Function that takes (train_df, val_df) and returns predictions.
            train_df: Training DataFrame.
            wrmse_func: W-RMSE calculation function.
            movie_counts: Movie rating counts.
            n_splits: Number of CV folds.
            
        Returns:
            Dictionary with fold scores and statistics.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            
            predictions = model_fn(train_fold, val_fold)
            
            score = wrmse_func(
                y_true=val_fold['rating'].values,
                y_pred=predictions,
                movie_ids=val_fold['movie_id'].values,
                movie_rating_counts=movie_counts
            )
            
            fold_scores.append(score)
            logger.info(f"Fold {fold + 1}: W-RMSE = {score:.4f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        logger.info(f"CV Results: W-RMSE = {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'fold_scores': fold_scores,
            'mean': mean_score,
            'std': std_score
        }
    
    def train_and_store(
        self,
        name: str,
        model: BaseModel,
        train_df: pd.DataFrame,
        **kwargs
    ) -> BaseModel:
        """
        Train a model and store it for later use.
        
        Args:
            name: Model identifier.
            model: Model instance to train.
            train_df: Training DataFrame.
            **kwargs: Additional arguments for model.fit().
            
        Returns:
            Fitted model instance.
        """
        model.fit(train_df, **kwargs)
        self.trained_models[name] = model
        
        logger.info(f"Trained and stored model: {name}")
        
        return model
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """
        Retrieve a trained model by name.
        
        Args:
            name: Model identifier.
            
        Returns:
            Model instance or None if not found.
        """
        return self.trained_models.get(name)

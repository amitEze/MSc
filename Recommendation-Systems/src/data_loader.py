"""
Data Loader Module
==================

This module handles all data ingestion operations for the MovieLens
recommendation system. It provides a clean interface for loading and
validating raw data files.

Classes:
    DataLoader: Main class for data loading operations.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data ingestion and validation for MovieLens dataset.
    
    This class provides methods to load movies, tags, training data,
    and submission files. It also separates explicit ratings from
    implicit feedback (null ratings).
    
    Attributes:
        data_dir (Path): Directory containing data files.
        movies (pd.DataFrame): Loaded movies data.
        tags (pd.DataFrame): Loaded tags data.
        train (pd.DataFrame): Loaded training data.
        explicit_train (pd.DataFrame): Training data with explicit ratings.
        implicit_train (pd.DataFrame): Training data with implicit feedback (nulls).
    
    Example:
        >>> loader = DataLoader(data_dir="data")
        >>> loader.load_all()
        >>> print(loader.get_data_summary())
    """
    
    def __init__(self, data_dir: str, sample_fraction: float = 1.0, sample_seed: int = 42) -> None:
        """
        Initialize the DataLoader with a data directory.
        
        Args:
            data_dir: Path to the directory containing data files.
            sample_fraction: Fraction of data to sample (0.0-1.0). Use < 1.0 for faster development.
            sample_seed: Random seed for sampling reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.movies: Optional[pd.DataFrame] = None
        self.tags: Optional[pd.DataFrame] = None
        self.train: Optional[pd.DataFrame] = None
        self.explicit_train: Optional[pd.DataFrame] = None
        self.implicit_train: Optional[pd.DataFrame] = None
        self.submission: Optional[pd.DataFrame] = None
        
        if sample_fraction < 1.0:
            logger.warning(
                f"⚠️  DEVELOPMENT MODE: Using {sample_fraction*100:.1f}% of training data. "
                "Set sample_fraction=1.0 for full training."
            )
        
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_movies(self, filename: str = "movies.csv") -> pd.DataFrame:
        """
        Load the movies dataset.
        
        Args:
            filename: Name of the movies file.
            
        Returns:
            DataFrame containing movie information.
            
        Raises:
            FileNotFoundError: If the movies file doesn't exist.
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading movies from {filepath}")
        
        self.movies = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.movies):,} movies")
        
        return self.movies
    
    def load_tags(self, filename: str = "tags.csv") -> pd.DataFrame:
        """
        Load the tags dataset.
        
        Args:
            filename: Name of the tags file.
            
        Returns:
            DataFrame containing user-generated tags.
            
        Raises:
            FileNotFoundError: If the tags file doesn't exist.
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading tags from {filepath}")
        
        self.tags = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.tags):,} tags")
        
        return self.tags
    
    def load_train(self, filename: str = "train.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split training data into explicit and implicit ratings.
        
        Explicit ratings are non-null values that users provided.
        Implicit feedback (null ratings) indicates user viewed but didn't rate.
        
        Args:
            filename: Name of the training file.
            
        Returns:
            Tuple of (explicit_train, implicit_train) DataFrames.
            
        Raises:
            FileNotFoundError: If the training file doesn't exist.
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading training data from {filepath}")
        
        self.train = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.train):,} total training records")
        
        # Apply sampling if requested
        if self.sample_fraction < 1.0:
            original_size = len(self.train)
            self.train = self.train.sample(
                frac=self.sample_fraction,
                random_state=self.sample_seed
            ).reset_index(drop=True)
            logger.info(
                f"📉 Sampled {len(self.train):,} records ({self.sample_fraction*100:.1f}%) "
                f"from {original_size:,} for faster development"
            )
        
        # Separate explicit ratings from implicit feedback
        self.explicit_train = self.train[self.train['rating'].notna()].copy()
        self.implicit_train = self.train[self.train['rating'].isna()].copy()
        
        explicit_pct = len(self.explicit_train) / len(self.train) * 100
        implicit_pct = len(self.implicit_train) / len(self.train) * 100
        
        logger.info(f"Explicit ratings: {len(self.explicit_train):,} ({explicit_pct:.1f}%)")
        logger.info(f"Implicit feedback: {len(self.implicit_train):,} ({implicit_pct:.1f}%)")
        
        return self.explicit_train, self.implicit_train
    
    def load_submission(self, filename: str = "ratings_submission.csv") -> pd.DataFrame:
        """
        Load the submission/test dataset.
        
        The submission file has format: id (user_id_movie_id), prediction
        This method parses the id column into separate user_id and movie_id columns.
        
        Args:
            filename: Name of the submission file.
            
        Returns:
            DataFrame containing test data for predictions with user_id and movie_id columns.
            
        Raises:
            FileNotFoundError: If the submission file doesn't exist.
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading submission data from {filepath}")
        
        self.submission = pd.read_csv(filepath)
        
        # Parse id column (format: user_id_movie_id) into separate columns
        if 'id' in self.submission.columns and 'user_id' not in self.submission.columns:
            # Split the id column on underscore
            id_parts = self.submission['id'].str.split('_', expand=True)
            self.submission['user_id'] = id_parts[0].astype(int)
            self.submission['movie_id'] = id_parts[1].astype(int)
            logger.info("Parsed 'id' column into 'user_id' and 'movie_id'")
        
        logger.info(f"Loaded {len(self.submission):,} submission records")
        
        return self.submission
    
    def load_all(
        self,
        movies_file: str = "movies.csv",
        tags_file: str = "tags.csv",
        train_file: str = "train.csv",
        submission_file: str = "ratings_submission.csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets at once.
        
        Args:
            movies_file: Name of the movies file.
            tags_file: Name of the tags file.
            train_file: Name of the training file.
            submission_file: Name of the submission file.
            
        Returns:
            Dictionary containing all loaded DataFrames.
        """
        logger.info("Loading all datasets...")
        
        self.load_movies(movies_file)
        self.load_tags(tags_file)
        self.load_train(train_file)
        self.load_submission(submission_file)
        
        logger.info("All datasets loaded successfully")
        
        return {
            "movies": self.movies,
            "tags": self.tags,
            "train": self.train,
            "explicit_train": self.explicit_train,
            "implicit_train": self.implicit_train,
            "submission": self.submission
        }
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Generate a summary of loaded data statistics.
        
        Returns:
            Dictionary containing data statistics.
        """
        summary = {
            "movies_count": len(self.movies) if self.movies is not None else 0,
            "tags_count": len(self.tags) if self.tags is not None else 0,
            "train_count": len(self.train) if self.train is not None else 0,
            "explicit_count": len(self.explicit_train) if self.explicit_train is not None else 0,
            "implicit_count": len(self.implicit_train) if self.implicit_train is not None else 0,
            "submission_count": len(self.submission) if self.submission is not None else 0,
        }
        
        if self.explicit_train is not None:
            summary["unique_users"] = self.explicit_train['user_id'].nunique()
            summary["unique_movies"] = self.explicit_train['movie_id'].nunique()
            summary["rating_mean"] = self.explicit_train['rating'].mean()
            summary["rating_std"] = self.explicit_train['rating'].std()
        
        return summary
    
    def create_movie_count_lookup(self) -> Dict[int, int]:
        """
        Create a lookup dictionary of movie_id to rating count.
        
        This is essential for W-RMSE weight calculation where
        weight = 1 / n_i (n_i = number of ratings for movie i).
        
        Returns:
            Dictionary mapping movie_id to its rating count.
            
        Raises:
            ValueError: If explicit_train hasn't been loaded.
        """
        if self.explicit_train is None:
            raise ValueError("Training data not loaded. Call load_train() first.")
        
        movie_counts = self.explicit_train.groupby('movie_id').size().to_dict()
        logger.info(f"Created movie count lookup for {len(movie_counts):,} movies")
        
        return movie_counts

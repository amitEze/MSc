"""
Preprocessing Module
====================

This module handles all feature engineering and data transformations
for the MovieLens recommendation system.

Key transformations include:
- Genre parsing and binary encoding
- Tag feature extraction (TF-IDF and BERT embeddings)
- Implicit feedback feature engineering
- User and movie statistics computation

Classes:
    FeatureEngineer: Main class for feature engineering operations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering for the MovieLens recommendation system.
    
    This class provides methods for extracting features from movies,
    tags, and user interactions to enhance model performance.
    
    Attributes:
        genre_mlb (MultiLabelBinarizer): Fitted genre encoder.
        genre_columns (List[str]): List of genre column names.
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        movies_enriched (pd.DataFrame): Movies with all extracted features.
    
    Example:
        >>> fe = FeatureEngineer()
        >>> movies_enriched = fe.parse_genres(movies)
        >>> movies_enriched = fe.create_tag_features_tfidf(tags, movies_enriched)
    """
    
    def __init__(self) -> None:
        """Initialize the FeatureEngineer."""
        self.genre_mlb: Optional[MultiLabelBinarizer] = None
        self.genre_columns: List[str] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_columns: List[str] = []
        self.movies_enriched: Optional[pd.DataFrame] = None
        
        logger.info("FeatureEngineer initialized")
    
    def parse_genres(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse pipe-separated genres into binary encoding.
        
        Applies MultiLabelBinarizer to create one-hot encoded columns
        for each unique genre (approximately 20 binary features).
        
        Args:
            movies_df: DataFrame with 'genres' column containing
                pipe-separated genre strings (e.g., "Action|Comedy").
                
        Returns:
            DataFrame with added binary genre columns.
            
        Example:
            >>> fe.parse_genres(movies)
            # Adds columns: genre_Action, genre_Comedy, etc.
        """
        logger.info("Parsing genres from movies data")
        movies_df = movies_df.copy()
        
        # Split genres into lists
        movies_df['genre_list'] = movies_df['genres'].fillna('').str.split('|')
        
        # Apply MultiLabelBinarizer
        self.genre_mlb = MultiLabelBinarizer()
        genre_matrix = self.genre_mlb.fit_transform(movies_df['genre_list'])
        
        # Create DataFrame with genre columns
        self.genre_columns = [f'genre_{g}' for g in self.genre_mlb.classes_]
        genre_df = pd.DataFrame(
            genre_matrix,
            columns=self.genre_columns,
            index=movies_df.index
        )
        
        # Merge back
        self.movies_enriched = pd.concat([movies_df, genre_df], axis=1)
        
        logger.info(f"Parsed {len(self.genre_columns)} unique genres: {self.genre_columns[:5]}...")
        
        return self.movies_enriched
    
    def create_tag_features_tfidf(
        self,
        tags_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        max_features: int = 500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        stop_words: str = "english"
    ) -> pd.DataFrame:
        """
        Create TF-IDF features from user-generated tags.
        
        Aggregates all tags per movie into a single string and applies
        TF-IDF vectorization to extract semantic features.
        
        Args:
            tags_df: DataFrame with 'movie_id' and 'tag' columns.
            movies_df: DataFrame with movie information.
            max_features: Maximum number of TF-IDF features to extract.
            ngram_range: Range of n-grams for TF-IDF (min, max).
            min_df: Minimum document frequency for terms.
            stop_words: Stop words setting for TfidfVectorizer.
            
        Returns:
            DataFrame with added TF-IDF feature columns.
        """
        logger.info(f"Creating TF-IDF features with max_features={max_features}")
        
        # Aggregate all tags per movie into single string
        movie_tags = tags_df.groupby('movie_id')['tag'].apply(
            lambda x: ' '.join(x.astype(str).str.lower())
        ).reset_index()
        movie_tags.columns = ['movie_id', 'tags_combined']
        
        # Merge with movies (handle movies without tags)
        movies_with_tags = movies_df.merge(movie_tags, on='movie_id', how='left')
        movies_with_tags['tags_combined'] = movies_with_tags['tags_combined'].fillna('')
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(movies_with_tags['tags_combined'])
        
        # Create DataFrame
        self.tfidf_columns = [f'tag_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.tfidf_columns,
            index=movies_with_tags.index
        )
        
        self.movies_enriched = pd.concat([movies_with_tags, tfidf_df], axis=1)
        
        logger.info(f"Created {len(self.tfidf_columns)} TF-IDF features")
        
        return self.movies_enriched
    
    def create_tag_features_bert(
        self,
        tags_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        model_name: str = 'all-MiniLM-L6-v2',
        batch_size: int = 256
    ) -> pd.DataFrame:
        """
        Create BERT embeddings for movie tags.
        
        Uses sentence-transformers to generate dense embeddings that
        capture semantic meaning of tags. Slower but higher quality
        than TF-IDF for final model boost.
        
        Args:
            tags_df: DataFrame with 'movie_id' and 'tag' columns.
            movies_df: DataFrame with movie information.
            model_name: Sentence transformer model name.
            batch_size: Batch size for encoding.
            
        Returns:
            DataFrame with added BERT embedding columns.
            
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        logger.info(f"Creating BERT embeddings with model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        
        # Aggregate tags per movie
        movie_tags = tags_df.groupby('movie_id')['tag'].apply(
            lambda x: ' '.join(x.astype(str).str.lower())
        ).reset_index()
        movie_tags.columns = ['movie_id', 'tags_combined']
        
        movies_with_tags = movies_df.merge(movie_tags, on='movie_id', how='left')
        movies_with_tags['tags_combined'] = movies_with_tags['tags_combined'].fillna('')
        
        # Load BERT model
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        logger.info("Generating BERT embeddings (this may take a while)...")
        embeddings = model.encode(
            movies_with_tags['tags_combined'].tolist(),
            show_progress_bar=True,
            batch_size=batch_size
        )
        
        # Create DataFrame
        bert_columns = [f'tag_bert_{i}' for i in range(embeddings.shape[1])]
        bert_df = pd.DataFrame(
            embeddings,
            columns=bert_columns,
            index=movies_with_tags.index
        )
        
        self.movies_enriched = pd.concat([movies_with_tags, bert_df], axis=1)
        
        logger.info(f"Created {len(bert_columns)} BERT embedding features")
        
        return self.movies_enriched
    
    def create_implicit_features(
        self,
        explicit_df: pd.DataFrame,
        implicit_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features from implicit feedback (null ratings).
        
        Implicit signals indicate the user viewed but didn't rate a movie,
        providing additional engagement information.
        
        Features created:
        - user_implicit_count: Number of implicit interactions per user
        - user_explicit_count: Number of explicit ratings per user
        - user_rating_ratio: Ratio of explicit to total interactions
        - movie_implicit_count: Number of implicit views per movie
        - movie_explicit_count: Number of explicit ratings per movie
        - movie_rating_ratio: Ratio of explicit to total interactions
        
        Args:
            explicit_df: DataFrame with explicit ratings.
            implicit_df: DataFrame with implicit feedback (null ratings).
            
        Returns:
            Tuple of (user_features, movie_features) DataFrames.
        """
        logger.info("Creating features from implicit feedback")
        
        # User-level implicit features
        user_implicit = implicit_df.groupby('user_id').size().reset_index(name='user_implicit_count')
        user_explicit = explicit_df.groupby('user_id').size().reset_index(name='user_explicit_count')
        
        user_features = user_implicit.merge(user_explicit, on='user_id', how='outer').fillna(0)
        user_features['user_rating_ratio'] = (
            user_features['user_explicit_count'] /
            (user_features['user_implicit_count'] + user_features['user_explicit_count'] + 1)
        )
        
        # Movie-level implicit features
        movie_implicit = implicit_df.groupby('movie_id').size().reset_index(name='movie_implicit_count')
        movie_explicit = explicit_df.groupby('movie_id').size().reset_index(name='movie_explicit_count')
        
        movie_features = movie_implicit.merge(movie_explicit, on='movie_id', how='outer').fillna(0)
        movie_features['movie_rating_ratio'] = (
            movie_features['movie_explicit_count'] /
            (movie_features['movie_implicit_count'] + movie_features['movie_explicit_count'] + 1)
        )
        
        logger.info(f"Created user features: {user_features.shape}")
        logger.info(f"Created movie features: {movie_features.shape}")
        
        return user_features, movie_features
    
    def compute_user_movie_stats(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Compute user and movie statistics from training data.
        
        These statistics serve as features for models and provide
        baseline predictions for bias models.
        
        Args:
            train_df: DataFrame with 'user_id', 'movie_id', 'rating' columns.
            
        Returns:
            Tuple of (user_stats, movie_stats) dictionaries.
            Each contains 'mean', 'std', 'count' sub-dictionaries.
        """
        logger.info("Computing user and movie statistics")
        
        user_stats = {
            'mean': train_df.groupby('user_id')['rating'].mean().to_dict(),
            'std': train_df.groupby('user_id')['rating'].std().fillna(1.0).to_dict(),
            'count': train_df.groupby('user_id').size().to_dict()
        }
        
        movie_stats = {
            'mean': train_df.groupby('movie_id')['rating'].mean().to_dict(),
            'std': train_df.groupby('movie_id')['rating'].std().fillna(1.0).to_dict(),
            'count': train_df.groupby('movie_id').size().to_dict()
        }
        
        logger.info(f"Computed stats for {len(user_stats['mean'])} users and {len(movie_stats['mean'])} movies")
        
        return user_stats, movie_stats
    
    def create_model_features(
        self,
        df: pd.DataFrame,
        movies_df: pd.DataFrame,
        genre_cols: List[str],
        movie_counts: Dict[int, int],
        user_stats: Dict[str, Dict],
        movie_stats: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Create feature matrix for gradient boosting models.
        
        Combines genre features with user/movie statistics to create
        a complete feature set for training.
        
        Args:
            df: DataFrame with 'user_id', 'movie_id' columns.
            movies_df: Movies DataFrame with genre columns.
            genre_cols: List of genre column names.
            movie_counts: Dictionary of movie_id to rating count.
            user_stats: User statistics dictionary.
            movie_stats: Movie statistics dictionary.
            
        Returns:
            DataFrame with all features for model training.
        """
        df = df.merge(movies_df[['movie_id'] + genre_cols], on='movie_id', how='left')
        
        # Add user statistics
        df['user_mean'] = df['user_id'].map(user_stats.get('mean', {})).fillna(3.5)
        df['user_std'] = df['user_id'].map(user_stats.get('std', {})).fillna(1.0)
        df['user_count'] = df['user_id'].map(user_stats.get('count', {})).fillna(1)
        
        # Add movie statistics
        df['movie_mean'] = df['movie_id'].map(movie_stats.get('mean', {})).fillna(3.5)
        df['movie_std'] = df['movie_id'].map(movie_stats.get('std', {})).fillna(1.0)
        df['movie_count'] = df['movie_id'].map(movie_counts).fillna(1)
        
        return df

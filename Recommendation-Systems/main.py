#!/usr/bin/env python
"""
MovieLens Recommendation System - Main Pipeline
================================================

This is the entry point for running the recommendation system pipeline.
It orchestrates data loading, preprocessing, model training, evaluation,
and submission generation.

Usage:
    python main.py --config config/config.yaml
    python main.py --config config/config.yaml --mode train
    python main.py --config config/config.yaml --mode evaluate
    python main.py --config config/config.yaml --mode submit

Author: ML Engineering Team
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import FeatureEngineer
from src.train import (
    BiasModel,
    BiasGenreModel,
    GlobalMeanModel,
    ColdStartContentModel,
    EnsembleModel,
    ModelTrainer,
)
from src.evaluate import Evaluator


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Configured logger instance.
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory for log file
    log_file = log_config.get('log_file', 'outputs/pipeline.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_models(
    trained_models: Dict[str, Any],
    results: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Save trained models and artifacts to disk.
    
    Args:
        trained_models: Dictionary of trained model instances.
        results: Cross-validation results dictionary.
        data: Preprocessed data dictionary (for saving preprocessing artifacts).
        config: Configuration dictionary.
        logger: Logger instance.
    """
    persistence_config = config.get('model_persistence', {})
    
    if not persistence_config.get('enabled', True):
        logger.info("Model persistence disabled, skipping save")
        return
    
    models_dir = Path(persistence_config.get('models_dir', 'outputs/models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    filenames = persistence_config.get('filenames', {})
    
    # Save each trained model
    for model_name, model in trained_models.items():
        filename = filenames.get(model_name, f"{model_name}_model.pkl")
        filepath = models_dir / filename
        model.save(str(filepath))
    
    # Save CV results
    cv_results_file = persistence_config.get('cv_results_file', 'cv_results.pkl')
    cv_results_path = models_dir / cv_results_file
    with open(cv_results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"CV results saved to {cv_results_path}")
    
    # Save preprocessing artifacts needed for prediction
    preprocessing_file = persistence_config.get('preprocessing_file', 'preprocessing_artifacts.pkl')
    preprocessing_path = models_dir / preprocessing_file
    
    preprocessing_artifacts = {
        'movie_counts': data.get('movie_counts'),
        'genre_cols': data.get('genre_cols'),
        'tfidf_cols': data.get('tfidf_cols'),
        'user_stats': data.get('user_stats'),
        'movie_stats': data.get('movie_stats'),
    }
    
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_artifacts, f)
    logger.info(f"Preprocessing artifacts saved to {preprocessing_path}")
    
    logger.info(f"All models saved to {models_dir}")


def load_models(
    config: Dict[str, Any],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Load trained models and artifacts from disk.
    
    Args:
        config: Configuration dictionary.
        logger: Logger instance.
        
    Returns:
        Dictionary with trained_models, results, and preprocessing artifacts,
        or None if models not found.
    """
    from src.train import BaseModel
    
    persistence_config = config.get('model_persistence', {})
    
    if not persistence_config.get('enabled', True):
        logger.info("Model persistence disabled")
        return None
    
    models_dir = Path(persistence_config.get('models_dir', 'outputs/models'))
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return None
    
    filenames = persistence_config.get('filenames', {})
    trained_models = {}
    
    # Load each model that exists
    for model_name, filename in filenames.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                trained_models[model_name] = BaseModel.load(str(filepath))
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    if not trained_models:
        logger.warning("No models found to load")
        return None
    
    # Load CV results
    results = {}
    cv_results_file = persistence_config.get('cv_results_file', 'cv_results.pkl')
    cv_results_path = models_dir / cv_results_file
    if cv_results_path.exists():
        with open(cv_results_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"CV results loaded from {cv_results_path}")
    
    # Load preprocessing artifacts
    preprocessing_artifacts = {}
    preprocessing_file = persistence_config.get('preprocessing_file', 'preprocessing_artifacts.pkl')
    preprocessing_path = models_dir / preprocessing_file
    if preprocessing_path.exists():
        with open(preprocessing_path, 'rb') as f:
            preprocessing_artifacts = pickle.load(f)
        logger.info(f"Preprocessing artifacts loaded from {preprocessing_path}")
    
    logger.info(f"Loaded {len(trained_models)} models from {models_dir}")
    
    return {
        'trained_models': trained_models,
        'results': results,
        'preprocessing_artifacts': preprocessing_artifacts
    }


def run_data_loading(config: Dict[str, Any], logger: logging.Logger) -> Dict:
    """
    Execute data loading phase.
    
    Args:
        config: Configuration dictionary.
        logger: Logger instance.
        
    Returns:
        Dictionary containing loaded data.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Loading")
    logger.info("=" * 60)
    
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data')
    sample_fraction = data_config.get('sample_fraction', 1.0)
    sample_seed = data_config.get('sample_seed', 42)
    
    loader = DataLoader(
        data_dir=data_dir,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed
    )
    data = loader.load_all(
        movies_file=data_config.get('movies_file', 'movies.csv'),
        tags_file=data_config.get('tags_file', 'tags.csv'),
        train_file=data_config.get('train_file', 'train.csv'),
        submission_file=data_config.get('submission_file', 'ratings_submission.csv')
    )
    
    # Log data summary
    summary = loader.get_data_summary()
    logger.info(f"Data Summary: {summary}")
    
    # Add movie counts to data dict
    data['movie_counts'] = loader.create_movie_count_lookup()
    data['loader'] = loader
    
    return data


def run_preprocessing(
    config: Dict[str, Any],
    data: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Execute preprocessing and feature engineering phase.
    
    Args:
        config: Configuration dictionary.
        data: Dictionary containing loaded data.
        logger: Logger instance.
        
    Returns:
        Updated data dictionary with engineered features.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Preprocessing & Feature Engineering")
    logger.info("=" * 60)
    
    preprocessing_config = config.get('preprocessing', {})
    fe = FeatureEngineer()
    
    # Parse genres
    movies_enriched = fe.parse_genres(data['movies'])
    
    # Create TF-IDF features
    tfidf_config = preprocessing_config.get('tfidf', {})
    movies_enriched = fe.create_tag_features_tfidf(
        tags_df=data['tags'],
        movies_df=movies_enriched,
        max_features=tfidf_config.get('max_features', 500),
        ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
        min_df=tfidf_config.get('min_df', 5),
        stop_words=tfidf_config.get('stop_words', 'english')
    )
    
    # Optionally create BERT features
    bert_config = preprocessing_config.get('bert', {})
    if bert_config.get('enabled', False):
        try:
            movies_enriched = fe.create_tag_features_bert(
                tags_df=data['tags'],
                movies_df=movies_enriched,
                model_name=bert_config.get('model_name', 'all-MiniLM-L6-v2'),
                batch_size=bert_config.get('batch_size', 256)
            )
        except ImportError:
            logger.warning("BERT embeddings disabled - sentence-transformers not installed")
    
    # Create implicit features
    user_implicit_feats, movie_implicit_feats = fe.create_implicit_features(
        explicit_df=data['explicit_train'],
        implicit_df=data['implicit_train']
    )
    
    # Compute user/movie statistics
    user_stats, movie_stats = fe.compute_user_movie_stats(data['explicit_train'])
    
    # Update data dictionary
    data['movies_enriched'] = movies_enriched
    data['genre_cols'] = fe.genre_columns
    data['tfidf_cols'] = fe.tfidf_columns
    data['user_implicit_feats'] = user_implicit_feats
    data['movie_implicit_feats'] = movie_implicit_feats
    data['user_stats'] = user_stats
    data['movie_stats'] = movie_stats
    data['feature_engineer'] = fe
    
    logger.info(f"Feature engineering complete: {movies_enriched.shape}")
    
    return data


def run_training(
    config: Dict[str, Any],
    data: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Execute model training phase.
    
    Args:
        config: Configuration dictionary.
        data: Dictionary containing preprocessed data.
        logger: Logger instance.
        
    Returns:
        Dictionary containing trained models and results.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Model Training & Cross-Validation")
    logger.info("=" * 60)
    
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    models_config = config.get('models', {})
    validation_config = config.get('validation', {})
    
    # Initialize evaluator and trainer
    evaluator = Evaluator(
        movie_counts=data['movie_counts'],
        output_dir=config.get('data', {}).get('output_dir', 'outputs')
    )
    trainer = ModelTrainer(random_state=seed)
    
    results = {}
    trained_models = {}
    
    # Determine number of CV splits based on fast_mode
    fast_mode = validation_config.get('fast_mode', False)
    n_splits = 2 if fast_mode else validation_config.get('n_splits', 5)
    
    if fast_mode:
        logger.warning(f"⚡ FAST MODE: Using {n_splits} CV splits instead of {validation_config.get('n_splits', 5)} for faster evaluation")
    
    # 1. Global Mean Baseline
    logger.info("-" * 40)
    logger.info("Training: Global Mean Model")
    
    def global_mean_cv(train_df, val_df):
        model = GlobalMeanModel()
        model.fit(train_df)
        return model.predict(val_df)
    
    results['global_mean'] = trainer.cross_validate(
        model_fn=global_mean_cv,
        train_df=data['explicit_train'],
        wrmse_func=evaluator.weighted_rmse,
        movie_counts=data['movie_counts'],
        n_splits=n_splits
    )
    
    # 2. Bias Model
    logger.info("-" * 40)
    logger.info("Training: Bias Model")
    
    bias_config = models_config.get('bias', {})
    
    def bias_cv(train_df, val_df):
        model = BiasModel(
            damping_user=bias_config.get('damping_user', 25),
            damping_movie=bias_config.get('damping_movie', 10)
        )
        model.fit(train_df)
        return model.predict(val_df)
    
    results['bias'] = trainer.cross_validate(
        model_fn=bias_cv,
        train_df=data['explicit_train'],
        wrmse_func=evaluator.weighted_rmse,
        movie_counts=data['movie_counts'],
        n_splits=n_splits
    )
    
    # Train final bias model on full data
    bias_model = BiasModel(
        damping_user=bias_config.get('damping_user', 25),
        damping_movie=bias_config.get('damping_movie', 10)
    )
    bias_model.fit(data['explicit_train'])
    trained_models['bias'] = bias_model
    
    # 3. Bias + Genre Model
    logger.info("-" * 40)
    logger.info("Training: Bias + Genre Model")
    
    genre_config = models_config.get('genre', {})
    
    def bias_genre_cv(train_df, val_df):
        model = BiasGenreModel(
            movies_df=data['movies_enriched'],
            genre_cols=data['genre_cols'],
            damping_user=bias_config.get('damping_user', 25),
            damping_movie=bias_config.get('damping_movie', 10),
            genre_damping=genre_config.get('damping', 5),
            adjustment_factor=genre_config.get('adjustment_factor', 0.1)
        )
        model.fit(train_df)
        return model.predict(val_df)
    
    results['bias_genre'] = trainer.cross_validate(
        model_fn=bias_genre_cv,
        train_df=data['explicit_train'],
        wrmse_func=evaluator.weighted_rmse,
        movie_counts=data['movie_counts'],
        n_splits=n_splits
    )
    
    # Train final bias_genre model on full data
    bias_genre_model = BiasGenreModel(
        movies_df=data['movies_enriched'],
        genre_cols=data['genre_cols'],
        damping_user=bias_config.get('damping_user', 25),
        damping_movie=bias_config.get('damping_movie', 10),
        genre_damping=genre_config.get('damping', 5),
        adjustment_factor=genre_config.get('adjustment_factor', 0.1)
    )
    bias_genre_model.fit(data['explicit_train'])
    trained_models['bias_genre'] = bias_genre_model
    
    # Save models and results
    save_models(trained_models, results, data, config, logger)
    
    return {
        'results': results,
        'trained_models': trained_models,
        'evaluator': evaluator,
        'trainer': trainer
    }


def run_cold_start_evaluation(
    config: Dict[str, Any],
    data: Dict,
    training_output: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Execute cold start evaluation phase.
    
    Args:
        config: Configuration dictionary.
        data: Dictionary containing preprocessed data.
        training_output: Dictionary from training phase.
        logger: Logger instance.
        
    Returns:
        Cold start evaluation results.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Cold Start Evaluation")
    logger.info("=" * 60)
    
    cold_start_config = config.get('cold_start', {})
    
    # Initialize content-based model
    content_model = ColdStartContentModel(
        movies_df=data['movies_enriched'],
        genre_cols=data['genre_cols'],
        tfidf_cols=data.get('tfidf_cols', []),
        n_similar=cold_start_config.get('n_similar', 20)
    )
    
    # Run evaluation
    evaluator = training_output['evaluator']
    cold_start_results = evaluator.evaluate_cold_start(
        explicit_df=data['explicit_train'],
        content_model=content_model,
        n_cold_movies=cold_start_config.get('n_cold_movies_for_evaluation', 50),
        min_ratings=cold_start_config.get('min_ratings_for_simulation', 50),
        seed=config.get('seed', 42)
    )
    
    return cold_start_results


def run_submission(
    config: Dict[str, Any],
    data: Dict,
    training_output: Dict,
    logger: logging.Logger
) -> None:
    """
    Generate submission file with best model predictions.
    
    Args:
        config: Configuration dictionary.
        data: Dictionary containing preprocessed data.
        training_output: Dictionary from training phase.
        logger: Logger instance.
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: Submission Generation")
    logger.info("=" * 60)
    
    submission_config = config.get('submission', {})
    
    # Find best model based on CV results
    results = training_output['results']
    best_model_name = min(results.keys(), key=lambda x: results[x]['mean'])
    best_model = training_output['trained_models'].get(best_model_name)
    
    if best_model is None:
        logger.warning(f"Best model '{best_model_name}' not available, using bias model")
        best_model = training_output['trained_models']['bias']
        best_model_name = 'bias'
    
    logger.info(f"Using best model: {best_model_name} (W-RMSE: {results[best_model_name]['mean']:.4f})")
    
    # Generate predictions for submission
    submission_df = data['submission']
    predictions = best_model.predict(submission_df)
    
    # Create submission file
    evaluator = training_output['evaluator']
    evaluator.create_submission(
        test_df=submission_df,
        predictions=predictions,
        output_file=submission_config.get('output_file', 'submission.csv'),
        clip_min=submission_config.get('clip_min', 0.5),
        clip_max=submission_config.get('clip_max', 5.0),
        decimal_places=submission_config.get('decimal_places', 4)
    )


def run_full_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Execute the complete recommendation pipeline.
    
    Args:
        config: Configuration dictionary.
        logger: Logger instance.
    """
    logger.info("=" * 60)
    logger.info("MovieLens Recommendation System Pipeline")
    logger.info("=" * 60)
    
    # Phase 1: Data Loading
    data = run_data_loading(config, logger)
    
    # Phase 2: Preprocessing
    data = run_preprocessing(config, data, logger)
    
    # Phase 3: Training
    training_output = run_training(config, data, logger)
    
    # Analyze weight distribution
    evaluator = training_output['evaluator']
    evaluator.analyze_weight_distribution(data['explicit_train'])
    
    # Phase 4: Cold Start Evaluation
    cold_start_results = run_cold_start_evaluation(config, data, training_output, logger)
    
    # Compare models
    evaluator.compare_models(training_output['results'])
    
    # Generate report
    evaluator.generate_report(
        model_results=training_output['results'],
        cold_start_results=cold_start_results
    )
    
    # Phase 5: Submission
    run_submission(config, data, training_output, logger)
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="MovieLens Recommendation System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --config config/config.yaml
    python main.py --config config/config.yaml --mode train
    python main.py --config config/config.yaml --mode evaluate
    python main.py --config config/config.yaml --mode submit
    python main.py --config config/config.yaml --mode submit --load-models
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'train', 'evaluate', 'submit'],
        default='full',
        help='Pipeline mode: full, train, evaluate, or submit (default: full)'
    )
    
    parser.add_argument(
        '--load-models',
        action='store_true',
        help='Load pre-trained models instead of training (for evaluate/submit modes)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Running in mode: {args.mode}")
    if args.load_models:
        logger.info("Will attempt to load pre-trained models")
    
    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    try:
        if args.mode == 'full':
            run_full_pipeline(config, logger)
        elif args.mode == 'train':
            data = run_data_loading(config, logger)
            data = run_preprocessing(config, data, logger)
            run_training(config, data, logger)
        elif args.mode == 'evaluate':
            data = run_data_loading(config, logger)
            data = run_preprocessing(config, data, logger)
            
            if args.load_models:
                loaded = load_models(config, logger)
                if loaded:
                    training_output = {
                        'results': loaded['results'],
                        'trained_models': loaded['trained_models'],
                        'evaluator': Evaluator(
                            movie_counts=data['movie_counts'],
                            output_dir=config.get('data', {}).get('output_dir', 'outputs')
                        ),
                        'trainer': None
                    }
                else:
                    logger.warning("No saved models found, training from scratch")
                    training_output = run_training(config, data, logger)
            else:
                training_output = run_training(config, data, logger)
            
            run_cold_start_evaluation(config, data, training_output, logger)
            
        elif args.mode == 'submit':
            data = run_data_loading(config, logger)
            data = run_preprocessing(config, data, logger)
            
            if args.load_models:
                loaded = load_models(config, logger)
                if loaded:
                    training_output = {
                        'results': loaded['results'],
                        'trained_models': loaded['trained_models'],
                        'evaluator': Evaluator(
                            movie_counts=data['movie_counts'],
                            output_dir=config.get('data', {}).get('output_dir', 'outputs')
                        ),
                        'trainer': None
                    }
                else:
                    logger.warning("No saved models found, training from scratch")
                    training_output = run_training(config, data, logger)
            else:
                training_output = run_training(config, data, logger)
            
            run_submission(config, data, training_output, logger)
            
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

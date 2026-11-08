"""
Evaluation Module
=================

This module provides evaluation metrics and analysis tools for the
MovieLens recommendation system.

Key functionality:
- Weighted RMSE calculation (matching Kaggle competition metric)
- Weight distribution analysis and visualization
- Cold start evaluation framework
- Submission file generation

Classes:
    Evaluator: Main class for evaluation and metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles evaluation metrics and analysis for recommendations.
    
    The primary metric is Weighted RMSE where w_i = 1/n_i
    (n_i = number of ratings for movie i). This weights niche
    movies more heavily than popular ones.
    
    Attributes:
        movie_counts (Dict): Movie ID to rating count mapping.
        output_dir (Path): Directory for saving outputs.
    
    Example:
        >>> evaluator = Evaluator(movie_counts)
        >>> score = evaluator.weighted_rmse(y_true, y_pred, movie_ids)
    """
    
    def __init__(
        self,
        movie_counts: Optional[Dict[int, int]] = None,
        output_dir: str = "outputs"
    ) -> None:
        """
        Initialize the Evaluator.
        
        Args:
            movie_counts: Dictionary mapping movie_id to rating count.
            output_dir: Directory for saving plots and outputs.
        """
        self.movie_counts = movie_counts or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluator initialized with {len(self.movie_counts)} movie counts")
    
    def set_movie_counts(self, movie_counts: Dict[int, int]) -> None:
        """
        Update the movie counts dictionary.
        
        Args:
            movie_counts: Dictionary mapping movie_id to rating count.
        """
        self.movie_counts = movie_counts
        logger.info(f"Updated movie counts: {len(movie_counts)} movies")
    
    def weighted_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        movie_ids: np.ndarray,
        movie_rating_counts: Optional[Dict[int, int]] = None
    ) -> float:
        """
        Calculate Weighted RMSE matching Kaggle's evaluation metric.
        
        Formula: sqrt(sum(w_i * (y_true - y_pred)^2) / sum(w_i))
        where w_i = 1 / n_i (n_i = number of ratings for movie i)
        
        Niche movies (low n_i) have HIGH weight → errors matter MORE
        Popular movies (high n_i) have LOW weight → errors matter LESS
        
        Args:
            y_true: Array of true ratings.
            y_pred: Array of predicted ratings.
            movie_ids: Array of movie IDs for each prediction.
            movie_rating_counts: Optional override for movie counts.
            
        Returns:
            Weighted RMSE score.
            
        Example:
            >>> score = evaluator.weighted_rmse(
            ...     y_true=[4.0, 3.5],
            ...     y_pred=[3.8, 3.2],
            ...     movie_ids=[123, 456]
            ... )
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        movie_ids = np.array(movie_ids)
        
        counts = movie_rating_counts or self.movie_counts
        
        # Calculate weights: w_i = 1 / n_i
        weights = np.array([1.0 / counts.get(mid, 1) for mid in movie_ids])
        
        # Squared errors
        squared_errors = (y_true - y_pred) ** 2
        
        # Weighted mean squared error
        wmse = np.sum(weights * squared_errors) / np.sum(weights)
        
        return np.sqrt(wmse)
    
    def standard_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate standard (unweighted) RMSE for comparison.
        
        Args:
            y_true: Array of true ratings.
            y_pred: Array of predicted ratings.
            
        Returns:
            Standard RMSE score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = np.mean((y_true - y_pred) ** 2)
        return np.sqrt(mse)
    
    def analyze_weight_distribution(
        self,
        explicit_df: pd.DataFrame,
        save_plot: bool = True
    ) -> pd.DataFrame:
        """
        Analyze and visualize the W-RMSE weight distribution.
        
        This reveals how much niche vs popular movies contribute
        to the final W-RMSE score.
        
        Args:
            explicit_df: DataFrame with explicit ratings.
            save_plot: Whether to save visualization to file.
            
        Returns:
            DataFrame with movie weight statistics.
        """
        logger.info("Analyzing W-RMSE weight distribution")
        
        # Calculate movie rating counts
        movie_counts = explicit_df.groupby('movie_id').size().reset_index(name='n_ratings')
        
        # Calculate weights: w_i = 1 / n_i
        movie_counts['weight'] = 1.0 / movie_counts['n_ratings']
        
        # Normalize weights for interpretation
        movie_counts['weight_normalized'] = (
            movie_counts['weight'] / movie_counts['weight'].sum()
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Distribution of rating counts (log scale)
        axes[0].hist(
            movie_counts['n_ratings'],
            bins=100,
            color='steelblue',
            edgecolor='white'
        )
        axes[0].set_xlabel('Number of Ratings per Movie')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Rating Count Distribution')
        axes[0].set_yscale('log')
        
        # 2. Weight distribution
        axes[1].hist(
            movie_counts['weight'],
            bins=100,
            color='coral',
            edgecolor='white'
        )
        axes[1].set_xlabel('Weight (1/n_i)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('W-RMSE Weight Distribution')
        axes[1].set_yscale('log')
        
        # 3. Cumulative weight contribution
        movie_counts_sorted = movie_counts.sort_values('n_ratings')
        movie_counts_sorted['cumulative_weight'] = (
            movie_counts_sorted['weight_normalized'].cumsum()
        )
        axes[2].plot(
            range(len(movie_counts_sorted)),
            movie_counts_sorted['cumulative_weight'],
            color='green'
        )
        axes[2].set_xlabel('Movies (sorted by popularity)')
        axes[2].set_ylabel('Cumulative Weight')
        axes[2].set_title('Niche Movies Dominate W-RMSE')
        axes[2].axhline(y=0.5, color='red', linestyle='--', label='50% weight')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'weight_distribution.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved weight distribution plot to {plot_path}")
        
        plt.close()
        
        # Log key statistics
        low_count_pct = (movie_counts['n_ratings'] <= 10).mean() * 100
        low_count_weight = movie_counts[
            movie_counts['n_ratings'] <= 10
        ]['weight_normalized'].sum() * 100
        
        top_1pct = movie_counts.nlargest(
            int(len(movie_counts) * 0.01), 'n_ratings'
        )['weight_normalized'].sum() * 100
        
        logger.info(f"Movies with 1-10 ratings: {low_count_pct:.1f}%")
        logger.info(f"Their weight contribution: {low_count_weight:.1f}%")
        logger.info(f"Top 1% popular movies contribute only: {top_1pct:.2f}%")
        
        return movie_counts
    
    def evaluate_cold_start(
        self,
        explicit_df: pd.DataFrame,
        content_model: 'ColdStartContentModel',
        n_cold_movies: int = 50,
        min_ratings: int = 50,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate cold-start strategy by simulating new items.
        
        Experiment Design:
        1. Select movies with >= min_ratings (enough data for evaluation)
        2. Hold out n_cold_movies as "new" items
        3. Train content-based model on remaining movies
        4. Predict ratings for "new" items
        5. Compare W-RMSE: Content-based vs. Global Mean baseline
        
        Args:
            explicit_df: DataFrame with explicit ratings.
            content_model: ColdStartContentModel instance.
            n_cold_movies: Number of movies to simulate as cold start.
            min_ratings: Minimum ratings required for a movie to be eligible.
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with evaluation results.
        """
        np.random.seed(seed)
        logger.info("Running cold-start evaluation experiment")
        
        # Find eligible movies for cold-start simulation
        movie_counts = explicit_df.groupby('movie_id').size()
        eligible_movies = movie_counts[movie_counts >= min_ratings].index.tolist()
        
        logger.info(f"Found {len(eligible_movies)} eligible movies")
        
        # Sample cold-start movies
        cold_movies = np.random.choice(
            eligible_movies,
            size=min(n_cold_movies, len(eligible_movies)),
            replace=False
        )
        
        # Split data
        cold_data = explicit_df[explicit_df['movie_id'].isin(cold_movies)]
        train_data = explicit_df[~explicit_df['movie_id'].isin(cold_movies)]
        
        logger.info(f"Cold items: {len(cold_movies)} movies, {len(cold_data):,} ratings")
        logger.info(f"Training data: {len(train_data):,} ratings")
        
        # Fit content model on training data
        content_model.fit(train_data)
        
        # Generate predictions for cold items
        predictions = content_model.predict(cold_data)
        actuals = cold_data['rating'].values
        movie_ids_for_weights = cold_data['movie_id'].values
        
        # Calculate cold-start movie counts (for weighting)
        cold_movie_counts = cold_data.groupby('movie_id').size().to_dict()
        
        # Calculate W-RMSE for content-based model
        content_wrmse = self.weighted_rmse(
            actuals, predictions, movie_ids_for_weights, cold_movie_counts
        )
        
        # Baseline: Global mean
        global_mean = train_data['rating'].mean()
        baseline_preds = np.full(len(actuals), global_mean)
        baseline_wrmse = self.weighted_rmse(
            actuals, baseline_preds, movie_ids_for_weights, cold_movie_counts
        )
        
        improvement = (baseline_wrmse - content_wrmse) / baseline_wrmse * 100
        
        logger.info(f"Content-Based W-RMSE: {content_wrmse:.4f}")
        logger.info(f"Global Mean W-RMSE: {baseline_wrmse:.4f}")
        logger.info(f"Improvement: {improvement:.1f}%")
        
        return {
            'content_wrmse': content_wrmse,
            'baseline_wrmse': baseline_wrmse,
            'improvement_pct': improvement,
            'cold_movies': cold_movies.tolist(),
            'n_cold_ratings': len(cold_data)
        }
    
    def create_submission(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        output_file: str = "submission.csv",
        clip_min: float = 0.5,
        clip_max: float = 5.0,
        decimal_places: int = 4
    ) -> pd.DataFrame:
        """
        Create Kaggle submission file.
        
        Format: id (user_id_movie_id), prediction
        
        Args:
            test_df: Test DataFrame with 'id' or 'user_id'/'movie_id' columns.
            predictions: Array of predicted ratings.
            output_file: Output filename.
            clip_min: Minimum rating value.
            clip_max: Maximum rating value.
            decimal_places: Decimal places for rounding.
            
        Returns:
            Submission DataFrame.
        """
        output_path = self.output_dir / output_file
        
        # Use original 'id' column if available, otherwise construct from user_id/movie_id
        if 'id' in test_df.columns:
            submission = pd.DataFrame({
                'id': test_df['id'],
                'prediction': predictions
            })
        else:
            submission = pd.DataFrame({
                'id': test_df['user_id'].astype(str) + '_' + test_df['movie_id'].astype(str),
                'prediction': predictions
            })
        
        # Clip predictions to valid range
        submission['prediction'] = submission['prediction'].clip(clip_min, clip_max)
        
        # Round to specified precision
        submission['prediction'] = submission['prediction'].round(decimal_places)
        
        submission.to_csv(output_path, index=False)
        
        logger.info(f"Submission saved: {output_path}")
        logger.info(f"Rows: {len(submission):,}")
        logger.info(
            f"Prediction range: [{submission['prediction'].min():.2f}, "
            f"{submission['prediction'].max():.2f}]"
        )
        
        return submission
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        save_plot: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple model results and visualize.
        
        Args:
            results_dict: Dictionary of model name to CV results.
            save_plot: Whether to save comparison plot.
            
        Returns:
            DataFrame with model comparison.
        """
        comparison = []
        
        for model_name, results in results_dict.items():
            comparison.append({
                'Model': model_name,
                'Mean W-RMSE': results.get('mean', 0),
                'Std W-RMSE': results.get('std', 0),
                'Min Fold': min(results.get('fold_scores', [0])),
                'Max Fold': max(results.get('fold_scores', [0]))
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('Mean W-RMSE')
        
        if save_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = comparison_df['Model']
            means = comparison_df['Mean W-RMSE']
            stds = comparison_df['Std W-RMSE']
            
            colors = sns.color_palette('viridis', len(models))
            bars = ax.barh(models, means, xerr=stds, color=colors, edgecolor='white')
            
            ax.set_xlabel('W-RMSE')
            ax.set_title('Model Comparison (Lower is Better)')
            ax.axvline(x=0.9446, color='red', linestyle='--', label='Baseline (0.9446)')
            ax.legend()
            
            plt.tight_layout()
            
            plot_path = self.output_dir / 'model_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {plot_path}")
            
            plt.close()
        
        logger.info(f"Model Comparison:\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def generate_report(
        self,
        model_results: Dict[str, Dict],
        cold_start_results: Optional[Dict] = None,
        output_file: str = "evaluation_report.txt"
    ) -> str:
        """
        Generate a text report of all evaluation results.
        
        Args:
            model_results: Dictionary of model results.
            cold_start_results: Optional cold start evaluation results.
            output_file: Output filename.
            
        Returns:
            Report text.
        """
        report_lines = [
            "=" * 60,
            "MovieLens Recommendation System - Evaluation Report",
            "=" * 60,
            "",
            "MODEL PERFORMANCE (Cross-Validation)",
            "-" * 40,
        ]
        
        for model_name, results in sorted(
            model_results.items(),
            key=lambda x: x[1].get('mean', float('inf'))
        ):
            report_lines.append(
                f"  {model_name}: W-RMSE = {results.get('mean', 0):.4f} "
                f"± {results.get('std', 0):.4f}"
            )
        
        report_lines.extend(["", "BASELINE COMPARISON", "-" * 40])
        report_lines.append("  Target: Beat 0.9446 W-RMSE")
        
        best_model = min(model_results.items(), key=lambda x: x[1].get('mean', float('inf')))
        best_score = best_model[1].get('mean', 0)
        
        if best_score < 0.9446:
            report_lines.append(f"  ✓ PASSED - Best model ({best_model[0]}): {best_score:.4f}")
        else:
            report_lines.append(f"  ✗ NOT PASSED - Best model ({best_model[0]}): {best_score:.4f}")
        
        if cold_start_results:
            report_lines.extend([
                "",
                "COLD START EVALUATION",
                "-" * 40,
                f"  Content-Based W-RMSE: {cold_start_results.get('content_wrmse', 0):.4f}",
                f"  Global Mean W-RMSE: {cold_start_results.get('baseline_wrmse', 0):.4f}",
                f"  Improvement: {cold_start_results.get('improvement_pct', 0):.1f}%",
                f"  Cold Movies Tested: {len(cold_start_results.get('cold_movies', []))}",
            ])
        
        report_lines.extend(["", "=" * 60])
        
        report_text = "\n".join(report_lines)
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text

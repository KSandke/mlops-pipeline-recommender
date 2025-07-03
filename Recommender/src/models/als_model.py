"""
ALS (Alternating Least Squares) model implementation.
This adapts the existing ALS model to the standardized interface.
"""

import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..core.interfaces import BaseRecommendationModel, RecommendationResult, ModelMetadata
from ..core.exceptions import (
    ModelNotLoadedError, 
    UserNotFoundError, 
    ItemNotFoundError,
    ModelLoadError,
    PredictionError
)

logger = logging.getLogger(__name__)


class ALSRecommendationModel(BaseRecommendationModel):
    """
    ALS model implementation following the standard interface.
    """
    
    def __init__(self, model_id: str = "als_model"):
        super().__init__(model_id)
        self.model = None
        self.user_map = None
        self.movie_map = None
        self.interaction_matrix = None
        self.movies_df = None
        self.ratings_stats = None
        
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load ALS model artifacts from disk.
        
        Args:
            model_path: Base path to model artifacts
            **kwargs: Additional parameters (e.g., movies_metadata_path)
        """
        try:
            logger.info(f"Loading ALS model from {model_path}")
            
            # Load model artifacts
            self.model = joblib.load(f"{model_path}/als_model.joblib")
            self.user_map = joblib.load(f"{model_path}/user_map.joblib")
            self.movie_map = joblib.load(f"{model_path}/movie_map.joblib")
            self.interaction_matrix = joblib.load(f"{model_path}/interaction_matrix.joblib")
            
            # Load movie metadata
            movies_path = kwargs.get('movies_metadata_path', 'Recommender/data/raw/movies.csv')
            self.movies_df = pd.read_csv(movies_path)
            
            # Extract year from title and preprocess
            self._preprocess_movie_metadata()
            
            # Load and calculate rating statistics
            ratings_path = kwargs.get('ratings_path', 'Recommender/data/raw/ratings.csv')
            self._load_rating_statistics(ratings_path)
            
            # Create metadata
            self._metadata = ModelMetadata(
                model_id=self.model_id,
                model_type="ALS",
                version=kwargs.get('version', '1.0.0'),
                trained_at=datetime.fromisoformat(kwargs.get('trained_at', '2025-07-01T10:00:00')),
                metrics={
                    'precision_at_10': 0.0144,
                    'recall_at_10': 0.1436,
                    'map_at_10': 0.0499
                },
                hyperparameters={
                    'factors': 50,
                    'regularization': 0.01,
                    'iterations': 20
                },
                data_version=kwargs.get('data_version', 'ml-latest-2023'),
                mlflow_run_id=kwargs.get('mlflow_run_id')
            )
            
            self._is_loaded = True
            logger.info(f"Successfully loaded ALS model with {len(self.user_map)} users and {len(self.movie_map)} items")
            
        except Exception as e:
            logger.error(f"Failed to load ALS model: {str(e)}")
            raise ModelLoadError(self.model_id, str(e))
    
    def predict_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Generate recommendations for a specific user.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
            
        if user_id not in self.user_map:
            raise UserNotFoundError(user_id)
        
        try:
            # Get user index
            user_idx = self.user_map.get_loc(user_id)
            
            # Generate recommendations
            recommended_items, scores = self.model.recommend(
                user_idx, 
                self.interaction_matrix[user_idx],
                N=n_recommendations * 3,  # Get extra to account for filtering
                filter_already_liked_items=True
            )
            
            # Convert to movie IDs
            recommended_movie_ids = [self.movie_map[idx] for idx in recommended_items]
            
            # Apply filters
            results = []
            for movie_id, score in zip(recommended_movie_ids, scores):
                if exclude_items and movie_id in exclude_items:
                    continue
                    
                if filter_criteria and not self._passes_filter(movie_id, filter_criteria):
                    continue
                
                results.append(RecommendationResult(
                    item_id=movie_id,
                    score=float(score),
                    rank=len(results) + 1,
                    metadata=self._get_movie_metadata(movie_id)
                ))
                
                if len(results) >= n_recommendations:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for user {user_id}: {str(e)}")
            raise PredictionError(self.model_id, str(e))
    
    def predict_similar_items(
        self,
        item_id: int,
        n_similar: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Find items similar to a given item.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
            
        if item_id not in self.movie_map:
            raise ItemNotFoundError(item_id)
        
        try:
            # Get item index
            item_idx = self.movie_map.get_loc(item_id)
            
            # Find similar items
            similar_items, scores = self.model.similar_items(
                item_idx, 
                N=n_similar * 2 + 1  # Extra for filtering + original item
            )
            
            # Skip the first item (itself)
            similar_items = similar_items[1:]
            scores = scores[1:]
            
            # Convert to movie IDs and create results
            results = []
            for idx, score in zip(similar_items, scores):
                movie_id = self.movie_map[idx]
                
                if filter_criteria and not self._passes_filter(movie_id, filter_criteria):
                    continue
                
                results.append(RecommendationResult(
                    item_id=movie_id,
                    score=float(score),
                    rank=len(results) + 1,
                    metadata=self._get_movie_metadata(movie_id)
                ))
                
                if len(results) >= n_similar:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Similar items prediction failed for item {item_id}: {str(e)}")
            raise PredictionError(self.model_id, str(e))
    
    def predict_batch(
        self,
        user_ids: List[int],
        n_recommendations: int = 10,
        **kwargs
    ) -> Dict[int, List[RecommendationResult]]:
        """
        Generate recommendations for multiple users.
        """
        results = {}
        
        for user_id in user_ids:
            try:
                results[user_id] = self.predict_for_user(
                    user_id, 
                    n_recommendations,
                    **kwargs
                )
            except UserNotFoundError:
                # Skip users not in training data
                results[user_id] = []
                logger.warning(f"User {user_id} not found in training data")
            except Exception as e:
                logger.error(f"Batch prediction failed for user {user_id}: {str(e)}")
                results[user_id] = []
        
        return results
    
    def get_model_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        if not self._metadata:
            raise ModelNotLoadedError(self.model_id)
        return self._metadata
    
    def get_user_embeddings(self, user_id: int) -> Optional[np.ndarray]:
        """Get user factors from ALS model."""
        if not self._is_loaded:
            return None
            
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map.get_loc(user_id)
        return self.model.user_factors[user_idx]
    
    def get_item_embeddings(self, item_id: int) -> Optional[np.ndarray]:
        """Get item factors from ALS model."""
        if not self._is_loaded:
            return None
            
        if item_id not in self.movie_map:
            return None
        
        item_idx = self.movie_map.get_loc(item_id)
        return self.model.item_factors[item_idx]
    
    def _passes_filter(self, movie_id: int, filter_criteria: Dict[str, Any]) -> bool:
        """Check if a movie passes the filter criteria."""
        movie_data = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if movie_data.empty:
            return False
        
        movie = movie_data.iloc[0]
        
        # Genre filter
        if 'genres' in filter_criteria and filter_criteria['genres']:
            movie_genres = movie['genres'].split('|') if pd.notna(movie['genres']) else []
            if not any(genre in movie_genres for genre in filter_criteria['genres']):
                return False
        
        # Year filter
        if 'year' in movie and pd.notna(movie['year']):
            year = int(movie['year'])
            if 'min_year' in filter_criteria and filter_criteria['min_year'] and year < filter_criteria['min_year']:
                return False
            if 'max_year' in filter_criteria and filter_criteria['max_year'] and year > filter_criteria['max_year']:
                return False
        
        return True
    
    def _preprocess_movie_metadata(self):
        """Extract year from title and preprocess movie metadata."""
        import re
        
        # Extract year from title (e.g., "Toy Story (1995)" -> 1995)
        def extract_year(title):
            match = re.search(r'\((\d{4})\)', title)
            return int(match.group(1)) if match else None
        
        self.movies_df['year'] = self.movies_df['title'].apply(extract_year)
        
        # Clean title by removing year
        def clean_title(title):
            return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        
        self.movies_df['clean_title'] = self.movies_df['title'].apply(clean_title)
    
    def _load_rating_statistics(self, ratings_path: str):
        """Load and calculate rating statistics for movies."""
        try:
            # Load ratings data (sample for efficiency if file is large)
            ratings_df = pd.read_csv(ratings_path)
            
            # Calculate statistics per movie
            self.ratings_stats = ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).round(2)
            
            # Flatten column names
            self.ratings_stats.columns = ['avg_rating', 'total_ratings']
            self.ratings_stats = self.ratings_stats.reset_index()
            
            logger.info(f"Loaded rating statistics for {len(self.ratings_stats)} movies")
            
        except Exception as e:
            logger.warning(f"Failed to load rating statistics: {str(e)}")
            self.ratings_stats = pd.DataFrame(columns=['movieId', 'avg_rating', 'total_ratings'])
    
    def _get_movie_metadata(self, movie_id: int) -> Dict[str, Any]:
        """Get movie metadata."""
        movie_data = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if movie_data.empty:
            return {'movie_id': movie_id}
        
        movie = movie_data.iloc[0]
        
        # Get rating statistics
        rating_stats = self.ratings_stats[self.ratings_stats['movieId'] == movie_id]
        avg_rating = rating_stats['avg_rating'].iloc[0] if not rating_stats.empty else None
        total_ratings = int(rating_stats['total_ratings'].iloc[0]) if not rating_stats.empty else None
        
        return {
            'title': movie['clean_title'],
            'genres': movie['genres'].split('|') if pd.notna(movie['genres']) else [],
            'year': int(movie['year']) if pd.notna(movie['year']) else None,
            'avg_rating': float(avg_rating) if avg_rating is not None else None,
            'total_ratings': total_ratings
        } 
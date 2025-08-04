"""
Neural Collaborative Filtering model implementation.
Implements the BaseRecommendationModel interface for FastAPI integration.
"""

import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..core.interfaces import BaseRecommendationModel, ModelMetadata, RecommendationResult
from ..core.exceptions import ModelNotLoadedError, UserNotFoundError, ItemNotFoundError
from ..train_neural import NeuralCollaborativeFiltering


class NeuralRecommendationModel(BaseRecommendationModel):
    """
    Neural Collaborative Filtering model for movie recommendations.
    Implements the BaseRecommendationModel interface for FastAPI integration.
    """
    
    def __init__(self, model_id: str = "neural_prod_v1"):
        self.model_id = model_id
        self.model_type = "neural_collaborative_filtering"
        self._is_loaded = False
        self._model: Optional[NeuralCollaborativeFiltering] = None
        self._encoders: Optional[Dict] = None
        self._movies_df: Optional[pd.DataFrame] = None
        self._metadata: Optional[ModelMetadata] = None
        
    def load(self, model_path: str, version: str = "1.0.0", trained_at: str = None) -> None:
        """
        Load the neural model and encoders.
        
        Args:
            model_path: Path to model artifacts
            version: Model version
            trained_at: Training timestamp
        """
        try:
            print(f"Loading neural model from {model_path}...")
            
            # Load encoders
            encoders_path = os.path.join(model_path, 'neural_encoders.joblib')
            if not os.path.exists(encoders_path):
                raise FileNotFoundError(f"Encoders not found at {encoders_path}")
            
            self._encoders = joblib.load(encoders_path)
            
            # Load model weights
            model_path_pth = os.path.join(model_path, 'ncf_model.pth')
            if not os.path.exists(model_path_pth):
                raise FileNotFoundError(f"Model weights not found at {model_path_pth}")
            
            # Get model dimensions from encoders
            num_users = len(self._encoders['user'].classes_)
            num_movies = len(self._encoders['movie'].classes_)
            
            # Initialize model with same architecture
            self._model = NeuralCollaborativeFiltering(
                num_users=num_users,
                num_movies=num_movies,
                embedding_dim=64,  # Default, will be overridden by loaded weights
                layers=[128, 64, 32],
                dropout=0.1
            )
            
            # Load trained weights
            model_state = torch.load(model_path_pth, map_location='cpu')
            self._model.load_state_dict(model_state)
            self._model.eval()
            
            # Load movie metadata
            movies_path = os.path.join(model_path.replace('models', 'data/raw'), 'movies.csv')
            if os.path.exists(movies_path):
                self._movies_df = pd.read_csv(movies_path)
            else:
                print(f"Warning: Movie metadata not found at {movies_path}")
            
            # Set metadata
            self._metadata = ModelMetadata(
                model_id=self.model_id,
                model_type=self.model_type,
                version=version,
                trained_at=datetime.fromisoformat(trained_at) if trained_at else datetime.now(),
                metrics={
                    "num_users": num_users,
                    "num_movies": num_movies,
                    "embedding_dim": 64
                }
            )
            
            self._is_loaded = True
            print(f"Neural model loaded successfully! Users: {num_users}, Movies: {num_movies}")
            
        except Exception as e:
            self._is_loaded = False
            raise Exception(f"Failed to load neural model: {str(e)}")
    
    def _encode_user_id(self, user_id: int) -> Optional[int]:
        """Encode user ID to internal representation."""
        if not self._is_loaded or user_id not in self._encoders['user'].classes_:
            return None
        return self._encoders['user'].transform([user_id])[0]
    
    def _encode_movie_id(self, movie_id: int) -> Optional[int]:
        """Encode movie ID to internal representation."""
        if not self._is_loaded or movie_id not in self._encoders['movie'].classes_:
            return None
        return self._encoders['movie'].transform([movie_id])[0]
    
    def _get_movie_metadata(self, movie_id: int) -> Dict[str, Any]:
        """Get movie metadata."""
        if self._movies_df is None:
            return {"title": f"Movie {movie_id}", "genres": [], "year": None}
        
        movie_data = self._movies_df[self._movies_df['movieId'] == movie_id]
        if len(movie_data) == 0:
            return {"title": f"Movie {movie_id}", "genres": [], "year": None}
        
        row = movie_data.iloc[0]
        return {
            "title": row.get('title', f"Movie {movie_id}"),
            "genres": row.get('genres', []),
            "year": row.get('year'),
            "avg_rating": None,  # Could be calculated from ratings data
            "total_ratings": None
        }
    
    def predict_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_items: Items to exclude
            filter_criteria: Additional filtering criteria
            
        Returns:
            List of recommendation results
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
        
        user_encoded = self._encode_user_id(user_id)
        if user_encoded is None:
            raise UserNotFoundError(user_id)
        
        # Get user's rated movies if excluding already rated
        rated_movies = set()
        if exclude_items is None:
            try:
                train_df = pd.read_csv("Recommender/data/processed/train.csv")
                user_ratings = train_df[train_df['userId'] == user_id]
                rated_movies = set(user_ratings['movieId'].values)
            except:
                pass  # If we can't load training data, continue without filtering
        
        if exclude_items:
            rated_movies.update(exclude_items)
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in self._encoders['movie'].classes_:
            if movie_id in rated_movies:
                continue
                
            predicted_rating = self._predict_rating(user_id, movie_id)
            if predicted_rating is not None:
                predictions.append({
                    'movie_id': movie_id,
                    'predicted_rating': predicted_rating
                })
        
        # Sort by predicted rating and get top recommendations
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        top_recommendations = predictions[:n_recommendations]
        
        # Convert to RecommendationResult format
        results = []
        for rank, pred in enumerate(top_recommendations, 1):
            movie_id = pred['movie_id']
            metadata = self._get_movie_metadata(movie_id)
            
            results.append(RecommendationResult(
                item_id=movie_id,
                score=pred['predicted_rating'],
                rank=rank,
                metadata=metadata
            ))
        
        return results
    
    def _predict_rating(self, user_id: int, movie_id: int) -> Optional[float]:
        """Predict rating for a user-movie pair."""
        user_encoded = self._encode_user_id(user_id)
        movie_encoded = self._encode_movie_id(movie_id)
        
        if user_encoded is None or movie_encoded is None:
            return None
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_encoded])
            movie_tensor = torch.LongTensor([movie_encoded])
            
            prediction = self._model(user_tensor, movie_tensor)
            return prediction.item()
    
    def predict_similar_items(
        self, 
        item_id: int, 
        n_similar: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Find items similar to a given item using embedding similarity.
        
        Args:
            item_id: Item ID to find similar items for
            n_similar: Number of similar items
            filter_criteria: Additional filtering criteria
            
        Returns:
            List of similar items
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
        
        movie_encoded = self._encode_movie_id(item_id)
        if movie_encoded is None:
            raise ItemNotFoundError(item_id)
        
        # Get movie embedding
        with torch.no_grad():
            movie_tensor = torch.LongTensor([movie_encoded])
            movie_embedding = self._model.movie_embedding(movie_tensor)
        
        # Calculate similarities with all other movies
        similarities = []
        for other_movie_id in self._encoders['movie'].classes_:
            if other_movie_id == item_id:
                continue
                
            other_movie_encoded = self._encode_movie_id(other_movie_id)
            if other_movie_encoded is not None:
                other_movie_tensor = torch.LongTensor([other_movie_encoded])
                other_embedding = self._model.movie_embedding(other_movie_tensor)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(movie_embedding, other_embedding).item()
                similarities.append({
                    'movie_id': other_movie_id,
                    'similarity_score': similarity
                })
        
        # Sort by similarity and get top similar movies
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_similar = similarities[:n_similar]
        
        # Convert to RecommendationResult format
        results = []
        for rank, sim in enumerate(top_similar, 1):
            movie_id = sim['movie_id']
            metadata = self._get_movie_metadata(movie_id)
            
            results.append(RecommendationResult(
                item_id=movie_id,
                score=sim['similarity_score'],
                rank=rank,
                metadata=metadata
            ))
        
        return results
    
    def predict_batch(
        self, 
        user_ids: List[int], 
        n_recommendations: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[RecommendationResult]]:
        """
        Get recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            filter_criteria: Additional filtering criteria
            
        Returns:
            Dictionary mapping user IDs to recommendation lists
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
        
        results = {}
        for user_id in user_ids:
            try:
                user_recs = self.predict_for_user(
                    user_id=user_id,
                    n_recommendations=n_recommendations,
                    filter_criteria=filter_criteria
                )
                results[user_id] = user_recs
            except UserNotFoundError:
                # Skip users not found in training data
                results[user_id] = []
        
        return results
    
    def get_popular_items(
        self, 
        n_recommendations: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Get popular items as a fallback for new users.
        
        Args:
            n_recommendations: Number of recommendations
            filter_criteria: Additional filtering criteria
            
        Returns:
            List of popular items
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(self.model_id)
        
        # Simple popularity based on movie ID (could be enhanced with actual popularity metrics)
        popular_movies = list(self._encoders['movie'].classes_)[:n_recommendations]
        
        results = []
        for rank, movie_id in enumerate(popular_movies, 1):
            metadata = self._get_movie_metadata(movie_id)
            
            results.append(RecommendationResult(
                item_id=movie_id,
                score=1.0 - (rank / len(popular_movies)),  # Decreasing score
                rank=rank,
                metadata=metadata
            ))
        
        return results
    
    def get_model_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        if not self._is_loaded or self._metadata is None:
            raise ModelNotLoadedError(self.model_id)
        return self._metadata
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded 
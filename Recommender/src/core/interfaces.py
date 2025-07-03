"""
Core interfaces for the recommendation system.
These abstract base classes define contracts that all models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np


@dataclass
class RecommendationResult:
    """Standardized recommendation result."""
    item_id: int
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning."""
    model_id: str
    model_type: str
    version: str
    trained_at: datetime
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    data_version: str
    mlflow_run_id: Optional[str] = None


class BaseRecommendationModel(ABC):
    """
    Abstract base class for all recommendation models.
    This interface ensures all models are compatible with the API.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._metadata: Optional[ModelMetadata] = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """
        Load model from disk or remote storage.
        
        Args:
            model_path: Path to model artifacts
            **kwargs: Additional loading parameters
        """
        pass
    
    @abstractmethod
    def predict_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: User to generate recommendations for
            n_recommendations: Number of items to recommend
            exclude_items: Items to exclude from recommendations
            filter_criteria: Additional filtering criteria (e.g., genre, year)
            
        Returns:
            List of RecommendationResult objects
        """
        pass
    
    @abstractmethod
    def predict_similar_items(
        self,
        item_id: int,
        n_similar: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Item to find similar items for
            n_similar: Number of similar items to return
            filter_criteria: Additional filtering criteria
            
        Returns:
            List of RecommendationResult objects
        """
        pass
    
    @abstractmethod
    def predict_batch(
        self,
        user_ids: List[int],
        n_recommendations: int = 10,
        **kwargs
    ) -> Dict[int, List[RecommendationResult]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping user IDs to their recommendations
        """
        pass
    
    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata:
        """
        Get model metadata including version, metrics, etc.
        
        Returns:
            ModelMetadata object
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform model health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "model_id": self.model_id,
            "is_loaded": self._is_loaded,
            "status": "healthy" if self._is_loaded else "not_loaded",
            "metadata": self._metadata.__dict__ if self._metadata else None
        }
    
    @abstractmethod
    def get_user_embeddings(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embeddings/factors (if applicable).
        
        Args:
            user_id: User ID
            
        Returns:
            User embedding vector or None
        """
        pass
    
    @abstractmethod
    def get_item_embeddings(self, item_id: int) -> Optional[np.ndarray]:
        """
        Get item embeddings/factors (if applicable).
        
        Args:
            item_id: Item ID
            
        Returns:
            Item embedding vector or None
        """
        pass


class BaseModelRegistry(ABC):
    """Abstract base class for model registry implementations."""
    
    @abstractmethod
    def register_model(self, model: BaseRecommendationModel) -> None:
        """Register a model in the registry."""
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> BaseRecommendationModel:
        """Retrieve a model from the registry."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List all available model IDs."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> BaseRecommendationModel:
        """Get the default model for serving."""
        pass 
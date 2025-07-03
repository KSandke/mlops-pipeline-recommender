"""
Model registry for managing multiple recommendation models.
This implements the registry pattern for model management.
"""

from typing import Dict, List, Optional
import logging
from threading import RLock

from ..core.interfaces import BaseRecommendationModel, BaseModelRegistry
from ..core.exceptions import ModelNotFoundError, ConfigurationError

logger = logging.getLogger(__name__)


class ModelRegistry(BaseModelRegistry):
    """
    Thread-safe model registry implementation.
    Manages multiple models and handles model selection.
    """
    
    def __init__(self):
        self._models: Dict[str, BaseRecommendationModel] = {}
        self._default_model_id: Optional[str] = None
        self._lock = RLock()  # Thread-safe operations
        
    def register_model(self, model: BaseRecommendationModel) -> None:
        """
        Register a model in the registry.
        
        Args:
            model: Model instance to register
        """
        with self._lock:
            model_id = model.model_id
            
            if model_id in self._models:
                logger.warning(f"Overwriting existing model with ID: {model_id}")
            
            self._models[model_id] = model
            logger.info(f"Registered model: {model_id}")
            
            # Set as default if it's the first model
            if self._default_model_id is None:
                self._default_model_id = model_id
                logger.info(f"Set {model_id} as default model")
    
    def get_model(self, model_id: str) -> BaseRecommendationModel:
        """
        Retrieve a model from the registry.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model not found
        """
        with self._lock:
            if model_id not in self._models:
                available = list(self._models.keys())
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found. Available models: {available}"
                )
            
            return self._models[model_id]
    
    def list_models(self) -> List[str]:
        """
        List all available model IDs.
        
        Returns:
            List of model IDs
        """
        with self._lock:
            return list(self._models.keys())
    
    def get_default_model(self) -> BaseRecommendationModel:
        """
        Get the default model for serving.
        
        Returns:
            Default model instance
            
        Raises:
            ConfigurationError: If no default model is set
        """
        with self._lock:
            if self._default_model_id is None:
                raise ConfigurationError(
                    "default_model", 
                    "No default model configured"
                )
            
            return self.get_model(self._default_model_id)
    
    def set_default_model(self, model_id: str) -> None:
        """
        Set the default model.
        
        Args:
            model_id: ID of the model to set as default
            
        Raises:
            ModelNotFoundError: If model not found
        """
        with self._lock:
            # Verify model exists
            if model_id not in self._models:
                raise ModelNotFoundError(model_id)
            
            self._default_model_id = model_id
            logger.info(f"Set default model to: {model_id}")
    
    def unregister_model(self, model_id: str) -> None:
        """
        Remove a model from the registry.
        
        Args:
            model_id: ID of the model to remove
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                logger.info(f"Unregistered model: {model_id}")
                
                # Update default if necessary
                if self._default_model_id == model_id:
                    self._default_model_id = None
                    if self._models:
                        # Set first available model as default
                        self._default_model_id = next(iter(self._models))
                        logger.info(f"Updated default model to: {self._default_model_id}")
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about all registered models.
        
        Returns:
            Dictionary with model information
        """
        with self._lock:
            info = {}
            for model_id, model in self._models.items():
                try:
                    metadata = model.get_model_metadata()
                    info[model_id] = {
                        'model_id': model_id,
                        'model_type': metadata.model_type,
                        'version': metadata.version,
                        'is_loaded': model._is_loaded,
                        'is_default': model_id == self._default_model_id,
                        'metrics': metadata.metrics,
                        'trained_at': metadata.trained_at.isoformat()
                    }
                except Exception as e:
                    logger.error(f"Failed to get info for model {model_id}: {str(e)}")
                    info[model_id] = {
                        'model_id': model_id,
                        'error': str(e),
                        'is_default': model_id == self._default_model_id
                    }
            
            return info


# Global registry instance
_global_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _global_registry 
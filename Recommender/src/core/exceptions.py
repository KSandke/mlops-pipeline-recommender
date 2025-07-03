"""
Custom exceptions for the recommendation system.
These provide specific error types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class RecommenderException(Exception):
    """Base exception for all recommender-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(RecommenderException):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model with ID '{model_id}' not found",
            details={"model_id": model_id}
        )


class UserNotFoundError(RecommenderException):
    """Raised when a user is not found in the system."""
    
    def __init__(self, user_id: int):
        super().__init__(
            message=f"User with ID {user_id} not found",
            details={"user_id": user_id}
        )


class ItemNotFoundError(RecommenderException):
    """Raised when an item is not found in the system."""
    
    def __init__(self, item_id: int):
        super().__init__(
            message=f"Item with ID {item_id} not found",
            details={"item_id": item_id}
        )


class ModelNotLoadedError(RecommenderException):
    """Raised when attempting to use a model that hasn't been loaded."""
    
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' is not loaded. Please load the model first.",
            details={"model_id": model_id}
        )


class InsufficientDataError(RecommenderException):
    """Raised when there's insufficient data to generate recommendations."""
    
    def __init__(self, user_id: int, min_interactions: int, actual_interactions: int):
        super().__init__(
            message=f"User {user_id} has insufficient interaction data. "
                    f"Minimum required: {min_interactions}, actual: {actual_interactions}",
            details={
                "user_id": user_id,
                "min_interactions": min_interactions,
                "actual_interactions": actual_interactions
            }
        )


class ModelLoadError(RecommenderException):
    """Raised when a model fails to load."""
    
    def __init__(self, model_id: str, reason: str):
        super().__init__(
            message=f"Failed to load model '{model_id}': {reason}",
            details={"model_id": model_id, "reason": reason}
        )


class PredictionError(RecommenderException):
    """Raised when prediction fails."""
    
    def __init__(self, model_id: str, reason: str):
        super().__init__(
            message=f"Prediction failed for model '{model_id}': {reason}",
            details={"model_id": model_id, "reason": reason}
        )


class ConfigurationError(RecommenderException):
    """Raised when there's a configuration error."""
    
    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Configuration error for '{config_key}': {reason}",
            details={"config_key": config_key, "reason": reason}
        )


class CacheError(RecommenderException):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Cache {operation} failed: {reason}",
            details={"operation": operation, "reason": reason}
        )


class ValidationError(RecommenderException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Validation failed for field '{field}': {reason}",
            details={"field": field, "reason": reason}
        ) 
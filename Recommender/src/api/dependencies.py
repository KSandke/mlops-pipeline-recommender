"""
Dependency injection for the API.
Provides reusable dependencies for endpoints.
"""

from fastapi import HTTPException, Header
from typing import Optional

from ..models.registry import get_model_registry
from ..core.interfaces import BaseRecommendationModel
from .config import get_settings

settings = get_settings()


async def get_current_model(
    model_id: Optional[str] = None,
    x_model_id: Optional[str] = Header(None)
) -> BaseRecommendationModel:
    """
    Get the current model for serving predictions.
    
    Model selection priority:
    1. model_id parameter in request
    2. X-Model-ID header
    3. Default model from registry
    
    Args:
        model_id: Optional model ID from request
        x_model_id: Optional model ID from header
        
    Returns:
        Model instance
        
    Raises:
        HTTPException: If model not found or not loaded
    """
    registry = get_model_registry()
    
    # Determine which model to use
    selected_model_id = model_id or x_model_id
    
    try:
        if selected_model_id:
            model = registry.get_model(selected_model_id)
        else:
            model = registry.get_default_model()
            
        # Verify model is loaded
        if not model._is_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model.model_id}' is not loaded"
            )
            
        return model
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> bool:
    """
    Verify API key if authentication is enabled.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        True if valid or auth disabled
        
    Raises:
        HTTPException: If invalid API key
    """
    if not settings.API_KEY_ENABLED:
        return True
        
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
        
    if x_api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
        
    return True 
"""
Model management endpoints for the API.
Provides model information and management capabilities.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ...core.schemas import ModelInfo
from ...models.registry import get_model_registry
from ...core.exceptions import ModelNotFoundError

router = APIRouter()


@router.get("/", response_model=List[ModelInfo])
async def list_available_models():
    """
    List all available models.
    
    Returns information about all registered models including their
    status, version, and performance metrics.
    """
    registry = get_model_registry()
    return list(registry.get_model_info().values())


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_details(model_id: str):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: The ID of the model to retrieve information for
        
    Returns:
        Detailed model information including metrics and configuration
    """
    registry = get_model_registry()
    
    try:
        model = registry.get_model(model_id)
        metadata = model.get_model_metadata()
        
        return ModelInfo(
            model_id=model_id,
            model_type=metadata.model_type,
            version=metadata.version,
            status="active" if model._is_loaded else "not_loaded",
            trained_at=metadata.trained_at,
            metrics=metadata.metrics,
            hyperparameters=metadata.hyperparameters,
            is_default=model_id == registry._default_model_id
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_id}/set-default")
async def set_default_model(model_id: str):
    """
    Set a model as the default for serving.
    
    Args:
        model_id: The ID of the model to set as default
        
    Returns:
        Success message
    """
    registry = get_model_registry()
    
    try:
        registry.set_default_model(model_id)
        return {"message": f"Model '{model_id}' set as default"}
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.get("/{model_id}/health")
async def model_health_check(model_id: str):
    """
    Check the health status of a specific model.
    
    Args:
        model_id: The ID of the model to check
        
    Returns:
        Model health status
    """
    registry = get_model_registry()
    
    try:
        model = registry.get_model(model_id)
        return model.health_check()
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found") 
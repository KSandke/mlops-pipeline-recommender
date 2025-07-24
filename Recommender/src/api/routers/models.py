"""
Model management endpoints for the API.
Provides model information and management capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict

from ...core.schemas import ModelInfo, HealthStatus
from ...models.registry import get_model_registry
from ...core.exceptions import ModelNotFoundError

router = APIRouter()


@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models.
    
    Returns information about all registered models including their
    status, version, and performance metrics.
    """
    registry = get_model_registry()
    model_info_dict = registry.get_model_info()
    
    models = []
    for model_id, info in model_info_dict.items():
        if 'error' not in info:
            models.append(ModelInfo(
                model_id=info['model_id'],
                model_type=ModelType(info['model_type'].lower()),
                version=info['version'],
                status="active" if info['is_loaded'] else "not_loaded",
                trained_at=info['trained_at'],
                metrics=info['metrics'],
                hyperparameters={},  # Not exposing full hyperparameters
                is_default=info['is_default']
            ))
    
    return models


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
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
            model_type=ModelType(metadata.model_type.lower()),
            version=metadata.version,
            status="active" if model._is_loaded else "not_loaded",
            trained_at=metadata.trained_at,
            metrics=metadata.metrics,
            hyperparameters=metadata.hyperparameters,
            is_default=model_id == registry._default_model_id
        )
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


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
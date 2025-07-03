"""
Health check endpoints for the API.
Provides system status and monitoring information.
"""

from fastapi import APIRouter
from datetime import datetime

from ...core.schemas import HealthResponse
from ...models.registry import get_model_registry
from ..config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    registry = get_model_registry()
    
    # Check model status
    models_loaded = len(registry.list_models())
    
    # Check cache status (simplified for now)
    cache_status = "connected" if settings.CACHE_ENABLED else "disabled"
    
    # Check database status (simplified for now)
    database_status = "connected" if settings.DATABASE_URL else "not_configured"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        models_loaded=models_loaded,
        cache_status=cache_status,
        database_status=database_status
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to accept traffic.
    """
    registry = get_model_registry()
    
    # Check if at least one model is loaded
    if len(registry.list_models()) == 0:
        return {"status": "not_ready", "reason": "No models loaded"}
    
    try:
        # Try to get the default model
        default_model = registry.get_default_model()
        if not default_model._is_loaded:
            return {"status": "not_ready", "reason": "Default model not loaded"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}
    
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()} 
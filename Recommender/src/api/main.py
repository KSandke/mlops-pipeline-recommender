"""
Main FastAPI application for the recommendation system.
This implements a production-ready API with proper error handling,
middleware, and documentation.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import uuid
from datetime import datetime

from .config import get_settings
from .routers import recommendations, health, models
from ..core.exceptions import RecommenderException
from ..core.schemas import ErrorResponse
from ..models.registry import get_model_registry
from ..models.als_model import ALSRecommendationModel
from ..models.neural_model import NeuralRecommendationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Load models on startup, cleanup on shutdown.
    """
    # Startup
    logger.info("Starting up recommendation API...")
    
    try:
        # Load models
        registry = get_model_registry()
        
        # Load ALS model
        als_model = ALSRecommendationModel(model_id="als_prod_v1")
        als_model.load(
            model_path=settings.MODEL_PATH,
            version="1.0.0",
            trained_at="2025-07-01T10:00:00"
        )
        registry.register_model(als_model)
        
        # Load Neural model (if available)
        try:
            neural_model = NeuralRecommendationModel(model_id="neural_prod_v1")
            neural_model.load(
                model_path=settings.MODEL_PATH,
                version="1.0.0",
                trained_at="2025-07-01T10:00:00"
            )
            registry.register_model(neural_model)
            logger.info("Neural model loaded successfully")
        except Exception as e:
            logger.warning(f"Neural model not available: {str(e)}")
        
        logger.info(f"Loaded {len(registry.list_models())} models")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        # In production, you might want to fail fast here
        # raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down recommendation API...")


# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Production-ready recommendation system with MLOps practices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add to logs
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    
    # Log response time
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Request {request_id} completed in {process_time:.2f}ms")
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


@app.exception_handler(RecommenderException)
async def recommender_exception_handler(request: Request, exc: RecommenderException):
    """Handle custom recommender exceptions."""
    error_response = ErrorResponse(
        error=exc.error_code,
        message=exc.message,
        details=exc.details,
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    # Map exceptions to HTTP status codes
    status_code_map = {
        "UserNotFoundError": 404,
        "ItemNotFoundError": 404,
        "ModelNotFoundError": 404,
        "ModelNotLoadedError": 503,
        "ValidationError": 400,
        "PredictionError": 500,
        "ConfigurationError": 500,
    }
    
    status_code = status_code_map.get(exc.error_code, 500)
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        timestamp=datetime.utcnow(),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode="json")
    )


# Include routers
app.include_router(
    recommendations.router,
    prefix="/api/v1/recommendations",
    tags=["recommendations"]
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["health"]
)

app.include_router(
    models.router,
    prefix="/api/v1/models",
    tags=["models"]
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    ) 
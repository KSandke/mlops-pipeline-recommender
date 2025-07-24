"""
Pydantic schemas for API request/response validation.
These provide automatic validation, serialization, and OpenAPI documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from ..api.config import get_settings

settings = get_settings()


class ModelType(str, Enum):
    """Supported model types."""
    ALS = "als"
    NCF = "ncf"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


class FilterCriteria(BaseModel):
    """Filtering criteria for recommendations."""
    genres: Optional[List[str]] = Field(None, description="Filter by genres")
    min_year: Optional[int] = Field(None, description="Minimum release year")
    max_year: Optional[int] = Field(None, description="Maximum release year")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum average rating")
    exclude_watched: bool = Field(True, description="Exclude already watched items")
    
    @validator('min_year', 'max_year')
    def validate_year(cls, v):
        if v is not None and (v < 1900 or v > datetime.now().year + 1):
            raise ValueError(f"Invalid year: {v}")
        return v


class RecommendationRequest(BaseModel):
    """Request model for single user recommendations."""
    user_id: int = Field(..., description="User ID to get recommendations for")
    n_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    model_id: Optional[str] = Field(None, description="Specific model to use")
    filter_criteria: Optional[FilterCriteria] = Field(None, description="Filtering options")
    exclude_items: Optional[List[int]] = Field(None, description="Item IDs to exclude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "n_recommendations": 10,
                "filter_criteria": {
                    "genres": ["Action", "Sci-Fi"],
                    "min_year": 2010
                }
            }
        }


class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations."""
    user_ids: List[int] = Field(..., min_items=1, max_items=100, description="List of user IDs")
    n_recommendations: int = Field(10, ge=1, le=50, description="Recommendations per user")
    model_id: Optional[str] = Field(None, description="Specific model to use")
    filter_criteria: Optional[FilterCriteria] = Field(None, description="Filtering options")


class SimilarItemsRequest(BaseModel):
    """Request model for similar items."""
    item_id: int = Field(..., description="Item ID to find similar items for")
    n_similar: int = Field(10, ge=1, le=50, description="Number of similar items")
    model_id: Optional[str] = Field(None, description="Specific model to use")
    filter_criteria: Optional[FilterCriteria] = Field(None, description="Filtering options")


class ItemInfo(BaseModel):
    """Item information in responses."""
    item_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    average_rating: Optional[float] = None
    total_ratings: Optional[int] = None


class RecommendationItem(BaseModel):
    """Single recommendation in response."""
    rank: int = Field(..., description="Recommendation rank (1-based)")
    score: float = Field(..., description="Recommendation score")
    item: ItemInfo = Field(..., description="Item details")
    explanation: Optional[str] = Field(None, description="Why this was recommended")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[RecommendationItem]
    model_id: str
    model_version: str
    generated_at: datetime
    processing_time_ms: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "recommendations": [
                    {
                        "rank": 1,
                        "score": 0.95,
                        "item": {
                            "item_id": 1,
                            "title": "Toy Story",
                            "genres": ["Animation", "Comedy"],
                            "year": 1995
                        }
                    }
                ],
                "model_id": "als_prod_v1",
                "model_version": "1.0.0",
                "generated_at": "2025-07-03T12:00:00Z",
                "processing_time_ms": 45.2
            }
        }


class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations."""
    recommendations: Dict[int, List[RecommendationItem]]
    model_id: str
    model_version: str
    generated_at: datetime
    total_users: int
    successful_users: int
    failed_users: List[int]
    processing_time_ms: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "recommendations": [
                    {
                        "rank": 1,
                        "score": 0.95,
                        "item": {
                            "item_id": 1,
                            "title": "Toy Story",
                            "genres": ["Animation", "Comedy"],
                            "year": 1995
                        }
                    }
                ],
                "model_id": "als_prod_v1",
                "model_version": "1.0.0",
                "generated_at": "2025-07-03T12:00:00Z",
                "processing_time_ms": 45.2
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    model_type: ModelType
    version: str
    status: str
    trained_at: datetime
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    is_default: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "als_prod_v1",
                "model_type": "als",
                "version": "1.0.0",
                "status": "active",
                "trained_at": "2025-07-01T10:00:00Z",
                "metrics": {
                    "precision_at_10": 0.0144,
                    "recall_at_10": 0.1436,
                    "map_at_10": 0.0499
                },
                "hyperparameters": {
                    "factors": 50,
                    "regularization": 0.01,
                    "iterations": 20
                },
                "is_default": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    models_loaded: int
    cache_status: str
    database_status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-07-03T12:00:00Z",
                "version": "1.0.0",
                "models_loaded": 2,
                "cache_status": "connected",
                "database_status": "connected"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "UserNotFoundError",
                "message": "User with ID 999999 not found",
                "timestamp": "2025-07-03T12:00:00Z",
                "request_id": "req-123456"
            }
        }


class ABTestConfig(BaseModel):
    """A/B test configuration."""
    test_id: str
    model_a: str
    model_b: str
    traffic_split: float = Field(0.5, ge=0, le=1, description="Fraction of traffic to model A")
    active: bool = True
    start_date: datetime
    end_date: Optional[datetime] = None 
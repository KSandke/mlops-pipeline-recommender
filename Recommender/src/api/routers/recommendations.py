"""
Recommendation endpoints for the API.
Handles user recommendations, similar items, and batch predictions.
"""

from fastapi import APIRouter, Depends, Query, Request
from typing import Optional
import time
from datetime import datetime

from ...core.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    SimilarItemsRequest,
    RecommendationItem,
    ItemInfo,
    FilterCriteria
)
from ...models.registry import get_model_registry
from ..dependencies import get_current_model

router = APIRouter()


@router.post("/user", response_model=RecommendationResponse)
async def get_user_recommendations(
    request: RecommendationRequest,
    req: Request,
    model=Depends(get_current_model)
):
    """
    Get personalized recommendations for a user.
    
    This endpoint generates movie recommendations based on the user's historical ratings
    and preferences using collaborative filtering.
    """
    start_time = time.time()
    
    # Get recommendations from model
    recommendations = model.predict_for_user(
        user_id=request.user_id,
        n_recommendations=request.n_recommendations,
        exclude_items=request.exclude_items,
        filter_criteria=request.filter_criteria.dict() if request.filter_criteria else None
    )
    
    # Convert to response format
    recommendation_items = []
    for rec in recommendations:
        item_info = ItemInfo(
            item_id=rec.item_id,
            title=rec.metadata.get('title', f'Movie {rec.item_id}'),
            genres=rec.metadata.get('genres', []),
            year=rec.metadata.get('year'),
            average_rating=rec.metadata.get('avg_rating'),
            total_ratings=rec.metadata.get('total_ratings')
        )
        
        recommendation_items.append(RecommendationItem(
            rank=rec.rank,
            score=rec.score,
            item=item_info
        ))
    
    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendation_items,
        model_id=model.model_id,
        model_version=model.get_model_metadata().version,
        generated_at=datetime.utcnow(),
        processing_time_ms=processing_time_ms
    )


@router.post("/batch", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    model=Depends(get_current_model)
):
    """
    Get recommendations for multiple users in a single request.
    
    This is more efficient than making multiple individual requests.
    Maximum batch size is controlled by MAX_BATCH_SIZE setting.
    """
    start_time = time.time()
    
    # Get batch predictions
    batch_results = model.predict_batch(
        user_ids=request.user_ids,
        n_recommendations=request.n_recommendations,
        filter_criteria=request.filter_criteria.dict() if request.filter_criteria else None
    )
    
    # Convert to response format
    recommendations = {}
    successful_users = 0
    failed_users = []
    
    for user_id, user_recs in batch_results.items():
        if user_recs:
            recommendation_items = []
            for rec in user_recs:
                item_info = ItemInfo(
                    item_id=rec.item_id,
                    title=rec.metadata.get('title', f'Movie {rec.item_id}'),
                    genres=rec.metadata.get('genres', []),
                    year=rec.metadata.get('year'),
                    average_rating=rec.metadata.get('avg_rating'),
                    total_ratings=rec.metadata.get('total_ratings')
                )
                
                recommendation_items.append(RecommendationItem(
                    rank=rec.rank,
                    score=rec.score,
                    item=item_info
                ))
            
            recommendations[user_id] = recommendation_items
            successful_users += 1
        else:
            failed_users.append(user_id)
    
    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000
    
    return BatchRecommendationResponse(
        recommendations=recommendations,
        model_id=model.model_id,
        model_version=model.get_model_metadata().version,
        generated_at=datetime.utcnow(),
        total_users=len(request.user_ids),
        successful_users=successful_users,
        failed_users=failed_users,
        processing_time_ms=processing_time_ms
    )


@router.post("/similar", response_model=RecommendationResponse)
async def get_similar_items(
    request: SimilarItemsRequest,
    model=Depends(get_current_model)
):
    """
    Find items similar to a given item.
    
    This uses the learned item embeddings to find movies with similar characteristics
    based on user interaction patterns.
    """
    start_time = time.time()
    
    # Get similar items from model
    similar_items = model.predict_similar_items(
        item_id=request.item_id,
        n_similar=request.n_similar,
        filter_criteria=request.filter_criteria.dict() if request.filter_criteria else None
    )
    
    # Convert to response format
    recommendation_items = []
    for item in similar_items:
        item_info = ItemInfo(
            item_id=item.item_id,
            title=item.metadata.get('title', f'Movie {item.item_id}'),
            genres=item.metadata.get('genres', []),
            year=item.metadata.get('year'),
            average_rating=item.metadata.get('avg_rating'),
            total_ratings=item.metadata.get('total_ratings')
        )
        
        recommendation_items.append(RecommendationItem(
            rank=item.rank,
            score=item.score,
            item=item_info
        ))
    
    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Use item_id as "user_id" in response for consistency
    return RecommendationResponse(
        user_id=request.item_id,
        recommendations=recommendation_items,
        model_id=model.model_id,
        model_version=model.get_model_metadata().version,
        generated_at=datetime.utcnow(),
        processing_time_ms=processing_time_ms
    ) 
import pytest  # noqa: F401
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Recommender.src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_user_recommendations():
    response = client.post("/api/v1/recommendations/user", json={"user_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

def test_similar_items():
    response = client.post("/api/v1/recommendations/similar", json={"item_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

def test_batch_recommendations():
    response = client.post("/api/v1/recommendations/batch", json={"user_ids": [1, 2, 3]})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], dict)
    assert "1" in data["recommendations"]
    assert "2" in data["recommendations"]
    assert "3" in data["recommendations"] 
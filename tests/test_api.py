import pytest
from fastapi.testclient import TestClient
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Recommender.src.api.main import app
from Recommender.src.api.dependencies import get_current_model
from Recommender.src.core.interfaces import BaseRecommendationModel, RecommendationResult, ModelMetadata

# Create a mock model that we can control for tests
class MockRecommendationModel(BaseRecommendationModel):
    def load(self, model_path: str, **kwargs) -> None:
        self._is_loaded = True

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="mock_model",
            model_type="mock",
            version="1.0",
            trained_at=datetime.utcnow(),
            metrics={},
            hyperparameters={},
            data_version="mock_data"
        )

    def predict_for_user(self, user_id, n_recommendations, **kwargs):
        results = []
        for i in range(n_recommendations):
            results.append(RecommendationResult(item_id=i, score=1.0 - (i * 0.1), rank=i + 1, metadata={}))
        return results

    def predict_similar_items(self, item_id, n_similar, **kwargs):
        results = []
        for i in range(n_similar):
            results.append(RecommendationResult(item_id=i + 10, score=1.0 - (i * 0.1), rank=i + 1, metadata={'item_id': i + 10}))
        return results

    def predict_batch(self, user_ids, n_recommendations, **kwargs):
        return {uid: self.predict_for_user(uid, n_recommendations) for uid in user_ids}
    
    def get_user_embeddings(self, user_id: int): return None
    def get_item_embeddings(self, item_id: int): return None

@pytest.fixture
def mock_model():
    return MockRecommendationModel(model_id="test_model")

@pytest.fixture(autouse=True)
def override_get_current_model(mock_model):
    app.dependency_overrides[get_current_model] = lambda: mock_model
    yield
    app.dependency_overrides.clear()

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Movie Recommendation API"

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_user_recommendations():
    response = client.post("/api/v1/recommendations/user", json={"user_id": 1, "n_recommendations": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["rank"] == 1

def test_similar_items():
    response = client.post("/api/v1/recommendations/similar", json={"item_id": 1, "n_similar": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1  # In similar items, the user_id field holds the original item_id
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["item"]["item_id"] == 10

def test_batch_recommendations():
    response = client.post("/api/v1/recommendations/batch", json={"user_ids": [1, 2, 3], "n_recommendations": 2})
    assert response.status_code == 200
    data = response.json()
    assert "1" in data["recommendations"]
    assert "2" in data["recommendations"]
    assert len(data["recommendations"]["1"]) == 2 
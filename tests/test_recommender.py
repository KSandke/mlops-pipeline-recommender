import pytest
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Recommender', 'src'))

class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_sample_data_creation(self):
        """Test that we can create sample data for testing."""
        # Create sample ratings data
        sample_ratings = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3],
            'movieId': [1, 2, 1, 3, 2],
            'rating': [4.0, 3.5, 5.0, 2.0, 4.5],
            'timestamp': [1000, 1001, 1002, 1003, 1004]
        })
        
        assert len(sample_ratings) == 5
        assert 'userId' in sample_ratings.columns
        assert 'movieId' in sample_ratings.columns
        assert 'rating' in sample_ratings.columns
        assert 'timestamp' in sample_ratings.columns

class TestEvaluationMetrics:
    """Test evaluation metrics calculations."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        # Mock data
        recommended_items = [1, 2, 3, 4, 5]
        relevant_items = {1, 3, 5}
        
        # Calculate precision@3
        k = 3
        top_k = recommended_items[:k]
        relevant_recommended = len(set(top_k) & relevant_items)
        precision = relevant_recommended / k
        
        expected_precision = 2/3  # Items 1 and 3 are relevant in top 3
        assert abs(precision - expected_precision) < 0.001
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended_items = [1, 2, 3, 4, 5]
        relevant_items = {1, 3, 5, 7, 9}  # 5 relevant items total
        
        k = 3
        top_k = recommended_items[:k]
        relevant_recommended = len(set(top_k) & relevant_items)
        recall = relevant_recommended / len(relevant_items)
        
        expected_recall = 2/5  # 2 out of 5 relevant items found
        assert abs(recall - expected_recall) < 0.001

class TestModelConfiguration:
    """Test model configuration handling."""
    
    def test_config_structure(self):
        """Test that config has required structure."""
        # Sample config structure
        config = {
            'als': {
                'factors': 50,
                'regularization': 0.01,
                'iterations': 20,
                'random_state': 42
            },
            'cuda': {
                'enabled': False,
                'device': 0
            }
        }
        
        assert 'als' in config
        assert 'factors' in config['als']
        assert 'regularization' in config['als']
        assert 'iterations' in config['als']
        assert 'random_state' in config['als']
        
        assert 'cuda' in config
        assert 'enabled' in config['cuda']
        assert 'device' in config['cuda']

class TestRecommendationSystem:
    """Test recommendation system functionality."""
    
    @pytest.fixture
    def sample_interaction_matrix(self):
        """Create a sample interaction matrix for testing."""
        # Simple 3x3 matrix for testing
        data = [4.0, 3.5, 5.0, 2.0, 4.5]
        row = [0, 0, 1, 1, 2]
        col = [0, 1, 0, 2, 1]
        return csr_matrix((data, (row, col)), shape=(3, 3))
    
    def test_matrix_sparsity_calculation(self, sample_interaction_matrix):
        """Test sparsity calculation."""
        matrix = sample_interaction_matrix
        sparsity = 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        
        # 5 non-zero elements out of 9 total = 4/9 sparsity
        expected_sparsity = 4/9
        assert abs(sparsity - expected_sparsity) < 0.001

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import implicit  # noqa: F401
        import yaml  # noqa: F401
        import joblib  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")

if __name__ == '__main__':
    pytest.main([__file__]) 
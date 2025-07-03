"""
Configuration management for the API.
Uses Pydantic settings for environment-based configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    API configuration settings.
    Values can be overridden by environment variables.
    """
    
    # API Settings
    API_VERSION: str = "v1"
    DEBUG: bool = Field(False, env="DEBUG")
    
    # Model Settings
    MODEL_PATH: str = Field("Recommender/models", env="MODEL_PATH")
    DEFAULT_MODEL_ID: str = Field("als_prod_v1", env="DEFAULT_MODEL_ID")
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(["*"], env="ALLOWED_HOSTS")
    
    # Cache Settings
    CACHE_ENABLED: bool = Field(True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(3600, env="CACHE_TTL")  # 1 hour
    REDIS_URL: Optional[str] = Field(None, env="REDIS_URL")
    
    # Database Settings (for future use)
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field("movielens-api", env="MLFLOW_EXPERIMENT_NAME")
    
    # Performance Settings
    MAX_BATCH_SIZE: int = Field(100, env="MAX_BATCH_SIZE")
    REQUEST_TIMEOUT: int = Field(30, env="REQUEST_TIMEOUT")
    
    # Security Settings
    API_KEY_ENABLED: bool = Field(False, env="API_KEY_ENABLED")
    API_KEYS: List[str] = Field([], env="API_KEYS")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    
    # A/B Testing
    AB_TEST_ENABLED: bool = Field(False, env="AB_TEST_ENABLED")
    AB_TEST_CONFIG: Optional[str] = Field(None, env="AB_TEST_CONFIG")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance
    """
    return Settings() 
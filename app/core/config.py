import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    app_name: str = "FlowScope AI"
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5178,http://localhost:3000"
    
    # Data settings
    max_dataset_size: int = 50000
    default_dataset_size: int = 10000
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins string to list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()

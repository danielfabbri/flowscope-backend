from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class PipelineStep(BaseModel):
    """Configuration for a single pipeline step."""
    name: str
    type: str  # ingestion, cleaning, transformation, feature_engineering, ml_modeling, output
    enabled: bool = True
    config: Dict[str, Any] = {}


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    name: str
    description: Optional[str] = None
    steps: List[PipelineStep]


class PipelineCreate(BaseModel):
    """Create pipeline request."""
    config: PipelineConfig


class PipelineStatus(BaseModel):
    """Pipeline execution status."""
    pipeline_id: str
    status: str  # created, running, completed, failed
    current_step: Optional[str] = None
    steps_completed: List[str] = []
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    """Pipeline creation response."""
    pipeline_id: str
    config: PipelineConfig
    status: str


class ChatQuery(BaseModel):
    """Chat query request."""
    pipeline_id: str
    query: str


class ChatResponse(BaseModel):
    """Chat response."""
    answer: str
    data: Optional[Dict[str, Any]] = None

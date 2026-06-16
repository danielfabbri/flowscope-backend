# Schemas for Solutions (grouping multiple pipelines)
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class SolutionCreate(BaseModel):
    """Schema for creating a new solution"""
    name: str
    description: str
    icon: Optional[str] = "folder"
    color: Optional[str] = "#6366f1"
    category: Optional[str] = "general"


class SolutionUpdate(BaseModel):
    """Schema for updating a solution"""
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    category: Optional[str] = None


class SolutionResponse(BaseModel):
    """Schema for solution response"""
    id: str
    name: str
    description: str
    icon: str
    color: str
    category: str
    created_at: str
    updated_at: str
    pipeline_count: int
    pipelines: List[str]  # List of pipeline IDs


class SolutionListResponse(BaseModel):
    """Schema for listing solutions"""
    solutions: List[SolutionResponse]
    total: int

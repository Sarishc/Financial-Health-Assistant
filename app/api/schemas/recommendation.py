# app/api/schemas/recommendation.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class RecommendationBase(BaseModel):
    """Base recommendation schema"""
    message: str
    type: str
    priority: int = Field(..., ge=1, le=10, description="Priority level from 1-10")

class Recommendation(RecommendationBase):
    """Complete recommendation schema with ID"""
    id: str
    category: Optional[str] = None
    amount: Optional[float] = None
    percentage: Optional[float] = None
    created_at: datetime

    class Config:
        orm_mode = True

class RecommendationList(BaseModel):
    """List of recommendations"""
    recommendations: List[Recommendation]
    total: int

class RecommendationFilters(BaseModel):
    """Recommendation filter options"""
    types: List[str]
    categories: List[str]
    priority_range: List[int]

class RecommendationReport(BaseModel):
    """Comprehensive recommendation report"""
    report_id: str
    generated_at: datetime
    top_recommendations: List[Recommendation]
    recommendation_by_type: Dict[str, List[Recommendation]]
    savings_potential: float
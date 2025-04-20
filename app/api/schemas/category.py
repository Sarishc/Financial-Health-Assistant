# app/api/schemas/category.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Category(BaseModel):
    """Category schema"""
    name: str
    description: Optional[str] = None
    color: Optional[str] = Field(None, description="Hex color code for display")
    icon: Optional[str] = None
    parent_category: Optional[str] = None

class CategoryStats(BaseModel):
    """Category statistics"""
    name: str
    transaction_count: int
    total_amount: float
    average_transaction: float
    percentage_of_total: float
    month_to_month_change: Optional[float] = None

class CategoryList(BaseModel):
    """List of categories"""
    categories: List[Category]
    total: int

class CategoryHierarchy(BaseModel):
    """Category hierarchy"""
    name: str
    subcategories: List["CategoryHierarchy"] = []
    
    class Config:
        orm_mode = True

CategoryHierarchy.update_forward_refs()
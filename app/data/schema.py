# app/data/schema.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime, date

class Transaction(BaseModel):
    """
    Schema for a financial transaction. This defines the expected structure 
    for transaction data throughout the application.
    """
    transaction_id: Optional[str] = None
    transaction_date: date
    description: str
    amount: float
    category: Optional[str] = None
    user_category: Optional[str] = None
    
    # Optional fields that might be useful
    merchant: Optional[str] = None
    location: Optional[str] = None
    is_recurring: Optional[bool] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount follows our convention: negative for expenses, positive for income"""
        # Optional validation logic
        return v

class TransactionBatch(BaseModel):
    """Schema for a batch of transactions (e.g., from CSV upload)"""
    transactions: List[Transaction]
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_date": "2023-01-15",
                        "description": "Grocery Store Purchase",
                        "amount": -85.42,
                        "category": "food"
                    },
                    {
                        "transaction_date": "2023-01-16",
                        "description": "Monthly Salary",
                        "amount": 3000.00,
                        "category": "income"
                    }
                ]
            }
        }

class Category(BaseModel):
    """Schema for transaction categories"""
    category_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    is_expense: bool = True
    color: Optional[str] = None  # For UI representation
    
    class Config:
        schema_extra = {
            "example": {
                "name": "dining",
                "description": "Restaurants and dining out",
                "is_expense": True,
                "color": "#FF5733"
            }
        }

class CategoryList(BaseModel):
    """Schema for a list of categories"""
    categories: List[Category]

class SpendingSummary(BaseModel):
    """Schema for spending summary by category"""
    category: str
    amount: float
    percentage: float
    count: int
    
    class Config:
        schema_extra = {
            "example": {
                "category": "food",
                "amount": 325.65,
                "percentage": 15.2,
                "count": 12
            }
        }

class MonthlySummary(BaseModel):
    """Schema for monthly spending summary"""
    month: str  # Format: YYYY-MM
    total_income: float
    total_expenses: float
    net: float
    top_categories: List[SpendingSummary]
    
    class Config:
        schema_extra = {
            "example": {
                "month": "2023-01",
                "total_income": 3500.00,
                "total_expenses": 2145.30,
                "net": 1354.70,
                "top_categories": [
                    {
                        "category": "food",
                        "amount": 325.65,
                        "percentage": 15.2,
                        "count": 12
                    }
                ]
            }
        }

class ForecastPoint(BaseModel):
    """Schema for a single forecast data point"""
    date: date
    amount: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2023-02-15",
                "amount": 95.20,
                "lower_bound": 85.40,
                "upper_bound": 105.00
            }
        }

class CategoryForecast(BaseModel):
    """Schema for category-specific forecast"""
    category: str
    forecast: List[ForecastPoint]
    total_forecasted: float
    
    class Config:
        schema_extra = {
            "example": {
                "category": "food",
                "forecast": [
                    {
                        "date": "2023-02-15",
                        "amount": 95.20,
                        "lower_bound": 85.40,
                        "upper_bound": 105.00
                    }
                ],
                "total_forecasted": 450.30
            }
        }

class FullForecast(BaseModel):
    """Schema for complete spending forecast"""
    total_forecast: List[ForecastPoint]
    categories: List[CategoryForecast]
    
    class Config:
        schema_extra = {
            "example": {
                "total_forecast": [
                    {
                        "date": "2023-02-15",
                        "amount": 215.40,
                        "lower_bound": 195.20,
                        "upper_bound": 235.60
                    }
                ],
                "categories": [
                    {
                        "category": "food",
                        "forecast": [
                            {
                                "date": "2023-02-15",
                                "amount": 95.20,
                                "lower_bound": 85.40,
                                "upper_bound": 105.00
                            }
                        ],
                        "total_forecasted": 450.30
                    }
                ]
            }
        }

class Recommendation(BaseModel):
    """Schema for a saving recommendation"""
    recommendation_id: Optional[str] = None
    type: str  # 'high_spending', 'recurring_charge', etc.
    category: str
    message: str
    potential_savings: Optional[float] = None
    priority: int  # 0-100
    
    class Config:
        schema_extra = {
            "example": {
                "type": "high_spending",
                "category": "dining",
                "message": "You've spent 30% more on dining than usual. Consider cooking at home to save $150 next month.",
                "potential_savings": 150.00,
                "priority": 85
            }
        }

class RecommendationList(BaseModel):
    """Schema for a list of recommendations"""
    recommendations: List[Recommendation]

class DatabaseSettings(BaseModel):
    """Database connection settings"""
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: str
    POSTGRES_DB: str
    
    def get_connection_string(self) -> str:
        """Return PostgreSQL connection string"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
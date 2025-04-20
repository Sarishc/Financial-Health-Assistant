# app/api/schemas/transaction.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class TransactionBase(BaseModel):
    """Base transaction schema"""
    description: str
    amount: float = Field(..., description="Positive for income, negative for expenses")
    transaction_date: datetime
    category: Optional[str] = None

class TransactionCreate(TransactionBase):
    """Transaction creation schema"""
    account_id: Optional[str] = None
    notes: Optional[str] = None

class Transaction(TransactionBase):
    """Complete transaction schema with ID"""
    id: str
    account_id: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class TransactionUpdate(BaseModel):
    """Transaction update schema"""
    description: Optional[str] = None
    amount: Optional[float] = None
    transaction_date: Optional[datetime] = None
    category: Optional[str] = None
    notes: Optional[str] = None

class TransactionList(BaseModel):
    """List of transactions"""
    transactions: List[Transaction]
    total: int

class TransactionStats(BaseModel):
    """Transaction statistics"""
    total_income: float
    total_expenses: float
    net_cashflow: float
    period_start: datetime
    period_end: datetime
    categories: Dict[str, float]
# app/api/schemas/auth.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Token(BaseModel):
    """Token schema for authentication response"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data schema"""
    email: Optional[str] = None

class User(BaseModel):
    """Base user schema"""
    email: EmailStr
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserCreate(BaseModel):
    """User creation schema"""
    email: EmailStr
    full_name: str
    password: str = Field(..., min_length=8)

class UserInDB(User):
    """User database schema"""
    hashed_password: str
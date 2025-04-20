from fastapi import APIRouter
from app.api.routes.transactions import router as transactions_router

api_router = APIRouter()
api_router.include_router(transactions_router, prefix="/api/v1", tags=["transactions"])

# app/api/routes/__init__.py
# Import routes modules individually to avoid circular imports
from fastapi import APIRouter

# Create empty router objects first
transactions = APIRouter()
categories = APIRouter()
recommendations = APIRouter()
forecasts = APIRouter()
users = APIRouter()

# Don't try to import the actual modules here
# The main file will import the actual router objects directly
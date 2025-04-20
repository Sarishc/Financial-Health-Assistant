# app/api/routes/__init__.py
# Routes module initialization

# Import individual routers
from app.api.routes.transactions import router as transactions_router
from app.api.routes.categories import router as categories_router
from app.api.routes.recommendations import router as recommendations_router
from app.api.routes.users import router as users_router

# Export the routers with the expected names
transactions = transactions_router
categories = categories_router
recommendations = recommendations_router
users = users_router

# Create a placeholder for forecasts if it's not yet implemented
from fastapi import APIRouter
forecasts = APIRouter()
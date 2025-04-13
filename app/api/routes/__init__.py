from fastapi import APIRouter
from app.api.routes.transactions import router as transactions_router

api_router = APIRouter()
api_router.include_router(transactions_router, prefix="/api/v1", tags=["transactions"])
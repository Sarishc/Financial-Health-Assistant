# app/api/routes/forecasts.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/forecasts")
async def get_forecasts():
    """Placeholder for forecasts endpoint"""
    return {"message": "Forecast functionality coming soon"}
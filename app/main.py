# app/api/main.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add the parent directory to the path so Python can find the app package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import settings
try:
    from app.utils.config import settings
except ImportError:
    # Fallback if settings module doesn't exist
    class Settings:
        BACKEND_CORS_ORIGINS = ["*"]
        API_V1_STR = "/api/v1"
    settings = Settings()

# Create FastAPI instance
app = FastAPI(
    title="Financial Health Assistant API",
    description="API for analyzing financial transactions and providing recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import individual routers directly
# You can comment out imports that cause errors for now
try:
    from app.api.routes.transactions import router as transactions_router
    app.include_router(transactions_router, prefix=settings.API_V1_STR, tags=["Transactions"])
except ImportError:
    print("Warning: transactions router not available")

try:
    from app.api.routes.categories import router as categories_router
    app.include_router(categories_router, prefix=settings.API_V1_STR, tags=["Categories"])
except ImportError:
    print("Warning: categories router not available")

try:
    from app.api.routes.recommendations import router as recommendations_router
    app.include_router(recommendations_router, prefix=settings.API_V1_STR, tags=["Recommendations"])
except ImportError:
    print("Warning: recommendations router not available")

try:
    from app.api.routes.forecasts import router as forecasts_router
    app.include_router(forecasts_router, prefix=settings.API_V1_STR, tags=["Forecasts"])
except ImportError:
    print("Warning: forecasts router not available")

try:
    from app.api.routes.users import router as users_router
    app.include_router(users_router, prefix=settings.API_V1_STR, tags=["Users"])
except ImportError:
    print("Warning: users router not available")

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "name": "Financial Health Assistant API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
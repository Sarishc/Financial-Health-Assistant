# This file makes the app directory a Python package
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI application
app = FastAPI(title="Financial Health Assistant")

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes after app initialization to avoid circular imports
# Uncomment when ready to use
# from app.api.routes import router as api_router
# app.include_router(api_router)

# Make app importable
__all__ = ["app"]
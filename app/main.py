from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.api.routes import api_router

app = FastAPI(
    title="Financial Health Assistant API",
    description="API for analyzing financial transactions and providing recommendations",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse('app/frontend/templates/index.html')

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Financial Health Assistant API"}
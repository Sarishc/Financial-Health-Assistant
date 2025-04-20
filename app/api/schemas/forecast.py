# app/api/routes/forecasts.py
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys
from datetime import datetime, timedelta, date

# Import schemas
from app.api.schemas.forecast import (
    ForecastPoint, CategoryForecast, ForecastList, 
    ForecastSummary, ForecastTrainingStatus
)
from app.api.schemas.messages import Message
from app.api.auth.auth import get_current_active_user
from app.api.schemas.auth import User

# Add parent directory to path for importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import forecaster if available
try:
    from app.models.forecasting.spending_forecaster import SpendingForecaster
    FORECASTER_AVAILABLE = True
except ImportError:
    FORECASTER_AVAILABLE = False

# Create router
router = APIRouter()

# Path to data files
TRANSACTIONS_PATH = "app/data/processed/transactions_clean.csv"
FORECASTS_PATH = "app/data/processed/forecasts.csv"
MODEL_DIR = "app/models/forecasting/saved_models"

# Helper function to check if forecasts are available
def forecasts_available():
    """Check if forecasting models are available"""
    return FORECASTER_AVAILABLE and os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR)

# Helper function to load transactions
def load_transactions():
    """Load transactions from CSV file"""
    if not os.path.exists(TRANSACTIONS_PATH):
        return pd.DataFrame()
    
    df = pd.read_csv(TRANSACTIONS_PATH)
    
    # Convert date columns to datetime
    date_columns = ['transaction_date', 'DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

# Forecast routes
@router.get("/forecasts", response_model=ForecastList)
async def get_forecasts(
    days: int = Query(30, description="Number of days to forecast"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get spending forecasts for all categories or a specific category
    
    - **days**: Number of days to forecast (default: 30 days)
    - **category**: Filter by specific category (optional)
    """
    # Check if forecasting is available
    if not forecasts_available():
        raise HTTPException(status_code=400, detail="Forecasting models not available")
    
    try:
        # Initialize forecaster
        forecaster = SpendingForecaster()
        forecaster.load_models(MODEL_DIR)
        
        # Generate forecasts
        forecast_data = forecaster.forecast(days=days)
        
        # If category filter specified, filter the forecasts
        if category:
            if category not in forecast_data:
                raise HTTPException(status_code=404, detail=f"No forecast available for category: {category}")
            
            forecast_data = {category: forecast_data[category]}
        
        # Process forecasts into response format
        forecasts = []
        
        for cat, forecast_df in forecast_data.items():
            # Determine amount column based on forecast format
            if 'yhat' in forecast_df.columns:  # Prophet forecast format
                amount_col = 'yhat'
                lower_bound_col = 'yhat_lower' if 'yhat_lower' in forecast_df.columns else None
                upper_bound_col = 'yhat_upper' if 'yhat_upper' in forecast_df.columns else None
                date_col = 'ds'
            else:
                amount_col = next((col for col in forecast_df.columns if col != 'date' and col != 'index'), forecast_df.columns[0])
                lower_bound_col = None
                upper_bound_col = None
                date_col = 'date' if 'date' in forecast_df.columns else forecast_df.index.name or 'index'
            
            # Get the dates and convert if necessary
            if date_col == 'index':
                forecast_dates = forecast_df.index
            else:
                forecast_dates = forecast_df[date_col]
            
            # Convert to date objects if necessary
            if isinstance(forecast_dates[0], datetime):
                forecast_dates = [d.date() for d in forecast_dates]
            elif not isinstance(forecast_dates[0], date):
                forecast_dates = [pd.Timestamp(d).date() for d in forecast_dates]
            
            # Create forecast points
            points = []
            for i, d in enumerate(forecast_dates):
                # Get amount
                amount = abs(float(forecast_df.iloc[i][amount_col]))
                
                # Get bounds if available
                lower_bound = abs(float(forecast_df.iloc[i][lower_bound_col])) if lower_bound_col else None
                upper_bound = abs(float(forecast_df.iloc[i][upper_bound_col])) if upper_bound_col else None
                
                points.append(ForecastPoint(
                    date=d,
                    amount=amount,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                ))
            
            # Calculate total forecast
            total_forecast = sum(p.amount for p in points)
            
            # Get date range
            start_date = min(forecast_dates)
            end_date = max(forecast_dates)
            period_days = (end_date - start_date).days + 1
            
            # Create category forecast
            category_forecast = CategoryForecast(
                category=cat,
                forecast_points=points,
                total_forecast=total_forecast,
                historical_average=None,  # Would need historical data for this
                percent_change=None,      # Would need historical data for this
                confidence_level=0.80,    # Default confidence level
                forecast_start_date=start_date,
                forecast_end_date=end_date,
                forecast_period_days=period_days
            )
            
            forecasts.append(category_forecast)
        
        return {"forecasts": forecasts, "total": len(forecasts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecasts: {str(e)}")

@router.get("/forecasts/summary", response_model=ForecastSummary)
async def get_forecast_summary(
    days: int = Query(30, description="Number of days to forecast"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a summary of forecasts across all categories
    
    - **days**: Number of days to forecast (default: 30 days)
    """
    # Check if forecasting is available
    if not forecasts_available():
        raise HTTPException(status_code=400, detail="Forecasting models not available")
    
    try:
        # Initialize forecaster
        forecaster = SpendingForecaster()
        forecaster.load_models(MODEL_DIR)
        
        # Generate forecasts
        forecast_data = forecaster.forecast(days=days)
        
        # Calculate summary statistics
        total_forecast = 0.0
        categories_forecast = {}
        forecast_start_date = None
        forecast_end_date = None
        
        for cat, forecast_df in forecast_data.items():
            # Determine amount column based on forecast format
            if 'yhat' in forecast_df.columns:  # Prophet forecast format
                amount_col = 'yhat'
                date_col = 'ds'
            else:
                amount_col = next((col for col in forecast_df.columns if col != 'date' and col != 'index'), forecast_df.columns[0])
                date_col = 'date' if 'date' in forecast_df.columns else forecast_df.index.name or 'index'
            
            # Calculate category total
            category_total = abs(forecast_df[amount_col].sum())
            total_forecast += category_total
            categories_forecast[cat] = float(category_total)
            
            # Get date range
            if date_col == 'index':
                dates = forecast_df.index
            else:
                dates = forecast_df[date_col]
            
            # Convert to date objects if necessary
            if isinstance(dates[0], datetime):
                dates = [d.date() for d in dates]
            elif not isinstance(dates[0], date):
                dates = [pd.Timestamp(d).date() for d in dates]
            
            # Update date range
            cat_start_date = min(dates)
            cat_end_date = max(dates)
            
            if forecast_start_date is None or cat_start_date < forecast_start_date:
                forecast_start_date = cat_start_date
            
            if forecast_end_date is None or cat_end_date > forecast_end_date:
                forecast_end_date = cat_end_date
        
        # Calculate period days
        period_days = (forecast_end_date - forecast_start_date).days + 1 if forecast_start_date and forecast_end_date else days
        
        return {
            "total_forecast": float(total_forecast),
            "categories_forecast": categories_forecast,
            "forecast_period_days": period_days,
            "forecast_start_date": forecast_start_date,
            "forecast_end_date": forecast_end_date,
            "generated_at": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast summary: {str(e)}")

@router.get("/forecasts/status", response_model=ForecastTrainingStatus)
async def get_forecast_status(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get status of forecast models
    """
    # Check if forecasting is available
    if not FORECASTER_AVAILABLE:
        return {
            "is_trained": False,
            "last_trained": None,
            "categories_trained": [],
            "models_available": [],
            "accuracy_metrics": None
        }
    
    # Check if models exist
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        return {
            "is_trained": False,
            "last_trained": None,
            "categories_trained": [],
            "models_available": [],
            "accuracy_metrics": None
        }
    
    try:
        # Get a list of available models
        models_available = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib') or f.endswith('.pkl')]
        
        # Get categories from model names
        categories_trained = [os.path.splitext(model)[0] for model in models_available]
        
        # Get last modified time of most recent model
        if models_available:
            model_paths = [os.path.join(MODEL_DIR, model) for model in models_available]
            last_trained = datetime.fromtimestamp(max(os.path.getmtime(path) for path in model_paths))
        else:
            last_trained = None
        
        # For accuracy metrics, we would need to load each model and get its metrics
        # This is a simplified version that doesn't load the actual metrics
        accuracy_metrics = {
            "average_mape": None,  # Mean Absolute Percentage Error
            "average_rmse": None,  # Root Mean Square Error
            "model_details": {}
        }
        
        return {
            "is_trained": True,
            "last_trained": last_trained,
            "categories_trained": categories_trained,
            "models_available": models_available,
            "accuracy_metrics": accuracy_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking forecast status: {str(e)}")

@router.post("/forecasts/train", response_model=Message)
async def train_forecast_models(
    days: int = Query(90, description="Number of days of history to use for training"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Train new forecast models using transaction data
    
    - **days**: Number of days of history to use for training (default: 90 days)
    """
    # Check if forecasting is available
    if not FORECASTER_AVAILABLE:
        raise HTTPException(status_code=400, detail="Forecasting module not available")
    
    # Check if transaction data exists
    if not os.path.exists(TRANSACTIONS_PATH):
        raise HTTPException(status_code=404, detail="No transaction data found")
    
    try:
        # Load transaction data
        df = load_transactions()
        
        # Ensure we have required columns
        date_col = 'transaction_date'
        amount_col = 'amount'
        category_col = 'category'
        
        # Map columns if needed
        if date_col not in df.columns and 'DATE' in df.columns:
            df[date_col] = df['DATE']
        
        if amount_col not in df.columns:
            if 'withdrawal' in df.columns and 'deposit' in df.columns:
                df[amount_col] = df['deposit'] - df['withdrawal']
            elif ' WITHDRAWAL AMT ' in df.columns and ' DEPOSIT AMT ' in df.columns:
                df[amount_col] = df[' DEPOSIT AMT '] - df[' WITHDRAWAL AMT ']
        
        if category_col not in df.columns:
            raise HTTPException(status_code=400, detail="Transaction data does not include categories")
        
        # Initialize forecaster
        forecaster = SpendingForecaster()
        
        # Prepare time series data
        category_dfs = forecaster.prepare_time_series(
            df, 
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col
        )
        
        if not category_dfs:
            raise HTTPException(status_code=400, detail="No valid time series data could be created from transactions")
        
        # Train models
        forecaster.train_models(category_dfs, column=amount_col)
        
        # Save models
        os.makedirs(MODEL_DIR, exist_ok=True)
        forecaster.save_models(MODEL_DIR)
        
        return {"message": f"Successfully trained {len(forecaster.models)} forecasting models"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training forecast models: {str(e)}")
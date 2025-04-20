# app/api/routes/recommendations.py
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
import pandas as pd
import uuid
import os
from datetime import datetime, timedelta
import sys

# Import the schemas
from app.api.schemas.recommendation import (
    Recommendation, RecommendationList, RecommendationFilters, RecommendationReport
)
from app.api.schemas.messages import Message
from app.api.auth.auth import get_current_active_user
from app.api.schemas.auth import User

# Add parent directory to path for importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import recommendation engine
from app.models.recommendation.recommendation_engine import RecommendationEngine

# Create router
router = APIRouter()

# Path to data files
TRANSACTIONS_PATH = "app/data/processed/transactions_clean.csv"
RECOMMENDATIONS_PATH = "app/data/processed/recommendations.csv"

# Helper functions
def load_recommendations():
    """Load recommendations from CSV file"""
    if not os.path.exists(RECOMMENDATIONS_PATH):
        # Create empty DataFrame if file doesn't exist
        df = pd.DataFrame(columns=[
            'id', 'message', 'type', 'priority', 'category', 
            'amount', 'percentage', 'created_at'
        ])
        df.to_csv(RECOMMENDATIONS_PATH, index=False)
        return df
    
    df = pd.read_csv(RECOMMENDATIONS_PATH)
    
    # Convert date columns to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    
    return df

def save_recommendations(df):
    """Save recommendations to CSV file"""
    df.to_csv(RECOMMENDATIONS_PATH, index=False)

def generate_recommendations():
    """Generate recommendations using the recommendation engine"""
    # Check if transaction data exists
    if not os.path.exists(TRANSACTIONS_PATH):
        raise HTTPException(status_code=404, detail="Transaction data not found")
    
    # Load transaction data
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    
    # Convert date column if it exists
    date_col = 'transaction_date'
    if 'DATE' in transactions_df.columns:
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['DATE'])
    elif date_col in transactions_df.columns:
        transactions_df[date_col] = pd.to_datetime(transactions_df[date_col])
    
    # Ensure amount column exists
    amount_col = 'amount'
    if amount_col not in transactions_df.columns:
        if ' WITHDRAWAL AMT ' in transactions_df.columns and ' DEPOSIT AMT ' in transactions_df.columns:
            transactions_df['withdrawal'] = pd.to_numeric(transactions_df[' WITHDRAWAL AMT '], errors='coerce').fillna(0)
            transactions_df['deposit'] = pd.to_numeric(transactions_df[' DEPOSIT AMT '], errors='coerce').fillna(0)
            transactions_df['amount'] = transactions_df['deposit'] - transactions_df['withdrawal']
    
    # Determine category and description columns
    category_col = 'category' if 'category' in transactions_df.columns else None
    desc_col = 'description' if 'description' in transactions_df.columns else (
        'TRANSACTION DETAILS' if 'TRANSACTION DETAILS' in transactions_df.columns else None
    )
    
    # Initialize recommendation engine
    engine = RecommendationEngine(threshold_percentile=75)
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(
        transactions_df,
        date_col=date_col,
        amount_col=amount_col,
        category_col=category_col,
        desc_col=desc_col,
        limit=15
    )
    
    # Convert to DataFrame with IDs and timestamps
    recommendations_list = []
    timestamp = datetime.now()
    
    for rec in recommendations:
        rec_id = str(uuid.uuid4())
        
        # Extract common fields
        rec_dict = {
            'id': rec_id,
            'message': rec['message'],
            'type': rec['type'],
            'priority': rec['priority'],
            'created_at': timestamp
        }
        
        # Extract optional fields if they exist
        for field in ['category', 'amount', 'percentage']:
            if field in rec:
                rec_dict[field] = rec[field]
        
        recommendations_list.append(rec_dict)
    
    # Create DataFrame and save
    recommendations_df = pd.DataFrame(recommendations_list)
    save_recommendations(recommendations_df)
    
    return recommendations_list

# Recommendation routes
@router.get("/recommendations", response_model=RecommendationList)
async def get_recommendations(
    skip: int = Query(0, description="Number of recommendations to skip"),
    limit: int = Query(100, description="Number of recommendations to return"),
    min_priority: Optional[int] = Query(None, ge=1, le=10, description="Filter by minimum priority"),
    max_priority: Optional[int] = Query(None, ge=1, le=10, description="Filter by maximum priority"),
    type: Optional[str] = Query(None, description="Filter by recommendation type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a list of recommendations with optional filters
    
    - **skip**: Number of recommendations to skip (pagination)
    - **limit**: Maximum number of recommendations to return
    - **min_priority**: Filter by minimum priority level (1-10)
    - **max_priority**: Filter by maximum priority level (1-10)
    - **type**: Filter by recommendation type
    - **category**: Filter by category
    """
    # Load recommendations
    df = load_recommendations()
    
    # Apply filters
    if min_priority is not None:
        df = df[df['priority'] >= min_priority]
    
    if max_priority is not None:
        df = df[df['priority'] <= max_priority]
    
    if type:
        df = df[df['type'] == type]
    
    if category and 'category' in df.columns:
        df = df[df['category'] == category]
    
    # Get total count
    total = len(df)
    
    # Apply pagination
    df = df.sort_values('priority', ascending=False)
    df = df.iloc[skip:skip + limit]
    
    # Convert to Pydantic models
    recommendations = []
    for _, row in df.iterrows():
        rec = {
            'id': row['id'],
            'message': row['message'],
            'type': row['type'],
            'priority': row['priority'],
            'created_at': row['created_at']
        }
        
        # Add optional fields if they exist and are not NaN
        for field in ['category', 'amount', 'percentage']:
            if field in row and pd.notna(row[field]):
                rec[field] = row[field]
        
        recommendations.append(Recommendation(**rec))
    
    return {"recommendations": recommendations, "total": total}

@router.get("/recommendations/filters", response_model=RecommendationFilters)
async def get_recommendation_filters(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get available filter options for recommendations
    """
    df = load_recommendations()
    
    # Get available recommendation types
    types = []
    if 'type' in df.columns:
        types = df['type'].dropna().unique().tolist()
    
    # Get available categories
    categories = []
    if 'category' in df.columns:
        categories = df['category'].dropna().unique().tolist()
    
    # Get priority range
    priority_range = [1, 10]  # Default range
    if 'priority' in df.columns and not df.empty:
        min_priority = int(df['priority'].min())
        max_priority = int(df['priority'].max())
        priority_range = [min_priority, max_priority]
    
    return {
        "types": types,
        "categories": categories,
        "priority_range": priority_range
    }

@router.get("/recommendations/{recommendation_id}", response_model=Recommendation)
async def get_recommendation(
    recommendation_id: str = Path(..., description="The ID of the recommendation to get"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a single recommendation by ID
    
    - **recommendation_id**: ID of the recommendation to retrieve
    """
    df = load_recommendations()
    
    # Find recommendation
    recommendation = df[df['id'] == recommendation_id]
    
    if recommendation.empty:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    
    # Convert to Pydantic model
    row = recommendation.iloc[0]
    rec = {
        'id': row['id'],
        'message': row['message'],
        'type': row['type'],
        'priority': row['priority'],
        'created_at': row['created_at']
    }
    
    # Add optional fields if they exist and are not NaN
    for field in ['category', 'amount', 'percentage']:
        if field in row and pd.notna(row[field]):
            rec[field] = row[field]
    
    return Recommendation(**rec)

@router.post("/recommendations/generate", response_model=RecommendationList)
async def regenerate_recommendations(
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate new recommendations based on current transaction data
    """
    try:
        recommendations_list = generate_recommendations()
        
        # Convert to Pydantic models
        recommendations = []
        for rec in recommendations_list:
            recommendations.append(Recommendation(**rec))
        
        return {"recommendations": recommendations, "total": len(recommendations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/recommendations/report", response_model=RecommendationReport)
async def get_recommendation_report(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a comprehensive recommendation report
    """
    df = load_recommendations()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations found")
    
    # Generate report ID
    report_id = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    generated_at = datetime.now()
    
    # Get top recommendations (highest priority)
    top_df = df.sort_values('priority', ascending=False).head(5)
    
    top_recommendations = []
    for _, row in top_df.iterrows():
        rec = {
            'id': row['id'],
            'message': row['message'],
            'type': row['type'],
            'priority': row['priority'],
            'created_at': row['created_at']
        }
        
        # Add optional fields if they exist and are not NaN
        for field in ['category', 'amount', 'percentage']:
            if field in row and pd.notna(row[field]):
                rec[field] = row[field]
        
        top_recommendations.append(Recommendation(**rec))
    
    # Group recommendations by type
    rec_by_type = {}
    if 'type' in df.columns:
        for rec_type in df['type'].unique():
            type_df = df[df['type'] == rec_type]
            type_recs = []
            
            for _, row in type_df.iterrows():
                rec = {
                    'id': row['id'],
                    'message': row['message'],
                    'type': row['type'],
                    'priority': row['priority'],
                    'created_at': row['created_at']
                }
                
                # Add optional fields if they exist and are not NaN
                for field in ['category', 'amount', 'percentage']:
                    if field in row and pd.notna(row[field]):
                        rec[field] = row[field]
                
                type_recs.append(Recommendation(**rec))
            
            rec_by_type[rec_type] = type_recs
    
    # Calculate savings potential
    savings_potential = 0.0
    if 'amount' in df.columns:
        # Sum amounts from saving_opportunity type recommendations
        savings_df = df[df['type'] == 'saving_opportunity']
        if not savings_df.empty and 'amount' in savings_df.columns:
            savings_potential = savings_df['amount'].sum()
    
    return {
        "report_id": report_id,
        "generated_at": generated_at,
        "top_recommendations": top_recommendations,
        "recommendation_by_type": rec_by_type,
        "savings_potential": float(savings_potential)
    }
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import io
import os

from app.data.processor import TransactionProcessor
from app.models.categorization.nlp_categorizer import TransactionCategorizer
from app.models.forecasting.spending_forecaster import SpendingForecaster
from app.models.recommendation.recommendation_engine import RecommendationEngine

router = APIRouter()
processor = TransactionProcessor()

# Load trained models if available
categorizer = None
categorizer_path = 'app/models/categorization/transaction_categorizer.joblib'
if os.path.exists(categorizer_path):
    categorizer = TransactionCategorizer(categorizer_path)

forecaster = SpendingForecaster()
forecasting_model_dir = 'app/models/forecasting/saved_models'
if os.path.exists(forecasting_model_dir) and os.listdir(forecasting_model_dir):
    try:
        forecaster.load_models(forecasting_model_dir)
    except:
        pass

recommendation_engine = RecommendationEngine()

@router.post("/transactions/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with transactions
    """
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Process transactions
        processed_df = processor.process_transactions(df)
        
        # Save processed data
        os.makedirs('app/data/processed', exist_ok=True)
        processed_df.to_csv('app/data/processed/transactions_clean.csv', index=False)
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_df)} transactions",
            "sample": processed_df.head(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@router.get("/transactions/categories")
async def get_transaction_categories():
    """
    Get transaction categories
    """
    try:
        # Check if we have processed transactions
        if not os.path.exists('app/data/processed/transactions_clean.csv'):
            return {"categories": []}
        
        # Load transactions
        df = pd.read_csv('app/data/processed/transactions_clean.csv')
        
        # Get categories
        if 'category' in df.columns:
            categories = df['category'].value_counts().to_dict()
            return {"categories": [{"name": k, "count": v} for k, v in categories.items()]}
        else:
            return {"categories": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@router.get("/transactions/summary")
async def get_transaction_summary(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get transaction summary statistics
    """
    try:
        # Check if we have processed transactions
        if not os.path.exists('app/data/processed/transactions_clean.csv'):
            raise HTTPException(status_code=404, detail="No transactions found")
        
        # Load transactions
        df = pd.read_csv('app/data/processed/transactions_clean.csv')
        
        # Map columns if needed
        column_mapping = {
            'DATE': 'transaction_date',
            ' WITHDRAWAL AMT ': 'withdrawal',
            ' DEPOSIT AMT ': 'deposit',
            'TRANSACTION DETAILS': 'description'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Filter by date if provided
        if start_date or end_date:
            if 'transaction_date' in df.columns:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
                
                if start_date:
                    df = df[df['transaction_date'] >= pd.to_datetime(start_date)]
                
                if end_date:
                    df = df[df['transaction_date'] <= pd.to_datetime(end_date)]
        
        # Create summary
        summary = {
            "total_transactions": len(df),
            "date_range": {
                "start": df['transaction_date'].min() if 'transaction_date' in df.columns else None,
                "end": df['transaction_date'].max() if 'transaction_date' in df.columns else None
            }
        }
        
        # Add financial summary
        if 'amount' in df.columns:
            summary["financials"] = {
                "total_spending": abs(df[df['amount'] < 0]['amount'].sum()),
                "total_income": df[df['amount'] > 0]['amount'].sum(),
                "net_cashflow": df['amount'].sum()
            }
        elif 'withdrawal' in df.columns and 'deposit' in df.columns:
            summary["financials"] = {
                "total_spending": df['withdrawal'].sum(),
                "total_income": df['deposit'].sum(),
                "net_cashflow": df['deposit'].sum() - df['withdrawal'].sum()
            }
        
        # Add category breakdown if available
        if 'category' in df.columns:
            if 'amount' in df.columns:
                category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
            elif 'withdrawal' in df.columns:
                category_spending = df.groupby('category')['withdrawal'].sum()
            else:
                category_spending = pd.Series()
                
            summary["category_breakdown"] = {
                cat: float(amount) for cat, amount in category_spending.items()
            }
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

@router.get("/transactions/forecast")
async def get_spending_forecast(days: int = 30):
    """
    Get spending forecast
    """
    try:
        # Check if we have a trained forecaster
        if not forecaster or not forecaster.models:
            raise HTTPException(status_code=404, detail="No forecasting models available")
        
        # Generate forecasts
        forecasts = forecaster.forecast(days=days)
        
        # Format response
        response = []
        
        for category, forecast_df in forecasts.items():
            forecast_data = []
            
            for _, row in forecast_df.iterrows():
                data_point = {
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "amount": float(row['amount'])
                }
                
                if 'lower_bound' in row and 'upper_bound' in row:
                    data_point["lower_bound"] = float(row['lower_bound'])
                    data_point["upper_bound"] = float(row['upper_bound'])
                
                forecast_data.append(data_point)
            
            response.append({
                "category": category,
                "forecast": forecast_data,
                "total_forecasted": float(forecast_df['amount'].sum())
            })
        
        return {"forecasts": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@router.get("/transactions/recommendations")
async def get_recommendations(limit: int = 10):
    """
    Get personalized recommendations
    """
    try:
        # Check if we have processed transactions
        if not os.path.exists('app/data/processed/transactions_clean.csv'):
            raise HTTPException(status_code=404, detail="No transactions found")
        
        # Load transactions
        df = pd.read_csv('app/data/processed/transactions_clean.csv')
        
        # Map columns if needed
        column_mapping = {
            'DATE': 'transaction_date',
            ' WITHDRAWAL AMT ': 'withdrawal',
            ' DEPOSIT AMT ': 'deposit',
            'TRANSACTION DETAILS': 'description'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Get forecasts if available
        forecasts = None
        if forecaster and forecaster.models:
            forecasts = forecaster.forecast(days=30)
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            df,
            forecasts=forecasts,
            date_col='transaction_date',
            amount_col='amount' if 'amount' in df.columns else 'withdrawal',
            category_col='category' if 'category' in df.columns else None,
            desc_col='description' if 'description' in df.columns else 'TRANSACTION DETAILS',
            limit=limit
        )
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/transactions/categorize")
async def categorize_transactions(transactions: List[Dict[str, Any]]):
    """
    Categorize transactions using the NLP model
    """
    try:
        # Check if we have a trained categorizer
        if not categorizer or not categorizer.is_trained:
            raise HTTPException(status_code=404, detail="No categorization model available")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure we have a description column
        if 'description' not in df.columns and 'TRANSACTION DETAILS' in df.columns:
            df['description'] = df['TRANSACTION DETAILS']
        
        if 'description' not in df.columns:
            raise HTTPException(status_code=400, detail="Transactions must include description field")
        
        # Categorize transactions
        descriptions = df['description'].fillna('').tolist()
        categories = categorizer.predict(descriptions)
        
        # Return categorized transa ctions
        result = []
        for i, transaction in enumerate(transactions):
            categorized = transaction.copy()
            categorized['category'] = categories[i]
            result.append(categorized)
        
        return {"categorized_transactions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing transactions: {str(e)}")
    
# app/api/routes/transactions.py
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from typing import List, Optional, Dict, Any
import pandas as pd
import uuid
import os
from datetime import datetime, timedelta

# Import the schemas
from app.api.schemas.transaction import (
    Transaction, TransactionCreate, TransactionUpdate, 
    TransactionList, TransactionStats
)
from app.api.schemas.messages import Message
from app.api.auth.auth import get_current_active_user
from app.api.schemas.auth import User

# Create router
router = APIRouter()

# Path to transaction data
DATA_PATH = "app/data/processed/transactions_clean.csv"

# Helper functions for data handling
def load_transactions():
    """Load transactions from CSV file"""
    if not os.path.exists(DATA_PATH):
        # Create empty DataFrame if file doesn't exist
        df = pd.DataFrame(columns=[
            'id', 'description', 'amount', 'transaction_date', 
            'category', 'account_id', 'notes', 'created_at', 'updated_at'
        ])
        df.to_csv(DATA_PATH, index=False)
        return df
    
    df = pd.read_csv(DATA_PATH)
    
    # Convert date columns to datetime
    date_columns = ['transaction_date', 'created_at', 'updated_at']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

def save_transactions(df):
    """Save transactions to CSV file"""
    df.to_csv(DATA_PATH, index=False)

# Transaction routes
@router.get("/transactions", response_model=TransactionList)
async def get_transactions(
    skip: int = Query(0, description="Number of transactions to skip"),
    limit: int = Query(100, description="Number of transactions to return"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_amount: Optional[float] = Query(None, description="Filter by minimum amount"),
    max_amount: Optional[float] = Query(None, description="Filter by maximum amount"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a list of transactions with optional filters
    
    - **skip**: Number of transactions to skip (pagination)
    - **limit**: Maximum number of transactions to return
    - **start_date**: Filter transactions after this date
    - **end_date**: Filter transactions before this date
    - **category**: Filter by transaction category
    - **min_amount**: Filter by minimum transaction amount
    - **max_amount**: Filter by maximum transaction amount
    """
    df = load_transactions()
    
    # Apply filters
    if start_date:
        df = df[df['transaction_date'] >= start_date]
    
    if end_date:
        df = df[df['transaction_date'] <= end_date]
    
    if category:
        df = df[df['category'] == category]
    
    if min_amount is not None:
        df = df[df['amount'] >= min_amount]
    
    if max_amount is not None:
        df = df[df['amount'] <= max_amount]
    
    # Get total count
    total = len(df)
    
    # Apply pagination
    df = df.sort_values('transaction_date', ascending=False)
    df = df.iloc[skip:skip + limit]
    
    # Convert to Pydantic models
    transactions = []
    for _, row in df.iterrows():
        transaction = Transaction(
            id=row['id'],
            description=row['description'],
            amount=row['amount'],
            transaction_date=row['transaction_date'],
            category=row['category'] if 'category' in row and pd.notna(row['category']) else None,
            account_id=row['account_id'] if 'account_id' in row and pd.notna(row['account_id']) else None,
            notes=row['notes'] if 'notes' in row and pd.notna(row['notes']) else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'] if 'updated_at' in row and pd.notna(row['updated_at']) else None
        )
        transactions.append(transaction)
    
    return {"transactions": transactions, "total": total}

@router.post("/transactions", response_model=Transaction)
async def create_transaction(
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new transaction
    
    - **description**: Transaction description
    - **amount**: Transaction amount (positive for income, negative for expenses)
    - **transaction_date**: Date of transaction
    - **category**: Transaction category (optional)
    - **account_id**: Account identifier (optional)
    - **notes**: Additional notes (optional)
    """
    df = load_transactions()
    
    # Generate new transaction ID
    transaction_id = str(uuid.uuid4())
    
    # Create new transaction
    new_transaction = {
        'id': transaction_id,
        'description': transaction.description,
        'amount': transaction.amount,
        'transaction_date': transaction.transaction_date,
        'category': transaction.category,
        'account_id': transaction.account_id,
        'notes': transaction.notes,
        'created_at': datetime.now(),
        'updated_at': None
    }
    
    # Add to DataFrame
    df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
    
    # Save to file
    save_transactions(df)
    
    # Return new transaction
    return Transaction(**new_transaction)

@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(
    transaction_id: str = Path(..., description="The ID of the transaction to get"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a single transaction by ID
    
    - **transaction_id**: ID of the transaction to retrieve
    """
    df = load_transactions()
    
    # Find transaction
    transaction = df[df['id'] == transaction_id]
    
    if transaction.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Convert to Pydantic model
    row = transaction.iloc[0]
    return Transaction(
        id=row['id'],
        description=row['description'],
        amount=row['amount'],
        transaction_date=row['transaction_date'],
        category=row['category'] if 'category' in row and pd.notna(row['category']) else None,
        account_id=row['account_id'] if 'account_id' in row and pd.notna(row['account_id']) else None,
        notes=row['notes'] if 'notes' in row and pd.notna(row['notes']) else None,
        created_at=row['created_at'],
        updated_at=row['updated_at'] if 'updated_at' in row and pd.notna(row['updated_at']) else None
    )

@router.put("/transactions/{transaction_id}", response_model=Transaction)
async def update_transaction(
    transaction_id: str = Path(..., description="The ID of the transaction to update"),
    transaction: TransactionUpdate = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a transaction
    
    - **transaction_id**: ID of the transaction to update
    - **transaction**: Updated transaction data
    """
    df = load_transactions()
    
    # Find transaction
    if transaction_id not in df['id'].values:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Update transaction
    mask = df['id'] == transaction_id
    
    # Update only provided fields
    if transaction.description is not None:
        df.loc[mask, 'description'] = transaction.description
    
    if transaction.amount is not None:
        df.loc[mask, 'amount'] = transaction.amount
    
    if transaction.transaction_date is not None:
        df.loc[mask, 'transaction_date'] = transaction.transaction_date
    
    if transaction.category is not None:
        df.loc[mask, 'category'] = transaction.category
    
    if transaction.notes is not None:
        df.loc[mask, 'notes'] = transaction.notes
    
    # Update timestamp
    df.loc[mask, 'updated_at'] = datetime.now()
    
    # Save changes
    save_transactions(df)
    
    # Return updated transaction
    updated_row = df[df['id'] == transaction_id].iloc[0]
    return Transaction(
        id=updated_row['id'],
        description=updated_row['description'],
        amount=updated_row['amount'],
        transaction_date=updated_row['transaction_date'],
        category=updated_row['category'] if 'category' in updated_row and pd.notna(updated_row['category']) else None,
        account_id=updated_row['account_id'] if 'account_id' in updated_row and pd.notna(updated_row['account_id']) else None,
        notes=updated_row['notes'] if 'notes' in updated_row and pd.notna(updated_row['notes']) else None,
        created_at=updated_row['created_at'],
        updated_at=updated_row['updated_at']
    )

@router.delete("/transactions/{transaction_id}", response_model=Message)
async def delete_transaction(
    transaction_id: str = Path(..., description="The ID of the transaction to delete"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a transaction
    
    - **transaction_id**: ID of the transaction to delete
    """
    df = load_transactions()
    
    # Find transaction
    if transaction_id not in df['id'].values:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Delete transaction
    df = df[df['id'] != transaction_id]
    
    # Save changes
    save_transactions(df)
    
    return {"message": "Transaction deleted successfully"}

@router.get("/transactions/stats", response_model=TransactionStats)
async def get_transaction_stats(
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get transaction statistics
    
    - **start_date**: Start date for calculating statistics (default: 30 days ago)
    - **end_date**: End date for calculating statistics (default: today)
    """
    df = load_transactions()
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Filter by date range
    mask = (df['transaction_date'] >= start_date) & (df['transaction_date'] <= end_date)
    filtered_df = df[mask]
    
    # Calculate statistics
    income = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
    expenses = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
    net_cashflow = income - expenses
    
    # Get category breakdown
    category_breakdown = {}
    
    if 'category' in filtered_df.columns:
        # Group by category and sum amounts
        category_sums = filtered_df.groupby('category')['amount'].sum()
        
        for category, amount in category_sums.items():
            if pd.notna(category):
                category_breakdown[category] = float(amount)
    
    return {
        "total_income": float(income),
        "total_expenses": float(expenses),
        "net_cashflow": float(net_cashflow),
        "period_start": start_date,
        "period_end": end_date,
        "categories": category_breakdown
    }

@router.post("/transactions/upload", response_model=Message)
async def upload_transactions(
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload transactions endpoint placeholder
    
    In a real implementation, this would accept a CSV or other file format
    for bulk transaction import.
    """
    return {"message": "Transaction upload functionality not implemented yet"}
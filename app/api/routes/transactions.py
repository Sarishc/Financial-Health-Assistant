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
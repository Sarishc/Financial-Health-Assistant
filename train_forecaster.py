# train_forecaster.py
import pandas as pd
import numpy as np
from app.models.forecasting.spending_forecaster import SpendingForecaster
import matplotlib.pyplot as plt
import os
import joblib

def train_forecasting_models():
    """Train and save spending forecasting models"""
    
    # Load transaction data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    
    if not os.path.exists(transactions_path):
        print(f"Error: {transactions_path} not found")
        return False
    
    df = pd.read_csv(transactions_path)
    print(f"Loaded {len(df)} transactions")
    
    # Check columns
    print("Available columns:", df.columns.tolist())
    
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
    
    # Ensure we have the required columns
    required_cols = ['transaction_date']
    amount_cols = ['amount', 'withdrawal', 'deposit']
    
    # Check if we have at least one amount column
    if not any(col in df.columns for col in amount_cols):
        print(f"Error: No amount columns found in data. Need one of {amount_cols}")
        return False
    
    # Create amount column if it doesn't exist
    if 'amount' not in df.columns and 'withdrawal' in df.columns and 'deposit' in df.columns:
        df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
        df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
        df['amount'] = df['deposit'] - df['withdrawal']
        print("Created 'amount' column from withdrawal and deposit columns")
    
    # Add basic categorization if category doesn't exist
    if 'category' not in df.columns and 'description' in df.columns:
        # This is a simplified version of what's in the visualization script
        def simple_categorize(description):
            if not isinstance(description, str):
                return 'other'
            
            description = str(description).lower()
            
            categories = {
                'food': ['grocery', 'restaurant', 'cafe', 'food', 'dining', 'eat'],
                'transport': ['uber', 'lyft', 'gas', 'fuel', 'transit', 'train'],
                'shopping': ['amazon', 'walmart', 'target', 'purchase', 'store'],
                'utilities': ['electric', 'water', 'gas', 'utility', 'bill', 'phone'],
                'entertainment': ['movie', 'netflix', 'spotify', 'hulu', 'game'],
                'other': []
            }
            
            for category, keywords in categories.items():
                if any(keyword in description for keyword in keywords):
                    return category
            
            return 'other'
        
        df['category'] = df['description'].apply(simple_categorize)
        print("Added basic categorization based on descriptions")
    
    # Initialize forecaster
    forecaster = SpendingForecaster()
    
    # Prepare time series data
    try:
        # Determine amount column to use
        amount_col = 'amount' if 'amount' in df.columns else 'withdrawal'
        category_dfs = forecaster.prepare_time_series(
            df, 
            date_col='transaction_date',
            amount_col=amount_col,
            category_col='category' if 'category' in df.columns else None
        )
        print(f"Prepared time series for {len(category_dfs)} categories")
    except Exception as e:
        print(f"Error preparing time series data: {str(e)}")
        return False
    
    # Train models
    try:
        forecaster.train_models(category_dfs, column=amount_col)
        print(f"Trained {len(forecaster.models)} forecasting models")
    except Exception as e:
        print(f"Error training forecasting models: {str(e)}")
        return False
    
    # Generate forecasts
    try:
        forecasts = forecaster.forecast(days=30)
        print(f"Generated forecasts for {len(forecasts)} categories")
    except Exception as e:
        print(f"Error generating forecasts: {str(e)}")
        return False
    
    # Create directory for visualizations
    viz_dir = 'notebooks/visualizations/forecasts'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize forecasts
    for category, forecast_df in forecasts.items():
        try:
            historical_df = category_dfs.get(category)
            output_path = os.path.join(viz_dir, f"{category}_forecast.png")
            
            forecaster.visualize_forecast(
                category,
                forecast_df,
                historical_df,
                column=amount_col,
                output_path=output_path
            )
            print(f"Created visualization for '{category}' forecast")
        except Exception as e:
            print(f"Error visualizing forecast for '{category}': {str(e)}")
    
    # Save models
    try:
        model_dir = 'app/models/forecasting/saved_models'
        forecaster.save_models(model_dir)
        print(f"Saved forecasting models to {model_dir}")
    except Exception as e:
        print(f"Error saving forecasting models: {str(e)}")
    
    return True

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('app/models/forecasting', exist_ok=True)
    os.makedirs('app/models/forecasting/saved_models', exist_ok=True)
    
    # Train forecasting models
    success = train_forecasting_models()
    
    if success:
        print("Forecasting models training completed successfully")
    else:
        print("Forecasting models training failed")
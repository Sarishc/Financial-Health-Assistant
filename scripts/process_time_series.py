# scripts/process_time_series.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from app.models.time_series.time_series_processor import TimeSeriesProcessor

def process_transaction_time_series():
    """Process transaction data into time series format with features"""
    
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
    
    # Create amount column if it doesn't exist
    if 'amount' not in df.columns and 'withdrawal' in df.columns and 'deposit' in df.columns:
        df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
        df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
        df['amount'] = df['deposit'] - df['withdrawal']
        print("Created 'amount' column from withdrawal and deposit columns")
    
    # Initialize time series processor
    processor = TimeSeriesProcessor()
    
    # Process data into time series format
    # Daily aggregation
    print("\nProcessing daily time series data...")
    daily_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='D'
    )
    
    # Weekly aggregation
    print("\nProcessing weekly time series data...")
    weekly_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='W'
    )
    
    # Monthly aggregation
    print("\nProcessing monthly time series data...")
    monthly_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='M'
    )
    
    # Create directory to save processed data
    os.makedirs('app/data/processed/time_series', exist_ok=True)
    
    # Save processed data
    daily_ts.to_csv('app/data/processed/time_series/daily_ts.csv', index=False)
    weekly_ts.to_csv('app/data/processed/time_series/weekly_ts.csv', index=False)
    monthly_ts.to_csv('app/data/processed/time_series/monthly_ts.csv', index=False)
    
    print(f"Saved time series data to app/data/processed/time_series/")
    
    # Create visualizations
    print("\nCreating time series visualizations...")
    
    # Visualize spending patterns
    figures = processor.visualize_spending_patterns(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='M'  # Monthly visualization
    )
    
    print(f"Created {len(figures)} visualizations in {processor.visualization_dir}")
    
    return True

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('app/models/time_series', exist_ok=True)
    
    # Process time series data
    success = process_transaction_time_series()
    
    if success:
        print("Time series processing completed successfully")
    else:
        print("Time series processing failed")
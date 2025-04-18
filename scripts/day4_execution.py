# scripts/day4_execution.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.time_series.time_series_processor import TimeSeriesProcessor
from scripts.generate_time_series_sample import generate_sample_time_series

def execute_day4_tasks():
    """Execute all Day 4 tasks for the Financial Health Assistant project"""
    
    print("=" * 80)
    print("FINANCIAL HEALTH ASSISTANT - DAY 4: TIME SERIES PREPROCESSING")
    print("=" * 80)
    
    # Step 1: Create necessary directories
    os.makedirs('app/models/time_series', exist_ok=True)
    os.makedirs('app/data/processed/time_series', exist_ok=True)
    os.makedirs('notebooks/visualizations/time_series', exist_ok=True)
    
    # Step 2: Check if transaction data exists, otherwise generate synthetic data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    
    if not os.path.exists(transactions_path):
        print("Transaction data not found. Generating synthetic data...")
        df = generate_sample_time_series(n_days=365, n_transactions_per_day=10)
    else:
        print(f"Loading transaction data from {transactions_path}...")
        df = pd.read_csv(transactions_path)
        
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
    
    print(f"Loaded {len(df)} transactions")
    
    # Step 3: Initialize time series processor
    processor = TimeSeriesProcessor()
    
    # Step 4: Process time series data at different frequencies
    print("\nProcessing daily time series data...")
    daily_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='D'
    )
    
    print("\nProcessing weekly time series data...")
    weekly_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='W'
    )
    
    print("\nProcessing monthly time series data...")
    monthly_ts = processor.process_time_series_data(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='M'
    )
    
    # Step 5: Save processed time series data
    daily_ts.to_csv('app/data/processed/time_series/daily_ts.csv', index=False)
    weekly_ts.to_csv('app/data/processed/time_series/weekly_ts.csv', index=False)
    monthly_ts.to_csv('app/data/processed/time_series/monthly_ts.csv', index=False)
    
    print(f"\nSaved time series data with the following dimensions:")
    print(f"Daily: {daily_ts.shape[0]} rows x {daily_ts.shape[1]} columns")
    print(f"Weekly: {weekly_ts.shape[0]} rows x {weekly_ts.shape[1]} columns")
    print(f"Monthly: {monthly_ts.shape[0]} rows x {monthly_ts.shape[1]} columns")
    
    # Step 6: Create visualizations
    print("\nCreating time series visualizations...")
    figures = processor.visualize_spending_patterns(
        df,
        date_col='transaction_date',
        amount_col='amount',
        category_col='category',
        freq='M'  # Monthly visualization
    )
    
    print(f"Created {len(figures)} visualizations in {processor.visualization_dir}")
    
    # Step 7: Run some analysis on the data
    print("\nAnalyzing spending patterns...")
    
    # Overall statistics
    expenses = df[df['amount'] < 0]['amount'].sum()
    income = df[df['amount'] > 0]['amount'].sum()
    
    print(f"Total expenses: ${abs(expenses):.2f}")
    print(f"Total income: ${income:.2f}")
    print(f"Net cash flow: ${(income + expenses):.2f}")
    
    # Category breakdown
    if 'category' in df.columns:
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        
        print("\nTop spending categories:")
        for category, amount in category_spending.head(5).items():
            print(f"{category}: ${amount:.2f}")
    
    # Temporal patterns
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['dayofweek'] = df['transaction_date'].dt.dayofweek
        df['month'] = df['transaction_date'].dt.month
        
        # Day of week pattern
        dow_spending = df[df['amount'] < 0].groupby('dayofweek')['amount'].sum().abs()
        
        # Month pattern
        month_spending = df[df['amount'] < 0].groupby('month')['amount'].sum().abs()
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Day of week visualization
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_spending.plot(kind='bar', ax=ax1)
        ax1.set_title('Spending by Day of Week')
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel('Total Amount')
        ax1.set_xticklabels(day_names, rotation=45)
        
        # Month visualization
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_spending.plot(kind='bar', ax=ax2)
        ax2.set_title('Spending by Month')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Amount')
        ax2.set_xticklabels(month_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig('notebooks/visualizations/time_series/temporal_patterns.png')
        
        print("\nIdentified temporal patterns:")
        highest_dow = day_names[dow_spending.idxmax()]
        lowest_dow = day_names[dow_spending.idxmin()]
        print(f"Highest spending day: {highest_dow}")
        print(f"Lowest spending day: {lowest_dow}")
        
        highest_month = month_names[month_spending.idxmax()-1]
        lowest_month = month_names[month_spending.idxmin()-1]
        print(f"Highest spending month: {highest_month}")
        print(f"Lowest spending month: {lowest_month}")
    
    print("\nDay 4 execution completed successfully!")
    return True

if __name__ == "__main__":
    execute_day4_tasks()
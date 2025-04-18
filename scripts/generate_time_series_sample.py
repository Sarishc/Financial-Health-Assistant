# scripts/generate_time_series_sample.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_time_series(n_days=365, n_transactions_per_day=10):
    """
    Generate synthetic time series data for testing
    
    Args:
        n_days: Number of days to generate data for
        n_transactions_per_day: Average number of transactions per day
        
    Returns:
        DataFrame with synthetic transaction data
    """
    print(f"Generating synthetic time series data for {n_days} days...")
    
    # Generate dates
    base_date = datetime(2023, 1, 1)
    end_date = base_date + timedelta(days=n_days-1)
    date_range = pd.date_range(start=base_date, end=end_date, freq='D')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate transactions
    n_transactions = n_days * n_transactions_per_day
    transactions = []
    
    # Categories with spending profiles
    # scripts/generate_time_series_sample.py (continued)
    # Categories with spending profiles
    categories = {
        'food': {'weight': 0.25, 'mean': 50, 'std': 20, 'weekly_pattern': [0.8, 1.0, 0.9, 1.0, 1.1, 1.5, 1.3]},
        'transport': {'weight': 0.15, 'mean': 30, 'std': 15, 'weekly_pattern': [1.2, 1.2, 1.1, 1.1, 1.3, 0.8, 0.5]},
        'shopping': {'weight': 0.20, 'mean': 80, 'std': 40, 'weekly_pattern': [0.7, 0.8, 0.9, 1.0, 1.1, 1.5, 1.4]},
        'utilities': {'weight': 0.10, 'mean': 100, 'std': 30, 'weekly_pattern': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        'entertainment': {'weight': 0.15, 'mean': 60, 'std': 30, 'weekly_pattern': [0.6, 0.7, 0.8, 1.0, 1.4, 1.8, 1.5]},
        'income': {'weight': 0.15, 'mean': 2000, 'std': 500, 'weekly_pattern': [0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.0]}
    }
    
    # Generate transactions based on category profiles
    for day_idx, date in enumerate(date_range):
        # Day of week (0=Monday, 6=Sunday)
        day_of_week = date.weekday()
        
        # Month seasonality factor (spending increases towards end of the year)
        month = date.month
        month_factor = 1.0 + (month - 1) * 0.02  # Gradual increase through the year
        
        # For each category, generate transactions
        for category, profile in categories.items():
            # Number of transactions for this category on this day
            # Affected by category weight and day of week pattern
            daily_weight = profile['weight'] * profile['weekly_pattern'][day_of_week] * month_factor
            n_cat_transactions = int(np.random.poisson(daily_weight * n_transactions_per_day / 5))
            
            for _ in range(n_cat_transactions):
                # Generate amount based on category profile
                if category == 'income':
                    # Income is positive
                    amount = np.random.normal(profile['mean'], profile['std'])
                    # Most income on specific days (e.g., payday on 1st and 15th)
                    if date.day not in [1, 15]:
                        if np.random.random() > 0.05:  # Small chance of income on non-payday
                            continue
                else:
                    # Expenses are negative
                    amount = -abs(np.random.normal(profile['mean'], profile['std']))
                
                # Add monthly pattern for utilities (higher in winter months)
                if category == 'utilities':
                    # Northern hemisphere winter (higher bills in Nov-Feb)
                    if month in [11, 12, 1, 2]:
                        amount *= 1.5
                
                # Add transaction
                transactions.append({
                    'transaction_date': date,
                    'category': category,
                    'amount': amount,
                    'description': f"Synthetic {category} transaction"
                })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Add some random noise to dates (not all transactions happen at midnight)
    df['transaction_date'] = df['transaction_date'].apply(
        lambda x: x + timedelta(hours=np.random.randint(0, 24), 
                               minutes=np.random.randint(0, 60))
    )
    
    # Add some recurring transactions
    recurring_templates = [
        {'category': 'utilities', 'description': 'Electricity Bill', 'amount': -120, 'day': 5},
        {'category': 'utilities', 'description': 'Water Bill', 'amount': -65, 'day': 10},
        {'category': 'utilities', 'description': 'Internet Service', 'amount': -80, 'day': 15},
        {'category': 'entertainment', 'description': 'Streaming Service', 'amount': -15, 'day': 7},
        {'category': 'housing', 'description': 'Rent Payment', 'amount': -1500, 'day': 1}
    ]
    
    recurring = []
    for month in range(1, 13):
        if base_date.month + month > 12:
            year = base_date.year + 1
            month_num = (base_date.month + month) % 12
            if month_num == 0:
                month_num = 12
        else:
            year = base_date.year
            month_num = base_date.month + month
        
        # Skip if beyond the date range
        if datetime(year, month_num, 1) > end_date:
            break
            
        for template in recurring_templates:
            # Skip some months randomly (for realism)
            if np.random.random() < 0.05:
                continue
                
            # Vary the amount slightly
            amount_variation = np.random.normal(0, abs(template['amount'] * 0.05))
            amount = template['amount'] + amount_variation
            
            # Create transaction
            try:
                trans_date = datetime(year, month_num, template['day'])
                # Skip if beyond the date range
                if trans_date > end_date:
                    continue
                    
                recurring.append({
                    'transaction_date': trans_date,
                    'category': template['category'],
                    'amount': amount,
                    'description': template['description']
                })
            except ValueError:
                # Skip invalid dates (e.g., Feb 30)
                continue
    
    # Append recurring transactions
    recurring_df = pd.DataFrame(recurring)
    df = pd.concat([df, recurring_df], ignore_index=True)
    
    # Sort by date
    df = df.sort_values('transaction_date')
    
    print(f"Generated {len(df)} synthetic transactions")
    
    # Save to CSV
    output_dir = 'app/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'synthetic_time_series.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Saved synthetic data to {output_path}")
    
    return df

if __name__ == "__main__":
    # Generate sample time series data
    df = generate_sample_time_series(n_days=365, n_transactions_per_day=10)
    
    # Display sample
    print("\nSample transactions:")
    print(df.head())
    
    # Display statistics
    print("\nTransaction statistics:")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Total transactions: {len(df)}")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    # Calculate spending by category
    category_spending = df.groupby('category')['amount'].sum().reset_index()
    category_spending['amount'] = category_spending['amount'].round(2)
    
    print("\nTotal by category:")
    for _, row in category_spending.iterrows():
        print(f"{row['category']}: ${abs(row['amount']):.2f}")
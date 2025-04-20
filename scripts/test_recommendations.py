# scripts/test_recommendations.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.recommendation.recommendation_engine import RecommendationEngine

def test_recommendation_engine():
    """Test the recommendation engine with sample data"""
    
    print("=" * 50)
    print("Testing Recommendation Engine")
    print("=" * 50)
    
    # Check if we have real data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    
    if os.path.exists(transactions_path):
        print(f"Using real transaction data from {transactions_path}")
        df = pd.read_csv(transactions_path)
    else:
        print("No real transaction data found. Creating synthetic data for testing...")
        df = create_synthetic_data()
    
    print(f"Loaded {len(df)} transactions")
    
    # Ensure columns are properly formatted
    date_col = 'transaction_date'
    amount_col = 'amount'
    
    # Convert date column to datetime if needed
    if date_col in df.columns and not pd.api.types.is_datetime64_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Check for amount column or create it
    if amount_col not in df.columns:
        if 'withdrawal' in df.columns and 'deposit' in df.columns:
            # Convert withdrawal to negative, deposit to positive
            df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
            df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
            df[amount_col] = df['deposit'] - df['withdrawal']
            print("Created 'amount' column from withdrawal and deposit columns")
    
    # Identify column names
    print("Available columns:", df.columns.tolist())
    
    # Determine category and description columns
    category_col = 'category' if 'category' in df.columns else None
    desc_col = 'description' if 'description' in df.columns else ('TRANSACTION DETAILS' if 'TRANSACTION DETAILS' in df.columns else None)
    
    # Initialize recommendation engine
    engine = RecommendationEngine(threshold_percentile=75)
    
    # Generate recommendations
    try:
        recommendations = engine.generate_recommendations(
            df,
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col,
            desc_col=desc_col,
            limit=10
        )
        
        print(f"Generated {len(recommendations)} recommendations")
        
        # Display recommendations
        print("\nTop Financial Recommendations:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['message']}")
            print(f"   Priority: {rec['priority']}")
            print(f"   Type: {rec['type']}")
            print("-" * 80)
        
        # Visualize recommendations
        viz_dir = 'notebooks/visualizations'
        os.makedirs(viz_dir, exist_ok=True)
        
        output_path = os.path.join(viz_dir, 'recommendations_priority.png')
        engine.visualize_recommendations(recommendations, output_path)
        
        # Save recommendations to CSV
        recommendations_df = pd.DataFrame(recommendations)
        output_csv = 'app/data/processed/recommendations.csv'
        recommendations_df.to_csv(output_csv, index=False)
        print(f"Saved recommendations to {output_csv}")
        
        return True
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return False

def create_synthetic_data(n_transactions=500):
    """Create synthetic transaction data for testing the recommendation engine"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define categories
    categories = [
        'food', 'transport', 'utilities', 'entertainment', 
        'shopping', 'health', 'housing', 'income', 'other'
    ]
    
    # Category to merchant mapping
    merchants = {
        'food': ['Grocery Store', 'Restaurant', 'Coffee Shop', 'Fast Food'],
        'transport': ['Gas Station', 'Uber', 'Lyft', 'Public Transit'],
        'utilities': ['Electric Company', 'Water Bill', 'Internet Provider', 'Mobile Phone'],
        'entertainment': ['Netflix', 'Movie Theater', 'Spotify', 'Gaming'],
        'shopping': ['Amazon', 'Target', 'Walmart', 'Department Store'],
        'health': ['Pharmacy', 'Doctor Visit', 'Gym Membership', 'Health Insurance'],
        'housing': ['Rent Payment', 'Mortgage', 'Home Improvement', 'Furniture'],
        'income': ['Salary Deposit', 'Freelance Payment', 'Tax Refund', 'Interest'],
        'other': ['Miscellaneous', 'ATM Withdrawal', 'Transfer', 'Unknown']
    }
    
    # Generate transactions
    transactions = []
    
    # Add regular income (twice a month)
    for month in range(6):
        for day in [15, 30]:
            payment_date = start_date + timedelta(days=30*month + day)
            if payment_date <= end_date:
                transactions.append({
                    'transaction_date': payment_date,
                    'description': 'Salary Deposit',
                    'amount': np.random.uniform(3000, 5000),
                    'category': 'income'
                })
    
    # Add regular expenses
    for month in range(7):  # 7 to ensure we have enough for the last month
        # Rent/Mortgage (monthly on the 1st)
        payment_date = start_date + timedelta(days=30*month)
        if payment_date <= end_date:
            transactions.append({
                'transaction_date': payment_date,
                'description': 'Rent Payment',
                'amount': -np.random.uniform(1200, 1500),
                'category': 'housing'
            })
        
        # Utilities (monthly, random days)
        for utility in merchants['utilities']:
            payment_date = start_date + timedelta(days=30*month + np.random.randint(1, 15))
            if payment_date <= end_date:
                transactions.append({
                    'transaction_date': payment_date,
                    'description': utility,
                    'amount': -np.random.uniform(50, 200),
                    'category': 'utilities'
                })
        
        # Subscriptions (monthly, consistent dates)
        for i, service in enumerate(['Netflix', 'Spotify', 'Gym Membership']):
            payment_date = start_date + timedelta(days=30*month + 5 + i*3)
            if payment_date <= end_date:
                transactions.append({
                    'transaction_date': payment_date,
                    'description': service,
                    'amount': -np.random.uniform(10, 50),
                    'category': 'entertainment' if service != 'Gym Membership' else 'health'
                })
    
    # Add random transactions throughout the period
    for _ in range(350):  # Additional random transactions
        # Pick a random date
        date = np.random.choice(date_range)
        
        # Pick a random category (weighted)
        category = np.random.choice(categories, p=[0.25, 0.15, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.1])
        
        # Pick a random merchant for that category
        merchant = np.random.choice(merchants[category])
        
        # Determine amount based on category
        if category == 'income':
            amount = np.random.uniform(100, 1000)  # Small incomes (side gigs, etc.)
        elif category == 'housing':
            amount = -np.random.uniform(100, 500)  # Housing-related expenses
        elif category == 'food':
            amount = -np.random.uniform(10, 100)
        elif category == 'transport':
            amount = -np.random.uniform(20, 80)
        elif category == 'utilities':
            amount = -np.random.uniform(30, 150)
        elif category == 'entertainment':
            amount = -np.random.uniform(10, 100)
        elif category == 'shopping':
            amount = -np.random.uniform(20, 200)
        elif category == 'health':
            amount = -np.random.uniform(10, 300)
        else:  # other
            amount = -np.random.uniform(10, 150)
        
        transactions.append({
            'transaction_date': date,
            'description': merchant,
            'amount': amount,
            'category': category
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('transaction_date')
    
    # Save synthetic data
    output_dir = 'app/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    synthetic_path = os.path.join(output_dir, 'synthetic_transactions.csv')
    df.to_csv(synthetic_path, index=False)
    print(f"Saved synthetic transaction data to {synthetic_path}")
    
    return df

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('app/data/processed', exist_ok=True)
    os.makedirs('notebooks/visualizations', exist_ok=True)
    
    # Test recommendation engine
    success = test_recommendation_engine()
    
    if success:
        print("Recommendation engine tested successfully!")
    else:
        print("Recommendation engine test failed")
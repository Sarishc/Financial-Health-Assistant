# scripts/day6_execution.py
"""
Day 6 execution script - Recommendation Engine Development
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.recommendation.recommendation_engine import RecommendationEngine
from app.models.forecasting.spending_forecaster import SpendingForecaster

def run_day6_tasks():
    """Execute Day 6 implementation tasks: Building the Recommendation Engine"""
    
    print("=" * 80)
    print("Day 6: Recommendation Engine Development")
    print("=" * 80)
    
    # 1. Create necessary directories
    print("\nCreating directories...")
    os.makedirs('app/models/recommendation', exist_ok=True)
    os.makedirs('app/data/processed', exist_ok=True)
    os.makedirs('notebooks/visualizations/recommendations', exist_ok=True)
    
    # 2. Check for transaction data
    transactions_path = 'app/data/processed/transactions_clean.csv'
    
    if not os.path.exists(transactions_path):
        print(f"Error: No processed transaction data found at {transactions_path}")
        print("Running with synthetic data instead...")
        from scripts.test_recommendations import create_synthetic_data
        df = create_synthetic_data(n_transactions=500)
    else:
        print(f"Loading transaction data from {transactions_path}")
        df = pd.read_csv(transactions_path)
        print(f"Loaded {len(df)} transactions")
    
    # 3. Ensure data is properly formatted
    # Map the columns to expected names
    column_mapping = {
        'DATE': 'transaction_date',
        'TRANSACTION DETAILS': 'description',
        ' WITHDRAWAL AMT ': 'withdrawal',
        ' DEPOSIT AMT ': 'deposit',
        'CHQ.NO.': 'check_number',
        'VALUE DATE': 'value_date',
        'Account No': 'account_number'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
            print(f"Mapped '{old_col}' to '{new_col}'")
    
    date_col = 'transaction_date'
    amount_col = 'amount'
    
    # Convert date column to datetime if needed
    if date_col in df.columns and not pd.api.types.is_datetime64_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"Converted '{date_col}' to datetime format")
    
    # Check for amount column or create it
    if amount_col not in df.columns:
        if 'withdrawal' in df.columns and 'deposit' in df.columns:
            # Convert withdrawal to negative, deposit to positive
            df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
            df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)
            df['amount'] = df['deposit'] - df['withdrawal']
            print("Created 'amount' column from withdrawal and deposit columns")
    
    # Identify column names
    print("\nAvailable columns after mapping:", df.columns.tolist())
    
    # Determine category and description columns
    category_col = 'category' if 'category' in df.columns else None
    desc_col = 'description' if 'description' in df.columns else None
    
    # 4. Check for forecasting models if available
    forecasts = None
    forecasting_model_dir = 'app/models/forecasting/saved_models'
    
    if os.path.exists(forecasting_model_dir) and os.listdir(forecasting_model_dir):
        try:
            print("\nLoading forecasting models...")
            # Load forecaster and forecasts
            forecaster = SpendingForecaster()
            forecaster.load_models(forecasting_model_dir)
            forecasts = forecaster.forecast(days=30)
            print(f"Loaded forecasts for {len(forecasts)} categories")
        except Exception as e:
            print(f"Error loading forecasts: {str(e)}")
            print("Proceeding without forecasting models")
    else:
        print("\nNo forecasting models found. Recommendations will not include forecast data.")
    
    # 5. Initialize and run recommendation engine
    print("\nGenerating financial recommendations...")
    engine = RecommendationEngine(threshold_percentile=75)
    
    try:
        recommendations = engine.generate_recommendations(
            df,
            forecasts=forecasts,
            date_col=date_col,
            amount_col=amount_col,
            category_col=category_col,
            desc_col=desc_col,
            limit=10
        )
        
        print(f"Generated {len(recommendations)} recommendations")
        
        # 6. Display recommendations
        print("\nTop Financial Recommendations:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['message']}")
            print(f"   Priority: {rec['priority']}")
            print(f"   Type: {rec['type']}")
            print("-" * 80)
        
        # 7. Visualize recommendations
        viz_dir = 'notebooks/visualizations/recommendations'
        
        # Priority chart
        priority_path = os.path.join(viz_dir, 'recommendations_priority.png')
        engine.visualize_recommendations(recommendations, priority_path)
        
        # Type distribution chart
        type_path = os.path.join(viz_dir, 'recommendation_types.png')
        
        # Count recommendations by type
        type_counts = {}
        for rec in recommendations:
            rec_type = rec['type']
            if rec_type not in type_counts:
                type_counts[rec_type] = 0
            type_counts[rec_type] += 1
        
        # Create pie chart of recommendation types
        plt.figure(figsize=(10, 7))
        plt.pie(type_counts.values(), labels=[t.replace('_', ' ').title() for t in type_counts.keys()], 
                autopct='%1.1f%%', startangle=140, shadow=True)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Recommendation Types Distribution')
        plt.tight_layout()
        plt.savefig(type_path)
        plt.close()
        
        print(f"Saved recommendation visualizations to {viz_dir}")
        
        # 8. Save recommendations to CSV
        recommendations_df = pd.DataFrame(recommendations)
        output_csv = 'app/data/processed/recommendations.csv'
        recommendations_df.to_csv(output_csv, index=False)
        print(f"Saved recommendations to {output_csv}")
        
        return True
    except Exception as e:
        print(f"Error in recommendation engine: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_day6_tasks()
    
    if success:
        print("\nDay 6 tasks completed successfully!")
    else:
        print("\nDay 6 tasks completed with errors.")
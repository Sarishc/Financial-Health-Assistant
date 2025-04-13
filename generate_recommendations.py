# generate_recommendations.py
import pandas as pd
import numpy as np
from app.models.recommendation.recommendation_engine import RecommendationEngine
from app.models.forecasting.spending_forecaster import SpendingForecaster
import os
import matplotlib.pyplot as plt

def generate_financial_recommendations():
    """Generate and test financial recommendations"""
    
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
    
    # Load forecasts if available
    forecasts = None
    forecasting_model_dir = 'app/models/forecasting/saved_models'
    
    if os.path.exists(forecasting_model_dir) and os.listdir(forecasting_model_dir):
        try:
            # Load forecaster and forecasts
            forecaster = SpendingForecaster()
            forecaster.load_models(forecasting_model_dir)
            forecasts = forecaster.forecast(days=30)
            print(f"Loaded forecasts for {len(forecasts)} categories")
        except Exception as e:
            print(f"Error loading forecasts: {str(e)}")
    
    # Initialize recommendation engine
    engine = RecommendationEngine(threshold_percentile=75)
    
    # Generate recommendations
    try:
        # Determine column names
        date_col = 'transaction_date'
        amount_col = 'amount' if 'amount' in df.columns else 'withdrawal'
        category_col = 'category' if 'category' in df.columns else None
        desc_col = 'description' if 'description' in df.columns else 'TRANSACTION DETAILS'
        
        # Generate recommendations
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

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('app/models/recommendation', exist_ok=True)
    
    # Generate recommendations
    success = generate_financial_recommendations()
    
    if success:
        print("Recommendation generation completed successfully")
    else:
        print("Recommendation generation failed")
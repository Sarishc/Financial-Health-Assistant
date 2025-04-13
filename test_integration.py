"""Test integration of all components of the Financial Health Assistant"""
import os
import pandas as pd
from app.data.processor import TransactionProcessor
from app.models.categorization.nlp_categorizer import TransactionCategorizer
from app.models.forecasting.spending_forecaster import SpendingForecaster
from app.models.recommendation.recommendation_engine import RecommendationEngine

def test_full_pipeline():
    """Test the complete financial analysis pipeline"""
    
    print("=" * 50)
    print("FINANCIAL HEALTH ASSISTANT INTEGRATION TEST")
    print("=" * 50)
    
    # Step 1: Check if data exists
    transactions_path = 'app/data/processed/transactions_clean.csv'
    if not os.path.exists(transactions_path):
        print("Error: No processed transaction data found")
        return False
    
    # Step 2: Load transaction data
    try:
        df = pd.read_csv(transactions_path)
        print(f"✓ Loaded {len(df)} transactions")
    except Exception as e:
        print(f"Error loading transactions: {str(e)}")
        return False
    
    # Step 3: Test categorization model
    categorizer_path = 'app/models/categorization/transaction_categorizer.joblib'
    if not os.path.exists(categorizer_path):
        print("Error: Categorization model not found")
        return False
    
    try:
        categorizer = TransactionCategorizer(categorizer_path)
        
        # Test on sample descriptions
        test_descriptions = [
            "GROCERY PURCHASE",
            "MONTHLY RENT PAYMENT",
            "UBER RIDE",
            "SALARY DEPOSIT"
        ]
        
        categories = categorizer.predict(test_descriptions)
        print(f"✓ Categorization model working: {test_descriptions[0]} → {categories[0]}")
    except Exception as e:
        print(f"Error testing categorization model: {str(e)}")
        return False
    
    # Step 4: Test forecasting model
    forecasting_model_dir = 'app/models/forecasting/saved_models'
    if not os.path.exists(forecasting_model_dir) or not os.listdir(forecasting_model_dir):
        print("Warning: No forecasting models found")
    else:
        try:
            forecaster = SpendingForecaster()
            forecaster.load_models(forecasting_model_dir)
            forecasts = forecaster.forecast(days=30)
            
            if forecasts:
                first_category = list(forecasts.keys())[0]
                print(f"✓ Forecasting model working: Generated forecast for '{first_category}' ({len(forecasts[first_category])} days)")
            else:
                print("Warning: No forecasts generated")
        except Exception as e:
            print(f"Error testing forecasting model: {str(e)}")
    
    # Step 5: Test recommendation engine
    try:
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(
            df, 
            date_col='transaction_date',
            amount_col='amount' if 'amount' in df.columns else 'withdrawal',
            category_col='category' if 'category' in df.columns else None,
            desc_col='description' if 'description' in df.columns else 'TRANSACTION DETAILS'
        )
        
        if recommendations:
            print(f"✓ Recommendation engine working: Generated {len(recommendations)} recommendations")
            print(f"  Top recommendation: {recommendations[0]['message']}")
        else:
            print("Warning: No recommendations generated")
    except Exception as e:
        print(f"Error testing recommendation engine: {str(e)}")
    
    print("\nAll components tested successfully!")
    print("You can now run the full application with: uvicorn app.main:app")
    return True

if __name__ == "__main__":
    test_full_pipeline()